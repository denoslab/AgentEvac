"""Inter-agent natural-language messaging bus.

Extracted from ``agentevac.simulation.main`` so that it can be tested
independently of SUMO / TraCI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class OutboxMessage(BaseModel):
    to: str = Field(..., description="Recipient vehicle ID, or '*' for broadcast to all active agents.")
    message: str = Field(..., description="Natural-language message content.")


class AgentMessagingBus:
    """Inter-agent natural-language messaging bus with delivery-round scheduling.

    Agents include optional outbox items in their LLM response.  The bus accepts
    these messages at round R and delivers them at round R+1 (one-round latency),
    simulating realistic communication delay.

    **Direct messages** (``to`` = specific vehicle ID):
        Delivered only to the named recipient if active at delivery time.
        Undelivered messages are held for up to ``ttl_rounds`` additional rounds,
        then dropped.

    **Broadcasts** (``to`` = ``"*"``, ``"all"``, or ``"broadcast"``):
        Fanned out to all round participants known at the time of sending (not of delivery).
        This includes both active vehicles and not-yet-departed households participating
        in the current decision round.
        A global cap (``max_broadcasts_per_round``) limits broadcast flooding.

    Per-agent message caps (``max_sends_per_agent_per_round``) prevent a single
    agent from saturating the bus.  Inboxes are capped at ``max_inbox_messages``
    entries; older messages are dropped from the front.

    Args:
        enabled: If ``False``, all methods are no-ops and inboxes are always empty.
        max_message_chars: Maximum length of a single message body (truncated).
        max_inbox_messages: Maximum messages retained per agent inbox.
        max_sends_per_agent_per_round: Per-agent send cap per decision round.
        max_broadcasts_per_round: Global broadcast cap per decision round.
        ttl_rounds: Rounds a direct message waits for an offline recipient.
        comm_radius_m: Spatial broadcast radius in metres.  ``0`` disables
            spatial filtering (broadcasts reach all agents).  Any positive value
            restricts broadcast delivery to agents within this Euclidean distance
            of the sender.  Direct messages are never filtered.
        event_stream: Optional event emitter (must have an ``emit`` method).
    """

    def __init__(
        self,
        enabled: bool,
        max_message_chars: int,
        max_inbox_messages: int,
        max_sends_per_agent_per_round: int,
        max_broadcasts_per_round: int,
        ttl_rounds: int,
        comm_radius_m: float = 0.0,
        event_stream: Any = None,
    ):
        self.enabled = bool(enabled)
        self.max_message_chars = max(1, int(max_message_chars))
        self.max_inbox_messages = max(1, int(max_inbox_messages))
        self.max_sends_per_agent_per_round = max(1, int(max_sends_per_agent_per_round))
        self.max_broadcasts_per_round = max(1, int(max_broadcasts_per_round))
        self.ttl_rounds = max(1, int(ttl_rounds))
        self.comm_radius_m = max(0.0, float(comm_radius_m))
        self._comm_radius_sq = self.comm_radius_m * self.comm_radius_m

        self._pending: List[Dict[str, Any]] = []
        self._inboxes: Dict[str, List[Dict[str, Any]]] = {}
        self._active_agents: List[str] = []
        self._positions: Dict[str, Tuple[float, float]] = {}
        self._current_round = 0
        self._broadcast_count = 0
        self._sender_sent_count: Dict[str, int] = {}
        self._sender_seq: Dict[str, int] = {}
        self._events = event_stream

    def _next_sender_seq(self, sender: str) -> int:
        nxt = self._sender_seq.get(sender, 0) + 1
        self._sender_seq[sender] = nxt
        return nxt

    def _push_inbox(self, recipient: str, msg: Dict[str, Any]):
        inbox = self._inboxes.setdefault(recipient, [])
        inbox.append({
            "from": msg["from"],
            "to": msg["to"],
            "message": msg["message"],
            "kind": "broadcast" if msg["is_broadcast"] else "direct",
            "sent_round": msg["sent_round"],
            "delivery_round": msg["deliver_round"],
        })
        if len(inbox) > self.max_inbox_messages:
            self._inboxes[recipient] = inbox[-self.max_inbox_messages:]
        if self._events:
            self._events.emit(
                "message_delivered",
                summary=f"{msg['from']} -> {recipient}",
                from_id=msg["from"],
                to_id=recipient,
                kind="broadcast" if msg["is_broadcast"] else "direct",
                sent_round=msg["sent_round"],
                delivery_round=msg["deliver_round"],
                message=msg["message"],
            )

    def begin_round(
        self,
        round_idx: int,
        participant_agent_ids: List[str],
        positions: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Start decision round R:
        - deliver messages scheduled for <= R
        - reset per-round send counters
        - store agent positions for spatial broadcast filtering
        Messages generated in round R are delivered at R+1.
        """
        if not self.enabled:
            return

        self._current_round = int(round_idx)
        self._active_agents = list(participant_agent_ids)
        self._positions = dict(positions) if positions else {}
        participant_set = set(participant_agent_ids)
        self._broadcast_count = 0
        self._sender_sent_count = {}

        remaining: List[Dict[str, Any]] = []
        for msg in self._pending:
            if msg["deliver_round"] > self._current_round:
                remaining.append(msg)
                continue

            recipient = msg["to"]
            if recipient in participant_set:
                self._push_inbox(recipient, msg)
                continue

            # Broadcast fanout is only to known round participants at send-time.
            if msg["is_broadcast"]:
                continue

            # Direct messages may wait for the recipient to appear (TTL-bound).
            if self._current_round <= msg["expire_round"]:
                remaining.append(msg)

        self._pending = remaining

    def get_inbox(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return a copy of the agent's inbox (delivered messages).

        Args:
            agent_id: Vehicle ID.

        Returns:
            List of message dicts; empty list if messaging is disabled or inbox is empty.
        """
        if not self.enabled:
            return []
        return list(self._inboxes.get(agent_id, []))

    def queue_outbox(self, sender: str, outbox: Optional[List[OutboxMessage]]):
        """
        Accept sender's outbox for current round R and enqueue for delivery at R+1.
        Enforces per-sender and global messaging caps.
        """
        if (not self.enabled) or (not outbox):
            return

        sender_count = self._sender_sent_count.get(sender, 0)
        for raw in outbox:
            if sender_count >= self.max_sends_per_agent_per_round:
                break

            recipient = str(getattr(raw, "to", "")).strip()
            recipient_norm = recipient.lower()
            text = str(getattr(raw, "message", "")).strip()
            if not recipient or not text:
                continue
            if len(text) > self.max_message_chars:
                text = text[:self.max_message_chars]

            sender_seq = self._next_sender_seq(sender)

            if recipient in {"*", "__all__"} or recipient_norm in {"all", "broadcast"}:
                if self._broadcast_count >= self.max_broadcasts_per_round:
                    continue
                self._broadcast_count += 1
                sender_count += 1
                self._sender_sent_count[sender] = sender_count

                # Spatial filtering: if comm_radius_m > 0, only fan out
                # to agents within Euclidean range of the sender.
                _s_pos = self._positions.get(sender) if self._comm_radius_sq > 0 else None
                for target in self._active_agents:
                    if target == sender:
                        continue
                    # --- spatial range check (skip if radius disabled or positions unknown) ---
                    if _s_pos is not None:
                        _t_pos = self._positions.get(target)
                        if _t_pos is not None:
                            _dx = _s_pos[0] - _t_pos[0]
                            _dy = _s_pos[1] - _t_pos[1]
                            if (_dx * _dx + _dy * _dy) > self._comm_radius_sq:
                                continue
                    if self._events:
                        self._events.emit(
                            "message_queued",
                            summary=f"{sender} -> {target}",
                            from_id=sender,
                            to_id=target,
                            kind="broadcast",
                            deliver_round=self._current_round + 1,
                            message=text,
                        )
                    self._pending.append({
                        "from": sender,
                        "to": target,
                        "message": text,
                        "sent_round": self._current_round,
                        "deliver_round": self._current_round + 1,
                        "expire_round": self._current_round + 1,
                        "sender_seq": sender_seq,
                        "is_broadcast": True,
                    })
            else:
                sender_count += 1
                self._sender_sent_count[sender] = sender_count
                if self._events:
                    self._events.emit(
                        "message_queued",
                        summary=f"{sender} -> {recipient}",
                        from_id=sender,
                        to_id=recipient,
                        kind="direct",
                        deliver_round=self._current_round + 1,
                        message=text,
                    )
                self._pending.append({
                    "from": sender,
                    "to": recipient,
                    "message": text,
                    "sent_round": self._current_round,
                    "deliver_round": self._current_round + 1,
                    "expire_round": self._current_round + self.ttl_rounds,
                    "sender_seq": sender_seq,
                    "is_broadcast": False,
                })
