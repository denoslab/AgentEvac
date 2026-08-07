"""Shared state between the simulation thread and the bridge HTTP threads.

Two kinds of thread touch this module. The simulation thread publishes snapshots
and reads control intent, and HTTP handler threads read snapshots and publish
control intent. Every mutable field is guarded by a lock, and every reader gets a
copy, so an HTTP thread never holds a reference into live simulation state and
never calls TraCI.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

# Phases the bridge reports on ``GET /status``. The console mirrors them.
PHASE_STARTING = "starting"
PHASE_LOADING = "loading"
PHASE_RUNNING = "running"
PHASE_FINISHING = "finishing"
PHASE_COMPLETE = "complete"
PHASE_FAILED = "failed"

# Speed multipliers the console offers. ``0`` means run with no throttle at all.
SPEED_MAX = 0.0


class EndRunRequested(Exception):
    """Raised inside the patched step call when the operator ends the run.

    The simulation loop is wrapped in ``try/finally``, so raising here lets the
    simulator's own cleanup flush metrics, the timeline, and the event log
    exactly as it does on a natural end.
    """


class BridgeState:
    """Thread-safe container for everything the console reads out of a run."""

    def __init__(self, *, max_events: int = 4000, run_meta: Optional[Dict[str, Any]] = None):
        self._lock = threading.Lock()

        # Lifecycle.
        self._phase: str = PHASE_STARTING
        self._phase_detail: str = "starting simulation process"
        self._started_wall: float = time.time()
        self._error: Optional[str] = None
        self._run_meta: Dict[str, Any] = dict(run_meta or {})

        # Operator intent.
        self._paused: bool = False
        self._speed: float = 16.0
        self._end_requested: bool = False
        self._shutdown_requested: bool = False

        # Continuous state, replaced wholesale by the simulation thread.
        self._snapshot: Optional[Dict[str, Any]] = None
        self._preview: Optional[Dict[str, Any]] = None

        # Discrete events, kept for late subscribers and fanned out live.
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._event_seq: int = 0
        self._subscribers: List[queue.Queue] = []

        # Decision-round progress, driven by the simulator's own event stream.
        self._round: Dict[str, Any] = {
            "in_progress": False,
            "index": 0,
            "dispatched": 0,
            "resolved": 0,
            "completed": 0,
            "total": None,
        }

        # Agent-detail requests, fulfilled on the simulation thread.
        self._agent_requests: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_phase(self, phase: str, detail: str = "", error: Optional[str] = None) -> None:
        with self._lock:
            self._phase = phase
            if detail:
                self._phase_detail = detail
            if error is not None:
                self._error = error

    def status(self) -> Dict[str, Any]:
        with self._lock:
            snap = self._snapshot
            return {
                "phase": self._phase,
                "phase_detail": self._phase_detail,
                "paused": self._paused,
                "speed": self._speed,
                "end_requested": self._end_requested,
                "elapsed_wall_s": round(time.time() - self._started_wall, 1),
                "error": self._error,
                "run": dict(self._run_meta),
                "sim_t_s": (snap or {}).get("sim_t_s"),
                "has_snapshot": snap is not None,
                "has_preview": self._preview is not None,
            }

    def run_meta(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._run_meta)

    def update_run_meta(self, **fields: Any) -> None:
        with self._lock:
            self._run_meta.update(fields)

    # ------------------------------------------------------------------
    # Operator intent
    # ------------------------------------------------------------------

    def apply_control(self, action: str, value: Any = None) -> Dict[str, Any]:
        """Apply one control action and return the resulting intent.

        Args:
            action: One of ``pause``, ``resume``, ``toggle_pause``, ``set_speed``,
                ``end``, or ``shutdown``. ``end`` stops the simulation loop and lets
                the simulator export its results. ``shutdown`` releases the process
                afterwards, once the console has read the closing state.
            value: Speed multiplier for ``set_speed``, where 0 means no throttle.

        Returns:
            The acknowledged intent, so the caller can echo it back immediately.

        Raises:
            ValueError: If the action is unknown or the speed is not a number.
        """
        with self._lock:
            if action == "pause":
                self._paused = True
            elif action == "resume":
                self._paused = False
            elif action == "toggle_pause":
                self._paused = not self._paused
            elif action == "set_speed":
                try:
                    speed = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"set_speed needs a number, got {value!r}")
                self._speed = max(0.0, speed)
            elif action == "end":
                self._end_requested = True
                self._paused = False
            elif action == "shutdown":
                self._shutdown_requested = True
            else:
                raise ValueError(f"unknown control action {action!r}")
            return self._intent_locked()

    def _intent_locked(self) -> Dict[str, Any]:
        return {
            "paused": self._paused,
            "speed": self._speed,
            "end_requested": self._end_requested,
            "shutdown_requested": self._shutdown_requested,
        }

    def intent(self) -> Dict[str, Any]:
        with self._lock:
            return self._intent_locked()

    # ------------------------------------------------------------------
    # Continuous state
    # ------------------------------------------------------------------

    def publish_snapshot(self, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._snapshot

    def publish_preview(self, preview: Dict[str, Any]) -> None:
        with self._lock:
            self._preview = preview

    def preview(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._preview

    # ------------------------------------------------------------------
    # Discrete events
    # ------------------------------------------------------------------

    def publish_event(self, record: Dict[str, Any]) -> None:
        """Record one simulator event and fan it out to live subscribers."""
        with self._lock:
            self._event_seq += 1
            record = dict(record)
            record["seq"] = self._event_seq
            self._events.append(record)
            dead = []
            for sub in self._subscribers:
                try:
                    sub.put_nowait(record)
                except queue.Full:
                    dead.append(sub)
            for sub in dead:
                self._subscribers.remove(sub)

    def subscribe(self, backlog: int = 200) -> "queue.Queue":
        """Register a live event queue, primed with the most recent events."""
        sub: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            recent = list(self._events)[-max(0, backlog):]
            self._subscribers.append(sub)
        for record in recent:
            try:
                sub.put_nowait(record)
            except queue.Full:
                break
        return sub

    def unsubscribe(self, sub: "queue.Queue") -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)

    def events_since(self, seq: int, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            return [rec for rec in self._events if rec.get("seq", 0) > seq][:limit]

    # ------------------------------------------------------------------
    # Decision-round progress
    # ------------------------------------------------------------------

    def round_begin(self, index: int, dispatched: int) -> None:
        with self._lock:
            self._round.update({
                "in_progress": True,
                "index": int(index),
                "dispatched": int(dispatched),
                "resolved": 0,
            })

    def round_resolve_one(self) -> None:
        with self._lock:
            if self._round["in_progress"]:
                self._round["resolved"] += 1

    def round_end(self) -> None:
        with self._lock:
            if self._round["in_progress"]:
                self._round["in_progress"] = False
                self._round["completed"] += 1

    def set_round_total(self, total: Optional[int]) -> None:
        with self._lock:
            self._round["total"] = total

    def round_progress(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._round)

    # ------------------------------------------------------------------
    # Agent detail, fulfilled on the simulation thread
    # ------------------------------------------------------------------

    def request_agent(self, agent_id: str, timeout_s: float = 5.0) -> Optional[Dict[str, Any]]:
        """Ask the simulation thread for one agent's detail and wait for it.

        Reading agent state from an HTTP thread would race the simulator's own
        writes, so the request is parked here and answered by
        ``fulfil_agent_requests`` the next time the simulation thread pumps the
        bridge. Each caller holds its own request record, so the answer cannot be
        collected by anyone else.
        """
        request: Dict[str, Any] = {
            "agent_id": str(agent_id),
            "done": threading.Event(),
            "result": None,
            "answered": False,
        }
        with self._lock:
            self._agent_requests.append(request)
        request["done"].wait(timeout_s)
        if not request["answered"]:
            with self._lock:
                if request in self._agent_requests:
                    self._agent_requests.remove(request)
            return None
        return request["result"]

    def fulfil_agent_requests(self, resolver) -> None:
        """Answer parked agent-detail requests. Call from the simulation thread."""
        with self._lock:
            pending = self._agent_requests
            self._agent_requests = []
        if not pending:
            return
        resolved: Dict[str, Any] = {}
        for request in pending:
            agent_id = request["agent_id"]
            if agent_id not in resolved:
                try:
                    resolved[agent_id] = resolver(agent_id)
                except Exception as exc:  # a bad agent id must not stop the run
                    resolved[agent_id] = {"error": "agent_lookup_failed", "detail": str(exc)}
            request["result"] = resolved[agent_id]
            request["answered"] = True
            request["done"].set()


class Pacer:
    """Wall-clock throttle that slows the simulation to a speed multiplier.

    The pacer only ever sleeps. It never advances the simulation, so a run paced
    at any speed produces the same trajectory as an unpaced one.
    """

    def __init__(self) -> None:
        self._deadline: Optional[float] = None

    def reset(self) -> None:
        self._deadline = None

    def wait(self, step_length_s: float, speed: float) -> None:
        """Sleep until this step's share of wall time has elapsed."""
        if speed <= 0:
            self._deadline = None
            return
        budget = step_length_s / speed
        now = time.perf_counter()
        if self._deadline is None:
            self._deadline = now + budget
            return
        self._deadline += budget
        remaining = self._deadline - now
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -1.0:
            # Running behind by more than a second means the machine cannot hold
            # this speed. Forget the debt so a later idle period is not skipped.
            self._deadline = now + budget
