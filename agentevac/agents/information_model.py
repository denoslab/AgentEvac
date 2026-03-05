"""Information sensing and social signal processing for evacuation agents.

This module handles the two information streams available to each agent each decision round:

**Environmental signals** (``sample_environment_signal``):
    The agent observes the closest fire margin on its current edge and the minimum margin
    across the head of its planned route.  Gaussian noise (``sigma_info`` metres, std-dev)
    is injected to model imperfect sensing — e.g., smoke obscuring visibility or GPS
    inaccuracy.  An optional delay (``INFO_DELAY_S`` seconds, converted to ``delay_rounds``)
    is applied by replaying a stale record from the agent's signal history, simulating
    delayed emergency broadcasts or slow rumour propagation.

**Social signals** (``build_social_signal``):
    Peer messages from the agent's inbox are parsed with a simple keyword-vote approach.
    Each message casts a vote for one of three hazard states based on which keyword
    category is detected first (danger > risky > safe).  The vote tally is converted to
    a probability triplet that is later fused with the environmental belief in
    ``belief_model.fuse_env_and_social_beliefs``.
"""

import random
from typing import Any, Dict, List, Optional


def _state_from_margin(margin_m: Optional[float]) -> str:
    """Classify a fire-edge margin into a discrete hazard state label.

    Args:
        margin_m: Distance in metres from the nearest fire edge; ``None`` if unknown.

    Returns:
        One of "danger" (≤ 100 m), "risky" (≤ 300 m), "safe" (> 300 m), or "unknown".
    """
    if margin_m is None:
        return "unknown"
    if margin_m <= 100.0:
        return "danger"
    if margin_m <= 300.0:
        return "risky"
    return "safe"


def inject_signal_noise(
    signal: Dict[str, Any],
    sigma_info: float,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Add zero-mean Gaussian noise to the observed fire margin.

    Simulates imperfect environmental sensing (e.g., smoke, sensor noise, GPS error).
    The noisy observation is clamped by the natural arithmetic (can go negative, meaning
    the agent *believes* the fire has reached it even if it hasn't, or vice-versa).

    If ``base_margin_m`` is absent (no fire active or edge not found), the function
    returns the signal unchanged with ``observed_margin_m=None``.

    Args:
        signal: Environment signal dict containing at least ``base_margin_m``.
        sigma_info: Standard deviation of the Gaussian noise in metres.
            A value of 0 disables noise injection.
        rng: Optional seeded ``random.Random`` instance for reproducible noise.
            Falls back to the global ``random`` module if not provided.

    Returns:
        A shallow copy of ``signal`` with added fields:
            - ``noise_delta_m``    : Sampled noise value (metres).
            - ``observed_margin_m``: Noisy observation (metres) or ``None``.
            - ``observed_state``   : Discrete state label from ``_state_from_margin``.
    """
    out = dict(signal)
    sigma = max(0.0, float(sigma_info))
    base_margin = out.get("base_margin_m")
    if base_margin is None:
        out["noise_delta_m"] = 0.0
        out["observed_margin_m"] = None
        out["observed_state"] = "unknown"
        return out

    src = rng if rng is not None else random
    noise_delta = float(src.gauss(0.0, sigma)) if sigma > 0.0 else 0.0
    observed_margin = float(base_margin) + noise_delta
    out["noise_delta_m"] = round(noise_delta, 2)
    out["observed_margin_m"] = round(observed_margin, 2)
    out["observed_state"] = _state_from_margin(observed_margin)
    return out


def apply_signal_delay(
    signal: Dict[str, Any],
    history: List[Dict[str, Any]],
    delay_rounds: int,
) -> Dict[str, Any]:
    """Return a delayed version of the signal by replaying a historical observation.

    When ``delay_rounds > 0``, the agent perceives the environment as it was
    ``delay_rounds`` decision periods ago, retrieved from its ``signal_history``.
    This simulates scenarios where official information is broadcast with a lag
    (e.g., emergency alerts that arrive minutes after the fire was detected).

    If the history is not yet deep enough to satisfy the requested delay, the current
    signal is returned without delay (the agent has not been alive long enough to have
    stale observations).

    Args:
        signal: The fresh environment signal for the current round.
        history: The agent's bounded signal history list (oldest-first).
        delay_rounds: Number of decision periods of delay to apply.

    Returns:
        A signal dict (either fresh or stale) with added fields:
            - ``is_delayed``            : Whether a historical record was substituted.
            - ``delay_rounds_applied``  : Actual delay in rounds applied.
            - ``delay_source_round``    : ``decision_round`` of the substituted record
                                          (only present when ``is_delayed=True``).
    """
    delay = max(0, int(delay_rounds))
    if delay <= 0:
        out = dict(signal)
        out["is_delayed"] = False
        out["delay_rounds_applied"] = 0
        return out

    if delay <= len(history):
        # Retrieve the record ``delay`` steps back (negative indexing from the most recent).
        source = dict(history[-delay])
        source["is_delayed"] = True
        source["delay_rounds_applied"] = delay
        source["delay_source_round"] = source.get("decision_round")
        return source

    # History is too short; return current signal without delay.
    out = dict(signal)
    out["is_delayed"] = False
    out["delay_rounds_applied"] = 0
    return out


def sample_environment_signal(
    agent_id: str,
    sim_t_s: float,
    current_edge: str,
    current_edge_margin_m: Optional[float],
    route_head_min_margin_m: Optional[float],
    decision_round: int,
    sigma_info: float,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Build a noisy environmental hazard signal for one agent at one decision round.

    Prefers the minimum margin across the route head (``route_head_min_margin_m``) as
    the base observation, since that reflects the most safety-critical upcoming segment.
    Falls back to the current edge margin if no route-head data is available.

    Noise is injected via ``inject_signal_noise`` before returning.

    Args:
        agent_id: Vehicle ID (included in signal for traceability).
        sim_t_s: Current simulation time in seconds.
        current_edge: SUMO edge ID where the vehicle is currently located.
        current_edge_margin_m: Fire margin (metres) on the current edge; may be ``None``.
        route_head_min_margin_m: Minimum fire margin across the upcoming route head
            edges; may be ``None`` if route data is unavailable.
        decision_round: Global decision-round counter (used as a history key).
        sigma_info: Noise standard deviation in metres (0 = noiseless).
        rng: Optional seeded RNG for reproducibility.

    Returns:
        A signal dict with fields including ``base_margin_m``, ``observed_margin_m``,
        ``observed_state``, ``noise_delta_m``, and metadata fields.
    """
    # Prefer route-head margin; it captures hazard on the path ahead, not just at feet.
    base_margin = route_head_min_margin_m
    source_metric = "route_head_min_margin_m"
    if base_margin is None:
        base_margin = current_edge_margin_m
        source_metric = "current_edge_margin_m"

    signal = {
        "agent_id": agent_id,
        "sim_t_s": round(float(sim_t_s), 2),
        "decision_round": int(decision_round),
        "current_edge": current_edge,
        "source_metric": source_metric,
        "sigma_info": float(max(0.0, sigma_info)),
        "base_margin_m": None if base_margin is None else round(float(base_margin), 2),
        "observed_margin_m": None,
        "observed_state": "unknown",
    }
    return inject_signal_noise(signal, sigma_info, rng=rng)


def build_social_signal(
    agent_id: str,
    inbox: List[Dict[str, Any]],
    *,
    max_messages: int = 5,
) -> Dict[str, Any]:
    """Aggregate inbox messages into a social hazard-belief signal via keyword voting.

    Processes the most recent ``max_messages`` entries in the agent's inbox.  Each
    message is classified into one hazard category by matching against prioritised
    keyword lists (danger > risky > safe):

        - **danger** keywords : "fire", "blocked", "danger", "smoke", "trapped"
        - **risky**  keywords : "slow", "crowded", "traffic", "risk", "near"
        - **safe**   keywords : "clear", "open", "safe", "passable"

    Messages not matching any keyword are ignored.  The vote counts are normalised
    into a probability triplet representing the aggregate social opinion.  If no votes
    were cast (empty inbox or no keyword matches), returns a uniform prior.

    Args:
        agent_id: Vehicle ID (included for traceability).
        inbox: List of message dicts, each containing at least a ``"message"`` key.
        max_messages: Maximum number of recent messages to consider.

    Returns:
        A social signal dict with:
            - ``message_count``  : Number of messages considered.
            - ``votes``          : Raw vote counts per category.
            - ``social_belief``  : Normalised {p_safe, p_risky, p_danger} triplet.
            - ``dominant_state`` : Category with the highest vote count, or "none".
    """
    considered = list(inbox[-max(1, int(max_messages)):]) if inbox else []
    votes = {"safe": 0, "risky": 0, "danger": 0}

    # Keyword lists ordered by severity; danger is checked first so that a message
    # containing both "fire" and "clear" is conservatively classified as danger.
    danger_words = ("fire", "blocked", "danger", "smoke", "trapped")
    risky_words = ("slow", "crowded", "traffic", "risk", "near")
    safe_words = ("clear", "open", "safe", "passable")

    for item in considered:
        text = str(item.get("message", "")).lower()
        if any(word in text for word in danger_words):
            votes["danger"] += 1
        elif any(word in text for word in risky_words):
            votes["risky"] += 1
        elif any(word in text for word in safe_words):
            votes["safe"] += 1

    total_votes = votes["safe"] + votes["risky"] + votes["danger"]
    if total_votes <= 0:
        # No interpretable messages: return a non-informative uniform prior.
        social_belief = {
            "p_safe": 1.0 / 3.0,
            "p_risky": 1.0 / 3.0,
            "p_danger": 1.0 / 3.0,
        }
        dominant_state = "none"
    else:
        social_belief = {
            "p_safe": votes["safe"] / float(total_votes),
            "p_risky": votes["risky"] / float(total_votes),
            "p_danger": votes["danger"] / float(total_votes),
        }
        dominant_state = max(votes, key=votes.get)

    return {
        "agent_id": agent_id,
        "message_count": len(considered),
        "votes": dict(votes),
        "social_belief": social_belief,
        "dominant_state": dominant_state,
    }
