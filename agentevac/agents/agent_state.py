"""Agent runtime state management for the AgentEvac wildfire evacuation simulator.

This module defines the per-agent in-memory state that persists across decision rounds.
All live agent states are stored in the global ``AGENT_STATES`` dict, which is keyed by
vehicle ID.  The main simulation loop (``agentevac.simulation.main``) creates or retrieves state via
``ensure_agent_state()`` before running the belief-update and departure-decision pipeline.

Psychological profile parameters stored in each agent's ``profile`` dict:
    - ``theta_trust``  : Weight given to social (neighbor) signals vs. own observations [0, 1].
                         Higher values mean the agent trusts peer messages more.
    - ``theta_r``      : Risk threshold; agent departs if ``p_danger > theta_r`` [0, 1].
    - ``theta_u``      : Urgency threshold; agent departs if the urgency term falls below
                         this value (see ``departure_model.py``) [0, 1].
    - ``gamma``        : Discount factor controlling urgency decay over time (≈ 0.99).
                         Each elapsed second reduces urgency by a factor of ``gamma``.
    - ``lambda_e``     : Exposure weight in the route utility function (≥ 0).
                         Larger values make agents more averse to hazardous routes.
    - ``lambda_t``     : Travel-time weight in the route utility function (≥ 0).
                         Larger values make agents prefer shorter travel times.
    - ``neighbor_window_s``      : Recency window for local neighborhood departure observations.
    - ``social_recent_weight``   : Weight on recent neighbor departures when computing
                                   social departure pressure.
    - ``social_total_weight``    : Weight on cumulative departed neighbors when computing
                                   social departure pressure.
    - ``social_trigger``         : Pressure threshold that can trigger social departure.
    - ``social_min_danger``      : Minimum danger floor required before social departure
                                   pressure can trigger departure.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AgentRuntimeState:
    """Holds all mutable per-agent state across decision rounds.

    Attributes:
        agent_id: Unique vehicle identifier matching the SUMO vehicle ID.
        created_sim_t_s: Simulation time (seconds) when this agent was first registered.
        last_sim_t_s: Simulation time of the most recent state update.
        profile: Immutable-ish psychological parameters (theta_trust, theta_r, etc.).
            Populated by ``ensure_agent_state`` and may be overridden by calibration sweeps.
        belief: Current Bayesian belief distribution {p_safe, p_risky, p_danger} plus
            derived fields (entropy, entropy_norm, uncertainty_bucket).
        psychology: Scalar summaries derived from the belief (perceived_risk, confidence).
        signal_history: Bounded list of recent environment signals (noisy margin observations).
            Used by the delay model to replay stale observations.
        social_history: Bounded list of recent social signals derived from inbox messages.
        decision_history: Bounded list of past decision records (predeparture + routing).
            Passed to the LLM as ``agent_self_history`` so agents can avoid repeated mistakes.
        observation_history: Bounded list of system-generated local neighborhood observations.
        has_departed: True once the vehicle has been added to the SUMO simulation.
    """

    agent_id: str
    created_sim_t_s: float
    last_sim_t_s: float
    profile: Dict[str, float] = field(default_factory=dict)
    belief: Dict[str, Any] = field(default_factory=dict)
    psychology: Dict[str, Any] = field(default_factory=dict)
    signal_history: List[Dict[str, Any]] = field(default_factory=list)
    social_history: List[Dict[str, Any]] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    observation_history: List[Dict[str, Any]] = field(default_factory=list)
    has_departed: bool = True
    last_input_hash: Optional[int] = None
    last_llm_choice_idx: Optional[int] = None
    last_llm_reason: Optional[str] = None
    last_llm_action: Optional[str] = None


# Global registry of all agent states, keyed by vehicle ID.
# Populated lazily as vehicles are registered in the simulation.
AGENT_STATES: Dict[str, AgentRuntimeState] = {}


def sample_profile_params(
    agent_id: str,
    means: Dict[str, float],
    spreads: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Sample per-agent profile parameters from truncated normal distributions.

    Each parameter is drawn from ``N(mean, spread)`` and clipped to ``[lo, hi]``.
    When ``spread <= 0`` the mean is returned unchanged (no heterogeneity).

    A deterministic RNG seeded by ``agent_id`` ensures that the same agent always
    receives the same profile regardless of which code path creates it first.

    Args:
        agent_id: Vehicle ID used to seed the per-agent RNG.
        means: Dict of parameter names to population means.
        spreads: Dict of parameter names to population standard deviations.
            Missing keys or values <= 0 disable sampling for that parameter.
        bounds: Dict of parameter names to ``(lo, hi)`` clipping bounds.

    Returns:
        A dict of sampled parameter values, one per key in ``means``.
    """
    rng = random.Random(hash(agent_id))
    result: Dict[str, float] = {}
    for key, mu in means.items():
        sigma = float(spreads.get(key, 0.0))
        lo, hi = bounds.get(key, (mu, mu))
        if sigma <= 0.0:
            result[key] = mu
        else:
            # Rejection-sample from truncated normal (bounded).
            for _ in range(100):
                v = rng.gauss(mu, sigma)
                if lo <= v <= hi:
                    result[key] = round(v, 4)
                    break
            else:
                result[key] = round(max(lo, min(hi, mu)), 4)
    return result


def ensure_agent_state(
    agent_id: str,
    sim_t_s: float,
    *,
    default_theta_trust: float = 0.5,
    default_theta_r: float = 0.45,
    default_theta_u: float = 0.30,
    default_gamma: float = 0.995,
    default_lambda_e: float = 1.0,
    default_lambda_t: float = 0.1,
    default_neighbor_window_s: float = 120.0,
    default_social_recent_weight: float = 0.7,
    default_social_total_weight: float = 0.3,
    default_social_trigger: float = 0.5,
    default_social_min_danger: float = 0.15,
) -> AgentRuntimeState:
    """Retrieve an existing agent state or create a new one with default parameters.

    On first call for a given ``agent_id``, initializes the belief distribution to a
    uniform prior (maximum entropy, ``uncertainty_bucket="High"``) and sets the
    psychology scalars to neutral values.  On subsequent calls, only updates
    ``last_sim_t_s`` and back-fills any missing profile keys via ``setdefault``.

    Args:
        agent_id: Vehicle ID used as the state registry key.
        sim_t_s: Current simulation time in seconds.
        default_theta_trust: Initial social-signal trust weight (see module docstring).
        default_theta_r: Initial risk-departure threshold.
        default_theta_u: Initial urgency-departure threshold.
        default_gamma: Per-second urgency discount factor.
        default_lambda_e: Initial exposure weight for route utility scoring.
        default_lambda_t: Initial travel-time weight for route utility scoring.
        default_neighbor_window_s: Window for recent neighborhood departure observations.
        default_social_recent_weight: Weight on recent departures in social pressure.
        default_social_total_weight: Weight on cumulative departures in social pressure.
        default_social_trigger: Pressure threshold for socially triggered departure.
        default_social_min_danger: Minimum danger probability required before social
            departure pressure can trigger departure.

    Returns:
        The (possibly newly created) ``AgentRuntimeState`` for this vehicle.
    """
    state = AGENT_STATES.get(agent_id)
    if state is None:
        # Uniform belief = maximum entropy for a 3-state distribution.
        uniform_entropy = math.log(3.0)
        state = AgentRuntimeState(
            agent_id=agent_id,
            created_sim_t_s=float(sim_t_s),
            last_sim_t_s=float(sim_t_s),
            profile={
                "theta_trust": float(default_theta_trust),
                "theta_r": float(default_theta_r),
                "theta_u": float(default_theta_u),
                "gamma": float(default_gamma),
                "lambda_e": float(default_lambda_e),
                "lambda_t": float(default_lambda_t),
                "neighbor_window_s": float(default_neighbor_window_s),
                "social_recent_weight": float(default_social_recent_weight),
                "social_total_weight": float(default_social_total_weight),
                "social_trigger": float(default_social_trigger),
                "social_min_danger": float(default_social_min_danger),
            },
            belief={
                "p_safe": 1.0 / 3.0,
                "p_risky": 1.0 / 3.0,
                "p_danger": 1.0 / 3.0,
                "entropy": uniform_entropy,
                "entropy_norm": 1.0,
                "uncertainty_bucket": "High",
            },
            psychology={
                "perceived_risk": 0.5,
                "confidence": 0.0,
            },
        )
        AGENT_STATES[agent_id] = state
    # Back-fill any profile keys added in later code versions without resetting existing ones.
    state.profile.setdefault("theta_trust", float(default_theta_trust))
    state.profile.setdefault("theta_r", float(default_theta_r))
    state.profile.setdefault("theta_u", float(default_theta_u))
    state.profile.setdefault("gamma", float(default_gamma))
    state.profile.setdefault("lambda_e", float(default_lambda_e))
    state.profile.setdefault("lambda_t", float(default_lambda_t))
    state.profile.setdefault("neighbor_window_s", float(default_neighbor_window_s))
    state.profile.setdefault("social_recent_weight", float(default_social_recent_weight))
    state.profile.setdefault("social_total_weight", float(default_social_total_weight))
    state.profile.setdefault("social_trigger", float(default_social_trigger))
    state.profile.setdefault("social_min_danger", float(default_social_min_danger))
    state.last_sim_t_s = float(sim_t_s)
    return state


def _append_bounded(items: List[Dict[str, Any]], value: Dict[str, Any], max_items: int) -> None:
    """Append ``value`` to ``items`` and trim the list to at most ``max_items`` entries.

    Older entries are dropped from the front so that ``items`` always contains the most
    recent ``max_items`` records.

    Args:
        items: The list to append to (mutated in-place).
        value: The record to append (copied shallowly).
        max_items: Maximum number of entries to retain.
    """
    items.append(dict(value))
    if len(items) > max(1, int(max_items)):
        del items[:-max(1, int(max_items))]


def append_signal_history(
    state: AgentRuntimeState,
    signal: Dict[str, Any],
    *,
    max_items: int = 16,
) -> None:
    """Append an environment signal record to the agent's signal history.

    The history is bounded to ``max_items`` entries (default 16).  The delay model in
    ``information_model.apply_signal_delay`` uses this history to retrieve stale
    observations when ``INFO_DELAY_S > 0``.

    Args:
        state: The agent whose history to update.
        signal: The environment signal dict produced by ``sample_environment_signal``.
        max_items: Maximum number of signal records to retain.
    """
    _append_bounded(state.signal_history, signal, max_items)


def append_social_history(
    state: AgentRuntimeState,
    signal: Dict[str, Any],
    *,
    max_items: int = 16,
) -> None:
    """Append a social signal record to the agent's social history.

    Social signals are derived from inbox messages by ``information_model.build_social_signal``.
    Keeping a bounded history supports future analysis of how peer influence evolved.

    Args:
        state: The agent whose history to update.
        signal: The social signal dict produced by ``build_social_signal``.
        max_items: Maximum number of social signal records to retain.
    """
    _append_bounded(state.social_history, signal, max_items)


def append_decision_history(
    state: AgentRuntimeState,
    decision: Dict[str, Any],
    *,
    max_items: int = 32,
) -> None:
    """Append a decision record to the agent's decision history.

    Decision records are passed to the LLM as ``agent_self_history`` so the model can
    avoid repeating previously ineffective choices.  The larger default cap (32) relative
    to signal/social histories reflects the higher value of decision context.

    Args:
        state: The agent whose history to update.
        decision: The decision record (predeparture or routing) to append.
        max_items: Maximum number of decision records to retain.
    """
    _append_bounded(state.decision_history, decision, max_items)


def append_observation_history(
    state: AgentRuntimeState,
    observation: Dict[str, Any],
    *,
    max_items: int = 16,
) -> None:
    """Append a system-generated observation to the agent's observation history.

    These observations are simulator-authored facts, such as nearby household departure
    updates. They are kept separately from peer-authored social messages.

    Args:
        state: The agent whose history to update.
        observation: The observation dict to append.
        max_items: Maximum number of observation records to retain.
    """
    _append_bounded(state.observation_history, observation, max_items)


def snapshot_agent_state(state: AgentRuntimeState) -> Dict[str, Any]:
    """Serialize an ``AgentRuntimeState`` to a plain dict for logging or replay.

    The returned dict is JSON-serializable and contains shallow copies of all mutable
    sub-structures.  Used by ``replay.record_agent_cognition`` to capture a point-in-time
    snapshot of the agent's internal state.

    Args:
        state: The agent state to serialize.

    Returns:
        A JSON-serializable dict representation of the agent state.
    """
    return {
        "agent_id": state.agent_id,
        "created_sim_t_s": float(state.created_sim_t_s),
        "last_sim_t_s": float(state.last_sim_t_s),
        "profile": dict(state.profile),
        "belief": dict(state.belief),
        "psychology": dict(state.psychology),
        "signal_history": [dict(item) for item in state.signal_history],
        "social_history": [dict(item) for item in state.social_history],
        "decision_history": [dict(item) for item in state.decision_history],
        "observation_history": [dict(item) for item in state.observation_history],
        "has_departed": bool(state.has_departed),
    }
