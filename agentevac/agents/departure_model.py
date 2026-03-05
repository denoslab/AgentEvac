"""Departure decision rule for pre-evacuation agents.

An agent waits at its spawn edge until this module's ``should_depart_now`` function
returns ``True``.  The function implements a three-clause OR rule:

    1. **Risk threshold** (``risk_threshold``):
       The agent's estimated danger probability exceeds its personal threshold
       ``theta_r``.  This is the primary, reactive trigger.

    2. **Urgency decay** (``urgency_threshold``):
       The urgency term ``gamma^elapsed_s * p_safe`` falls below ``theta_u``.
       As time passes, the discount factor ``gamma`` (< 1) erodes the "stay safe"
       signal so that an agent that has been waiting a long time will eventually
       feel compelled to act — even if ``p_danger`` is still low.  This captures
       the real-world observation that people eventually evacuate out of a growing
       general unease rather than a discrete danger trigger.

    3. **Low-confidence precaution** (``low_confidence_precaution``):
       If the agent is highly uncertain (``confidence < 0.15``) *and* the danger
       probability has reached at least 60 % of ``theta_r``, it departs
       pre-emptively.  This prevents agents from staying frozen in high-entropy
       situations where they should arguably err on the side of caution.
"""

from typing import Any, Dict, Tuple

from agentevac.agents.agent_state import AgentRuntimeState


def should_depart_now(
    agent_state: AgentRuntimeState,
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    sim_t_s: float,
) -> Tuple[bool, str]:
    """Evaluate whether an agent should depart from its spawn edge at the current tick.

    Applies three departure clauses in priority order (first match wins):
        1. ``risk_threshold``        : ``p_danger > theta_r``
        2. ``urgency_threshold``     : ``gamma^elapsed_s * p_safe < theta_u``
        3. ``low_confidence_precaution``: ``confidence < 0.15`` and
                                         ``p_danger >= max(0.20, theta_r * 0.6)``

    If none of the clauses fire, returns ``(False, "wait")``.

    Args:
        agent_state: The agent's runtime state (supplies profile parameters and creation time).
        belief: Current Bayesian belief dict with keys "p_danger", "p_safe", etc.
        psychology: Current psychology dict with key "confidence".
        sim_t_s: Current simulation time in seconds.

    Returns:
        A ``(should_depart, reason)`` tuple where ``reason`` is one of
        "risk_threshold", "urgency_threshold", "low_confidence_precaution", or "wait".
    """
    p_danger = float(belief.get("p_danger", 0.0))
    p_safe = float(belief.get("p_safe", 0.0))
    theta_r = float(agent_state.profile.get("theta_r", 0.45))
    theta_u = float(agent_state.profile.get("theta_u", 0.30))
    gamma = float(agent_state.profile.get("gamma", 0.995))
    created_t = float(agent_state.created_sim_t_s)
    elapsed_s = max(0.0, float(sim_t_s) - created_t)

    # Clause 1: Direct danger threshold.
    if p_danger > theta_r:
        return True, "risk_threshold"

    # Clause 2: Urgency decay — the longer the agent waits, the lower gamma^t * p_safe
    # becomes, eventually dropping below theta_u.  This forces eventual action even in
    # low-information environments.
    urgency_term = (gamma ** elapsed_s) * p_safe
    if urgency_term < theta_u:
        return True, "urgency_threshold"

    # Clause 3: High uncertainty + non-trivial danger probability → err on the side of
    # caution.  Only fires when the agent is genuinely uncertain (low confidence) and
    # danger is already elevated relative to its own risk threshold.
    confidence = float(psychology.get("confidence", 0.0))
    if confidence < 0.15 and p_danger >= max(0.20, theta_r * 0.6):
        return True, "low_confidence_precaution"

    return False, "wait"
