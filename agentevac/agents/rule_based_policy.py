"""Rule-based (softmax) baseline policy for ablation experiments.

Provides drop-in replacements for the three LLM call sites in
``agentevac.simulation.main``.  The belief equations, utility scores,
departure logic, and SUMO interaction are identical to the LLM agent —
only the decision mechanism differs: a temperature-controlled softmax
over precomputed utility scores replaces the GPT-4o-mini call.

This lets experiments isolate *what value the LLM's language reasoning
adds* on top of the mathematical framework.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from agentevac.agents.agent_state import AgentRuntimeState
from agentevac.agents.departure_model import should_depart_now


def _softmax_sample(utilities: List[float], tau: float, rng: random.Random) -> int:
    """Return an index sampled from a softmax distribution over *utilities*.

    Args:
        utilities: Utility scores (higher = better).
        tau: Temperature parameter.  ``tau <= 0`` → deterministic argmax;
            higher values → more uniform sampling.
        rng: Seeded random number generator for reproducibility.

    Returns:
        Selected index into *utilities*.
    """
    if not utilities:
        raise ValueError("utilities must be non-empty")
    if len(utilities) == 1:
        return 0

    # Deterministic argmax when temperature is zero or negative.
    if tau <= 0.0:
        return max(range(len(utilities)), key=lambda i: utilities[i])

    # Numerically stable softmax: subtract max before exp.
    max_u = max(utilities)
    exps = [math.exp((u - max_u) / tau) for u in utilities]
    total = sum(exps)
    probs = [e / total for e in exps]

    # Weighted random sample.
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1  # float rounding guard


def rule_based_predeparture(
    agent_state: AgentRuntimeState,
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    sim_t_s: float,
    neighborhood_observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rule-based predeparture decision — delegates to the heuristic departure model.

    Returns a dict matching the shape of ``PreDepartureDecisionModel``.
    """
    should_depart, reason = should_depart_now(
        agent_state, belief, psychology, sim_t_s,
        neighborhood_observation=neighborhood_observation,
    )
    return {
        "action": "depart" if should_depart else "wait",
        "reason": reason,
        "situation_summary": "rule_based",
        "conflict_assessment": None,
    }


def rule_based_destination_choice(
    menu: List[Dict[str, Any]],
    reachable_indices: List[int],
    tau: float,
    rng: random.Random,
) -> Tuple[int, str]:
    """Choose a destination via softmax over expected utility scores.

    Args:
        menu: Annotated destination menu (each item has ``expected_utility``).
        reachable_indices: Indices of reachable menu items.
        tau: Softmax temperature.
        rng: Seeded RNG for reproducibility.

    Returns:
        ``(choice_idx, reason)`` where ``choice_idx`` is the menu index
        or ``-1`` if no reachable options exist.
    """
    if not reachable_indices:
        return -1, "no_reachable"

    idx_to_menu = {item["idx"]: item for item in menu}
    utilities = []
    valid_indices = []
    for idx in reachable_indices:
        item = idx_to_menu.get(idx)
        if item is None:
            continue
        u = item.get("expected_utility")
        if u is None:
            continue
        utilities.append(float(u))
        valid_indices.append(idx)

    if not valid_indices:
        return -1, "no_reachable"

    selected_pos = _softmax_sample(utilities, tau, rng)
    return valid_indices[selected_pos], "softmax_utility"


def rule_based_routing_choice(
    menu: List[Dict[str, Any]],
    reachable_indices: List[int],
    current_choice_idx: Optional[int],
    tau: float,
    rng: random.Random,
) -> Tuple[int, str]:
    """Choose a route/destination via softmax, with KEEP logic.

    If the softmax-sampled option equals the agent's current destination,
    returns ``(-1, "softmax_keep")`` to signal no change — mirroring the
    LLM convention where ``choice_index=-1`` means "stay on current route".

    Args:
        menu: Annotated menu (each item has ``expected_utility``).
        reachable_indices: Indices of reachable menu items.
        current_choice_idx: The agent's current destination/route index,
            or ``None`` if unknown.
        tau: Softmax temperature.
        rng: Seeded RNG for reproducibility.

    Returns:
        ``(choice_idx, reason)`` — ``-1`` means KEEP.
    """
    if not reachable_indices:
        return -1, "no_reachable"

    idx_to_menu = {item["idx"]: item for item in menu}
    utilities = []
    valid_indices = []
    for idx in reachable_indices:
        item = idx_to_menu.get(idx)
        if item is None:
            continue
        u = item.get("expected_utility")
        if u is None:
            continue
        utilities.append(float(u))
        valid_indices.append(idx)

    if not valid_indices:
        return -1, "no_reachable"

    selected_pos = _softmax_sample(utilities, tau, rng)
    chosen_idx = valid_indices[selected_pos]

    if chosen_idx == current_choice_idx:
        return -1, "softmax_keep"

    return chosen_idx, "softmax_utility"
