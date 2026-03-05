"""Route and destination utility scoring for evacuation agents.

Each agent evaluates a menu of destinations or pre-defined routes by computing an
expected utility score.  Higher (less negative) scores indicate safer, faster options.
The utility function is:

    U(option) = -[lambda_e * E(option) + lambda_t * C(option)]

where:
    - ``lambda_e`` : exposure-aversion weight from the agent's profile (default 1.0).
    - ``lambda_t`` : travel-time-aversion weight from the agent's profile (default 0.1).
    - ``E(option)`` : expected exposure score (``_expected_exposure``).
    - ``C(option)`` : travel cost in equivalent minutes (``_travel_cost``).

Expected exposure combines four components:
    1. **risk_sum** : Sum of edge-level fire risk scores along the route (or fastest path).
       Scaled by a severity multiplier that increases with ``p_risky`` and ``p_danger``.
    2. **blocked_edges** : Number of edges along the route that are currently inside a
       fire perimeter.  Each blocked edge adds a heavy fixed penalty (8.0) because
       a blocked edge typically makes the route impassable.
    3. **margin_penalty** : Lookup-table penalty based on the closest fire approach
       along the route (see ``_effective_margin_penalty``).
    4. **uncertainty_penalty** : A penalty proportional to ``(1 - confidence)`` that
       discourages fragile choices when the agent is unsure of the hazard.

Annotated menus (``annotate_menu_with_expected_utility``) are used in the *advice_guided*
scenario so the LLM receives pre-computed utility context alongside each option.
"""

from typing import Any, Dict, List


def _num(value: Any, default: float = 0.0) -> float:
    """Safely coerce ``value`` to float, returning ``default`` on failure or ``None``.

    Args:
        value: The value to convert.
        default: Fallback if ``value`` is ``None`` or not numeric.

    Returns:
        ``float(value)`` or ``default``.
    """
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _effective_margin_penalty(min_margin_m: Any) -> float:
    """Return a scalar penalty encoding how close the fire gets to a route.

    The penalty is a non-linear lookup table calibrated so that a fire immediately
    at the route (0 m margin) is 5× worse than a fire that never appears (∞ margin).
    The ``inf``-margin case receives a small non-zero penalty (0.25) to account for
    model uncertainty: even routes with no fire currently nearby may become risky.

    Thresholds:
        - ∞ (no fire detected)      → 0.25  (nominal baseline)
        - ≤ 0 m (inside fire)       → 5.0   (highest risk)
        - ≤ 100 m (very close)      → 3.0
        - ≤ 300 m (near)            → 1.5
        - ≤ 700 m (buffered)        → 0.6
        - > 700 m (clear)           → 0.15  (lowest non-zero risk)

    Args:
        min_margin_m: Minimum fire margin in metres along the route; may be ``None``
            (treated as ∞).

    Returns:
        A non-negative penalty scalar.
    """
    margin = _num(min_margin_m, default=float("inf"))
    if margin == float("inf"):
        return 0.25
    if margin <= 0.0:
        return 5.0
    if margin <= 100.0:
        return 3.0
    if margin <= 300.0:
        return 1.5
    if margin <= 700.0:
        return 0.6
    return 0.15


def _travel_cost(menu_item: Dict[str, Any]) -> float:
    """Estimate travel cost in equivalent minutes for a menu option.

    Prefers ``travel_time_s_fastest_path`` (converted to minutes) when available.
    Falls back to an edge-count heuristic (each edge ≈ 0.25 min) using
    ``len_edges`` or ``len_edges_fastest_path``.

    Args:
        menu_item: A destination or route dict from the menu library.

    Returns:
        Estimated travel cost in minutes (≥ 0).
    """
    travel_time = menu_item.get("travel_time_s_fastest_path")
    if travel_time is not None:
        return _num(travel_time, 0.0) / 60.0
    edge_count = menu_item.get("len_edges")
    if edge_count is None:
        edge_count = menu_item.get("len_edges_fastest_path")
    return _num(edge_count, 0.0) * 0.25


def _expected_exposure(
    menu_item: Dict[str, Any],
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
) -> float:
    """Compute the expected hazard exposure for one route or destination option.

    Formula:
        severity_scale = 1 + 0.8 * p_risky + 1.6 * p_danger + 0.6 * perceived_risk
        uncertainty_penalty = max(0, 1 - confidence) * 0.75
        exposure = risk_sum * severity_scale
                   + blocked_edges * 8.0
                   + margin_penalty(min_margin_m)
                   + uncertainty_penalty

    Weight rationale:
        - ``p_danger`` coefficient (1.6) > ``p_risky`` (0.8): danger is penalised
          more aggressively because it signals imminent threat.
        - ``perceived_risk`` (0.6): a secondary subjective signal that blends with
          the Bayesian posterior.
        - ``blocked_edges * 8.0``: a large fixed penalty; a blocked edge on a route
          often means the route is impassable, so the option should be strongly
          deprioritised.
        - ``uncertainty_penalty * 0.75``: agents that are highly uncertain about the
          hazard should avoid options that are already somewhat risky.

    Args:
        menu_item: A destination or route dict.
        belief: The agent's current Bayesian belief dict.
        psychology: The agent's current psychology dict (perceived_risk, confidence).

    Returns:
        Expected exposure score (≥ 0; higher = more hazardous).
    """
    risk_sum = _num(menu_item.get("risk_sum", menu_item.get("risk_sum_on_fastest_path")), 0.0)
    blocked_edges = _num(menu_item.get("blocked_edges", menu_item.get("blocked_edges_on_fastest_path")), 0.0)
    min_margin_m = menu_item.get("min_margin_m", menu_item.get("min_margin_m_on_fastest_path"))

    p_risky = _num(belief.get("p_risky"), 1.0 / 3.0)
    p_danger = _num(belief.get("p_danger"), 1.0 / 3.0)
    perceived_risk = _num(psychology.get("perceived_risk"), p_danger)
    confidence = _num(psychology.get("confidence"), 0.0)

    # Severity scale: amplifies risk_sum based on how dangerous the agent believes things are.
    severity_scale = 1.0 + (0.8 * p_risky) + (1.6 * p_danger) + (0.6 * perceived_risk)
    # Uncertainty penalty: less confident agents should avoid options that are already risky.
    uncertainty_penalty = max(0.0, 1.0 - confidence) * 0.75

    return (
        risk_sum * severity_scale
        + (blocked_edges * 8.0)
        + _effective_margin_penalty(min_margin_m)
        + uncertainty_penalty
    )


def score_destination_utility(
    menu_item: Dict[str, Any],
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    profile: Dict[str, Any],
) -> float:
    """Compute the utility score for a destination option.

    Utility is the negative weighted sum of expected exposure and travel cost:
        U = -(lambda_e * exposure + lambda_t * travel_cost)

    Higher (less negative) values indicate preferable destinations.

    Args:
        menu_item: A destination dict from ``DESTINATION_LIBRARY`` enriched with
            risk metrics (risk_sum, blocked_edges, min_margin_m, travel_time_s_fastest_path).
        belief: The agent's current Bayesian belief dict.
        psychology: The agent's current psychology dict.
        profile: The agent's profile dict (supplies ``lambda_e``, ``lambda_t``).

    Returns:
        Utility score (negative float; higher = better).
    """
    lambda_e = max(0.0, _num(profile.get("lambda_e"), 1.0))
    lambda_t = max(0.0, _num(profile.get("lambda_t"), 0.1))
    expected_exposure = _expected_exposure(menu_item, belief, psychology)
    travel_cost = _travel_cost(menu_item)
    return -((lambda_e * expected_exposure) + (lambda_t * travel_cost))


def score_route_utility(
    menu_item: Dict[str, Any],
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    profile: Dict[str, Any],
) -> float:
    """Compute the utility score for a route option.

    Identical formula to ``score_destination_utility``; kept as a separate function
    to allow future divergence (e.g., additional route-specific penalty terms).

    Args:
        menu_item: A route dict from ``ROUTE_LIBRARY`` enriched with risk metrics.
        belief: The agent's current Bayesian belief dict.
        psychology: The agent's current psychology dict.
        profile: The agent's profile dict (supplies ``lambda_e``, ``lambda_t``).

    Returns:
        Utility score (negative float; higher = better).
    """
    lambda_e = max(0.0, _num(profile.get("lambda_e"), 1.0))
    lambda_t = max(0.0, _num(profile.get("lambda_t"), 0.1))
    expected_exposure = _expected_exposure(menu_item, belief, psychology)
    travel_cost = _travel_cost(menu_item)
    return -((lambda_e * expected_exposure) + (lambda_t * travel_cost))


def annotate_menu_with_expected_utility(
    menu: List[Dict[str, Any]],
    *,
    mode: str,
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Annotate each menu option in-place with its expected utility and component breakdown.

    For *destination* mode, unreachable options (``reachable=False``) receive
    ``expected_utility=None`` and a minimal component dict to signal their exclusion.

    The annotated menu is later filtered by ``scenarios.filter_menu_for_scenario`` so
    that utility scores are only visible to agents in the *advice_guided* regime.

    Args:
        menu: List of destination or route dicts (mutated in-place).
        mode: ``"destination"`` or ``"route"`` — selects the scoring function.
        belief: The agent's current Bayesian belief dict.
        psychology: The agent's current psychology dict.
        profile: The agent's profile dict (supplies ``lambda_e``, ``lambda_t``).

    Returns:
        The same ``menu`` list, with each item updated to include:
            - ``expected_utility``  : Scalar utility score or ``None`` if unreachable.
            - ``utility_components``: Dict with lambda_e, lambda_t, expected_exposure,
                                      travel_cost (and ``reachable=False`` if unreachable).
    """
    for item in menu:
        if mode == "destination":
            if not item.get("reachable", False):
                # Unreachable destinations get null utility to signal exclusion to the LLM.
                item["expected_utility"] = None
                item["utility_components"] = {
                    "lambda_e": max(0.0, _num(profile.get("lambda_e"), 1.0)),
                    "lambda_t": max(0.0, _num(profile.get("lambda_t"), 0.1)),
                    "reachable": False,
                }
                continue
            expected_exposure = _expected_exposure(item, belief, psychology)
            travel_cost = _travel_cost(item)
            utility = score_destination_utility(item, belief, psychology, profile)
        else:
            expected_exposure = _expected_exposure(item, belief, psychology)
            travel_cost = _travel_cost(item)
            utility = score_route_utility(item, belief, psychology, profile)

        item["expected_utility"] = round(utility, 4)
        item["utility_components"] = {
            "lambda_e": round(max(0.0, _num(profile.get("lambda_e"), 1.0)), 4),
            "lambda_t": round(max(0.0, _num(profile.get("lambda_t"), 0.1)), 4),
            "expected_exposure": round(expected_exposure, 4),
            "travel_cost": round(travel_cost, 4),
        }
    return menu
