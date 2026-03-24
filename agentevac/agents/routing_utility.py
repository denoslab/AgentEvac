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

Expected exposure combines four components (in ``alert_guided`` and ``advice_guided``):
    1. **risk_sum** : Sum of edge-level fire risk scores along the route (or fastest path).
       Scaled by a severity multiplier that increases with ``p_risky`` and ``p_danger``.
    2. **blocked_edges** : Number of edges along the route that are currently inside a
       fire perimeter.  Each blocked edge adds a heavy fixed penalty (8.0) because
       a blocked edge typically makes the route impassable.
    3. **margin_penalty** : Lookup-table penalty based on the closest fire approach
       along the route (see ``_effective_margin_penalty``).
    4. **uncertainty_penalty** : A penalty proportional to ``(1 - confidence)`` that
       discourages fragile choices when the agent is unsure of the hazard.

In ``no_notice`` mode, agents lack route-specific fire data.  Exposure is instead
estimated from the agent's general belief state scaled by route length — longer routes
mean more time exposed to whatever danger the agent perceives.

Annotated menus (``annotate_menu_with_expected_utility``) are computed for all three
scenarios so the LLM always receives a utility score.  The *precision* of the exposure
estimate varies by information regime: belief-only (no_notice), current fire state
(alert_guided), or current fire state with full route-head data (advice_guided).
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
        - ∞ (no fire detected)       → 0.25  (nominal baseline)
        - ≤ 0 m (inside fire)        → 5.0   (highest risk)
        - ≤ 1200 m (very close)      → 3.0
        - ≤ 2500 m (near)            → 1.5
        - ≤ 5000 m (buffered)        → 0.6
        - > 5000 m (clear)           → 0.15  (lowest non-zero risk)

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
    if margin <= 1200.0:
        return 3.0
    if margin <= 2500.0:
        return 1.5
    if margin <= 5000.0:
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


def _observation_based_exposure(
    menu_item: Dict[str, Any],
    belief: Dict[str, Any],
    psychology: Dict[str, Any],
) -> float:
    """Estimate hazard exposure when route-specific fire data is unavailable.

    Used in the ``no_notice`` scenario where agents have only their own noisy
    observation of the current edge.  Without per-route fire metrics (risk_sum,
    blocked_edges, min_margin_m), exposure is derived from the agent's general
    belief state scaled by route length:

        hazard_level = 0.3 * p_risky + 0.7 * p_danger + 0.4 * perceived_risk
        length_factor = len_edges * 0.15
        exposure = hazard_level * length_factor + uncertainty_penalty

    Longer routes are penalised more because a longer route means more time
    spent driving through a potentially hazardous environment.  The coefficients
    prioritise ``p_danger`` (0.7) over ``p_risky`` (0.3) to maintain consistency
    with the severity weighting in ``_expected_exposure``.

    Args:
        menu_item: A destination or route dict.
        belief: The agent's current Bayesian belief dict.
        psychology: The agent's current psychology dict (perceived_risk, confidence).

    Returns:
        Expected exposure score (>= 0; higher = more hazardous).
    """
    p_risky = _num(belief.get("p_risky"), 1.0 / 3.0)
    p_danger = _num(belief.get("p_danger"), 1.0 / 3.0)
    perceived_risk = _num(psychology.get("perceived_risk"), p_danger)
    confidence = _num(psychology.get("confidence"), 0.0)

    hazard_level = 0.3 * p_risky + 0.7 * p_danger + 0.4 * perceived_risk
    len_edges = _num(
        menu_item.get("len_edges", menu_item.get("len_edges_fastest_path")),
        1.0,
    )
    length_factor = len_edges * 0.15
    uncertainty_penalty = max(0.0, 1.0 - confidence) * 0.75

    return hazard_level * length_factor + uncertainty_penalty


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
    scenario: str = "advice_guided",
) -> List[Dict[str, Any]]:
    """Annotate each menu option in-place with its expected utility and component breakdown.

    For *destination* mode, unreachable options (``reachable=False``) receive
    ``expected_utility=None`` and a minimal component dict to signal their exclusion.

    The ``scenario`` parameter controls which exposure function is used:

    - ``"no_notice"``: ``_observation_based_exposure`` — uses only the agent's belief
      state and route length (no route-specific fire data).
    - ``"alert_guided"`` / ``"advice_guided"``: ``_expected_exposure`` — uses route-
      specific fire metrics (risk_sum, blocked_edges, min_margin_m).

    Args:
        menu: List of destination or route dicts (mutated in-place).
        mode: ``"destination"`` or ``"route"`` — selects the scoring function.
        belief: The agent's current Bayesian belief dict.
        psychology: The agent's current psychology dict.
        profile: The agent's profile dict (supplies ``lambda_e``, ``lambda_t``).
        scenario: Active information regime (``"no_notice"``, ``"alert_guided"``,
            or ``"advice_guided"``).  Controls which exposure function is used.

    Returns:
        The same ``menu`` list, with each item updated to include:
            - ``expected_utility``  : Scalar utility score or ``None`` if unreachable.
            - ``utility_components``: Dict with lambda_e, lambda_t, expected_exposure,
                                      travel_cost (and ``reachable=False`` if unreachable).
    """
    use_observation_exposure = str(scenario).strip().lower() == "no_notice"
    exposure_fn = _observation_based_exposure if use_observation_exposure else _expected_exposure

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

        lambda_e = max(0.0, _num(profile.get("lambda_e"), 1.0))
        lambda_t = max(0.0, _num(profile.get("lambda_t"), 0.1))
        expected_exposure = exposure_fn(item, belief, psychology)
        travel_cost = _travel_cost(item)
        utility = -((lambda_e * expected_exposure) + (lambda_t * travel_cost))

        item["expected_utility"] = round(utility, 4)
        item["utility_components"] = {
            "lambda_e": round(lambda_e, 4),
            "lambda_t": round(lambda_t, 4),
            "expected_exposure": round(expected_exposure, 4),
            "travel_cost": round(travel_cost, 4),
        }
    return menu
