"""Fire forecast construction and route-head risk summarization.

This module transforms raw fire geometry data (circles with positions and radii from
``Traci_GPT2.active_fires``) into structured forecast objects and natural-language
briefings that are embedded in the LLM prompt.

**Fire forecast** (``build_fire_forecast``):
    Compares current fire circles against projected circles (queried at
    ``sim_t + FORECAST_HORIZON_S``) to derive growth metrics: max and average radius
    growth per fire, and any new fire IDs that appear in the projected snapshot.

**Edge risk** (``estimate_edge_forecast_risk``):
    Queries the caller-supplied ``edge_risk_fn`` (a thin wrapper around
    ``Traci_GPT2.compute_edge_risk_for_fires``) to obtain ``(blocked, risk_score,
    margin_m)`` for a single edge, then classifies the result into a margin band.

**Route-head summary** (``summarize_route_forecast``):
    Evaluates the first ``max_edges`` non-junction edges of the agent's planned route,
    finding the minimum margin (worst-case proximity to fire) and counting blocked edges.

**Briefing** (``render_forecast_briefing``):
    Assembles a concise natural-language sentence from the above data, providing the
    LLM with a human-readable situational summary without requiring it to parse raw
    numbers.
"""

from typing import Any, Callable, Dict, List, Optional


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    """Round a float to ``digits`` decimal places, returning ``None`` for missing values.

    Args:
        value: The value to round; may be ``None``.
        digits: Number of decimal places.

    Returns:
        Rounded float, or ``None`` if ``value`` is ``None`` or non-numeric.
    """
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _margin_band(margin_m: Optional[float]) -> str:
    """Classify a fire margin into a named proximity band.

    Used in forecast briefings and menu annotations to provide a consistent, human-
    readable description of how close a fire is to a road edge.

    Thresholds:
        - ``None``   : "unknown"
        - ≤ 0 m      : "inside_predicted_fire"  (fire has overtaken the edge)
        - ≤ 100 m    : "very_close"
        - ≤ 300 m    : "near"
        - ≤ 700 m    : "buffered"
        - > 700 m    : "clear"

    Args:
        margin_m: Minimum distance (metres) from fire edge to road edge.

    Returns:
        A string band label.
    """
    if margin_m is None:
        return "unknown"
    if margin_m <= 0.0:
        return "inside_predicted_fire"
    if margin_m <= 100.0:
        return "very_close"
    if margin_m <= 300.0:
        return "near"
    if margin_m <= 700.0:
        return "buffered"
    return "clear"


def build_fire_forecast(
    sim_t_s: float,
    current_fires: List[Dict[str, Any]],
    projected_fires: List[Dict[str, Any]],
    horizon_s: float,
) -> Dict[str, Any]:
    """Compute a fire growth forecast by comparing current and projected fire circles.

    Each fire is identified by an ``"id"`` key.  For fires present in both snapshots,
    growth is ``projected_r - current_r``.  For fires that appear only in the projected
    snapshot (new ignitions within the horizon), their full projected radius is counted
    as growth.

    Args:
        sim_t_s: Current simulation time in seconds.
        current_fires: List of active fire dicts at ``sim_t_s`` (each with "id", "r").
        projected_fires: List of active fire dicts at ``sim_t_s + horizon_s``.
        horizon_s: Forecast horizon in seconds.

    Returns:
        A forecast summary dict with:
            - ``horizon_s``                : Forecast horizon in seconds.
            - ``generated_at_s``           : Simulation time of forecast generation.
            - ``current_fire_count``       : Number of currently active fires.
            - ``projected_fire_count``     : Number of fires projected at horizon.
            - ``new_projected_fire_ids``   : IDs of fires that ignite within horizon.
            - ``max_projected_radius_m``   : Largest fire radius at horizon.
            - ``max_radius_growth_m``      : Largest radius growth across all fires.
            - ``avg_radius_growth_m``      : Average radius growth across all fires.
    """
    current_by_id = {str(item.get("id")): item for item in current_fires}
    projected_by_id = {str(item.get("id")): item for item in projected_fires}

    growth_values: List[float] = []
    for fire_id, projected in projected_by_id.items():
        current = current_by_id.get(fire_id)
        if current is None:
            # New fire ignition within the forecast horizon: count full radius as growth.
            growth_values.append(float(projected.get("r", 0.0)))
            continue
        growth_values.append(float(projected.get("r", 0.0)) - float(current.get("r", 0.0)))

    max_growth = max(growth_values) if growth_values else 0.0
    avg_growth = (sum(growth_values) / float(len(growth_values))) if growth_values else 0.0
    new_fire_ids = sorted([fire_id for fire_id in projected_by_id.keys() if fire_id not in current_by_id])

    return {
        "horizon_s": _round_or_none(horizon_s, 2),
        "generated_at_s": _round_or_none(sim_t_s, 2),
        "current_fire_count": len(current_fires),
        "projected_fire_count": len(projected_fires),
        "new_projected_fire_ids": new_fire_ids,
        "max_projected_radius_m": _round_or_none(
            max((float(item.get("r", 0.0)) for item in projected_fires), default=0.0),
            2,
        ),
        "max_radius_growth_m": _round_or_none(max_growth, 2),
        "avg_radius_growth_m": _round_or_none(avg_growth, 2),
    }


def estimate_edge_forecast_risk(
    edge_id: str,
    edge_risk_fn: Callable[[str], Any],
) -> Dict[str, Any]:
    """Query the risk of a single edge against the projected fire field.

    Delegates the geometry computation to ``edge_risk_fn``, which is typically a
    closure around ``Traci_GPT2.compute_edge_risk_for_fires`` evaluated at the
    *projected* fire positions (``sim_t + FORECAST_HORIZON_S``).

    Args:
        edge_id: SUMO edge ID to evaluate.
        edge_risk_fn: Callable that accepts an edge ID and returns
            ``(blocked: bool, risk_score: float, margin_m: float)``.

    Returns:
        A dict with:
            - ``edge_id``    : The queried edge.
            - ``blocked``    : True if the fire overlaps the edge at the forecast horizon.
            - ``risk_score`` : Exponential decay score ∈ [0, 1].
            - ``margin_m``   : Distance (metres) from fire edge to road edge.
            - ``band``       : Margin band label from ``_margin_band``.
    """
    blocked, risk_score, margin_m = edge_risk_fn(edge_id)
    return {
        "edge_id": edge_id,
        "blocked": bool(blocked),
        "risk_score": _round_or_none(risk_score, 4),
        "margin_m": _round_or_none(margin_m, 2),
        "band": _margin_band(_round_or_none(margin_m, 2)),
    }


def summarize_route_forecast(
    route_edges: List[str],
    edge_risk_fn: Callable[[str], Any],
    max_edges: int,
) -> Dict[str, Any]:
    """Evaluate the first ``max_edges`` non-junction edges of a route against the forecast.

    Junction edges (IDs starting with ``":"``) are skipped because they are short
    connector segments without meaningful geometry in the SUMO network.

    Finds the minimum margin (most dangerous approach) and counts blocked edges to
    provide a worst-case picture of the upcoming route segment.

    Args:
        route_edges: Ordered list of edge IDs along the planned route.
        edge_risk_fn: Callable returning ``(blocked, risk_score, margin_m)`` per edge.
        max_edges: Maximum number of route-head edges to evaluate.

    Returns:
        A dict with:
            - ``head_edges_evaluated`` : Number of non-junction edges checked.
            - ``blocked_edges``        : Number of blocked edges found.
            - ``min_margin_m``         : Minimum margin across evaluated edges (metres).
            - ``max_risk_score``       : Maximum risk score across evaluated edges.
            - ``band``                 : Margin band for the minimum margin.
    """
    checked = 0
    blocked_edges = 0
    min_margin: Optional[float] = None
    max_risk = 0.0

    for edge_id in route_edges:
        if checked >= max(1, int(max_edges)):
            break
        if not edge_id or edge_id.startswith(":"):
            # Skip SUMO internal junction connector edges.
            continue
        blocked, risk_score, margin_m = edge_risk_fn(edge_id)
        checked += 1
        if blocked:
            blocked_edges += 1
        if margin_m is not None:
            margin_val = float(margin_m)
            if min_margin is None or margin_val < min_margin:
                min_margin = margin_val
        if risk_score is not None:
            max_risk = max(max_risk, float(risk_score))

    return {
        "head_edges_evaluated": checked,
        "blocked_edges": blocked_edges,
        "min_margin_m": _round_or_none(min_margin, 2),
        "max_risk_score": _round_or_none(max_risk, 4),
        "band": _margin_band(_round_or_none(min_margin, 2)),
    }


def render_forecast_briefing(
    agent_id: str,
    forecast: Dict[str, Any],
    belief: Dict[str, Any],
    edge_forecast: Dict[str, Any],
    route_forecast: Dict[str, Any],
) -> str:
    """Render a one-sentence natural-language situational briefing for the LLM prompt.

    Combines three information streams into a concise, human-readable sentence:
        - The overall hazard tone (based on ``p_danger`` and uncertainty).
        - The current edge status (edge band or "may be overtaken").
        - The route-head status (band or blocked-segment count).

    This briefing is embedded in the LLM prompt as a plain-English complement to the
    structured forecast JSON, making key risk indicators immediately legible to the model.

    Args:
        agent_id: Vehicle ID (currently unused; retained for future per-agent logging).
        forecast: Fire forecast summary from ``build_fire_forecast``.
        belief: The agent's current Bayesian belief dict (for p_danger, uncertainty_bucket).
        edge_forecast: Per-edge forecast dict from ``estimate_edge_forecast_risk``.
        route_forecast: Route-head forecast dict from ``summarize_route_forecast``.

    Returns:
        A one-sentence briefing string.
    """
    horizon_s = int(round(float(forecast.get("horizon_s") or 0.0)))
    edge_band = edge_forecast.get("band", "unknown").replace("_", " ")
    route_band = route_forecast.get("band", "unknown").replace("_", " ")
    blocked_edges = int(route_forecast.get("blocked_edges") or 0)
    p_danger = float(belief.get("p_danger") or 0.0)
    uncertainty = str(belief.get("uncertainty_bucket") or "High")

    if blocked_edges > 0:
        route_clause = f"{blocked_edges} route-head segment(s) may be blocked"
    else:
        route_clause = f"route head looks {route_band}"

    if edge_forecast.get("blocked"):
        edge_clause = "your current edge may be overtaken"
    else:
        edge_clause = f"your current edge looks {edge_band}"

    # Select a tone word based on belief danger probability and uncertainty.
    if p_danger >= 0.5:
        tone = "Forecast suggests the threat is building"
    elif uncertainty == "High":
        tone = "Forecast is uncertain, but conditions may tighten"
    else:
        tone = "Forecast suggests a manageable window"

    return f"{tone} within {horizon_s}s: {edge_clause}, and {route_clause}."
