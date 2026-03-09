"""System-generated neighborhood departure observations for pre-departure agents.

This module models passive social observation rather than agent-authored messaging.
When one household departs, the simulator can synthesize an objective update for its
neighbors, such as:

    "Two neighbors have departed to evacuate. Three neighbors are still staying."

The updates are intended to be:
    - factual
    - local
    - time-bounded
    - auditable in replay/history

They can then be used in two ways:
    1. surfaced to prompts as structured context
    2. converted into a scalar social-pressure term for the pre-departure rule
"""

from __future__ import annotations

from typing import Any, Dict, List


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _pluralize(count: int, singular: str, plural: str) -> str:
    return singular if int(count) == 1 else plural


def build_neighbor_map(
    spawn_events: List[tuple],
    scope: str = "same_spawn_edge",
) -> Dict[str, List[str]]:
    """Build a neighbor list for each agent from the spawn-event table."""
    scope_norm = str(scope).strip().lower()
    if scope_norm != "same_spawn_edge":
        raise ValueError(f"Unsupported neighborhood scope: {scope}")

    by_edge: Dict[str, List[str]] = {}
    for event in spawn_events:
        if len(event) < 2:
            continue
        agent_id = str(event[0])
        spawn_edge = str(event[1])
        by_edge.setdefault(spawn_edge, []).append(agent_id)

    neighbor_map: Dict[str, List[str]] = {}
    for peer_ids in by_edge.values():
        for agent_id in peer_ids:
            neighbor_map[agent_id] = [peer for peer in peer_ids if peer != agent_id]
    return neighbor_map


def render_neighborhood_summary(obs: Dict[str, Any]) -> str:
    """Render a neutral, count-based summary sentence for one neighborhood snapshot."""
    neighbor_count = int(obs.get("neighbor_count", 0) or 0)
    recent_departures = int(obs.get("recent_departures_count", 0) or 0)
    still_staying = int(obs.get("still_staying_count", 0) or 0)

    if neighbor_count <= 0:
        return "No neighbors are associated with this observation scope."

    if recent_departures <= 0:
        if still_staying <= 0:
            return "No neighbors have departed recently. No neighbors are still staying."
        staying_word = _pluralize(still_staying, "neighbor is", "neighbors are")
        return (
            f"No neighbors have departed recently. "
            f"{still_staying} {staying_word} still staying."
        )

    departed_word = _pluralize(recent_departures, "neighbor has", "neighbors have")
    if still_staying <= 0:
        return (
            f"{recent_departures} {departed_word} departed to evacuate. "
            f"No neighbors are still staying."
        )

    staying_word = _pluralize(still_staying, "neighbor is", "neighbors are")
    return (
        f"{recent_departures} {departed_word} departed to evacuate. "
        f"{still_staying} {staying_word} still staying."
    )


def summarize_neighborhood_observation(
    agent_id: str,
    sim_t_s: float,
    neighbor_map: Dict[str, List[str]],
    spawn_edge_by_agent: Dict[str, str],
    departure_times: Dict[str, float],
    *,
    scope: str = "same_spawn_edge",
    window_s: float = 120.0,
) -> Dict[str, Any]:
    """Compute a structured local-departure observation for one focal agent."""
    neighbors = list(neighbor_map.get(agent_id, []))
    neighbor_count = len(neighbors)
    obs_time = float(sim_t_s)
    recency_window = max(0.0, float(window_s))

    departed_total_count = 0
    recent_departures_count = 0
    for peer_id in neighbors:
        depart_t = departure_times.get(peer_id)
        if depart_t is None:
            continue
        if float(depart_t) <= obs_time:
            departed_total_count += 1
            if (obs_time - float(depart_t)) <= recency_window:
                recent_departures_count += 1

    still_staying_count = max(0, neighbor_count - departed_total_count)
    recent_departure_fraction = recent_departures_count / float(max(1, neighbor_count))
    departed_total_fraction = departed_total_count / float(max(1, neighbor_count))

    obs: Dict[str, Any] = {
        "available": True,
        "kind": "neighbor_departure_observation",
        "source": "system",
        "subject_agent_id": str(agent_id),
        "scope": str(scope),
        "anchor_spawn_edge": spawn_edge_by_agent.get(agent_id),
        "window_s": round(recency_window, 2),
        "observation_time_s": round(obs_time, 2),
        "neighbor_count": neighbor_count,
        "departed_total_count": departed_total_count,
        "still_staying_count": still_staying_count,
        "recent_departures_count": recent_departures_count,
        "recent_departure_fraction": round(recent_departure_fraction, 4),
        "departed_total_fraction": round(departed_total_fraction, 4),
    }
    obs["summary"] = render_neighborhood_summary(obs)
    return obs


def build_departure_observation_update(
    focal_agent_id: str,
    departed_agent_id: str,
    sim_t_s: float,
    neighbor_map: Dict[str, List[str]],
    spawn_edge_by_agent: Dict[str, str],
    departure_times: Dict[str, float],
    *,
    scope: str = "same_spawn_edge",
    window_s: float = 120.0,
) -> Dict[str, Any]:
    """Build a system observation update for one focal agent after a neighbor departs."""
    obs = summarize_neighborhood_observation(
        focal_agent_id,
        sim_t_s,
        neighbor_map,
        spawn_edge_by_agent,
        departure_times,
        scope=scope,
        window_s=window_s,
    )
    obs["event_time_s"] = round(float(sim_t_s), 2)
    obs["departed_neighbor_id"] = str(departed_agent_id)
    return obs


def compute_social_departure_pressure(
    obs: Dict[str, Any],
    *,
    w_recent: float = 0.7,
    w_total: float = 0.3,
) -> float:
    """Convert a neighborhood observation into a bounded social-pressure scalar."""
    recent_fraction = float(obs.get("recent_departure_fraction", 0.0) or 0.0)
    total_fraction = float(obs.get("departed_total_fraction", 0.0) or 0.0)
    pressure = (float(w_recent) * recent_fraction) + (float(w_total) * total_fraction)
    return round(_clamp(pressure, 0.0, 1.0), 4)
