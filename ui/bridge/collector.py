"""Builds console payloads out of live simulation state.

Every function here runs on the simulation thread, which is the only thread
allowed to touch TraCI or to read the simulator's mutable dictionaries. The
results are plain dictionaries that the HTTP layer serves without ever reaching
back into the simulation.

Nothing in this module writes to simulation state. Reads are defensive so a
simulator that grows or renames an internal field degrades to a thinner payload
instead of stopping a run.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Household lifecycle as the console shows it.
STATUS_WAITING = "waiting"
STATUS_EVACUATING = "evacuating"
STATUS_ARRIVED = "arrived"


def _get(module: Any, name: str, default: Any = None) -> Any:
    return getattr(module, name, default)


class GeoProjector:
    """Converts simulation coordinates to longitude and latitude.

    The conversion itself lives in :mod:`ui.bridge.projection`. This wrapper adds
    rounding to roughly a tenth of a metre, which keeps the payloads small, and
    tolerance for points a projection cannot place.
    """

    def __init__(self, convert: Callable[[float, float], Tuple[float, float]], report: Optional[Dict[str, Any]] = None):
        self._convert = convert
        self.report: Dict[str, Any] = dict(report or {})

    @property
    def available(self) -> bool:
        return self._convert is not None

    def to_lonlat(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        try:
            lon, lat = self._convert(float(x), float(y))
        except Exception:
            return None
        if not (math.isfinite(lon) and math.isfinite(lat)):
            return None
        return round(lon, 6), round(lat, 6)

    def line_to_lonlat(self, shape: Sequence[Sequence[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for point in shape:
            converted = self.to_lonlat(point[0], point[1])
            if converted is not None:
                out.append([converted[0], converted[1]])
        return out


def build_preview(module: Any, projector: GeoProjector) -> Dict[str, Any]:
    """Assemble the static geography of a run, built once after the net loads.

    Returns spawn households, destinations, fire sources with their ignition
    schedule, alert areas, and the road geometry around the incident. The console
    draws all of it before the first vehicle moves.
    """
    spawn_events: Iterable[Sequence[Any]] = _get(module, "SPAWN_EVENTS", []) or []
    spawn_midpoint: Dict[str, Tuple[float, float]] = _get(module, "SPAWN_EDGE_MIDPOINT", {}) or {}
    spawn_edge_by_agent: Dict[str, str] = _get(module, "SPAWN_EDGE_BY_AGENT", {}) or {}
    edge_shape: Dict[str, Sequence[Sequence[float]]] = _get(module, "EDGE_SHAPE", {}) or {}
    destinations: List[Dict[str, Any]] = _get(module, "DESTINATION_LIBRARY", []) or []
    schedule = _get(module, "ALERT_SCHEDULE")

    households: List[Dict[str, Any]] = []
    for spawn in spawn_events:
        agent_id = str(spawn[0])
        xy = spawn_midpoint.get(agent_id)
        if xy is None:
            shape = edge_shape.get(str(spawn[1]))
            xy = shape[len(shape) // 2] if shape else None
        lonlat = projector.to_lonlat(*xy) if xy else None
        if lonlat is None:
            continue
        households.append({
            "id": agent_id,
            "lon": lonlat[0],
            "lat": lonlat[1],
            "spawn_edge": str(spawn[1]),
            "dest_edge": str(spawn[2]),
        })

    dest_points: List[Dict[str, Any]] = []
    for entry in destinations:
        edge_id = str(entry.get("edge", ""))
        shape = edge_shape.get(edge_id)
        if not shape:
            continue
        lonlat = projector.to_lonlat(*shape[len(shape) // 2])
        if lonlat is None:
            continue
        dest_points.append({
            "name": str(entry.get("name", edge_id)),
            "edge": edge_id,
            "lon": lonlat[0],
            "lat": lonlat[1],
        })

    fire_sources: List[Dict[str, Any]] = []
    raw_fires = list(_get(module, "FIRE_SOURCES", []) or []) + list(_get(module, "NEW_FIRE_EVENTS", []) or [])
    for src in raw_fires:
        lonlat = projector.to_lonlat(float(src["x"]), float(src["y"]))
        if lonlat is None:
            continue
        fire_sources.append({
            "id": str(src["id"]),
            "lon": lonlat[0],
            "lat": lonlat[1],
            "t0_s": float(src.get("t0", 0.0)),
            "r0_m": float(src.get("r0", 0.0)),
            "growth_m_per_s": float(src.get("growth_m_per_s", 0.0)),
            "max_r_m": float(src.get("max_r_m", 0.0)) or None,
        })
    fire_sources.sort(key=lambda item: item["t0_s"])

    areas: List[Dict[str, Any]] = []
    if schedule is not None:
        try:
            ordered = schedule.ordered_areas()
        except Exception:
            ordered = {}
        members_by_area: Dict[str, List[str]] = {}
        for agent_id, edge_id in spawn_edge_by_agent.items():
            try:
                for area_name in schedule.areas_for_edge(edge_id):
                    members_by_area.setdefault(area_name, []).append(str(agent_id))
            except Exception:
                continue
        for area_name, spec in sorted(ordered.items(), key=lambda kv: kv[1].get("order_t_s", 0.0)):
            member_ids = sorted(members_by_area.get(area_name, []))
            points = [
                projector.to_lonlat(*spawn_midpoint[aid])
                for aid in member_ids
                if aid in spawn_midpoint
            ]
            points = [p for p in points if p is not None]
            areas.append({
                "name": area_name,
                "order_t_s": float(spec.get("order_t_s", 0.0)),
                "channel": str(spec.get("channel", "")),
                "households": len(member_ids),
                "member_ids": member_ids,
                "hull": _convex_hull(points),
            })

    alert_events: List[Dict[str, Any]] = []
    if schedule is not None:
        try:
            for event in schedule.scheduled_events():
                alert_events.append({
                    "id": str(getattr(event, "event_id", "") or getattr(event, "id", "")),
                    "issue_time_s": float(getattr(event, "issue_time_s", 0.0)),
                    "instruction": str(getattr(event, "instruction", "")),
                    "areas": [str(a) for a in getattr(event, "areas", []) or []],
                })
        except Exception:
            alert_events = []
        alert_events.sort(key=lambda item: item["issue_time_s"])

    # Road geometry is selected in simulation coordinates, because projecting every
    # edge of a metropolitan network before filtering would cost minutes.
    incident_xy = _xy_bounds(
        [spawn_midpoint[h["id"]] for h in households if h["id"] in spawn_midpoint]
        + [(float(f["x"]), float(f["y"])) for f in raw_fires]
    )
    reach_xy = _xy_bounds(
        ([] if incident_xy is None else [(incident_xy[0], incident_xy[1]), (incident_xy[2], incident_xy[3])])
        + [
            tuple(edge_shape[str(d.get("edge", ""))][len(edge_shape[str(d.get("edge", ""))]) // 2])
            for d in destinations
            if str(d.get("edge", "")) in edge_shape and edge_shape[str(d.get("edge", ""))]
        ]
    )
    roads = _build_road_geometry(
        _get(module, "net"),
        edge_shape,
        projector,
        incident_xy=_pad_xy(incident_xy, 2500.0),
        reach_xy=_pad_xy(reach_xy, 4000.0),
    )

    return {
        "households": households,
        "destinations": dest_points,
        "fire_sources": fire_sources,
        "areas": areas,
        "alert_events": alert_events,
        "roads": roads,
        "bbox": _bbox_of(households + fire_sources),
        "reach_bbox": _bbox_of(households + dest_points + fire_sources),
    }


def _xy_bounds(points: Iterable[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _pad_xy(
    box: Optional[Tuple[float, float, float, float]], pad_m: float
) -> Optional[Tuple[float, float, float, float]]:
    if box is None:
        return None
    return box[0] - pad_m, box[1] - pad_m, box[2] + pad_m, box[3] + pad_m


def _bbox_of(points: List[Dict[str, Any]], pad_deg: float = 0.02) -> Optional[List[float]]:
    lons = [p["lon"] for p in points if "lon" in p]
    lats = [p["lat"] for p in points if "lat" in p]
    if not lons or not lats:
        return None
    return [
        round(min(lons) - pad_deg, 6),
        round(min(lats) - pad_deg, 6),
        round(max(lons) + pad_deg, 6),
        round(max(lats) + pad_deg, 6),
    ]


#: Roads at or above this free-flow speed are kept outside the incident area, so the
#: console shows the corridors that connect the fire ground to the shelters without
#: drawing every residential street of a metropolitan network.
ARTERIAL_SPEED_MPS = 16.6


def _build_road_geometry(
    net: Any,
    edge_shape: Dict[str, Sequence[Sequence[float]]],
    projector: GeoProjector,
    *,
    incident_xy: Optional[Tuple[float, float, float, float]],
    reach_xy: Optional[Tuple[float, float, float, float]],
    max_edges: int = 40000,
) -> Dict[str, Any]:
    """Convert road geometry around the incident into a line collection.

    Two bands are kept. Inside the incident box every street is drawn, because that
    is where households live and where the fire moves. Between there and the
    shelters only arterials survive, which is what an operator needs to read the
    evacuation corridors. Selection happens in simulation coordinates so a
    metropolitan network costs a numeric comparison per edge instead of a
    projection.
    """
    features: List[Dict[str, Any]] = []
    if not edge_shape:
        return {"type": "FeatureCollection", "features": features}

    speeds: Dict[str, float] = {}
    if net is not None:
        try:
            for edge in net.getEdges(withInternal=False):
                speeds[str(edge.getID())] = float(edge.getSpeed())
        except Exception:
            speeds = {}

    for edge_id, shape in edge_shape.items():
        if len(features) >= max_edges:
            break
        if not shape or len(shape) < 2:
            continue
        in_incident = _shape_in_box(shape, incident_xy)
        if not in_incident:
            if not _shape_in_box(shape, reach_xy):
                continue
            if speeds.get(str(edge_id), 0.0) < ARTERIAL_SPEED_MPS:
                continue
        line = projector.line_to_lonlat(shape)
        if len(line) < 2:
            continue
        features.append({
            "type": "Feature",
            "properties": {"c": 1 if in_incident else 0},
            "geometry": {"type": "LineString", "coordinates": line},
        })
    return {"type": "FeatureCollection", "features": features}


def _shape_in_box(
    shape: Sequence[Sequence[float]], box: Optional[Tuple[float, float, float, float]]
) -> bool:
    if box is None:
        return True
    min_x, min_y, max_x, max_y = box
    for point in shape:
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
            return True
    return False


def _convex_hull(points: List[Tuple[float, float]]) -> List[List[float]]:
    """Return the convex hull of a point set as a closed ring, or an empty list.

    Alert areas are defined by edge membership, so their outline is drawn from
    the households they contain. Fewer than three points has no interior, so the
    console falls back to drawing the member households alone.
    """
    unique = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(unique) < 3:
        return []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    ring = lower[:-1] + upper[:-1]
    if len(ring) < 3:
        return []
    ring.append(ring[0])
    return [[x, y] for x, y in ring]


def build_snapshot(
    module: Any,
    projector: GeoProjector,
    *,
    sim_t_s: float,
    step_idx: int,
    intent: Dict[str, Any],
    round_progress: Dict[str, Any],
    area_members: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Assemble one continuous-state frame for the console."""
    agent_live_status: Dict[str, Dict[str, Any]] = _get(module, "agent_live_status", {}) or {}
    spawn_midpoint: Dict[str, Tuple[float, float]] = _get(module, "SPAWN_EDGE_MIDPOINT", {}) or {}
    spawn_events: Iterable[Sequence[Any]] = _get(module, "SPAWN_EVENTS", []) or []
    awareness: Dict[str, Dict[str, Any]] = _get(module, "AWARENESS_LOG", {}) or {}
    metrics = _get(module, "metrics")
    schedule = _get(module, "ALERT_SCHEDULE")
    spawn_edge_by_agent: Dict[str, str] = _get(module, "SPAWN_EDGE_BY_AGENT", {}) or {}

    arrival_times: Dict[str, float] = getattr(metrics, "_arrival_times", {}) if metrics else {}
    depart_times: Dict[str, float] = getattr(metrics, "_depart_times", {}) if metrics else {}
    fire_contact = set(getattr(metrics, "_fire_contact_agents", set()) if metrics else set())

    agents: List[Dict[str, Any]] = []
    counts = {
        "total": 0,
        STATUS_WAITING: 0,
        STATUS_EVACUATING: 0,
        STATUS_ARRIVED: 0,
        "fire_contact": len(fire_contact),
        "aware": len(awareness),
    }

    for spawn in spawn_events:
        agent_id = str(spawn[0])
        counts["total"] += 1
        live = agent_live_status.get(agent_id) or {}
        arrived = agent_id in arrival_times
        active = bool(live.get("active")) and not arrived

        if arrived:
            status = STATUS_ARRIVED
        elif active:
            status = STATUS_EVACUATING
        else:
            status = STATUS_WAITING
        counts[status] += 1

        if active and live.get("pos_xy"):
            xy = live["pos_xy"]
        else:
            xy = spawn_midpoint.get(agent_id)
        lonlat = projector.to_lonlat(*xy) if xy else None
        if lonlat is None:
            continue

        record: Dict[str, Any] = {
            "id": agent_id,
            "lon": lonlat[0],
            "lat": lonlat[1],
            "status": status,
            "aware": agent_id in awareness,
        }
        if status == STATUS_EVACUATING:
            record["edge"] = live.get("current_edge")
        if agent_id in fire_contact:
            record["fire_contact"] = True
        agents.append(record)

    fires: List[Dict[str, Any]] = []
    active_fires = _get(module, "active_fires")
    if callable(active_fires):
        try:
            for fire in active_fires(sim_t_s):
                lonlat = projector.to_lonlat(float(fire["x"]), float(fire["y"]))
                if lonlat is None:
                    continue
                fires.append({
                    "id": str(fire["id"]),
                    "lon": lonlat[0],
                    "lat": lonlat[1],
                    "r_m": round(float(fire["r"]), 1),
                })
        except Exception:
            fires = []

    alerts = _alert_state(schedule, sim_t_s)
    areas = _area_progress(area_members, depart_times, arrival_times, schedule, sim_t_s, spawn_edge_by_agent)

    return {
        "sim_t_s": round(float(sim_t_s), 2),
        "step_idx": int(step_idx),
        "paused": bool(intent.get("paused")),
        "speed_target": intent.get("speed"),
        "round": round_progress,
        "counts": counts,
        "agents": agents,
        "fires": fires,
        "alerts": alerts,
        "areas": areas,
    }


def _alert_state(schedule: Any, sim_t_s: float) -> Dict[str, Any]:
    if schedule is None:
        return {"issued": [], "pending": [], "next": None}
    try:
        events = schedule.scheduled_events()
    except Exception:
        return {"issued": [], "pending": [], "next": None}

    issued: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for event in sorted(events, key=lambda e: getattr(e, "issue_time_s", 0.0)):
        row = {
            "id": str(getattr(event, "event_id", "") or getattr(event, "id", "")),
            "issue_time_s": float(getattr(event, "issue_time_s", 0.0)),
            "instruction": str(getattr(event, "instruction", "")),
            "areas": [str(a) for a in getattr(event, "areas", []) or []],
        }
        (issued if row["issue_time_s"] <= sim_t_s else pending).append(row)
    return {
        "issued": issued,
        "pending": pending,
        "next": pending[0] if pending else None,
    }


def _area_progress(
    area_members: Dict[str, List[str]],
    depart_times: Dict[str, float],
    arrival_times: Dict[str, float],
    schedule: Any,
    sim_t_s: float,
    spawn_edge_by_agent: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Per-community clearance as it stands at ``sim_t_s``."""
    rows: List[Dict[str, Any]] = []
    ordered: Dict[str, Dict[str, Any]] = {}
    if schedule is not None:
        try:
            ordered = schedule.ordered_areas()
        except Exception:
            ordered = {}
    for area_name, members in area_members.items():
        spec = ordered.get(area_name, {})
        order_t = spec.get("order_t_s")
        rows.append({
            "name": area_name,
            "households": len(members),
            "departed": sum(1 for aid in members if aid in depart_times),
            "arrived": sum(1 for aid in members if aid in arrival_times),
            "order_t_s": float(order_t) if order_t is not None else None,
            "ordered": order_t is not None and sim_t_s >= float(order_t),
            "channel": str(spec.get("channel", "")),
        })
    rows.sort(key=lambda row: (row["order_t_s"] is None, row["order_t_s"] or 0.0, row["name"]))
    return rows


def build_area_members(module: Any) -> Dict[str, List[str]]:
    """Map each alert area to the households whose home edge belongs to it."""
    schedule = _get(module, "ALERT_SCHEDULE")
    spawn_edge_by_agent: Dict[str, str] = _get(module, "SPAWN_EDGE_BY_AGENT", {}) or {}
    members: Dict[str, List[str]] = {}
    if schedule is None:
        return members
    for agent_id, edge_id in spawn_edge_by_agent.items():
        try:
            areas = schedule.areas_for_edge(edge_id)
        except Exception:
            continue
        for area_name in areas:
            members.setdefault(str(area_name), []).append(str(agent_id))
    for area_name in members:
        members[area_name].sort()
    return members
