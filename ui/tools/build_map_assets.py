"""Build the map assets the Setup view draws before a run exists.

The console shows a scenario package on a map the moment an operator selects it,
which cannot wait for SUMO to load a metropolitan network. This script does that
work once, offline, and writes one JSON bundle per package into
``ui/assets/previews/``.

Run it after adding or editing a scenario package::

    python -m ui.tools.build_map_assets                     # every package
    python -m ui.tools.build_map_assets halifax_3town_e0    # one package

It reads the configuration files and the road network and writes only into
``ui/assets/``. It starts no simulation and touches nothing the research
campaigns depend on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agentevac.agents.alert_schedule import AlertSchedule
from agentevac.config_loader import load_map_config, load_spawns, validate_spawn_positions
from ui.bridge import collector, projection

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "ui" / "assets" / "previews"

#: How far beyond the households and fire origins to keep buildings, in degrees. About
#: 2.2 km of latitude, which matches the padding the road layer uses for the incident
#: area, so buildings and streets cover the same ground.
BUILDING_PAD_DEG = 0.02
#: Beyond this many buildings in view the bundle keeps centroids and drops footprints,
#: so a package covering a whole city still loads.  Selection is by centroid either way.
BUILDING_FOOTPRINT_LIMIT = 25000
#: A building further than this from any drivable edge gets no spawn edge, matching the
#: default the offline spawn generator uses.
BUILDING_SNAP_MAX_M = 200.0
#: Grid cell size for the nearest-edge index, in metres.  Large enough that a cell holds
#: several edges and small enough that a query inspects few of them.
SNAP_GRID_CELL_M = 200.0
#: Road classes a household never spawns onto.  A driveway does not meet a motorway, so a
#: building whose nearest road is one of these is matched to the nearest community street
#: instead, and is left unsnapped when there is none in range.  Passenger cars are allowed
#: on all of them, which is why drivability alone is not the right test.
SNAP_EXCLUDED_TYPES = frozenset({
    "highway.motorway",
    "highway.motorway_link",
    "highway.trunk",
    "highway.trunk_link",
})


def _simulation_view(package_id: str) -> SimpleNamespace:
    """Assemble the same fields the bridge reads off a live run.

    Building this view from the configuration files lets the preview reuse the
    geometry code the live map uses, so the Setup view and the Operations view
    cannot drift apart.
    """
    cfg = load_map_config(package_id)
    net_file = REPO_ROOT / str(cfg["map"]["net_file"])
    if not net_file.exists():
        raise FileNotFoundError(f"network file missing for {package_id}: {net_file}")

    # Imported here, not at module scope, so the building-layer helpers stay importable
    # on an interpreter without SUMO on its path.
    import sumolib

    net = sumolib.net.readNet(str(net_file), withInternal=False)

    edge_shape: Dict[str, List[Any]] = {}
    edge_length: Dict[str, float] = {}
    for edge in net.getEdges(withInternal=False):
        lanes = edge.getLanes()
        if not lanes:
            continue
        edge_shape[edge.getID()] = [(float(p[0]), float(p[1])) for p in lanes[0].getShape()]
        edge_length[edge.getID()] = float(edge.getLength())

    spawns = validate_spawn_positions(
        load_spawns(cfg["spawns"], cfg["destinations"]), edge_length
    )
    spawn_edge_by_agent = {str(s[0]): str(s[1]) for s in spawns}
    spawn_midpoint: Dict[str, Any] = {}
    for agent_id, edge_id in spawn_edge_by_agent.items():
        shape = edge_shape.get(edge_id)
        if shape:
            spawn_midpoint[agent_id] = shape[len(shape) // 2]

    return SimpleNamespace(
        net=net,
        EDGE_SHAPE=edge_shape,
        EDGE_LENGTH=edge_length,
        SPAWN_EVENTS=spawns,
        SPAWN_EDGE_BY_AGENT=spawn_edge_by_agent,
        SPAWN_EDGE_MIDPOINT=spawn_midpoint,
        DESTINATION_LIBRARY=cfg["destinations"],
        FIRE_SOURCES=(cfg.get("fires") or {}).get("sources") or [],
        NEW_FIRE_EVENTS=(cfg.get("fires") or {}).get("events") or [],
        ALERT_SCHEDULE=AlertSchedule.from_config(cfg.get("alerts") or None),
    )


def _buildings_file(package_id: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Locate the building polygons for a package.

    ``map.json`` may name one directly as ``buildings_file``.  Otherwise the SUMO
    configuration's ``additional-files`` entry is followed, which is where polyconvert
    output is already wired in.  Returns ``None`` when nothing is found, which is not an
    error, since a package without buildings simply has no building layer.
    """
    declared = (cfg.get("map") or {}).get("buildings_file")
    if declared:
        candidate = REPO_ROOT / str(declared)
        return candidate if candidate.exists() else None

    sumo_cfg = (cfg.get("map") or {}).get("sumo_cfg")
    if not sumo_cfg:
        return None
    cfg_path = REPO_ROOT / str(sumo_cfg)
    if not cfg_path.exists():
        return None
    try:
        root = ET.parse(cfg_path).getroot()
    except ET.ParseError:
        return None
    node = root.find(".//additional-files")
    if node is None or not node.get("value"):
        return None
    for entry in str(node.get("value")).split(","):
        candidate = (cfg_path.parent / entry.strip()).resolve()
        if candidate.exists():
            return candidate
    return None


def _buildings_in_view(
    path: Path, bbox: Optional[List[float]]
) -> List[Dict[str, Any]]:
    """Parse building polygons inside ``bbox`` and return centroids with footprints.

    polyconvert wrote these with ``proj.plain-geo``, so the shapes are already longitude
    and latitude and need no projection.  A building is in view when its centroid is
    inside the box, which is the same rule a selection box in the console would apply.
    """
    if bbox is None:
        return []
    min_lon, min_lat, max_lon, max_lat = bbox
    out: List[Dict[str, Any]] = []
    for _event, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "poly":
            continue
        raw = elem.get("shape", "")
        if not raw:
            elem.clear()
            continue
        points: List[List[float]] = []
        for pair in raw.split():
            parts = pair.split(",")
            if len(parts) < 2:
                continue
            try:
                points.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
        if not points:
            elem.clear()
            continue
        lon = sum(p[0] for p in points) / len(points)
        lat = sum(p[1] for p in points) / len(points)
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            out.append({
                "id": elem.get("id", ""),
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "poly": [[round(p[0], 6), round(p[1], 6)] for p in points],
            })
        elem.clear()
    return out


class EdgeSnapper:
    """Nearest drivable edge for a point, measured to the edge polyline.

    The browser cannot do this, since snapping needs the road network and reading a
    metropolitan one costs about fifteen seconds.  The builder already holds the network,
    so the mapping is computed here once and travels in the bundle.

    Distance is ``polygonOffsetAndDistanceToPoint``, the same call the hazard model uses
    for fire margin, so an authored package and the simulation agree on where a household
    sits.  ``sumolib.net.getNeighboringEdges`` would be the obvious alternative, but
    without the optional ``rtree`` package it degrades to scanning every edge per query,
    which is why this builds a uniform grid instead.
    """

    def __init__(
        self,
        edge_shape: Dict[str, List[Any]],
        drivable: Optional[set] = None,
        cell_m: float = SNAP_GRID_CELL_M,
    ):
        self.cell_m = float(cell_m)
        self.shapes: Dict[str, List[Any]] = {}
        self.grid: Dict[Any, List[str]] = {}
        for edge_id, shape in edge_shape.items():
            if drivable is not None and edge_id not in drivable:
                continue
            if not shape or len(shape) < 2:
                continue
            self.shapes[edge_id] = shape
            for point in shape:
                self.grid.setdefault(self._cell(point[0], point[1]), []).append(edge_id)

    def _cell(self, x: float, y: float):
        return (int(x // self.cell_m), int(y // self.cell_m))

    def nearest(self, x: float, y: float, max_distance_m: float) -> Tuple[Optional[str], Optional[float]]:
        """Return ``(edge_id, distance_m)``, or ``(None, None)`` if nothing is in range.

        Rings of cells are searched outward.  A ring is only worth entering while its
        nearest possible point could still beat the best match found so far, so the
        search stops as soon as the geometry says no closer edge can exist.
        """
        from sumolib import geomhelper

        cx, cy = self._cell(x, y)
        max_ring = max(1, int(max_distance_m // self.cell_m) + 1)
        best_id: Optional[str] = None
        best_dist = float("inf")
        seen: set = set()

        for ring in range(max_ring + 1):
            # Nothing in this ring or beyond can improve on what we already have.
            if best_id is not None and (ring - 1) * self.cell_m > best_dist:
                break
            for gx in range(cx - ring, cx + ring + 1):
                for gy in range(cy - ring, cy + ring + 1):
                    if ring > 0 and abs(gx - cx) != ring and abs(gy - cy) != ring:
                        continue  # interior of the ring, already searched
                    for edge_id in self.grid.get((gx, gy), ()):
                        if edge_id in seen:
                            continue
                        seen.add(edge_id)
                        _, dist = geomhelper.polygonOffsetAndDistanceToPoint(
                            (x, y), self.shapes[edge_id], perpendicular=False
                        )
                        if dist < best_dist:
                            best_dist = float(dist)
                            best_id = edge_id

        if best_id is None or best_dist > max_distance_m:
            return (None, None)
        return (best_id, round(best_dist, 1))


class LocalFrame:
    """A local metric frame about a reference longitude and latitude.

    Snapping has to happen in metres, and the network's own XY is unreachable from
    longitude and latitude because ``net.convertLonLat2XY`` needs ``pyproj``, which is
    not installed.  ``ui.bridge.projection`` supplies the other direction, verified
    against SUMO's ``convertGeo``, so both buildings and edges are brought into this
    frame instead.  The per-degree scales are the WGS84 series evaluated at the
    reference latitude, so over the twenty-odd kilometres a package spans the error is
    far below the metre that matters here.
    """

    def __init__(self, lon0: float, lat0: float):
        import math

        phi = math.radians(lat0)
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.m_per_deg_lat = (
            111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
        )
        self.m_per_deg_lon = (
            111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
        )

    def to_m(self, lon: float, lat: float) -> Tuple[float, float]:
        return (
            (float(lon) - self.lon0) * self.m_per_deg_lon,
            (float(lat) - self.lat0) * self.m_per_deg_lat,
        )


def _drivable_edges(net: Any, exclude_types: Any = SNAP_EXCLUDED_TYPES) -> Optional[set]:
    """Edge IDs a household may spawn onto.

    A lane a passenger car may use, and not a road class a driveway never meets.  The
    second test matters because motorways allow passenger cars, so drivability alone
    would place a household on a limited-access highway whenever one runs nearer than
    the street the building actually fronts.
    """
    if net is None:
        return None
    try:
        out = set()
        for edge in net.getEdges(withInternal=False):
            if exclude_types and str(edge.getType()) in exclude_types:
                continue
            for lane in edge.getLanes():
                if lane.allows("passenger"):
                    out.add(str(edge.getID()))
                    break
        return out
    except Exception:
        return None


def _metric_edge_shapes(
    edge_shape: Dict[str, List[Any]],
    projector: Any,
    frame: "LocalFrame",
    box: List[float],
) -> Dict[str, List[Tuple[float, float]]]:
    """Bring the edges near the building box into the local metric frame.

    A metropolitan network holds far more edges than a package covers, so each edge is
    first tested by its midpoint alone, which is one projection per edge.  Only the
    survivors have every vertex projected.  The test box is padded well beyond the snap
    radius, so an edge reaching into the area is kept even when its midpoint sits outside.
    """
    min_lon, min_lat, max_lon, max_lat = box
    out: Dict[str, List[Tuple[float, float]]] = {}
    for edge_id, shape in edge_shape.items():
        if not shape or len(shape) < 2:
            continue
        mid = shape[len(shape) // 2]
        lonlat = projector.to_lonlat(float(mid[0]), float(mid[1]))
        if lonlat is None:
            continue
        if not (min_lon <= lonlat[0] <= max_lon and min_lat <= lonlat[1] <= max_lat):
            continue
        line = projector.line_to_lonlat(shape)
        if len(line) < 2:
            continue
        out[edge_id] = [frame.to_m(p[0], p[1]) for p in line]
    return out


def _snap_buildings(buildings: List[Dict[str, Any]], view: Any, projector: Any = None) -> Dict[str, int]:
    """Attach ``edge`` and ``edge_dist_m`` to each building, in place.

    Returns a small report of how many snapped and how many found nothing in range.
    """
    edge_shape = getattr(view, "EDGE_SHAPE", None) or {}
    if projector is None or not edge_shape or not buildings:
        for item in buildings:
            item["edge"] = None
            item["edge_dist_m"] = None
        return {"snapped": 0, "unsnapped": len(buildings)}

    lons = [b["lon"] for b in buildings]
    lats = [b["lat"] for b in buildings]
    frame = LocalFrame(sum(lons) / len(lons), sum(lats) / len(lats))
    # Padded well past the snap radius, so an edge entering the area is never missed.
    pad = 0.02
    box = [min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad]

    metric = _metric_edge_shapes(edge_shape, projector, frame, box)
    drivable = _drivable_edges(getattr(view, "net", None))
    if drivable is not None:
        # Strict. An empty result here means no road a household may spawn onto is in
        # range, which is a real answer, so it must not fall back to the unfiltered set
        # and quietly put someone on a motorway.
        metric = {k: v for k, v in metric.items() if k in drivable}

    # Simulation coordinates travel with each building, because a household authored on
    # the map is a longitude and a latitude while spawns.json records simulation XY.
    to_xy = None
    try:
        to_xy = projection.build_inverse_projection(getattr(view, "net", None))
    except Exception:
        to_xy = None

    snapper = EdgeSnapper(metric, drivable=None)
    snapped = 0
    for item in buildings:
        x, y = frame.to_m(item["lon"], item["lat"])
        edge_id, dist = snapper.nearest(x, y, BUILDING_SNAP_MAX_M)
        item["edge"] = edge_id
        item["edge_dist_m"] = dist
        if to_xy is not None:
            sim_x, sim_y = to_xy(item["lon"], item["lat"])
            item["x"] = round(float(sim_x), 2)
            item["y"] = round(float(sim_y), 2)
        if edge_id is not None:
            snapped += 1
    return {"snapped": snapped, "unsnapped": len(buildings) - snapped}


def _attach_buildings(
    preview: Dict[str, Any],
    package_id: str,
    cfg: Dict[str, Any],
    view: Any = None,
    projector: Any = None,
) -> None:
    """Add the building layer to a preview bundle in place.

    The layer is what a selection box in the console would draw and hit-test against.
    It covers the same ground as the incident road band, so a box drawn over the
    households finds the buildings those households live in.
    """
    path = _buildings_file(package_id, cfg)
    if path is None:
        preview["buildings"] = []
        preview["buildings_meta"] = {"source": None, "count": 0, "footprints": False}
        return

    base = preview.get("bbox")
    box = None
    if base:
        box = [
            base[0] - BUILDING_PAD_DEG, base[1] - BUILDING_PAD_DEG,
            base[2] + BUILDING_PAD_DEG, base[3] + BUILDING_PAD_DEG,
        ]
    buildings = _buildings_in_view(path, box)

    footprints = len(buildings) <= BUILDING_FOOTPRINT_LIMIT
    if not footprints:
        for item in buildings:
            item.pop("poly", None)

    snap = _snap_buildings(buildings, view, projector)

    preview["buildings"] = buildings
    preview["buildings_meta"] = {
        "source": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "count": len(buildings),
        "footprints": footprints,
        "bbox": [round(v, 6) for v in box] if box else None,
        "snapped": snap["snapped"],
        "unsnapped": snap["unsnapped"],
        "snap_max_m": BUILDING_SNAP_MAX_M,
    }


def build_package(package_id: str, *, out_dir: Path = ASSETS_DIR) -> Optional[Path]:
    """Write one package's preview bundle. Returns the path, or None on failure."""
    started = time.monotonic()
    print(f"[preview] {package_id}: reading configuration and network", flush=True)
    try:
        view = _simulation_view(package_id)
    except Exception as exc:
        print(f"[preview] {package_id}: skipped, {exc}", flush=True)
        return None

    try:
        convert, report = projection.build_projection(view.net)
    except projection.ProjectionError as exc:
        print(f"[preview] {package_id}: skipped, {exc}", flush=True)
        return None

    projector = collector.GeoProjector(convert, report)
    preview = collector.build_preview(view, projector)
    preview["package"] = package_id
    preview["generated_wall"] = time.time()
    preview["projection"] = report
    try:
        _attach_buildings(preview, package_id, load_map_config(package_id), view=view, projector=projector)
    except Exception as exc:
        print(f"[preview] {package_id}: buildings skipped, {exc}", flush=True)
        preview["buildings"] = []
        preview["buildings_meta"] = {"source": None, "count": 0, "footprints": False}

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{package_id}.json"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(preview, handle, ensure_ascii=False, separators=(",", ":"))
    size_mb = target.stat().st_size / 1e6
    meta = preview["buildings_meta"]
    building_note = f"{meta['count']} buildings"
    if not meta.get("footprints"):
        building_note += " (centroids only)"
    if meta.get("unsnapped"):
        building_note += f", {meta['unsnapped']} unsnapped"
    print(
        f"[preview] {package_id}: {len(preview['households'])} households, "
        f"{len(preview['fire_sources'])} fire sources, "
        f"{len(preview['roads']['features'])} road segments, "
        f"{building_note}, "
        f"{size_mb:.1f} MB in {time.monotonic() - started:.0f} s",
        flush=True,
    )
    return target


def discover_packages() -> List[str]:
    configs = REPO_ROOT / "configs"
    if not configs.is_dir():
        return []
    return sorted(p.name for p in configs.iterdir() if (p / "map.json").exists())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ui.tools.build_map_assets",
        description="Generate the Setup-view map bundles for scenario packages.",
    )
    parser.add_argument("packages", nargs="*", help="Package ids. Defaults to every package.")
    parser.add_argument("--out-dir", default=str(ASSETS_DIR))
    parser.add_argument("--force", action="store_true",
                        help="Rebuild bundles that already exist.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    targets = args.packages or discover_packages()
    if not targets:
        print("no scenario packages found under configs/")
        return 1

    built = 0
    for package_id in targets:
        existing = out_dir / f"{package_id}.json"
        if existing.exists() and not args.force:
            print(f"[preview] {package_id}: already built, pass --force to rebuild", flush=True)
            continue
        if build_package(package_id, out_dir=out_dir) is not None:
            built += 1
    print(f"[preview] wrote {built} bundle(s) into {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
