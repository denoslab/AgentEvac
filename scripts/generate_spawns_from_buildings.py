#!/usr/bin/env python3
"""Generate spawn configuration from building polygons and a SUMO network.

For each building polygon, computes the centroid, finds the nearest drivable
edge in the SUMO network, and writes a ``spawns.json`` file in either compact
or detailed format.

Usage examples::

    # One agent per building, compact format (default)
    python scripts/generate_spawns_from_buildings.py \
        --net sumo/halifax.net.xml \
        --buildings sumo/Halifax_buildings.xml \
        --output configs/halifax/spawns.json

    # 3 agents per building, within a SUMO XY bounding box
    python scripts/generate_spawns_from_buildings.py \
        --net sumo/halifax.net.xml \
        --buildings sumo/Halifax_buildings.xml \
        --output configs/halifax/spawns.json \
        --mode per-building --count 3 \
        --bbox "10000,20000,30000,35000"

    # 2 agents per unique edge (deduplicated)
    python scripts/generate_spawns_from_buildings.py \
        --net sumo/halifax.net.xml \
        --buildings sumo/Halifax_buildings.xml \
        --output configs/halifax/spawns.json \
        --mode per-edge --count 2

    # Detailed format output
    python scripts/generate_spawns_from_buildings.py \
        --net sumo/halifax.net.xml \
        --buildings sumo/Halifax_buildings.xml \
        --output configs/halifax/spawns.json \
        --format detailed --dest-edge "E#S0"

Modes::

    per-building : One group entry per building (buildings sharing an edge are
                   merged into one group with summed counts).
    per-edge     : One group entry per unique edge, regardless of how many
                   buildings map to it.

The ``--count`` flag sets how many agents spawn per building (``per-building``
mode) or per unique edge (``per-edge`` mode).  Default is 1.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# SUMO_HOME setup
# ---------------------------------------------------------------------------
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    # Try common locations
    for candidate in ["/usr/share/sumo/tools", "/opt/sumo/tools"]:
        if os.path.isdir(candidate):
            sys.path.append(candidate)
            break

import sumolib  # noqa: E402 (must follow path setup)

# Edge types that allow passenger vehicles (cars).
_DRIVABLE_TYPES = {
    "highway.residential",
    "highway.tertiary",
    "highway.secondary",
    "highway.unclassified",
    "highway.primary",
    "highway.motorway",
    "highway.motorway_link",
    "highway.primary_link",
    "highway.secondary_link",
    "highway.tertiary_link",
    "highway.living_street",
    "highway.trunk",
    "highway.trunk_link",
    "highway.service",
}


# ---------------------------------------------------------------------------
# Building polygon parsing
# ---------------------------------------------------------------------------

def parse_buildings(buildings_path: str) -> List[Dict]:
    """Parse building polygons and compute centroids.

    Args:
        buildings_path: Path to the SUMO ``<additional>`` XML file containing
            ``<poly>`` elements (e.g., output of ``polyconvert``).

    Returns:
        List of dicts with keys ``id``, ``lon``, ``lat`` (centroid coordinates).
    """
    buildings = []
    tree = ET.iterparse(buildings_path, events=("end",))
    for _, elem in tree:
        if elem.tag != "poly":
            continue
        poly_id = elem.get("id", "")
        shape_str = elem.get("shape", "")
        if not shape_str:
            elem.clear()
            continue

        # Parse shape points (lon,lat pairs separated by spaces)
        points = []
        for pair in shape_str.strip().split():
            parts = pair.split(",")
            if len(parts) >= 2:
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    points.append((lon, lat))
                except ValueError:
                    continue

        if not points:
            elem.clear()
            continue

        # Centroid (average of vertices)
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        buildings.append({"id": poly_id, "lon": cx, "lat": cy})
        elem.clear()

    return buildings


def filter_buildings_by_bbox(
    buildings: List[Dict],
    net,
    bbox_xy: Tuple[float, float, float, float],
) -> List[Dict]:
    """Filter buildings whose SUMO XY centroid falls outside a bounding box.

    Args:
        buildings: List of building dicts with ``lon``, ``lat``.
        net: sumolib network object (used for lon/lat → XY conversion).
        bbox_xy: Bounding box in SUMO XY coordinates: ``(x1, y1, x2, y2)``.

    Returns:
        Filtered list of building dicts.
    """
    x1, y1, x2, y2 = bbox_xy
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    result = []
    for b in buildings:
        x, y = net.convertLonLat2XY(b["lon"], b["lat"])
        if x_min <= x <= x_max and y_min <= y <= y_max:
            result.append(b)
    return result


# ---------------------------------------------------------------------------
# Nearest-edge lookup (KD-tree accelerated)
# ---------------------------------------------------------------------------

def _build_edge_index(net, drivable_types):
    """Build a KD-tree of drivable edge midpoints for fast nearest-neighbor lookup.

    Returns:
        (kdtree, edge_ids, edge_midpoints) where edge_ids[i] corresponds to the
        i-th point in the KD-tree.
    """
    from scipy.spatial import cKDTree

    edge_ids = []
    midpoints = []
    for e in net.getEdges(withInternal=False):
        if e.getType() not in drivable_types:
            continue
        # Check that at least one lane allows passenger
        allows_passenger = False
        for lane in e.getLanes():
            if lane.allows("passenger"):
                allows_passenger = True
                break
        if not allows_passenger:
            continue

        # Use midpoint of the first lane's shape
        shape = e.getLanes()[0].getShape()
        if not shape:
            continue
        mid_idx = len(shape) // 2
        mx, my = float(shape[mid_idx][0]), float(shape[mid_idx][1])
        edge_ids.append(e.getID())
        midpoints.append((mx, my))

    if not midpoints:
        raise RuntimeError("No drivable edges found in the network.")

    import numpy as np
    points = np.array(midpoints)
    tree = cKDTree(points)
    return tree, edge_ids, points


def find_nearest_edges(
    net,
    buildings: List[Dict],
    max_distance_m: float,
    drivable_types=None,
) -> List[Dict]:
    """For each building, find the nearest drivable edge.

    Args:
        net: sumolib network object.
        buildings: List of building dicts with ``lon``, ``lat``.
        max_distance_m: Skip buildings farther than this from any edge.
        drivable_types: Set of edge type strings to consider.

    Returns:
        List of dicts with ``building_id``, ``edge_id``, ``distance_m``.
    """
    import numpy as np

    if drivable_types is None:
        drivable_types = _DRIVABLE_TYPES

    print(f"[SPAWNS] Building edge spatial index for {len(buildings)} buildings...")
    tree, edge_ids, _ = _build_edge_index(net, drivable_types)

    # Convert all building centroids to SUMO XY
    xy_points = []
    for b in buildings:
        x, y = net.convertLonLat2XY(b["lon"], b["lat"])
        xy_points.append((x, y))

    query = np.array(xy_points)
    distances, indices = tree.query(query, k=1)

    results = []
    skipped = 0
    for i, b in enumerate(buildings):
        dist = float(distances[i])
        if dist > max_distance_m:
            skipped += 1
            continue
        results.append({
            "building_id": b["id"],
            "edge_id": edge_ids[indices[i]],
            "distance_m": round(dist, 1),
        })

    print(f"[SPAWNS] Matched {len(results)} buildings to edges "
          f"(skipped {skipped} beyond {max_distance_m}m)")
    return results


# ---------------------------------------------------------------------------
# Spawn generation
# ---------------------------------------------------------------------------

def generate_spawn_config(
    matches: List[Dict],
    mode: str,
    count: int,
    dest_edge: Optional[str],
    output_format: str,
):
    """Convert building-to-edge matches into spawn configuration.

    Args:
        matches: Output of ``find_nearest_edges``.
        mode: ``"per-building"`` or ``"per-edge"``.
        count: Agents per building (per-building mode) or per edge (per-edge mode).
        dest_edge: Default destination edge. If None, omitted from output
            (will use destinations.json default at load time).
        output_format: ``"compact"`` or ``"detailed"``.

    Returns:
        JSON-serializable spawn config (list for detailed, dict for compact).
    """
    # Aggregate by edge
    edge_building_counts: Dict[str, int] = defaultdict(int)
    for m in matches:
        edge_building_counts[m["edge_id"]] += 1

    if mode == "per-building":
        # Each building contributes `count` agents → edge total = buildings * count
        groups = []
        for edge_id, n_buildings in sorted(edge_building_counts.items()):
            total = n_buildings * count
            entry = {"edge": edge_id, "count": total}
            if dest_edge:
                entry["dest_edge"] = dest_edge
            groups.append(entry)
    elif mode == "per-edge":
        # Ignore how many buildings map here; each unique edge gets `count` agents
        groups = []
        for edge_id in sorted(edge_building_counts.keys()):
            entry = {"edge": edge_id, "count": count}
            if dest_edge:
                entry["dest_edge"] = dest_edge
            groups.append(entry)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    total_agents = sum(g["count"] for g in groups)
    total_edges = len(groups)
    print(f"[SPAWNS] Mode={mode} count={count} → "
          f"{total_agents} agents across {total_edges} edges")

    if output_format == "compact":
        result = {"groups": groups}
        if dest_edge:
            result["default_dest_edge"] = dest_edge
        return result

    # Detailed format: expand groups into per-agent dicts
    palette = [
        [255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255],
        [255, 125, 0, 255], [125, 255, 0, 255], [0, 255, 255, 255],
        [255, 255, 0, 255], [0, 125, 255, 255], [125, 0, 255, 255],
        [255, 0, 255, 255],
    ]
    agents = []
    for g in groups:
        edge = g["edge"]
        for i in range(1, g["count"] + 1):
            agent = {
                "veh_id": f"{edge}_{i}",
                "spawn_edge": edge,
                "dest_edge": dest_edge or "",
                "depart_time": 0.0,
                "lane": "first",
                "pos": str(10.0 + 10.0 * (i - 1)),
                "speed": "max",
                "color": palette[(i - 1) % len(palette)],
            }
            agents.append(agent)
    return agents


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bbox(raw: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not raw:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must have 4 comma-separated values: x1,y1,x2,y2")
    return (parts[0], parts[1], parts[2], parts[3])


def main():
    parser = argparse.ArgumentParser(
        description="Generate spawn configuration from building polygons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--net", required=True, help="Path to SUMO .net.xml file.")
    parser.add_argument("--buildings", required=True, help="Path to buildings polygon XML.")
    parser.add_argument("--output", required=True, help="Output spawns JSON path.")
    parser.add_argument(
        "--mode", choices=["per-building", "per-edge"], default="per-building",
        help="per-building: count agents per building. per-edge: count agents per unique edge. (default: per-building)",
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Agents per building (per-building mode) or per edge (per-edge mode). (default: 1)",
    )
    parser.add_argument(
        "--format", choices=["compact", "detailed"], default="compact", dest="output_format",
        help="Output format. (default: compact)",
    )
    parser.add_argument(
        "--max-distance", type=float, default=200.0,
        help="Skip buildings farther than this (metres) from any drivable edge. (default: 200)",
    )
    parser.add_argument(
        "--bbox",
        help="Bounding box in SUMO XY coordinates: x1,y1,x2,y2. "
             "Only buildings whose centroid (converted to XY) falls inside are included. "
             "Read the coordinates from the SUMO GUI status bar.",
    )
    parser.add_argument(
        "--dest-edge",
        help="Default destination edge for all agents. If omitted, uses destinations.json at load time.",
    )
    parser.add_argument(
        "--extra-types", nargs="*", default=[],
        help="Additional edge types to consider drivable (e.g., highway.track).",
    )
    args = parser.parse_args()

    # 1. Parse buildings
    print(f"[SPAWNS] Parsing buildings from {args.buildings}...")
    buildings = parse_buildings(args.buildings)
    print(f"[SPAWNS] Parsed {len(buildings)} buildings total")

    if not buildings:
        print("[SPAWNS] No buildings found. Check building file.")
        return 1

    # 2. Load network
    print(f"[SPAWNS] Loading network {args.net}...")
    net = sumolib.net.readNet(args.net, withInternal=False)

    # 3. Apply bbox filter in SUMO XY coordinates
    bbox = _parse_bbox(args.bbox)
    if bbox:
        buildings = filter_buildings_by_bbox(buildings, net, bbox)
        print(f"[SPAWNS] {len(buildings)} buildings within bbox "
              f"x=[{min(bbox[0],bbox[2]):.0f},{max(bbox[0],bbox[2]):.0f}] "
              f"y=[{min(bbox[1],bbox[3]):.0f},{max(bbox[1],bbox[3]):.0f}]")
        if not buildings:
            print("[SPAWNS] No buildings in bbox. Check coordinates in SUMO GUI.")
            return 1

    # 4. Find nearest edges
    drivable = set(_DRIVABLE_TYPES)
    for extra in args.extra_types:
        drivable.add(extra.strip())

    matches = find_nearest_edges(
        net, buildings,
        max_distance_m=args.max_distance,
        drivable_types=drivable,
    )

    if not matches:
        print("[SPAWNS] No buildings matched to edges. Try increasing --max-distance.")
        return 1

    # 5. Generate spawn config
    config = generate_spawn_config(
        matches,
        mode=args.mode,
        count=args.count,
        dest_edge=args.dest_edge,
        output_format=args.output_format,
    )

    # 6. Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"[SPAWNS] Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
