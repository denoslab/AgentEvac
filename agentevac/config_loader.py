"""Map configuration loader for AgentEvac.

Loads network-specific configuration (spawns, fires, destinations, routes) from
JSON files under ``configs/<map_name>/``.  This allows switching between maps
(e.g., Lytton → Halifax) with a single ``--map`` CLI flag.

Expected directory layout::

    configs/<map_name>/
        map.json           — net_file, sumo_cfg
        spawns.json        — vehicle spawn events (detailed or compact format)
        fires.json         — fire sources and mid-simulation ignition events
        destinations.json  — destination menu for CONTROL_MODE="destination"
        routes.json        — route menu for CONTROL_MODE="route"

Spawn formats
-------------
**Detailed** (list of per-agent dicts)::

    [
      {"veh_id": "veh1_1", "spawn_edge": "42006672", "dest_edge": "E#S2", ...},
      ...
    ]

**Compact** (dict with ``"groups"`` key)::

    {
      "groups": [
        {"edge": "42006672", "count": 3},
        {"edge": "42006514#4", "count": 2, "dest_edge": "E#S0"},
        ...
      ],
      "default_dest_edge": "E#S2"
    }

Compact groups are expanded into the same tuple format by ``expand_spawn_groups``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Spawn tuple type alias for readability
SpawnTuple = Tuple[str, str, str, float, str, str, str, Tuple[int, int, int, int]]

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# 10-colour palette matching spawn_events.py, cycled for auto-assignment.
_COLOR_PALETTE: List[Tuple[int, int, int, int]] = [
    (255, 0, 0, 255),      # red
    (0, 0, 255, 255),       # blue
    (0, 255, 0, 255),       # green
    (255, 125, 0, 255),     # orange
    (125, 255, 0, 255),     # spring
    (0, 255, 255, 255),     # cyan
    (255, 255, 0, 255),     # yellow
    (0, 125, 255, 255),     # ocean
    (125, 0, 255, 255),     # violet
    (255, 0, 255, 255),     # magenta
]

# Default spacing between agents on the same edge (metres).
_DEFAULT_POS_SPACING_M = 10.0
# Minimum margin from the end of the edge (metres).
_DEFAULT_POS_MARGIN_M = 5.0
# Starting position for the first agent on an edge (metres).
_DEFAULT_POS_START_M = 10.0


def load_map_config(map_name: str) -> Dict[str, Any]:
    """Load all JSON config files for a given map name.

    Args:
        map_name: Directory name under ``configs/`` (e.g., ``"lytton"``).

    Returns:
        Dict with keys ``"map"``, ``"spawns"``, ``"fires"``, ``"destinations"``,
        ``"routes"``, ``"alerts"`` — each holding the parsed JSON content.  Missing
        optional files (``routes.json``, ``alerts.json``) default to empty list/dict.

    Raises:
        FileNotFoundError: If the map directory does not exist.
    """
    map_dir = _CONFIGS_DIR / map_name
    if not map_dir.is_dir():
        raise FileNotFoundError(
            f"Map config directory not found: {map_dir}. "
            f"Available maps: {[p.name for p in _CONFIGS_DIR.iterdir() if p.is_dir()]}"
        )

    cfg: Dict[str, Any] = {}
    for name, required in [
        ("map", True),
        ("spawns", True),
        ("fires", True),
        ("destinations", True),
        ("routes", False),
        ("alerts", False),
        ("corridors", False),
    ]:
        path = map_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                cfg[name] = json.load(f)
        elif required:
            raise FileNotFoundError(f"Required config file missing: {path}")
        else:
            cfg[name] = []
    return cfg


def spawns_to_tuples(
    spawns: List[Dict[str, Any]],
    home_points_out: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[SpawnTuple]:
    """Convert spawn dicts from JSON into the legacy tuple format used by main.py.

    Args:
        spawns: List of spawn event dicts from ``spawns.json``.
        home_points_out: Optional dict filled with ``{veh_id: (x, y)}`` for any agent
            carrying a ``home_xy`` building centroid in SUMO XY coordinates.  Left alone
            when ``None``, so callers that do not need home geometry are unaffected.

    Returns:
        List of ``(veh_id, spawn_edge, dest_edge, depart_time, lane, pos, speed, color)``
        tuples matching the format expected by the simulation loop.
    """
    result = []
    for s in spawns:
        color = tuple(s["color"]) if "color" in s else (255, 0, 0, 255)
        veh_id = str(s["veh_id"])
        if home_points_out is not None:
            point = _parse_home_xy(s.get("home_xy"))
            if point is not None:
                home_points_out[veh_id] = point
        result.append((
            veh_id,
            str(s["spawn_edge"]),
            str(s["dest_edge"]),
            float(s["depart_time"]),
            str(s.get("lane", "first")),
            str(s.get("pos", "20")),
            str(s.get("speed", "max")),
            color,
        ))
    return result


def _parse_home_xy(raw: Any) -> Optional[Tuple[float, float]]:
    """Coerce a ``home_xy`` entry into an ``(x, y)`` pair, or ``None`` if unusable."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError):
        return None


def expand_spawn_groups(
    groups: List[Dict[str, Any]],
    default_dest_edge: str,
    home_points_out: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[SpawnTuple]:
    """Expand compact spawn groups into the full tuple list.

    Each group dict has the form::

        {"edge": "<edge_id>", "count": <int>, ...}

    Optional per-group overrides:
        ``dest_edge``        — override default destination edge.
        ``id_prefix``        — override auto-generated ID prefix (default: edge ID).
        ``depart_interval``  — seconds between agents (default: 0.0, all simultaneous).
        ``lane``             — SUMO departure lane (default: ``"first"``).
        ``speed``            — SUMO departure speed (default: ``"max"``).
        ``color``            — fixed RGBA for all agents in group (default: palette cycle).
        ``home_xy``          — list of ``[x, y]`` building centroids in SUMO XY, one per
                               agent in ``count`` order.  Entries beyond the list length,
                               and groups without the key, simply carry no home point.
        ``building_id``      — list of source building IDs parallel to ``home_xy``, kept
                               for provenance and for later building-level area selection.

    Agent IDs are ``"<edge_id>_1"``, ``"<edge_id>_2"``, etc.  If the same edge
    appears in multiple groups, a group suffix ``"_g<n>"`` is appended to avoid
    collisions.

    Positions are auto-staggered starting at 10 m with 10 m spacing.  Overflow
    is handled later by ``validate_spawn_positions`` once edge lengths are known.

    Args:
        groups: List of compact group dicts.
        default_dest_edge: Fallback destination edge (e.g., first entry in
            ``destinations.json``).
        home_points_out: Optional dict filled with ``{veh_id: (x, y)}`` from each group's
            ``home_xy`` list.  Filled in the same loop that generates the IDs, so a home
            point can never drift onto the wrong agent.  Left alone when ``None``.

    Returns:
        List of spawn tuples in the same format as ``spawns_to_tuples``.
    """
    result: List[SpawnTuple] = []
    # Track which edge IDs have been seen to detect duplicates.
    edge_occurrence: Dict[str, int] = {}

    for group in groups:
        edge = str(group["edge"])
        count = int(group["count"])
        if count < 1:
            continue

        # Detect duplicate edges across groups.
        occurrence = edge_occurrence.get(edge, 0)
        edge_occurrence[edge] = occurrence + 1
        needs_group_suffix = occurrence > 0

        prefix = str(group.get("id_prefix", edge))
        if needs_group_suffix:
            prefix = f"{prefix}_g{occurrence + 1}"

        dest = str(group.get("dest_edge", default_dest_edge))
        interval = float(group.get("depart_interval", 0.0))
        lane = str(group.get("lane", "first"))
        speed = str(group.get("speed", "max"))
        fixed_color = tuple(group["color"]) if "color" in group else None
        home_xy = group.get("home_xy") or []

        for i in range(1, count + 1):
            veh_id = f"{prefix}_{i}"
            depart_time = interval * (i - 1)
            pos = str(_DEFAULT_POS_START_M + _DEFAULT_POS_SPACING_M * (i - 1))
            color = fixed_color or _COLOR_PALETTE[(i - 1) % len(_COLOR_PALETTE)]
            if home_points_out is not None and i - 1 < len(home_xy):
                point = _parse_home_xy(home_xy[i - 1])
                if point is not None:
                    home_points_out[veh_id] = point
            result.append((veh_id, edge, dest, depart_time, lane, pos, speed, color))

    return result


def validate_spawn_positions(
    spawns: List[SpawnTuple],
    edge_lengths: Dict[str, float],
) -> List[SpawnTuple]:
    """Clamp spawn positions to stay within edge bounds.

    For any spawn whose ``pos`` exceeds its edge length (minus a safety margin),
    the position is clamped so the vehicle fits on the edge.  Spawns on edges
    not present in ``edge_lengths`` are left unchanged.

    When multiple agents share an edge and would all be clamped to the same
    position, they are redistributed evenly across the available edge length.

    Args:
        spawns: List of spawn tuples (may contain out-of-bounds positions from
            ``expand_spawn_groups``).
        edge_lengths: ``{edge_id: length_m}`` dict built from the SUMO network.

    Returns:
        New list of spawn tuples with valid positions.
    """
    if not edge_lengths:
        return list(spawns)

    # Group spawns by edge to handle redistribution.
    from collections import defaultdict
    edge_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (vid, edge, *_rest) in enumerate(spawns):
        edge_indices[edge].append(idx)

    result = list(spawns)
    for edge, indices in edge_indices.items():
        length = edge_lengths.get(edge)
        if length is None:
            continue

        max_pos = max(length - _DEFAULT_POS_MARGIN_M, 1.0)
        n = len(indices)

        # Check if any position overflows.
        needs_redistribution = False
        for idx in indices:
            try:
                pos_val = float(result[idx][5])
            except (ValueError, IndexError):
                continue
            if pos_val > max_pos:
                needs_redistribution = True
                break

        if needs_redistribution:
            # Redistribute all agents on this edge evenly.
            if n == 1:
                positions = [min(_DEFAULT_POS_START_M, max_pos)]
            else:
                start = min(_DEFAULT_POS_START_M, max_pos * 0.1)
                step = (max_pos - start) / max(n - 1, 1)
                positions = [start + step * j for j in range(n)]

            for j, idx in enumerate(indices):
                old = result[idx]
                result[idx] = (
                    old[0], old[1], old[2], old[3],
                    old[4], str(round(positions[j], 1)), old[6], old[7],
                )

    return result


def load_spawns(
    raw: Any,
    destinations: List[Dict[str, Any]],
    home_points_out: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[SpawnTuple]:
    """Detect spawn format (detailed list or compact groups) and produce tuples.

    Args:
        raw: Parsed JSON from ``spawns.json`` — either a list (detailed) or a
            dict with a ``"groups"`` key (compact).
        destinations: Parsed ``destinations.json`` list (used for default
            ``dest_edge`` in compact mode).
        home_points_out: Optional dict filled with ``{veh_id: (x, y)}`` building centroids
            in SUMO XY.  Stays empty for configs without ``home_xy``, which is how the
            simulation decides between the edge and building-centroid hazard basis.

    Returns:
        List of spawn tuples ready for the simulation loop.
    """
    if isinstance(raw, list):
        return spawns_to_tuples(raw, home_points_out=home_points_out)

    if isinstance(raw, dict) and "groups" in raw:
        default_dest = str(raw.get("default_dest_edge", ""))
        if not default_dest and destinations:
            default_dest = str(destinations[0].get("edge", ""))
        if not default_dest:
            raise ValueError(
                "Compact spawn format requires either 'default_dest_edge' in "
                "spawns.json or at least one entry in destinations.json."
            )
        return expand_spawn_groups(
            raw["groups"], default_dest, home_points_out=home_points_out
        )

    raise ValueError(
        "spawns.json must be either a JSON array (detailed format) or "
        "a JSON object with a 'groups' key (compact format)."
    )
