"""Composing a new scenario package from a map selection.

An operator draws a box on the map, the buildings inside become households, a second
selection becomes the area an order covers, and fire origins are placed by click. This
module turns that draft into the JSON files the simulator reads.

Two rules govern every path here.

A package directory is created or nothing happens. Writing into a directory that already
exists is refused, so a package a campaign has run against can never be altered by the
console. That is the only write path the console has into ``configs/``.

Buildings are the unit of selection and edges are the unit the simulator inserts on, so
the translation happens once, here, using the ``edge`` each building already carries in
its preview bundle. Nothing downstream has to know a building existed.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
PREVIEWS_DIR = REPO_ROOT / "ui" / "assets" / "previews"

#: Files a draft inherits from its source package unchanged. The network, the shelter
#: menu, and the route menu are properties of the map, not of the selection.
INHERITED_FILES = ("map.json", "destinations.json", "routes.json")

#: A package id becomes a directory name and appears in run paths, so it is kept to
#: characters that need no escaping anywhere.
PACKAGE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{2,63}$")

#: The alert instruction vocabulary the schedule resolver accepts.
INSTRUCTION_CHOICES = ("evacuate_now", "prepare", "shelter", "none")

#: Channels an order can arrive on, matching the M1 alert engine.
CHANNEL_CHOICES = ("broadcast", "door", "wireless_emergency_alert")

MAX_AGENTS_PER_BUILDING = 50


@dataclass
class DraftValidation:
    """Outcome of checking one proposed package, shaped like the run validator's."""

    ok: bool
    problems: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": self.problems,
            "warnings": self.warnings,
            "summary": self.summary,
        }


def _problem(field_name: str, message: str, hint: str) -> Dict[str, str]:
    return {"field": field_name, "message": message, "hint": hint}


def load_buildings_index(package_id: str) -> Dict[str, Dict[str, Any]]:
    """Return ``{building_id: record}`` from a package's preview bundle.

    The bundle is what the browser drew the selection on, so resolving against it means
    the package records exactly the buildings the operator saw.
    """
    path = PREVIEWS_DIR / f"{package_id}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        bundle = json.load(handle)
    return {str(b["id"]): b for b in bundle.get("buildings", []) if b.get("id")}


def _resolve_households(
    rows: List[Dict[str, Any]],
    index: Dict[str, Dict[str, Any]],
    problems: List[Dict[str, str]],
    warnings: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Attach geometry to each selected building and drop the ones that cannot spawn."""
    resolved: List[Dict[str, Any]] = []
    unsnapped: List[str] = []
    unknown: List[str] = []
    seen: set = set()

    for row in rows:
        building_id = str(row.get("building_id", "") or "")
        if not building_id:
            problems.append(_problem(
                "households", "a selected household carries no building id",
                "Reselect the area so every household names its building.",
            ))
            continue
        if building_id in seen:
            warnings.append(_problem(
                "households", f"building {building_id} was selected more than once",
                "The duplicate is ignored and the first selection stands.",
            ))
            continue
        seen.add(building_id)

        record = index.get(building_id) if index else None
        # A caller may supply geometry directly, which is what tests and any client
        # holding the bundle already do.
        edge = row.get("edge") or (record or {}).get("edge")
        x = row.get("x", (record or {}).get("x"))
        y = row.get("y", (record or {}).get("y"))

        if index and record is None:
            unknown.append(building_id)
            continue
        if not edge:
            unsnapped.append(building_id)
            continue
        if x is None or y is None:
            unsnapped.append(building_id)
            continue

        try:
            count = int(row.get("count", 1))
        except (TypeError, ValueError):
            count = 0
        if count < 1:
            warnings.append(_problem(
                "households", f"building {building_id} was given {row.get('count')!r} agents",
                "A household needs at least one agent, so this building is skipped.",
            ))
            continue
        if count > MAX_AGENTS_PER_BUILDING:
            problems.append(_problem(
                "households",
                f"building {building_id} was given {count} agents",
                f"Keep it to {MAX_AGENTS_PER_BUILDING} or fewer per building.",
            ))
            continue

        resolved.append({
            "building_id": building_id,
            "edge": str(edge),
            "x": float(x),
            "y": float(y),
            "count": count,
        })

    if unknown:
        problems.append(_problem(
            "households",
            f"{len(unknown)} selected buildings are not in the source package's map bundle",
            "Rebuild the bundle with ui.tools.build_map_assets, or reselect on the current map.",
        ))
    if unsnapped:
        warnings.append(_problem(
            "households",
            f"{len(unsnapped)} selected buildings sit too far from any drivable road",
            "They are left out, because a vehicle cannot be inserted for them.",
        ))
    return resolved


def compose_spawns(households: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build ``spawns.json`` from resolved households.

    Buildings that share an edge merge into one group, and their centroids and ids are
    listed in the same order the loader generates agent ids, so a household keeps its
    own home point.
    """
    order: List[str] = []
    by_edge: Dict[str, List[Dict[str, Any]]] = {}
    for row in sorted(households, key=lambda r: (r["edge"], r["building_id"])):
        if row["edge"] not in by_edge:
            by_edge[row["edge"]] = []
            order.append(row["edge"])
        by_edge[row["edge"]].append(row)

    groups = []
    for edge in order:
        rows = by_edge[edge]
        home_xy: List[Any] = []
        building_id: List[Any] = []
        for row in rows:
            for _ in range(row["count"]):
                home_xy.append([round(row["x"], 2), round(row["y"], 2)])
                building_id.append(row["building_id"])
        groups.append({
            "edge": edge,
            "count": sum(r["count"] for r in rows),
            "home_xy": home_xy,
            "building_id": building_id,
        })
    return {"groups": groups}


def compose_alerts(
    areas: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    index: Dict[str, Dict[str, Any]],
    households: List[Dict[str, Any]],
    problems: List[Dict[str, str]],
    warnings: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Build ``alerts.json``, translating each area's households into its edges.

    An area covers buildings because that is what an operator selects and what the
    record describes. The simulator scopes an order by edge, so the edge set is the
    distinct edges of the area's buildings, derived here and never at run time.

    An area only covers households. A building holding no agents cannot be evacuated, so
    ordering it changes nothing, and including it would still pull its road into the
    ordered area and over-scope the order onto households that were never chosen.
    """
    if not areas and not events:
        return None

    household_edge = {row["building_id"]: row["edge"] for row in households}
    composed_areas: Dict[str, Any] = {}
    for area in areas:
        name = str(area.get("name", "") or "").strip()
        if not name:
            problems.append(_problem(
                "alert_areas", "an alert area has no name",
                "Give every area a short name, which the schedule refers to.",
            ))
            continue
        if name in composed_areas:
            problems.append(_problem(
                "alert_areas", f"two alert areas are both named {name!r}",
                "Area names must be unique.",
            ))
            continue

        edges: List[str] = []
        members: List[str] = []
        not_household = 0
        for building_id in area.get("building_ids", []) or []:
            building_id = str(building_id)
            edge = household_edge.get(building_id)
            if not edge:
                not_household += 1
                continue
            members.append(building_id)
            if edge not in edges:
                edges.append(edge)
        if not_household:
            warnings.append(_problem(
                "alert_areas",
                f"{not_household} buildings in area {name!r} are not households",
                "They hold no agents, so the order skips them and they are left out of the area.",
            ))
        if not edges:
            problems.append(_problem(
                "alert_areas", f"area {name!r} covers no household",
                "Select buildings that are already households under the spawn selection.",
            ))
            continue

        composed_areas[name] = {
            "label": str(area.get("label", name)),
            "agents": sum(
                row["count"] for row in households if household_edge.get(row["building_id"]) in edges
            ),
            "building_ids": members,
            "edges": edges,
        }

    schedule = []
    for event in events:
        event_areas = [str(a) for a in (event.get("areas") or [])]
        unknown = [a for a in event_areas if a not in composed_areas]
        if unknown:
            problems.append(_problem(
                "alert_events",
                f"alert {event.get('id', '?')} names areas that do not exist: {', '.join(unknown)}",
                "Every alert must refer to an area defined in the selection.",
            ))
            continue
        instruction = str(event.get("instruction", "evacuate_now"))
        if instruction not in INSTRUCTION_CHOICES:
            problems.append(_problem(
                "alert_events", f"unknown instruction {instruction!r}",
                f"Choose one of {', '.join(INSTRUCTION_CHOICES)}.",
            ))
            continue
        try:
            issue_time = float(event.get("issue_time_s"))
        except (TypeError, ValueError):
            problems.append(_problem(
                "alert_events", f"alert {event.get('id', '?')} has no issue time",
                "Give a time in seconds after ignition.",
            ))
            continue
        channel = str(event.get("channel", "broadcast"))
        if channel not in CHANNEL_CHOICES:
            problems.append(_problem(
                "alert_events", f"unknown channel {channel!r}",
                f"Choose one of {', '.join(CHANNEL_CHOICES)}.",
            ))
            continue

        schedule.append({
            "id": str(event.get("id") or f"EA-{len(schedule) + 1}"),
            "issue_time_s": issue_time,
            "areas": event_areas,
            "instruction": instruction,
            "channel": channel,
            "hazard_text": str(event.get("hazard_text", "") or ""),
            "routing_text": event.get("routing_text") or None,
        })
    schedule.sort(key=lambda row: row["issue_time_s"])

    return {
        "_meta": {"authored_in": "operator console", "areas_are_cumulative": True},
        "areas": composed_areas,
        "schedule": schedule,
    }


def compose_fires(
    fires: List[Dict[str, Any]],
    problems: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Build ``fires.json`` from placed sources."""
    sources = []
    for i, fire in enumerate(fires or []):
        try:
            source = {
                "id": str(fire.get("id") or f"source_{i + 1}"),
                "x": float(fire["x"]),
                "y": float(fire["y"]),
                "t0": float(fire.get("t0", 0.0)),
                "r0": float(fire.get("r0", 30.0)),
                "growth_m_per_s": float(fire.get("growth_m_per_s", 0.3)),
            }
        except (KeyError, TypeError, ValueError):
            problems.append(_problem(
                "fires", f"fire source {fire.get('id', i + 1)!r} is missing a position or a growth rate",
                "Place the source on the map and give it a growth rate in metres per second.",
            ))
            continue
        if fire.get("max_r_m") is not None:
            try:
                source["max_r_m"] = float(fire["max_r_m"])
            except (TypeError, ValueError):
                problems.append(_problem(
                    "fires", f"fire source {source['id']!r} has an unreadable radius cap",
                    "Leave the cap empty for an uncapped front, or give it in metres.",
                ))
                continue
        if source["growth_m_per_s"] < 0:
            problems.append(_problem(
                "fires", f"fire source {source['id']!r} grows at a negative rate",
                "A front spreads outward, so the rate cannot be below zero.",
            ))
            continue
        sources.append(source)
    sources.sort(key=lambda row: row["t0"])
    return {"sources": sources}


def validate_draft(
    draft: Dict[str, Any],
    index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[DraftValidation, Optional[Dict[str, Any]]]:
    """Check a proposed package and, when it holds, compose its files.

    Returns ``(validation, files)`` where ``files`` maps file name to parsed content and
    is ``None`` when the draft cannot be written.
    """
    problems: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    draft = draft or {}

    package_id = str(draft.get("id", "") or "").strip()
    if not PACKAGE_ID_PATTERN.match(package_id):
        problems.append(_problem(
            "id", f"{package_id!r} is not a usable package name",
            "Use 3 to 64 characters, lowercase letters, digits, and underscores.",
        ))
    elif (CONFIGS_DIR / package_id).exists():
        problems.append(_problem(
            "id", f"a package named {package_id!r} already exists",
            "Choose a different name. The console never writes over an existing package.",
        ))

    source_id = str(draft.get("source_package", "") or "").strip()
    source_dir = CONFIGS_DIR / source_id if source_id else None
    if not source_id:
        problems.append(_problem(
            "source_package", "no source package was named",
            "Pick the package whose network and shelters this one inherits.",
        ))
    elif not source_dir.is_dir():
        problems.append(_problem(
            "source_package", f"no package named {source_id!r} to inherit from",
            "Pick a package that exists.",
        ))
    else:
        for name in INHERITED_FILES:
            if name == "routes.json":
                continue
            if not (source_dir / name).exists():
                problems.append(_problem(
                    "source_package", f"{source_id} has no {name} to inherit",
                    "Pick a package with a complete map definition.",
                ))

    if index is None and source_id:
        index = load_buildings_index(source_id)
    index = index or {}

    households = _resolve_households(list(draft.get("households") or []), index, problems, warnings)
    if not households:
        problems.append(_problem(
            "households", "the selection contains no household that can spawn",
            "Draw a box over buildings that sit beside a road.",
        ))

    alerts = compose_alerts(
        list(draft.get("alert_areas") or []),
        list(draft.get("alert_events") or []),
        index,
        households,
        problems,
        warnings,
    )
    fires = compose_fires(list(draft.get("fires") or []), problems)
    if not fires["sources"]:
        problems.append(_problem(
            "fires", "the package has no fire source",
            "Place at least one origin, since without a fire nothing drives the evacuation.",
        ))

    total_agents = sum(row["count"] for row in households)
    summary = {
        "package": package_id,
        "source_package": source_id,
        "households": len(households),
        "agents": total_agents,
        "edges": len({row["edge"] for row in households}),
        "fire_sources": len(fires["sources"]),
        "alert_areas": len((alerts or {}).get("areas", {})),
        "alert_events": len((alerts or {}).get("schedule", [])),
    }

    if alerts is None:
        warnings.append(_problem(
            "alert_areas", "the package carries no alert schedule",
            "Households will run with no official order, which is the no-notice baseline.",
        ))
    if total_agents > 2000:
        warnings.append(_problem(
            "households", f"{total_agents} agents is a large population",
            "Runtime grows with the population, so try a smaller selection first.",
        ))

    if problems:
        return DraftValidation(ok=False, problems=problems, warnings=warnings, summary=summary), None

    files: Dict[str, Any] = {
        "spawns.json": compose_spawns(households),
        "fires.json": fires,
    }
    if alerts is not None:
        files["alerts.json"] = alerts
    return DraftValidation(ok=True, problems=[], warnings=warnings, summary=summary), files


def write_package(
    draft: Dict[str, Any],
    index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[DraftValidation, Optional[Path]]:
    """Validate a draft and create its package directory.

    The directory is created only when it does not already exist, and a failure part way
    through removes what was written, so a half-built package never appears in the list.
    """
    validation, files = validate_draft(draft, index)
    if not validation.ok or files is None:
        return validation, None

    package_id = str(draft["id"]).strip()
    source_dir = CONFIGS_DIR / str(draft["source_package"]).strip()
    target = CONFIGS_DIR / package_id

    try:
        # Fails if the directory exists, which is the guarantee this module rests on.
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        validation.ok = False
        validation.problems.append(_problem(
            "id", f"a package named {package_id!r} already exists",
            "Choose a different name.",
        ))
        return validation, None

    try:
        for name in INHERITED_FILES:
            source_file = source_dir / name
            if source_file.exists():
                shutil.copyfile(source_file, target / name)
        for name, content in files.items():
            with open(target / name, "w", encoding="utf-8") as handle:
                json.dump(content, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        _write_readme(target, draft, validation.summary)
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)
        validation.ok = False
        validation.problems.append(_problem(
            "id", f"the package could not be written, {exc}",
            "Check that configs/ is writable and try again.",
        ))
        return validation, None

    return validation, target


def _write_readme(target: Path, draft: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """Record where the package came from, so it is never mistaken for a curated one."""
    label = str(draft.get("label") or target.name)
    note = str(draft.get("description") or "").strip()
    lines = [
        f"# {label}",
        "",
        "Authored in the operator console from a map selection. Households are the "
        "buildings inside the drawn area, one agent each unless the count was raised, "
        f"and the network and shelters are inherited from `{summary.get('source_package')}`.",
        "",
        f"- Households {summary.get('households')} across {summary.get('edges')} roads, "
        f"{summary.get('agents')} agents",
        f"- Fire sources {summary.get('fire_sources')}",
        f"- Alert areas {summary.get('alert_areas')}, orders {summary.get('alert_events')}",
    ]
    if note:
        lines += ["", note]
    lines.append("")
    with open(target / "README.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
