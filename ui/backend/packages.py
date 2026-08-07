"""Scenario package discovery and run-configuration validation.

A scenario package is a directory under ``configs/`` holding the JSON files the
simulator reads for one incident. This module summarises each package for the
Setup view and checks a proposed run before a process is spawned, so an operator
sees a named problem instead of a simulation that dies at startup.

Package files are read, never written. The console has no path that edits them.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentevac.config_loader import load_map_config, load_spawns

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

SCENARIO_CHOICES = ("no_notice", "alert_guided", "advice_guided")
ENGINE_CHOICES = ("llm", "rule_based", "replay")

#: Plain-language description of each information regime, for the Setup cards.
SCENARIO_DESCRIPTIONS = {
    "no_notice": "Households see only what they can observe themselves and what neighbours tell them.",
    "alert_guided": "Households also receive the official alert and the fire forecast.",
    "advice_guided": "Households also receive route guidance and the expected cost of each option.",
}

ENGINE_DESCRIPTIONS = {
    "llm": "Live language-model decisions. Needs an API key and adds latency at every decision round.",
    "rule_based": "Deterministic policy over the same beliefs and utilities. No API calls, responds instantly.",
    "replay": "Replays the decisions recorded in an earlier language-model run. Authentic wording, no API calls.",
}


@dataclass
class ScenarioPackage:
    """One incident package as the Setup view presents it."""

    id: str
    label: str
    households: int
    fire_sources: int
    destinations: List[str]
    alert_waves: List[Dict[str, Any]]
    has_alert_schedule: bool
    net_file: str
    sumo_cfg: str
    net_exists: bool
    net_size_mb: float
    anchor_clock: Optional[str]
    recommended_horizon_s: Optional[float]
    required_horizon_s: Optional[float]
    first_ignition_s: float
    last_ignition_s: float
    description: str
    problems: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["usable"] = not self.problems
        return payload


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _package_description(map_dir: Path) -> str:
    """First prose paragraph of the package README, or an empty string."""
    readme = map_dir / "README.md"
    if not readme.exists():
        return ""
    lines: List[str] = []
    for raw in readme.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            if lines:
                break
            continue
        lines.append(line)
        if len(" ".join(lines)) > 320:
            break
    return " ".join(lines)[:400]


def _anchor_clock(alerts_cfg: Any) -> Optional[str]:
    """Local wall-clock time that simulation second zero represents."""
    if not isinstance(alerts_cfg, dict):
        return None
    clock = ((alerts_cfg.get("_meta") or {}).get("clock") or {})
    raw = str(clock.get("sim_t0_equals", ""))
    match = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", raw)
    if not match:
        return None
    hour, minute, second = match.group(1), match.group(2), match.group(3) or "00"
    return f"{int(hour):02d}:{minute}:{second}"


def load_package(package_id: str) -> ScenarioPackage:
    """Summarise one scenario package, recording rather than raising on problems."""
    map_dir = CONFIGS_DIR / package_id
    problems: List[str] = []
    try:
        cfg = load_map_config(package_id)
    except Exception as exc:
        return ScenarioPackage(
            id=package_id, label=package_id, households=0, fire_sources=0, destinations=[],
            alert_waves=[], has_alert_schedule=False, net_file="", sumo_cfg="", net_exists=False,
            net_size_mb=0.0, anchor_clock=None, recommended_horizon_s=None, required_horizon_s=None,
            first_ignition_s=0.0, last_ignition_s=0.0, description="",
            problems=[f"configuration could not be read: {exc}"],
        )

    try:
        spawns = load_spawns(cfg["spawns"], cfg["destinations"])
    except Exception as exc:
        spawns = []
        problems.append(f"spawns.json is not usable: {exc}")

    fires = list((cfg.get("fires") or {}).get("sources") or [])
    fires += list((cfg.get("fires") or {}).get("events") or [])
    ignition_times = sorted(float(f.get("t0", 0.0)) for f in fires) or [0.0]

    alerts_cfg = cfg.get("alerts") or None
    waves: List[Dict[str, Any]] = []
    if isinstance(alerts_cfg, dict):
        for event in alerts_cfg.get("schedule") or []:
            waves.append({
                "id": str(event.get("id", "")),
                "issue_time_s": float(event.get("issue_time_s", 0.0)),
                "instruction": str(event.get("instruction", "")),
                "areas": [str(a) for a in event.get("areas") or []],
            })
        waves.sort(key=lambda w: w["issue_time_s"])

    clock = ((alerts_cfg or {}).get("_meta") or {}).get("clock") or {}
    net_file = str((cfg.get("map") or {}).get("net_file", ""))
    sumo_cfg = str((cfg.get("map") or {}).get("sumo_cfg", ""))
    net_path = REPO_ROOT / net_file if net_file else None
    net_exists = bool(net_path and net_path.exists())
    if net_file and not net_exists:
        problems.append(f"network file is missing: {net_file}")
    if not spawns:
        problems.append("the package defines no households")

    return ScenarioPackage(
        id=package_id,
        label=package_id.replace("_", " "),
        households=len(spawns),
        fire_sources=len(fires),
        destinations=[str(d.get("name", d.get("edge", ""))) for d in cfg.get("destinations") or []],
        alert_waves=waves,
        has_alert_schedule=bool(waves),
        net_file=net_file,
        sumo_cfg=sumo_cfg,
        net_exists=net_exists,
        net_size_mb=round(net_path.stat().st_size / 1e6, 1) if net_exists else 0.0,
        anchor_clock=_anchor_clock(alerts_cfg),
        recommended_horizon_s=_opt_float(clock.get("recommended_sim_end_time_s")),
        required_horizon_s=_opt_float(clock.get("required_sim_end_time_s")),
        first_ignition_s=ignition_times[0],
        last_ignition_s=ignition_times[-1],
        description=_package_description(map_dir),
        problems=problems,
    )


def _opt_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_packages() -> List[ScenarioPackage]:
    """Every readable scenario package, flagship packages first."""
    if not CONFIGS_DIR.is_dir():
        return []
    packages = []
    for entry in sorted(CONFIGS_DIR.iterdir()):
        if not entry.is_dir() or not (entry / "map.json").exists():
            continue
        packages.append(load_package(entry.name))

    def sort_key(pkg: ScenarioPackage):
        return (bool(pkg.problems), not pkg.has_alert_schedule, -pkg.households, pkg.id)

    packages.sort(key=sort_key)
    return packages


def incident_schedule(package: ScenarioPackage) -> List[Dict[str, Any]]:
    """Ignitions and alert waves on one timeline, for the Setup view."""
    rows: List[Dict[str, Any]] = []
    try:
        cfg = load_map_config(package.id)
    except Exception:
        return rows
    fires = list((cfg.get("fires") or {}).get("sources") or [])
    fires += list((cfg.get("fires") or {}).get("events") or [])
    for fire in fires:
        rows.append({
            "kind": "ignition",
            "t_s": float(fire.get("t0", 0.0)),
            "label": str(fire.get("id", "")),
            "detail": f"{float(fire.get('r0', 0.0)):.0f} m initial radius",
        })
    for wave in package.alert_waves:
        rows.append({
            "kind": "alert",
            "t_s": wave["issue_time_s"],
            "label": wave["id"] or wave["instruction"],
            "detail": ", ".join(wave["areas"]),
        })
    rows.sort(key=lambda row: row["t_s"])
    return rows


# ----------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "package": "halifax_3town_e0",
    "scenario": "advice_guided",
    "engine": "rule_based",
    "replay_run_id": None,
    "seed": 1024,
    "sim_end_time_s": 28800,
    "alert_minutes_earlier": 0,
    "messaging": True,
    "decision_period_s": 240.0,
    "initial_speed": 16,
    "label": "",
}


@dataclass
class Validation:
    """Outcome of checking one proposed run."""

    ok: bool
    problems: List[Dict[str, str]]
    warnings: List[Dict[str, str]]
    normalized: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": self.problems,
            "warnings": self.warnings,
            "config": self.normalized,
        }


def validate_config(raw: Dict[str, Any], recordings: Optional[List[Dict[str, Any]]] = None) -> Validation:
    """Check a proposed run and normalise it into the form the launcher takes."""
    problems: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    config = dict(DEFAULT_CONFIG)
    config.update({k: v for k, v in (raw or {}).items() if v is not None})

    package_id = str(config.get("package", ""))
    package = load_package(package_id) if package_id else None
    if package is None or package.problems:
        detail = package.problems[0] if package and package.problems else "no scenario package selected"
        problems.append({"field": "package", "message": detail,
                         "hint": "Choose a package whose files and network are present."})

    scenario = str(config.get("scenario", ""))
    if scenario not in SCENARIO_CHOICES:
        problems.append({"field": "scenario", "message": f"unknown information regime {scenario!r}",
                         "hint": f"Choose one of {', '.join(SCENARIO_CHOICES)}."})

    engine = str(config.get("engine", ""))
    if engine not in ENGINE_CHOICES:
        problems.append({"field": "engine", "message": f"unknown decision engine {engine!r}",
                         "hint": f"Choose one of {', '.join(ENGINE_CHOICES)}."})
    if engine == "llm" and not os.getenv("OPENAI_API_KEY"):
        problems.append({"field": "engine", "message": "live language-model decisions need OPENAI_API_KEY",
                         "hint": "Set the key in the environment, or run with rule_based or replay."})
    if engine == "replay":
        run_id = str(config.get("replay_run_id") or "")
        known = {rec["run_id"]: rec for rec in (recordings or [])}
        if not run_id:
            problems.append({"field": "replay_run_id", "message": "replay needs a recorded run to play back",
                             "hint": "Pick a recording, or switch the engine to rule_based."})
        elif run_id not in known:
            problems.append({"field": "replay_run_id", "message": f"no recording found for run {run_id}",
                             "hint": "Pick a recording from the list."})
        else:
            recorded = known[run_id].get("params") or {}
            for field_name, config_key in (("map_name", "package"), ("scenario_mode", "scenario")):
                recorded_value = recorded.get(field_name)
                if recorded_value and str(recorded_value) != str(config.get(config_key)):
                    warnings.append({
                        "field": config_key,
                        "message": f"the recording was made with {config_key}={recorded_value}",
                        "hint": "Replay follows the recorded decisions, so the run is set to match.",
                    })
                    config[config_key] = recorded_value

    try:
        config["seed"] = int(config["seed"])
    except (TypeError, ValueError):
        problems.append({"field": "seed", "message": "the seed must be a whole number",
                         "hint": "Enter a number, or press randomize."})

    try:
        horizon = float(config["sim_end_time_s"])
        if horizon <= 0:
            raise ValueError
        config["sim_end_time_s"] = horizon
    except (TypeError, ValueError):
        problems.append({"field": "sim_end_time_s", "message": "the horizon must be a positive number of seconds",
                         "hint": "The Halifax reconstruction is designed for 28800 seconds."})
        horizon = 0.0

    if package and package.required_horizon_s and horizon and horizon < package.required_horizon_s:
        warnings.append({
            "field": "sim_end_time_s",
            "message": f"the last alert wave is issued at {package.required_horizon_s:.0f} s",
            "hint": "A shorter horizon ends the run before that order reaches anyone.",
        })

    try:
        minutes = float(config["alert_minutes_earlier"])
        config["alert_minutes_earlier"] = minutes
    except (TypeError, ValueError):
        problems.append({"field": "alert_minutes_earlier", "message": "alert timing must be a number of minutes",
                         "hint": "Use 0 for the historical timing."})
        minutes = 0.0

    if minutes and package and not package.has_alert_schedule:
        warnings.append({
            "field": "alert_minutes_earlier",
            "message": "this package ships no alert schedule, so the timing shift changes nothing",
            "hint": "Choose a package with alert waves to run the timing counterfactual.",
        })
    if minutes and scenario == "no_notice":
        warnings.append({
            "field": "alert_minutes_earlier",
            "message": "under no notice, households never receive the official orders",
            "hint": "The timing shift only shows up under alert_guided or advice_guided.",
        })

    config["messaging"] = bool(config.get("messaging"))
    config["label"] = str(config.get("label") or "").strip()[:120]

    return Validation(ok=not problems, problems=problems, warnings=warnings, normalized=config)
