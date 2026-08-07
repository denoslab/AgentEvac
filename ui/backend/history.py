"""Past runs and replay recordings, read out of the outputs directory.

Everything here is read-only. The Debrief view reads finished runs through these
functions, and the Setup view reads the list of recordings that replay can play
back. No function writes to, moves, or deletes an existing run artifact.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentevac.utils.run_parameters import companion_parameter_path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

#: Runs the console itself launches land here, well away from the campaign folders.
UI_RUNS_DIR = OUTPUTS_DIR / "ui_runs"

_RUN_ID = re.compile(r"(\d{8}_\d{6})")

#: How long a directory listing is reused before the disk is walked again.
_CACHE_TTL_S = 5.0
_cache: Dict[str, Any] = {"runs": (0.0, []), "recordings": (0.0, [])}


def _run_id_of(path: Path) -> Optional[str]:
    match = _RUN_ID.search(path.stem)
    return match.group(1) if match else None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _params_for(path: Path) -> Optional[Dict[str, Any]]:
    try:
        companion = companion_parameter_path(path)
    except Exception:
        return None
    if companion.exists():
        return _load_json(companion)
    run_id = _run_id_of(path)
    if not run_id:
        return None
    for candidate in sorted(path.parent.glob(f"*params*{run_id}*.json")):
        data = _load_json(candidate)
        if data is not None:
            return data
    return None


@dataclass
class RunRecord:
    """One finished run as the history list and Debrief view show it."""

    run_id: str
    metrics_path: str
    modified: float
    campaign: str
    label: str
    package: Optional[str]
    scenario: Optional[str]
    agent_type: Optional[str]
    seed: Optional[int]
    horizon_s: Optional[float]
    alert_offset_s: Optional[float]
    total_agents: Optional[int]
    arrived: Optional[int]
    departed: Optional[int]
    from_console: bool

    def to_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _headline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_agents": metrics.get("total_agents"),
        "arrived": metrics.get("arrived_agents"),
        "departed": metrics.get("departed_agents"),
    }


def _summarise(metrics_path: Path) -> Optional[RunRecord]:
    metrics = _load_json(metrics_path)
    if metrics is None or "total_agents" not in metrics:
        return None
    run_id = _run_id_of(metrics_path) or metrics_path.stem
    params = _params_for(metrics_path) or {}
    alerts = params.get("alerts") or {}
    seeding = params.get("seeding") or {}
    try:
        relative = metrics_path.relative_to(OUTPUTS_DIR)
        campaign = str(relative.parent) if str(relative.parent) != "." else "outputs"
    except ValueError:
        campaign = str(metrics_path.parent)
    console_label = None
    console_note = metrics_path.parent / "console_run.json"
    if console_note.exists():
        console_label = (_load_json(console_note) or {}).get("label")
    headline = _headline(metrics)
    return RunRecord(
        run_id=run_id,
        metrics_path=str(metrics_path.relative_to(REPO_ROOT)),
        modified=metrics_path.stat().st_mtime,
        campaign=campaign,
        label=console_label or f"{campaign} · {run_id}",
        package=params.get("map"),
        scenario=params.get("scenario"),
        agent_type=params.get("agent_type"),
        seed=seeding.get("master_seed") if isinstance(seeding, dict) else None,
        horizon_s=params.get("sim_end_time_s"),
        alert_offset_s=alerts.get("alert_time_offset_s") if isinstance(alerts, dict) else None,
        total_agents=headline["total_agents"],
        arrived=headline["arrived"],
        departed=headline["departed"],
        from_console=console_note.exists(),
    )


def list_runs(limit: int = 200) -> List[Dict[str, Any]]:
    """Every finished run under ``outputs/``, most recent first."""
    cached_at, cached = _cache["runs"]
    if time.monotonic() - cached_at < _CACHE_TTL_S:
        return cached[:limit]
    records: List[RunRecord] = []
    if OUTPUTS_DIR.is_dir():
        for path in OUTPUTS_DIR.rglob("*metrics*.json"):
            if not path.is_file() or "agent_profiles" in path.name:
                continue
            record = _summarise(path)
            if record is not None:
                records.append(record)
    records.sort(key=lambda rec: rec.modified, reverse=True)
    payload = [rec.to_json() for rec in records]
    _cache["runs"] = (time.monotonic(), payload)
    return payload[:limit]


def find_run(run_id: str) -> Optional[Dict[str, Any]]:
    for record in list_runs(limit=10000):
        if record["run_id"] == run_id:
            return record
    return None


def run_artifacts(run_id: str) -> Dict[str, Optional[str]]:
    """Locate the four files a run leaves behind, as repo-relative paths."""
    record = find_run(run_id)
    if record is None:
        return {}
    metrics_path = REPO_ROOT / record["metrics_path"]
    folder = metrics_path.parent
    artifacts: Dict[str, Optional[str]] = {"metrics": record["metrics_path"]}

    def first(pattern: str) -> Optional[str]:
        for candidate in sorted(folder.glob(pattern)):
            if candidate.is_file():
                return str(candidate.relative_to(REPO_ROOT))
        return None

    artifacts["params"] = first(f"*params*{run_id}*.json")
    artifacts["timeline"] = first(f"*timeline*{run_id}*.jsonl")
    artifacts["events"] = first(f"*events*{run_id}*.jsonl")
    artifacts["profiles"] = first(f"*agent_profiles*{run_id}*.json")
    artifacts["decisions"] = first(f"*llm_routes*{run_id}*.jsonl")
    return artifacts


def load_metrics(run_id: str) -> Optional[Dict[str, Any]]:
    record = find_run(run_id)
    if record is None:
        return None
    return _load_json(REPO_ROOT / record["metrics_path"])


def load_timeline(run_id: str, limit: int = 20000) -> List[Dict[str, Any]]:
    """The per-run timeline rows, which carry departures and arrivals over time."""
    artifacts = run_artifacts(run_id)
    path = artifacts.get("timeline")
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(REPO_ROOT / path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
                if len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def list_recordings(limit: int = 100) -> List[Dict[str, Any]]:
    """Language-model runs that replay can play back, most recent first.

    A recording is usable when its decision log carries entries and its parameter
    companion is present, because replay needs the companion to reproduce the
    same package and information regime.
    """
    cached_at, cached = _cache["recordings"]
    if time.monotonic() - cached_at < _CACHE_TTL_S:
        return cached[:limit]
    found: List[Dict[str, Any]] = []
    if OUTPUTS_DIR.is_dir():
        for path in OUTPUTS_DIR.rglob("*llm_routes*.jsonl"):
            if not path.is_file():
                continue
            run_id = _run_id_of(path)
            if not run_id:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size < 512:
                continue
            params = _params_for(path) or {}
            if str(params.get("agent_type", "")) == "rule_based":
                continue
            found.append({
                "run_id": run_id,
                "path": str(path.relative_to(REPO_ROOT)),
                "size_mb": round(size / 1e6, 2),
                "modified": path.stat().st_mtime,
                "package": params.get("map"),
                "scenario": params.get("scenario"),
                "model": params.get("openai_model"),
                "horizon_s": params.get("sim_end_time_s"),
                "params": {
                    "map_name": params.get("map"),
                    "scenario_mode": params.get("scenario"),
                    "sim_end_time_s": params.get("sim_end_time_s"),
                    "agent_type": params.get("agent_type"),
                },
                "has_params": bool(params),
            })
    found.sort(key=lambda rec: rec["modified"], reverse=True)
    _cache["recordings"] = (time.monotonic(), found)
    return found[:limit]


def evacuation_curve(run_id: str) -> Dict[str, List[List[float]]]:
    """Cumulative departures and arrivals over simulation time, from the timeline."""
    departures: List[float] = []
    arrivals: List[float] = []
    for row in load_timeline(run_id):
        layer = str(row.get("layer", ""))
        t_s = row.get("t_s", row.get("sim_t_s"))
        if t_s is None:
            continue
        if layer == "departure":
            departures.append(float(t_s))
        elif layer == "arrival":
            arrivals.append(float(t_s))
    departures.sort()
    arrivals.sort()
    return {
        "departures": [[t, i + 1] for i, t in enumerate(departures)],
        "arrivals": [[t, i + 1] for i, t in enumerate(arrivals)],
    }
