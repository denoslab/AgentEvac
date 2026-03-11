"""Helpers for recording and locating per-run parameter snapshots."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

_REFERENCE_PREFIXES = (
    "run_params_",
    "run_metrics_",
    "metrics_",
    "events_",
    "llm_routes_",
    "routes_",
)


def reference_suffix(reference_path: str | Path) -> str:
    """Return the variable suffix portion of a run artifact filename.

    Examples:
        ``run_metrics_20260311_012202.json`` -> ``20260311_012202``
        ``metrics_sigma-40_20260311_012202.json`` -> ``sigma-40_20260311_012202``
    """
    stem = Path(reference_path).stem
    for prefix in _REFERENCE_PREFIXES:
        if stem.startswith(prefix):
            suffix = stem[len(prefix):]
            if suffix:
                return suffix
    return stem


def build_parameter_log_path(base_path: str, *, reference_path: Optional[str | Path] = None) -> str:
    """Build a parameter-log path, preserving a companion artifact suffix when possible."""
    base = Path(base_path)
    ext = base.suffix or ".json"
    stem = base.stem if base.suffix else base.name

    if reference_path:
        suffix = reference_suffix(reference_path)
        candidate = base.with_name(f"{stem}_{suffix}{ext}")
        idx = 1
        while candidate.exists():
            candidate = base.with_name(f"{stem}_{suffix}_{idx:02d}{ext}")
            idx += 1
        return str(candidate)

    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate = base.with_name(f"{stem}_{ts}{ext}")
    idx = 1
    while candidate.exists():
        candidate = base.with_name(f"{stem}_{ts}_{idx:02d}{ext}")
        idx += 1
    return str(candidate)


def write_run_parameter_log(
    base_path: str,
    payload: Mapping[str, Any],
    *,
    reference_path: Optional[str | Path] = None,
) -> str:
    """Write one JSON parameter snapshot to disk and return its path."""
    target = Path(build_parameter_log_path(base_path, reference_path=reference_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return str(target)


def companion_parameter_path(reference_path: str | Path, *, base_name: str = "run_params") -> Path:
    """Derive the expected companion parameter-log path for a run artifact."""
    ref = Path(reference_path)
    suffix = reference_suffix(ref)
    return ref.with_name(f"{base_name}_{suffix}.json")
