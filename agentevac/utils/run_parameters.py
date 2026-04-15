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


class _CompactLeafEncoder(json.JSONEncoder):
    """JSON encoder that renders dicts of only scalar values on a single line.

    Nested structures are indented normally, but "leaf" dicts (whose values are
    all str, int, float, bool, or None) are kept compact so they can be directly
    copy-pasted as Python dict literals.
    """

    def __init__(self, **kw):
        self._sort_keys = kw.pop("sort_keys", False)
        kw.pop("indent", None)  # we handle indentation ourselves
        super().__init__(**kw)
        self._indent = "  "

    def encode(self, o):
        return self._fmt(o, 0)

    @staticmethod
    def _is_leaf_dict(d):
        return isinstance(d, dict) and all(
            isinstance(v, (str, int, float, bool, type(None))) for v in d.values()
        )

    def _fmt(self, o, level):
        ind = self._indent * level
        ind1 = self._indent * (level + 1)
        if isinstance(o, dict):
            if self._is_leaf_dict(o):
                keys = sorted(o) if self._sort_keys else list(o)
                pairs = ", ".join(
                    f"{json.dumps(k)}: {json.dumps(o[k], ensure_ascii=False)}"
                    for k in keys
                )
                return "{" + pairs + "}"
            keys = sorted(o) if self._sort_keys else list(o)
            items = ",\n".join(
                f"{ind1}{json.dumps(k)}: {self._fmt(o[k], level + 1)}"
                for k in keys
            )
            return "{\n" + items + f"\n{ind}}}"
        if isinstance(o, list):
            if not o:
                return "[]"
            items = ",\n".join(f"{ind1}{self._fmt(item, level + 1)}" for item in o)
            return "[\n" + items + f"\n{ind}]"
        return json.dumps(o, ensure_ascii=False)


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
        fh.write(_CompactLeafEncoder(sort_keys=True, ensure_ascii=False).encode(dict(payload)))
        fh.write("\n")
    return str(target)


def companion_parameter_path(reference_path: str | Path, *, base_name: str = "run_params") -> Path:
    """Derive the expected companion parameter-log path for a run artifact."""
    ref = Path(reference_path)
    suffix = reference_suffix(ref)
    return ref.with_name(f"{base_name}_{suffix}.json")
