"""Shared helpers for plotting completed simulation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List

from agentevac.utils.run_parameters import companion_parameter_path


def newest_file(pattern: str) -> Path:
    """Return the newest file matching ``pattern``.

    Raises:
        FileNotFoundError: If no matching files exist.
    """
    matches = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[0]


def resolve_input(path_arg: str | None, pattern: str) -> Path:
    """Resolve an explicit input path or fall back to the newest matching file."""
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path
    return newest_file(pattern)


def resolve_optional_run_params(path_arg: str | None, reference_path: Path | None) -> Path | None:
    """Resolve an explicit or companion run-parameter log path if available."""
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path
    if reference_path is None:
        return None
    candidate = companion_parameter_path(reference_path)
    return candidate if candidate.exists() else None


def load_json(path: Path) -> Any:
    """Load a JSON document from ``path``."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> List[dict[str, Any]]:
    """Load JSON Lines from ``path`` into a list of dicts."""
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def ensure_output_path(
    input_path: Path,
    output_arg: str | None,
    *,
    suffix: str,
) -> Path:
    """Resolve output path and ensure its parent directory exists."""
    if output_arg:
        out = Path(output_arg)
    else:
        out = input_path.with_suffix("")
        out = out.with_name(f"{out.name}.{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def top_items(mapping: dict[str, float], limit: int) -> list[tuple[str, float]]:
    """Return up to ``limit`` items sorted by descending value then key."""
    items = sorted(mapping.items(), key=lambda item: (-item[1], item[0]))
    return items[: max(1, int(limit))]


def bin_counts(
    times_s: Iterable[float],
    *,
    bin_s: float,
) -> list[tuple[float, int]]:
    """Bin event times into fixed-width buckets.

    Returns:
        List of ``(bin_start_s, count)`` tuples in ascending order.
    """
    counts: dict[float, int] = {}
    width = max(float(bin_s), 1e-9)
    for t in times_s:
        bucket = width * int(float(t) // width)
        counts[bucket] = counts.get(bucket, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0])


def require_matplotlib():
    """Import matplotlib lazily with a useful error message."""
    # Constrain thread-hungry numeric backends before importing matplotlib/numpy.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "`pip install -e .[plot]` or `pip install matplotlib`."
        ) from exc
    return plt
