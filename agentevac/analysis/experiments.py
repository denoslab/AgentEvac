"""Parameter sweep driver for AgentEvac calibration experiments.

This module builds a Cartesian-product grid of agent parameters and runs one
simulation subprocess per grid cell, collecting the resulting metrics and replay files.

**Parameter axes:**
    - ``info_sigma``    : Gaussian noise standard deviation on margin observations (metres).
    - ``info_delay_s``  : Information delay in seconds (stale observation replay).
    - ``theta_trust``   : Social-signal trust weight ∈ [0, 1].
    - ``scenario``      : Information regime ("no_notice", "alert_guided", "advice_guided").

Each case is run by spawning ``Traci_GPT2.py`` as a subprocess with the appropriate
environment variables set (``INFO_SIGMA``, ``INFO_DELAY_S``, ``DEFAULT_THETA_TRUST``).
The SUMO GUI is suppressed (``--sumo-binary sumo``) for headless batch execution.

**Outputs** (written to ``output_dir``):
    - ``routes_<case_id>.jsonl``    : Recorded LLM route decisions (for replay).
    - ``metrics_<case_id>.json``    : Run-level KPI summary.
    - ``stdout_<case_id>.log``      : Full stdout + stderr of the subprocess.
    - ``experiment_results.json``   : Aggregated list of all case result dicts.
    - ``experiment_results.csv``    : Flat CSV of key result fields.
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_float_list(raw: str) -> List[float]:
    """Parse a comma-separated string of floats into a list.

    Args:
        raw: Comma-separated numeric string (e.g., ``"20,40,60"``).

    Returns:
        List of float values.
    """
    values = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _parse_str_list(raw: str) -> List[str]:
    """Parse a comma-separated string into a list of non-empty strings.

    Args:
        raw: Comma-separated string (e.g., ``"no_notice,advice_guided"``).

    Returns:
        List of stripped non-empty strings.
    """
    values = []
    for part in str(raw).split(","):
        item = part.strip()
        if item:
            values.append(item)
    return values


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-") or "case"


def _case_id(case_cfg: Dict[str, Any], idx: int) -> str:
    return (
        f"{idx:03d}_"
        f"scn-{_slug(case_cfg['scenario'])}_"
        f"sigma-{_slug(case_cfg['info_sigma'])}_"
        f"delay-{_slug(case_cfg['info_delay_s'])}_"
        f"trust-{_slug(case_cfg['theta_trust'])}"
    )


def build_experiment_grid(
    sigma_values: Optional[List[float]] = None,
    delay_values: Optional[List[float]] = None,
    trust_values: Optional[List[float]] = None,
    scenario_modes: Optional[List[str]] = None,
    base_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build a Cartesian-product experiment grid from parameter value lists.

    Defaults apply when a parameter list is not provided:
        - sigma_values    : [40.0]
        - delay_values    : [0.0]
        - trust_values    : [0.5]
        - scenario_modes  : ["advice_guided"]

    Args:
        sigma_values: List of ``INFO_SIGMA`` values to sweep.
        delay_values: List of ``INFO_DELAY_S`` values to sweep.
        trust_values: List of ``DEFAULT_THETA_TRUST`` values to sweep.
        scenario_modes: List of scenario mode strings to sweep.
        base_overrides: Additional key-value pairs merged into every case dict
            (e.g., ``{"messaging_enabled": True}``).

    Returns:
        List of case config dicts, one per grid cell.
    """
    sigma_seq = sigma_values if sigma_values is not None else [40.0]
    delay_seq = delay_values if delay_values is not None else [0.0]
    trust_seq = trust_values if trust_values is not None else [0.5]
    scenario_seq = scenario_modes if scenario_modes is not None else ["advice_guided"]

    grid: List[Dict[str, Any]] = []
    for info_sigma, info_delay_s, theta_trust, scenario in itertools.product(
        sigma_seq,
        delay_seq,
        trust_seq,
        scenario_seq,
    ):
        case = {
            "info_sigma": float(info_sigma),
            "info_delay_s": float(info_delay_s),
            "theta_trust": float(theta_trust),
            "scenario": str(scenario),
        }
        if base_overrides:
            case.update(dict(base_overrides))
        grid.append(case)
    return grid


def _extract_path(stdout: str, prefix: str) -> Optional[str]:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def _extract_events_path(stdout: str) -> Optional[str]:
    pattern = re.compile(r"^\[EVENTS\] enabled=.* path=(.+?) stdout=.*$")
    for line in stdout.splitlines():
        m = pattern.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def run_experiment_case(
    case_cfg: Dict[str, Any],
    *,
    script_path: str = "Traci_GPT2.py",
    python_executable: Optional[str] = None,
    output_dir: str = "outputs/experiments",
    sumo_binary: str = "sumo",
    run_mode: str = "record",
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute one parameter-grid case by spawning a ``Traci_GPT2.py`` subprocess.

    Constructs the CLI command and environment from ``case_cfg``, runs the process,
    captures stdout/stderr, and extracts the metrics and replay file paths from the
    subprocess stdout using fixed prefix patterns.

    Args:
        case_cfg: Case configuration dict (from ``build_experiment_grid``), containing
            at minimum ``info_sigma``, ``info_delay_s``, ``theta_trust``, and
            ``scenario`` keys.
        script_path: Path to the main simulation script.
        python_executable: Python interpreter to use; defaults to ``sys.executable``.
        output_dir: Directory for output files.
        sumo_binary: SUMO binary name; use ``"sumo"`` for headless batch runs.
        run_mode: ``"record"`` or ``"replay"``.
        timeout_s: Optional subprocess timeout in seconds.

    Returns:
        A result dict with fields including ``case_id``, ``status``, ``returncode``,
        ``elapsed_s``, ``replay_path``, ``metrics_path``, ``events_path``,
        ``stdout_log``, and ``stdout_tail``.
    """
    python_bin = python_executable or sys.executable
    script_file = Path(script_path).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_index = int(case_cfg.get("_case_index", 0))
    case_id = str(case_cfg.get("case_id") or _case_id(case_cfg, case_index))
    replay_base = out_dir / f"routes_{case_id}.jsonl"
    metrics_base = out_dir / f"metrics_{case_id}.json"
    stdout_log = out_dir / f"stdout_{case_id}.log"

    cmd = [
        python_bin,
        str(script_file),
        "--run-mode", run_mode,
        "--scenario", str(case_cfg["scenario"]),
        "--sumo-binary", str(sumo_binary),
        "--events", "off",
        "--events-stdout", "off",
        "--web-dashboard", "off",
        "--overlays", "off",
        "--metrics", "on",
        "--replay-log-path", str(replay_base),
        "--metrics-log-path", str(metrics_base),
    ]

    messaging_enabled = bool(case_cfg.get("messaging_enabled", True))
    cmd.extend(["--messaging", "on" if messaging_enabled else "off"])

    env = os.environ.copy()
    env.update({
        "INFO_SIGMA": str(float(case_cfg["info_sigma"])),
        "INFO_DELAY_S": str(float(case_cfg["info_delay_s"])),
        "DEFAULT_THETA_TRUST": str(float(case_cfg["theta_trust"])),
        "SUMO_BINARY": str(sumo_binary),
    })
    if "DEFAULT_LAMBDA_E" in case_cfg:
        env["DEFAULT_LAMBDA_E"] = str(float(case_cfg["DEFAULT_LAMBDA_E"]))
    if "DEFAULT_LAMBDA_T" in case_cfg:
        env["DEFAULT_LAMBDA_T"] = str(float(case_cfg["DEFAULT_LAMBDA_T"]))

    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script_file.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed_s = time.time() - started_at
        stdout_text = proc.stdout or ""
        status = "ok" if proc.returncode == 0 else "failed"
        timeout_hit = False
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.time() - started_at
        stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        status = "timeout"
        timeout_hit = True
        returncode = -1

    stdout_log.write_text(stdout_text, encoding="utf-8")

    result = {
        "case_id": case_id,
        "case": dict(case_cfg),
        "command": cmd,
        "status": status,
        "returncode": returncode,
        "timeout": timeout_hit,
        "elapsed_s": round(elapsed_s, 3),
        "stdout_log": str(stdout_log),
        "replay_path": _extract_path(stdout_text, "[REPLAY] mode=record path="),
        "metrics_path": _extract_path(stdout_text, "[METRICS] summary_path="),
        "events_path": _extract_events_path(stdout_text),
        "stdout_tail": stdout_text.splitlines()[-20:],
    }
    return result


def run_parameter_sweep(
    grid: List[Dict[str, Any]],
    *,
    script_path: str = "Traci_GPT2.py",
    python_executable: Optional[str] = None,
    output_dir: str = "outputs/experiments",
    sumo_binary: str = "sumo",
    run_mode: str = "record",
    timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run all cases in the experiment grid sequentially.

    Cases run one at a time (no parallelism) to avoid SUMO port conflicts and to
    keep resource usage predictable.

    Args:
        grid: List of case config dicts from ``build_experiment_grid``.
        script_path: Path to the main simulation script.
        python_executable: Python interpreter to use.
        output_dir: Directory for all case output files.
        sumo_binary: SUMO binary name.
        run_mode: ``"record"`` or ``"replay"``.
        timeout_s: Per-case subprocess timeout in seconds.

    Returns:
        List of result dicts (one per grid case) from ``run_experiment_case``.
    """
    results: List[Dict[str, Any]] = []
    for idx, raw_case in enumerate(grid, start=1):
        case_cfg = dict(raw_case)
        case_cfg["_case_index"] = idx
        case_cfg["case_id"] = _case_id(case_cfg, idx)
        results.append(
            run_experiment_case(
                case_cfg,
                script_path=script_path,
                python_executable=python_executable,
                output_dir=output_dir,
                sumo_binary=sumo_binary,
                run_mode=run_mode,
                timeout_s=timeout_s,
            )
        )
    return results


def export_experiment_results(
    results: List[Dict[str, Any]],
    *,
    output_dir: str,
    stem: str = "experiment_results",
) -> Dict[str, str]:
    """Write experiment results to JSON (full) and CSV (flat key subset) files.

    The JSON file contains all result fields.  The CSV file contains a flattened
    subset suitable for quick inspection in spreadsheet tools.

    Args:
        results: List of result dicts from ``run_parameter_sweep``.
        output_dir: Directory to write output files.
        stem: Base filename stem (without extension).

    Returns:
        Dict with ``"json"`` and ``"csv"`` keys mapping to the written file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "status",
                "returncode",
                "timeout",
                "elapsed_s",
                "scenario",
                "info_sigma",
                "info_delay_s",
                "theta_trust",
                "replay_path",
                "metrics_path",
                "stdout_log",
            ],
        )
        writer.writeheader()
        for row in results:
            case = row.get("case") or {}
            writer.writerow({
                "case_id": row.get("case_id"),
                "status": row.get("status"),
                "returncode": row.get("returncode"),
                "timeout": row.get("timeout"),
                "elapsed_s": row.get("elapsed_s"),
                "scenario": case.get("scenario"),
                "info_sigma": case.get("info_sigma"),
                "info_delay_s": case.get("info_delay_s"),
                "theta_trust": case.get("theta_trust"),
                "replay_path": row.get("replay_path"),
                "metrics_path": row.get("metrics_path"),
                "stdout_log": row.get("stdout_log"),
            })

    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--script-path", default="Traci_GPT2.py")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--sumo-binary", default="sumo", help="Use 'sumo' for headless batch runs.")
    parser.add_argument("--run-mode", choices=["record", "replay"], default="record")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--sigma-values", default="40.0")
    parser.add_argument("--delay-values", default="0.0")
    parser.add_argument("--trust-values", default="0.5")
    parser.add_argument("--scenario-values", default="advice_guided")
    parser.add_argument("--messaging", choices=["on", "off"], default="on")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    grid = build_experiment_grid(
        sigma_values=_parse_float_list(args.sigma_values),
        delay_values=_parse_float_list(args.delay_values),
        trust_values=_parse_float_list(args.trust_values),
        scenario_modes=_parse_str_list(args.scenario_values),
        base_overrides={
            "messaging_enabled": (args.messaging == "on"),
        },
    )
    results = run_parameter_sweep(
        grid,
        script_path=args.script_path,
        python_executable=args.python_executable,
        output_dir=args.output_dir,
        sumo_binary=args.sumo_binary,
        run_mode=args.run_mode,
        timeout_s=args.timeout_s,
    )
    exported = export_experiment_results(results, output_dir=args.output_dir)
    print(f"[EXPERIMENTS] cases={len(results)}")
    print(f"[EXPERIMENTS] json={exported['json']}")
    print(f"[EXPERIMENTS] csv={exported['csv']}")
    failed = sum(1 for row in results if row.get("status") != "ok")
    print(f"[EXPERIMENTS] failed_cases={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
