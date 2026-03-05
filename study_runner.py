"""End-to-end calibration study orchestrator.

``study_runner.py`` is the top-level entry point for running a complete parameter
calibration study.  It chains three stages:

    1. **Experiment sweep** (``experiments.run_parameter_sweep``):
       Spawns one headless ``Traci_GPT2.py`` subprocess per grid cell and collects
       the metrics JSON and replay JSONL for each case.

    2. **Calibration fit** (``calibration.fit_agent_parameters``):
       Scores each run against the reference metrics and ranks candidates by
       goodness of fit.

    3. **Report export**:
       Writes three output artifacts to a timestamped study directory:
           - ``experiments/experiment_results.json``   : Full sweep results.
           - ``experiments/experiment_results.csv``    : Flat CSV summary.
           - ``calibration_report.json``               : Ranked calibration results.
           - ``study_report.json``                     : Top-level study summary.

Each call to ``run_study`` creates a fresh timestamped directory under ``output_dir``
so studies do not overwrite each other.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from calibration import export_calibration_report, fit_agent_parameters, load_reference_scenario
from experiments import build_experiment_grid, export_experiment_results, run_parameter_sweep


def _parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw).split(","):
        item = part.strip()
        if item:
            values.append(float(item))
    return values


def _parse_str_list(raw: str) -> List[str]:
    values: List[str] = []
    for part in str(raw).split(","):
        item = part.strip()
        if item:
            values.append(item)
    return values


def _parse_weights(raw: Optional[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if not raw:
        return weights
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight assignment: {item}")
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value.strip())
    return weights


def _timestamped_study_dir(base_dir: str) -> str:
    base = Path(base_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"study_{ts}"
    idx = 1
    while candidate.exists():
        candidate = base / f"study_{ts}_{idx:02d}"
        idx += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def run_study(
    *,
    reference_path: str,
    script_path: str = "Traci_GPT2.py",
    python_executable: Optional[str] = None,
    output_dir: str = "outputs/studies",
    sumo_binary: str = "sumo",
    run_mode: str = "record",
    timeout_s: Optional[float] = None,
    sigma_values: Optional[List[float]] = None,
    delay_values: Optional[List[float]] = None,
    trust_values: Optional[List[float]] = None,
    scenario_values: Optional[List[str]] = None,
    messaging_enabled: bool = True,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run a complete parameter calibration study: sweep → fit → report.

    Creates a timestamped study directory, runs all grid cases, fits against the
    reference, and writes summary reports.

    Args:
        reference_path: Path to the reference metrics JSON (target behaviour).
        script_path: Path to ``Traci_GPT2.py``.
        python_executable: Python interpreter; defaults to ``sys.executable``.
        output_dir: Base directory for all study outputs.
        sumo_binary: SUMO binary name (``"sumo"`` for headless).
        run_mode: ``"record"`` or ``"replay"``.
        timeout_s: Per-case subprocess timeout in seconds.
        sigma_values: ``INFO_SIGMA`` values to sweep.
        delay_values: ``INFO_DELAY_S`` values to sweep.
        trust_values: ``DEFAULT_THETA_TRUST`` values to sweep.
        scenario_values: Scenario mode strings to sweep.
        messaging_enabled: Whether inter-agent messaging is active for all cases.
        weights: Per-metric calibration weight overrides.
        top_k: Number of top-ranked candidates to include in the calibration report.

    Returns:
        A study summary dict containing timing information, case counts, file paths,
        and the best-fit parameter configuration.
    """
    study_dir = _timestamped_study_dir(output_dir)
    experiments_dir = str(Path(study_dir) / "experiments")
    reference = load_reference_scenario(reference_path)

    grid = build_experiment_grid(
        sigma_values=sigma_values,
        delay_values=delay_values,
        trust_values=trust_values,
        scenario_modes=scenario_values,
        base_overrides={
            "messaging_enabled": bool(messaging_enabled),
        },
    )

    started_at = time.time()
    results = run_parameter_sweep(
        grid,
        script_path=script_path,
        python_executable=python_executable,
        output_dir=experiments_dir,
        sumo_binary=sumo_binary,
        run_mode=run_mode,
        timeout_s=timeout_s,
    )
    finished_experiments_at = time.time()

    exported = export_experiment_results(results, output_dir=experiments_dir)
    fit = fit_agent_parameters(
        {
            "weights": dict(weights or {}),
            "top_k": int(top_k),
        },
        reference=reference,
        experiments_results=results,
    )
    calibration_report = {
        "mode": "batch_fit",
        "reference_path": reference_path,
        "results_json": exported["json"],
        "fit": fit,
    }
    calibration_path = export_calibration_report(
        calibration_report,
        str(Path(study_dir) / "calibration_report.json"),
    )

    failed_cases = sum(1 for row in results if row.get("status") != "ok")
    completed_at = time.time()
    summary = {
        "mode": "experiment_and_calibration",
        "study_dir": study_dir,
        "reference_path": reference_path,
        "started_at_epoch_s": round(started_at, 3),
        "experiments_completed_at_epoch_s": round(finished_experiments_at, 3),
        "completed_at_epoch_s": round(completed_at, 3),
        "durations_s": {
            "experiments": round(finished_experiments_at - started_at, 3),
            "total": round(completed_at - started_at, 3),
        },
        "experiment": {
            "case_count": len(results),
            "failed_cases": failed_cases,
            "results_json": exported["json"],
            "results_csv": exported["csv"],
        },
        "calibration": {
            "report_path": calibration_path,
            "candidate_count": fit.get("candidate_count", 0),
            "best_case": fit.get("best_case"),
        },
    }

    study_report_path = Path(study_dir) / "study_report.json"
    with open(study_report_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    summary["study_report_path"] = str(study_report_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--reference", required=True, help="Reference metrics JSON.")
    parser.add_argument("--script-path", default="Traci_GPT2.py")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-dir", default="outputs/studies")
    parser.add_argument("--sumo-binary", default="sumo", help="Use 'sumo' for headless batch runs.")
    parser.add_argument("--run-mode", choices=["record", "replay"], default="record")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--sigma-values", default="40.0")
    parser.add_argument("--delay-values", default="0.0")
    parser.add_argument("--trust-values", default="0.5")
    parser.add_argument("--scenario-values", default="advice_guided")
    parser.add_argument("--messaging", choices=["on", "off"], default="on")
    parser.add_argument(
        "--weights",
        help="Comma-separated calibration weights, e.g. average_travel_time=2.0,route_choice_entropy=0.5",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_study(
        reference_path=args.reference,
        script_path=args.script_path,
        python_executable=args.python_executable,
        output_dir=args.output_dir,
        sumo_binary=args.sumo_binary,
        run_mode=args.run_mode,
        timeout_s=args.timeout_s,
        sigma_values=_parse_float_list(args.sigma_values),
        delay_values=_parse_float_list(args.delay_values),
        trust_values=_parse_float_list(args.trust_values),
        scenario_values=_parse_str_list(args.scenario_values),
        messaging_enabled=(args.messaging == "on"),
        weights=_parse_weights(args.weights),
        top_k=args.top_k,
    )
    print(f"[STUDY] dir={summary['study_dir']}")
    print(
        f"[STUDY] cases={summary['experiment']['case_count']} "
        f"failed={summary['experiment']['failed_cases']}"
    )
    print(f"[STUDY] experiment_json={summary['experiment']['results_json']}")
    print(f"[STUDY] calibration_report={summary['calibration']['report_path']}")
    best = summary["calibration"].get("best_case")
    if best:
        print(
            f"[STUDY] best_case={best.get('case_id')} "
            f"fit_score={best['score']['fit_score']} "
            f"normalized_loss={best['score']['normalized_loss']}"
        )
    else:
        print("[STUDY] best_case=")
    print(f"[STUDY] report={summary['study_report_path']}")
    return 0 if summary["experiment"]["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
