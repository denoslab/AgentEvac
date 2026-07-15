"""Parameter sweep driver for AgentEvac calibration experiments.

This module builds a Cartesian-product grid of agent parameters and runs one
simulation subprocess per grid cell, collecting the resulting metrics and replay files.

**Parameter axes:**
    - ``info_sigma``    : Gaussian noise standard deviation on margin observations (metres).
    - ``info_delay_s``  : Information delay in seconds (stale observation replay).
    - ``theta_trust``   : Social-signal trust weight ∈ [0, 1].
    - ``scenario``      : Information regime ("no_notice", "alert_guided", "advice_guided").

Each case is run by spawning ``agentevac.simulation.main`` as a subprocess with the appropriate
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
import shlex
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


def _parse_int_list(raw: str) -> List[int]:
    """Parse a comma-separated string of integers into a list."""
    values = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
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


SPREAD_PRESETS: Dict[str, Dict[str, float]] = {
    "none": {
        "theta_trust": 0.0, "theta_r": 0.0, "theta_u": 0.0,
        "gamma": 0.0, "lambda_e": 0.0, "lambda_t": 0.0,
    },
    "low": {
        "theta_trust": 0.05, "theta_r": 0.03, "theta_u": 0.03,
        "gamma": 0.001, "lambda_e": 0.15, "lambda_t": 0.03,
    },
    "moderate": {
        "theta_trust": 0.12, "theta_r": 0.08, "theta_u": 0.08,
        "gamma": 0.003, "lambda_e": 0.4, "lambda_t": 0.08,
    },
    "high": {
        "theta_trust": 0.20, "theta_r": 0.15, "theta_u": 0.15,
        "gamma": 0.005, "lambda_e": 0.8, "lambda_t": 0.15,
    },
}


def _case_id(case_cfg: Dict[str, Any], idx: int) -> str:
    parts = (
        f"{idx:03d}_"
        f"scn-{_slug(case_cfg['scenario'])}_"
        f"sigma-{_slug(case_cfg['info_sigma'])}_"
        f"delay-{_slug(case_cfg['info_delay_s'])}_"
        f"trust-{_slug(case_cfg['theta_trust'])}"
    )
    spread = case_cfg.get("spread")
    if spread and spread != "none":
        parts += f"_spread-{_slug(spread)}"
    agent_type = case_cfg.get("agent_type", "llm")
    if agent_type != "llm":
        parts += f"_agent-{_slug(agent_type)}"
    if "sumo_seed" in case_cfg:
        parts += f"_seed-{_slug(case_cfg['sumo_seed'])}"
    fpr = case_cfg.get("fire_perception_range_m")
    if fpr is not None and float(fpr) != 1200.0:
        parts += f"_fpr-{_slug(fpr)}"
    return parts


def build_experiment_grid(
    sigma_values: Optional[List[float]] = None,
    delay_values: Optional[List[float]] = None,
    trust_values: Optional[List[float]] = None,
    scenario_modes: Optional[List[str]] = None,
    spread_values: Optional[List[str]] = None,
    base_overrides: Optional[Dict[str, Any]] = None,
    agent_type: str = "llm",
    seed_values: Optional[List[int]] = None,
    perception_range_values: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Build a Cartesian-product experiment grid from parameter value lists.

    Defaults apply when a parameter list is not provided:
        - sigma_values    : [40.0]
        - delay_values    : [0.0]
        - trust_values    : [0.5]
        - scenario_modes  : ["advice_guided"]
        - spread_values   : [None]  (no heterogeneity)
        - seed_values     : [42]
        - perception_range_values : [None]  (not swept)

    Args:
        sigma_values: List of ``INFO_SIGMA`` values to sweep.
        delay_values: List of ``INFO_DELAY_S`` values to sweep.
        trust_values: List of ``DEFAULT_THETA_TRUST`` values to sweep.
        scenario_modes: List of scenario mode strings to sweep.
        spread_values: List of population spread preset names to sweep
            (``"none"``, ``"low"``, ``"moderate"``, ``"high"``).
            When ``None``, spread is omitted from the grid (homogeneous population).
        base_overrides: Additional key-value pairs merged into every case dict
            (e.g., ``{"messaging_enabled": True}``).
        seed_values: List of master seeds for multi-seed replication.  Each value
            seeds every stochastic stream of the run (SUMO traffic, agent
            psychological profiles, rule-based softmax sampling, information
            noise) via the master-seed scheme in ``agentevac/utils/seeding.py``,
            so iterating across this list produces honest Monte-Carlo replicates.
            Stored in each case dict under the legacy key ``sumo_seed``.
        perception_range_values: List of ``FIRE_PERCEPTION_RANGE_M`` values to sweep.
            When ``None``, perception range is not included as a grid axis.

    Returns:
        List of case config dicts, one per grid cell.
    """
    sigma_seq = sigma_values if sigma_values is not None else [40.0]
    delay_seq = delay_values if delay_values is not None else [0.0]
    trust_seq = trust_values if trust_values is not None else [0.5]
    scenario_seq = scenario_modes if scenario_modes is not None else ["advice_guided"]
    spread_seq: List[Optional[str]] = spread_values if spread_values is not None else [None]
    seed_seq = seed_values if seed_values is not None else [42]
    fpr_seq: List[Optional[float]] = (
        [float(v) for v in perception_range_values]
        if perception_range_values is not None
        else [None]
    )

    grid: List[Dict[str, Any]] = []
    for info_sigma, info_delay_s, theta_trust, scenario, spread, seed, fpr in itertools.product(
        sigma_seq,
        delay_seq,
        trust_seq,
        scenario_seq,
        spread_seq,
        seed_seq,
        fpr_seq,
    ):
        case = {
            "info_sigma": float(info_sigma),
            "info_delay_s": float(info_delay_s),
            "theta_trust": float(theta_trust),
            "scenario": str(scenario),
            "agent_type": str(agent_type),
            "sumo_seed": int(seed),
        }
        if spread is not None:
            case["spread"] = str(spread)
        if fpr is not None:
            case["fire_perception_range_m"] = float(fpr)
        if base_overrides:
            case.update(dict(base_overrides))
        grid.append(case)
    return grid


def load_resume_cases(
    resume_path: str,
    *,
    statuses: Optional[List[str]] = None,
    stem: str = "experiment_results",
) -> List[Dict[str, Any]]:
    """Load failed (or otherwise non-ok) cases from a prior experiment_results.json.

    Used by ``--resume-from`` to re-run only the cases that previously failed,
    while skipping ones that completed successfully.

    Args:
        resume_path: Either an ``experiment_results.json`` file or a directory
            containing one. When a directory is given, ``<stem>.json`` is read
            from inside it.
        statuses: Statuses considered "needs re-run". Defaults to ``["failed"]``.
            Pass e.g. ``["failed", "timeout"]`` to also resume timeouts.
        stem: Base filename when ``resume_path`` is a directory.

    Returns:
        A list of case config dicts (each with ``info_sigma``, ``info_delay_s``,
        ``theta_trust``, ``scenario``, and any other keys carried in the prior
        run's ``case`` payload), ready to be passed to ``run_parameter_sweep``.

    Raises:
        FileNotFoundError: If the resolved results JSON does not exist.
        ValueError: If the file does not contain a list of result rows.
    """
    target_statuses = {str(s).strip() for s in (statuses or ["failed"]) if str(s).strip()}
    path = Path(resume_path)
    if path.is_dir():
        path = path / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Resume results file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of result rows in {path}")

    cases: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) not in target_statuses:
            continue
        case_payload = row.get("case") or {}
        if not isinstance(case_payload, dict):
            continue
        case = dict(case_payload)
        case.pop("_case_index", None)
        # Preserve the original case_id so re-run artifacts (metrics_*,
        # routes_*, run_params_*, stdout_*) share filenames with the failed
        # run and can be merged back in place.  Fall back to the row's
        # top-level case_id if the nested copy is missing.
        if not case.get("case_id") and row.get("case_id"):
            case["case_id"] = str(row.get("case_id"))
        cases.append(case)
    return cases


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


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in cmd)


def run_experiment_case(
    case_cfg: Dict[str, Any],
    *,
    script_path: str = "agentevac/simulation/main.py",
    python_executable: Optional[str] = None,
    output_dir: str = "outputs/experiments",
    sumo_binary: str = "sumo",
    run_mode: str = "record",
    timeout_s: Optional[float] = None,
    sumo_seed: Optional[int] = None,
    map_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one parameter-grid case by spawning a simulation subprocess.

    Constructs the CLI command and environment from ``case_cfg``, runs the process,
    captures stdout/stderr, and extracts the metrics and replay file paths from the
    subprocess stdout using fixed prefix patterns.

    Args:
        case_cfg: Case configuration dict (from ``build_experiment_grid``), containing
            at minimum ``info_sigma``, ``info_delay_s``, ``theta_trust``, and
            ``scenario`` keys.
        script_path: Path to the main simulation script (relative to project root).
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
    # Run subprocess from the project root (three levels above this file:
    # agentevac/analysis/experiments.py → agentevac/analysis → agentevac → root).
    project_root = Path(__file__).parents[2]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_index = int(case_cfg.get("_case_index", 0))
    case_id = str(case_cfg.get("case_id") or _case_id(case_cfg, case_index))
    replay_base = out_dir / f"routes_{case_id}.jsonl"
    metrics_base = out_dir / f"metrics_{case_id}.json"
    params_base = out_dir / "run_params.json"
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
        "--params-log-path", str(params_base),
    ]

    messaging_enabled = bool(case_cfg.get("messaging_enabled", True))
    cmd.extend(["--messaging", "on" if messaging_enabled else "off"])
    _agent_type = case_cfg.get("agent_type", "llm")
    cmd.extend(["--agent-type", str(_agent_type)])
    _map = map_name or case_cfg.get("map_name")
    if _map:
        cmd.extend(["--map", str(_map)])
    print(f"[SIM_CLI] case_id={case_id} {_format_cmd(cmd[2:])}")

    env = os.environ.copy()
    env.update({
        "INFO_SIGMA": str(float(case_cfg["info_sigma"])),
        "INFO_DELAY_S": str(float(case_cfg["info_delay_s"])),
        "DEFAULT_THETA_TRUST": str(float(case_cfg["theta_trust"])),
        "SUMO_BINARY": str(sumo_binary),
        "AGENT_TYPE": str(_agent_type),
    })
    if "SOFTMAX_TAU" in case_cfg:
        env["SOFTMAX_TAU"] = str(float(case_cfg["SOFTMAX_TAU"]))
    # The case-config key is named ``sumo_seed`` for legacy compatibility but is
    # actually the *master* seed for this replicate -- it derives every stochastic
    # stream (SUMO traffic, agent psychological profiles, rule-based softmax,
    # information noise) via the master-seed scheme in agentevac/utils/seeding.py.
    # We export it as both MASTER_SEED (so main.py's master-keyed streams pick it up)
    # and SUMO_SEED (so SUMO's --seed sees the same value verbatim, preserving
    # bit-level continuity of the traffic stream against pre-master-seed runs).
    _effective_seed = case_cfg.get("sumo_seed", sumo_seed)
    if _effective_seed is not None:
        env["MASTER_SEED"] = str(int(_effective_seed))
        env["SUMO_SEED"] = str(int(_effective_seed))
    if "fire_perception_range_m" in case_cfg:
        env["FIRE_PERCEPTION_RANGE_M"] = str(float(case_cfg["fire_perception_range_m"]))
    if "grounding_instruction" in case_cfg:
        env["GROUNDING_INSTRUCTION"] = str(case_cfg["grounding_instruction"])
    if "DEFAULT_LAMBDA_E" in case_cfg:
        env["DEFAULT_LAMBDA_E"] = str(float(case_cfg["DEFAULT_LAMBDA_E"]))
    if "DEFAULT_LAMBDA_T" in case_cfg:
        env["DEFAULT_LAMBDA_T"] = str(float(case_cfg["DEFAULT_LAMBDA_T"]))
    _spread_name = case_cfg.get("spread")
    if _spread_name and _spread_name in SPREAD_PRESETS:
        for _sp_key, _sp_val in SPREAD_PRESETS[_spread_name].items():
            env[f"{_sp_key.upper()}_SPREAD"] = str(_sp_val)
    else:
        # Explicitly zero all spread env vars to prevent leakage from the
        # parent shell (e.g. leftover exports from a prior RQ4 run).
        for _sp_key in ("theta_trust", "theta_r", "theta_u", "gamma", "lambda_e", "lambda_t"):
            env[f"{_sp_key.upper()}_SPREAD"] = "0.0"

    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
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
    script_path: str = "agentevac/simulation/main.py",
    python_executable: Optional[str] = None,
    output_dir: str = "outputs/experiments",
    sumo_binary: str = "sumo",
    run_mode: str = "record",
    timeout_s: Optional[float] = None,
    sumo_seed: Optional[int] = None,
    map_name: Optional[str] = None,
    start_index: int = 1,
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
        map_name: Map config directory name (e.g., ``"lytton"``).
        start_index: Starting case index for ID naming (default: 1).

    Returns:
        List of result dicts (one per grid case) from ``run_experiment_case``.
    """
    results: List[Dict[str, Any]] = []
    for idx, raw_case in enumerate(grid, start=start_index):
        case_cfg = dict(raw_case)
        case_cfg["_case_index"] = idx
        if not case_cfg.get("case_id"):
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
                sumo_seed=sumo_seed,
                map_name=map_name,
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
                "spread",
                "agent_type",
                "sumo_seed",
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
                "spread": case.get("spread"),
                "agent_type": case.get("agent_type", "llm"),
                "sumo_seed": case.get("sumo_seed"),
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
    parser.add_argument("--script-path", default="agentevac/simulation/main.py")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--sumo-binary", default="sumo", help="Use 'sumo' for headless batch runs.")
    parser.add_argument("--run-mode", choices=["record", "replay"], default="record")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--sigma-values", default="40.0")
    parser.add_argument("--delay-values", default="0.0")
    parser.add_argument("--trust-values", default="0.5")
    parser.add_argument("--scenario-values", default="advice_guided")
    parser.add_argument(
        "--spread-values",
        default=None,
        help="Comma-separated spread preset names to sweep "
             "(none, low, moderate, high). "
             "Omit to disable population heterogeneity (all spreads = 0).",
    )
    parser.add_argument(
        "--agent-type",
        choices=["llm", "rule_based"],
        default="llm",
        help="Agent type: llm (default) or rule_based (softmax baseline).",
    )
    parser.add_argument("--messaging", choices=["on", "off"], default="on")
    parser.add_argument("--sumo-seed", type=int, default=None,
                        help="Master seed for the run (legacy flag name). Derives SUMO traffic, "
                             "agent psychological profiles, rule-based softmax, and information "
                             "noise -- not just SUMO. Shorthand for --seed-values with a single value.")
    parser.add_argument(
        "--seed-values",
        default=None,
        help="Comma-separated master seeds for multi-seed replication (e.g. '42,43,44'). "
             "Each value seeds every stochastic stream of the run, so iterating across "
             "this list produces honest Monte-Carlo replicates. "
             "Takes priority over --sumo-seed when both are specified.",
    )
    parser.add_argument(
        "--perception-range-values",
        default=None,
        help="Comma-separated FIRE_PERCEPTION_RANGE_M values to sweep (e.g. '800,1200,1600'). "
             "Omit to use the simulation default (1200).",
    )
    parser.add_argument(
        "--map",
        default=os.getenv("MAP_NAME", "lytton"),
        help="Map config directory name under configs/ (default: lytton).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting case index for ID naming (default: 1). "
             "Use e.g. 14 to start from 014 when rerunning a subset.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a prior experiment_results.json (or directory containing one). "
             "When set, the grid is rebuilt from the failed cases in that file and the "
             "--sigma-values / --delay-values / --trust-values / --scenario-values / "
             "--messaging flags are ignored. New runs are written to --output-dir, which "
             "may differ from the resume directory so results can be merged manually.",
    )
    parser.add_argument(
        "--resume-statuses",
        default="failed",
        help="Comma-separated list of statuses to treat as 'needs re-run' when "
             "--resume-from is set (default: failed). Use e.g. 'failed,timeout' to "
             "also re-run timeouts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.resume_from:
        resume_statuses = _parse_str_list(args.resume_statuses)
        grid = load_resume_cases(args.resume_from, statuses=resume_statuses)
        if not grid:
            print(
                f"[EXPERIMENTS] resume_from={args.resume_from} "
                f"statuses={','.join(resume_statuses)} matched_cases=0 — nothing to do."
            )
            return 0
        print(
            f"[EXPERIMENTS] resume_from={args.resume_from} "
            f"statuses={','.join(resume_statuses)} matched_cases={len(grid)}"
        )
    else:
        # --seed-values takes priority; fall back to --sumo-seed as single-element list
        if args.seed_values is not None:
            _seeds = _parse_int_list(args.seed_values)
        elif args.sumo_seed is not None:
            _seeds = [args.sumo_seed]
        else:
            _seeds = [42]
        _fpr = (
            _parse_float_list(args.perception_range_values)
            if args.perception_range_values is not None
            else None
        )
        grid = build_experiment_grid(
            sigma_values=_parse_float_list(args.sigma_values),
            delay_values=_parse_float_list(args.delay_values),
            trust_values=_parse_float_list(args.trust_values),
            scenario_modes=_parse_str_list(args.scenario_values),
            spread_values=_parse_str_list(args.spread_values) if args.spread_values else None,
            base_overrides={
                "messaging_enabled": (args.messaging == "on"),
            },
            agent_type=args.agent_type,
            seed_values=_seeds,
            perception_range_values=_fpr,
        )
    results = run_parameter_sweep(
        grid,
        script_path=args.script_path,
        python_executable=args.python_executable,
        output_dir=args.output_dir,
        sumo_binary=args.sumo_binary,
        run_mode=args.run_mode,
        timeout_s=args.timeout_s,
        sumo_seed=args.sumo_seed,
        map_name=args.map,
        start_index=args.start_index,
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
