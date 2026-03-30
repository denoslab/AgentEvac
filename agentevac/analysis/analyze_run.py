"""Compact post-run analysis report.

Usage:
    python -m agentevac.analysis.analyze_run <run_id_or_path>

Examples:
    python -m agentevac.analysis.analyze_run 20260325_175136
    python -m agentevac.analysis.analyze_run outputs/run_metrics_20260325_175136.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

OUTPUTS_DIR = Path("outputs")

_ARTIFACT_PREFIXES = ("run_metrics_", "run_params_", "agent_profiles_",
                      "events_", "llm_routes_")


def _resolve_suffix(arg: str) -> str:
    """Turn a CLI argument into a timestamp suffix like '20260325_175136'."""
    p = Path(arg)
    if p.exists():
        from agentevac.utils.run_parameters import reference_suffix
        return reference_suffix(p)
    # Treat as bare suffix
    return arg


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _fmt(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"


def _top_n(d: Dict[str, float], n: int = 3, *, reverse: bool = True) -> List[str]:
    """Return top-n agent ids sorted by value (highest first by default)."""
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)
    return [f"{k} ({_fmt(v)})" for k, v in items[:n]]


def _bottom_n(d: Dict[str, float], n: int = 3) -> List[str]:
    return _top_n(d, n, reverse=False)


def analyze(suffix: str, outputs_dir: Path = OUTPUTS_DIR) -> str:
    """Build a compact text report for the given run suffix."""
    metrics = _load_json(outputs_dir / f"run_metrics_{suffix}.json")
    params = _load_json(outputs_dir / f"run_params_{suffix}.json")
    profiles = _load_json(outputs_dir / f"agent_profiles_{suffix}.json")

    if metrics is None:
        return f"ERROR: run_metrics_{suffix}.json not found in {outputs_dir}"

    lines: List[str] = []
    lines.append(f"=== Run Analysis: {suffix} ===\n")

    # --- Configuration ---
    lines.append("Configuration:")
    if params:
        lines.append(f"  Scenario:    {params.get('scenario', '?')}")
        lines.append(f"  Run mode:    {params.get('run_mode', '?')}")
        lines.append(f"  Sim end:     {params.get('sim_end_time_s', '?')}s")
        lines.append(f"  SUMO binary: {params.get('sumo_binary', '?')}")

        fires = params.get("fire_sources", [])
        lines.append(f"  Fire sources: {len(fires)}")
        if fires:
            ids = [f.get("id", "?") for f in fires]
            lines.append(f"    IDs: {', '.join(ids)}")

        cog = params.get("cognition", {})
        lines.append(f"  info_sigma:  {cog.get('info_sigma', '?')}")
        lines.append(f"  info_delay:  {cog.get('info_delay_s', '?')}s")
        lines.append(f"  theta_trust: {cog.get('theta_trust', '?')} (default)")
        lines.append(f"  belief_inertia: {cog.get('belief_inertia', '?')}")

        dep = params.get("departure", {})
        lines.append(f"  departure theta_r: {dep.get('theta_r', '?')} (default)")

        util = params.get("utility", {})
        lines.append(f"  utility lambda_e: {util.get('lambda_e', '?')}, lambda_t: {util.get('lambda_t', '?')} (default)")

        msg = params.get("messaging_controls", {})
        if msg.get("enabled"):
            lines.append(f"  Messaging: ON (inbox={msg.get('max_inbox_messages')}, "
                         f"sends/round={msg.get('max_sends_per_agent_per_round')}, "
                         f"ttl={msg.get('ttl_rounds')})")
        else:
            lines.append("  Messaging: OFF")
    else:
        lines.append("  (run_params file not found)")

    # --- Agents overview ---
    departed = metrics.get("departed_agents", 0)
    arrived = metrics.get("arrived_agents", 0)
    pct = f"{arrived / departed * 100:.0f}%" if departed else "N/A"
    lines.append(f"\nAgents: {departed} departed, {arrived} arrived ({pct})")
    lines.append(f"Decision snapshots: {metrics.get('decision_snapshot_count', '?')}")

    # --- Destination shares ---
    dest = metrics.get("destination_choice_share", {})
    counts = dest.get("counts", {})
    fracs = dest.get("fractions", {})
    if counts:
        lines.append("\nDestination Shares:")
        for name in sorted(counts, key=lambda k: counts[k], reverse=True):
            f = fracs.get(name, 0)
            lines.append(f"  {name}: {counts[name]:>2} ({f * 100:.1f}%)")
        lines.append(f"  Route choice entropy: {_fmt(metrics.get('route_choice_entropy', 0), 4)}")

    # --- Travel time ---
    tt = metrics.get("average_travel_time", {})
    per_agent_tt = tt.get("per_agent", {})
    if per_agent_tt:
        vals = list(per_agent_tt.values())
        lines.append(f"\nTravel Time:")
        lines.append(f"  Average: {_fmt(tt.get('average', 0), 1)}s")
        lines.append(f"  Median:  {_fmt(statistics.median(vals), 1)}s")
        lines.append(f"  Std dev: {_fmt(statistics.stdev(vals), 1)}s" if len(vals) > 1 else "")
        mn_agent = min(per_agent_tt, key=per_agent_tt.get)
        mx_agent = max(per_agent_tt, key=per_agent_tt.get)
        lines.append(f"  Fastest: {mn_agent} ({_fmt(per_agent_tt[mn_agent], 1)}s)")
        lines.append(f"  Slowest: {mx_agent} ({_fmt(per_agent_tt[mx_agent], 1)}s)")
        lines.append(f"  Slowest 3: {', '.join(_top_n(per_agent_tt, 3))}")

    # --- Hazard exposure ---
    haz = metrics.get("average_hazard_exposure", {})
    per_agent_haz = haz.get("per_agent_average", {})
    if per_agent_haz:
        lines.append(f"\nHazard Exposure:")
        lines.append(f"  Global average: {_fmt(haz.get('global_average', 0), 4)}")
        lines.append(f"  Sample count:   {haz.get('sample_count', '?')}")
        fully_exposed = [k for k, v in per_agent_haz.items() if v >= 1.0]
        low_exposure = [k for k, v in per_agent_haz.items() if v < 0.2]
        if fully_exposed:
            lines.append(f"  Fully exposed (>=1.0): {', '.join(fully_exposed)}")
        if low_exposure:
            lines.append(f"  Low exposure  (<0.2):  {', '.join(_bottom_n({k: per_agent_haz[k] for k in low_exposure}, 5))}")
        lines.append(f"  Highest 3: {', '.join(_top_n(per_agent_haz, 3))}")
        lines.append(f"  Lowest  3: {', '.join(_bottom_n(per_agent_haz, 3))}")

    # --- Decision instability ---
    inst = metrics.get("decision_instability", {})
    per_agent_inst = inst.get("per_agent_changes", {})
    if per_agent_inst:
        lines.append(f"\nDecision Instability:")
        lines.append(f"  Average changes: {_fmt(inst.get('average_changes', 0), 2)}")
        lines.append(f"  Max changes:     {inst.get('max_changes', 0)}")
        stable = [k for k, v in per_agent_inst.items() if v == 0]
        lines.append(f"  Stable (0 changes): {len(stable)} agents")
        unstable = {k: float(v) for k, v in per_agent_inst.items() if v > 0}
        if unstable:
            lines.append(f"  Most unstable: {', '.join(_top_n(unstable, 3))}")

    # --- Signal conflict ---
    sc = metrics.get("average_signal_conflict", {})
    if sc:
        lines.append(f"\nSignal Conflict:")
        lines.append(f"  Global average: {_fmt(sc.get('global_average', 0), 4)}")
        lines.append(f"  Sample count:   {sc.get('sample_count', '?')}")

    # --- Departure time variability ---
    dtv = metrics.get("departure_time_variability")
    if dtv is not None:
        lines.append(f"\nDeparture time variability: {_fmt(dtv, 4)}")

    # --- Agent profiles ---
    if profiles:
        lines.append(f"\nAgent Profiles ({len(profiles)} agents):")
        _param_keys = ["theta_r", "lambda_e", "lambda_t", "gamma", "theta_trust"]
        for pk in _param_keys:
            vals = [p[pk] for p in profiles.values() if pk in p]
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lines.append(f"  {pk}: all {_fmt(lo, 4)}")
            else:
                lines.append(f"  {pk}: {_fmt(lo, 4)} — {_fmt(hi, 4)}  "
                             f"(mean {_fmt(statistics.mean(vals), 4)})")
    else:
        lines.append("\n  (agent_profiles file not found)")

    # --- Behavioral flags ---
    flags: List[str] = []
    if per_agent_haz:
        n_full = len([v for v in per_agent_haz.values() if v >= 1.0])
        if n_full:
            flags.append(f"{n_full} agent(s) fully exposed (hazard >= 1.0)")
    if per_agent_tt and per_agent_haz:
        # Agents with high travel time but low exposure → likely rerouted successfully
        med_tt = statistics.median(per_agent_tt.values())
        for vid in per_agent_tt:
            t = per_agent_tt[vid]
            h = per_agent_haz.get(vid, 0)
            if t > med_tt * 2.5 and h < 0.2:
                flags.append(f"{vid}: very slow ({_fmt(t, 0)}s) but low exposure "
                             f"({_fmt(h, 3)}) — likely rerouted to avoid fire")
            if t < med_tt * 0.5 and h >= 0.9:
                flags.append(f"{vid}: fast ({_fmt(t, 0)}s) but high exposure "
                             f"({_fmt(h, 3)}) — drove through hazard zone")
    if per_agent_inst:
        high_inst = [k for k, v in per_agent_inst.items() if v >= 5]
        if high_inst:
            flags.append(f"Highly indecisive (>=5 changes): {', '.join(high_inst)}")

    if flags:
        lines.append("\nBehavioral Flags:")
        for f in flags:
            lines.append(f"  * {f}")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    suffix = _resolve_suffix(sys.argv[1])

    # Allow overriding outputs dir via second arg
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUTS_DIR

    report = analyze(suffix, out_dir)
    print(report)


if __name__ == "__main__":
    main()
