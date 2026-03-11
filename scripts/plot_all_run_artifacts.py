#!/usr/bin/env python3
"""Generate all standard figures for one completed AgentEvac run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from scripts._plot_common import newest_file
    from scripts.plot_agent_communication import plot_agent_communication
    from scripts.plot_agent_round_timeline import plot_agent_round_timeline
    from scripts.plot_departure_timeline import plot_timeline
    from scripts.plot_experiment_comparison import load_cases, plot_experiment_comparison
    from scripts.plot_run_metrics import plot_metrics_dashboard
except ModuleNotFoundError:
    from _plot_common import newest_file
    from plot_agent_communication import plot_agent_communication
    from plot_agent_round_timeline import plot_agent_round_timeline
    from plot_departure_timeline import plot_timeline
    from plot_experiment_comparison import load_cases, plot_experiment_comparison
    from plot_run_metrics import plot_metrics_dashboard


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the standard dashboard, timeline, comparison, and communication plots for one run."
    )
    parser.add_argument("--run-id", help="Timestamp token such as 20260309_030340.")
    parser.add_argument("--metrics", help="Explicit run_metrics JSON path.")
    parser.add_argument("--events", help="Explicit events JSONL path.")
    parser.add_argument("--replay", help="Explicit llm_routes JSONL path.")
    parser.add_argument("--dialogs", help="Explicit dialogs CSV path.")
    parser.add_argument(
        "--results-json",
        help="Optional experiment_results.json to also generate the multi-run comparison figure.",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory. Defaults to outputs/figures/<run-id-or-latest>/.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively as they are generated.")
    parser.add_argument("--top-n", type=int, default=15, help="Top-N bars for agent-level charts.")
    parser.add_argument("--bin-s", type=float, default=30.0, help="Time-bin width in seconds for timeline counts.")
    return parser.parse_args()


def _maybe_path(path_arg: str | None) -> Path | None:
    if not path_arg:
        return None
    path = Path(path_arg)
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")
    return path


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    for path_arg in (args.events, args.metrics, args.replay, args.dialogs):
        if path_arg:
            match = re.search(r"(\d{8}_\d{6})", Path(path_arg).name)
            if match:
                return match.group(1)
    newest = newest_file("outputs/events_*.jsonl")
    stem = newest.stem
    return stem.replace("events_", "", 1)


def _resolve_paths(args: argparse.Namespace, run_id: str) -> dict[str, Path | None]:
    metrics = _maybe_path(args.metrics)
    events = _maybe_path(args.events)
    replay = _maybe_path(args.replay)
    dialogs = _maybe_path(args.dialogs)

    if metrics is None:
        candidate = Path(f"outputs/run_metrics_{run_id}.json")
        metrics = candidate if candidate.exists() else newest_file("outputs/run_metrics_*.json")
    if events is None:
        candidate = Path(f"outputs/events_{run_id}.jsonl")
        events = candidate if candidate.exists() else newest_file("outputs/events_*.jsonl")
    if replay is None:
        candidate = Path(f"outputs/llm_routes_{run_id}.jsonl")
        replay = candidate if candidate.exists() else None
    if dialogs is None:
        candidate = Path(f"outputs/llm_routes_{run_id}.dialogs.csv")
        dialogs = candidate if candidate.exists() else newest_file("outputs/*.dialogs.csv")

    return {
        "metrics": metrics,
        "events": events,
        "replay": replay,
        "dialogs": dialogs,
    }


def main() -> None:
    args = _parse_args()
    run_id = _resolve_run_id(args)
    paths = _resolve_paths(args, run_id)

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/figures") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = paths["metrics"]
    events_path = paths["events"]
    replay_path = paths["replay"]
    dialogs_path = paths["dialogs"]
    assert metrics_path is not None
    assert events_path is not None
    assert dialogs_path is not None

    plot_metrics_dashboard(
        metrics_path,
        out_path=out_dir / "run_metrics.dashboard.png",
        show=args.show,
        top_n=args.top_n,
    )
    plot_timeline(
        events_path,
        replay_path=replay_path,
        out_path=out_dir / "run_timeline.png",
        show=args.show,
        bin_s=args.bin_s,
    )
    plot_agent_communication(
        events_path=events_path,
        dialogs_path=dialogs_path,
        out_path=out_dir / "agent_communication.png",
        show=args.show,
        top_n=args.top_n,
    )
    if replay_path is not None:
        plot_agent_round_timeline(
            events_path=events_path,
            replay_path=replay_path,
            metrics_path=metrics_path,
            out_path=out_dir / "agent_round_timeline.png",
            show=args.show,
            include_no_departure=False,
        )
    comparison_source: Path | None = None
    if args.results_json:
        results_path = Path(args.results_json)
        if not results_path.exists():
            raise SystemExit(f"Results JSON does not exist: {results_path}")
        comparison_rows, comparison_source = load_cases(results_path, "outputs/run_metrics_*.json")
        plot_experiment_comparison(
            comparison_rows,
            source_path=comparison_source,
            out_path=out_dir / "experiment_comparison.png",
            show=args.show,
        )
    else:
        metrics_matches = sorted(Path().glob("outputs/run_metrics_*.json"))
        if len(metrics_matches) > 1:
            comparison_rows, comparison_source = load_cases(None, "outputs/run_metrics_*.json")
            plot_experiment_comparison(
                comparison_rows,
                source_path=comparison_source,
                out_path=out_dir / "experiment_comparison.png",
                show=args.show,
            )

    print(f"[PLOT] run_id={run_id}")
    print(f"[PLOT] figures_dir={out_dir}")
    print(f"[PLOT] metrics={metrics_path}")
    print(f"[PLOT] events={events_path}")
    if replay_path:
        print(f"[PLOT] replay={replay_path}")
    print(f"[PLOT] dialogs={dialogs_path}")
    if comparison_source:
        print(f"[PLOT] comparison_source={comparison_source}")


if __name__ == "__main__":
    main()
