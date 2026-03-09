#!/usr/bin/env python3
"""Plot departure and communication timelines from completed simulation logs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._plot_common import (
        bin_counts,
        ensure_output_path,
        load_jsonl,
        require_matplotlib,
        resolve_input,
    )
except ModuleNotFoundError:
    from _plot_common import (
        bin_counts,
        ensure_output_path,
        load_jsonl,
        require_matplotlib,
        resolve_input,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize departures, messages, and route changes over time."
    )
    parser.add_argument(
        "--events",
        help="Path to an events_*.jsonl file. Defaults to the newest outputs/events_*.jsonl.",
    )
    parser.add_argument(
        "--replay",
        help="Optional llm_routes_*.jsonl replay log for route-change counts.",
    )
    parser.add_argument(
        "--out",
        help="Output PNG path. Defaults to <events>.timeline.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure window in addition to saving the PNG.",
    )
    parser.add_argument(
        "--bin-s",
        type=float,
        default=30.0,
        help="Time-bin width in seconds for event counts (default: 30).",
    )
    return parser.parse_args()


def _extract_times(rows: list[dict], event_type: str) -> list[float]:
    out = []
    for rec in rows:
        if rec.get("event") == event_type and rec.get("time_s") is not None:
            out.append(float(rec["time_s"]))
    return sorted(out)


def _plot_cumulative(ax, times: list[float], title: str, color: str) -> None:
    if not times:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        ax.set_axis_off()
        return
    y = list(range(1, len(times) + 1))
    ax.step(times, y, where="post", color=color, linewidth=2)
    ax.scatter(times, y, color=color, s=16)
    ax.set_title(title)
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Cumulative Count")


def _plot_binned(ax, series: list[tuple[str, list[float], str]], *, bin_s: float) -> None:
    plotted = False
    for label, times, color in series:
        binned = bin_counts(times, bin_s=bin_s)
        if not binned:
            continue
        xs = [x for x, _ in binned]
        ys = [y for _, y in binned]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=label, color=color)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return
    ax.set_title(f"Event Volume per {int(bin_s) if float(bin_s).is_integer() else bin_s}s Bin")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Event Count")
    ax.legend()


def plot_timeline(events_path: Path, *, replay_path: Path | None, out_path: Path, show: bool, bin_s: float) -> None:
    plt = require_matplotlib()
    event_rows = load_jsonl(events_path)
    replay_rows = load_jsonl(replay_path) if replay_path else []

    departure_times = _extract_times(event_rows, "departure_release")
    message_times = _extract_times(event_rows, "message_delivered") + _extract_times(event_rows, "message_queued")
    observation_times = _extract_times(event_rows, "system_observation_generated")
    llm_times = _extract_times(event_rows, "llm_decision") + _extract_times(event_rows, "predeparture_llm_decision")
    route_change_times = _extract_times(replay_rows, "route_change")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        f"AgentEvac Timeline\n{events_path.name}" + (f" | replay={replay_path.name}" if replay_path else ""),
        fontsize=14,
    )

    _plot_cumulative(axes[0], departure_times, "Cumulative Departures", "#E45756")
    _plot_binned(
        axes[1],
        [
            ("Messages", sorted(message_times), "#4C78A8"),
            ("System observations", sorted(observation_times), "#54A24B"),
            ("LLM decisions", sorted(llm_times), "#F58518"),
            ("Route changes", sorted(route_change_times), "#B279A2"),
        ],
        bin_s=bin_s,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[PLOT] events={events_path}")
    if replay_path:
        print(f"[PLOT] replay={replay_path}")
    print(f"[PLOT] output={out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    events_path = resolve_input(args.events, "outputs/events_*.jsonl")
    replay_path = Path(args.replay) if args.replay else None
    if replay_path and not replay_path.exists():
        raise SystemExit(f"Replay file does not exist: {replay_path}")
    out_path = ensure_output_path(events_path, args.out, suffix="timeline")
    plot_timeline(events_path, replay_path=replay_path, out_path=out_path, show=args.show, bin_s=args.bin_s)


if __name__ == "__main__":
    main()
