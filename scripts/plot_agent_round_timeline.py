#!/usr/bin/env python3
"""Plot a round-based agent timeline with departure, arrival, and route-change overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from scripts._plot_common import load_json, load_jsonl, require_matplotlib, resolve_input
except ModuleNotFoundError:
    from _plot_common import load_json, load_jsonl, require_matplotlib, resolve_input


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one row per agent from departure round to arrival round, "
                    "with route-change rounds highlighted."
    )
    parser.add_argument(
        "--run-id",
        help="Timestamp token such as 20260309_030340. Used to resolve matching outputs files.",
    )
    parser.add_argument(
        "--events",
        help="Path to an events_*.jsonl file. Defaults to the newest outputs/events_*.jsonl.",
    )
    parser.add_argument(
        "--replay",
        help="Path to an llm_routes_*.jsonl file. Defaults to the newest outputs/llm_routes_*.jsonl.",
    )
    parser.add_argument(
        "--metrics",
        help="Path to a run_metrics_*.json file. Defaults to the newest outputs/run_metrics_*.json.",
    )
    parser.add_argument(
        "--out",
        help="Output PNG path. Defaults to <events>.round_timeline.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure window in addition to saving the PNG.",
    )
    parser.add_argument(
        "--include-no-departure",
        action="store_true",
        help="Also show agents without a departure_release event, starting from their first route change.",
    )
    return parser.parse_args()


def _round_table(event_rows: list[dict[str, Any]]) -> list[tuple[int, float]]:
    rounds = []
    for rec in event_rows:
        if rec.get("event") != "decision_round_start":
            continue
        if rec.get("round") is None or rec.get("sim_t_s") is None:
            continue
        rounds.append((int(rec["round"]), float(rec["sim_t_s"])))
    rounds = sorted(set(rounds), key=lambda item: item[0])
    if not rounds:
        raise SystemExit("No decision_round_start events found; cannot build round timeline.")
    return rounds


def _round_for_time(t: float, rounds: list[tuple[int, float]]) -> int:
    """Return the latest decision round whose time is <= ``t``."""
    selected = rounds[0][0]
    for round_idx, round_t in rounds:
        if round_t <= float(t) + 1e-9:
            selected = round_idx
        else:
            break
    return selected


def _departure_times(event_rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for rec in event_rows:
        if rec.get("event") != "departure_release":
            continue
        vid = rec.get("veh_id")
        sim_t = rec.get("sim_t_s")
        if vid is None or sim_t is None:
            continue
        out.setdefault(str(vid), float(sim_t))
    return out


def _route_change_times(replay_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for rec in replay_rows:
        if rec.get("event") != "route_change":
            continue
        vid = rec.get("veh_id")
        sim_t = rec.get("time_s")
        if vid is None or sim_t is None:
            continue
        out.setdefault(str(vid), []).append(float(sim_t))
    for vid in out:
        out[vid] = sorted(set(out[vid]))
    return out


def _timeline_rows(
    event_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    include_no_departure: bool,
) -> tuple[list[dict[str, Any]], int]:
    rounds = _round_table(event_rows)
    final_round = rounds[-1][0]
    departures = _departure_times(event_rows)
    route_changes = _route_change_times(replay_rows)
    travel_times = metrics.get("average_travel_time", {}).get("per_agent", {}) or {}

    all_agent_ids = set(departures.keys())
    if include_no_departure:
        all_agent_ids.update(route_changes.keys())

    rows: list[dict[str, Any]] = []
    for vid in sorted(all_agent_ids):
        depart_time = departures.get(vid)
        change_times = route_changes.get(vid, [])

        if depart_time is None:
            if not include_no_departure or not change_times:
                continue
            start_round = _round_for_time(change_times[0], rounds)
            status = "no_departure_event"
        else:
            start_round = _round_for_time(depart_time, rounds)
            status = "completed" if vid in travel_times else "incomplete"

        if vid in travel_times and depart_time is not None:
            arrival_time = float(depart_time) + float(travel_times[vid])
            end_round = _round_for_time(arrival_time, rounds)
            status = "completed"
        else:
            end_round = final_round

        end_round = max(end_round, start_round)
        change_rounds = sorted({_round_for_time(t, rounds) for t in change_times if _round_for_time(t, rounds) >= start_round})

        rows.append({
            "veh_id": vid,
            "start_round": start_round,
            "end_round": end_round,
            "change_rounds": change_rounds,
            "status": status,
        })

    rows.sort(key=lambda row: (row["start_round"], row["veh_id"]))
    return rows, final_round


def plot_agent_round_timeline(
    *,
    events_path: Path,
    replay_path: Path,
    metrics_path: Path,
    out_path: Path,
    show: bool,
    include_no_departure: bool,
) -> None:
    plt = require_matplotlib()
    from matplotlib.patches import Patch

    event_rows = load_jsonl(events_path)
    replay_rows = load_jsonl(replay_path)
    metrics = load_json(metrics_path)
    timeline_rows, final_round = _timeline_rows(
        event_rows,
        replay_rows,
        metrics,
        include_no_departure=include_no_departure,
    )
    if not timeline_rows:
        raise SystemExit("No agent timeline rows could be constructed from the provided artifacts.")

    fig_h = max(6.0, 0.32 * len(timeline_rows) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.suptitle(
        f"Agent Round Timeline\n{events_path.name} | {replay_path.name} | {metrics_path.name}",
        fontsize=14,
    )

    yticks = []
    ylabels = []
    base_colors = {
        "completed": "#4C78A8",
        "incomplete": "#999999",
        "no_departure_event": "#BBBBBB",
    }

    for idx, row in enumerate(timeline_rows):
        y = idx
        yticks.append(y)
        ylabels.append(row["veh_id"])
        start = float(row["start_round"]) - 0.5
        width = float(row["end_round"] - row["start_round"] + 1)
        color = base_colors.get(row["status"], "#4C78A8")
        hatch = "//" if row["status"] != "completed" else None
        ax.broken_barh(
            [(start, width)],
            (y - 0.35, 0.7),
            facecolors=color,
            edgecolors="black",
            linewidth=0.4,
            hatch=hatch,
            alpha=0.9,
        )
        change_segments = [(float(round_idx) - 0.5, 1.0) for round_idx in row["change_rounds"]]
        if change_segments:
            ax.broken_barh(
                change_segments,
                (y - 0.35, 0.7),
                facecolors="#F58518",
                edgecolors="#C04B00",
                linewidth=0.4,
            )

    ax.set_xlim(0.5, final_round + 0.5)
    ax.set_ylim(-1, len(timeline_rows))
    ax.set_xlabel("Decision Round")
    ax.set_ylabel("Agent")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax.legend(
        handles=[
            Patch(facecolor="#4C78A8", edgecolor="black", label="Active interval"),
            Patch(facecolor="#F58518", edgecolor="#C04B00", label="Route/destination change round"),
            Patch(facecolor="#999999", edgecolor="black", hatch="//", label="Still active at run end / inferred"),
        ],
        loc="upper right",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[PLOT] events={events_path}")
    print(f"[PLOT] replay={replay_path}")
    print(f"[PLOT] metrics={metrics_path}")
    print(f"[PLOT] output={out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.run_id:
        run_id = str(args.run_id)
        events_default = f"outputs/events_{run_id}.jsonl"
        replay_default = f"outputs/llm_routes_{run_id}.jsonl"
        metrics_default = f"outputs/run_metrics_{run_id}.json"
    else:
        events_default = "outputs/events_*.jsonl"
        replay_default = "outputs/llm_routes_*.jsonl"
        metrics_default = "outputs/run_metrics_*.json"

    events_path = resolve_input(args.events, events_default)
    replay_path = resolve_input(args.replay, replay_default)
    metrics_path = resolve_input(args.metrics, metrics_default)
    out_path = (
        Path(args.out)
        if args.out
        else events_path.with_suffix("").with_name(f"{events_path.with_suffix('').name}.round_timeline.png")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_agent_round_timeline(
        events_path=events_path,
        replay_path=replay_path,
        metrics_path=metrics_path,
        out_path=out_path,
        show=args.show,
        include_no_departure=args.include_no_departure,
    )


if __name__ == "__main__":
    main()
