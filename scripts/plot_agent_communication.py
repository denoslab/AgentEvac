#!/usr/bin/env python3
"""Visualize agent-to-agent messaging and LLM dialog volume for one run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    from scripts._plot_common import ensure_output_path, load_jsonl, require_matplotlib, resolve_input, top_items
except ModuleNotFoundError:
    from _plot_common import ensure_output_path, load_jsonl, require_matplotlib, resolve_input, top_items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize messaging and dialog activity from events JSONL and dialogs CSV."
    )
    parser.add_argument(
        "--events",
        help="Path to an events_*.jsonl file. Defaults to the newest outputs/events_*.jsonl.",
    )
    parser.add_argument(
        "--dialogs",
        help="Path to a *.dialogs.csv file. Defaults to the newest outputs/*.dialogs.csv.",
    )
    parser.add_argument(
        "--out",
        help="Output PNG path. Defaults to <events>.communication.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure window in addition to saving the PNG.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Maximum number of bars to draw in sender/recipient charts (default: 15).",
    )
    return parser.parse_args()


def _load_dialog_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _draw_bar(ax, items: list[tuple[str, float]], title: str, ylabel: str, color: str) -> None:
    if not items:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        ax.set_axis_off()
        return
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    ax.bar(range(len(values)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def _round_value(rec: dict[str, Any]) -> int | None:
    for key in ("delivery_round", "deliver_round", "sent_round", "round"):
        value = rec.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _plot_round_series(ax, event_rows: list[dict[str, Any]]) -> None:
    series = {
        "queued": {},
        "delivered": {},
        "llm": {},
        "predeparture": {},
    }
    for rec in event_rows:
        event = rec.get("event")
        round_idx = _round_value(rec)
        if round_idx is None:
            continue
        if event == "message_queued":
            series["queued"][round_idx] = series["queued"].get(round_idx, 0) + 1
        elif event == "message_delivered":
            series["delivered"][round_idx] = series["delivered"].get(round_idx, 0) + 1
        elif event == "llm_decision":
            series["llm"][round_idx] = series["llm"].get(round_idx, 0) + 1
        elif event == "predeparture_llm_decision":
            series["predeparture"][round_idx] = series["predeparture"].get(round_idx, 0) + 1

    plotted = False
    colors = {
        "queued": "#4C78A8",
        "delivered": "#54A24B",
        "llm": "#F58518",
        "predeparture": "#E45756",
    }
    for name, mapping in series.items():
        if not mapping:
            continue
        xs = sorted(mapping.keys())
        ys = [mapping[x] for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=name, color=colors[name])
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return
    ax.set_title("Message and Decision Volume by Round")
    ax.set_xlabel("Decision Round")
    ax.set_ylabel("Event Count")
    ax.legend()


def _plot_dialog_modes(ax, dialog_rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    response_lengths: dict[str, list[int]] = {}
    for row in dialog_rows:
        mode = str(row.get("control_mode") or "unknown")
        counts[mode] = counts.get(mode, 0) + 1
        response_text = row.get("response_text") or ""
        response_lengths.setdefault(mode, []).append(len(response_text))

    labels = sorted(counts.keys())
    if not labels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    xs = list(range(len(labels)))
    count_vals = [counts[label] for label in labels]
    avg_lens = [
        (sum(response_lengths[label]) / float(len(response_lengths[label])))
        if response_lengths[label] else 0.0
        for label in labels
    ]

    ax.bar(xs, count_vals, color="#72B7B2", label="dialogs")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Dialog Volume and Avg Response Length")
    ax.set_ylabel("Dialog Count")

    ax2 = ax.twinx()
    ax2.plot(xs, avg_lens, color="#B279A2", marker="o", linewidth=1.8, label="avg response chars")
    ax2.set_ylabel("Average Response Length (chars)")


def plot_agent_communication(
    *,
    events_path: Path,
    dialogs_path: Path,
    out_path: Path,
    show: bool,
    top_n: int,
) -> None:
    plt = require_matplotlib()
    event_rows = load_jsonl(events_path)
    dialog_rows = _load_dialog_rows(dialogs_path)

    sender_counts: dict[str, int] = {}
    recipient_counts: dict[str, int] = {}
    for rec in event_rows:
        event = rec.get("event")
        if event == "message_queued":
            sender = str(rec.get("from_id") or "unknown")
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        elif event == "message_delivered":
            recipient = str(rec.get("to_id") or "unknown")
            recipient_counts[recipient] = recipient_counts.get(recipient, 0) + 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AgentEvac Communication Analysis\n{events_path.name} | {dialogs_path.name}",
        fontsize=14,
    )

    _draw_bar(
        axes[0, 0],
        top_items({k: float(v) for k, v in sender_counts.items()}, top_n),
        f"Top Message Senders (top {top_n})",
        "Queued Messages",
        "#4C78A8",
    )
    _draw_bar(
        axes[0, 1],
        top_items({k: float(v) for k, v in recipient_counts.items()}, top_n),
        f"Top Message Recipients (top {top_n})",
        "Delivered Messages",
        "#54A24B",
    )
    _plot_round_series(axes[1, 0], event_rows)
    _plot_dialog_modes(axes[1, 1], dialog_rows)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[PLOT] events={events_path}")
    print(f"[PLOT] dialogs={dialogs_path}")
    print(f"[PLOT] output={out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    events_path = resolve_input(args.events, "outputs/events_*.jsonl")
    dialogs_path = resolve_input(args.dialogs, "outputs/*.dialogs.csv")
    out_path = ensure_output_path(events_path, args.out, suffix="communication")
    plot_agent_communication(
        events_path=events_path,
        dialogs_path=dialogs_path,
        out_path=out_path,
        show=args.show,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
