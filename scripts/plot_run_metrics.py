#!/usr/bin/env python3
"""Plot a compact dashboard for one completed simulation metrics JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._plot_common import ensure_output_path, load_json, require_matplotlib, resolve_input, top_items
except ModuleNotFoundError:
    from _plot_common import ensure_output_path, load_json, require_matplotlib, resolve_input, top_items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one run_metrics_*.json file as a 2x2 dashboard."
    )
    parser.add_argument(
        "--metrics",
        help="Path to a metrics JSON file. Defaults to the newest outputs/run_metrics_*.json.",
    )
    parser.add_argument(
        "--out",
        help="Output PNG path. Defaults to <metrics>.dashboard.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure window in addition to saving the PNG.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of per-agent bars to draw in each panel (default: 20).",
    )
    return parser.parse_args()


def _draw_or_empty(ax, items: list[tuple[str, float]], title: str, ylabel: str, color: str, *, highest_first: bool = True):
    if not items:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        ax.set_axis_off()
        return
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    if not highest_first:
        labels = list(reversed(labels))
        values = list(reversed(values))
    ax.bar(range(len(values)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_metrics_dashboard(metrics_path: Path, *, out_path: Path, show: bool, top_n: int) -> None:
    plt = require_matplotlib()
    metrics = load_json(metrics_path)

    kpis = {
        "Departure variance": float(metrics.get("departure_time_variability", 0.0)),
        "Route entropy": float(metrics.get("route_choice_entropy", 0.0)),
        "Hazard exposure": float(metrics.get("average_hazard_exposure", {}).get("global_average", 0.0)),
        "Avg travel time": float(metrics.get("average_travel_time", {}).get("average", 0.0)),
    }
    exposure = metrics.get("average_hazard_exposure", {}).get("per_agent_average", {}) or {}
    travel = metrics.get("average_travel_time", {}).get("per_agent", {}) or {}
    instability = metrics.get("decision_instability", {}).get("per_agent_changes", {}) or {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AgentEvac Run Metrics\n{metrics_path.name} | mode={metrics.get('run_mode', 'unknown')} "
        f"| departed={metrics.get('departed_agents', 0)} | arrived={metrics.get('arrived_agents', 0)}",
        fontsize=14,
    )

    axes[0, 0].bar(range(len(kpis)), list(kpis.values()), color=["#4C78A8", "#F58518", "#E45756", "#54A24B"])
    axes[0, 0].set_xticks(range(len(kpis)))
    axes[0, 0].set_xticklabels(list(kpis.keys()), rotation=20, ha="right")
    axes[0, 0].set_title("Run KPI Summary")
    axes[0, 0].set_ylabel("Value")

    _draw_or_empty(
        axes[0, 1],
        top_items(travel, top_n),
        f"Per-Agent Travel Time (top {top_n})",
        "Seconds",
        "#4C78A8",
    )
    _draw_or_empty(
        axes[1, 0],
        top_items(exposure, top_n),
        f"Per-Agent Hazard Exposure (top {top_n})",
        "Average Risk Score",
        "#E45756",
    )
    _draw_or_empty(
        axes[1, 1],
        top_items({k: float(v) for k, v in instability.items()}, top_n),
        f"Per-Agent Decision Instability (top {top_n})",
        "Choice Changes",
        "#72B7B2",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[PLOT] metrics={metrics_path}")
    print(f"[PLOT] output={out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    metrics_path = resolve_input(args.metrics, "outputs/run_metrics_*.json")
    out_path = ensure_output_path(metrics_path, args.out, suffix="dashboard")
    plot_metrics_dashboard(metrics_path, out_path=out_path, show=args.show, top_n=args.top_n)


if __name__ == "__main__":
    main()
