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
    """Parse CLI arguments for the run-metrics dashboard."""
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
    """Draw a bar panel, or a centered placeholder if no rows are available."""
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


def _kpi_specs(metrics: dict) -> list[dict[str, object]]:
    """Build the four top-level KPI descriptors used in the dashboard header panel."""
    return [
        {
            "title": "Departure variance",
            "value": float(metrics.get("departure_time_variability", 0.0)),
            "ylabel": "Seconds^2",
            "color": "#4C78A8",
            "fmt": "{:.3f}",
        },
        {
            "title": "Route entropy",
            "value": float(metrics.get("route_choice_entropy", 0.0)),
            "ylabel": "Entropy (nats)",
            "color": "#F58518",
            "fmt": "{:.3f}",
        },
        {
            "title": "Hazard exposure",
            "value": float(metrics.get("average_hazard_exposure", {}).get("global_average", 0.0)),
            "ylabel": "Average risk score",
            "color": "#E45756",
            "fmt": "{:.3f}",
        },
        {
            "title": "Avg travel time",
            "value": float(metrics.get("average_travel_time", {}).get("average", 0.0)),
            "ylabel": "Seconds",
            "color": "#54A24B",
            "fmt": "{:.2f}",
        },
    ]


def _plot_kpi_grid(fig, slot, metrics: dict) -> None:
    """Render the KPI summary as four mini subplots with independent y scales."""
    kpi_grid = slot.subgridspec(2, 2, wspace=0.35, hspace=0.45)
    for idx, spec in enumerate(_kpi_specs(metrics)):
        ax = fig.add_subplot(kpi_grid[idx // 2, idx % 2])
        value = float(spec["value"])
        ymax = max(1.0, value * 1.15) if value >= 0.0 else max(1.0, abs(value) * 1.15)
        ax.bar([0], [value], color=str(spec["color"]), width=0.5)
        ax.set_title(str(spec["title"]), fontsize=10)
        ax.set_ylabel(str(spec["ylabel"]), fontsize=9)
        ax.set_xticks([])
        ax.set_ylim(min(0.0, value * 1.1), ymax)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        label = str(spec["fmt"]).format(value)
        text_y = value if value > 0.0 else ymax * 0.04
        va = "bottom"
        if value < 0.0:
            text_y = value
            va = "top"
        ax.text(0, text_y, label, ha="center", va=va, fontsize=10)


def plot_metrics_dashboard(metrics_path: Path, *, out_path: Path, show: bool, top_n: int) -> None:
    """Render the run-metrics dashboard and save it to ``out_path``."""
    plt = require_matplotlib()
    metrics = load_json(metrics_path)
    exposure = metrics.get("average_hazard_exposure", {}).get("per_agent_average", {}) or {}
    travel = metrics.get("average_travel_time", {}).get("per_agent", {}) or {}
    instability = metrics.get("decision_instability", {}).get("per_agent_changes", {}) or {}

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.3)
    fig.suptitle(
        f"AgentEvac Run Metrics\n{metrics_path.name} | mode={metrics.get('run_mode', 'unknown')} "
        f"| departed={metrics.get('departed_agents', 0)} | arrived={metrics.get('arrived_agents', 0)}",
        fontsize=14,
    )

    _plot_kpi_grid(fig, grid[0, 0], metrics)
    ax_travel = fig.add_subplot(grid[0, 1])
    ax_exposure = fig.add_subplot(grid[1, 0])
    ax_instability = fig.add_subplot(grid[1, 1])

    _draw_or_empty(
        ax_travel,
        top_items(travel, top_n),
        f"Per-Agent Travel Time (top {top_n})",
        "Seconds",
        "#4C78A8",
    )
    _draw_or_empty(
        ax_exposure,
        top_items(exposure, top_n),
        f"Per-Agent Hazard Exposure (top {top_n})",
        "Average Risk Score",
        "#E45756",
    )
    _draw_or_empty(
        ax_instability,
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
    """CLI entry point for the run-metrics dashboard."""
    args = _parse_args()
    metrics_path = resolve_input(args.metrics, "outputs/run_metrics_*.json")
    out_path = ensure_output_path(metrics_path, args.out, suffix="dashboard")
    plot_metrics_dashboard(metrics_path, out_path=out_path, show=args.show, top_n=args.top_n)


if __name__ == "__main__":
    main()
