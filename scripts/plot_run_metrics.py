#!/usr/bin/env python3
"""Plot a compact dashboard for one completed simulation metrics JSON."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

try:
    from scripts._plot_common import (
        ensure_output_path,
        load_json,
        require_matplotlib,
        resolve_input,
        resolve_optional_run_params,
        top_items,
    )
    from scripts.plot_experiment_comparison import (
        SCENARIO_COLORS,
        SCENARIO_ORDER,
        load_cases,
    )
except ModuleNotFoundError:
    from _plot_common import (
        ensure_output_path,
        load_json,
        require_matplotlib,
        resolve_input,
        resolve_optional_run_params,
        top_items,
    )
    from plot_experiment_comparison import (
        SCENARIO_COLORS,
        SCENARIO_ORDER,
        load_cases,
    )


_CASE_INDEX_RE = re.compile(r"_(\d{3,})_")


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
        "--metrics-glob",
        help="Glob of metrics JSON files. When set, plots a multi-run KPI comparison "
             "(2x2 grid of departure variance, route entropy, hazard exposure, avg travel time) "
             "with bars sorted contiguously by scenario, instead of the single-run dashboard.",
    )
    parser.add_argument(
        "--params",
        help="Optional companion run_params JSON path. Defaults to the matching run_params_<id>.json when present.",
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


def _briefing_summary(params: dict | None) -> str | None:
    """Format driver-briefing thresholds for the dashboard footer."""
    if not params:
        return None
    briefing = params.get("driver_briefing_thresholds") or {}
    if not briefing:
        return None
    return (
        "Briefing thresholds: "
        f"margin_m={briefing.get('margin_very_close_m', '?')}/"
        f"{briefing.get('margin_near_m', '?')}/"
        f"{briefing.get('margin_buffered_m', '?')} "
        f"risk_density={briefing.get('risk_density_low', '?')}/"
        f"{briefing.get('risk_density_medium', '?')}/"
        f"{briefing.get('risk_density_high', '?')} "
        f"delay_ratio={briefing.get('delay_fast_ratio', '?')}/"
        f"{briefing.get('delay_moderate_ratio', '?')}/"
        f"{briefing.get('delay_heavy_ratio', '?')} "
        f"advisory_margin_m={briefing.get('caution_min_margin_m', '?')}/"
        f"{briefing.get('recommended_min_margin_m', '?')}"
    )


def _kpi_multirun_specs() -> list[dict[str, str]]:
    """Field/title/color descriptors for the multi-run KPI panels."""
    return [
        {
            "field": "departure_variability",
            "title": "Departure variance",
            "ylabel": "Seconds^2",
            "fmt": "{:.2f}",
        },
        {
            "field": "route_entropy",
            "title": "Route entropy",
            "ylabel": "Entropy (nats)",
            "fmt": "{:.3f}",
        },
        {
            "field": "hazard_exposure",
            "title": "Hazard exposure",
            "ylabel": "Average risk score",
            "fmt": "{:.3f}",
        },
        {
            "field": "avg_travel_time",
            "title": "Avg travel time",
            "ylabel": "Seconds",
            "fmt": "{:.1f}",
        },
    ]


def _short_run_label(row: dict[str, Any]) -> str:
    """Return a compact bar label such as ``001`` extracted from the row label."""
    label = str(row.get("label", ""))
    match = _CASE_INDEX_RE.search(label)
    if match:
        return match.group(1)
    return label[:12] or "?"


def _sort_rows_by_scenario(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows so each scenario forms a contiguous group, then by label."""
    order_index = {name: i for i, name in enumerate(SCENARIO_ORDER)}

    def key(row: dict[str, Any]) -> tuple[int, str]:
        scenario = str(row.get("scenario", "unknown"))
        return (order_index.get(scenario, len(SCENARIO_ORDER)), str(row.get("label", "")))

    return sorted(rows, key=key)


def plot_kpi_multirun(
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    out_path: Path,
    show: bool,
) -> None:
    """Render a 2x2 KPI comparison across multiple runs, grouped by scenario."""
    plt = require_matplotlib()
    if not rows:
        raise SystemExit("No runs to plot.")

    ordered = _sort_rows_by_scenario(rows)
    short_labels = [_short_run_label(row) for row in ordered]
    scenarios = [str(row.get("scenario", "unknown")) for row in ordered]
    colors = [SCENARIO_COLORS.get(scn, SCENARIO_COLORS["unknown"]) for scn in scenarios]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AgentEvac Multi-Run KPIs\n{source_path.name} | runs={len(ordered)}",
        fontsize=14,
    )

    for idx, spec in enumerate(_kpi_multirun_specs()):
        ax = axes[idx // 2, idx % 2]
        values = [float(row.get(spec["field"], 0.0)) for row in ordered]
        positions = list(range(len(values)))
        ax.bar(positions, values, color=colors)
        ax.set_title(spec["title"], fontsize=11)
        ax.set_ylabel(spec["ylabel"], fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(short_labels, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ymax = max(values) if values else 0.0
        ymin = min(values) if values else 0.0
        if ymax > 0.0:
            ax.set_ylim(min(0.0, ymin * 1.1), ymax * 1.18)
        for pos, val in zip(positions, values):
            ax.text(
                pos,
                val if val >= 0.0 else 0.0,
                spec["fmt"].format(val),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    seen_scenarios: list[str] = []
    for scn in scenarios:
        if scn not in seen_scenarios:
            seen_scenarios.append(scn)
    ordered_legend = [scn for scn in SCENARIO_ORDER if scn in seen_scenarios]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SCENARIO_COLORS.get(scn, SCENARIO_COLORS["unknown"]))
        for scn in ordered_legend
    ]
    if handles:
        fig.legend(
            handles,
            ordered_legend,
            loc="lower center",
            ncol=len(ordered_legend),
            frameon=False,
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[PLOT] source={source_path}")
    print(f"[PLOT] output={out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_metrics_dashboard(
    metrics_path: Path,
    *,
    out_path: Path,
    show: bool,
    top_n: int,
    params_path: Path | None = None,
) -> None:
    """Render the run-metrics dashboard and save it to ``out_path``."""
    plt = require_matplotlib()
    metrics = load_json(metrics_path)
    params = load_json(params_path) if params_path else None
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

    footer = _briefing_summary(params)
    rect_bottom = 0.04 if footer else 0.0
    if footer:
        fig.text(0.02, 0.012, footer, ha="left", va="bottom", fontsize=8)

    fig.tight_layout(rect=(0, rect_bottom, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[PLOT] metrics={metrics_path}")
    if params_path:
        print(f"[PLOT] params={params_path}")
    print(f"[PLOT] output={out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    """CLI entry point for the run-metrics dashboard."""
    args = _parse_args()
    if args.metrics_glob:
        rows, source_path = load_cases(None, args.metrics_glob)
        out_path = ensure_output_path(source_path, args.out, suffix="kpi_comparison")
        plot_kpi_multirun(
            rows,
            source_path=source_path,
            out_path=out_path,
            show=args.show,
        )
        return
    metrics_path = resolve_input(args.metrics, "outputs/run_metrics_*.json")
    params_path = resolve_optional_run_params(args.params, metrics_path)
    out_path = ensure_output_path(metrics_path, args.out, suffix="dashboard")
    plot_metrics_dashboard(
        metrics_path,
        out_path=out_path,
        show=args.show,
        top_n=args.top_n,
        params_path=params_path,
    )


if __name__ == "__main__":
    main()
