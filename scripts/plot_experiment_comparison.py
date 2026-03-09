#!/usr/bin/env python3
"""Compare multiple completed runs from an experiment sweep or metrics glob."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from scripts._plot_common import ensure_output_path, load_json, require_matplotlib
except ModuleNotFoundError:
    from _plot_common import ensure_output_path, load_json, require_matplotlib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple AgentEvac runs from experiment_results.json or a metrics glob."
    )
    parser.add_argument(
        "--results-json",
        help="Path to experiment_results.json from agentevac.analysis.experiments.",
    )
    parser.add_argument(
        "--metrics-glob",
        default="outputs/run_metrics_*.json",
        help="Glob of metrics JSON files used if --results-json is omitted "
             "(default: outputs/run_metrics_*.json).",
    )
    parser.add_argument(
        "--out",
        help="Output PNG path. Defaults to <results>.comparison.png or outputs/metrics_comparison.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure window in addition to saving the PNG.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metrics_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "departure_variability": _safe_float(metrics.get("departure_time_variability")),
        "route_entropy": _safe_float(metrics.get("route_choice_entropy")),
        "hazard_exposure": _safe_float(metrics.get("average_hazard_exposure", {}).get("global_average")),
        "avg_travel_time": _safe_float(metrics.get("average_travel_time", {}).get("average")),
        "arrived_agents": _safe_float(metrics.get("arrived_agents")),
        "departed_agents": _safe_float(metrics.get("departed_agents")),
    }


def load_cases(results_json: Path | None, metrics_glob: str) -> tuple[list[dict[str, Any]], Path]:
    rows: list[dict[str, Any]] = []
    if results_json is not None:
        payload = load_json(results_json)
        if not isinstance(payload, list):
            raise SystemExit(f"Expected a list in {results_json}")
        for item in payload:
            metrics_path = item.get("metrics_path")
            if not metrics_path:
                continue
            path = Path(str(metrics_path))
            if not path.exists():
                continue
            metrics = load_json(path)
            case = item.get("case") or {}
            row = {
                "label": str(item.get("case_id") or path.stem),
                "scenario": str(case.get("scenario", "unknown")),
                "info_sigma": _safe_float(case.get("info_sigma")),
                "info_delay_s": _safe_float(case.get("info_delay_s")),
                "theta_trust": _safe_float(case.get("theta_trust")),
                "metrics_path": str(path),
            }
            row.update(_metrics_row(metrics))
            rows.append(row)
        return rows, results_json

    matches = sorted(Path().glob(metrics_glob))
    if not matches:
        raise SystemExit(f"No metrics files match pattern: {metrics_glob}")
    for path in matches:
        metrics = load_json(path)
        row = {
            "label": path.stem,
            "scenario": "unknown",
            "info_sigma": 0.0,
            "info_delay_s": 0.0,
            "theta_trust": 0.0,
            "metrics_path": str(path),
        }
        row.update(_metrics_row(metrics))
        rows.append(row)
    return rows, matches[-1]


def _scatter_by_scenario(ax, rows: list[dict[str, Any]]) -> None:
    scenario_colors = {
        "no_notice": "#E45756",
        "alert_guided": "#F58518",
        "advice_guided": "#4C78A8",
        "unknown": "#777777",
    }
    seen = set()
    for row in rows:
        scenario = str(row.get("scenario", "unknown"))
        label = scenario if scenario not in seen else None
        seen.add(scenario)
        size = max(30.0, 20.0 + 20.0 * row.get("theta_trust", 0.0))
        ax.scatter(
            row["hazard_exposure"],
            row["avg_travel_time"],
            s=size,
            color=scenario_colors.get(scenario, "#777777"),
            alpha=0.85,
            label=label,
        )
    ax.set_title("Hazard Exposure vs Travel Time")
    ax.set_xlabel("Global Hazard Exposure")
    ax.set_ylabel("Average Travel Time (s)")
    if seen:
        ax.legend()


def _line_vs_sigma(ax, rows: list[dict[str, Any]]) -> None:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get("scenario", "unknown")), []).append(row)
    if not by_scenario:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    for scenario, scenario_rows in sorted(by_scenario.items()):
        ordered = sorted(scenario_rows, key=lambda item: item.get("info_sigma", 0.0))
        xs = [r.get("info_sigma", 0.0) for r in ordered]
        ys = [r.get("route_entropy", 0.0) for r in ordered]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=scenario)
    ax.set_title("Route Entropy vs Info Sigma")
    ax.set_xlabel("INFO_SIGMA")
    ax.set_ylabel("Route Choice Entropy")
    ax.legend()


def _bar_mean_by_scenario(ax, rows: list[dict[str, Any]], field: str, title: str, ylabel: str, color: str) -> None:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row.get("scenario", "unknown")), []).append(float(row.get(field, 0.0)))
    labels = sorted(groups.keys())
    if not labels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    means = [sum(groups[label]) / float(len(groups[label])) for label in labels]
    ax.bar(range(len(labels)), means, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_experiment_comparison(rows: list[dict[str, Any]], *, source_path: Path, out_path: Path, show: bool) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AgentEvac Experiment Comparison\n{source_path.name} | runs={len(rows)}",
        fontsize=14,
    )

    _scatter_by_scenario(axes[0, 0], rows)
    _line_vs_sigma(axes[0, 1], rows)
    _bar_mean_by_scenario(
        axes[1, 0],
        rows,
        field="avg_travel_time",
        title="Mean Travel Time by Scenario",
        ylabel="Average Travel Time (s)",
        color="#4C78A8",
    )
    _bar_mean_by_scenario(
        axes[1, 1],
        rows,
        field="hazard_exposure",
        title="Mean Hazard Exposure by Scenario",
        ylabel="Global Hazard Exposure",
        color="#E45756",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[PLOT] source={source_path}")
    print(f"[PLOT] output={out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    results_path = Path(args.results_json) if args.results_json else None
    if results_path and not results_path.exists():
        raise SystemExit(f"Results JSON does not exist: {results_path}")
    rows, source_path = load_cases(results_path, args.metrics_glob)
    out_path = ensure_output_path(source_path, args.out, suffix="comparison")
    plot_experiment_comparison(rows, source_path=source_path, out_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
