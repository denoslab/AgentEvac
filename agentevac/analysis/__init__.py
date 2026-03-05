"""Calibration, experiment sweep, and metrics analysis modules.

    calibration   — weighted-loss fit of run metrics against a reference scenario.
    experiments   — Cartesian-product parameter sweep driver.
    metrics       — per-run KPI collection (travel time, entropy, exposure, etc.).
    study_runner  — end-to-end orchestrator: sweep → fit → report.
"""
