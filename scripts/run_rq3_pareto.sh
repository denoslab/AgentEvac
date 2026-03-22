#!/usr/bin/env bash
# ==============================================================================
# RQ3: Safety–Efficiency Pareto Frontier Across Alerting Regimes
#
# Research question:
#   How does the safety-efficiency trade-off (average hazard exposure E_avg
#   vs average travel time T_avg) change across combinations of information
#   quality, delay, and trust? How does the resulting Pareto frontier shift
#   under the three alerting regimes (no-notice, alert-guided, advice-guided)?
#
# Design:
#   IV1: INFO_SIGMA         = {20, 40, 80}
#   IV2: INFO_DELAY_S       = {0, 30, 60}
#   IV3: DEFAULT_THETA_TRUST = {0.25, 0.5, 0.75}
#   Grouping: scenario      = {no_notice, alert_guided, advice_guided}
#   Messaging: on           (fixed)
#   Population: homogeneous (all spreads = 0)
#   Seeds: 5                (stochastic replication)
#
# Grid: 3 sigma × 3 delay × 3 trust × 3 scenario × 5 seeds = 405 runs
#
# Primary DVs:
#   - average_hazard_exposure (E_avg — safety axis)
#   - average_travel_time     (T_avg — efficiency axis)
#   - arrived_agents          (completion filter)
#
# Per-agent DVs (for equity analysis):
#   - per_agent hazard exposure
#   - per_agent travel time
#
# Analysis-level metrics (computed post-hoc):
#   - Pareto frontier per scenario
#   - Hypervolume indicator per scenario
#   - Frontier shift between scenarios
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

for seed in 12345 12346 12347 12348 12349; do
  echo "============================================"
  echo "[RQ3] seed=${seed}"
  echo "============================================"
  python3 -m agentevac.analysis.experiments \
    --output-dir "outputs/rq3/pareto_seed_${seed}" \
    --sumo-binary sumo \
    --sumo-seed "$seed" \
    --sigma-values 20,40,80 \
    --delay-values 0,30,60 \
    --trust-values 0.25,0.5,0.75 \
    --scenario-values no_notice,alert_guided,advice_guided \
    --messaging on
done

echo "[RQ3] All seeds complete."
