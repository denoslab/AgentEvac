#!/usr/bin/env bash
# ==============================================================================
# RQ1: Information Quality → Departure Timing, Route Choice, Decision Instability
#
# Research question:
#   How do information quality factors (observation noise σ_info, information
#   delay, and conflicting signals) affect departure timing, route-choice
#   distribution, and decision instability?
#
# Design:
#   IV1: INFO_SIGMA         = {0, 20, 40, 80}        (observation noise)
#   IV2: INFO_DELAY_S       = {0, 15, 30, 60}        (information delay)
#   Moderator: scenario     = {no_notice, alert_guided, advice_guided}
#   Messaging: on           (fixed — required for signal conflict to emerge)
#   theta_trust: 0.5        (fixed — social channel held neutral)
#   Population: homogeneous (all spreads = 0)
#   Seeds: 5                (stochastic replication)
#
# Grid: 4 sigma × 4 delay × 3 scenario × 5 seeds = 240 runs
#
# Primary DVs:
#   - departure_time_variability
#   - route_choice_entropy
#   - decision_instability (average_changes, max_changes)
#   - average_signal_conflict (JSD — measures conflicting signals)
#   - destination_choice_share
#   - departed_agents, arrived_agents
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(12345 12346 12347 12348 12349)

for seed in "${SEEDS[@]}"; do
  echo "============================================"
  echo "[RQ1] seed=${seed}"
  echo "============================================"
  SUMO_SEED="$seed" python3 -m agentevac.analysis.experiments \
    --output-dir "outputs/rq1/info_quality_seed_${seed}" \
    --sumo-binary sumo \
    --sigma-values 0,20,40,80 \
    --delay-values 0,15,30,60 \
    --trust-values 0.5 \
    --scenario-values no_notice,alert_guided,advice_guided \
    --messaging on
done

echo "[RQ1] All seeds complete."
