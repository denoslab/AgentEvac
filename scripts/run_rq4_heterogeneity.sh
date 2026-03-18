#!/usr/bin/env bash
# ==============================================================================
# RQ4: Population Heterogeneity — Diversity as Resilience or Fragility
#
# Research question:
#   Does heterogeneity in risk tolerance and decision weights improve
#   system-level evacuation resilience, and does the optimal diversity
#   level depend on the information regime?
#
# Design:
#   IV1: spread_level       = {none, low, moderate, high}
#   Moderator: scenario     = {no_notice, alert_guided, advice_guided}
#   INFO_SIGMA: 40          (fixed — moderate noise)
#   INFO_DELAY_S: 30        (fixed — moderate delay)
#   theta_trust mean: 0.5   (fixed)
#   Messaging: on           (fixed — social channel needed for canary cascade)
#   Seeds: 10               (more seeds for stochastic spread effects)
#
# Spread levels (std-dev of truncated normal around population means):
#   ┌──────────────┬──────┬──────┬──────────┬──────┐
#   │ Parameter    │ none │ low  │ moderate │ high │
#   ├──────────────┼──────┼──────┼──────────┼──────┤
#   │ theta_trust  │ 0.0  │ 0.05 │ 0.12    │ 0.20 │
#   │ theta_r      │ 0.0  │ 0.03 │ 0.08    │ 0.15 │
#   │ theta_u      │ 0.0  │ 0.03 │ 0.08    │ 0.15 │
#   │ gamma        │ 0.0  │ 0.001│ 0.003   │ 0.005│
#   │ lambda_e     │ 0.0  │ 0.15 │ 0.4     │ 0.8  │
#   │ lambda_t     │ 0.0  │ 0.03 │ 0.08    │ 0.15 │
#   └──────────────┴──────┴──────┴──────────┴──────┘
#
# Grid: 4 spread × 3 scenario × 10 seeds = 120 runs
#
# Primary DVs:
#   - arrived_agents, average_hazard_exposure, average_travel_time
#   - departure_time_variability, route_choice_entropy, decision_instability
#
# Per-agent DVs (for equity / Gini analysis):
#   - per_agent hazard exposure, per_agent travel time
#
# Time-series DVs (post-process from replay JSONL):
#   - departure reason distribution (canary cascade detection)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(12345 12346 12347 12348 12349 12350 12351 12352 12353 12354)

# Spread level definitions: (label, theta_trust, theta_r, theta_u, gamma, lambda_e, lambda_t)
SPREAD_LABELS=( "none"     "low"   "moderate" "high" )
SPREAD_TRUST=(  "0.0"      "0.05"  "0.12"     "0.20" )
SPREAD_TR=(     "0.0"      "0.03"  "0.08"     "0.15" )
SPREAD_TU=(     "0.0"      "0.03"  "0.08"     "0.15" )
SPREAD_GAMMA=(  "0.0"      "0.001" "0.003"    "0.005")
SPREAD_LE=(     "0.0"      "0.15"  "0.4"      "0.8"  )
SPREAD_LT=(     "0.0"      "0.03"  "0.08"     "0.15" )

for i in "${!SPREAD_LABELS[@]}"; do
  spread="${SPREAD_LABELS[$i]}"
  for seed in "${SEEDS[@]}"; do
    echo "============================================"
    echo "[RQ4] spread=${spread} seed=${seed}"
    echo "============================================"
    SUMO_SEED="$seed" \
    THETA_TRUST_SPREAD="${SPREAD_TRUST[$i]}" \
    THETA_R_SPREAD="${SPREAD_TR[$i]}" \
    THETA_U_SPREAD="${SPREAD_TU[$i]}" \
    GAMMA_SPREAD="${SPREAD_GAMMA[$i]}" \
    LAMBDA_E_SPREAD="${SPREAD_LE[$i]}" \
    LAMBDA_T_SPREAD="${SPREAD_LT[$i]}" \
    python3 -m agentevac.analysis.experiments \
      --output-dir "outputs/rq4/heterogeneity_spread_${spread}_seed_${seed}" \
      --sumo-binary sumo \
      --sigma-values 40 \
      --delay-values 30 \
      --trust-values 0.5 \
      --scenario-values no_notice,alert_guided,advice_guided \
      --messaging on
  done
done

echo "[RQ4] All conditions complete."
