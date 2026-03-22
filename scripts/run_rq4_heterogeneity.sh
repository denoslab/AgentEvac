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

# Run each spread level as a separate block to avoid bash indexed arrays.
run_spread() {
  local spread="$1"
  local s_trust="$2" s_tr="$3" s_tu="$4" s_gamma="$5" s_le="$6" s_lt="$7"
  for seed in 12345 12346 12347 12348 12349 12350 12351 12352 12353 12354; do
    echo "============================================"
    echo "[RQ4] spread=${spread} seed=${seed}"
    echo "============================================"
    THETA_TRUST_SPREAD="$s_trust" \
    THETA_R_SPREAD="$s_tr" \
    THETA_U_SPREAD="$s_tu" \
    GAMMA_SPREAD="$s_gamma" \
    LAMBDA_E_SPREAD="$s_le" \
    LAMBDA_T_SPREAD="$s_lt" \
    python3 -m agentevac.analysis.experiments \
      --output-dir "outputs/rq4/heterogeneity_spread_${spread}_seed_${seed}" \
      --sumo-binary sumo \
      --sumo-seed "$seed" \
      --sigma-values 40 \
      --delay-values 30 \
      --trust-values 0.5 \
      --scenario-values no_notice,alert_guided,advice_guided \
      --messaging on
  done
}

run_spread none     0.0   0.0   0.0   0.0   0.0   0.0
run_spread low      0.05  0.03  0.03  0.001 0.15  0.03
run_spread moderate 0.12  0.08  0.08  0.003 0.4   0.08
run_spread high     0.20  0.15  0.15  0.005 0.8   0.15

echo "[RQ4] All conditions complete."
