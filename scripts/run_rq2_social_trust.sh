#!/usr/bin/env bash
# ==============================================================================
# RQ2: Social Cues × Trust → Cascades, Herding, Belief Dynamics
#
# Research question:
#   Given fixed information quality, how do social cues and trust (θ_trust)
#   interact to shape collective departure cascades, herding in route/
#   destination choice, and changes in decision instability and belief
#   uncertainty over time?
#
# Design:
#   IV1: DEFAULT_THETA_TRUST = {0.0, 0.25, 0.5, 0.75, 1.0}
#   IV2: messaging           = {on, off}
#   Moderator: scenario      = {no_notice, alert_guided, advice_guided}
#   INFO_SIGMA: 40           (fixed — moderate noise)
#   INFO_DELAY_S: 30         (fixed — moderate delay)
#   Population: homogeneous  (all spreads = 0)
#   Seeds: 5                 (stochastic replication)
#
# Grid: 5 trust × 2 messaging × 3 scenario × 5 seeds = 150 runs
#
# Primary DVs (from metrics JSON):
#   - departure_time_variability  (cascade synchronization)
#   - route_choice_entropy        (herding)
#   - decision_instability        (flip-flopping)
#   - average_signal_conflict     (env vs. social disagreement)
#   - destination_choice_share    (which destinations attract herds)
#   - departed_agents, arrived_agents
#
# Time-series DVs (post-process from replay JSONL):
#   - belief entropy over time    (agent_cognition events)
#   - confidence over time        (agent_cognition events)
#   - departure reason counts     (departure_release events)
#   - cascade chain analysis      (departure_release reason=neighbor_departure_activity)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

for messaging in on off; do
  for seed in 12345 12346 12347 12348 12349; do
    echo "============================================"
    echo "[RQ2] messaging=${messaging} seed=${seed}"
    echo "============================================"
    python3 -m agentevac.analysis.experiments \
      --output-dir "outputs/rq2/social_trust_msg_${messaging}_seed_${seed}" \
      --sumo-binary sumo \
      --sumo-seed "$seed" \
      --sigma-values 40 \
      --delay-values 30 \
      --trust-values 0.0,0.25,0.5,0.75,1.0 \
      --scenario-values no_notice,alert_guided,advice_guided \
      --messaging "$messaging"
  done
done

echo "[RQ2] All conditions complete."
