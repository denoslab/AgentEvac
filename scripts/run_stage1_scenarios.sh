#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(12345 12346 12347 12348 12349)

for messaging in on off; do
  for seed in "${SEEDS[@]}"; do
    echo "[STAGE1] messaging=${messaging} seed=${seed}"
    SUMO_SEED="$seed" python3 -m agentevac.analysis.experiments \
      --output-dir "outputs/stage1/scenarios_msg_${messaging}_seed_${seed}" \
      --sumo-binary sumo \
      --sigma-values 40 \
      --delay-values 30 \
      --trust-values 0.5 \
      --scenario-values no_notice,alert_guided,advice_guided \
      --messaging "$messaging"
  done
done
