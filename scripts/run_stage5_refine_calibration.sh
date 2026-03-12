#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(12345 12346 12347 12348 12349)

for seed in "${SEEDS[@]}"; do
  echo "[STAGE5] seed=${seed}"
  SUMO_SEED="$seed" python3 -m agentevac.analysis.experiments \
    --output-dir "outputs/stage5/refined_seed_${seed}" \
    --sumo-binary sumo \
    --sigma-values 20,40,60 \
    --delay-values 15,30,45 \
    --trust-values 0.25,0.5,0.75 \
    --scenario-values alert_guided,advice_guided \
    --messaging on
done
