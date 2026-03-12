#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(12345 12346 12347 12348 12349)

for seed in "${SEEDS[@]}"; do
  echo "[STAGE2] seed=${seed}"
  SUMO_SEED="$seed" python3 -m agentevac.analysis.experiments \
    --output-dir "outputs/stage2/uncertainty_seed_${seed}" \
    --sumo-binary sumo \
    --sigma-values 0,20,40,80 \
    --delay-values 0,30,60 \
    --trust-values 0.5 \
    --scenario-values no_notice,alert_guided,advice_guided \
    --messaging on
done
