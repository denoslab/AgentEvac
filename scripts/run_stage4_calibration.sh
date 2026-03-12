#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

REFERENCE_PATH="${1:-outputs/reference_metrics.json}"

python3 -m agentevac.analysis.study_runner \
  --reference "$REFERENCE_PATH" \
  --output-dir "outputs/stage4" \
  --sumo-binary sumo \
  --sigma-values 0,20,40,80 \
  --delay-values 0,30,60,120 \
  --trust-values 0.0,0.25,0.5,0.75,1.0 \
  --scenario-values no_notice,alert_guided,advice_guided \
  --messaging on \
  --top-k 10
