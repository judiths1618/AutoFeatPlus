#!/usr/bin/env bash
# Run every AutoFeat benchmark scenario end-to-end via auto_pipeline.
# Logs per-scenario output to results/6g_data/run_logs/<label>.log.
set -uo pipefail

cd "$(dirname "$0")/.."
RUN() { conda run --no-capture-output -n autofeat-6g python -m feature_discovery.auto_pipeline "$@"; }
LOGDIR="results/6g_data/run_logs"
mkdir -p "$LOGDIR"

declare -a STATUS

run_scenario() {
  local label="$1"; shift
  echo "============================================================"
  echo ">>> $label"
  echo "============================================================"
  if RUN "$@" --label "$label" > "$LOGDIR/$label.log" 2>&1; then
    echo "    OK   $label"
    STATUS+=("OK   $label")
  else
    echo "    FAIL $label (see $LOGDIR/$label.log)"
    STATUS+=("FAIL $label")
  fi
}

run_scenario scenario1 \
  --base-table datasets/scenario1/rabbitmq-s1.csv --target lat99 \
  --data-dir datasets/scenario1 --dataset-type regression --algorithms XGB

run_scenario scenario2c \
  --base-table datasets/scenario2c/rabbitmq-reduced.csv --target lat99 \
  --data-dir datasets/scenario2c --dataset-type regression \
  --temporal-key time --temporal-tolerance 0 --no-transformer-discovery --algorithms XGB

run_scenario scenarioA_lat95 \
  --base-table datasets/scenarioA_lat95/rabbitmq-reduced.csv --target lat95 \
  --data-dir datasets/scenarioA_lat95 --dataset-type regression \
  --temporal-key time --temporal-tolerance 60 --algorithms XGB

run_scenario scenarioA_lat99 \
  --base-table datasets/scenarioA_lat99/rabbitmq-reduced.csv --target lat99 \
  --data-dir datasets/scenarioA_lat99 --dataset-type regression \
  --temporal-key time --temporal-tolerance 60 --algorithms XGB

run_scenario scenarioB_amf_seg01 \
  --base-table datasets/scenarioB_seg01/amf-seg01-reduced.csv --target lat99 \
  --data-dir datasets/scenarioB_seg01 --dataset-type regression --algorithms XGB

run_scenario scenarioN_target_n \
  --base-table datasets/EUR/6907619/rabbitmq-performance.csv --target n \
  --data-dir datasets/EUR/6907619 --dataset-type regression \
  --temporal-key time --temporal-tolerance 60 --algorithms XGB

run_scenario scenarioK_kul \
  --base-table datasets/scenarioK_kul/samples_base.csv --target target_x \
  --data-dir datasets/scenarioK_kul --dataset-type binary \
  --no-transformer-discovery --algorithms XGB

run_scenario scenarioK_csi \
  --base-table datasets/scenarioK_csi/samples_base.csv --target target_x \
  --data-dir datasets/scenarioK_csi --dataset-type binary \
  --no-transformer-discovery --algorithms XGB

echo
echo "==================== RUN SUMMARY ===================="
printf '%s\n' "${STATUS[@]}"
echo "====================================================="
