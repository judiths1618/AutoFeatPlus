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
  --base-table scenarios/scenario1/rabbitmq-s1.csv --target lat99 \
  --data-dir scenarios/scenario1 --dataset-type regression --algorithms XGB

run_scenario scenario2c \
  --base-table scenarios/scenario2c/rabbitmq-reduced.csv --target lat99 \
  --data-dir scenarios/scenario2c --dataset-type regression \
  --temporal-key time --temporal-tolerance 0 --no-transformer-discovery --algorithms XGB

run_scenario scenarioA_lat95 \
  --base-table scenarios/scenarioA_lat95/rabbitmq-reduced.csv --target lat95 \
  --data-dir scenarios/scenarioA_lat95 --dataset-type regression \
  --temporal-key time --temporal-tolerance 60 --algorithms XGB

run_scenario scenarioA_lat99 \
  --base-table scenarios/scenarioA_lat99/rabbitmq-reduced.csv --target lat99 \
  --data-dir scenarios/scenarioA_lat99 --dataset-type regression \
  --temporal-key time --temporal-tolerance 60 --algorithms XGB

run_scenario scenarioB_amf_seg01 \
  --base-table scenarios/scenarioB_seg01/amf-seg01-reduced.csv --target lat99 \
  --data-dir scenarios/scenarioB_seg01 --dataset-type regression --algorithms XGB

run_scenario scenarioN_target_n \
  --base-table scenarios/scenarioN_target_n/rabbitmq-performance.csv --target n \
  --data-dir scenarios/scenarioN_target_n --dataset-type regression \
  --temporal-key time --temporal-tolerance 60 --algorithms XGB

run_scenario scenarioK_csi \
  --base-table scenarios/scenarioK_csi/samples_base.csv --target target_x \
  --data-dir scenarios/scenarioK_csi --dataset-type binary \
  --no-transformer-discovery --algorithms XGB

run_scenario scenarioR_resource \
  --base-table scenarios/scenarioR_resource/rabbitmq-reduced.csv --target lat99 \
  --data-dir scenarios/scenarioR_resource --dataset-type regression \
  --temporal-key time --temporal-tolerance 60 --no-transformer-discovery --algorithms XGB

run_scenario scenarioU_unrelated \
  --base-table scenarios/scenarioU_unrelated/rabbitmq-reduced.csv --target lat99 \
  --data-dir scenarios/scenarioU_unrelated --dataset-type regression \
  --no-transformer-discovery --algorithms XGB

echo
echo "==================== RUN SUMMARY ===================="
printf '%s\n' "${STATUS[@]}"
echo "====================================================="
