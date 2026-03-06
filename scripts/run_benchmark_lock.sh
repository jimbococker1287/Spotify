#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "$1" != --* ]]; then
  BENCHMARK_ID="$1"
  shift
else
  BENCHMARK_ID="$(date +%Y%m%d-%H%M%S)"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

SEEDS="${BENCHMARK_SEEDS:-11 42 77}"
DEEP_MODELS="${DEEP_MODELS:-dense,gru_artist,lstm}"
CLASSICAL_MODELS="${CLASSICAL_MODELS:-logreg,random_forest,extra_trees}"
EPOCHS="${EPOCHS:-6}"

export SPOTIFY_FORCE_CPU="${SPOTIFY_FORCE_CPU:-1}"
export SPOTIFY_RUN_EAGER="${SPOTIFY_RUN_EAGER:-0}"
export SPOTIFY_DISABLE_MONITOR="${SPOTIFY_DISABLE_MONITOR:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/outputs/history/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/outputs/history/.cache}"

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "Benchmark lock id: $BENCHMARK_ID"
echo "Seeds: $SEEDS"
echo "Deep models: $DEEP_MODELS"
echo "Classical models: $CLASSICAL_MODELS"

for seed in $SEEDS; do
  RUN_NAME="benchmark-lock-${BENCHMARK_ID}-seed-${seed}"
  echo ">>> Running ${RUN_NAME}"
  "$PYTHON_CMD" -m spotify \
    --profile small \
    --run-name "$RUN_NAME" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --models "$DEEP_MODELS" \
    --classical-models "$CLASSICAL_MODELS" \
    --mlflow \
    --no-optuna \
    --no-temporal-backtest \
    --no-shap \
    "$@"
done

"$PYTHON_CMD" scripts/aggregate_benchmark.py --benchmark-id "$BENCHMARK_ID"
echo "Benchmark lock complete: outputs/history/benchmark_lock_${BENCHMARK_ID}_summary.csv"
