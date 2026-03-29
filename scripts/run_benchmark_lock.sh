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
BENCHMARK_ENABLE_MLFLOW="${BENCHMARK_ENABLE_MLFLOW:-1}"
BENCHMARK_CLASSICAL_ONLY="${BENCHMARK_CLASSICAL_ONLY:-0}"
BENCHMARK_RESUME="${BENCHMARK_RESUME:-1}"
BENCHMARK_HISTORY_CSV="${BENCHMARK_HISTORY_CSV:-$ROOT_DIR/outputs/history/experiment_history.csv}"

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
echo "Classical only: $BENCHMARK_CLASSICAL_ONLY"

for seed in $SEEDS; do
  RUN_NAME="benchmark-lock-${BENCHMARK_ID}-seed-${seed}"
  if [[ "$BENCHMARK_RESUME" == "1" ]] && [[ -f "$BENCHMARK_HISTORY_CSV" ]]; then
    if "$PYTHON_CMD" - "$BENCHMARK_HISTORY_CSV" "$RUN_NAME" <<'PY'
import csv
import sys
from pathlib import Path

history_csv = Path(sys.argv[1])
run_name = sys.argv[2]
with history_csv.open("r", encoding="utf-8") as infile:
    found = any(str(row.get("run_name", "")).strip() == run_name for row in csv.DictReader(infile))
raise SystemExit(0 if found else 1)
PY
    then
      echo ">>> Skipping ${RUN_NAME} (already present in experiment history)"
      continue
    fi
  fi
  echo ">>> Running ${RUN_NAME}"
  CMD=(
    "$PYTHON_CMD" -m spotify
    --profile small \
    --run-name "$RUN_NAME" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --models "$DEEP_MODELS" \
    --classical-models "$CLASSICAL_MODELS" \
    --no-mlflow \
    --no-optuna \
    --no-temporal-backtest \
    --no-shap \
  )
  if [[ "$BENCHMARK_ENABLE_MLFLOW" == "1" ]]; then
    CMD+=("--mlflow")
  fi
  if [[ "$BENCHMARK_CLASSICAL_ONLY" == "1" ]]; then
    CMD+=("--classical-only")
  fi
  CMD+=("$@")
  "${CMD[@]}"
done

"$PYTHON_CMD" scripts/aggregate_benchmark.py --benchmark-id "$BENCHMARK_ID"
echo "Benchmark lock complete: outputs/history/benchmark_lock_${BENCHMARK_ID}_summary.csv"
