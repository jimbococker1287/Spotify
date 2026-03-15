#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
else
  RUN_NAME="experimental-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

EPOCHS="${EPOCHS:-18}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-10}"
BACKTEST_FOLDS="${BACKTEST_FOLDS:-3}"

export SPOTIFY_TF_DATA_CACHE="${SPOTIFY_TF_DATA_CACHE:-off}"
export SPOTIFY_CLASSICAL_MODEL_WORKERS="${SPOTIFY_CLASSICAL_MODEL_WORKERS:-1}"
export SPOTIFY_BACKTEST_WORKERS="${SPOTIFY_BACKTEST_WORKERS:-1}"
export SPOTIFY_OPTUNA_JOBS="${SPOTIFY_OPTUNA_JOBS:-1}"
export SPOTIFY_CHAMPION_GATE_METRIC="${SPOTIFY_CHAMPION_GATE_METRIC:-backtest_top1}"
export SPOTIFY_CHAMPION_GATE_MATCH_PROFILE="${SPOTIFY_CHAMPION_GATE_MATCH_PROFILE:-1}"
export SPOTIFY_CHAMPION_GATE_MAX_REGRESSION="${SPOTIFY_CHAMPION_GATE_MAX_REGRESSION:-0.02}"

exec "$PYTHON_CMD" -m spotify \
  --profile experimental \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --optuna \
  --optuna-trials "$OPTUNA_TRIALS" \
  --temporal-backtest \
  --backtest-folds "$BACKTEST_FOLDS" \
  --mlflow \
  --no-shap \
  "$@"
