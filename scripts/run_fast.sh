#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
else
  RUN_NAME="fast-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

DEEP_MODELS="${DEEP_MODELS:-dense,gru_artist,lstm}"
CLASSICAL_MODELS="${CLASSICAL_MODELS:-logreg,extra_trees,mlp}"
EPOCHS="${EPOCHS:-4}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-8}"
BACKTEST_FOLDS="${BACKTEST_FOLDS:-2}"

export SPOTIFY_TF_DATA_CACHE="${SPOTIFY_TF_DATA_CACHE:-off}"
export SPOTIFY_CLASSICAL_MODEL_WORKERS="${SPOTIFY_CLASSICAL_MODEL_WORKERS:-1}"
export SPOTIFY_BACKTEST_WORKERS="${SPOTIFY_BACKTEST_WORKERS:-1}"
export SPOTIFY_OPTUNA_JOBS="${SPOTIFY_OPTUNA_JOBS:-1}"
export SPOTIFY_CHAMPION_GATE_METRIC="${SPOTIFY_CHAMPION_GATE_METRIC:-backtest_top1}"
export SPOTIFY_CHAMPION_GATE_MAX_REGRESSION="${SPOTIFY_CHAMPION_GATE_MAX_REGRESSION:-0.01}"
export SPOTIFY_DISABLE_MONITOR="${SPOTIFY_DISABLE_MONITOR:-1}"

exec "$PYTHON_CMD" -m spotify \
  --profile fast \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --models "$DEEP_MODELS" \
  --classical-models "$CLASSICAL_MODELS" \
  --optuna \
  --optuna-trials "$OPTUNA_TRIALS" \
  --optuna-models "$CLASSICAL_MODELS" \
  --temporal-backtest \
  --backtest-folds "$BACKTEST_FOLDS" \
  --backtest-models "$CLASSICAL_MODELS" \
  --mlflow \
  --no-shap \
  "$@"
