#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
else
  RUN_NAME="refresh-backtest-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

CLASSICAL_MODELS="${CLASSICAL_MODELS:-logreg,random_forest,extra_trees,hist_gbm,knn,gaussian_nb,mlp}"
BACKTEST_FOLDS="${BACKTEST_FOLDS:-5}"

export SPOTIFY_TF_DATA_CACHE="${SPOTIFY_TF_DATA_CACHE:-off}"
export SPOTIFY_CLASSICAL_MODEL_WORKERS="${SPOTIFY_CLASSICAL_MODEL_WORKERS:-1}"
export SPOTIFY_BACKTEST_WORKERS="${SPOTIFY_BACKTEST_WORKERS:-1}"
export SPOTIFY_OPTUNA_JOBS="${SPOTIFY_OPTUNA_JOBS:-1}"
export SPOTIFY_CHAMPION_GATE_METRIC="${SPOTIFY_CHAMPION_GATE_METRIC:-backtest_top1}"
export SPOTIFY_CHAMPION_GATE_MATCH_PROFILE="${SPOTIFY_CHAMPION_GATE_MATCH_PROFILE:-1}"
export SPOTIFY_DISABLE_MONITOR="${SPOTIFY_DISABLE_MONITOR:-1}"

exec "$PYTHON_CMD" -m spotify \
  --profile full \
  --classical-only \
  --run-name "$RUN_NAME" \
  --classical-models "$CLASSICAL_MODELS" \
  --no-optuna \
  --temporal-backtest \
  --backtest-folds "$BACKTEST_FOLDS" \
  --backtest-models "$CLASSICAL_MODELS" \
  --mlflow \
  --no-shap \
  "$@"
