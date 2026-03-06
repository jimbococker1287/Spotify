#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
else
  RUN_NAME="everything-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

CLASSICAL_ALL="logreg,random_forest,extra_trees,hist_gbm,knn,gaussian_nb,mlp"
DEEP_ALL="dense,lstm,gru,transformer,cnn,tcn,cnn_lstm,attention_rnn,tft,transformer_xl,memory_net,graph_seq,gru_artist,memory_net_artist"
EPOCHS="${EPOCHS:-12}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
OPTUNA_TIMEOUT_SECONDS="${OPTUNA_TIMEOUT_SECONDS:-1800}"
BACKTEST_FOLDS="${BACKTEST_FOLDS:-5}"

export SPOTIFY_FORCE_CPU="${SPOTIFY_FORCE_CPU:-1}"
export SPOTIFY_RUN_EAGER="${SPOTIFY_RUN_EAGER:-1}"

exec "$PYTHON_CMD" -m spotify \
  --profile full \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --models "$DEEP_ALL" \
  --mlflow \
  --optuna \
  --optuna-trials "$OPTUNA_TRIALS" \
  --optuna-timeout-seconds "$OPTUNA_TIMEOUT_SECONDS" \
  --temporal-backtest \
  --backtest-folds "$BACKTEST_FOLDS" \
  --classical-models "$CLASSICAL_ALL" \
  --optuna-models "$CLASSICAL_ALL" \
  --backtest-models "$CLASSICAL_ALL" \
  "$@"
