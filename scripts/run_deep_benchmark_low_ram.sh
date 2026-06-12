#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs outputs/history/.mplconfig outputs/history/.cache

if [[ -z "${PYTHON_BIN:-}" && -x ".venv-metal/bin/python" ]]; then
  export PYTHON_BIN="$PWD/.venv-metal/bin/python"
elif [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  export PYTHON_BIN="$PWD/.venv/bin/python"
elif [[ -z "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$PWD"

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
else
  RUN_NAME="deep-benchmark-low-ram-$(date +%Y%m%d-%H%M%S)"
fi
LOG_FILE="outputs/logs/${RUN_NAME}.log"

DEEP_ALL="${DEEP_ALL:-sasrec,bert4rec,srgnn,dense,gru,transformer,lstm,cnn,tcn,cnn_lstm,attention_rnn,tft,transformer_xl,memory_net,graph_seq,gru_artist,memory_net_artist}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXPECTED_MODELS="$DEEP_ALL"

EXTRA_ARGS=("$@")
for ((idx = 0; idx < ${#EXTRA_ARGS[@]}; idx++)); do
  case "${EXTRA_ARGS[$idx]}" in
    --models)
      if (( idx + 1 < ${#EXTRA_ARGS[@]} )); then
        EXPECTED_MODELS="${EXTRA_ARGS[$((idx + 1))]}"
      fi
      ;;
    --models=*)
      EXPECTED_MODELS="${EXTRA_ARGS[$idx]#--models=}"
      ;;
  esac
done

# This launcher is intentionally deep-only: it turns the expensive all-14 model
# training pass into a bounded benchmark instead of a full ops/control-room run.
export SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS="${SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS:-1}"
export SPOTIFY_FULL_DEEP_MODE_POLICY="${SPOTIFY_FULL_DEEP_MODE_POLICY:-on}"
export SPOTIFY_DEEP_SCREENING="${SPOTIFY_DEEP_SCREENING:-off}"
export SPOTIFY_CACHE_DEEP="${SPOTIFY_CACHE_DEEP:-0}"
export SPOTIFY_CACHE_DEEP_REPORTING="${SPOTIFY_CACHE_DEEP_REPORTING:-0}"
export SPOTIFY_WARM_START_DEEP="${SPOTIFY_WARM_START_DEEP:-0}"
export SPOTIFY_DISABLE_MONITOR="${SPOTIFY_DISABLE_MONITOR:-1}"

# Lower peak RAM by preferring small queues, no dataset cache, and limited
# TensorFlow/BLAS parallelism. This trades wall-clock speed for stability.
export SPOTIFY_TF_DATA_CACHE="${SPOTIFY_TF_DATA_CACHE:-off}"
export SPOTIFY_TF_DATA_CACHE_FRACTION="${SPOTIFY_TF_DATA_CACHE_FRACTION:-0.10}"
export SPOTIFY_TF_PREFETCH="${SPOTIFY_TF_PREFETCH:-1}"
export SPOTIFY_TF_DATA_THREADPOOL="${SPOTIFY_TF_DATA_THREADPOOL:-1}"
export SPOTIFY_SHUFFLE_BUFFER="${SPOTIFY_SHUFFLE_BUFFER:-8192}"
export SPOTIFY_STEPS_PER_EXECUTION="${SPOTIFY_STEPS_PER_EXECUTION:-16}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/outputs/history/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/outputs/history/.cache}"

echo "Using PYTHON_BIN=$PYTHON_BIN"
echo "Run name: $RUN_NAME"
echo "Log file: $LOG_FILE"
echo "Deep models: $DEEP_ALL"
echo "Expected benchmark models: $EXPECTED_MODELS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"

CMD=(
  "$PYTHON_BIN" -m spotify
  --profile full
  --run-name "$RUN_NAME"
  --models "$DEEP_ALL"
  --epochs "$EPOCHS"
  --batch "$BATCH_SIZE"
  --no-classical-models
  --no-optuna
  --no-retrieval-stack
  --no-self-supervised-pretrain
  --no-temporal-backtest
  --no-mlflow
  --no-shap
  --no-friction-analysis
  --no-moonshot-lab
)
CMD+=("$@")

set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}

"$PYTHON_BIN" -m spotify.deep_benchmark_finalizer \
  --outputs-dir outputs \
  --run-name "$RUN_NAME" \
  --expected-models "$EXPECTED_MODELS"
FINALIZER_STATUS=$?
set -e

if [[ "$RUN_STATUS" -ne 0 ]]; then
  exit "$RUN_STATUS"
fi
exit "$FINALIZER_STATUS"
