#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" && -x ".venv-metal/bin/python" ]]; then
  export PYTHON_BIN="$PWD/.venv-metal/bin/python"
fi

# Train every configured deep architecture, but trade throughput for lower RAM.
export SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS="${SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS:-1}"
export SPOTIFY_DEEP_SCREENING="${SPOTIFY_DEEP_SCREENING:-off}"
export DEEP_ALL="${DEEP_ALL:-sasrec,bert4rec,srgnn,dense,gru,transformer,lstm,cnn,tcn,cnn_lstm,attention_rnn,tft,transformer_xl,memory_net,graph_seq,gru_artist,memory_net_artist}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export EPOCHS="${EPOCHS:-10}"

# Keep TensorFlow input memory conservative. This may run much longer.
export SPOTIFY_TF_DATA_CACHE="${SPOTIFY_TF_DATA_CACHE:-off}"
export SPOTIFY_TF_DATA_CACHE_FRACTION="${SPOTIFY_TF_DATA_CACHE_FRACTION:-0.10}"
export SPOTIFY_TF_PREFETCH="${SPOTIFY_TF_PREFETCH:-1}"
export SPOTIFY_TF_DATA_THREADPOOL="${SPOTIFY_TF_DATA_THREADPOOL:-1}"
export SPOTIFY_SHUFFLE_BUFFER="${SPOTIFY_SHUFFLE_BUFFER:-8192}"
export SPOTIFY_STEPS_PER_EXECUTION="${SPOTIFY_STEPS_PER_EXECUTION:-16}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"

# Avoid nested classical/Optuna/backtest parallelism while deep models run.
export SPOTIFY_CLASSICAL_MODEL_WORKERS="${SPOTIFY_CLASSICAL_MODEL_WORKERS:-1}"
export SPOTIFY_MAX_CLASSICAL_WORKERS="${SPOTIFY_MAX_CLASSICAL_WORKERS:-1}"
export SPOTIFY_BACKTEST_WORKERS="${SPOTIFY_BACKTEST_WORKERS:-1}"
export SPOTIFY_OPTUNA_MODEL_WORKERS="${SPOTIFY_OPTUNA_MODEL_WORKERS:-1}"
export SPOTIFY_OPTUNA_JOBS="${SPOTIFY_OPTUNA_JOBS:-1}"
export SPOTIFY_SKLEARN_NJOBS="${SPOTIFY_SKLEARN_NJOBS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# No deep cache or warm-start reuse: this is a true all-14 training pass.
export SPOTIFY_CACHE_DEEP="${SPOTIFY_CACHE_DEEP:-0}"
export SPOTIFY_CACHE_DEEP_REPORTING="${SPOTIFY_CACHE_DEEP_REPORTING:-0}"
export SPOTIFY_WARM_START_DEEP="${SPOTIFY_WARM_START_DEEP:-0}"

if [[ $# -eq 0 || "$1" == --* ]]; then
  set -- "everything-all-deep-low-ram-$(date +%Y%m%d-%H%M%S)" "$@"
fi

RUN_NAME="$1"
PLAN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|--preflight)
      PLAN_ONLY=1
      ;;
  esac
done

set +e
bash scripts/run_everything_balanced_full.sh "$@"
STATUS=$?
set -e

if [[ "$PLAN_ONLY" == "1" ]]; then
  exit "$STATUS"
fi

# If the later full-pipeline stages are interrupted, preserve the all-14 deep
# benchmark as a structured partial artifact instead of leaving only logs.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  "$PYTHON_BIN" -m spotify.deep_benchmark_finalizer --outputs-dir outputs --run-name "$RUN_NAME" --expected-models "$DEEP_ALL" || true
else
  python3 -m spotify.deep_benchmark_finalizer --outputs-dir outputs --run-name "$RUN_NAME" --expected-models "$DEEP_ALL" || true
fi

exit "$STATUS"
