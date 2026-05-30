#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs

if [[ -x ".venv-metal/bin/python" ]]; then
  VENV_DIR="${VENV_DIR:-.venv-metal}"
elif [[ -x ".venv/bin/python" ]]; then
  VENV_DIR="${VENV_DIR:-.venv}"
else
  VENV_DIR="${VENV_DIR:-}"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "$VENV_DIR" ]]; then
  export PYTHON_BIN="$PWD/$VENV_DIR/bin/python"
else
  export PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$PWD"

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
else
  RUN_NAME="everything-balanced-full-$(date +%Y%m%d-%H%M%S)"
fi
LOG_FILE="outputs/logs/${RUN_NAME}.log"

# Fresh model/search work, with lower-risk data/reporting caches retained.
export SPOTIFY_CACHE_BACKTEST="${SPOTIFY_CACHE_BACKTEST:-0}"
export SPOTIFY_CACHE_CLASSICAL="${SPOTIFY_CACHE_CLASSICAL:-0}"
export SPOTIFY_CACHE_DEEP="${SPOTIFY_CACHE_DEEP:-0}"
export SPOTIFY_CACHE_OPTUNA="${SPOTIFY_CACHE_OPTUNA:-0}"
export SPOTIFY_CACHE_RETRIEVAL="${SPOTIFY_CACHE_RETRIEVAL:-0}"
export SPOTIFY_CACHE_SHAP="${SPOTIFY_CACHE_SHAP:-0}"
export SPOTIFY_CACHE_PREPARED="${SPOTIFY_CACHE_PREPARED:-1}"
export SPOTIFY_CACHE_DEEP_REPORTING="${SPOTIFY_CACHE_DEEP_REPORTING:-1}"
export SPOTIFY_CACHE_ANALYSIS_ARTIFACTS="${SPOTIFY_CACHE_ANALYSIS_ARTIFACTS:-1}"
export SPOTIFY_CACHE_POSTRUN_RESEARCH_ARTIFACTS="${SPOTIFY_CACHE_POSTRUN_RESEARCH_ARTIFACTS:-1}"

# Keep memory lower than the no-skip run without reducing project coverage.
export SPOTIFY_CLASSICAL_MODEL_WORKERS="${SPOTIFY_CLASSICAL_MODEL_WORKERS:-2}"
export SPOTIFY_MAX_CLASSICAL_WORKERS="${SPOTIFY_MAX_CLASSICAL_WORKERS:-2}"
export SPOTIFY_BACKTEST_WORKERS="${SPOTIFY_BACKTEST_WORKERS:-1}"
export SPOTIFY_OPTUNA_MODEL_WORKERS="${SPOTIFY_OPTUNA_MODEL_WORKERS:-1}"
export SPOTIFY_OPTUNA_JOBS="${SPOTIFY_OPTUNA_JOBS:-1}"
export SPOTIFY_SKLEARN_NJOBS="${SPOTIFY_SKLEARN_NJOBS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-3}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export SPOTIFY_TF_DATA_CACHE_FRACTION="${SPOTIFY_TF_DATA_CACHE_FRACTION:-0.25}"

# Full branch/model coverage, but with bounded samples and deep screening.
export SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS="${SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS:-1}"
export SPOTIFY_DEEP_SCREENING="${SPOTIFY_DEEP_SCREENING:-auto}"
export SPOTIFY_DEEP_SCREENING_TOP_N="${SPOTIFY_DEEP_SCREENING_TOP_N:-4}"
export SPOTIFY_DEEP_SCREENING_MIN_MODELS="${SPOTIFY_DEEP_SCREENING_MIN_MODELS:-6}"
export SPOTIFY_DEEP_SCREENING_MAX_TRAIN_ROWS="${SPOTIFY_DEEP_SCREENING_MAX_TRAIN_ROWS:-16000}"
export SPOTIFY_DEEP_SCREENING_MAX_VAL_ROWS="${SPOTIFY_DEEP_SCREENING_MAX_VAL_ROWS:-5000}"

export CLASSICAL_MAX_TRAIN_SAMPLES="${CLASSICAL_MAX_TRAIN_SAMPLES:-45000}"
export CLASSICAL_MAX_EVAL_SAMPLES="${CLASSICAL_MAX_EVAL_SAMPLES:-20000}"
export BACKTEST_MAX_TRAIN_SAMPLES="${BACKTEST_MAX_TRAIN_SAMPLES:-25000}"
export BACKTEST_MAX_EVAL_SAMPLES="${BACKTEST_MAX_EVAL_SAMPLES:-10000}"
export SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES="${SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES:-$BACKTEST_MAX_TRAIN_SAMPLES}"
export SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES="${SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES:-$BACKTEST_MAX_EVAL_SAMPLES}"
export SPOTIFY_STRESS_TEST_MAX_SESSIONS="${SPOTIFY_STRESS_TEST_MAX_SESSIONS:-5000}"
export SPOTIFY_STRESS_TEST_BATCH_SIZE="${SPOTIFY_STRESS_TEST_BATCH_SIZE:-128}"
export SPOTIFY_PRETRAIN_MAX_PAIRS="${SPOTIFY_PRETRAIN_MAX_PAIRS:-750000}"
export SPOTIFY_RETRIEVAL_BACKTEST_PRETRAIN_MAX_PAIRS="${SPOTIFY_RETRIEVAL_BACKTEST_PRETRAIN_MAX_PAIRS:-150000}"

export CLASSICAL_ALL="${CLASSICAL_ALL:-logreg,random_forest,extra_trees,hist_gbm,knn,gaussian_nb,mlp}"
export OPTUNA_MODELS="${OPTUNA_MODELS:-logreg,random_forest,extra_trees,hist_gbm,knn,gaussian_nb,mlp}"
export BACKTEST_MODELS="${BACKTEST_MODELS:-logreg,random_forest,extra_trees,hist_gbm,knn,gaussian_nb,mlp,retrieval_dual_encoder,retrieval_reranker,blended_ensemble}"
export OPTUNA_TRIALS="${OPTUNA_TRIALS:-12}"
export OPTUNA_TIMEOUT_SECONDS="${OPTUNA_TIMEOUT_SECONDS:-1200}"
export SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS="${SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS:-120}"
export EPOCHS="${EPOCHS:-10}"

echo "Using PYTHON_BIN=$PYTHON_BIN"
echo "Run name: $RUN_NAME"
echo "Log file: $LOG_FILE"

bash scripts/run_everything.sh "$RUN_NAME" \
  --retrieval-stack \
  --self-supervised-pretrain \
  --friction-analysis \
  --moonshot-lab \
  "$@" \
  2>&1 | tee "$LOG_FILE"

LATEST_RUN_DIR="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
from spotify.run_artifacts import latest_manifest_run_dir
print(latest_manifest_run_dir(Path("outputs")) or "")
PY
)"

run_make() {
  if [[ -n "$VENV_DIR" ]]; then
    make VENV="$VENV_DIR" "$@"
  else
    make PYTHON="$PYTHON_BIN" "$@"
  fi
}

run_make research-claims
run_make show-ready-backfill
run_make listener-archetypes
run_make quant-decision-lab
run_make creator-market-intelligence
run_make research-platform-lab
run_make analytics-db
run_make scope-expansion-lab
run_make control-room
run_make branch-portfolio
run_make claim-to-demo
run_make outward-package
run_make front-door
run_make day-90-launch
run_make show-ready-maintenance

set +e
"$PYTHON_BIN" scripts/regression_alert.py --run-dir "$LATEST_RUN_DIR" --review-threshold off
ALERT_STATUS=$?

"$PYTHON_BIN" scripts/control_room_guard.py --run-dir "$LATEST_RUN_DIR"
GUARD_STATUS=$?
set -e

echo "Run dir: $LATEST_RUN_DIR"
echo "Log file: $LOG_FILE"
echo "regression_alert_status=$ALERT_STATUS control_room_guard_status=$GUARD_STATUS"

exit $(( ALERT_STATUS != 0 ? ALERT_STATUS : GUARD_STATUS ))
