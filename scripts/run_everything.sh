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
OPTUNA_MODELS="${OPTUNA_MODELS:-logreg,extra_trees,mlp}"
BACKTEST_MODELS="${BACKTEST_MODELS:-logreg,extra_trees,mlp}"
DEEP_ALL="dense,lstm,gru,transformer,cnn,tcn,cnn_lstm,attention_rnn,tft,transformer_xl,memory_net,graph_seq,gru_artist,memory_net_artist"
EPOCHS="${EPOCHS:-12}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-18}"
OPTUNA_TIMEOUT_SECONDS="${OPTUNA_TIMEOUT_SECONDS:-1200}"
BACKTEST_FOLDS="${BACKTEST_FOLDS:-4}"
CLASSICAL_MAX_TRAIN_SAMPLES="${CLASSICAL_MAX_TRAIN_SAMPLES:-50000}"
CLASSICAL_MAX_EVAL_SAMPLES="${CLASSICAL_MAX_EVAL_SAMPLES:-25000}"
BACKTEST_MAX_TRAIN_SAMPLES="${BACKTEST_MAX_TRAIN_SAMPLES:-30000}"
BACKTEST_MAX_EVAL_SAMPLES="${BACKTEST_MAX_EVAL_SAMPLES:-12000}"

export SPOTIFY_FORCE_CPU="${SPOTIFY_FORCE_CPU:-0}"
export SPOTIFY_RUN_EAGER="${SPOTIFY_RUN_EAGER:-0}"
export SPOTIFY_STEPS_PER_EXECUTION="${SPOTIFY_STEPS_PER_EXECUTION:-64}"
export SPOTIFY_BATCH_LOG_INTERVAL="${SPOTIFY_BATCH_LOG_INTERVAL:-100}"
export SPOTIFY_CACHE_PREPARED="${SPOTIFY_CACHE_PREPARED:-1}"
export SPOTIFY_OPTUNA_PRUNER="${SPOTIFY_OPTUNA_PRUNER:-median}"
export SPOTIFY_OPTUNA_PRUNING_FIDELITIES="${SPOTIFY_OPTUNA_PRUNING_FIDELITIES:-0.25,0.60,1.0}"
export SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS="${SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS:-120}"
export SPOTIFY_OPTUNA_MODEL_TIMEOUTS="${SPOTIFY_OPTUNA_MODEL_TIMEOUTS:-logreg=300,random_forest=900,extra_trees=600,hist_gbm=900,knn=180,gaussian_nb=120,mlp=600}"
export SPOTIFY_PRETRAIN_OBJECTIVES="${SPOTIFY_PRETRAIN_OBJECTIVES:-cooccurrence,masked_tail}"
export SPOTIFY_PRETRAIN_MAX_PAIRS="${SPOTIFY_PRETRAIN_MAX_PAIRS:-1000000}"
export SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES="${SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES:-$BACKTEST_MAX_TRAIN_SAMPLES}"
export SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES="${SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES:-$BACKTEST_MAX_EVAL_SAMPLES}"
export SPOTIFY_ROBUSTNESS_GUARDRAIL_SEGMENT="${SPOTIFY_ROBUSTNESS_GUARDRAIL_SEGMENT:-repeat_from_prev}"
export SPOTIFY_ROBUSTNESS_GUARDRAIL_BUCKET="${SPOTIFY_ROBUSTNESS_GUARDRAIL_BUCKET:-new}"
export SPOTIFY_STRESS_BENCHMARK_SCENARIO="${SPOTIFY_STRESS_BENCHMARK_SCENARIO:-evening_drift}"
export SPOTIFY_STRESS_BENCHMARK_POLICY="${SPOTIFY_STRESS_BENCHMARK_POLICY:-safe_global}"
export SPOTIFY_STRESS_BENCHMARK_REFERENCE_POLICY="${SPOTIFY_STRESS_BENCHMARK_REFERENCE_POLICY:-baseline_exploit}"
export SPOTIFY_CONFORMAL_TARGET_SELECTIVE_RISK="${SPOTIFY_CONFORMAL_TARGET_SELECTIVE_RISK:-0.40}"
export SPOTIFY_CONFORMAL_MIN_ACCEPTED_RATE="${SPOTIFY_CONFORMAL_MIN_ACCEPTED_RATE:-0.25}"
export SPOTIFY_CONFORMAL_MIN_RISK_DROP="${SPOTIFY_CONFORMAL_MIN_RISK_DROP:-0.02}"
export SPOTIFY_CHAMPION_GATE_MAX_REGRESSION="${SPOTIFY_CHAMPION_GATE_MAX_REGRESSION:-0.005}"
export SPOTIFY_CHAMPION_GATE_METRIC="${SPOTIFY_CHAMPION_GATE_METRIC:-backtest_top1}"
export SPOTIFY_CHAMPION_GATE_MATCH_PROFILE="${SPOTIFY_CHAMPION_GATE_MATCH_PROFILE:-1}"
export SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK="${SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK:-0.50}"
export SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE="${SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE:-0.30}"
export SPOTIFY_CHAMPION_GATE_STRICT="${SPOTIFY_CHAMPION_GATE_STRICT:-0}"
export SPOTIFY_TF_DATA_CACHE="${SPOTIFY_TF_DATA_CACHE:-auto}"
export SPOTIFY_TF_DATA_CACHE_FRACTION="${SPOTIFY_TF_DATA_CACHE_FRACTION:-0.40}"
export SPOTIFY_TF_PREFETCH="${SPOTIFY_TF_PREFETCH:-auto}"
export SPOTIFY_DISTRIBUTION_STRATEGY="${SPOTIFY_DISTRIBUTION_STRATEGY:-auto}"
export SPOTIFY_MIXED_PRECISION="${SPOTIFY_MIXED_PRECISION:-auto}"
export SPOTIFY_ISOLATE_MPL_CACHE="${SPOTIFY_ISOLATE_MPL_CACHE:-0}"

LOGICAL_CPUS="$("$PYTHON_CMD" - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"

TOTAL_RAM_GB="$("$PYTHON_CMD" - <<'PY'
try:
    import psutil  # type: ignore
    total = int(psutil.virtual_memory().total // (1024 ** 3))
except Exception:
    total = 0
print(total)
PY
)"

if [[ -z "${TF_NUM_INTRAOP_THREADS:-}" ]]; then
  export TF_NUM_INTRAOP_THREADS="$LOGICAL_CPUS"
fi
if [[ -z "${TF_NUM_INTEROP_THREADS:-}" ]]; then
  if (( LOGICAL_CPUS > 8 )); then
    export TF_NUM_INTEROP_THREADS="4"
  elif (( LOGICAL_CPUS > 2 )); then
    export TF_NUM_INTEROP_THREADS="2"
  else
    export TF_NUM_INTEROP_THREADS="1"
  fi
fi

if [[ -z "${SPOTIFY_CLASSICAL_MODEL_WORKERS:-}" ]]; then
  cpu_based_workers=1
  if (( LOGICAL_CPUS >= 12 )); then
    cpu_based_workers=8
  elif (( LOGICAL_CPUS >= 8 )); then
    cpu_based_workers=6
  elif (( LOGICAL_CPUS >= 4 )); then
    cpu_based_workers=2
  fi

  mem_cap_workers=4
  if (( TOTAL_RAM_GB > 0 )); then
    if (( TOTAL_RAM_GB < 12 )); then
      mem_cap_workers=1
    elif (( TOTAL_RAM_GB < 18 )); then
      mem_cap_workers=2
    elif (( TOTAL_RAM_GB < 26 )); then
      mem_cap_workers=3
    else
      mem_cap_workers=4
    fi
  fi

  if (( cpu_based_workers < mem_cap_workers )); then
    export SPOTIFY_CLASSICAL_MODEL_WORKERS="$cpu_based_workers"
  else
    export SPOTIFY_CLASSICAL_MODEL_WORKERS="$mem_cap_workers"
  fi
fi

if [[ -z "${SPOTIFY_BACKTEST_WORKERS:-}" ]]; then
  if (( LOGICAL_CPUS >= 10 )) && (( TOTAL_RAM_GB >= 24 )) && (( SPOTIFY_CLASSICAL_MODEL_WORKERS > 2 )); then
    export SPOTIFY_BACKTEST_WORKERS="3"
  elif (( SPOTIFY_CLASSICAL_MODEL_WORKERS > 2 )); then
    export SPOTIFY_BACKTEST_WORKERS="2"
  else
    export SPOTIFY_BACKTEST_WORKERS="$SPOTIFY_CLASSICAL_MODEL_WORKERS"
  fi
fi

if [[ -z "${SPOTIFY_OPTUNA_JOBS:-}" ]]; then
  if (( SPOTIFY_CLASSICAL_MODEL_WORKERS > 2 )); then
    export SPOTIFY_OPTUNA_JOBS="2"
  else
    export SPOTIFY_OPTUNA_JOBS="$SPOTIFY_CLASSICAL_MODEL_WORKERS"
  fi
fi

if [[ -z "${SPOTIFY_OPTUNA_MODEL_WORKERS:-}" ]]; then
  if (( LOGICAL_CPUS >= 8 )) && (( TOTAL_RAM_GB >= 24 )); then
    export SPOTIFY_OPTUNA_MODEL_WORKERS="2"
  else
    export SPOTIFY_OPTUNA_MODEL_WORKERS="1"
  fi
fi

# Avoid nested BLAS thread oversubscription when many sklearn jobs run in parallel.
if (( SPOTIFY_CLASSICAL_MODEL_WORKERS > 1 )); then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
fi

export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-$LOGICAL_CPUS}"

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
  --classical-max-train-samples "$CLASSICAL_MAX_TRAIN_SAMPLES" \
  --classical-max-eval-samples "$CLASSICAL_MAX_EVAL_SAMPLES" \
  --classical-models "$CLASSICAL_ALL" \
  --optuna-models "$OPTUNA_MODELS" \
  --backtest-models "$BACKTEST_MODELS" \
  "$@"
