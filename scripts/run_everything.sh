#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESOURCE_PROFILE="${SPOTIFY_RESOURCE_PROFILE:-auto}"
DRY_RUN=0
PREFLIGHT=0
pipeline_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-profile)
      if [[ $# -lt 2 ]]; then
        echo "--resource-profile requires auto, cpu, or gpu." >&2
        exit 2
      fi
      RESOURCE_PROFILE="$2"
      shift 2
      ;;
    --resource-profile=*)
      RESOURCE_PROFILE="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --preflight)
      PREFLIGHT=1
      shift
      ;;
    --)
      shift
      pipeline_args+=("$@")
      break
      ;;
    *)
      pipeline_args+=("$1")
      shift
      ;;
  esac
done
if (( ${#pipeline_args[@]} > 0 )); then
  set -- "${pipeline_args[@]}"
else
  set --
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
else
  RUN_NAME="everything-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -x ".venv/bin/python" ]]; then
  PLANNER_PYTHON=".venv/bin/python"
else
  PLANNER_PYTHON="python3"
fi
plan_exports="$(
  PYTHONPATH="$ROOT_DIR" "$PLANNER_PYTHON" -m spotify.resource_planning \
    --root-dir "$ROOT_DIR" \
    --profile "$RESOURCE_PROFILE" \
    --format shell
)"
eval "$plan_exports"
PYTHON_CMD="$PYTHON_BIN"
if [[ "$PREFLIGHT" == "1" || "$DRY_RUN" == "1" ]]; then
  printf '%s\n' "$SPOTIFY_RESOURCE_PLAN_REPORT"
else
  printf '%s\n' "$SPOTIFY_RESOURCE_PLAN_SUMMARY"
fi

CLASSICAL_ALL="${CLASSICAL_ALL:-logreg,extra_trees,knn,gaussian_nb,mlp}"
OPTUNA_MODELS="${OPTUNA_MODELS:-logreg,mlp}"
BACKTEST_MODELS="${BACKTEST_MODELS:-logreg,extra_trees,mlp,retrieval_reranker,blended_ensemble}"
required_modules=()
case ",$CLASSICAL_ALL,$OPTUNA_MODELS,$BACKTEST_MODELS," in
  *,lightgbm,*) required_modules+=(lightgbm) ;;
esac
case ",$CLASSICAL_ALL,$OPTUNA_MODELS,$BACKTEST_MODELS," in
  *,xgboost,*) required_modules+=(xgboost) ;;
esac
if (( ${#required_modules[@]} > 0 )); then
  "$PYTHON_CMD" - "${required_modules[@]}" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    names = ", ".join(missing)
    raise SystemExit(
        f"Missing selected model dependencies in {sys.executable}: {names}. "
        f"Run '{sys.executable} -m pip install -e .' before starting the pipeline."
    )
PY
fi
DEEP_CORE_DEFAULT="dense,gru,transformer"
DEEP_RESEARCH_DEFAULT="sasrec,bert4rec,srgnn,lstm,cnn,tcn,cnn_lstm,attention_rnn,tft,transformer_xl,memory_net,graph_seq,gru_artist,memory_net_artist"
if [[ -z "${DEEP_ALL:-}" ]]; then
  if [[ "${SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS:-0}" == "1" || "${SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS:-0}" == "true" || "${SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS:-0}" == "yes" || "${SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS:-0}" == "on" ]]; then
    DEEP_ALL="${DEEP_CORE_DEFAULT},${DEEP_RESEARCH_DEFAULT}"
  else
    DEEP_ALL="${DEEP_CORE_DEFAULT}"
  fi
fi
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-10}"
OPTUNA_TIMEOUT_SECONDS="${OPTUNA_TIMEOUT_SECONDS:-600}"
BACKTEST_FOLDS="${BACKTEST_FOLDS:-4}"
CLASSICAL_MAX_TRAIN_SAMPLES="${CLASSICAL_MAX_TRAIN_SAMPLES:-50000}"
CLASSICAL_MAX_EVAL_SAMPLES="${CLASSICAL_MAX_EVAL_SAMPLES:-25000}"
BACKTEST_MAX_TRAIN_SAMPLES="${BACKTEST_MAX_TRAIN_SAMPLES:-30000}"
BACKTEST_MAX_EVAL_SAMPLES="${BACKTEST_MAX_EVAL_SAMPLES:-12000}"

export SPOTIFY_RUN_EAGER="${SPOTIFY_RUN_EAGER:-0}"
export SPOTIFY_BATCH_LOG_INTERVAL="${SPOTIFY_BATCH_LOG_INTERVAL:-100}"
export SPOTIFY_WARM_START_DEEP="${SPOTIFY_WARM_START_DEEP:-1}"
export SPOTIFY_WARM_START_OPTUNA="${SPOTIFY_WARM_START_OPTUNA:-1}"
export SPOTIFY_DEEP_SCREENING="${SPOTIFY_DEEP_SCREENING:-auto}"
export SPOTIFY_DEEP_SCREENING_TOP_N="${SPOTIFY_DEEP_SCREENING_TOP_N:-3}"
export SPOTIFY_DEEP_SCREENING_EPOCHS="${SPOTIFY_DEEP_SCREENING_EPOCHS:-1}"
export SPOTIFY_DEEP_SCREENING_MIN_MODELS="${SPOTIFY_DEEP_SCREENING_MIN_MODELS:-5}"
export SPOTIFY_OPTUNA_WARM_START_TRIAL_FRACTION="${SPOTIFY_OPTUNA_WARM_START_TRIAL_FRACTION:-0.60}"
export SPOTIFY_OPTUNA_WARM_START_MIN_TRIALS="${SPOTIFY_OPTUNA_WARM_START_MIN_TRIALS:-4}"
export SPOTIFY_OPTUNA_PRUNER="${SPOTIFY_OPTUNA_PRUNER:-median}"
export SPOTIFY_OPTUNA_PRUNING_FIDELITIES="${SPOTIFY_OPTUNA_PRUNING_FIDELITIES:-0.25,0.60,1.0}"
export SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS="${SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS:-120}"
export SPOTIFY_OPTUNA_MODEL_TIMEOUTS="${SPOTIFY_OPTUNA_MODEL_TIMEOUTS:-logreg=300,random_forest=900,extra_trees=600,hist_gbm=900,knn=180,gaussian_nb=120,mlp=600}"
export SPOTIFY_OPTUNA_SHORTLIST_TOP_N="${SPOTIFY_OPTUNA_SHORTLIST_TOP_N:-2}"
export SPOTIFY_PRETRAIN_OBJECTIVES="${SPOTIFY_PRETRAIN_OBJECTIVES:-cooccurrence,masked_tail}"
export SPOTIFY_PRETRAIN_MAX_PAIRS="${SPOTIFY_PRETRAIN_MAX_PAIRS:-1000000}"
export SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES="${SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES:-$BACKTEST_MAX_TRAIN_SAMPLES}"
export SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES="${SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES:-$BACKTEST_MAX_EVAL_SAMPLES}"
export SPOTIFY_BACKTEST_SHORTLIST_TOP_N="${SPOTIFY_BACKTEST_SHORTLIST_TOP_N:-2}"
export SPOTIFY_ROBUSTNESS_GUARDRAIL_SEGMENT="${SPOTIFY_ROBUSTNESS_GUARDRAIL_SEGMENT:-repeat_from_prev}"
export SPOTIFY_ROBUSTNESS_GUARDRAIL_BUCKET="${SPOTIFY_ROBUSTNESS_GUARDRAIL_BUCKET:-new}"
export SPOTIFY_STRESS_BENCHMARK_SCENARIO="${SPOTIFY_STRESS_BENCHMARK_SCENARIO:-evening_drift}"
export SPOTIFY_STRESS_BENCHMARK_POLICY="${SPOTIFY_STRESS_BENCHMARK_POLICY:-safe_routed}"
export SPOTIFY_STRESS_BENCHMARK_REFERENCE_POLICY="${SPOTIFY_STRESS_BENCHMARK_REFERENCE_POLICY:-baseline_exploit}"
export SPOTIFY_CONFORMAL_TARGET_SELECTIVE_RISK="${SPOTIFY_CONFORMAL_TARGET_SELECTIVE_RISK:-0.40}"
export SPOTIFY_CONFORMAL_MIN_ACCEPTED_RATE="${SPOTIFY_CONFORMAL_MIN_ACCEPTED_RATE:-0.25}"
export SPOTIFY_CONFORMAL_MIN_RISK_DROP="${SPOTIFY_CONFORMAL_MIN_RISK_DROP:-0.02}"
export SPOTIFY_CLASSICAL_CONFORMAL_TARGET_SELECTIVE_RISK="${SPOTIFY_CLASSICAL_CONFORMAL_TARGET_SELECTIVE_RISK:-0.45}"
export SPOTIFY_CLASSICAL_CONFORMAL_MIN_ACCEPTED_RATE="${SPOTIFY_CLASSICAL_CONFORMAL_MIN_ACCEPTED_RATE:-0.70}"
export SPOTIFY_CLASSICAL_CONFORMAL_MIN_RISK_DROP="${SPOTIFY_CLASSICAL_CONFORMAL_MIN_RISK_DROP:-0.02}"
export SPOTIFY_CHAMPION_GATE_MAX_REGRESSION="${SPOTIFY_CHAMPION_GATE_MAX_REGRESSION:-0.005}"
export SPOTIFY_CHAMPION_GATE_METRIC="${SPOTIFY_CHAMPION_GATE_METRIC:-backtest_top1}"
export SPOTIFY_CHAMPION_GATE_MATCH_PROFILE="${SPOTIFY_CHAMPION_GATE_MATCH_PROFILE:-1}"
export SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK="${SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK:-0.50}"
export SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE="${SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE:-0.30}"
export SPOTIFY_CHAMPION_GATE_STRICT="${SPOTIFY_CHAMPION_GATE_STRICT:-0}"
export SPOTIFY_DISTRIBUTION_STRATEGY="${SPOTIFY_DISTRIBUTION_STRATEGY:-auto}"
export SPOTIFY_FAIL_FAST_PY313_DEEP="${SPOTIFY_FAIL_FAST_PY313_DEEP:-1}"

enable_shap_raw="$(printf '%s' "${SPOTIFY_ENABLE_SHAP:-0}" | tr '[:upper:]' '[:lower:]')"
shap_flag_present=0
for arg in "$@"; do
  case "$arg" in
    --no-shap|--shap)
      shap_flag_present=1
      ;;
  esac
done
if [[ "$SPOTIFY_RESOURCE_PROFILE_RESOLVED" == "gpu" && "$shap_flag_present" == "0" ]]; then
  case "$enable_shap_raw" in
    1|true|yes|on) ;;
    *) set -- "$@" --no-shap ;;
  esac
fi

command=(
  "$PYTHON_CMD" -m spotify
  --profile full
  --run-name "$RUN_NAME"
  --epochs "$EPOCHS"
)
if [[ -n "$BATCH_SIZE" ]]; then
  command+=(--batch "$BATCH_SIZE")
fi
command+=(
  --models "$DEEP_ALL"
  --mlflow
  --optuna
  --optuna-trials "$OPTUNA_TRIALS"
  --optuna-timeout-seconds "$OPTUNA_TIMEOUT_SECONDS"
  --temporal-backtest
  --backtest-folds "$BACKTEST_FOLDS"
  --classical-max-train-samples "$CLASSICAL_MAX_TRAIN_SAMPLES"
  --classical-max-eval-samples "$CLASSICAL_MAX_EVAL_SAMPLES"
  --classical-models "$CLASSICAL_ALL"
  --optuna-models "$OPTUNA_MODELS"
  --backtest-models "$BACKTEST_MODELS"
  "$@"
)

if [[ "$PREFLIGHT" == "1" ]]; then
  exit 0
fi
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
