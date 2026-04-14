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

_python_is_ge_313() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 13) else 1)
PY
}

_python_has_tensorflow() {
  "$1" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)
PY
}

_resolve_python_cmd() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD="$PYTHON_BIN"
    return
  fi
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_CMD=".venv/bin/python"
  else
    PYTHON_CMD="python3"
  fi

  case "$(printf '%s' "${SPOTIFY_AUTO_ROUTE_TF_PYTHON:-auto}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      return
      ;;
  esac
  if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    return
  fi
  if [[ ! -x ".venv-metal/bin/python" ]]; then
    return
  fi
  if ! _python_is_ge_313 "$PYTHON_CMD"; then
    return
  fi
  if ! _python_has_tensorflow ".venv-metal/bin/python"; then
    return
  fi

  PYTHON_CMD=".venv-metal/bin/python"
  export SPOTIFY_TF_COMPAT_VENV_ROUTED="${SPOTIFY_TF_COMPAT_VENV_ROUTED:-1}"
  echo "Auto-routing fast deep runtime to ${PYTHON_CMD} to avoid Apple Silicon Python 3.13 TensorFlow instability." >&2
}

_resolve_python_cmd

DEEP_MODELS="${DEEP_MODELS:-dense,gru_artist,lstm}"
CLASSICAL_MODELS="${CLASSICAL_MODELS:-logreg,extra_trees,mlp}"
EPOCHS="${EPOCHS:-4}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-8}"
BACKTEST_FOLDS="${BACKTEST_FOLDS:-2}"

export SPOTIFY_TF_DATA_CACHE="${SPOTIFY_TF_DATA_CACHE:-off}"
export SPOTIFY_CACHE_BACKTEST="${SPOTIFY_CACHE_BACKTEST:-1}"
export SPOTIFY_CACHE_CLASSICAL="${SPOTIFY_CACHE_CLASSICAL:-1}"
export SPOTIFY_CACHE_DEEP="${SPOTIFY_CACHE_DEEP:-1}"
export SPOTIFY_CACHE_DEEP_REPORTING="${SPOTIFY_CACHE_DEEP_REPORTING:-1}"
export SPOTIFY_CACHE_RETRIEVAL="${SPOTIFY_CACHE_RETRIEVAL:-1}"
export SPOTIFY_CACHE_SHAP="${SPOTIFY_CACHE_SHAP:-1}"
export SPOTIFY_WARM_START_DEEP="${SPOTIFY_WARM_START_DEEP:-1}"
export SPOTIFY_WARM_START_OPTUNA="${SPOTIFY_WARM_START_OPTUNA:-1}"
export SPOTIFY_DEEP_SCREENING="${SPOTIFY_DEEP_SCREENING:-auto}"
export SPOTIFY_DEEP_SCREENING_TOP_N="${SPOTIFY_DEEP_SCREENING_TOP_N:-2}"
export SPOTIFY_DEEP_SCREENING_EPOCHS="${SPOTIFY_DEEP_SCREENING_EPOCHS:-1}"
export SPOTIFY_DEEP_SCREENING_MIN_MODELS="${SPOTIFY_DEEP_SCREENING_MIN_MODELS:-4}"
export SPOTIFY_OPTUNA_WARM_START_TRIAL_FRACTION="${SPOTIFY_OPTUNA_WARM_START_TRIAL_FRACTION:-0.60}"
export SPOTIFY_OPTUNA_WARM_START_MIN_TRIALS="${SPOTIFY_OPTUNA_WARM_START_MIN_TRIALS:-4}"
export SPOTIFY_CLASSICAL_MODEL_WORKERS="${SPOTIFY_CLASSICAL_MODEL_WORKERS:-1}"
export SPOTIFY_BACKTEST_WORKERS="${SPOTIFY_BACKTEST_WORKERS:-1}"
export SPOTIFY_OPTUNA_JOBS="${SPOTIFY_OPTUNA_JOBS:-1}"
export SPOTIFY_ROBUSTNESS_GUARDRAIL_SEGMENT="${SPOTIFY_ROBUSTNESS_GUARDRAIL_SEGMENT:-repeat_from_prev}"
export SPOTIFY_ROBUSTNESS_GUARDRAIL_BUCKET="${SPOTIFY_ROBUSTNESS_GUARDRAIL_BUCKET:-new}"
export SPOTIFY_STRESS_BENCHMARK_SCENARIO="${SPOTIFY_STRESS_BENCHMARK_SCENARIO:-evening_drift}"
export SPOTIFY_STRESS_BENCHMARK_POLICY="${SPOTIFY_STRESS_BENCHMARK_POLICY:-safe_global}"
export SPOTIFY_STRESS_BENCHMARK_REFERENCE_POLICY="${SPOTIFY_STRESS_BENCHMARK_REFERENCE_POLICY:-baseline_exploit}"
export SPOTIFY_CONFORMAL_TARGET_SELECTIVE_RISK="${SPOTIFY_CONFORMAL_TARGET_SELECTIVE_RISK:-0.45}"
export SPOTIFY_CONFORMAL_MIN_ACCEPTED_RATE="${SPOTIFY_CONFORMAL_MIN_ACCEPTED_RATE:-0.20}"
export SPOTIFY_CONFORMAL_MIN_RISK_DROP="${SPOTIFY_CONFORMAL_MIN_RISK_DROP:-0.02}"
export SPOTIFY_CLASSICAL_CONFORMAL_TARGET_SELECTIVE_RISK="${SPOTIFY_CLASSICAL_CONFORMAL_TARGET_SELECTIVE_RISK:-0.45}"
export SPOTIFY_CLASSICAL_CONFORMAL_MIN_ACCEPTED_RATE="${SPOTIFY_CLASSICAL_CONFORMAL_MIN_ACCEPTED_RATE:-0.70}"
export SPOTIFY_CLASSICAL_CONFORMAL_MIN_RISK_DROP="${SPOTIFY_CLASSICAL_CONFORMAL_MIN_RISK_DROP:-0.02}"
export SPOTIFY_CHAMPION_GATE_METRIC="${SPOTIFY_CHAMPION_GATE_METRIC:-backtest_top1}"
export SPOTIFY_CHAMPION_GATE_MATCH_PROFILE="${SPOTIFY_CHAMPION_GATE_MATCH_PROFILE:-1}"
export SPOTIFY_CHAMPION_GATE_MAX_REGRESSION="${SPOTIFY_CHAMPION_GATE_MAX_REGRESSION:-0.01}"
export SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK="${SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK:-0.55}"
export SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE="${SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE:-0.35}"
export SPOTIFY_DISABLE_MONITOR="${SPOTIFY_DISABLE_MONITOR:-1}"
export SPOTIFY_FAIL_FAST_PY313_DEEP="${SPOTIFY_FAIL_FAST_PY313_DEEP:-1}"

FAST_SAFE_MODE="${SPOTIFY_FAST_SAFE_MODE:-auto}"
FAST_PROGRESS_TIMEOUT_SECONDS="${SPOTIFY_FAST_PROGRESS_TIMEOUT_SECONDS:-75}"
FAST_PROGRESS_POLL_SECONDS="${SPOTIFY_FAST_PROGRESS_POLL_SECONDS:-5}"
FAST_PROGRESS_PATTERN='Epoch progress: step=|Epoch [0-9]+ done|Training classical model|Retrieval baseline|Temporal backtesting|Pipeline completed successfully'
SAFE_MODE_ACTIVE=0

_fast_python_is_313_or_newer() {
  "$PYTHON_CMD" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)'
}

_fast_should_enable_safe_mode() {
  local raw
  raw="$(printf '%s' "$FAST_SAFE_MODE" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    1|true|yes|on)
      return 0
      ;;
    0|false|no|off)
      return 1
      ;;
    auto)
      _fast_python_is_313_or_newer
      return $?
      ;;
    *)
      return 1
      ;;
  esac
}

_apply_safe_mode_defaults() {
  [[ -n "${SPOTIFY_RUN_EAGER:-}" ]] || export SPOTIFY_RUN_EAGER=1
  [[ -n "${SPOTIFY_STEPS_PER_EXECUTION:-}" ]] || export SPOTIFY_STEPS_PER_EXECUTION=1
  [[ -n "${SPOTIFY_TF_PREFETCH:-}" ]] || export SPOTIFY_TF_PREFETCH=1
  SAFE_MODE_ACTIVE=1
  echo "Fast safe mode enabled: run_eagerly=${SPOTIFY_RUN_EAGER} steps_per_execution=${SPOTIFY_STEPS_PER_EXECUTION} prefetch=${SPOTIFY_TF_PREFETCH}" >&2
}

_find_latest_matching_run_dir() {
  "$PYTHON_CMD" - "$ROOT_DIR" "$RUN_NAME" <<'PY'
from pathlib import Path
import sys

root_dir = Path(sys.argv[1])
run_name = sys.argv[2].strip()
runs_dir = root_dir / "outputs" / "runs"
if not runs_dir.exists():
    print("")
    raise SystemExit(0)

candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
if run_name:
    suffix = f"_{run_name}"
    filtered = [path for path in candidates if path.name.endswith(suffix)]
    if filtered:
        candidates = filtered
if not candidates:
    print("")
    raise SystemExit(0)
latest = max(candidates, key=lambda path: (path.name, path.stat().st_mtime))
print(str(latest.resolve()))
PY
}

_wait_for_initial_progress() {
  local pid="$1"
  local previous_run_dir="${2:-}"
  local deadline=$((SECONDS + FAST_PROGRESS_TIMEOUT_SECONDS))
  local run_dir=""
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -z "$run_dir" ]]; then
      local candidate_run_dir
      candidate_run_dir="$(_find_latest_matching_run_dir)"
      if [[ -n "$candidate_run_dir" ]]; then
        if [[ -z "$previous_run_dir" || "$candidate_run_dir" != "$previous_run_dir" ]]; then
          run_dir="$candidate_run_dir"
        fi
      fi
    fi
    if [[ -n "$run_dir" && -f "$run_dir/train.log" ]]; then
      if grep -Eq "$FAST_PROGRESS_PATTERN" "$run_dir/train.log"; then
        return 0
      fi
    fi
    if (( SECONDS >= deadline )); then
      echo "Fast run stalled before initial progress marker; run_dir=${run_dir:-unknown}" >&2
      return 124
    fi
    sleep "$FAST_PROGRESS_POLL_SECONDS"
  done
  wait "$pid"
  return $?
}

_run_fast_once() {
  local execution_mode="${1:-standard}"
  shift || true
  local previous_run_dir
  previous_run_dir="$(_find_latest_matching_run_dir)"
  local -a cmd=(
    "$PYTHON_CMD" -m spotify
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
  )
  if [[ "$execution_mode" == "classical_only" ]]; then
    cmd+=(--classical-only)
  fi
  cmd+=("$@")
  "${cmd[@]}" &
  local pid=$!
  if _wait_for_initial_progress "$pid" "$previous_run_dir"; then
    wait "$pid"
    return $?
  else
    local status=$?
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    return "$status"
  fi
}

if _fast_should_enable_safe_mode; then
  _apply_safe_mode_defaults
fi

if _run_fast_once standard "$@"; then
  exit 0
else
  status=$?
  if [[ "$status" -eq 124 && "$SAFE_MODE_ACTIVE" -eq 0 ]]; then
    echo "Retrying fast lane in safe mode after a startup stall." >&2
    _apply_safe_mode_defaults
    if _run_fast_once standard "$@"; then
      exit 0
    fi
    status=$?
  fi
  if [[ "$status" -eq 124 ]]; then
    echo "Deep startup remained stalled in safe mode; falling back to a classical-only fast refresh." >&2
    if _run_fast_once classical_only "$@"; then
      exit 0
    fi
    status=$?
  fi
  exit "$status"
fi
