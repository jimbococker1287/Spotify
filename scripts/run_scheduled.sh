#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-fast}"
shift || true

if [[ "$MODE" == "fast" ]]; then
  bash scripts/run_fast.sh "$@"
elif [[ "$MODE" == "full" ]]; then
  bash scripts/run_full.sh "$@"
else
  echo "Unknown mode: $MODE (expected fast|full)" >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

if [[ "$MODE" == "fast" ]]; then
  CONTROL_ROOM_MAX_ROBUSTNESS_GAP="${SPOTIFY_CONTROL_ROOM_MAX_ROBUSTNESS_GAP_FAST:-0.40}"
  CONTROL_ROOM_MAX_REPEAT_FROM_PREV_NEW_GAP="${SPOTIFY_CONTROL_ROOM_MAX_REPEAT_FROM_PREV_NEW_GAP_FAST:-0.45}"
  CONTROL_ROOM_MAX_STRESS_SKIP_RISK="${SPOTIFY_CONTROL_ROOM_MAX_STRESS_SKIP_RISK_FAST:-off}"
  CONTROL_ROOM_MAX_STRESS_BENCHMARK_SKIP_RISK="${SPOTIFY_CONTROL_ROOM_MAX_STRESS_BENCHMARK_SKIP_RISK_FAST:-off}"
  CONTROL_ROOM_MAX_TARGET_DRIFT_JSD="${SPOTIFY_CONTROL_ROOM_MAX_TARGET_DRIFT_JSD_FAST:-0.25}"
  CONTROL_ROOM_MAX_SELECTIVE_RISK="${SPOTIFY_CONTROL_ROOM_MAX_SELECTIVE_RISK_FAST:-off}"
else
  CONTROL_ROOM_MAX_ROBUSTNESS_GAP="${SPOTIFY_CONTROL_ROOM_MAX_ROBUSTNESS_GAP:-0.35}"
  CONTROL_ROOM_MAX_REPEAT_FROM_PREV_NEW_GAP="${SPOTIFY_CONTROL_ROOM_MAX_REPEAT_FROM_PREV_NEW_GAP:-0.35}"
  CONTROL_ROOM_MAX_STRESS_SKIP_RISK="${SPOTIFY_CONTROL_ROOM_MAX_STRESS_SKIP_RISK:-0.45}"
  CONTROL_ROOM_MAX_STRESS_BENCHMARK_SKIP_RISK="${SPOTIFY_CONTROL_ROOM_MAX_STRESS_BENCHMARK_SKIP_RISK:-0.45}"
  CONTROL_ROOM_MAX_TARGET_DRIFT_JSD="${SPOTIFY_CONTROL_ROOM_MAX_TARGET_DRIFT_JSD:-0.20}"
  CONTROL_ROOM_MAX_SELECTIVE_RISK="${SPOTIFY_CONTROL_ROOM_MAX_SELECTIVE_RISK:-0.50}"
fi

ALERT_EXIT=0
GUARD_EXIT=0
LATEST_RUN_DIR="$("$PYTHON_CMD" -c 'from pathlib import Path; from spotify.run_artifacts import latest_manifest_run_dir; path = latest_manifest_run_dir(Path("outputs")); print("" if path is None else path)' || true)"
ALERT_ARGS=()

if [[ -z "$LATEST_RUN_DIR" || ! -d "$LATEST_RUN_DIR" ]]; then
  echo "Unable to resolve the latest manifest-backed run directory under outputs/runs" >&2
  exit 1
fi

if [[ "$MODE" == "fast" ]]; then
  ALERT_ARGS+=(--allow-fail)
fi

"$PYTHON_CMD" scripts/regression_alert.py --run-dir "$LATEST_RUN_DIR" --review-threshold off "${ALERT_ARGS[@]}" || ALERT_EXIT=$?
"$PYTHON_CMD" scripts/control_room_guard.py \
  --run-dir "$LATEST_RUN_DIR" \
  --max-robustness-gap "$CONTROL_ROOM_MAX_ROBUSTNESS_GAP" \
  --max-repeat-from-prev-new-gap "$CONTROL_ROOM_MAX_REPEAT_FROM_PREV_NEW_GAP" \
  --max-stress-skip-risk "$CONTROL_ROOM_MAX_STRESS_SKIP_RISK" \
  --max-stress-benchmark-skip-risk "$CONTROL_ROOM_MAX_STRESS_BENCHMARK_SKIP_RISK" \
  --max-target-drift-jsd "$CONTROL_ROOM_MAX_TARGET_DRIFT_JSD" \
  --max-selective-risk "$CONTROL_ROOM_MAX_SELECTIVE_RISK" \
  || GUARD_EXIT=$?

if [[ "$ALERT_EXIT" -ne 0 ]]; then
  exit "$ALERT_EXIT"
fi

if [[ "$GUARD_EXIT" -ne 0 ]]; then
  exit "$GUARD_EXIT"
fi
