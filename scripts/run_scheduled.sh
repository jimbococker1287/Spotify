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

ALERT_EXIT=0
GUARD_EXIT=0

"$PYTHON_CMD" scripts/regression_alert.py --review-threshold off || ALERT_EXIT=$?
"$PYTHON_CMD" scripts/control_room_guard.py \
  --max-robustness-gap "${SPOTIFY_CONTROL_ROOM_MAX_ROBUSTNESS_GAP:-0.35}" \
  --max-stress-skip-risk "${SPOTIFY_CONTROL_ROOM_MAX_STRESS_SKIP_RISK:-0.45}" \
  --max-target-drift-jsd "${SPOTIFY_CONTROL_ROOM_MAX_TARGET_DRIFT_JSD:-0.20}" \
  --max-selective-risk "${SPOTIFY_CONTROL_ROOM_MAX_SELECTIVE_RISK:-0.50}" \
  || GUARD_EXIT=$?

if [[ "$ALERT_EXIT" -ne 0 ]]; then
  exit "$ALERT_EXIT"
fi

if [[ "$GUARD_EXIT" -ne 0 ]]; then
  exit "$GUARD_EXIT"
fi
