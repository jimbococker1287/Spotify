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

"$PYTHON_CMD" scripts/regression_alert.py
