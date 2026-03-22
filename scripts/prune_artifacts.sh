#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

MODE="${SPOTIFY_ARTIFACT_CLEANUP:-light}"
MIN_SIZE_MB="${SPOTIFY_ARTIFACT_CLEANUP_MIN_MB:-100}"
KEEP_FULL_RUNS="${SPOTIFY_KEEP_FULL_RUNS:-2}"

PYTHONPATH=. "$PYTHON_BIN" scripts/prune_artifacts.py --mode "$MODE" --min-size-mb "$MIN_SIZE_MB" --keep-full-runs "$KEEP_FULL_RUNS" "$@"
