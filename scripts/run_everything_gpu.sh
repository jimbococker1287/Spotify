#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
exec bash scripts/run_everything.sh --resource-profile gpu "$@"
