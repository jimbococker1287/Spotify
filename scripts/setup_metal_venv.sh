#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON311_BIN="${PYTHON311_BIN:-/opt/homebrew/opt/python@3.11/bin/python3.11}"
VENV_DIR="${VENV_DIR:-.venv-metal}"
USER_PLUGIN_DIR="$HOME/Library/Python/3.11/lib/python/site-packages/tensorflow-plugins"

if [[ ! -x "$PYTHON311_BIN" ]]; then
  echo "Python 3.11 interpreter not found at $PYTHON311_BIN" >&2
  echo "Set PYTHON311_BIN to a Python 3.11 executable and retry." >&2
  exit 1
fi

if [[ -d "$USER_PLUGIN_DIR" ]]; then
  backup_path="${USER_PLUGIN_DIR}.backup-$(date +%Y%m%d-%H%M%S)"
  mv "$USER_PLUGIN_DIR" "$backup_path"
  echo "Backed up conflicting user-site tensorflow plugin dir to: $backup_path"
fi

"$PYTHON311_BIN" -m venv "$VENV_DIR"

PYTHONNOUSERSITE=1 "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

tmp_requirements="$(mktemp)"
trap 'rm -f "$tmp_requirements"' EXIT
awk '$1 !~ /^tensorflow/' requirements.txt > "$tmp_requirements"

PYTHONNOUSERSITE=1 "$VENV_DIR/bin/python" -m pip install \
  -r "$tmp_requirements" \
  pytest \
  ruff \
  mypy \
  pre-commit \
  tensorflow-macos==2.16.2 \
  tensorflow-metal==1.2.0

PYTHONNOUSERSITE=1 "$VENV_DIR/bin/python" -m pip install -e . --no-deps

echo
echo "Metal environment ready at $VENV_DIR"
echo "Verifying GPU visibility..."
PYTHONNOUSERSITE=1 "$VENV_DIR/bin/python" scripts/check_acceleration.py
