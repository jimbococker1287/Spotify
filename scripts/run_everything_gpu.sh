#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv-metal/bin/python" ]]; then
  echo "Missing .venv-metal/bin/python. Run bash scripts/setup_metal_venv.sh first." >&2
  exit 1
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHON_BIN="${PYTHON_BIN:-.venv-metal/bin/python}"
export SPOTIFY_FORCE_CPU="${SPOTIFY_FORCE_CPU:-0}"
export SPOTIFY_MIXED_PRECISION="${SPOTIFY_MIXED_PRECISION:-auto}"

TOTAL_RAM_GB="$("$PYTHON_BIN" - <<'PY'
try:
    import psutil  # type: ignore
    total = int(psutil.virtual_memory().total // (1024 ** 3))
except Exception:
    total = 0
print(total)
PY
)"

# Apple Silicon uses unified memory, so leave more headroom for the handoff
# from Metal training into the classical stages on 16-18 GB machines unless
# the user explicitly overrides the worker counts.
if [[ -z "${SPOTIFY_CLASSICAL_MODEL_WORKERS:-}" ]] && (( TOTAL_RAM_GB > 0 )) && (( TOTAL_RAM_GB <= 18 )); then
  export SPOTIFY_CLASSICAL_MODEL_WORKERS="2"
fi
if [[ -z "${SPOTIFY_MAX_CLASSICAL_WORKERS:-}" ]] && [[ -n "${SPOTIFY_CLASSICAL_MODEL_WORKERS:-}" ]]; then
  export SPOTIFY_MAX_CLASSICAL_WORKERS="$SPOTIFY_CLASSICAL_MODEL_WORKERS"
fi
if [[ -z "${SPOTIFY_BACKTEST_WORKERS:-}" ]] && (( TOTAL_RAM_GB > 0 )) && (( TOTAL_RAM_GB <= 18 )); then
  export SPOTIFY_BACKTEST_WORKERS="1"
fi
if [[ -z "${SPOTIFY_OPTUNA_JOBS:-}" ]] && (( TOTAL_RAM_GB > 0 )) && (( TOTAL_RAM_GB <= 18 )); then
  export SPOTIFY_OPTUNA_JOBS="2"
fi
if [[ -z "${SPOTIFY_OPTUNA_MODEL_WORKERS:-}" ]] && (( TOTAL_RAM_GB > 0 )) && (( TOTAL_RAM_GB <= 18 )); then
  export SPOTIFY_OPTUNA_MODEL_WORKERS="1"
fi
if [[ -z "${SPOTIFY_SKLEARN_NJOBS:-}" ]] && (( TOTAL_RAM_GB > 0 )) && (( TOTAL_RAM_GB <= 18 )); then
  export SPOTIFY_SKLEARN_NJOBS="1"
fi
if [[ -z "${SPOTIFY_TF_DATA_CACHE_FRACTION:-}" ]] && (( TOTAL_RAM_GB > 0 )) && (( TOTAL_RAM_GB <= 18 )); then
  export SPOTIFY_TF_DATA_CACHE_FRACTION="0.40"
fi

enable_shap_raw="${SPOTIFY_ENABLE_SHAP:-0}"
extra_args=("$@")
if [[ "$enable_shap_raw" != "1" && "$enable_shap_raw" != "true" && "$enable_shap_raw" != "yes" && "$enable_shap_raw" != "on" ]]; then
  extra_args+=("--no-shap")
fi

# Reuse the higher-parallelism launcher so Metal deep training and CPU-heavy
# classical/Optuna/backtest stages both run in their faster configuration.
exec bash scripts/run_everything_cpu_boost.sh "${extra_args[@]}"
