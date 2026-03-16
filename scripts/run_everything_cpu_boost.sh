#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

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
  if (( LOGICAL_CPUS >= 12 )); then
    export TF_NUM_INTEROP_THREADS="6"
  elif (( LOGICAL_CPUS >= 8 )); then
    export TF_NUM_INTEROP_THREADS="4"
  else
    export TF_NUM_INTEROP_THREADS="2"
  fi
fi

if [[ -z "${SPOTIFY_CLASSICAL_MODEL_WORKERS:-}" ]]; then
  cpu_boost_workers=1
  if (( TOTAL_RAM_GB >= 24 )); then
    if (( LOGICAL_CPUS >= 12 )); then
      cpu_boost_workers=6
    elif (( LOGICAL_CPUS >= 8 )); then
      cpu_boost_workers=5
    elif (( LOGICAL_CPUS >= 6 )); then
      cpu_boost_workers=4
    else
      cpu_boost_workers=2
    fi
  elif (( TOTAL_RAM_GB >= 16 )); then
    if (( LOGICAL_CPUS >= 10 )); then
      cpu_boost_workers=4
    elif (( LOGICAL_CPUS >= 8 )); then
      cpu_boost_workers=3
    else
      cpu_boost_workers=2
    fi
  elif (( TOTAL_RAM_GB >= 12 )); then
    if (( LOGICAL_CPUS >= 8 )); then
      cpu_boost_workers=3
    else
      cpu_boost_workers=2
    fi
  fi
  export SPOTIFY_CLASSICAL_MODEL_WORKERS="$cpu_boost_workers"
fi

if [[ -z "${SPOTIFY_MAX_CLASSICAL_WORKERS:-}" ]]; then
  export SPOTIFY_MAX_CLASSICAL_WORKERS="$SPOTIFY_CLASSICAL_MODEL_WORKERS"
fi

if [[ -z "${SPOTIFY_BACKTEST_WORKERS:-}" ]]; then
  if (( SPOTIFY_CLASSICAL_MODEL_WORKERS >= 5 )); then
    export SPOTIFY_BACKTEST_WORKERS="4"
  elif (( SPOTIFY_CLASSICAL_MODEL_WORKERS >= 4 )); then
    export SPOTIFY_BACKTEST_WORKERS="3"
  else
    export SPOTIFY_BACKTEST_WORKERS="$SPOTIFY_CLASSICAL_MODEL_WORKERS"
  fi
fi

if [[ -z "${SPOTIFY_OPTUNA_JOBS:-}" ]]; then
  if (( SPOTIFY_CLASSICAL_MODEL_WORKERS >= 4 )); then
    export SPOTIFY_OPTUNA_JOBS="4"
  else
    export SPOTIFY_OPTUNA_JOBS="$SPOTIFY_CLASSICAL_MODEL_WORKERS"
  fi
fi

if [[ -z "${SPOTIFY_OPTUNA_MODEL_WORKERS:-}" ]]; then
  if (( TOTAL_RAM_GB >= 16 )) && (( SPOTIFY_CLASSICAL_MODEL_WORKERS >= 4 )); then
    export SPOTIFY_OPTUNA_MODEL_WORKERS="2"
  else
    export SPOTIFY_OPTUNA_MODEL_WORKERS="1"
  fi
fi

if [[ -z "${SPOTIFY_TF_DATA_THREADPOOL:-}" ]]; then
  if (( LOGICAL_CPUS >= 10 )); then
    export SPOTIFY_TF_DATA_THREADPOOL="4"
  elif (( LOGICAL_CPUS >= 8 )); then
    export SPOTIFY_TF_DATA_THREADPOOL="3"
  else
    export SPOTIFY_TF_DATA_THREADPOOL="0"
  fi
fi

export SPOTIFY_TF_DATA_CACHE_FRACTION="${SPOTIFY_TF_DATA_CACHE_FRACTION:-0.50}"

if (( SPOTIFY_CLASSICAL_MODEL_WORKERS > 1 )); then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
fi

exec bash scripts/run_everything.sh "$@"
