PYTHON ?= python3
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
PROFILE ?= dev
MODE ?= fast
RUN_NAME ?=
EXTRA_ARGS ?=

ifeq ($(wildcard $(VENV_PY)),)
RUN_PY := $(PYTHON)
else
RUN_PY := $(VENV_PY)
endif

.PHONY: setup setup-metal train train-fast train-full train-core train-experimental train-classical train-deep train-elite train-everything train-everything-cpu-boost train-everything-gpu check-acceleration refresh-backtest benchmark-lock regression-guard regression-alert analytics-db compare-public public-insights prune-artifacts storage-report predict-next serve-predict schedule-run lint typecheck qa test clean clean-all

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt
	$(VENV_PIP) install -e . --no-deps
	mkdir -p data/raw outputs
	touch data/raw/.gitkeep outputs/.gitkeep

setup-metal:
	bash scripts/setup_metal_venv.sh

train:
	mkdir -p outputs
	$(RUN_PY) -m spotify --profile $(PROFILE) $(EXTRA_ARGS)

train-fast:
	mkdir -p outputs
	bash scripts/run_fast.sh $(RUN_NAME) $(EXTRA_ARGS)

train-full:
	mkdir -p outputs
	bash scripts/run_full.sh $(RUN_NAME) $(EXTRA_ARGS)

train-core:
	mkdir -p outputs
	bash scripts/run_core.sh $(RUN_NAME) $(EXTRA_ARGS)

train-experimental:
	mkdir -p outputs
	bash scripts/run_experimental.sh $(RUN_NAME) $(EXTRA_ARGS)

train-classical:
	mkdir -p outputs
	$(RUN_PY) -m spotify --profile $(PROFILE) --classical-only $(EXTRA_ARGS)

train-deep:
	mkdir -p outputs
	$(RUN_PY) -m spotify --profile $(PROFILE) --no-classical-models $(EXTRA_ARGS)

train-elite:
	mkdir -p outputs
	$(RUN_PY) -m spotify --profile small --mlflow --optuna --temporal-backtest $(EXTRA_ARGS)

train-everything:
	mkdir -p outputs
	bash scripts/run_everything.sh $(RUN_NAME) $(EXTRA_ARGS)

train-everything-cpu-boost:
	mkdir -p outputs
	bash scripts/run_everything_cpu_boost.sh $(RUN_NAME) $(EXTRA_ARGS)

train-everything-gpu:
	mkdir -p outputs
	bash scripts/run_everything_gpu.sh $(RUN_NAME) $(EXTRA_ARGS)

check-acceleration:
	$(RUN_PY) scripts/check_acceleration.py $(EXTRA_ARGS)

refresh-backtest:
	mkdir -p outputs
	bash scripts/rerun_classical_backtest.sh $(RUN_NAME) $(EXTRA_ARGS)

benchmark-lock:
	mkdir -p outputs
	bash scripts/run_benchmark_lock.sh $(RUN_NAME) $(EXTRA_ARGS)

regression-guard:
	mkdir -p outputs
	$(RUN_PY) scripts/regression_guard.py $(EXTRA_ARGS)

regression-alert:
	$(RUN_PY) scripts/regression_alert.py $(EXTRA_ARGS)

analytics-db:
	$(RUN_PY) scripts/build_analytics_db.py $(EXTRA_ARGS)

compare-public:
	$(RUN_PY) -m spotify.compare_public $(EXTRA_ARGS)

public-insights:
	$(RUN_PY) -m spotify.public_insights $(EXTRA_ARGS)

prune-artifacts:
	bash scripts/prune_artifacts.sh $(EXTRA_ARGS)

storage-report:
	PYTHONPATH=. $(RUN_PY) scripts/storage_report.py $(EXTRA_ARGS)

predict-next:
	$(RUN_PY) -m spotify.predict_next $(EXTRA_ARGS)

serve-predict:
	$(RUN_PY) -m spotify.predict_service $(EXTRA_ARGS)

schedule-run:
	mkdir -p outputs
	bash scripts/run_scheduled.sh $(MODE) $(RUN_NAME) $(EXTRA_ARGS)

lint:
	$(RUN_PY) -m ruff check .

typecheck:
	$(RUN_PY) -m mypy

qa: lint typecheck test

test:
	$(RUN_PY) -m pip install -e . --no-deps
	$(RUN_PY) -m pytest

clean:
	rm -rf outputs/runs outputs/.cache outputs/.mplconfig
	mkdir -p outputs/runs outputs/history
	touch outputs/.gitkeep
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

clean-all:
	rm -rf outputs/*
	mkdir -p outputs
	touch outputs/.gitkeep
