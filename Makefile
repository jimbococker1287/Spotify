PYTHON ?= python3
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
PROFILE ?= dev
RUN_NAME ?=
EXTRA_ARGS ?=

ifeq ($(wildcard $(VENV_PY)),)
RUN_PY := $(PYTHON)
else
RUN_PY := $(VENV_PY)
endif

.PHONY: setup train train-classical train-deep train-elite train-everything test clean clean-all

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt
	mkdir -p data/raw outputs
	touch data/raw/.gitkeep outputs/.gitkeep

train:
	mkdir -p outputs
	$(RUN_PY) -m spotify --profile $(PROFILE) $(EXTRA_ARGS)

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

test:
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
