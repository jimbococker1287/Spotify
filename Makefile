PYTHON ?= python3
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
PROFILE ?= dev
EXTRA_ARGS ?=

ifeq ($(wildcard $(VENV_PY)),)
RUN_PY := $(PYTHON)
else
RUN_PY := $(VENV_PY)
endif

.PHONY: setup train test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt
	mkdir -p data/raw outputs
	touch data/raw/.gitkeep outputs/.gitkeep

train:
	mkdir -p outputs
	$(RUN_PY) -m spotify --profile $(PROFILE) $(EXTRA_ARGS)

test:
	$(RUN_PY) -m pytest

clean:
	rm -rf outputs/*
	mkdir -p outputs
	touch outputs/.gitkeep
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
