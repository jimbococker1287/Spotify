from __future__ import annotations

import logging
from types import SimpleNamespace

import spotify.pipeline_runtime as pipeline_runtime


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_shortlist_classical_model_names_ranks_by_val_top1_then_fit_seconds() -> None:
    rows = [
        SimpleNamespace(model_name="logreg", val_top1=0.31, fit_seconds=12.0),
        SimpleNamespace(model_name="mlp", val_top1=0.41, fit_seconds=15.0),
        SimpleNamespace(model_name="extra_trees", val_top1=0.41, fit_seconds=28.0),
    ]

    selected = pipeline_runtime._shortlist_classical_model_names(
        ("logreg", "extra_trees", "mlp"),
        rows,
        top_n=2,
        logger=_logger("spotify.test.shortlist.rank"),
        stage_label="Optuna",
    )

    assert selected == ("mlp", "extra_trees")


def test_shortlist_classical_model_names_skips_when_baseline_metrics_missing() -> None:
    rows = [
        SimpleNamespace(model_name="logreg", val_top1=0.31, fit_seconds=12.0),
    ]

    selected = pipeline_runtime._shortlist_classical_model_names(
        ("logreg", "mlp"),
        rows,
        top_n=1,
        logger=_logger("spotify.test.shortlist.missing"),
        stage_label="Temporal backtest",
    )

    assert selected == ("logreg", "mlp")


def test_shortlist_classical_model_names_preserves_passthrough_backtest_candidates() -> None:
    rows = [
        SimpleNamespace(model_name="logreg", val_top1=0.31, fit_seconds=12.0),
        SimpleNamespace(model_name="mlp", val_top1=0.41, fit_seconds=15.0),
        SimpleNamespace(model_name="extra_trees", val_top1=0.37, fit_seconds=20.0),
    ]

    selected = pipeline_runtime._shortlist_classical_model_names(
        ("logreg", "extra_trees", "mlp", "retrieval_reranker", "blended_ensemble"),
        rows,
        top_n=2,
        logger=_logger("spotify.test.shortlist.passthrough"),
        stage_label="Temporal backtest",
        passthrough_names=("retrieval_reranker", "blended_ensemble"),
    )

    assert selected == ("mlp", "extra_trees", "retrieval_reranker", "blended_ensemble")


def test_resolve_shortlist_top_n_parses_positive_ints(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_OPTUNA_SHORTLIST_TOP_N", "2")
    assert pipeline_runtime._resolve_shortlist_top_n("SPOTIFY_OPTUNA_SHORTLIST_TOP_N") == 2

    monkeypatch.setenv("SPOTIFY_OPTUNA_SHORTLIST_TOP_N", "0")
    assert pipeline_runtime._resolve_shortlist_top_n("SPOTIFY_OPTUNA_SHORTLIST_TOP_N") == 0

    monkeypatch.setenv("SPOTIFY_OPTUNA_SHORTLIST_TOP_N", "bad")
    assert pipeline_runtime._resolve_shortlist_top_n("SPOTIFY_OPTUNA_SHORTLIST_TOP_N") == 0
