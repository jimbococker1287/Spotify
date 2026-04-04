from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import spotify.backtesting as backtesting
from spotify.backtesting import BacktestFoldResult, run_temporal_backtest
from spotify.data import PreparedData
from spotify.drift import run_drift_diagnostics


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _prepared_data() -> PreparedData:
    return PreparedData(
        df=pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=10, freq="h"),
                "artist_label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
                "master_metadata_album_artist_name": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "B"],
                "hour": list(range(10)),
                "dayofweek": [3] * 10,
                "session_position": list(range(10)),
                "is_artist_repeat_from_prev": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                "skipped": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            }
        ),
        context_features=["hour", "session_position"],
        X_seq_train=np.array([[0, 1], [1, 2], [2, 0], [0, 1]], dtype="int32"),
        X_seq_val=np.array([[1, 2], [2, 1]], dtype="int32"),
        X_seq_test=np.array([[0, 2], [2, 1]], dtype="int32"),
        X_ctx_train=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype="float32"),
        X_ctx_val=np.array([[6.0, 6.0], [7.0, 7.0]], dtype="float32"),
        X_ctx_test=np.array([[8.0, 8.0], [9.0, 9.0]], dtype="float32"),
        y_train=np.array([1, 2, 0, 1], dtype="int32"),
        y_val=np.array([2, 1], dtype="int32"),
        y_test=np.array([2, 1], dtype="int32"),
        y_skip_train=np.array([0, 1, 0, 0], dtype="float32"),
        y_skip_val=np.array([1, 0], dtype="float32"),
        y_skip_test=np.array([0, 1], dtype="float32"),
        num_artists=3,
        num_ctx=2,
    )


def test_run_drift_diagnostics_writes_expected_artifacts(tmp_path: Path) -> None:
    data = _prepared_data()

    artifacts = run_drift_diagnostics(
        data=data,
        sequence_length=2,
        output_dir=tmp_path,
        logger=_logger("spotify.test.drift"),
    )

    expected = {
        tmp_path / "data_drift_summary.json",
        tmp_path / "context_feature_drift.csv",
        tmp_path / "segment_drift.csv",
        tmp_path / "context_feature_drift_by_group.csv",
        tmp_path / "context_feature_drift.png",
        tmp_path / "segment_drift.png",
    }
    assert expected.issubset(set(artifacts))
    summary = json.loads((tmp_path / "data_drift_summary.json").read_text(encoding="utf-8"))
    assert summary["train_rows"] == 4
    assert summary["context_feature_count"] == 2
    assert "target_drift" in summary
    assert "context_drift_by_group" in summary
    assert "drift_interpretation" in summary


def test_run_temporal_backtest_supports_deep_models_via_injected_runner(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6), (6, 8)])

    def _fake_deep_job(**kwargs) -> BacktestFoldResult:
        assert kwargs["model_name"] == "gru_artist"
        return BacktestFoldResult(
            model_name="gru_artist",
            model_type="deep",
            model_family="neural",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_seq_fit"]),
            test_rows=len(kwargs["X_seq_test"]),
            fit_seconds=0.25,
            top1=0.5,
            top5=1.0,
        )

    monkeypatch.setattr(backtesting, "_run_deep_backtest_job", _fake_deep_job)

    rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path,
        selected_models=("gru_artist",),
        random_seed=42,
        folds=2,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.deep_backtest"),
        deep_model_builders={"gru_artist": lambda: object()},
        strategy=object(),
    )

    assert len(rows) == 2
    assert all(row.model_type == "deep" for row in rows)
    csv_payload = (tmp_path / "temporal_backtest.csv").read_text(encoding="utf-8")
    assert "model_type" in csv_payload
    assert "gru_artist" in csv_payload


def test_run_temporal_backtest_honors_env_sample_caps(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    captured: list[tuple[int, int]] = []

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 8)])

    def _fake_backtest_job(**kwargs) -> BacktestFoldResult:
        captured.append((len(kwargs["X_train"]), len(kwargs["X_test"])))
        return BacktestFoldResult(
            model_name=str(kwargs["model_name"]),
            model_type="classical",
            model_family="linear",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_train"]),
            test_rows=len(kwargs["X_test"]),
            fit_seconds=0.01,
            top1=0.5,
            top5=1.0,
        )

    monkeypatch.setattr(backtesting, "_run_backtest_job", _fake_backtest_job)
    monkeypatch.setenv("SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES", "2")
    monkeypatch.setenv("SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES", "1")

    rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path,
        selected_models=("logreg",),
        random_seed=42,
        folds=1,
        max_train_samples=10,
        max_eval_samples=10,
        logger=_logger("spotify.test.backtest_caps"),
    )

    assert captured == [(2, 1)]
    assert len(rows) == 1
    assert rows[0].train_rows == 2
    assert rows[0].test_rows == 1


def test_run_temporal_backtest_scores_each_classical_eval_split_once(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    predict_proba_calls: list[int] = []

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6)])

    class _CountingEstimator:
        classes_ = np.array([0, 1, 2], dtype="int32")

        def fit(self, X: np.ndarray, y: np.ndarray) -> "_CountingEstimator":
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            predict_proba_calls.append(len(X))
            return np.array(
                [
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                ],
                dtype="float32",
            )[: len(X)]

    monkeypatch.setattr(
        backtesting,
        "build_classical_estimator",
        lambda _model_name, _seed, params=None, estimator_n_jobs=-1: ("linear", _CountingEstimator()),
    )

    rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path,
        selected_models=("logreg",),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_predict_proba"),
    )

    assert len(rows) == 1
    assert predict_proba_calls == [2]
    assert rows[0].top1 == 1.0


def test_resolve_backtest_workers_defaults_to_two_jobs_when_available(monkeypatch) -> None:
    monkeypatch.setattr(backtesting.os, "cpu_count", lambda: 8)

    assert backtesting._resolve_backtest_workers(None, job_count=1) == 1
    assert backtesting._resolve_backtest_workers("", job_count=2) == 2
    assert backtesting._resolve_backtest_workers("auto", job_count=4) == 2
    assert backtesting._resolve_backtest_workers("1", job_count=4) == 1


def test_run_temporal_backtest_supports_tuned_classical_aliases(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6)])

    def _fake_backtest_job(**kwargs) -> BacktestFoldResult:
        captured.append(
            {
                "model_name": kwargs["model_name"],
                "estimator_model_name": kwargs.get("estimator_model_name"),
                "estimator_params": kwargs.get("estimator_params"),
                "result_model_type": kwargs.get("result_model_type"),
            }
        )
        return BacktestFoldResult(
            model_name=str(kwargs["model_name"]),
            model_type=str(kwargs.get("result_model_type", "classical")),
            model_family="mlp",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_train"]),
            test_rows=len(kwargs["X_test"]),
            fit_seconds=0.01,
            top1=0.5,
            top5=1.0,
        )

    monkeypatch.setattr(backtesting, "_run_backtest_job", _fake_backtest_job)
    monkeypatch.setenv("SPOTIFY_BACKTEST_WORKERS", "1")

    rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path,
        selected_models=("mlp", "mlp_optuna"),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_tuned_alias"),
        tuned_model_specs={
            "mlp_optuna": {
                "base_model_name": "mlp",
                "best_params": {"hidden_1": 128, "hidden_2": 64},
            }
        },
    )

    assert len(rows) == 2
    tuned_call = next(item for item in captured if item["model_name"] == "mlp_optuna")
    assert tuned_call["estimator_model_name"] == "mlp"
    assert tuned_call["estimator_params"] == {"hidden_1": 128, "hidden_2": 64}
    assert tuned_call["result_model_type"] == "classical_tuned"


def test_run_temporal_backtest_supports_retrieval_models(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6)])

    def _fake_retrieval_backtest_job(**kwargs) -> BacktestFoldResult:
        captured.append(
            {
                "model_name": kwargs["model_name"],
                "train_rows": len(kwargs["X_seq_train"]),
                "test_rows": len(kwargs["X_seq_test"]),
            }
        )
        return BacktestFoldResult(
            model_name=str(kwargs["model_name"]),
            model_type="retrieval_reranker",
            model_family="candidate_reranker",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_seq_train"]),
            test_rows=len(kwargs["X_seq_test"]),
            fit_seconds=0.05,
            top1=0.44,
            top5=0.88,
        )

    monkeypatch.setattr(backtesting, "_run_retrieval_backtest_job", _fake_retrieval_backtest_job)

    rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path,
        selected_models=("retrieval_reranker",),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_retrieval"),
    )

    assert len(rows) == 1
    assert rows[0].model_name == "retrieval_reranker"
    assert rows[0].model_type == "retrieval_reranker"
    assert captured == [{"model_name": "retrieval_reranker", "train_rows": 4, "test_rows": 2}]
