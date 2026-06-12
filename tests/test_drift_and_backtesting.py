from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

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
        tmp_path / "data_drift_brief.json",
        tmp_path / "data_drift_brief.md",
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
    brief = json.loads((tmp_path / "data_drift_brief.json").read_text(encoding="utf-8"))
    assert brief["status"] in {"stable", "attention"}
    assert brief["inspect_paths"][0] == "analysis/data_drift_summary.json"
    assert "Data Drift Brief" in (tmp_path / "data_drift_brief.md").read_text(encoding="utf-8")


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


def test_run_temporal_backtest_supports_blended_ensemble_models(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6)])
    monkeypatch.setenv("SPOTIFY_BACKTEST_WORKERS", "1")

    def _fake_backtest_job(**kwargs) -> BacktestFoldResult:
        return BacktestFoldResult(
            model_name=str(kwargs["model_name"]),
            model_type="classical",
            model_family="tree_ensemble",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_train"]),
            test_rows=len(kwargs["X_test"]),
            fit_seconds=0.02,
            top1=0.35,
            top5=0.70,
        )

    def _fake_retrieval_backtest_job(**kwargs) -> BacktestFoldResult:
        return BacktestFoldResult(
            model_name=str(kwargs["model_name"]),
            model_type="retrieval_reranker",
            model_family="candidate_reranker",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_seq_train"]),
            test_rows=len(kwargs["X_seq_test"]),
            fit_seconds=0.03,
            top1=0.44,
            top5=0.88,
        )

    def _fake_ensemble_backtest_job(**kwargs) -> BacktestFoldResult:
        captured.append(
            {
                "model_name": kwargs["model_name"],
                "candidate_model_names": kwargs["candidate_model_names"],
                "tuned_model_specs": kwargs["tuned_model_specs"],
            }
        )
        return BacktestFoldResult(
            model_name="blended_ensemble",
            model_type="ensemble",
            model_family="blend",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_seq_train"]),
            test_rows=len(kwargs["X_seq_test"]),
            fit_seconds=0.05,
            top1=0.52,
            top5=0.93,
        )

    monkeypatch.setattr(backtesting, "_run_backtest_job", _fake_backtest_job)
    monkeypatch.setattr(backtesting, "_run_retrieval_backtest_job", _fake_retrieval_backtest_job)
    monkeypatch.setattr(backtesting, "_run_ensemble_backtest_job", _fake_ensemble_backtest_job)

    rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path,
        selected_models=("extra_trees", "retrieval_reranker", "blended_ensemble"),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_ensemble"),
    )

    assert {row.model_name for row in rows} == {"extra_trees", "retrieval_reranker", "blended_ensemble"}
    assert next(row for row in rows if row.model_name == "blended_ensemble").model_type == "ensemble"
    assert captured == [
        {
            "model_name": "blended_ensemble",
            "candidate_model_names": ("extra_trees", "retrieval_reranker", "blended_ensemble"),
            "tuned_model_specs": {},
        }
    ]


def test_run_temporal_backtest_shares_retrieval_fit_across_fold_consumers(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    fit_calls: list[int] = []

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6)])
    monkeypatch.setenv("SPOTIFY_BACKTEST_WORKERS", "1")

    def _fake_fit_retrieval_backtest_models(**kwargs):
        fit_calls.append(int(kwargs["fold_idx"]))
        baseline = SimpleNamespace(
            retrieval_metrics_val={"top1": 0.4, "top5": 0.8},
            retrieval_metrics_test={"top1": 0.45, "top5": 0.85},
            val_retrieval_scores=np.array([[2.0, 0.0, -1.0]], dtype="float32"),
            test_retrieval_scores=np.array(
                [[0.0, -1.0, 2.0], [0.0, 2.0, -1.0]],
                dtype="float32",
            ),
        )
        reranker = SimpleNamespace(
            rerank_metrics_val={"top1": 0.5, "top5": 1.0},
            rerank_metrics_test={"top1": 1.0, "top5": 1.0},
            val_rerank_proba=np.array([[0.8, 0.1, 0.1]], dtype="float32"),
            test_rerank_proba=np.array(
                [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
                dtype="float32",
            ),
        )
        return baseline, reranker, 2.5

    monkeypatch.setattr(backtesting, "_fit_retrieval_backtest_models", _fake_fit_retrieval_backtest_models)
    cache_stats: dict[str, object] = {}

    rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path,
        selected_models=("retrieval_dual_encoder", "retrieval_reranker", "blended_ensemble"),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_shared_retrieval"),
        cache_stats_out=cache_stats,
    )

    assert fit_calls == [1]
    assert [row.model_name for row in rows] == [
        "retrieval_dual_encoder",
        "retrieval_reranker",
        "blended_ensemble",
    ]
    assert cache_stats["retrieval_fit_count"] == 1
    assert cache_stats["retrieval_fit_reuse_count"] == 2


def test_ensemble_backtest_skips_base_candidate_shadowed_by_best_tuned_alias(monkeypatch) -> None:
    fit_calls: list[str] = []

    def _fake_fit_classical_backtest_probabilities(**kwargs):
        model_name = str(kwargs["model_name"])
        fit_calls.append(model_name)
        score = {"mlp_optuna": 0.9, "mlp": 0.4, "extra_trees": 0.7}[model_name]
        return {
            "model_name": model_name,
            "model_type": "classical_tuned" if model_name == "mlp_optuna" else "classical",
            "model_family": "test",
            "base_model_name": "mlp" if model_name == "mlp_optuna" else model_name,
            "fit_seconds": 0.1,
            "val_proba": np.array([[0.1, 0.8, 0.1]], dtype="float32"),
            "test_proba": np.array(
                [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
                dtype="float32",
            ),
            "val_top1": score,
            "val_top5": 1.0,
            "test_top1": 1.0,
            "test_top5": 1.0,
        }

    monkeypatch.setattr(
        backtesting,
        "_fit_classical_backtest_probabilities",
        _fake_fit_classical_backtest_probabilities,
    )

    row = backtesting._run_ensemble_backtest_job(
        model_name="blended_ensemble",
        fold_idx=1,
        random_seed=42,
        estimator_n_jobs=1,
        X_train_tab=np.arange(12, dtype="float32").reshape(4, 3),
        y_train=np.array([1, 2, 0, 1], dtype="int32"),
        X_test_tab=np.arange(6, dtype="float32").reshape(2, 3),
        y_test=np.array([2, 1], dtype="int32"),
        X_seq_train=np.array([[0, 1], [1, 2], [2, 0], [0, 1]], dtype="int32"),
        X_ctx_train=np.arange(8, dtype="float32").reshape(4, 2),
        X_seq_test=np.array([[0, 2], [2, 1]], dtype="int32"),
        X_ctx_test=np.arange(4, dtype="float32").reshape(2, 2),
        num_artists=3,
        logger=_logger("spotify.test.backtest_ensemble_alias"),
        candidate_model_names=("mlp", "mlp_optuna", "extra_trees", "blended_ensemble"),
        tuned_model_specs={
            "mlp_optuna": {
                "base_model_name": "mlp",
                "best_params": {"hidden_1": 128},
            }
        },
    )

    assert fit_calls == ["mlp_optuna", "extra_trees"]
    assert row.model_name == "blended_ensemble"


def test_retrieval_backtest_runtime_config_tracks_extended_knobs(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_ANN_BITS", "12")
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_BATCH_SIZE", "512")
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_LR", "0.045")
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_L2", "0.0001")

    config = backtesting._retrieval_backtest_runtime_config(num_artists=200)

    assert config["ann_bits"] == 12
    assert config["batch_size"] == 512
    assert config["learning_rate"] == 0.045
    assert config["l2"] == 0.0001


def test_run_retrieval_backtest_job_forwards_backtest_specific_retrieval_knobs(monkeypatch) -> None:
    import spotify.retrieval_runtime_eval as retrieval_runtime_eval
    import spotify.retrieval_seed_selection as retrieval_seed_selection

    captured_env: dict[str, str | None] = {}

    def _fake_train_pretraining_seed(**kwargs):
        captured_env["ann_bits"] = os.getenv("SPOTIFY_RETRIEVAL_ANN_BITS")
        captured_env["batch_size"] = os.getenv("SPOTIFY_RETRIEVAL_BATCH_SIZE")
        captured_env["learning_rate"] = os.getenv("SPOTIFY_RETRIEVAL_LR")
        captured_env["l2"] = os.getenv("SPOTIFY_RETRIEVAL_L2")
        result = SimpleNamespace(
            artist_embeddings=np.eye(3, dtype="float32"),
            artist_frequency=np.array([0.4, 0.35, 0.25], dtype="float32"),
        )
        return result, Path("/tmp/pretrain.joblib"), np.eye(3, dtype="float32"), np.zeros((2, 3), dtype="float32"), np.zeros(3, dtype="float32"), 8, []

    def _fake_evaluate_retrieval_baseline(**kwargs):
        return SimpleNamespace(retrieval_metrics_test={"top1": 0.2, "top5": 0.4})

    def _fake_train_and_evaluate_reranker(**kwargs):
        return SimpleNamespace(rerank_metrics_test={"top1": 0.33, "top5": 0.66})

    monkeypatch.setattr(retrieval_seed_selection, "train_pretraining_seed", _fake_train_pretraining_seed)
    monkeypatch.setattr(retrieval_runtime_eval, "evaluate_retrieval_baseline", _fake_evaluate_retrieval_baseline)
    monkeypatch.setattr(retrieval_runtime_eval, "train_and_evaluate_reranker", _fake_train_and_evaluate_reranker)
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_ANN_BITS", "12")
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_BATCH_SIZE", "512")
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_LR", "0.045")
    monkeypatch.setenv("SPOTIFY_RETRIEVAL_BACKTEST_L2", "0.0001")

    row = backtesting._run_retrieval_backtest_job(
        model_name="retrieval_reranker",
        fold_idx=1,
        random_seed=42,
        X_seq_train=np.array([[0, 1], [1, 2], [2, 0], [0, 1]], dtype="int32"),
        X_ctx_train=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype="float32"),
        y_train=np.array([1, 2, 0, 1], dtype="int32"),
        X_seq_test=np.array([[0, 2], [2, 1]], dtype="int32"),
        X_ctx_test=np.array([[8.0, 8.0], [9.0, 9.0]], dtype="float32"),
        y_test=np.array([2, 1], dtype="int32"),
        num_artists=3,
        logger=_logger("spotify.test.retrieval_backtest_knobs"),
    )

    assert captured_env == {
        "ann_bits": "12",
        "batch_size": "512",
        "learning_rate": "0.045",
        "l2": "0.0001",
    }
    assert row.model_name == "retrieval_reranker"
    assert row.top1 == 0.33
    assert row.top5 == 0.66


def test_run_temporal_backtest_reuses_cached_phase_artifacts_without_recomputing(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "run"
    cache_fingerprint = "prepared123"
    model_name = "gru_artist"

    monkeypatch.setenv("SPOTIFY_CACHE_BACKTEST", "1")

    cache_payload = backtesting._build_temporal_backtest_cache_payload(
        cache_fingerprint=cache_fingerprint,
        selected_models=(model_name,),
        classical_models=(),
        deep_models=(model_name,),
        retrieval_models=(),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        adaptation_mode="cold",
        tuned_model_specs={},
        sequence_length=int(data.X_seq_train.shape[1]),
        num_artists=int(data.num_artists),
        num_ctx=int(data.num_ctx),
        total_rows=int(len(data.y_train) + len(data.y_val) + len(data.y_test)),
    )
    cache_key = backtesting._build_temporal_backtest_cache_key(cache_payload)
    cache_paths = backtesting._resolve_temporal_backtest_cache_paths(
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        cache_key=cache_key,
    )
    cache_paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    cached_artifacts = {
        "temporal_backtest.csv": b"csv",
        "temporal_backtest.json": b"json",
        "temporal_backtest_summary.csv": b"summary-csv",
        "temporal_backtest_summary.json": b"summary-json",
        "temporal_backtest_top1.png": b"plot",
    }
    for name, payload in cached_artifacts.items():
        (cache_paths.artifact_dir / name).write_bytes(payload)
    backtesting.write_json(
        cache_paths.result_path,
        {
            "cache_schema_version": backtesting.BACKTEST_CACHE_SCHEMA_VERSION,
            "rows": [
                {
                    "model_name": model_name,
                    "model_type": "deep",
                    "model_family": "neural",
                    "fold": 1,
                    "train_rows": 4,
                    "test_rows": 2,
                    "fit_seconds": 3.5,
                    "top1": 0.5,
                    "top5": 1.0,
                    "adaptation_mode": "cold",
                }
            ],
            "artifact_names": sorted(cached_artifacts),
        },
        sort_keys=True,
    )
    backtesting.write_json(cache_paths.metadata_path, cache_payload, sort_keys=True)

    monkeypatch.setattr(backtesting, "build_full_tabular_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dataset build should be skipped on a cache hit")))
    monkeypatch.setattr(backtesting, "_resolve_deep_builders", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("deep builders should be skipped on a cache hit")))
    cache_stats: dict[str, object] = {}

    rows = run_temporal_backtest(
        data=data,
        output_dir=output_dir,
        selected_models=(model_name,),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_cache"),
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        cache_stats_out=cache_stats,
    )

    assert len(rows) == 1
    assert rows[0].model_name == model_name
    assert rows[0].fit_seconds == 3.5
    for name, payload in cached_artifacts.items():
        assert (output_dir / name).read_bytes() == payload
    wall_seconds = float(cache_stats.pop("wall_seconds"))
    assert wall_seconds >= 0.0
    assert cache_stats == {
        "enabled": True,
        "fingerprint": cache_fingerprint,
        "cache_key": cache_key,
        "hit": True,
        "selected_models": [model_name],
        "cache_scope": "whole_run",
        "model_cache_hit_names": [model_name],
        "model_cache_miss_names": [],
        "reused_row_count": 1,
        "retrieval_fit_count": 0,
        "retrieval_fit_reuse_count": 0,
    }


def test_run_temporal_backtest_reuses_model_cache_when_shortlist_changes(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    cache_root = tmp_path / "cache"
    fit_calls: list[str] = []

    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6)])
    monkeypatch.setenv("SPOTIFY_CACHE_BACKTEST", "1")
    monkeypatch.setenv("SPOTIFY_BACKTEST_WORKERS", "1")

    def _fake_backtest_job(**kwargs) -> BacktestFoldResult:
        model_name = str(kwargs["model_name"])
        fit_calls.append(model_name)
        return BacktestFoldResult(
            model_name=model_name,
            model_type="classical",
            model_family="test",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_train"]),
            test_rows=len(kwargs["X_test"]),
            fit_seconds=1.0,
            top1=0.5,
            top5=1.0,
        )

    monkeypatch.setattr(backtesting, "_run_backtest_job", _fake_backtest_job)

    first_rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path / "first",
        selected_models=("logreg",),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_model_cache_first"),
        cache_root=cache_root,
        cache_fingerprint="prepared123",
    )
    second_cache_stats: dict[str, object] = {}
    second_rows = run_temporal_backtest(
        data=data,
        output_dir=tmp_path / "second",
        selected_models=("logreg", "extra_trees"),
        random_seed=42,
        folds=1,
        max_train_samples=0,
        max_eval_samples=0,
        logger=_logger("spotify.test.backtest_model_cache_second"),
        cache_root=cache_root,
        cache_fingerprint="prepared123",
        cache_stats_out=second_cache_stats,
    )

    assert [row.model_name for row in first_rows] == ["logreg"]
    assert [row.model_name for row in second_rows] == ["logreg", "extra_trees"]
    assert fit_calls == ["logreg", "extra_trees"]
    assert second_cache_stats["cache_scope"] == "partial_model"
    assert second_cache_stats["model_cache_hit_names"] == ["logreg"]
    assert second_cache_stats["model_cache_miss_names"] == ["extra_trees"]
    assert second_cache_stats["reused_row_count"] == 1
