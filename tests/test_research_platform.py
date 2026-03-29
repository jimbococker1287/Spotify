from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import spotify.backtesting as backtesting
from spotify.backtesting import BacktestFoldResult, run_temporal_backtest
from spotify.data import PreparedData
from spotify.governance import evaluate_champion_gate
from spotify.policy_eval import run_policy_simulation
from spotify.probability_bundles import save_prediction_bundle
from spotify.research_artifacts import (
    write_ablation_summary,
    write_benchmark_protocol,
    write_experiment_registry,
    write_significance_summary,
)
from spotify.robustness import run_robustness_slice_evaluation


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _prepared_data() -> PreparedData:
    return PreparedData(
        df=pd.DataFrame(
            {
                "ts": pd.date_range("2026-03-01", periods=12, freq="h"),
                "artist_label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0],
                "master_metadata_album_artist_name": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "B", "C", "A"],
                "hour": list(range(12)),
                "dayofweek": [6] * 12,
                "session_position": list(range(12)),
                "is_artist_repeat_from_prev": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                "skipped": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                "platform_code": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
                "session_repeat_ratio_so_far": [0.0, 0.0, 0.0, 0.3, 0.1, 0.0, 0.4, 0.2, 0.1, 0.5, 0.3, 0.2],
                "tech_playback_errors_24h": [0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 1, 0],
                "offline": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            }
        ),
        context_features=["hour", "tech_playback_errors_24h", "offline"],
        X_seq_train=np.array([[0, 1], [1, 2], [2, 0], [0, 1]], dtype="int32"),
        X_seq_val=np.array([[1, 2], [2, 1]], dtype="int32"),
        X_seq_test=np.array([[0, 2], [2, 1]], dtype="int32"),
        X_ctx_train=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 1.0]], dtype="float32"),
        X_ctx_val=np.array([[6.0, 0.0, 0.0], [7.0, 2.0, 1.0]], dtype="float32"),
        X_ctx_test=np.array([[8.0, 0.0, 0.0], [9.0, 3.0, 1.0]], dtype="float32"),
        y_train=np.array([1, 2, 0, 1], dtype="int32"),
        y_val=np.array([2, 1], dtype="int32"),
        y_test=np.array([2, 1], dtype="int32"),
        y_skip_train=np.array([0, 1, 0, 0], dtype="float32"),
        y_skip_val=np.array([0, 1], dtype="float32"),
        y_skip_test=np.array([0, 1], dtype="float32"),
        num_artists=3,
        num_ctx=3,
    )


def test_research_artifact_writers_emit_expected_files(tmp_path: Path) -> None:
    data = _prepared_data()
    config = SimpleNamespace(
        random_seed=42,
        sequence_length=2,
        max_artists=3,
        enable_temporal_backtest=True,
        temporal_backtest_folds=2,
        temporal_backtest_model_names=("gru_artist", "logreg"),
        enable_conformal=True,
        enable_retrieval_stack=True,
        enable_friction_analysis=True,
    )
    results = [
        {"model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "model_family": "candidate_reranker", "val_top1": 0.6, "test_top1": 0.5, "fit_seconds": 1.2},
        {"model_name": "gru_artist", "model_type": "deep", "model_family": "neural", "val_top1": 0.55, "test_top1": 0.45, "fit_seconds": 2.0},
    ]
    backtest_rows = [
        {"model_name": "retrieval_reranker", "top1": 0.60, "fold": 1},
        {"model_name": "retrieval_reranker", "top1": 0.58, "fold": 2},
        {"model_name": "gru_artist", "top1": 0.54, "fold": 1},
        {"model_name": "gru_artist", "top1": 0.50, "fold": 2},
    ]

    protocol_paths = write_benchmark_protocol(
        output_dir=tmp_path,
        run_id="run_x",
        profile="full",
        data=data,
        cache_info={"fingerprint": "abc123", "source_file_count": 2},
        config=config,
    )
    registry_path = write_experiment_registry(
        output_dir=tmp_path,
        run_id="run_x",
        profile="full",
        results=results,
        backtest_rows=backtest_rows,
        config=config,
    )
    ablation_paths = write_ablation_summary(output_dir=tmp_path / "analysis", results=results)
    significance_paths = write_significance_summary(output_dir=tmp_path / "analysis", results=results, backtest_rows=backtest_rows)

    assert all(path.exists() for path in protocol_paths)
    assert registry_path.exists()
    assert all(path.exists() for path in ablation_paths)
    assert all(path.exists() for path in significance_paths)
    protocol_payload = json.loads((tmp_path / "benchmark_protocol.json").read_text(encoding="utf-8"))
    assert protocol_payload["benchmark_contract"]["contract_version"] == "2026-week10-v1"
    assert protocol_payload["protocol"]["benchmark_lock"]["minimum_repeated_runs"] == 3


def test_robustness_and_policy_outputs_are_written(tmp_path: Path) -> None:
    data = _prepared_data()
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True)
    bundle_path = save_prediction_bundle(
        run_dir / "prediction_bundles" / "demo.npz",
        val_proba=np.array([[0.1, 0.2, 0.7], [0.1, 0.8, 0.1]], dtype="float32"),
        test_proba=np.array([[0.2, 0.2, 0.6], [0.1, 0.7, 0.2]], dtype="float32"),
    )
    results = [{"model_name": "demo_model", "model_type": "retrieval_reranker", "prediction_bundle_path": str(bundle_path)}]

    robustness_artifacts = run_robustness_slice_evaluation(
        data=data,
        results=results,
        sequence_length=2,
        run_dir=run_dir,
        logger=_logger("spotify.test.robustness"),
    )
    policy_artifacts = run_policy_simulation(
        data=data,
        results=results,
        run_dir=run_dir,
        logger=_logger("spotify.test.policy"),
    )

    assert (run_dir / "analysis" / "robustness_summary.json").exists()
    assert (run_dir / "analysis" / "policy_simulation_summary.json").exists()
    assert robustness_artifacts
    assert policy_artifacts


def test_backtest_warm_adaptation_reuses_previous_weights(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    monkeypatch.setattr(backtesting, "_build_expanding_windows", lambda _n_rows, _folds: [(4, 6), (6, 8)])
    calls: list[dict[str, object]] = []

    def _fake_deep_job(**kwargs) -> BacktestFoldResult:
        calls.append({"fold_idx": kwargs["fold_idx"], "initial_weights": kwargs.get("initial_weights"), "adaptation_mode": kwargs.get("adaptation_mode")})
        if kwargs.get("weight_sink") is not None:
            kwargs["weight_sink"]["weights"] = [np.array([float(kwargs["fold_idx"])], dtype="float32")]
        return BacktestFoldResult(
            model_name="gru_artist",
            model_type="deep",
            model_family="neural",
            fold=int(kwargs["fold_idx"]),
            train_rows=len(kwargs["X_seq_fit"]),
            test_rows=len(kwargs["X_seq_test"]),
            fit_seconds=0.1,
            top1=0.5,
            top5=1.0,
            adaptation_mode=str(kwargs.get("adaptation_mode", "cold")),
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
        logger=_logger("spotify.test.adaptation"),
        deep_model_builders={"gru_artist": lambda: object()},
        strategy=object(),
        adaptation_mode="warm",
    )

    assert len(rows) == 2
    assert calls[0]["initial_weights"] is None
    assert calls[1]["initial_weights"] is not None
    assert all(row.adaptation_mode == "warm" for row in rows)


def test_champion_gate_can_block_on_selective_risk(tmp_path: Path) -> None:
    history_csv = tmp_path / "history.csv"
    history_csv.write_text("run_id,model_name,val_top1\n", encoding="utf-8")

    result = evaluate_champion_gate(
        history_csv=history_csv,
        current_run_id="run_a",
        current_results=[{"model_name": "retrieval_reranker", "val_top1": 0.6}],
        regression_threshold=0.01,
        metric_source="val_top1",
        require_profile_match=False,
        current_risk_metrics={
            "retrieval_reranker": {
                "val_selective_risk": 0.35,
                "val_abstention_rate": 0.05,
            }
        },
        max_selective_risk=0.20,
    )

    assert result["promoted"] is False
    assert result["status"] == "fail_selective_risk"
