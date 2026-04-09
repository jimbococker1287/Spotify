from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spotify.data import PreparedData
from spotify.friction import run_friction_proxy_analysis
import spotify.retrieval_runtime as retrieval_runtime
from spotify.retrieval import train_retrieval_stack, train_self_supervised_artist_embeddings
from spotify.serving import load_predictor, resolve_model_row


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _prepared_data() -> PreparedData:
    return PreparedData(
        df=pd.DataFrame(
            {
                "ts": pd.date_range("2026-02-01", periods=14, freq="h"),
                "artist_label": [0, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3],
                "master_metadata_album_artist_name": [
                    "A",
                    "B",
                    "C",
                    "B",
                    "C",
                    "D",
                    "B",
                    "C",
                    "D",
                    "B",
                    "C",
                    "D",
                    "C",
                    "D",
                ],
                "hour": list(range(14)),
                "dayofweek": [6] * 14,
                "session_position": list(range(14)),
                "is_artist_repeat_from_prev": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "skipped": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
            }
        ),
        context_features=["hour", "tech_playback_errors_24h", "tech_stutter_events_24h", "offline"],
        X_seq_train=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype="int32",
        ),
        X_seq_val=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype="int32",
        ),
        X_seq_test=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype="int32",
        ),
        X_ctx_train=np.array(
            [
                [8.0, 0.0, 0.0, 0.0],
                [9.0, 2.0, 2.0, 1.0],
                [10.0, 0.0, 0.0, 0.0],
                [11.0, 2.0, 2.0, 1.0],
                [12.0, 0.0, 0.0, 0.0],
                [13.0, 2.0, 2.0, 1.0],
            ],
            dtype="float32",
        ),
        X_ctx_val=np.array(
            [
                [14.0, 0.0, 0.0, 0.0],
                [15.0, 2.0, 2.0, 1.0],
            ],
            dtype="float32",
        ),
        X_ctx_test=np.array(
            [
                [16.0, 0.0, 0.0, 0.0],
                [17.0, 2.0, 2.0, 1.0],
            ],
            dtype="float32",
        ),
        y_train=np.array([3, 1, 3, 1, 3, 1], dtype="int32"),
        y_val=np.array([3, 1], dtype="int32"),
        y_test=np.array([3, 1], dtype="int32"),
        y_skip_train=np.array([0, 1, 0, 1, 0, 1], dtype="int32"),
        y_skip_val=np.array([0, 1], dtype="int32"),
        y_skip_test=np.array([0, 1], dtype="int32"),
        num_artists=4,
        num_ctx=4,
    )


def test_train_retrieval_stack_writes_artifacts_and_rows(tmp_path: Path) -> None:
    data = _prepared_data()

    result = train_retrieval_stack(
        data=data,
        output_dir=tmp_path,
        random_seed=42,
        candidate_k=3,
        enable_self_supervised_pretraining=True,
        logger=_logger("spotify.test.retrieval"),
    )

    assert [row["model_type"] for row in result.rows] == ["retrieval", "retrieval_reranker"]
    assert all(Path(str(row["prediction_bundle_path"])).exists() for row in result.rows)
    assert all(Path(str(row["retrieval_artifact_path"])).exists() for row in result.rows)
    assert any((tmp_path / "pretraining").glob("self_supervised_artist_embeddings_*.joblib"))
    summary = json.loads((tmp_path / "retrieval" / "retrieval_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_k"] == 3
    assert summary["selected_pretraining_objective"] in {"cooccurrence", "masked_tail", "contrastive_session"} or str(
        summary["selected_pretraining_objective"]
    ).startswith("blend_")
    assert "ann_validation" in summary
    assert "retrieval" in summary
    assert "reranker" in summary
    assert summary["reranker"]["model_name"] in {"logreg", "hist_gbm"}
    assert result.rows[1]["reranker_model_name"] in {"logreg", "hist_gbm"}


def test_self_supervised_pretraining_caps_pair_count(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    monkeypatch.setenv("SPOTIFY_PRETRAIN_MAX_PAIRS", "4")
    monkeypatch.setenv("SPOTIFY_PRETRAIN_EPOCHS", "1")

    result, artifact_path = train_self_supervised_artist_embeddings(
        data=data,
        output_dir=tmp_path,
        random_seed=11,
        logger=_logger("spotify.test.retrieval.pretrain_cap"),
        embedding_dim=8,
        objective_name="contrastive_session",
    )

    assert result.objective_name == "contrastive_session"
    assert result.pair_count == 4
    assert artifact_path.exists()


def test_retrieval_reranker_artifact_is_serveable(tmp_path: Path) -> None:
    data = _prepared_data()
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True)

    result = train_retrieval_stack(
        data=data,
        output_dir=run_dir,
        random_seed=7,
        candidate_k=3,
        enable_self_supervised_pretraining=True,
        logger=_logger("spotify.test.retrieval.serving"),
    )
    (run_dir / "feature_metadata.json").write_text(
        json.dumps(
            {
                "artist_labels": ["A", "B", "C", "D"],
                "sequence_length": 3,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(json.dumps(result.rows, indent=2), encoding="utf-8")

    row = resolve_model_row(run_dir, explicit_model_name="retrieval_reranker", alias_model_name=None)
    predictor = load_predictor(run_dir=run_dir, row=row, artist_labels=["A", "B", "C", "D"])
    proba = predictor.predict_proba(
        np.array([[0, 1, 2]], dtype="int32"),
        np.array([[12.0, 0.0, 0.0, 0.0]], dtype="float32"),
    )

    assert predictor.model_type == "retrieval_reranker"
    assert proba.shape == (1, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_train_retrieval_stack_reuses_cached_phase_artifacts_without_recomputing(tmp_path: Path, monkeypatch) -> None:
    data = _prepared_data()
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "run"
    cache_fingerprint = "prepared123"

    monkeypatch.setenv("SPOTIFY_CACHE_RETRIEVAL", "1")

    cache_payload = retrieval_runtime._build_retrieval_cache_payload(
        cache_fingerprint=cache_fingerprint,
        data=data,
        random_seed=42,
        candidate_k=3,
        enable_self_supervised_pretraining=True,
    )
    cache_key = retrieval_runtime._build_retrieval_cache_key(cache_payload)
    cache_paths = retrieval_runtime._resolve_retrieval_cache_paths(
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        cache_key=cache_key,
    )
    cache_paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    cached_artifacts = {
        "prediction_bundles/retrieval_dual_encoder.npz": b"dual-bundle",
        "prediction_bundles/retrieval_reranker.npz": b"rerank-bundle",
        "retrieval/retrieval_dual_encoder.joblib": b"dual-model",
        "retrieval/retrieval_reranker.joblib": b"rerank-model",
        "retrieval/retrieval_reranker_estimator.joblib": b"rerank-estimator",
        "retrieval/retrieval_summary.json": b"{}",
        "pretraining/self_supervised_artist_embeddings_cached.joblib": b"pretrain",
    }
    for rel_path, payload in cached_artifacts.items():
        artifact_path = cache_paths.artifact_dir / rel_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(payload)

    dual_row = {
        "model_name": "retrieval_dual_encoder",
        "model_type": "retrieval",
        "model_family": "dual_encoder",
        "val_top1": 0.25,
        "val_top5": 0.75,
        "val_ndcg_at5": 0.3,
        "val_mrr_at5": 0.2,
        "val_coverage_at5": 0.5,
        "val_diversity_at5": 0.4,
        "test_top1": 0.2,
        "test_top5": 0.7,
        "test_ndcg_at5": 0.28,
        "test_mrr_at5": 0.18,
        "test_coverage_at5": 0.45,
        "test_diversity_at5": 0.35,
        "fit_seconds": 1.5,
        "epochs": 6,
        "prediction_bundle_path": str(output_dir / "prediction_bundles" / "retrieval_dual_encoder.npz"),
        "retrieval_artifact_path": str(output_dir / "retrieval" / "retrieval_dual_encoder.joblib"),
        "pretraining_artifact_path": str(output_dir / "pretraining" / "self_supervised_artist_embeddings_cached.joblib"),
        "pretraining_objective": "cooccurrence",
        "val_recall_at3": 0.75,
        "test_recall_at3": 0.7,
        "val_ann_recall_at3": 0.72,
        "test_ann_recall_at3": 0.68,
    }
    reranker_row = {
        "model_name": "retrieval_reranker",
        "model_type": "retrieval_reranker",
        "model_family": "candidate_reranker",
        "val_top1": 0.3,
        "val_top5": 0.8,
        "val_ndcg_at5": 0.34,
        "val_mrr_at5": 0.24,
        "val_coverage_at5": 0.55,
        "val_diversity_at5": 0.42,
        "test_top1": 0.27,
        "test_top5": 0.78,
        "test_ndcg_at5": 0.31,
        "test_mrr_at5": 0.22,
        "test_coverage_at5": 0.5,
        "test_diversity_at5": 0.39,
        "fit_seconds": 1.5,
        "epochs": 6,
        "prediction_bundle_path": str(output_dir / "prediction_bundles" / "retrieval_reranker.npz"),
        "retrieval_artifact_path": str(output_dir / "retrieval" / "retrieval_reranker.joblib"),
        "estimator_artifact_path": str(output_dir / "retrieval" / "retrieval_reranker_estimator.joblib"),
        "pretraining_artifact_path": str(output_dir / "pretraining" / "self_supervised_artist_embeddings_cached.joblib"),
        "pretraining_objective": "cooccurrence",
        "val_recall_at3": 0.75,
        "test_recall_at3": 0.7,
        "val_ann_recall_at3": 0.72,
        "test_ann_recall_at3": 0.68,
    }
    retrieval_runtime.write_json(
        cache_paths.result_path,
        {
            "cache_schema_version": retrieval_runtime.RETRIEVAL_CACHE_SCHEMA_VERSION,
            "rows": [
                retrieval_runtime._serialize_cached_retrieval_row(dual_row, output_dir=output_dir),
                retrieval_runtime._serialize_cached_retrieval_row(reranker_row, output_dir=output_dir),
            ],
            "artifact_names": sorted(cached_artifacts),
        },
        sort_keys=True,
    )
    retrieval_runtime.write_json(cache_paths.metadata_path, cache_payload, sort_keys=True)

    monkeypatch.setattr(
        retrieval_runtime,
        "train_pretraining_seed",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pretraining should be skipped on a cache hit")),
    )
    monkeypatch.setattr(
        retrieval_runtime,
        "evaluate_retrieval_baseline",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("baseline evaluation should be skipped on a cache hit")),
    )
    monkeypatch.setattr(
        retrieval_runtime,
        "train_and_evaluate_reranker",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("reranker training should be skipped on a cache hit")),
    )
    cache_stats: dict[str, object] = {}

    result = train_retrieval_stack(
        data=data,
        output_dir=output_dir,
        random_seed=42,
        candidate_k=3,
        enable_self_supervised_pretraining=True,
        logger=_logger("spotify.test.retrieval.cache"),
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        cache_stats_out=cache_stats,
    )

    assert [row["model_name"] for row in result.rows] == ["retrieval_dual_encoder", "retrieval_reranker"]
    assert cache_stats == {
        "enabled": True,
        "fingerprint": cache_fingerprint,
        "cache_key": cache_key,
        "hit": True,
        "candidate_k": 3,
    }
    for rel_path, payload in cached_artifacts.items():
        assert (output_dir / rel_path).read_bytes() == payload
    assert Path(str(result.rows[0]["prediction_bundle_path"])).exists()
    assert Path(str(result.rows[0]["retrieval_artifact_path"])).exists()
    assert Path(str(result.rows[1]["estimator_artifact_path"])).exists()
    assert Path(str(result.rows[1]["pretraining_artifact_path"])).exists()


def test_friction_proxy_analysis_writes_expected_artifacts(tmp_path: Path) -> None:
    data = _prepared_data()

    artifacts = run_friction_proxy_analysis(
        data=data,
        output_dir=tmp_path,
        logger=_logger("spotify.test.friction"),
    )

    expected = {
        tmp_path / "friction_proxy_summary.json",
        tmp_path / "friction_feature_coefficients.csv",
        tmp_path / "friction_counterfactual_delta.csv",
        tmp_path / "friction_feature_coefficients.png",
    }
    assert expected.issubset(set(artifacts))
    summary = json.loads((tmp_path / "friction_proxy_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["interpretation"] == "proxy_counterfactual_not_causal"
    assert summary["friction_feature_count"] == 3
