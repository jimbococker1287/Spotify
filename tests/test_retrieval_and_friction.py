from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spotify.data import PreparedData
from spotify.friction import run_friction_proxy_analysis
from spotify.retrieval import train_retrieval_stack
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
    assert summary["selected_pretraining_objective"] in {"cooccurrence", "masked_tail", "contrastive_session"}
    assert "ann_validation" in summary
    assert "retrieval" in summary
    assert "reranker" in summary


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
