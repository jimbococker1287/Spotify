from __future__ import annotations

from spotify.champion_alias import (
    best_deep_model_name,
    best_serveable_model,
    preferred_serveable_model,
    read_champion_alias,
    resolve_prediction_run_dir,
    write_champion_alias,
)


def test_best_deep_model_name_uses_top1_metric() -> None:
    result_rows = [
        {"model_name": "logreg", "model_type": "classical", "val_top1": 0.80},
        {"model_name": "dense", "model_type": "deep", "val_top1": 0.41},
        {"model_name": "gru", "model_type": "deep", "val_top1": 0.46},
    ]

    assert best_deep_model_name(result_rows) == "gru"


def test_write_and_read_champion_alias_round_trip(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    run_dir.mkdir(parents=True)

    alias_file = write_champion_alias(
        output_dir=output_dir,
        run_id="run_a",
        run_dir=run_dir,
        model_name="gru",
    )
    alias = read_champion_alias(alias_file)

    assert alias is not None
    assert alias.run_id == "run_a"
    assert alias.model_name == "gru"
    assert alias.model_type == "deep"
    assert alias.run_dir == run_dir.resolve()


def test_resolve_prediction_run_dir_uses_default_alias(tmp_path) -> None:
    project_root = tmp_path
    output_dir = project_root / "outputs"
    run_dir = output_dir / "runs" / "run_b"
    run_dir.mkdir(parents=True)

    write_champion_alias(
        output_dir=output_dir,
        run_id="run_b",
        run_dir=run_dir,
        model_name="lstm",
    )

    resolved_run_dir, alias_model_name = resolve_prediction_run_dir(None, project_root=project_root)
    assert resolved_run_dir == run_dir.resolve()
    assert alias_model_name == "lstm"


def test_best_serveable_model_prefers_highest_valid_row(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    estimator_path = run_dir / "estimators" / "classical_mlp.joblib"
    estimator_path.parent.mkdir(parents=True)
    estimator_path.write_bytes(b"stub")

    result_rows = [
        {"model_name": "gru", "model_type": "deep", "val_top1": 0.25},
        {
            "model_name": "mlp",
            "model_type": "classical",
            "val_top1": 0.31,
            "estimator_artifact_path": str(estimator_path),
        },
    ]

    assert best_serveable_model(result_rows, run_dir=run_dir) == ("mlp", "classical")


def test_preferred_serveable_model_uses_requested_promoted_model_when_valid(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    retrieval_path = run_dir / "retrieval" / "retrieval_reranker.joblib"
    retrieval_path.parent.mkdir(parents=True)
    retrieval_path.write_bytes(b"stub")
    estimator_path = run_dir / "estimators" / "classical_mlp.joblib"
    estimator_path.parent.mkdir(parents=True)
    estimator_path.write_bytes(b"stub")

    result_rows = [
        {
            "model_name": "mlp",
            "model_type": "classical",
            "val_top1": 0.60,
            "estimator_artifact_path": str(estimator_path),
        },
        {
            "model_name": "retrieval_reranker",
            "model_type": "retrieval_reranker",
            "val_top1": 0.55,
            "retrieval_artifact_path": str(retrieval_path),
        },
    ]

    assert preferred_serveable_model(
        result_rows,
        run_dir=run_dir,
        preferred_model_name="retrieval_reranker",
    ) == ("retrieval_reranker", "retrieval_reranker")
