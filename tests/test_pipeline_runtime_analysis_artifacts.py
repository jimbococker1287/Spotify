from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spotify.pipeline_runtime_analysis import PipelineAnalysisContext
from spotify.pipeline_runtime_analysis_artifacts import run_analysis_artifacts
from spotify.run_timing import RunPhaseRecorder


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _write_artifact(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_run_analysis_artifacts_reuses_cached_phase_outputs(tmp_path: Path) -> None:
    counters = {
        "extended": 0,
        "drift": 0,
        "robustness": 0,
        "policy": 0,
        "friction": 0,
    }

    def build_context(run_dir: Path) -> PipelineAnalysisContext:
        return PipelineAnalysisContext(
            artifact_paths=[],
            artist_labels=["A", "B"],
            backtest_rows=[],
            cache_info_payload={"fingerprint": "prepared123"},
            classical_feature_bundle=None,
            config=SimpleNamespace(
                output_dir=tmp_path / "outputs",
                sequence_length=5,
                random_seed=42,
                enable_conformal=False,
                conformal_alpha=0.10,
                classical_max_train_samples=256,
                enable_friction_analysis=True,
                enable_moonshot_lab=False,
            ),
            history_dir=tmp_path / "outputs" / "history",
            logger=_logger(f"analysis-{run_dir.name}"),
            manifest_path=run_dir / "run_manifest.json",
            optuna_rows=[],
            phase_recorder=RunPhaseRecorder(run_id=run_dir.name),
            prepared=SimpleNamespace(),
            raw_df=None,
            result_rows=[
                {
                    "model_name": "dense",
                    "model_type": "deep",
                    "val_top1": 0.61,
                    "test_top1": 0.58,
                    "prediction_bundle_path": str(run_dir / "prediction_bundles" / "deep_dense.npz"),
                }
            ],
            run_classical_models=True,
            run_dir=run_dir,
            run_id=run_dir.name,
        )

    def _extended(*, run_dir: Path, **_kwargs):
        counters["extended"] += 1
        return [_write_artifact(run_dir / "analysis" / "extended.json", "extended")]

    def _drift(*, output_dir: Path, **_kwargs):
        counters["drift"] += 1
        return [_write_artifact(output_dir / "drift.json", "drift")]

    def _robustness(*, run_dir: Path, **_kwargs):
        counters["robustness"] += 1
        return [_write_artifact(run_dir / "analysis" / "robustness_summary.json", "robustness")]

    def _policy(*, run_dir: Path, **_kwargs):
        counters["policy"] += 1
        return [_write_artifact(run_dir / "analysis" / "policy_simulation_summary.json", "policy")]

    def _friction(*, output_dir: Path, **_kwargs):
        counters["friction"] += 1
        return [_write_artifact(output_dir / "friction_summary.json", "friction")]

    deps = SimpleNamespace(
        build_probability_ensemble=lambda **_kwargs: None,
        run_extended_evaluation=_extended,
        run_drift_diagnostics=_drift,
        run_robustness_slice_evaluation=_robustness,
        run_policy_simulation=_policy,
        run_friction_proxy_analysis=_friction,
        run_moonshot_lab=lambda **_kwargs: [],
    )

    first_run_dir = tmp_path / "outputs" / "runs" / "run_a"
    first_context = build_context(first_run_dir)
    run_analysis_artifacts(context=first_context, deps=deps)

    second_run_dir = tmp_path / "outputs" / "runs" / "run_b"
    second_context = build_context(second_run_dir)
    run_analysis_artifacts(context=second_context, deps=deps)

    assert counters == {
        "extended": 1,
        "drift": 1,
        "robustness": 1,
        "policy": 1,
        "friction": 1,
    }
    assert (second_run_dir / "analysis" / "extended.json").exists()
    assert (second_run_dir / "analysis" / "drift.json").exists()
    assert (second_run_dir / "analysis" / "robustness_summary.json").exists()
    assert (second_run_dir / "analysis" / "policy_simulation_summary.json").exists()
    assert (second_run_dir / "analysis" / "friction_summary.json").exists()


def test_run_analysis_artifacts_reuses_cached_probability_ensemble(tmp_path: Path) -> None:
    build_count = 0

    def build_context(run_dir: Path) -> PipelineAnalysisContext:
        bundle_path = run_dir / "prediction_bundles" / "classical_logreg.npz"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bundle_path,
            val_proba=np.array([[0.7, 0.3]], dtype="float32"),
            test_proba=np.array([[0.6, 0.4]], dtype="float32"),
        )
        fixed_timestamp_ns = 1_780_000_000_000_000_000
        os.utime(bundle_path, ns=(fixed_timestamp_ns, fixed_timestamp_ns))
        return PipelineAnalysisContext(
            artifact_paths=[],
            artist_labels=["A", "B"],
            backtest_rows=[],
            cache_info_payload={"fingerprint": "prepared-ensemble"},
            classical_feature_bundle=None,
            config=SimpleNamespace(
                output_dir=tmp_path / "outputs",
                sequence_length=5,
                random_seed=42,
                enable_conformal=False,
                conformal_alpha=0.10,
                classical_max_train_samples=256,
                enable_friction_analysis=False,
                enable_moonshot_lab=False,
            ),
            history_dir=tmp_path / "outputs" / "history",
            logger=_logger(f"ensemble-{run_dir.name}"),
            manifest_path=run_dir / "run_manifest.json",
            optuna_rows=[],
            phase_recorder=RunPhaseRecorder(run_id=run_dir.name),
            prepared=SimpleNamespace(),
            raw_df=None,
            result_rows=[
                {
                    "model_name": "logreg",
                    "model_type": "classical",
                    "val_top1": 0.7,
                    "test_top1": 0.6,
                    "prediction_bundle_path": str(bundle_path),
                }
            ],
            run_classical_models=True,
            run_dir=run_dir,
            run_id=run_dir.name,
        )

    def _build_ensemble(*, run_dir: Path, **_kwargs):
        nonlocal build_count
        build_count += 1
        bundle_path = _write_artifact(
            run_dir / "prediction_bundles" / "ensemble_blended_ensemble.npz",
            "ensemble",
        )
        summary_path = run_dir / "analysis" / "ensemble_blended_ensemble_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            f'{{"prediction_bundle_path": "{bundle_path}"}}',
            encoding="utf-8",
        )
        return SimpleNamespace(
            row={
                "model_name": "blended_ensemble",
                "model_type": "ensemble",
                "val_top1": 0.8,
                "test_top1": 0.75,
                "fit_seconds": 12.0,
                "prediction_bundle_path": str(bundle_path),
            },
            artifact_paths=[bundle_path, summary_path],
        )

    deps = SimpleNamespace(
        build_probability_ensemble=_build_ensemble,
        run_extended_evaluation=lambda **_kwargs: [],
        run_drift_diagnostics=lambda **_kwargs: [],
        run_robustness_slice_evaluation=lambda **_kwargs: [],
        run_policy_simulation=lambda **_kwargs: [],
        run_friction_proxy_analysis=lambda **_kwargs: [],
        run_moonshot_lab=lambda **_kwargs: [],
    )

    first_context = build_context(tmp_path / "outputs" / "runs" / "run_a")
    run_analysis_artifacts(context=first_context, deps=deps)
    second_context = build_context(tmp_path / "outputs" / "runs" / "run_b")
    run_analysis_artifacts(context=second_context, deps=deps)

    assert build_count == 1
    ensemble_row = next(
        row for row in second_context.result_rows if row["model_name"] == "blended_ensemble"
    )
    expected_bundle_path = (
        second_context.run_dir / "prediction_bundles" / "ensemble_blended_ensemble.npz"
    )
    assert ensemble_row["prediction_bundle_path"] == str(expected_bundle_path)
    assert expected_bundle_path.exists()
    phase_records = second_context.phase_recorder.summary(final_status="success")["phases"]
    assert any(
        phase["phase_name"] == "probability_ensemble"
        and phase["metadata"].get("cache_hit") is True
        for phase in phase_records
    )
