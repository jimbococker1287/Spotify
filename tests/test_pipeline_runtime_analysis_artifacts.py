from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

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
