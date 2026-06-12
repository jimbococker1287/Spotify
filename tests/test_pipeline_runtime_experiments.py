from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import spotify.pipeline_runtime_experiments as pipeline_runtime_experiments
import spotify.pipeline_runtime_tensorflow_stage as pipeline_runtime_tensorflow_stage
from spotify.pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps
from spotify.run_timing import RunPhaseRecorder


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_run_experiment_stages_skips_tensorflow_when_deep_and_backtest_are_cached(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    recorder = RunPhaseRecorder(run_id="pipeline-runtime-test")
    logger = _logger("spotify.test.pipeline_runtime.cached")
    config = SimpleNamespace(
        model_names=("dense",),
        batch_size=8,
        epochs=1,
        output_dir=tmp_path,
        random_seed=7,
        sequence_length=3,
        enable_shap=False,
        classical_model_names=(),
        classical_max_train_samples=0,
        classical_max_eval_samples=0,
        enable_optuna=False,
        optuna_model_names=(),
        optuna_trials=0,
        optuna_timeout_seconds=0,
        enable_retrieval_stack=False,
        retrieval_candidate_k=5,
        enable_self_supervised_pretraining=False,
        enable_temporal_backtest=True,
        temporal_backtest_folds=1,
        temporal_backtest_model_names=("dense",),
    )
    context = PipelineExperimentContext(
        artifact_paths=[],
        backtest_rows=[],
        cache_fingerprint="prepared123",
        config=config,
        logger=logger,
        optuna_rows=[],
        phase_recorder=recorder,
        prepared=SimpleNamespace(num_artists=4, num_ctx=2, df=[]),
        result_rows=[],
        run_classical_models=False,
        run_deep_backtest=True,
        run_deep_models=True,
        run_dir=run_dir,
    )

    deep_cache_plan = SimpleNamespace(
        enabled=True,
        fingerprint="prepared123",
        hit_model_names=("dense",),
        miss_model_names=(),
    )
    deep_training_artifacts = SimpleNamespace(
        histories={},
        val_metrics={},
        test_metrics={},
        fit_seconds={},
        prediction_bundle_paths={},
    )
    backtest_cache_inspection = SimpleNamespace(
        enabled=True,
        fingerprint="prepared123",
        cache_key="backtest-cache-key",
        hit=True,
        selected_models=("dense",),
        classical_models=(),
        deep_models=("dense",),
        retrieval_models=(),
        adaptation_mode="cold",
        max_train_samples=0,
        max_eval_samples=0,
        cache_paths=None,
        cache_payload={},
    )
    captured: dict[str, object] = {}

    def _unreachable(*args, **kwargs):
        raise AssertionError("This path should not run in the cached TensorFlow skip test")

    def _fake_train_and_evaluate_models(**kwargs):
        captured["deep_training_strategy"] = kwargs.get("strategy")
        return deep_training_artifacts

    def _fake_run_temporal_backtest(**kwargs):
        captured["backtest_strategy"] = kwargs.get("strategy")
        captured["backtest_builders"] = kwargs.get("deep_model_builders")
        return []

    monkeypatch.setattr(
        pipeline_runtime_tensorflow_stage,
        "load_tensorflow_runtime",
        lambda logger: (_ for _ in ()).throw(AssertionError("TensorFlow should not initialize on a fully cached run")),
    )

    deps = PipelineExperimentDeps(
        ResourceMonitor=lambda logger: None,
        VAL_KEY="val_top1",
        build_classical_feature_bundle=_unreachable,
        build_model_builders=_unreachable,
        persist_to_sqlite=_unreachable,
        plot_learning_curves=_unreachable,
        plot_model_comparison=_unreachable,
        resolve_cached_deep_training_artifacts=lambda **kwargs: deep_cache_plan,
        restore_deep_reporting_artifacts=lambda **kwargs: (
            run_dir / "model_comparison.png",
            [],
            run_dir / "histories.json",
            run_dir / "utilization.png",
            run_dir / "spotify_training.db",
        ),
        inspect_temporal_backtest_cache=lambda **kwargs: backtest_cache_inspection,
        run_classical_benchmarks=_unreachable,
        run_optuna_tuning=_unreachable,
        run_shap_analysis=_unreachable,
        run_temporal_backtest=_fake_run_temporal_backtest,
        save_deep_reporting_artifacts=_unreachable,
        save_histories_json=_unreachable,
        save_utilization_plot=_unreachable,
        train_and_evaluate_models=_fake_train_and_evaluate_models,
        train_retrieval_stack=_unreachable,
    )

    outputs = pipeline_runtime_experiments.run_experiment_stages(context=context, deps=deps)
    summary = recorder.summary(final_status="FINISHED")
    phases = {record["phase_name"]: record for record in summary["phases"]}

    assert outputs.classical_feature_bundle is None
    assert captured["deep_training_strategy"] is None
    assert captured["backtest_strategy"] is None
    assert captured["backtest_builders"] is None
    assert phases["tensorflow_runtime_init"]["status"] == "skipped"
    assert phases["tensorflow_runtime_init"]["metadata"]["reason"] == "deep_training_fully_cached_and_deep_backtest_cached"
    assert phases["release_deep_runtime_resources"]["status"] == "skipped"
    assert phases["release_deep_runtime_resources"]["metadata"]["release_point"] == "not_initialized"
    assert phases["backtest_stage_planning"]["metadata"]["tensorflow_required"] is False
    assert phases["backtest_stage_planning"]["metadata"]["backtest_cache_hit"] is True


def _orchestration_context(tmp_path: Path) -> PipelineExperimentContext:
    return PipelineExperimentContext(
        artifact_paths=[],
        backtest_rows=[],
        cache_fingerprint="prepared123",
        config=SimpleNamespace(),
        logger=_logger("spotify.test.pipeline_runtime.orchestration"),
        optuna_rows=[],
        phase_recorder=RunPhaseRecorder(run_id="pipeline-runtime-orchestration"),
        prepared=SimpleNamespace(),
        result_rows=[],
        run_classical_models=True,
        run_deep_backtest=False,
        run_deep_models=True,
        run_dir=tmp_path,
    )


def _experiment_plan(*, needs_deep_backtest: bool) -> SimpleNamespace:
    return SimpleNamespace(
        classical_feature_bundle="features",
        deep_backtest_cache_inspection=SimpleNamespace(
            enabled=True,
            hit=False,
            cache_key="backtest-key",
            classical_models=("logreg",),
            deep_models=("dense",) if needs_deep_backtest else (),
            retrieval_models=(),
        ),
        deep_cache_plan=SimpleNamespace(),
        deep_cache_stats={
            "enabled": True,
            "hit_model_names": [],
            "miss_model_names": ["dense"],
        },
        needs_tf_for_deep_backtest=needs_deep_backtest,
        needs_tf_for_deep_training=True,
    )


def test_run_experiment_stages_releases_tensorflow_before_cpu_only_backtest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _orchestration_context(tmp_path)
    initial_plan = _experiment_plan(needs_deep_backtest=False)
    initial_plan.deep_backtest_cache_inspection = None
    final_plan = _experiment_plan(needs_deep_backtest=False)
    tensorflow_runtime = object()
    strategy = object()
    events: list[str] = []
    releases: list[dict[str, object]] = []

    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "build_experiment_plan",
        lambda **kwargs: events.append("initial_plan") or initial_plan,
    )
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "run_classical_benchmarks",
        lambda **kwargs: events.append("classical") or [],
    )
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "run_optuna_tuning",
        lambda **kwargs: events.append("optuna") or (("logreg",), {}),
    )
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "run_retrieval_stack",
        lambda **kwargs: events.append("retrieval"),
    )
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "update_experiment_plan_for_backtest",
        lambda **kwargs: events.append("backtest_plan") or final_plan,
    )
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "init_tensorflow_runtime",
        lambda **kwargs: events.append("tensorflow_init") or (tensorflow_runtime, strategy),
    )
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "run_deep_model_training",
        lambda **kwargs: events.append("deep_training"),
    )

    def _release(**kwargs) -> None:
        events.append("release")
        releases.append(kwargs)

    def _backtest(**kwargs) -> None:
        events.append("backtest")
        assert kwargs["strategy"] is None

    monkeypatch.setattr(pipeline_runtime_experiments, "release_deep_runtime_resources", _release)
    monkeypatch.setattr(pipeline_runtime_experiments, "run_temporal_backtest", _backtest)

    pipeline_runtime_experiments.run_experiment_stages(context=context, deps=SimpleNamespace())

    assert events == [
        "initial_plan",
        "classical",
        "optuna",
        "retrieval",
        "backtest_plan",
        "tensorflow_init",
        "deep_training",
        "release",
        "backtest",
    ]
    assert releases == [
        {
            "context": context,
            "tf": tensorflow_runtime,
            "release_point": "before_temporal_backtest",
            "next_stage": "temporal_backtest",
            "deep_backtest_required": False,
        }
    ]
    phases = {
        record["phase_name"]: record
        for record in context.phase_recorder.summary(final_status="FINISHED")["phases"]
    }
    assert phases["backtest_stage_planning"]["metadata"]["tensorflow_release_point"] == (
        "before_temporal_backtest"
    )


def test_run_experiment_stages_releases_tensorflow_when_deep_backtest_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _orchestration_context(tmp_path)
    initial_plan = _experiment_plan(needs_deep_backtest=False)
    initial_plan.deep_backtest_cache_inspection = None
    final_plan = _experiment_plan(needs_deep_backtest=True)
    tensorflow_runtime = object()
    strategy = object()
    events: list[str] = []
    releases: list[dict[str, object]] = []

    monkeypatch.setattr(pipeline_runtime_experiments, "build_experiment_plan", lambda **kwargs: initial_plan)
    monkeypatch.setattr(pipeline_runtime_experiments, "run_classical_benchmarks", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "run_optuna_tuning",
        lambda **kwargs: (("dense",), {}),
    )
    monkeypatch.setattr(pipeline_runtime_experiments, "run_retrieval_stack", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "update_experiment_plan_for_backtest",
        lambda **kwargs: final_plan,
    )
    monkeypatch.setattr(
        pipeline_runtime_experiments,
        "init_tensorflow_runtime",
        lambda **kwargs: (tensorflow_runtime, strategy),
    )
    monkeypatch.setattr(pipeline_runtime_experiments, "run_deep_model_training", lambda **kwargs: None)

    def _backtest(**kwargs) -> None:
        events.append("backtest")
        assert kwargs["strategy"] is strategy
        raise RuntimeError("backtest failed")

    def _release(**kwargs) -> None:
        events.append("release")
        releases.append(kwargs)

    monkeypatch.setattr(pipeline_runtime_experiments, "run_temporal_backtest", _backtest)
    monkeypatch.setattr(pipeline_runtime_experiments, "release_deep_runtime_resources", _release)

    with pytest.raises(RuntimeError, match="backtest failed"):
        pipeline_runtime_experiments.run_experiment_stages(context=context, deps=SimpleNamespace())

    assert events == ["backtest", "release"]
    assert releases[0]["tf"] is tensorflow_runtime
    assert releases[0]["release_point"] == "after_temporal_backtest"
    assert releases[0]["deep_backtest_required"] is True
