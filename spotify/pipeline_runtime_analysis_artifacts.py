from __future__ import annotations

from .pipeline_runtime_analysis_types import PipelineAnalysisContext, PipelineAnalysisDeps


def run_analysis_artifacts(*, context: PipelineAnalysisContext, deps: PipelineAnalysisDeps) -> None:
    ensemble_result = deps.build_probability_ensemble(
        data=context.prepared,
        results=context.result_rows,
        sequence_length=context.config.sequence_length,
        run_dir=context.run_dir,
        logger=context.logger,
    )
    if ensemble_result is not None:
        context.result_rows.append(dict(ensemble_result.row))
        context.artifact_paths.extend(ensemble_result.artifact_paths)

    with context.phase_recorder.phase(
        "extended_evaluation",
        enable_conformal=context.config.enable_conformal,
        conformal_alpha=context.config.conformal_alpha,
    ) as phase:
        extended_eval_artifacts = deps.run_extended_evaluation(
            data=context.prepared,
            results=context.result_rows,
            sequence_length=context.config.sequence_length,
            run_dir=context.run_dir,
            random_seed=context.config.random_seed,
            max_train_samples=context.config.classical_max_train_samples,
            enable_conformal=context.config.enable_conformal,
            conformal_alpha=context.config.conformal_alpha,
            logger=context.logger,
            feature_bundle=context.classical_feature_bundle,
        )
        context.artifact_paths.extend(extended_eval_artifacts)
        phase["artifact_count"] = int(len(extended_eval_artifacts))

    with context.phase_recorder.phase("drift_diagnostics") as phase:
        drift_artifacts = deps.run_drift_diagnostics(
            data=context.prepared,
            sequence_length=context.config.sequence_length,
            output_dir=context.run_dir / "analysis",
            logger=context.logger,
        )
        context.artifact_paths.extend(drift_artifacts)
        phase["artifact_count"] = int(len(drift_artifacts))

    with context.phase_recorder.phase("robustness_slice_evaluation") as phase:
        robustness_artifacts = deps.run_robustness_slice_evaluation(
            data=context.prepared,
            results=context.result_rows,
            sequence_length=context.config.sequence_length,
            run_dir=context.run_dir,
            logger=context.logger,
        )
        context.artifact_paths.extend(robustness_artifacts)
        phase["artifact_count"] = int(len(robustness_artifacts))

    with context.phase_recorder.phase("policy_simulation") as phase:
        policy_artifacts = deps.run_policy_simulation(
            data=context.prepared,
            results=context.result_rows,
            run_dir=context.run_dir,
            logger=context.logger,
        )
        context.artifact_paths.extend(policy_artifacts)
        phase["artifact_count"] = int(len(policy_artifacts))

    if context.config.enable_friction_analysis:
        with context.phase_recorder.phase("friction_proxy_analysis") as phase:
            friction_artifacts = deps.run_friction_proxy_analysis(
                data=context.prepared,
                output_dir=context.run_dir / "analysis",
                logger=context.logger,
            )
            context.artifact_paths.extend(friction_artifacts)
            phase["artifact_count"] = int(len(friction_artifacts))
    else:
        context.phase_recorder.skip("friction_proxy_analysis", reason="friction_analysis_disabled")
        context.logger.info("Skipping friction proxy analysis for this run.")

    if context.config.enable_moonshot_lab:
        with context.phase_recorder.phase("moonshot_lab") as phase:
            moonshot_artifacts = deps.run_moonshot_lab(
                data=context.prepared,
                results=context.result_rows,
                run_dir=context.run_dir,
                sequence_length=context.config.sequence_length,
                artist_labels=context.artist_labels,
                random_seed=context.config.random_seed,
                logger=context.logger,
            )
            context.artifact_paths.extend(moonshot_artifacts)
            phase["artifact_count"] = int(len(moonshot_artifacts))
    else:
        context.phase_recorder.skip("moonshot_lab", reason="moonshot_lab_disabled")
        context.logger.info("Skipping moonshot lab for this run.")


__all__ = ["run_analysis_artifacts"]
