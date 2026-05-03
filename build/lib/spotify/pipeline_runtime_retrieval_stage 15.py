from __future__ import annotations

from .pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps


def run_retrieval_stack(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> None:
    if context.config.enable_retrieval_stack:
        retrieval_cache_stats: dict[str, object] = {}
        with context.phase_recorder.phase(
            "retrieval_stack",
            candidate_k=context.config.retrieval_candidate_k,
            enable_self_supervised_pretraining=context.config.enable_self_supervised_pretraining,
        ) as phase:
            retrieval_result = deps.train_retrieval_stack(
                data=context.prepared,
                output_dir=context.run_dir,
                random_seed=context.config.random_seed,
                candidate_k=context.config.retrieval_candidate_k,
                enable_self_supervised_pretraining=context.config.enable_self_supervised_pretraining,
                logger=context.logger,
                cache_root=context.config.output_dir / "cache" / "retrieval_stack",
                cache_fingerprint=context.cache_fingerprint,
                cache_stats_out=retrieval_cache_stats,
            )
            for row in retrieval_result.rows:
                context.result_rows.append(dict(row))
            context.artifact_paths.extend(retrieval_result.artifact_paths)
            phase["result_count"] = int(len(retrieval_result.rows))
            phase["cache_enabled"] = bool(retrieval_cache_stats.get("enabled", False))
            phase["cache_fingerprint"] = str(retrieval_cache_stats.get("fingerprint", ""))
            phase["cache_key"] = str(retrieval_cache_stats.get("cache_key", ""))
            phase["cache_hit"] = bool(retrieval_cache_stats.get("hit", False))
            phase["resolved_candidate_k"] = int(
                retrieval_cache_stats.get("candidate_k", context.config.retrieval_candidate_k)
            )
        return

    context.phase_recorder.skip("retrieval_stack", reason="retrieval_disabled")
    context.logger.info("Skipping retrieval stack for this run.")


__all__ = ["run_retrieval_stack"]
