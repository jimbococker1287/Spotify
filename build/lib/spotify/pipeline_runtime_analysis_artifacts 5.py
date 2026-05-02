from __future__ import annotations

from pathlib import Path

from .pipeline_artifact_cache import (
    phase_artifact_cache_enabled,
    resolve_phase_artifact_cache_paths,
    restore_phase_artifact_cache,
    save_phase_artifact_cache,
    source_digest_for_paths,
    stable_payload_digest,
)
from .pipeline_runtime_analysis_types import PipelineAnalysisContext, PipelineAnalysisDeps


_THIS_SOURCE = Path(__file__).resolve()


def _normalized_result_rows_for_cache(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    ignored_keys = {"fit_seconds", "epochs"}
    for row in rows:
        payload = {
            str(key): value
            for key, value in row.items()
            if not str(key).endswith("_path") and str(key) not in ignored_keys
        }
        normalized.append(payload)
    normalized.sort(
        key=lambda row: (
            str(row.get("model_name", "")),
            str(row.get("model_type", "")),
            str(row.get("base_model_name", "")),
        )
    )
    return normalized


def _analysis_cache_root(context: PipelineAnalysisContext) -> Path:
    return context.config.output_dir / "cache" / "analysis_phase_artifacts"


def _analysis_cache_fingerprint(context: PipelineAnalysisContext) -> str:
    return str(context.cache_info_payload.get("fingerprint", "")).strip()


def _phase_source_digest(*callables) -> str:
    paths = [_THIS_SOURCE]
    for item in callables:
        code = getattr(item, "__code__", None)
        filename = getattr(code, "co_filename", "")
        text = str(filename).strip()
        if text:
            paths.append(Path(text))
    return source_digest_for_paths(tuple(paths))


def _run_cached_analysis_phase(
    *,
    context: PipelineAnalysisContext,
    phase_name: str,
    cache_payload: dict[str, object],
    source_digest: str,
    build_artifacts,
) -> None:
    cache_fingerprint = _analysis_cache_fingerprint(context)
    cache_enabled = bool(cache_fingerprint) and phase_artifact_cache_enabled(
        env_var="SPOTIFY_CACHE_ANALYSIS_ARTIFACTS",
        default="1",
    )
    payload = {
        **cache_payload,
        "prepared_fingerprint": cache_fingerprint,
        "source_digest": source_digest,
    }
    cache_key = stable_payload_digest(payload) if cache_enabled else ""
    cache_paths = (
        resolve_phase_artifact_cache_paths(
            cache_root=_analysis_cache_root(context),
            namespace="analysis",
            phase_name=phase_name,
            cache_key=cache_key,
        )
        if cache_enabled
        else None
    )

    with context.phase_recorder.phase(
        phase_name,
        cache_enabled=cache_enabled,
        cache_key=cache_key,
    ) as phase:
        if cache_enabled and cache_paths is not None:
            restored = restore_phase_artifact_cache(
                cache_paths=cache_paths,
                run_dir=context.run_dir,
                logger=context.logger,
            )
            if restored is not None:
                context.artifact_paths.extend(restored)
                phase["artifact_count"] = int(len(restored))
                phase["cache_hit"] = True
                return

        artifacts = list(build_artifacts())
        context.artifact_paths.extend(artifacts)
        phase["artifact_count"] = int(len(artifacts))
        phase["cache_hit"] = False
        if cache_enabled and cache_paths is not None:
            save_phase_artifact_cache(
                cache_paths=cache_paths,
                cache_payload=payload,
                run_dir=context.run_dir,
                artifact_paths=artifacts,
                logger=context.logger,
            )


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

    result_rows_digest = stable_payload_digest(_normalized_result_rows_for_cache(context.result_rows))
    base_payload = {
        "sequence_length": int(context.config.sequence_length),
        "random_seed": int(context.config.random_seed),
        "result_rows_digest": result_rows_digest,
    }

    _run_cached_analysis_phase(
        context=context,
        phase_name="extended_evaluation",
        cache_payload={
            **base_payload,
            "enable_conformal": bool(context.config.enable_conformal),
            "conformal_alpha": float(context.config.conformal_alpha),
            "max_train_samples": int(context.config.classical_max_train_samples),
        },
        source_digest=_phase_source_digest(deps.run_extended_evaluation),
        build_artifacts=lambda: deps.run_extended_evaluation(
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
        ),
    )

    _run_cached_analysis_phase(
        context=context,
        phase_name="drift_diagnostics",
        cache_payload={
            "sequence_length": int(context.config.sequence_length),
            "prepared_fingerprint": _analysis_cache_fingerprint(context),
        },
        source_digest=_phase_source_digest(deps.run_drift_diagnostics),
        build_artifacts=lambda: deps.run_drift_diagnostics(
            data=context.prepared,
            sequence_length=context.config.sequence_length,
            output_dir=context.run_dir / "analysis",
            logger=context.logger,
        ),
    )

    _run_cached_analysis_phase(
        context=context,
        phase_name="robustness_slice_evaluation",
        cache_payload=base_payload,
        source_digest=_phase_source_digest(deps.run_robustness_slice_evaluation),
        build_artifacts=lambda: deps.run_robustness_slice_evaluation(
            data=context.prepared,
            results=context.result_rows,
            sequence_length=context.config.sequence_length,
            run_dir=context.run_dir,
            logger=context.logger,
        ),
    )

    _run_cached_analysis_phase(
        context=context,
        phase_name="policy_simulation",
        cache_payload=base_payload,
        source_digest=_phase_source_digest(deps.run_policy_simulation),
        build_artifacts=lambda: deps.run_policy_simulation(
            data=context.prepared,
            results=context.result_rows,
            run_dir=context.run_dir,
            logger=context.logger,
        ),
    )

    if context.config.enable_friction_analysis:
        _run_cached_analysis_phase(
            context=context,
            phase_name="friction_proxy_analysis",
            cache_payload={
                "sequence_length": int(context.config.sequence_length),
                "prepared_fingerprint": _analysis_cache_fingerprint(context),
            },
            source_digest=_phase_source_digest(deps.run_friction_proxy_analysis),
            build_artifacts=lambda: deps.run_friction_proxy_analysis(
                data=context.prepared,
                output_dir=context.run_dir / "analysis",
                logger=context.logger,
            ),
        )
    else:
        context.phase_recorder.skip("friction_proxy_analysis", reason="friction_analysis_disabled")
        context.logger.info("Skipping friction proxy analysis for this run.")

    if context.config.enable_moonshot_lab:
        _run_cached_analysis_phase(
            context=context,
            phase_name="moonshot_lab",
            cache_payload={
                **base_payload,
                "artist_labels_digest": stable_payload_digest(list(context.artist_labels)),
            },
            source_digest=_phase_source_digest(deps.run_moonshot_lab),
            build_artifacts=lambda: deps.run_moonshot_lab(
                data=context.prepared,
                results=context.result_rows,
                run_dir=context.run_dir,
                sequence_length=context.config.sequence_length,
                artist_labels=context.artist_labels,
                random_seed=context.config.random_seed,
                logger=context.logger,
            ),
        )
    else:
        context.phase_recorder.skip("moonshot_lab", reason="moonshot_lab_disabled")
        context.logger.info("Skipping moonshot lab for this run.")


__all__ = ["run_analysis_artifacts"]
