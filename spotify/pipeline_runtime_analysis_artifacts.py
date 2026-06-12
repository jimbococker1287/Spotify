from __future__ import annotations

import hashlib
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
from .run_artifacts import safe_read_json, write_json


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


def _prediction_bundle_digest(rows: list[dict[str, object]]) -> str:
    hasher = hashlib.sha256()
    bundle_rows = sorted(
        (
            str(row.get("model_name", "")).strip(),
            Path(str(row.get("prediction_bundle_path", "")).strip()),
        )
        for row in rows
        if str(row.get("prediction_bundle_path", "")).strip()
    )
    for model_name, bundle_path in bundle_rows:
        hasher.update(model_name.encode("utf-8"))
        hasher.update(bundle_path.name.encode("utf-8"))
        try:
            stat = bundle_path.stat()
            hasher.update(str(stat.st_size).encode("ascii"))
            hasher.update(str(stat.st_mtime_ns).encode("ascii"))
        except OSError:
            hasher.update(b"missing")
    return hasher.hexdigest()[:24]


def _ensemble_result_metadata_path(run_dir: Path) -> Path:
    return run_dir / "analysis" / "ensemble_blended_ensemble_result.json"


def _restore_ensemble_result(
    *,
    run_dir: Path,
    restored_paths: list[Path],
) -> dict[str, object] | None:
    metadata_path = _ensemble_result_metadata_path(run_dir)
    if metadata_path not in restored_paths:
        return None
    payload = safe_read_json(metadata_path, default=None)
    if not isinstance(payload, dict):
        return None
    row = payload.get("row")
    if not isinstance(row, dict):
        return None
    restored_row = dict(row)
    bundle_path = run_dir / "prediction_bundles" / "ensemble_blended_ensemble.npz"
    if bundle_path.exists():
        restored_row["prediction_bundle_path"] = str(bundle_path)
        summary_path = run_dir / "analysis" / "ensemble_blended_ensemble_summary.json"
        summary = safe_read_json(summary_path, default=None)
        if isinstance(summary, dict):
            summary["prediction_bundle_path"] = str(bundle_path)
            write_json(summary_path, summary)
    return restored_row


def _run_probability_ensemble(
    *,
    context: PipelineAnalysisContext,
    deps: PipelineAnalysisDeps,
) -> None:
    cache_fingerprint = _analysis_cache_fingerprint(context)
    cache_enabled = bool(cache_fingerprint) and phase_artifact_cache_enabled(
        env_var="SPOTIFY_CACHE_ANALYSIS_ARTIFACTS",
        default="1",
    )
    cache_payload = {
        "prepared_fingerprint": cache_fingerprint,
        "sequence_length": int(context.config.sequence_length),
        "result_rows": _normalized_result_rows_for_cache(context.result_rows),
        "prediction_bundle_digest": _prediction_bundle_digest(context.result_rows),
        "source_digest": _phase_source_digest(deps.build_probability_ensemble),
    }
    cache_key = stable_payload_digest(cache_payload) if cache_enabled else ""
    cache_paths = (
        resolve_phase_artifact_cache_paths(
            cache_root=_analysis_cache_root(context),
            namespace="analysis",
            phase_name="probability_ensemble",
            cache_key=cache_key,
        )
        if cache_enabled
        else None
    )

    with context.phase_recorder.phase(
        "probability_ensemble",
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
                restored_row = _restore_ensemble_result(
                    run_dir=context.run_dir,
                    restored_paths=restored,
                )
                if restored_row is not None:
                    context.result_rows.append(restored_row)
                context.artifact_paths.extend(restored)
                phase["artifact_count"] = int(len(restored))
                phase["cache_hit"] = True
                phase["result_built"] = restored_row is not None
                return

        ensemble_result = deps.build_probability_ensemble(
            data=context.prepared,
            results=context.result_rows,
            sequence_length=context.config.sequence_length,
            run_dir=context.run_dir,
            logger=context.logger,
        )
        artifacts: list[Path] = []
        result_row: dict[str, object] | None = None
        if ensemble_result is not None:
            result_row = dict(ensemble_result.row)
            context.result_rows.append(result_row)
            artifacts.extend(ensemble_result.artifact_paths)

        metadata_path = write_json(
            _ensemble_result_metadata_path(context.run_dir),
            {"row": result_row},
        )
        artifacts.append(metadata_path)
        context.artifact_paths.extend(artifacts)
        phase["artifact_count"] = int(len(artifacts))
        phase["cache_hit"] = False
        phase["result_built"] = result_row is not None
        if cache_enabled and cache_paths is not None:
            save_phase_artifact_cache(
                cache_paths=cache_paths,
                cache_payload=cache_payload,
                run_dir=context.run_dir,
                artifact_paths=artifacts,
                logger=context.logger,
            )


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
    _run_probability_ensemble(context=context, deps=deps)

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
