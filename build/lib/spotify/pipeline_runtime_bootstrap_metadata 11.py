from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .pipeline_helpers import _write_json_artifact
from .pipeline_runtime_dependency_types import PipelineRuntimeDeps


def build_cache_info_payload(*, cache_info) -> dict[str, object]:
    return {
        "enabled": cache_info.enabled,
        "hit": cache_info.hit,
        "fingerprint": cache_info.fingerprint,
        "cache_path": str(cache_info.cache_path) if cache_info.cache_path else "",
        "metadata_path": str(cache_info.metadata_path) if cache_info.metadata_path else "",
        "source_file_count": cache_info.source_file_count,
    }


def collect_artist_labels(*, prepared) -> list[str]:
    artist_label_frame = (
        prepared.df[["artist_label", "master_metadata_album_artist_name"]]
        .drop_duplicates(subset=["artist_label"])
        .sort_values("artist_label")
    )
    return artist_label_frame["master_metadata_album_artist_name"].astype(str).tolist()


def write_feature_metadata(
    *,
    artifact_paths: list[Path],
    artist_labels: list[str],
    config: PipelineConfig,
    deps: PipelineRuntimeDeps,
    prepared,
    run_dir: Path,
) -> None:
    metadata_path = run_dir / "feature_metadata.json"
    _write_json_artifact(
        metadata_path,
        {
            "sequence_length": config.sequence_length,
            "context_features": list(prepared.context_features),
            "skew_context_features": list(deps.SKEW_CONTEXT_FEATURES),
            "artist_labels": artist_labels,
        },
        artifact_paths,
    )


__all__ = [
    "build_cache_info_payload",
    "collect_artist_labels",
    "write_feature_metadata",
]
