from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .run_artifacts import copy_file_if_changed, safe_read_json, write_json


PHASE_ARTIFACT_CACHE_SCHEMA_VERSION = "phase-artifact-cache-v1"


@dataclass(frozen=True)
class PhaseArtifactCachePaths:
    cache_key: str
    cache_dir: Path
    manifest_path: Path
    artifacts_dir: Path


def phase_artifact_cache_enabled(*, env_var: str = "SPOTIFY_CACHE_PHASE_ARTIFACTS", default: str = "1") -> bool:
    import os

    raw = os.getenv(env_var, default).strip().lower()
    return raw not in ("0", "false", "no", "off")


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_ready(item_method())
        except Exception:
            return str(value)
    return str(value)


def stable_payload_digest(payload: Any) -> str:
    serialized = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def source_digest_for_paths(paths: list[Path] | tuple[Path, ...]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        hasher.update(str(resolved).encode("utf-8"))
        try:
            hasher.update(resolved.read_bytes())
        except Exception:
            continue
    return hasher.hexdigest()[:24]


def source_digest_for_callables(*callables: Any) -> str:
    paths: list[Path] = []
    for item in callables:
        code = getattr(item, "__code__", None)
        filename = getattr(code, "co_filename", "")
        text = str(filename).strip()
        if not text:
            continue
        paths.append(Path(text))
    return source_digest_for_paths(tuple(paths))


def resolve_phase_artifact_cache_paths(
    *,
    cache_root: Path,
    namespace: str,
    phase_name: str,
    cache_key: str,
) -> PhaseArtifactCachePaths:
    cache_dir = (cache_root / namespace / phase_name / cache_key).resolve()
    return PhaseArtifactCachePaths(
        cache_key=cache_key,
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.json",
        artifacts_dir=cache_dir / "artifacts",
    )


def restore_phase_artifact_cache(
    *,
    cache_paths: PhaseArtifactCachePaths,
    run_dir: Path,
    logger,
) -> list[Path] | None:
    payload = safe_read_json(cache_paths.manifest_path, default=None)
    if not isinstance(payload, dict):
        return None
    rel_paths = payload.get("artifact_rel_paths", [])
    if not isinstance(rel_paths, list):
        return None
    restored: list[Path] = []
    try:
        for rel_path in rel_paths:
            rel_text = str(rel_path).strip()
            if not rel_text:
                continue
            source = cache_paths.artifacts_dir / rel_text
            if not source.exists():
                return None
            destination = run_dir / rel_text
            copy_file_if_changed(source, destination)
            restored.append(destination)
    except Exception as exc:
        logger.warning("Phase artifact cache restore failed for %s (%s). Rebuilding.", cache_paths.cache_dir, exc)
        return None
    return restored


def save_phase_artifact_cache(
    *,
    cache_paths: PhaseArtifactCachePaths,
    cache_payload: dict[str, object],
    run_dir: Path,
    artifact_paths: list[Path],
    logger,
) -> None:
    rel_paths: list[str] = []
    seen: set[str] = set()
    for path in artifact_paths:
        if not path.exists():
            continue
        try:
            rel = path.resolve().relative_to(run_dir.resolve())
        except Exception:
            continue
        rel_text = str(rel)
        if rel_text in seen:
            continue
        seen.add(rel_text)
        rel_paths.append(rel_text)

    try:
        cache_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        for rel_text in rel_paths:
            source = run_dir / rel_text
            destination = cache_paths.artifacts_dir / rel_text
            copy_file_if_changed(source, destination)
        write_json(
            cache_paths.manifest_path,
            {
                "schema_version": PHASE_ARTIFACT_CACHE_SCHEMA_VERSION,
                "cache_key": cache_paths.cache_key,
                "cache_payload": _json_ready(cache_payload),
                "artifact_rel_paths": rel_paths,
            },
            sort_keys=True,
        )
    except Exception as exc:
        logger.warning("Phase artifact cache save failed for %s (%s).", cache_paths.cache_dir, exc)


__all__ = [
    "PHASE_ARTIFACT_CACHE_SCHEMA_VERSION",
    "PhaseArtifactCachePaths",
    "phase_artifact_cache_enabled",
    "resolve_phase_artifact_cache_paths",
    "restore_phase_artifact_cache",
    "save_phase_artifact_cache",
    "source_digest_for_callables",
    "source_digest_for_paths",
    "stable_payload_digest",
]
