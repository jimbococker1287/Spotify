from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import time

import joblib
import pandas as pd

from .data_preparation import CONTEXT_FEATURES
from .data_preparation import SKEW_CONTEXT_FEATURES
from .data_preparation import TECHNICAL_LOG_FILENAMES
from .data_preparation import PreparedData
from .data_preparation import _load_json_records
from .data_preparation import _rolling_artist_counts
from .data_preparation import _rolling_artist_counts_multi
from .data_preparation import append_audio_features as _append_audio_features_impl
from .data_preparation import append_technical_log_features as _append_technical_log_features_impl
from .data_preparation import engineer_features as _engineer_features_impl
from .data_preparation import prepare_training_data as _prepare_training_data_impl

CACHE_SCHEMA_VERSION = "prepared-data-v4"


@dataclass
class PreparedDataCacheInfo:
    enabled: bool
    hit: bool
    fingerprint: str
    cache_path: Path | None
    metadata_path: Path | None
    source_file_count: int


@dataclass(frozen=True)
class PreparedDataCachePaths:
    fingerprint: str
    cache_dir: Path
    bundle_path: Path
    metadata_path: Path


def _select_preferred_grouped_files(
    data_dir: Path,
    grouped_files: dict[Path, list[Path]],
    *,
    preferred_dir_name: str,
    logger,
    multi_dir_label: str,
    nested_label: str,
    count_label: str,
) -> tuple[Path, list[Path]]:
    preferred_dir = max(
        grouped_files,
        key=lambda parent: (
            len(grouped_files[parent]),
            parent == data_dir,
            parent.name.lower() == preferred_dir_name,
            str(parent),
        ),
    )
    selected = list(grouped_files[preferred_dir])
    if len(grouped_files) > 1:
        logger.warning(
            "Found %s in multiple directories under %s. Using %s (%d %s).",
            multi_dir_label,
            data_dir,
            preferred_dir,
            len(selected),
            count_label,
        )
    elif preferred_dir != data_dir:
        logger.info("Using nested %s export folder: %s", nested_label, preferred_dir)
    return preferred_dir, selected


def discover_streaming_files(data_dir: Path, include_video: bool, logger) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()

    grouped_audio_files: dict[Path, list[Path]] = {}
    for path in sorted(candidate for candidate in data_dir.rglob("Streaming_History_Audio_*.json") if candidate.is_file()):
        grouped_audio_files.setdefault(path.parent, []).append(path)

    if grouped_audio_files:
        preferred_dir, discovered = _select_preferred_grouped_files(
            data_dir,
            grouped_audio_files,
            preferred_dir_name="spotify extended streaming history",
            logger=logger,
            multi_dir_label="streaming history files",
            nested_label="streaming history",
            count_label="audio files",
        )
        if include_video:
            discovered.extend(
                sorted(candidate for candidate in preferred_dir.glob("Streaming_History_Video_*.json") if candidate.is_file())
            )
        return discovered

    raise FileNotFoundError(
        f"No streaming history JSON files found in {data_dir}. "
        "Expected files named Streaming_History_Audio_*.json directly in that directory "
        "or inside a nested Spotify export folder."
    )


def discover_technical_log_files(data_dir: Path, logger) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()

    grouped_files: dict[Path, list[Path]] = {}
    target_names = set(TECHNICAL_LOG_FILENAMES)
    for path in sorted(candidate for candidate in data_dir.rglob("*.json") if candidate.is_file() and candidate.name in target_names):
        parent_name = path.parent.name.lower()
        if "technical log information" not in parent_name:
            continue
        grouped_files.setdefault(path.parent, []).append(path)

    if not grouped_files:
        logger.info("No Spotify technical-log export found under %s; using zero-filled technical context features.", data_dir)
        return []

    preferred_dir, selected = _select_preferred_grouped_files(
        data_dir,
        grouped_files,
        preferred_dir_name="spotify technical log information",
        logger=logger,
        multi_dir_label="technical log files",
        nested_label="technical log",
        count_label="files",
    )
    discovered_by_name = {path.name: path for path in selected}
    discovered = [discovered_by_name[name] for name in TECHNICAL_LOG_FILENAMES if name in discovered_by_name]
    logger.info("Discovered %d technical log files for feature augmentation.", len(discovered))
    return discovered


def load_streaming_history(data_dir: Path, include_video: bool, logger) -> pd.DataFrame:
    files = discover_streaming_files(data_dir, include_video, logger)
    all_records: list[dict] = []
    fast_json = None
    try:
        import orjson as fast_json  # type: ignore
        logger.info("Using orjson for faster streaming-history parsing.")
    except Exception:
        fast_json = None

    for path in files:
        records = _load_json_records(path, fast_json)
        logger.info("Loaded %s with %d records", path.name, len(records))
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("Streaming history files were found, but no records were loaded.")

    df = pd.DataFrame(all_records)
    logger.info("Concatenated data shape: %s", df.shape)
    return df


def _cache_enabled_from_env() -> bool:
    raw = os.getenv("SPOTIFY_CACHE_PREPARED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _build_prepared_cache_fingerprint(
    files: list[Path],
    technical_files: list[Path],
    include_video: bool,
    enable_spotify_features: bool,
    max_artists: int,
    sequence_length: int,
) -> tuple[str, dict[str, object]]:
    payload_files: list[dict[str, object]] = []
    for path in sorted(files, key=lambda item: str(item.resolve())):
        stat = path.stat()
        payload_files.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            }
        )
    payload_technical_files: list[dict[str, object]] = []
    for path in sorted(technical_files, key=lambda item: str(item.resolve())):
        stat = path.stat()
        payload_technical_files.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            }
        )

    fingerprint_payload: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "include_video": bool(include_video),
        "enable_spotify_features": bool(enable_spotify_features),
        "max_artists": int(max_artists),
        "sequence_length": int(sequence_length),
        "context_features": list(CONTEXT_FEATURES),
        "skew_context_features": list(SKEW_CONTEXT_FEATURES),
        "files": payload_files,
        "technical_files": payload_technical_files,
    }
    serialized = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return fingerprint, fingerprint_payload


def _resolve_prepared_cache_paths(cache_root: Path, fingerprint: str) -> PreparedDataCachePaths:
    cache_dir = (cache_root / fingerprint).resolve()
    return PreparedDataCachePaths(
        fingerprint=fingerprint,
        cache_dir=cache_dir,
        bundle_path=cache_dir / "prepared_bundle.joblib",
        metadata_path=cache_dir / "cache_meta.json",
    )


def _load_prepared_cache(
    *,
    cache_paths: PreparedDataCachePaths,
    scaler_path: Path,
    source_file_count: int,
    logger,
) -> tuple[PreparedData, PreparedDataCacheInfo] | None:
    if not cache_paths.bundle_path.exists():
        return None

    try:
        payload = joblib.load(cache_paths.bundle_path)
        prepared = payload.get("prepared")
        scaler = payload.get("scaler")
        if not isinstance(prepared, PreparedData):
            raise TypeError("cached payload has unexpected prepared object type")
        if scaler is not None:
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
        logger.info("Prepared-data cache hit: %s", cache_paths.bundle_path)
        return prepared, PreparedDataCacheInfo(
            enabled=True,
            hit=True,
            fingerprint=cache_paths.fingerprint,
            cache_path=cache_paths.bundle_path,
            metadata_path=(cache_paths.metadata_path if cache_paths.metadata_path.exists() else None),
            source_file_count=source_file_count,
        )
    except Exception as exc:
        logger.warning("Prepared-data cache load failed (%s). Rebuilding cache.", exc)
        return None


def _save_prepared_cache(
    *,
    cache_paths: PreparedDataCachePaths,
    prepared: PreparedData,
    scaler_path: Path,
    fingerprint_payload: dict[str, object],
    logger,
) -> None:
    try:
        scaler = joblib.load(scaler_path)
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"prepared": prepared, "scaler": scaler}, cache_paths.bundle_path, compress=3)
        cache_paths.metadata_path.write_text(
            json.dumps(
                {
                    "fingerprint": cache_paths.fingerprint,
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "created_at_epoch_s": int(time.time()),
                    "fingerprint_payload": fingerprint_payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Prepared-data cache saved: %s", cache_paths.bundle_path)
    except Exception as exc:
        logger.warning("Prepared-data cache save skipped due to error: %s", exc)


def load_or_prepare_training_data(
    *,
    data_dir: Path,
    include_video: bool,
    enable_spotify_features: bool,
    max_artists: int,
    sequence_length: int,
    scaler_path: Path,
    cache_root: Path,
    raw_df: pd.DataFrame | None = None,
    logger,
) -> tuple[PreparedData, PreparedDataCacheInfo]:
    files = discover_streaming_files(data_dir, include_video, logger)
    technical_files = discover_technical_log_files(data_dir, logger)
    source_file_count = len(files) + len(technical_files)
    fingerprint, fingerprint_payload = _build_prepared_cache_fingerprint(
        files=files,
        technical_files=technical_files,
        include_video=include_video,
        enable_spotify_features=enable_spotify_features,
        max_artists=max_artists,
        sequence_length=sequence_length,
    )

    cache_enabled = _cache_enabled_from_env()
    cache_paths = _resolve_prepared_cache_paths(cache_root, fingerprint)

    if cache_enabled:
        cached = _load_prepared_cache(
            cache_paths=cache_paths,
            scaler_path=scaler_path,
            source_file_count=source_file_count,
            logger=logger,
        )
        if cached is not None:
            return cached

    df = raw_df.copy() if raw_df is not None else load_streaming_history(data_dir, include_video, logger)
    df = engineer_features(df, max_artists, logger)
    df = append_technical_log_features(df, data_dir=data_dir, logger=logger, technical_files=technical_files)
    df = append_audio_features(df, enable_spotify_features, logger)
    prepared = prepare_training_data(
        df=df,
        sequence_length=sequence_length,
        scaler_path=scaler_path,
        logger=logger,
    )

    if cache_enabled:
        _save_prepared_cache(
            cache_paths=cache_paths,
            prepared=prepared,
            scaler_path=scaler_path,
            fingerprint_payload=fingerprint_payload,
            logger=logger,
        )

    return prepared, PreparedDataCacheInfo(
        enabled=cache_enabled,
        hit=False,
        fingerprint=fingerprint,
        cache_path=(cache_paths.bundle_path if cache_enabled else None),
        metadata_path=(cache_paths.metadata_path if cache_enabled else None),
        source_file_count=source_file_count,
    )


def append_technical_log_features(
    df: pd.DataFrame,
    *,
    data_dir: Path,
    logger,
    technical_files: list[Path] | None = None,
) -> pd.DataFrame:
    return _append_technical_log_features_impl(
        df,
        data_dir=data_dir,
        logger=logger,
        technical_files=technical_files,
        discover_technical_log_files_fn=discover_technical_log_files,
    )


def append_audio_features(df: pd.DataFrame, enable_spotify_features: bool, logger) -> pd.DataFrame:
    return _append_audio_features_impl(df, enable_spotify_features, logger)


def engineer_features(
    df: pd.DataFrame,
    max_artists: int,
    logger,
    artist_classes: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return _engineer_features_impl(
        df,
        max_artists=max_artists,
        logger=logger,
        artist_classes=artist_classes,
    )


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int,
    scaler_path: Path,
    logger,
) -> PreparedData:
    return _prepare_training_data_impl(
        df=df,
        sequence_length=sequence_length,
        scaler_path=scaler_path,
        logger=logger,
    )
