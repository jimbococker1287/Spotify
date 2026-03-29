from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

CONTEXT_FEATURES: tuple[str, ...] = (
    "hour",
    "dayofweek",
    "month",
    "platform_code",
    "reason_start_code",
    "reason_end_code",
    "shuffle",
    "skipped",
    "offline",
    "incognito_mode",
    "danceability",
    "energy",
    "tempo",
    "time_diff",
    "session_position",
    "session_elapsed_seconds",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "artist_play_count",
    "days_since_last",
    "hours_since_last_artist",
    "skip_streak",
    "listen_streak",
    "artist_play_count_24h",
    "artist_play_count_7d",
    "artist_freq_smooth",
    "plays_since_last_artist",
    "artist_session_play_count",
    "session_skip_rate_so_far",
    "session_unique_artists_so_far",
    "session_repeat_ratio_so_far",
    "is_artist_repeat_from_prev",
    "transition_repeat_count",
    "recent_skip_rate_5",
    "recent_skip_rate_20",
    "recent_artist_unique_ratio_5",
    "recent_artist_unique_ratio_20",
    "artist_hour_rate_smooth",
    "artist_dow_rate_smooth",
    "prev_artist_transition_rate_smooth",
    "artist_skip_rate_hist",
    "artist_skip_rate_smooth",
    "tech_connection_events_1h",
    "tech_connection_none_24h",
    "tech_ipv6_failures_24h",
    "tech_playback_errors_24h",
    "tech_playback_fatal_errors_24h",
    "tech_stutter_events_24h",
    "tech_track_not_played_24h",
    "tech_cloud_stats_events_24h",
    "tech_cloud_stalls_24h",
    "tech_last_reachability_wifi",
    "tech_last_reachability_cellular",
    "tech_last_reachability_offline",
    "tech_last_ipv6_failed",
    "tech_allow_downgrade",
    "tech_bitrate_wifi_kbps",
    "tech_bitrate_cellular_kbps",
)

SKEW_CONTEXT_FEATURES: tuple[str, ...] = (
    "time_diff",
    "session_position",
    "session_elapsed_seconds",
    "artist_play_count",
    "days_since_last",
    "hours_since_last_artist",
    "skip_streak",
    "listen_streak",
    "artist_play_count_24h",
    "artist_play_count_7d",
    "artist_freq_smooth",
    "plays_since_last_artist",
    "artist_session_play_count",
    "session_skip_rate_so_far",
    "session_unique_artists_so_far",
    "transition_repeat_count",
    "tech_connection_events_1h",
    "tech_connection_none_24h",
    "tech_ipv6_failures_24h",
    "tech_playback_errors_24h",
    "tech_playback_fatal_errors_24h",
    "tech_stutter_events_24h",
    "tech_track_not_played_24h",
    "tech_cloud_stats_events_24h",
    "tech_cloud_stalls_24h",
)

CACHE_SCHEMA_VERSION = "prepared-data-v4"
TECHNICAL_LOG_FILENAMES: tuple[str, ...] = (
    "ConnectionInfo.json",
    "AudioStreamingSettingsReport.json",
    "PlaybackError.json",
    "Stutter.json",
    "TrackNotPlayed.json",
    "CloudPlaybackPlaybackStats.json",
)


@dataclass
class PreparedData:
    df: pd.DataFrame
    context_features: list[str]

    X_seq_train: np.ndarray
    X_seq_val: np.ndarray
    X_seq_test: np.ndarray

    X_ctx_train: np.ndarray
    X_ctx_val: np.ndarray
    X_ctx_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    y_skip_train: np.ndarray
    y_skip_val: np.ndarray
    y_skip_test: np.ndarray

    num_artists: int
    num_ctx: int


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


def _ensure_column(df: pd.DataFrame, column: str, default_value):
    if column not in df.columns:
        df[column] = default_value


def _rolling_artist_counts(
    ts_seconds: np.ndarray,
    artists: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    return _rolling_artist_counts_multi(ts_seconds, artists, window_seconds=(window_seconds,))[0]


def _rolling_artist_counts_multi(
    ts_seconds: np.ndarray,
    artists: np.ndarray,
    *,
    window_seconds: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    normalized_windows = tuple(max(1, int(window)) for window in window_seconds)
    counts = tuple(np.zeros(len(artists), dtype="float32") for _ in normalized_windows)
    buffers: dict[object, list[deque[int]]] = {}

    for idx, (ts_value, artist_key) in enumerate(zip(ts_seconds, artists)):
        artist_buffers = buffers.get(artist_key)
        if artist_buffers is None:
            artist_buffers = [deque() for _ in normalized_windows]
            buffers[artist_key] = artist_buffers

        ts_int = int(ts_value)
        for out, bucket, window in zip(counts, artist_buffers, normalized_windows):
            threshold = ts_int - window
            while bucket and bucket[0] < threshold:
                bucket.popleft()
            out[idx] = float(len(bucket))
            bucket.append(ts_int)

    return counts


def _ensure_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in df.columns:
        return df.reset_index(drop=True)
    ts_values = df["ts"]
    if ts_values.is_monotonic_increasing:
        if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
            return df
        return df.reset_index(drop=True)
    return df.sort_values("ts").reset_index(drop=True)


def _load_json_records(path: Path, fast_json) -> list[dict]:
    if fast_json is not None:
        payload = fast_json.loads(path.read_bytes())
    else:
        with path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
    if isinstance(payload, list):
        return payload
    return []


def _coerce_epoch_ms(value) -> int | None:
    if value is None:
        return None
    try:
        epoch = int(float(value))
    except (TypeError, ValueError):
        return None
    if abs(epoch) < 10_000_000_000:
        epoch *= 1000
    return epoch


def _normalize_device_family(value) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null", "unknown", "not_applicable"}:
        return "other"
    if any(token in text for token in ("ios", "iphone", "ipad", "android", "mobile")):
        return "mobile"
    if any(token in text for token in ("osx", "os x", "mac", "desktop", "windows", "linux", "web_player", "web player")):
        return "desktop"
    return "other"


def _parse_bool_flag(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    return 1.0 if text in {"1", "true", "yes", "y", "on"} else 0.0


def _windowed_sums_for_all_rows(
    row_ts_ms: np.ndarray,
    event_ts_ms: np.ndarray,
    event_values: np.ndarray,
    *,
    window_ms: int,
) -> np.ndarray:
    if len(row_ts_ms) == 0 or len(event_ts_ms) == 0:
        return np.zeros(len(row_ts_ms), dtype="float32")

    order = np.argsort(event_ts_ms, kind="mergesort")
    sorted_ts = event_ts_ms[order]
    sorted_values = event_values[order]
    cumulative = np.concatenate(([0.0], np.cumsum(sorted_values, dtype="float64")))
    right = np.searchsorted(sorted_ts, row_ts_ms, side="right")
    left = np.searchsorted(sorted_ts, row_ts_ms - window_ms, side="left")
    return (cumulative[right] - cumulative[left]).astype("float32", copy=False)


def _windowed_sums_by_family(
    row_ts_ms: np.ndarray,
    row_families: np.ndarray,
    event_ts_ms: np.ndarray,
    event_families: np.ndarray | None,
    event_values: np.ndarray,
    *,
    window_ms: int,
) -> np.ndarray:
    if event_families is None:
        return _windowed_sums_for_all_rows(row_ts_ms, event_ts_ms, event_values, window_ms=window_ms)

    out = np.zeros(len(row_ts_ms), dtype="float32")
    for family in np.unique(row_families):
        row_mask = row_families == family
        family_events = event_families == family
        if not np.any(family_events):
            continue
        out[row_mask] = _windowed_sums_for_all_rows(
            row_ts_ms[row_mask],
            event_ts_ms[family_events],
            event_values[family_events],
            window_ms=window_ms,
        )
    return out


def _latest_values_for_all_rows(
    row_ts_ms: np.ndarray,
    event_ts_ms: np.ndarray,
    event_values: np.ndarray,
    *,
    default_value: float = 0.0,
) -> np.ndarray:
    out = np.full(len(row_ts_ms), float(default_value), dtype="float32")
    if len(row_ts_ms) == 0 or len(event_ts_ms) == 0:
        return out

    order = np.argsort(event_ts_ms, kind="mergesort")
    sorted_ts = event_ts_ms[order]
    sorted_values = event_values[order]
    positions = np.searchsorted(sorted_ts, row_ts_ms, side="right") - 1
    valid = positions >= 0
    if np.any(valid):
        out[valid] = sorted_values[positions[valid]].astype("float32", copy=False)
    return out


def _latest_values_by_family(
    row_ts_ms: np.ndarray,
    row_families: np.ndarray,
    event_ts_ms: np.ndarray,
    event_families: np.ndarray | None,
    event_values: np.ndarray,
    *,
    default_value: float = 0.0,
) -> np.ndarray:
    if event_families is None:
        return _latest_values_for_all_rows(row_ts_ms, event_ts_ms, event_values, default_value=default_value)

    out = np.full(len(row_ts_ms), float(default_value), dtype="float32")
    for family in np.unique(row_families):
        row_mask = row_families == family
        family_events = event_families == family
        if not np.any(family_events):
            continue
        out[row_mask] = _latest_values_for_all_rows(
            row_ts_ms[row_mask],
            event_ts_ms[family_events],
            event_values[family_events],
            default_value=default_value,
        )
    return out


def append_technical_log_features(
    df: pd.DataFrame,
    *,
    data_dir: Path,
    logger,
    technical_files: list[Path] | None = None,
) -> pd.DataFrame:
    defaults = {
        "tech_connection_events_1h": 0.0,
        "tech_connection_none_24h": 0.0,
        "tech_ipv6_failures_24h": 0.0,
        "tech_playback_errors_24h": 0.0,
        "tech_playback_fatal_errors_24h": 0.0,
        "tech_stutter_events_24h": 0.0,
        "tech_track_not_played_24h": 0.0,
        "tech_cloud_stats_events_24h": 0.0,
        "tech_cloud_stalls_24h": 0.0,
        "tech_last_reachability_wifi": 0.0,
        "tech_last_reachability_cellular": 0.0,
        "tech_last_reachability_offline": 0.0,
        "tech_last_ipv6_failed": 0.0,
        "tech_allow_downgrade": 0.0,
        "tech_bitrate_wifi_kbps": 0.0,
        "tech_bitrate_cellular_kbps": 0.0,
    }
    for column, default_value in defaults.items():
        df[column] = default_value

    if technical_files is None:
        technical_files = discover_technical_log_files(data_dir, logger)
    if not technical_files:
        return df

    try:
        import orjson as fast_json  # type: ignore
        logger.info("Using orjson for faster technical-log parsing.")
    except Exception:
        fast_json = None

    files_by_name = {path.name: path for path in technical_files}
    if "ts" not in df.columns or "platform" not in df.columns or df.empty:
        return df

    row_ts_ms = (pd.to_datetime(df["ts"], errors="coerce").astype("int64") // 10**6).to_numpy(dtype="int64", copy=False)
    row_families = np.array([_normalize_device_family(value) for value in df["platform"].tolist()], dtype=object)
    rows = len(df)
    day_ms = 24 * 60 * 60 * 1000
    hour_ms = 60 * 60 * 1000

    connection_path = files_by_name.get("ConnectionInfo.json")
    if connection_path is not None:
        connection_records = _load_json_records(connection_path, fast_json)
        logger.info("Loaded %s with %d technical records", connection_path.name, len(connection_records))
        connection_rows: list[tuple[int, str, float, float, float, float, float, float]] = []
        for record in connection_records:
            ts_ms = _coerce_epoch_ms(record.get("context_time") or record.get("timestamp_utc"))
            if ts_ms is None:
                continue
            family = _normalize_device_family(record.get("context_device_type") or record.get("context_os_name") or record.get("context_device_model"))
            reachability = str(record.get("message_reachability_type") or "").strip().lower()
            wifi = 1.0 if ("wlan" in reachability or "wifi" in reachability) else 0.0
            cellular = 1.0 if any(token in reachability for token in ("2g", "3g", "4g", "5g", "cell")) else 0.0
            offline = 1.0 if reachability in {"none", "offline", "unreachable"} else 0.0
            ipv6_failed = _parse_bool_flag(record.get("message_ipv6_failed"))
            connection_rows.append((ts_ms, family, 1.0, offline, ipv6_failed, wifi, cellular, offline))

        if connection_rows:
            connection_frame = pd.DataFrame(
                connection_rows,
                columns=[
                    "ts_ms",
                    "device_family",
                    "event_count",
                    "none_count",
                    "ipv6_failed_count",
                    "reachability_wifi",
                    "reachability_cellular",
                    "reachability_offline",
                ],
            )
            event_ts = connection_frame["ts_ms"].to_numpy(dtype="int64", copy=False)
            event_families = connection_frame["device_family"].to_numpy(dtype=object, copy=False)
            df["tech_connection_events_1h"] = _windowed_sums_by_family(
                row_ts_ms, row_families, event_ts, event_families, connection_frame["event_count"].to_numpy(dtype="float32", copy=False), window_ms=hour_ms
            )
            df["tech_connection_none_24h"] = _windowed_sums_by_family(
                row_ts_ms, row_families, event_ts, event_families, connection_frame["none_count"].to_numpy(dtype="float32", copy=False), window_ms=day_ms
            )
            df["tech_ipv6_failures_24h"] = _windowed_sums_by_family(
                row_ts_ms, row_families, event_ts, event_families, connection_frame["ipv6_failed_count"].to_numpy(dtype="float32", copy=False), window_ms=day_ms
            )
            df["tech_last_reachability_wifi"] = _latest_values_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                connection_frame["reachability_wifi"].to_numpy(dtype="float32", copy=False),
            )
            df["tech_last_reachability_cellular"] = _latest_values_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                connection_frame["reachability_cellular"].to_numpy(dtype="float32", copy=False),
            )
            df["tech_last_reachability_offline"] = _latest_values_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                connection_frame["reachability_offline"].to_numpy(dtype="float32", copy=False),
            )
            df["tech_last_ipv6_failed"] = _latest_values_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                connection_frame["ipv6_failed_count"].to_numpy(dtype="float32", copy=False),
            )

    settings_path = files_by_name.get("AudioStreamingSettingsReport.json")
    if settings_path is not None:
        settings_records = _load_json_records(settings_path, fast_json)
        logger.info("Loaded %s with %d technical records", settings_path.name, len(settings_records))
        settings_rows: list[tuple[int, str, float, float, float]] = []
        for record in settings_records:
            ts_ms = _coerce_epoch_ms(record.get("context_time") or record.get("timestamp_utc"))
            if ts_ms is None:
                continue
            family = _normalize_device_family(record.get("context_device_type") or record.get("context_os_name") or record.get("context_device_model"))
            wifi_bitrate = record.get("message_user_selected_play_bitrate_wifi")
            if wifi_bitrate is None:
                wifi_bitrate = record.get("message_play_bitrate_wifi")
            cellular_bitrate = record.get("message_user_selected_play_bitrate_cellular")
            if cellular_bitrate is None:
                cellular_bitrate = record.get("message_play_bitrate_cellular")
            settings_rows.append(
                (
                    ts_ms,
                    family,
                    _parse_bool_flag(record.get("message_allow_downgrade")),
                    float(wifi_bitrate or 0.0) / 1000.0,
                    float(cellular_bitrate or 0.0) / 1000.0,
                )
            )

        if settings_rows:
            settings_frame = pd.DataFrame(
                settings_rows,
                columns=["ts_ms", "device_family", "allow_downgrade", "bitrate_wifi_kbps", "bitrate_cellular_kbps"],
            )
            event_ts = settings_frame["ts_ms"].to_numpy(dtype="int64", copy=False)
            event_families = settings_frame["device_family"].to_numpy(dtype=object, copy=False)
            df["tech_allow_downgrade"] = _latest_values_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                settings_frame["allow_downgrade"].to_numpy(dtype="float32", copy=False),
            )
            df["tech_bitrate_wifi_kbps"] = _latest_values_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                settings_frame["bitrate_wifi_kbps"].to_numpy(dtype="float32", copy=False),
            )
            df["tech_bitrate_cellular_kbps"] = _latest_values_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                settings_frame["bitrate_cellular_kbps"].to_numpy(dtype="float32", copy=False),
            )

    for filename, output_name, fatal_name in (
        ("PlaybackError.json", "tech_playback_errors_24h", "tech_playback_fatal_errors_24h"),
        ("Stutter.json", "tech_stutter_events_24h", None),
        ("TrackNotPlayed.json", "tech_track_not_played_24h", None),
    ):
        path = files_by_name.get(filename)
        if path is None:
            continue
        records = _load_json_records(path, fast_json)
        logger.info("Loaded %s with %d technical records", path.name, len(records))
        rows_payload: list[tuple[int, str, float, float]] = []
        for record in records:
            ts_ms = _coerce_epoch_ms(record.get("context_time") or record.get("timestamp_utc"))
            if ts_ms is None:
                continue
            family = _normalize_device_family(record.get("context_device_type") or record.get("context_os_name") or record.get("context_device_model"))
            fatal_value = _parse_bool_flag(record.get("message_fatal")) if fatal_name else 0.0
            rows_payload.append((ts_ms, family, 1.0, fatal_value))
        if not rows_payload:
            continue

        event_frame = pd.DataFrame(rows_payload, columns=["ts_ms", "device_family", "event_count", "fatal_count"])
        event_ts = event_frame["ts_ms"].to_numpy(dtype="int64", copy=False)
        event_families = event_frame["device_family"].to_numpy(dtype=object, copy=False)
        df[output_name] = _windowed_sums_by_family(
            row_ts_ms,
            row_families,
            event_ts,
            event_families,
            event_frame["event_count"].to_numpy(dtype="float32", copy=False),
            window_ms=day_ms,
        )
        if fatal_name:
            df[fatal_name] = _windowed_sums_by_family(
                row_ts_ms,
                row_families,
                event_ts,
                event_families,
                event_frame["fatal_count"].to_numpy(dtype="float32", copy=False),
                window_ms=day_ms,
            )

    cloud_path = files_by_name.get("CloudPlaybackPlaybackStats.json")
    if cloud_path is not None:
        cloud_records = _load_json_records(cloud_path, fast_json)
        logger.info("Loaded %s with %d technical records", cloud_path.name, len(cloud_records))
        cloud_rows: list[tuple[int, float, float]] = []
        for record in cloud_records:
            ts_ms = _coerce_epoch_ms(record.get("context_time") or record.get("timestamp_utc") or record.get("message_client_timestamp"))
            if ts_ms is None:
                continue
            cloud_rows.append((ts_ms, 1.0, float(record.get("message_num_stalls") or 0.0)))
        if cloud_rows:
            cloud_frame = pd.DataFrame(cloud_rows, columns=["ts_ms", "event_count", "stall_count"])
            event_ts = cloud_frame["ts_ms"].to_numpy(dtype="int64", copy=False)
            df["tech_cloud_stats_events_24h"] = _windowed_sums_for_all_rows(
                row_ts_ms,
                event_ts,
                cloud_frame["event_count"].to_numpy(dtype="float32", copy=False),
                window_ms=day_ms,
            )
            df["tech_cloud_stalls_24h"] = _windowed_sums_for_all_rows(
                row_ts_ms,
                event_ts,
                cloud_frame["stall_count"].to_numpy(dtype="float32", copy=False),
                window_ms=day_ms,
            )

    for column in defaults:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype("float32")

    logger.info("Added %d technical context columns to %d listening rows.", len(defaults), rows)
    return df


def append_audio_features(df: pd.DataFrame, enable_spotify_features: bool, logger) -> pd.DataFrame:
    for column in ("danceability", "energy", "tempo"):
        df[column] = 0.0

    if not enable_spotify_features:
        logger.info("Spotify audio feature fetch disabled; filling audio feature columns with 0.0.")
        return df

    logger.warning(
        "Spotify audio feature fetch is disabled. Spotify's audio-features endpoint is deprecated and "
        "Spotify's Developer Policy does not allow Spotify content to train ML/AI models. "
        "Filling audio feature columns with 0.0 instead."
    )
    return df


def engineer_features(
    df: pd.DataFrame,
    max_artists: int,
    logger,
    artist_classes: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    _ensure_column(df, "ts", pd.NaT)
    _ensure_column(df, "master_metadata_album_artist_name", None)
    _ensure_column(df, "platform", "unknown")
    _ensure_column(df, "reason_start", "unknown")
    _ensure_column(df, "reason_end", "unknown")

    for col in ("shuffle", "skipped", "offline", "incognito_mode"):
        _ensure_column(df, col, 0)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df[df["ts"].notnull()]
    df = df[df["master_metadata_album_artist_name"].notna()]

    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month

    df = _ensure_time_sorted(df)
    df["time_diff"] = df["ts"].diff().dt.total_seconds().fillna(0)
    session_threshold = 30 * 60
    df["session_id"] = (df["time_diff"] > session_threshold).cumsum()
    session_group = df.groupby("session_id", sort=False)
    df["session_position"] = session_group.cumcount()
    session_start = session_group["ts"].transform("min")
    df["session_elapsed_seconds"] = (df["ts"] - session_start).dt.total_seconds().fillna(0.0).astype("float32")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    artist_name_group = df.groupby("master_metadata_album_artist_name", sort=False)
    df["artist_play_count"] = artist_name_group.cumcount()
    previous_artist_ts = artist_name_group["ts"].shift(1)
    df["hours_since_last_artist"] = (
        (df["ts"] - previous_artist_ts).dt.total_seconds().div(3600.0).fillna(0.0).clip(lower=0.0).astype("float32")
    )
    df["days_since_last"] = (df["hours_since_last_artist"] / 24.0).astype("float32")

    df["skip_flag"] = df["skipped"]
    streak_grp = (df["skip_flag"] != df["skip_flag"].shift()).cumsum()
    streak_counts = df.groupby(streak_grp, sort=False)["skip_flag"].cumcount()
    df["skip_streak"] = streak_counts.where(df["skip_flag"] == 1, 0)
    df["listen_streak"] = streak_counts.where(df["skip_flag"] == 0, 0)

    df["platform_code"] = df["platform"].astype("category").cat.codes
    df["reason_start_code"] = df["reason_start"].astype("category").cat.codes
    df["reason_end_code"] = df["reason_end"].astype("category").cat.codes

    for col in ("shuffle", "skipped", "offline", "incognito_mode"):
        df[col] = pd.Series(df[col], dtype="boolean").fillna(False).astype("int8")

    if artist_classes is None:
        top_artists = (
            df["master_metadata_album_artist_name"]
            .value_counts()
            .nlargest(max_artists)
            .index
            .tolist()
        )
        df = df[df["master_metadata_album_artist_name"].isin(top_artists)].copy()
        encoder = LabelEncoder()
        df["artist_label"] = encoder.fit_transform(df["master_metadata_album_artist_name"])
    else:
        top_artists = [str(item) for item in artist_classes]
        label_map = {artist_name: idx for idx, artist_name in enumerate(top_artists)}
        df = df[df["master_metadata_album_artist_name"].isin(top_artists)].copy()
        df["artist_label"] = df["master_metadata_album_artist_name"].map(label_map).astype("int32")

    df = df.reset_index(drop=True)

    artist_labels = df["artist_label"].to_numpy(dtype="int32", copy=False)
    ts_seconds = (df["ts"].astype("int64") // 10**9).to_numpy(dtype="int64")

    artist_play_count_24h, artist_play_count_7d = _rolling_artist_counts_multi(
        ts_seconds,
        artist_labels,
        window_seconds=(24 * 60 * 60, 7 * 24 * 60 * 60),
    )
    df["artist_play_count_24h"] = artist_play_count_24h
    df["artist_play_count_7d"] = artist_play_count_7d

    alpha = 1.0
    vocab_size = max(1, len(top_artists))
    row_index = np.arange(len(df), dtype="float32")
    artist_seen = df["artist_play_count"].to_numpy(dtype="float32", copy=False)
    artist_freq_smooth = (artist_seen + alpha) / (row_index + alpha * float(vocab_size))
    df["artist_freq_smooth"] = artist_freq_smooth

    session_group = df.groupby("session_id", sort=False)
    artist_group = df.groupby("artist_label", sort=False)
    session_artist_group = df.groupby(["session_id", "artist_label"], sort=False)

    prev_seen_idx = pd.Series(np.arange(len(df), dtype="float32"), index=df.index).groupby(df["artist_label"], sort=False).shift(1)
    prev_seen_vals = prev_seen_idx.to_numpy(dtype="float32", copy=False)
    plays_since_last_artist = np.where(
        np.isnan(prev_seen_vals),
        row_index + 1.0,
        row_index - prev_seen_vals,
    ).astype("float32")
    df["plays_since_last_artist"] = plays_since_last_artist

    df["artist_session_play_count"] = session_artist_group.cumcount().astype("float32")

    skip_values = df["skipped"].to_numpy(dtype="float32", copy=False)
    session_position = df["session_position"].to_numpy(dtype="float32", copy=False)
    session_skip_cumsum = session_group["skipped"].cumsum().to_numpy(dtype="float32", copy=False)
    prior_session_skip = session_skip_cumsum - skip_values
    session_skip_rate_so_far = np.divide(
        prior_session_skip,
        session_position,
        out=np.zeros(len(df), dtype="float32"),
        where=session_position > 0,
    )
    df["session_skip_rate_so_far"] = session_skip_rate_so_far

    first_in_session_artist = (~df.duplicated(["session_id", "artist_label"])).astype("int8")
    first_in_session_vals = first_in_session_artist.to_numpy(dtype="int8", copy=False)
    session_unique_cumsum = first_in_session_artist.groupby(df["session_id"], sort=False).cumsum().to_numpy(dtype="int32", copy=False)
    session_unique_artists_so_far = (session_unique_cumsum - first_in_session_vals).astype("float32")
    session_repeat_ratio_so_far = np.divide(
        session_position - session_unique_artists_so_far,
        session_position,
        out=np.zeros(len(df), dtype="float32"),
        where=session_position > 0,
    )
    df["session_unique_artists_so_far"] = session_unique_artists_so_far
    df["session_repeat_ratio_so_far"] = session_repeat_ratio_so_far

    repeat_from_prev = (
        (df["master_metadata_album_artist_name"] == df["master_metadata_album_artist_name"].shift(1))
        & (df["session_id"] == df["session_id"].shift(1))
    )
    df["is_artist_repeat_from_prev"] = repeat_from_prev.fillna(False).astype("int8")

    transition_repeat_count = np.zeros(len(df), dtype="float32")
    prev_artist_transition_rate_smooth = np.zeros(len(df), dtype="float32")
    session_ids = df["session_id"]
    valid_transition_mask = session_ids.eq(session_ids.shift(1)).to_numpy(dtype=bool, copy=False)
    if np.any(valid_transition_mask):
        valid_prev = df["artist_label"].shift(1)[valid_transition_mask].astype("int32")
        valid_curr = df.loc[valid_transition_mask, "artist_label"].astype("int32")
        pair_frame = pd.DataFrame(
            {
                "prev_artist_label": valid_prev.to_numpy(copy=False),
                "artist_label": valid_curr.to_numpy(copy=False),
            }
        )
        pair_counts = pair_frame.groupby(["prev_artist_label", "artist_label"], sort=False).cumcount().to_numpy(dtype="float32", copy=False)
        transition_repeat_count[valid_transition_mask] = pair_counts

        outgoing_counts = valid_prev.groupby(valid_prev, sort=False).cumcount().to_numpy(dtype="float32", copy=False)
        prev_artist_transition_rate_smooth[valid_transition_mask] = (
            (pair_counts + 1.0) / (outgoing_counts + float(vocab_size))
        ).astype("float32", copy=False)
    df["transition_repeat_count"] = transition_repeat_count

    artist_hour_seen = df.groupby(["artist_label", "hour"], sort=False).cumcount().to_numpy(dtype="float32", copy=False)
    artist_dow_seen = df.groupby(["artist_label", "dayofweek"], sort=False).cumcount().to_numpy(dtype="float32", copy=False)
    artist_hour_rate_smooth = ((artist_hour_seen + 1.0) / (artist_seen + 24.0)).astype("float32", copy=False)
    artist_dow_rate_smooth = ((artist_dow_seen + 1.0) / (artist_seen + 7.0)).astype("float32", copy=False)

    def _trailing_mean_before(values: np.ndarray, width: int) -> np.ndarray:
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype="float64")))
        idx = np.arange(len(values), dtype="int64")
        starts = np.maximum(0, idx - width)
        totals = cumulative[idx] - cumulative[starts]
        counts = idx - starts
        out = np.zeros(len(values), dtype="float32")
        valid = counts > 0
        out[valid] = (totals[valid] / counts[valid]).astype("float32", copy=False)
        return out

    recent_skip_rate_5 = _trailing_mean_before(skip_values, 5)
    recent_skip_rate_20 = _trailing_mean_before(skip_values, 20)
    recent_artist_unique_ratio_5 = np.zeros(len(df), dtype="float32")
    recent_artist_unique_ratio_20 = np.zeros(len(df), dtype="float32")
    recent_artist_window_5: deque[int] = deque(maxlen=5)
    recent_artist_window_20: deque[int] = deque(maxlen=20)
    recent_artist_counts_5: dict[int, int] = {}
    recent_artist_counts_20: dict[int, int] = {}
    for idx, artist_label in enumerate(artist_labels):
        artist_key = int(artist_label)
        recent_artist_unique_ratio_5[idx] = (
            float(len(recent_artist_counts_5) / len(recent_artist_window_5)) if recent_artist_window_5 else 0.0
        )
        recent_artist_unique_ratio_20[idx] = (
            float(len(recent_artist_counts_20) / len(recent_artist_window_20)) if recent_artist_window_20 else 0.0
        )

        if len(recent_artist_window_5) == recent_artist_window_5.maxlen:
            evicted_5 = recent_artist_window_5.popleft()
            next_count_5 = recent_artist_counts_5[evicted_5] - 1
            if next_count_5 <= 0:
                del recent_artist_counts_5[evicted_5]
            else:
                recent_artist_counts_5[evicted_5] = next_count_5
        recent_artist_window_5.append(artist_key)
        recent_artist_counts_5[artist_key] = recent_artist_counts_5.get(artist_key, 0) + 1

        if len(recent_artist_window_20) == recent_artist_window_20.maxlen:
            evicted_20 = recent_artist_window_20.popleft()
            next_count_20 = recent_artist_counts_20[evicted_20] - 1
            if next_count_20 <= 0:
                del recent_artist_counts_20[evicted_20]
            else:
                recent_artist_counts_20[evicted_20] = next_count_20
        recent_artist_window_20.append(artist_key)
        recent_artist_counts_20[artist_key] = recent_artist_counts_20.get(artist_key, 0) + 1

    df["recent_skip_rate_5"] = recent_skip_rate_5
    df["recent_skip_rate_20"] = recent_skip_rate_20
    df["recent_artist_unique_ratio_5"] = recent_artist_unique_ratio_5
    df["recent_artist_unique_ratio_20"] = recent_artist_unique_ratio_20
    df["artist_hour_rate_smooth"] = artist_hour_rate_smooth
    df["artist_dow_rate_smooth"] = artist_dow_rate_smooth
    df["prev_artist_transition_rate_smooth"] = prev_artist_transition_rate_smooth

    alpha_skip = 1.0
    beta_skip = 1.0
    artist_skip_cumsum = artist_group["skipped"].cumsum().to_numpy(dtype="float32", copy=False)
    prior_artist_skip = artist_skip_cumsum - skip_values
    artist_skip_rate_hist = np.divide(
        prior_artist_skip,
        artist_seen,
        out=np.zeros(len(df), dtype="float32"),
        where=artist_seen > 0,
    )
    artist_skip_rate_smooth = ((prior_artist_skip + alpha_skip) / (artist_seen + alpha_skip + beta_skip)).astype("float32", copy=False)
    df["artist_skip_rate_hist"] = artist_skip_rate_hist
    df["artist_skip_rate_smooth"] = artist_skip_rate_smooth

    logger.info(
        "After filtering to top %d artists, training frame shape is %s",
        len(top_artists),
        df.shape,
    )

    return df


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int,
    scaler_path: Path,
    logger,
) -> PreparedData:
    context_features = list(CONTEXT_FEATURES)

    df = _ensure_time_sorted(df)
    missing_context = [key for key in context_features if key not in df.columns]
    if missing_context:
        raise RuntimeError(f"Missing engineered context features: {', '.join(missing_context)}")

    labels = df["artist_label"].to_numpy(dtype="int32", copy=False)
    context_vals = df[context_features].to_numpy(dtype="float32", copy=False)
    skipped_vals = df["skipped"].to_numpy(dtype="float32", copy=False)

    if len(labels) <= sequence_length + 1:
        raise RuntimeError(
            "Not enough rows after preprocessing to build sequences. "
            f"Need > {sequence_length + 1}, found {len(labels)}."
        )

    sample_count = len(labels) - sequence_length - 1
    X_seq = np.ascontiguousarray(sliding_window_view(labels, sequence_length)[:sample_count])
    target_slice = slice(sequence_length, sequence_length + sample_count)
    X_ctx = context_vals[target_slice]
    y_seq = labels[target_slice]
    y_skip = skipped_vals[target_slice]

    n = len(X_seq)
    test_start = int(n * 0.80)
    val_start = int(n * 0.64)

    X_seq_train, X_seq_val, X_seq_test = X_seq[:val_start], X_seq[val_start:test_start], X_seq[test_start:]
    X_ctx_train, X_ctx_val, X_ctx_test = X_ctx[:val_start], X_ctx[val_start:test_start], X_ctx[test_start:]

    y_train, y_val, y_test = y_seq[:val_start], y_seq[val_start:test_start], y_seq[test_start:]
    y_skip_train, y_skip_val, y_skip_test = y_skip[:val_start], y_skip[val_start:test_start], y_skip[test_start:]

    skew_targets = list(SKEW_CONTEXT_FEATURES)
    skew_idx = [context_features.index(key) for key in skew_targets if key in context_features]

    def _log1p_cols(arr: np.ndarray) -> np.ndarray:
        out = arr.astype("float32", copy=True)
        if skew_idx:
            skew_block = np.maximum(out[:, skew_idx], 0.0)
            out[:, skew_idx] = np.log1p(skew_block).astype("float32", copy=False)
        return out

    X_ctx_train = _log1p_cols(X_ctx_train)
    X_ctx_val = _log1p_cols(X_ctx_val)
    X_ctx_test = _log1p_cols(X_ctx_test)

    scaler = StandardScaler()
    X_ctx_train = scaler.fit_transform(X_ctx_train).astype("float32")
    X_ctx_val = scaler.transform(X_ctx_val).astype("float32")
    X_ctx_test = scaler.transform(X_ctx_test).astype("float32")

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    logger.info(
        "Total sequences=%d | train=%d | val=%d | test=%d",
        len(X_seq),
        len(X_seq_train),
        len(X_seq_val),
        len(X_seq_test),
    )

    num_artists = int(np.max(y_seq)) + 1
    num_ctx = X_ctx.shape[1]

    keep_cols = [
        "ts",
        "master_metadata_album_artist_name",
        "artist_label",
        "skipped",
        *context_features,
    ]
    deduped_keep_cols: list[str] = []
    seen_cols: set[str] = set()
    for col in keep_cols:
        if col in seen_cols or col not in df.columns:
            continue
        seen_cols.add(col)
        deduped_keep_cols.append(col)
    keep_cols = deduped_keep_cols
    slim_df = df[keep_cols].copy()
    if "master_metadata_album_artist_name" in slim_df.columns:
        slim_df["master_metadata_album_artist_name"] = slim_df["master_metadata_album_artist_name"].astype("category")

    return PreparedData(
        df=slim_df,
        context_features=context_features,
        X_seq_train=X_seq_train,
        X_seq_val=X_seq_val,
        X_seq_test=X_seq_test,
        X_ctx_train=X_ctx_train,
        X_ctx_val=X_ctx_val,
        X_ctx_test=X_ctx_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_skip_train=y_skip_train.astype("float32"),
        y_skip_val=y_skip_val.astype("float32"),
        y_skip_test=y_skip_test.astype("float32"),
        num_artists=num_artists,
        num_ctx=num_ctx,
    )
