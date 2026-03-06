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

from .config import CANONICAL_AUDIO_FILES, CANONICAL_VIDEO_FILE

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
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "artist_play_count",
    "days_since_last",
    "skip_streak",
    "listen_streak",
    "artist_play_count_24h",
    "artist_play_count_7d",
    "plays_since_last_artist",
    "artist_session_play_count",
    "session_unique_artists_so_far",
    "is_artist_repeat_from_prev",
    "transition_repeat_count",
    "artist_skip_rate_hist",
)

SKEW_CONTEXT_FEATURES: tuple[str, ...] = (
    "time_diff",
    "session_position",
    "artist_play_count",
    "days_since_last",
    "skip_streak",
    "listen_streak",
    "artist_play_count_24h",
    "artist_play_count_7d",
    "plays_since_last_artist",
    "artist_session_play_count",
    "session_unique_artists_so_far",
    "transition_repeat_count",
)

CACHE_SCHEMA_VERSION = "prepared-data-v1"


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


def discover_streaming_files(data_dir: Path, include_video: bool, logger) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()

    canonical_files = [data_dir / name for name in CANONICAL_AUDIO_FILES]
    if include_video:
        canonical_files.append(data_dir / CANONICAL_VIDEO_FILE)

    if all(path.exists() for path in canonical_files):
        return canonical_files

    discovered = sorted(data_dir.glob("Streaming_History_Audio_*.json"))
    if include_video:
        discovered.extend(sorted(data_dir.glob("Streaming_History_Video_*.json")))

    if discovered:
        logger.warning(
            "Using discovered history files because canonical filenames were missing. Found %d files.",
            len(discovered),
        )
        return discovered

    raise FileNotFoundError(
        f"No streaming history JSON files found in {data_dir}. "
        "Expected files named Streaming_History_Audio_*.json."
    )


def load_streaming_history(data_dir: Path, include_video: bool, logger) -> pd.DataFrame:
    files = discover_streaming_files(data_dir, include_video, logger)
    all_records: list[dict] = []

    for path in files:
        with path.open("r", encoding="utf-8") as infile:
            records = json.load(infile)
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

    fingerprint_payload: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "include_video": bool(include_video),
        "enable_spotify_features": bool(enable_spotify_features),
        "max_artists": int(max_artists),
        "sequence_length": int(sequence_length),
        "context_features": list(CONTEXT_FEATURES),
        "skew_context_features": list(SKEW_CONTEXT_FEATURES),
        "files": payload_files,
    }
    serialized = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return fingerprint, fingerprint_payload


def load_or_prepare_training_data(
    *,
    data_dir: Path,
    include_video: bool,
    enable_spotify_features: bool,
    max_artists: int,
    sequence_length: int,
    scaler_path: Path,
    cache_root: Path,
    logger,
) -> tuple[PreparedData, PreparedDataCacheInfo]:
    files = discover_streaming_files(data_dir, include_video, logger)
    fingerprint, fingerprint_payload = _build_prepared_cache_fingerprint(
        files=files,
        include_video=include_video,
        enable_spotify_features=enable_spotify_features,
        max_artists=max_artists,
        sequence_length=sequence_length,
    )

    cache_enabled = _cache_enabled_from_env()
    cache_dir = (cache_root / fingerprint).resolve()
    bundle_path = cache_dir / "prepared_bundle.joblib"
    metadata_path = cache_dir / "cache_meta.json"

    if cache_enabled and bundle_path.exists():
        try:
            payload = joblib.load(bundle_path)
            prepared = payload.get("prepared")
            scaler = payload.get("scaler")
            if not isinstance(prepared, PreparedData):
                raise TypeError("cached payload has unexpected prepared object type")
            if scaler is not None:
                scaler_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(scaler, scaler_path)
            logger.info("Prepared-data cache hit: %s", bundle_path)
            return prepared, PreparedDataCacheInfo(
                enabled=True,
                hit=True,
                fingerprint=fingerprint,
                cache_path=bundle_path,
                metadata_path=(metadata_path if metadata_path.exists() else None),
                source_file_count=len(files),
            )
        except Exception as exc:
            logger.warning("Prepared-data cache load failed (%s). Rebuilding cache.", exc)

    df = load_streaming_history(data_dir, include_video, logger)
    df = engineer_features(df, max_artists, logger)
    df = append_audio_features(df, enable_spotify_features, logger)
    prepared = prepare_training_data(
        df=df,
        sequence_length=sequence_length,
        scaler_path=scaler_path,
        logger=logger,
    )

    if cache_enabled:
        try:
            scaler = joblib.load(scaler_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"prepared": prepared, "scaler": scaler}, bundle_path, compress=3)
            metadata_path.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "created_at_epoch_s": int(time.time()),
                        "fingerprint_payload": fingerprint_payload,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Prepared-data cache saved: %s", bundle_path)
        except Exception as exc:
            logger.warning("Prepared-data cache save skipped due to error: %s", exc)

    return prepared, PreparedDataCacheInfo(
        enabled=cache_enabled,
        hit=False,
        fingerprint=fingerprint,
        cache_path=(bundle_path if cache_enabled else None),
        metadata_path=(metadata_path if cache_enabled else None),
        source_file_count=len(files),
    )


def _ensure_column(df: pd.DataFrame, column: str, default_value):
    if column not in df.columns:
        df[column] = default_value


def _rolling_artist_counts(
    ts_seconds: np.ndarray,
    artists: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    counts = np.zeros(len(artists), dtype="float32")
    buffers: dict[str, deque[int]] = {}

    for idx, (ts_value, artist_name) in enumerate(zip(ts_seconds, artists)):
        bucket = buffers.get(str(artist_name))
        if bucket is None:
            bucket = deque()
            buffers[str(artist_name)] = bucket

        threshold = int(ts_value) - window_seconds
        while bucket and bucket[0] < threshold:
            bucket.popleft()
        counts[idx] = float(len(bucket))
        bucket.append(int(ts_value))

    return counts


def append_audio_features(df: pd.DataFrame, enable_spotify_features: bool, logger) -> pd.DataFrame:
    if not enable_spotify_features:
        logger.info("Spotipy feature fetch disabled via CLI flag.")
        df["danceability"] = 0.0
        df["energy"] = 0.0
        df["tempo"] = 0.0
        return df

    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError:
        logger.warning("spotipy not installed. Skipping audio feature fetch.")
        df["danceability"] = 0.0
        df["energy"] = 0.0
        df["tempo"] = 0.0
        return df

    try:
        cid = os.getenv("SPOTIPY_CLIENT_ID")
        secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        if not cid or not secret:
            raise RuntimeError("SPOTIPY_CLIENT_ID/SECRET not set")

        if "spotify_track_uri" not in df.columns:
            raise RuntimeError("spotify_track_uri column is missing")

        client_credentials_manager = SpotifyClientCredentials()
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        uris = df["spotify_track_uri"].dropna().unique().tolist()
        features_list: list[dict] = []

        for i in range(0, len(uris), 100):
            chunk = uris[i : i + 100]
            try:
                feats_batch = sp.audio_features(chunk)
                for uri, feats in zip(chunk, feats_batch):
                    if feats is None:
                        continue
                    features_list.append(
                        {
                            "spotify_track_uri": uri,
                            "danceability": feats.get("danceability", 0.0),
                            "energy": feats.get("energy", 0.0),
                            "tempo": feats.get("tempo", 0.0),
                        }
                    )
            except Exception:
                time.sleep(0.2)
                continue

        audio_feats = pd.DataFrame(features_list)
        if not audio_feats.empty:
            df = df.merge(audio_feats, on="spotify_track_uri", how="left")

        df[["danceability", "energy", "tempo"]] = df[["danceability", "energy", "tempo"]].fillna(0.0)
        logger.info("Added Spotify audio features for %d tracks", len(audio_feats))
        return df
    except Exception as exc:
        logger.warning("Spotipy audio feature fetch skipped: %s", exc)
        df["danceability"] = 0.0
        df["energy"] = 0.0
        df["tempo"] = 0.0
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

    df = df.sort_values("ts").reset_index(drop=True)
    df["time_diff"] = df["ts"].diff().dt.total_seconds().fillna(0)
    session_threshold = 30 * 60
    df["session_id"] = (df["time_diff"] > session_threshold).cumsum()
    df["session_position"] = df.groupby("session_id").cumcount()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["artist_play_count"] = df.groupby("master_metadata_album_artist_name").cumcount()
    df["days_since_last"] = (
        (df["ts"] - df.groupby("master_metadata_album_artist_name")["ts"].shift(1)).dt.days.fillna(0)
    )

    df["skip_flag"] = df["skipped"]
    streak_grp = (df["skip_flag"] != df["skip_flag"].shift()).cumsum()
    df["skip_streak"] = df.groupby(streak_grp)["skip_flag"].cumcount().where(df["skip_flag"] == 1, 0)
    df["listen_streak"] = df.groupby(streak_grp)["skip_flag"].cumcount().where(df["skip_flag"] == 0, 0)

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

    df = df.sort_values("ts").reset_index(drop=True)

    artist_names = df["master_metadata_album_artist_name"].astype(str).to_numpy()
    ts_seconds = (df["ts"].astype("int64") // 10**9).to_numpy(dtype="int64")

    df["artist_play_count_24h"] = _rolling_artist_counts(ts_seconds, artist_names, window_seconds=24 * 60 * 60)
    df["artist_play_count_7d"] = _rolling_artist_counts(ts_seconds, artist_names, window_seconds=7 * 24 * 60 * 60)

    plays_since_last_artist = np.zeros(len(df), dtype="float32")
    last_seen_idx: dict[str, int] = {}
    for idx, artist_name in enumerate(artist_names):
        prev_idx = last_seen_idx.get(artist_name)
        plays_since_last_artist[idx] = float(idx - prev_idx) if prev_idx is not None else float(idx + 1)
        last_seen_idx[artist_name] = idx
    df["plays_since_last_artist"] = plays_since_last_artist

    df["artist_session_play_count"] = (
        df.groupby(["session_id", "master_metadata_album_artist_name"]).cumcount().astype("float32")
    )

    session_unique_artists_so_far = np.zeros(len(df), dtype="float32")
    current_session = None
    session_seen: set[str] = set()
    for idx, (session_id, artist_name) in enumerate(zip(df["session_id"].to_numpy(), artist_names)):
        if current_session != session_id:
            current_session = session_id
            session_seen = set()
        session_unique_artists_so_far[idx] = float(len(session_seen))
        session_seen.add(artist_name)
    df["session_unique_artists_so_far"] = session_unique_artists_so_far

    repeat_from_prev = (
        (df["master_metadata_album_artist_name"] == df["master_metadata_album_artist_name"].shift(1))
        & (df["session_id"] == df["session_id"].shift(1))
    )
    df["is_artist_repeat_from_prev"] = repeat_from_prev.fillna(False).astype("int8")

    transition_repeat_count = np.zeros(len(df), dtype="float32")
    transition_counts: dict[tuple[int, int], int] = {}
    prev_session_id = None
    prev_artist_label = None
    for idx, (session_id, artist_label) in enumerate(zip(df["session_id"].to_numpy(), df["artist_label"].to_numpy())):
        if prev_artist_label is None or prev_session_id != session_id:
            transition_repeat_count[idx] = 0.0
        else:
            transition = (int(prev_artist_label), int(artist_label))
            transition_repeat_count[idx] = float(transition_counts.get(transition, 0))
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
        prev_session_id = session_id
        prev_artist_label = int(artist_label)
    df["transition_repeat_count"] = transition_repeat_count

    artist_skip_rate_hist = np.zeros(len(df), dtype="float32")
    artist_skip_sum: dict[str, float] = {}
    artist_skip_count: dict[str, int] = {}
    for idx, (artist_name, skipped_value) in enumerate(zip(artist_names, df["skipped"].to_numpy(dtype="float32"))):
        seen = artist_skip_count.get(artist_name, 0)
        total = artist_skip_sum.get(artist_name, 0.0)
        artist_skip_rate_hist[idx] = float(total / seen) if seen > 0 else 0.0
        artist_skip_count[artist_name] = seen + 1
        artist_skip_sum[artist_name] = total + float(skipped_value)
    df["artist_skip_rate_hist"] = artist_skip_rate_hist

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

    df = df.sort_values("ts").reset_index(drop=True)
    missing_context = [key for key in context_features if key not in df.columns]
    if missing_context:
        raise RuntimeError(f"Missing engineered context features: {', '.join(missing_context)}")

    labels = df["artist_label"].tolist()
    context_vals = df[context_features].values

    if len(labels) <= sequence_length + 1:
        raise RuntimeError(
            "Not enough rows after preprocessing to build sequences. "
            f"Need > {sequence_length + 1}, found {len(labels)}."
        )

    X_seq, X_ctx, y_seq, y_skip = [], [], [], []
    for i in range(len(labels) - sequence_length - 1):
        X_seq.append(labels[i : i + sequence_length])
        X_ctx.append(context_vals[i + sequence_length])
        y_seq.append(labels[i + sequence_length])
        y_skip.append(df["skipped"].iloc[i + sequence_length])

    X_seq = np.array(X_seq, dtype="int32")
    X_ctx = np.array(X_ctx, dtype="float32")
    y_seq = np.array(y_seq, dtype="int32")
    y_skip = np.array(y_skip, dtype="float32")

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
        for idx in skew_idx:
            out[:, idx] = np.log1p(np.maximum(out[:, idx], 0.0))
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

    return PreparedData(
        df=df,
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
