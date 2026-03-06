from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import CANONICAL_AUDIO_FILES, CANONICAL_VIDEO_FILE


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


def _ensure_column(df: pd.DataFrame, column: str, default_value):
    if column not in df.columns:
        df[column] = default_value


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


def engineer_features(df: pd.DataFrame, max_artists: int, logger) -> pd.DataFrame:
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
        df[col] = df[col].fillna(False).astype(int)

    top_artists = df["master_metadata_album_artist_name"].value_counts().nlargest(max_artists).index
    df = df[df["master_metadata_album_artist_name"].isin(top_artists)].copy()

    encoder = LabelEncoder()
    df["artist_label"] = encoder.fit_transform(df["master_metadata_album_artist_name"])

    logger.info(
        "After filtering to top %d artists, training frame shape is %s",
        max_artists,
        df.shape,
    )

    return df


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int,
    scaler_path: Path,
    logger,
) -> PreparedData:
    context_features = [
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
    ]

    df = df.sort_values("ts").reset_index(drop=True)
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

    skew_targets = [
        "time_diff",
        "session_position",
        "artist_play_count",
        "days_since_last",
        "skip_streak",
        "listen_streak",
    ]
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
