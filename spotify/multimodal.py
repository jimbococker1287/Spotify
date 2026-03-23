from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultimodalArtistSpace:
    artist_labels: list[str]
    feature_names: list[str]
    raw_features: np.ndarray
    embeddings: np.ndarray
    popularity: np.ndarray
    energy: np.ndarray
    danceability: np.ndarray
    tempo: np.ndarray


def _safe_series(df: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), default, dtype="float32"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim != 2:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms <= 1e-8] = 1.0
    return (arr / norms).astype("float32", copy=False)


def _load_retrieval_embeddings(results: list[dict[str, object]]) -> np.ndarray | None:
    candidate_rows = [
        row
        for row in results
        if str(row.get("model_type", "")).strip() in ("retrieval", "retrieval_reranker")
        and str(row.get("retrieval_artifact_path", "")).strip()
        and Path(str(row.get("retrieval_artifact_path", "")).strip()).exists()
    ]
    if not candidate_rows:
        return None
    best = max(candidate_rows, key=lambda row: float(row.get("val_top1", float("-inf"))))
    try:
        payload = joblib.load(Path(str(best.get("retrieval_artifact_path", "")).strip()))
    except Exception:
        return None
    artist_embeddings = getattr(payload, "artist_embeddings", None)
    if artist_embeddings is None:
        return None
    return np.asarray(artist_embeddings, dtype="float32")


def _artist_frame(df: pd.DataFrame, *, artist_labels: list[str]) -> pd.DataFrame:
    working = df.copy()
    defaults = {
        "skipped": 0.0,
        "energy": 0.0,
        "danceability": 0.0,
        "tempo": 0.0,
        "hour": 0.0,
        "hour_sin": 0.0,
        "hour_cos": 0.0,
        "session_position": 0.0,
        "offline": 0.0,
        "tech_playback_errors_24h": 0.0,
    }
    for column, default in defaults.items():
        working[column] = _safe_series(working, column, default=default)

    artist_stats = (
        working.groupby("artist_label", sort=True)
        .agg(
            play_count=("artist_label", "size"),
            skip_rate=("skipped", "mean"),
            avg_energy=("energy", "mean"),
            avg_danceability=("danceability", "mean"),
            avg_tempo=("tempo", "mean"),
            avg_hour=("hour", "mean"),
            avg_hour_sin=("hour_sin", "mean"),
            avg_hour_cos=("hour_cos", "mean"),
            avg_session_position=("session_position", "mean"),
            avg_offline=("offline", "mean"),
            avg_playback_errors=("tech_playback_errors_24h", "mean"),
        )
        .reset_index()
    )
    full = pd.DataFrame({"artist_label": np.arange(len(artist_labels), dtype="int32")})
    full = full.merge(artist_stats, on="artist_label", how="left")
    full = full.fillna(0.0)
    return full


def _transition_features(df: pd.DataFrame, *, num_artists: int) -> np.ndarray:
    ordered = df.sort_values("ts").reset_index(drop=True)
    current = ordered["artist_label"].to_numpy(dtype="int32", copy=False)
    session_ids = ordered["session_id"].to_numpy(dtype="int64", copy=False) if "session_id" in ordered.columns else np.zeros(len(ordered), dtype="int64")
    valid = session_ids[1:] == session_ids[:-1]
    prev_items = current[:-1][valid]
    next_items = current[1:][valid]
    out_degree = np.bincount(prev_items, minlength=num_artists).astype("float32")
    in_degree = np.bincount(next_items, minlength=num_artists).astype("float32")
    transition_entropy = np.zeros(num_artists, dtype="float32")
    if prev_items.size:
        counts = np.ones((num_artists, num_artists), dtype="float32")
        np.add.at(counts, (prev_items, next_items), 1.0)
        probs = counts / counts.sum(axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
        transition_entropy = entropy.astype("float32", copy=False)
    return np.stack([out_degree, in_degree, transition_entropy], axis=1).astype("float32", copy=False)


def _top_neighbors(space: MultimodalArtistSpace, top_k: int = 5) -> list[dict[str, object]]:
    embeddings = np.asarray(space.embeddings, dtype="float32")
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -1.0)
    rows: list[dict[str, object]] = []
    kk = max(1, min(top_k, len(space.artist_labels) - 1 if len(space.artist_labels) > 1 else 1))
    for idx, artist_name in enumerate(space.artist_labels):
        neighbor_ids = np.argsort(similarities[idx])[::-1][:kk]
        for rank, neighbor_idx in enumerate(neighbor_ids.tolist(), start=1):
            rows.append(
                {
                    "artist_label": idx,
                    "artist_name": artist_name,
                    "neighbor_rank": rank,
                    "neighbor_label": int(neighbor_idx),
                    "neighbor_name": space.artist_labels[int(neighbor_idx)],
                    "cosine_similarity": float(similarities[idx, int(neighbor_idx)]),
                }
            )
    return rows


def compute_multimodal_artist_space(
    *,
    df: pd.DataFrame,
    artist_labels: list[str],
    results: list[dict[str, object]],
) -> MultimodalArtistSpace:
    artist_frame = _artist_frame(df, artist_labels=artist_labels)
    transition_features = _transition_features(df, num_artists=len(artist_labels))

    base_features = artist_frame[
        [
            "play_count",
            "skip_rate",
            "avg_energy",
            "avg_danceability",
            "avg_tempo",
            "avg_hour",
            "avg_hour_sin",
            "avg_hour_cos",
            "avg_session_position",
            "avg_offline",
            "avg_playback_errors",
        ]
    ].to_numpy(dtype="float32", copy=False)
    feature_names = [
        "play_count",
        "skip_rate",
        "avg_energy",
        "avg_danceability",
        "avg_tempo",
        "avg_hour",
        "avg_hour_sin",
        "avg_hour_cos",
        "avg_session_position",
        "avg_offline",
        "avg_playback_errors",
        "transition_out_degree",
        "transition_in_degree",
        "transition_entropy",
    ]
    feature_matrix = np.concatenate([base_features, transition_features], axis=1)

    retrieval_embeddings = _load_retrieval_embeddings(results)
    if retrieval_embeddings is not None and len(retrieval_embeddings) == len(artist_labels):
        feature_matrix = np.concatenate([feature_matrix, retrieval_embeddings.astype("float32", copy=False)], axis=1)
        feature_names.extend([f"retrieval_embed_{idx}" for idx in range(retrieval_embeddings.shape[1])])

    centered = feature_matrix - np.mean(feature_matrix, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    scale[scale <= 1e-6] = 1.0
    standardized = centered / scale
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    dim = max(2, min(16, vt.shape[0]))
    embeddings = standardized @ vt[:dim].T
    embeddings = _normalize_rows(embeddings.astype("float32", copy=False))

    space = MultimodalArtistSpace(
        artist_labels=list(artist_labels),
        feature_names=feature_names,
        raw_features=feature_matrix.astype("float32", copy=False),
        embeddings=embeddings,
        popularity=(artist_frame["play_count"].to_numpy(dtype="float32", copy=False) / max(1.0, float(artist_frame["play_count"].sum()))).astype("float32", copy=False),
        energy=artist_frame["avg_energy"].to_numpy(dtype="float32", copy=False),
        danceability=artist_frame["avg_danceability"].to_numpy(dtype="float32", copy=False),
        tempo=artist_frame["avg_tempo"].to_numpy(dtype="float32", copy=False),
    )
    return space


def build_multimodal_artist_space(
    *,
    df: pd.DataFrame,
    artist_labels: list[str],
    results: list[dict[str, object]],
    output_dir: Path,
    logger,
) -> tuple[MultimodalArtistSpace, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    space = compute_multimodal_artist_space(
        df=df,
        artist_labels=artist_labels,
        results=results,
    )

    artifact_path = output_dir / "multimodal_artist_space.joblib"
    joblib.dump(space, artifact_path, compress=3)
    summary_path = output_dir / "multimodal_artist_space_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "artist_count": len(artist_labels),
                "feature_count": len(space.feature_names),
                "embedding_dim": int(space.embeddings.shape[1]),
                "retrieval_fusion_enabled": any(
                    str(feature_name).startswith("retrieval_embed_")
                    for feature_name in space.feature_names
                ),
                "mean_popularity": float(np.mean(space.popularity)) if len(space.popularity) else float("nan"),
                "mean_energy": float(np.mean(space.energy)) if len(space.energy) else float("nan"),
                "mean_danceability": float(np.mean(space.danceability)) if len(space.danceability) else float("nan"),
                "mean_tempo": float(np.mean(space.tempo)) if len(space.tempo) else float("nan"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    neighbors_path = _write_csv(
        output_dir / "multimodal_artist_neighbors.csv",
        _top_neighbors(space),
        ["artist_label", "artist_name", "neighbor_rank", "neighbor_label", "neighbor_name", "cosine_similarity"],
    )
    logger.info(
        "Built multimodal artist space: artists=%d dim=%d retrieval_fused=%s",
        len(artist_labels),
        int(space.embeddings.shape[1]),
        any(str(feature_name).startswith("retrieval_embed_") for feature_name in space.feature_names),
    )
    return space, [artifact_path, summary_path, neighbors_path]
