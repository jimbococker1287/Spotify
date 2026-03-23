from __future__ import annotations

from collections import Counter, defaultdict
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data import engineer_features
from .multimodal import MultimodalArtistSpace, compute_multimodal_artist_space
from .public_catalog import SpotifyArtistMetadata, SpotifyPublicCatalogClient


def _normalize_name(value: str) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _artist_labels_from_engineered(df: pd.DataFrame) -> list[str]:
    if df.empty or not {"artist_label", "master_metadata_album_artist_name"}.issubset(df.columns):
        return []
    return (
        df[["artist_label", "master_metadata_album_artist_name"]]
        .drop_duplicates(subset=["artist_label"])
        .sort_values("artist_label")["master_metadata_album_artist_name"]
        .astype(str)
        .tolist()
    )


def prepare_creator_intelligence_inputs(
    *,
    history_df: pd.DataFrame,
    logger: logging.Logger,
    multimodal_space: MultimodalArtistSpace | None = None,
    max_artists: int = 250,
) -> tuple[pd.DataFrame, MultimodalArtistSpace, dict[str, object]]:
    if history_df.empty:
        raise RuntimeError("Creator intelligence requires non-empty listening history.")

    if multimodal_space is None:
        engineered = engineer_features(history_df.copy(), max_artists=max(2, int(max_artists)), logger=logger)
        artist_labels = _artist_labels_from_engineered(engineered)
        if len(artist_labels) < 2:
            raise RuntimeError("Creator intelligence needs at least two artists in the selected history window.")
        derived_space = compute_multimodal_artist_space(
            df=engineered,
            artist_labels=artist_labels,
            results=[],
        )
        return (
            engineered,
            derived_space,
            {
                "mode": "derived",
                "artist_count": int(len(artist_labels)),
                "embedding_dim": int(derived_space.embeddings.shape[1]),
                "retrieval_fusion_enabled": False,
            },
        )

    artist_labels = list(multimodal_space.artist_labels)
    engineered = engineer_features(
        history_df.copy(),
        max_artists=max(2, len(artist_labels)),
        logger=logger,
        artist_classes=artist_labels,
    )
    if engineered.empty:
        raise RuntimeError("No artists from the selected history window were present in the multimodal artist space.")
    return (
        engineered,
        multimodal_space,
        {
            "mode": "artifact",
            "artist_count": int(len(artist_labels)),
            "embedding_dim": int(multimodal_space.embeddings.shape[1]),
            "retrieval_fusion_enabled": any(
                str(feature_name).startswith("retrieval_embed_") for feature_name in multimodal_space.feature_names
            ),
        },
    )


def _local_artist_stats(history_df: pd.DataFrame, *, artist_labels: list[str]) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["artist_label", "artist_name", "play_count", "play_share"])
    counts = (
        history_df.groupby("artist_label", sort=True)
        .size()
        .rename("play_count")
        .reset_index()
    )
    counts["artist_name"] = counts["artist_label"].map({idx: name for idx, name in enumerate(artist_labels)})
    total = float(counts["play_count"].sum())
    counts["play_share"] = counts["play_count"].astype("float64") / max(total, 1.0)
    return counts


def _transition_frame(history_df: pd.DataFrame, *, artist_labels: list[str]) -> pd.DataFrame:
    required = {"artist_label", "session_id", "ts"}
    if history_df.empty or not required.issubset(history_df.columns):
        return pd.DataFrame(
            columns=[
                "source_label",
                "target_label",
                "transition_count",
                "source_out_share",
                "target_in_share",
                "source_artist",
                "target_artist",
            ]
        )
    ordered = history_df.sort_values("ts").reset_index(drop=True)
    current = ordered["artist_label"].to_numpy(dtype="int32", copy=False)
    session_ids = ordered["session_id"].to_numpy(dtype="int64", copy=False)
    valid = session_ids[1:] == session_ids[:-1]
    prev_items = current[:-1][valid]
    next_items = current[1:][valid]
    if prev_items.size == 0:
        return pd.DataFrame(
            columns=[
                "source_label",
                "target_label",
                "transition_count",
                "source_out_share",
                "target_in_share",
                "source_artist",
                "target_artist",
            ]
        )

    frame = pd.DataFrame({"source_label": prev_items, "target_label": next_items})
    grouped = (
        frame.groupby(["source_label", "target_label"], sort=False)
        .size()
        .rename("transition_count")
        .reset_index()
    )
    source_totals = grouped.groupby("source_label", sort=False)["transition_count"].sum().rename("source_total")
    target_totals = grouped.groupby("target_label", sort=False)["transition_count"].sum().rename("target_total")
    grouped = grouped.merge(source_totals, on="source_label", how="left").merge(target_totals, on="target_label", how="left")
    grouped["source_out_share"] = grouped["transition_count"] / grouped["source_total"].clip(lower=1)
    grouped["target_in_share"] = grouped["transition_count"] / grouped["target_total"].clip(lower=1)
    grouped["source_artist"] = grouped["source_label"].map({idx: name for idx, name in enumerate(artist_labels)})
    grouped["target_artist"] = grouped["target_label"].map({idx: name for idx, name in enumerate(artist_labels)})
    return grouped.sort_values(["transition_count", "source_out_share"], ascending=[False, False]).reset_index(drop=True)


def _top_similarity_neighbors(
    space: MultimodalArtistSpace,
    *,
    artist_ids: list[int],
    top_k: int,
) -> dict[int, list[tuple[int, float]]]:
    embeddings = np.asarray(space.embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.size == 0:
        return {artist_id: [] for artist_id in artist_ids}
    similarities = embeddings @ embeddings.T
    results: dict[int, list[tuple[int, float]]] = {}
    for artist_id in artist_ids:
        if artist_id < 0 or artist_id >= similarities.shape[0]:
            results[artist_id] = []
            continue
        row = similarities[artist_id].copy()
        row[artist_id] = -1.0
        neighbor_ids = np.argsort(row)[::-1]
        pairs: list[tuple[int, float]] = []
        for neighbor_id in neighbor_ids.tolist():
            score = float(row[int(neighbor_id)])
            if score <= 0.0:
                continue
            pairs.append((int(neighbor_id), score))
            if len(pairs) >= max(1, int(top_k)):
                break
        results[artist_id] = pairs
    return results


def _genre_overlap(left: list[str], right: list[str]) -> float:
    left_set = {_normalize_name(item) for item in left if item}
    right_set = {_normalize_name(item) for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set & right_set) / len(left_set | right_set))


def _median_gap_days(release_dates: list[pd.Timestamp]) -> float | None:
    if len(release_dates) < 2:
        return None
    ordered = sorted(release_dates)
    deltas = [
        float((ordered[idx] - ordered[idx - 1]).total_seconds() / 86400.0)
        for idx in range(1, len(ordered))
        if ordered[idx] > ordered[idx - 1]
    ]
    if not deltas:
        return None
    return float(np.median(np.asarray(deltas, dtype="float32")))


def _parse_release_date(raw_value: str, precision: str) -> pd.Timestamp | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    if precision == "year":
        value = f"{value}-01-01"
    elif precision == "month":
        value = f"{value}-01"
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp


def _dedupe_albums(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("name", "")).casefold(),
            str(row.get("release_date", "")),
            str(row.get("album_type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _release_profile(
    *,
    client: SpotifyPublicCatalogClient,
    metadata: SpotifyArtistMetadata,
    market: str,
    album_limit: int,
    label_album_limit: int,
    now: pd.Timestamp,
) -> dict[str, object]:
    albums = client.get_artist_albums(
        metadata.spotify_id,
        include_groups="album,single",
        limit=max(6, int(album_limit) * 3),
        market=market,
    )
    deduped = _dedupe_albums(albums)
    release_rows: list[dict[str, object]] = []
    release_dates: list[pd.Timestamp] = []
    label_counter: Counter[str] = Counter()

    for album in deduped[: max(1, int(album_limit))]:
        precision = str(album.get("release_date_precision", "day")).strip()
        release_ts = _parse_release_date(str(album.get("release_date", "")).strip(), precision)
        if release_ts is None:
            continue
        release_dates.append(release_ts)
        release_rows.append(
            {
                "album_id": str(album.get("id", "")).strip(),
                "album_name": str(album.get("name", "")).strip(),
                "album_type": str(album.get("album_type", "")).strip(),
                "release_date": str(album.get("release_date", "")).strip(),
                "release_date_precision": precision,
                "total_tracks": int(album.get("total_tracks", 0) or 0),
                "spotify_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
            }
        )

    for row in release_rows[: max(1, int(label_album_limit))]:
        album_id = str(row.get("album_id", "")).strip()
        if not album_id:
            continue
        album_payload = client.get_album(album_id, market=market)
        label = str(album_payload.get("label", "")).strip()
        if label:
            label_counter[label] += 1

    release_rows.sort(key=lambda row: str(row.get("release_date", "")), reverse=True)
    ordered_dates = sorted(release_dates)
    latest_release = ordered_dates[-1] if ordered_dates else None
    median_gap_days = _median_gap_days(ordered_dates)
    days_since_latest = (
        int(max((now - latest_release).total_seconds() / 86400.0, 0.0)) if latest_release is not None else None
    )
    if days_since_latest is None:
        whitespace_score = 0.0
    elif median_gap_days is None:
        whitespace_score = float(days_since_latest / 180.0)
    else:
        whitespace_score = float(days_since_latest / max(median_gap_days, 1.0))

    return {
        "release_count": int(len(release_rows)),
        "recent_releases": release_rows,
        "latest_release_date": latest_release.date().isoformat() if latest_release is not None else None,
        "days_since_latest_release": days_since_latest,
        "median_gap_days": round(median_gap_days, 2) if median_gap_days is not None else None,
        "release_whitespace_score": round(whitespace_score, 4),
        "dominant_labels": [label for label, _count in label_counter.most_common(3)],
    }


def _cluster_local_scenes(
    *,
    space: MultimodalArtistSpace,
    local_artist_ids: list[int],
    scene_count: int | None,
) -> dict[int, int]:
    unique_ids = sorted({int(artist_id) for artist_id in local_artist_ids if 0 <= int(artist_id) < len(space.artist_labels)})
    if not unique_ids:
        return {}
    if len(unique_ids) == 1:
        return {unique_ids[0]: 0}
    if scene_count is None:
        target_clusters = max(2, min(6, int(round(np.sqrt(len(unique_ids))))))
    else:
        target_clusters = max(1, int(scene_count))
    target_clusters = min(target_clusters, len(unique_ids))
    if target_clusters <= 1:
        return {artist_id: 0 for artist_id in unique_ids}

    features = np.asarray(space.embeddings, dtype="float32")[unique_ids]
    model = KMeans(n_clusters=target_clusters, n_init=10, random_state=42)
    assignments = model.fit_predict(features)
    return {artist_id: int(scene_id) for artist_id, scene_id in zip(unique_ids, assignments.tolist())}


def build_creator_label_intelligence(
    *,
    history_df: pd.DataFrame,
    space: MultimodalArtistSpace,
    seed_artists: list[str],
    client: SpotifyPublicCatalogClient,
    market: str,
    related_limit: int = 8,
    neighbor_k: int = 5,
    release_limit: int = 10,
    scene_count: int | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, object]:
    if history_df.empty:
        raise RuntimeError("Creator intelligence requires non-empty engineered listening history.")

    artist_labels = list(space.artist_labels)
    play_stats = _local_artist_stats(history_df, artist_labels=artist_labels)
    transitions = _transition_frame(history_df, artist_labels=artist_labels)
    local_name_to_id = {_normalize_name(name): idx for idx, name in enumerate(artist_labels)}
    play_count_map = {int(row["artist_label"]): int(row["play_count"]) for row in play_stats.to_dict(orient="records")}
    play_share_map = {int(row["artist_label"]): float(row["play_share"]) for row in play_stats.to_dict(orient="records")}

    seed_local_ids = [local_name_to_id[key] for key in [_normalize_name(item) for item in seed_artists] if key in local_name_to_id]
    if not seed_local_ids and not play_stats.empty:
        seed_local_ids = play_stats.sort_values(["play_count", "artist_label"], ascending=[False, True])["artist_label"].head(
            max(1, min(3, len(play_stats)))
        ).astype(int).tolist()

    neighbor_map = _top_similarity_neighbors(space, artist_ids=seed_local_ids, top_k=max(1, int(neighbor_k)))
    candidate_local_ids: set[int] = set(seed_local_ids)
    for pairs in neighbor_map.values():
        for neighbor_id, _score in pairs:
            candidate_local_ids.add(int(neighbor_id))
    if not transitions.empty and seed_local_ids:
        for seed_id in seed_local_ids:
            seed_routes = transitions[transitions["source_label"] == int(seed_id)].head(max(1, int(neighbor_k)))
            candidate_local_ids.update(seed_routes["target_label"].astype(int).tolist())
            inbound_routes = transitions[transitions["target_label"] == int(seed_id)].head(max(1, int(neighbor_k)))
            candidate_local_ids.update(inbound_routes["source_label"].astype(int).tolist())

    metadata_lookup: dict[str, SpotifyArtistMetadata] = {}
    spotify_id_lookup: dict[str, SpotifyArtistMetadata] = {}

    def register_metadata(metadata: SpotifyArtistMetadata) -> SpotifyArtistMetadata:
        metadata_lookup[_normalize_name(metadata.name)] = metadata
        spotify_id_lookup[metadata.spotify_id] = metadata
        return metadata

    seed_metadata: list[SpotifyArtistMetadata] = []
    public_related_ids: dict[str, set[str]] = defaultdict(set)
    external_scene_votes: dict[str, list[int]] = defaultdict(list)
    external_seed_names: dict[str, list[str]] = defaultdict(list)
    public_related_rows: list[dict[str, object]] = []

    for seed_name in seed_artists:
        metadata = client.search_artist(seed_name)
        if metadata is None:
            continue
        seed_metadata.append(register_metadata(metadata))

    for local_id in sorted(candidate_local_ids):
        artist_name = artist_labels[int(local_id)]
        metadata = client.search_artist(artist_name)
        if metadata is not None:
            register_metadata(metadata)

    scene_map = _cluster_local_scenes(space=space, local_artist_ids=sorted(candidate_local_ids), scene_count=scene_count)

    for metadata in seed_metadata:
        related_artists = client.get_related_artists(metadata.spotify_id, limit=max(1, int(related_limit)))
        source_local_id = local_name_to_id.get(_normalize_name(metadata.query))
        source_scene = scene_map.get(source_local_id) if source_local_id is not None else None
        for related in related_artists:
            register_metadata(related)
            public_related_ids[metadata.spotify_id].add(related.spotify_id)
            if source_scene is not None:
                external_scene_votes[related.spotify_id].append(int(source_scene))
            external_seed_names[related.spotify_id].append(metadata.name)
            public_related_rows.append(
                {
                    "source_artist": metadata.name,
                    "source_spotify_id": metadata.spotify_id,
                    "target_artist": related.name,
                    "target_spotify_id": related.spotify_id,
                    "edge_type": "public_related",
                    "public_related": 1.0,
                    "embedding_similarity": None,
                    "transition_share": None,
                    "hybrid_score": 1.0,
                }
            )

    release_profiles: dict[str, dict[str, object]] = {}
    candidate_external_metadata = [spotify_id_lookup[spotify_id] for spotify_id in sorted(external_scene_votes)]
    candidate_metadata = list({item.spotify_id: item for item in [*seed_metadata, *candidate_external_metadata, *metadata_lookup.values()]}.values())
    now_ts = now or pd.Timestamp.now(tz="UTC")
    for metadata in candidate_metadata:
        release_profiles[metadata.spotify_id] = _release_profile(
            client=client,
            metadata=metadata,
            market=market,
            album_limit=max(3, int(release_limit)),
            label_album_limit=2,
            now=now_ts,
        )

    adjacency_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    adjacency_scores_by_artist: dict[str, list[float]] = defaultdict(list)
    migration_scores_by_artist: dict[str, list[float]] = defaultdict(list)

    for seed_id in seed_local_ids:
        source_name = artist_labels[int(seed_id)]
        source_metadata = metadata_lookup.get(_normalize_name(source_name))
        source_related_ids = public_related_ids.get(source_metadata.spotify_id, set()) if source_metadata is not None else set()
        for target_id, similarity in neighbor_map.get(int(seed_id), []):
            target_name = artist_labels[int(target_id)]
            target_metadata = metadata_lookup.get(_normalize_name(target_name))
            transition_share = 0.0
            if not transitions.empty:
                route = transitions[
                    (transitions["source_label"] == int(seed_id)) & (transitions["target_label"] == int(target_id))
                ].head(1)
                if not route.empty:
                    transition_share = _safe_float(route.iloc[0]["source_out_share"])
            public_related = float(
                1.0 if target_metadata is not None and target_metadata.spotify_id in source_related_ids else 0.0
            )
            similarity_norm = max(0.0, min(1.0, (float(similarity) + 1.0) / 2.0))
            hybrid_score = (0.55 * similarity_norm) + (0.25 * transition_share) + (0.20 * public_related)
            row = {
                "source_artist": source_name,
                "target_artist": target_name,
                "edge_type": "hybrid_adjacency",
                "embedding_similarity": round(float(similarity), 4),
                "transition_share": round(float(transition_share), 4),
                "public_related": round(float(public_related), 4),
                "hybrid_score": round(float(hybrid_score), 4),
            }
            adjacency_rows.append(row)
            edge_rows.append(row)
            adjacency_scores_by_artist[target_name].append(float(hybrid_score))

    transition_rows = transitions[
        transitions["source_label"].isin(sorted(candidate_local_ids)) & transitions["target_label"].isin(sorted(candidate_local_ids))
    ].copy()
    fan_migration_rows: list[dict[str, object]] = []
    for row in transition_rows.to_dict(orient="records"):
        source_label = int(row["source_label"])
        target_label = int(row["target_label"])
        source_name = artist_labels[source_label]
        target_name = artist_labels[target_label]
        migration_row = {
            "source_artist": source_name,
            "target_artist": target_name,
            "source_scene_id": scene_map.get(source_label),
            "target_scene_id": scene_map.get(target_label),
            "transition_count": int(row["transition_count"]),
            "source_out_share": round(_safe_float(row["source_out_share"]), 4),
            "target_in_share": round(_safe_float(row["target_in_share"]), 4),
        }
        fan_migration_rows.append(migration_row)
        edge_rows.append(
            {
                "source_artist": source_name,
                "target_artist": target_name,
                "edge_type": "fan_migration",
                "embedding_similarity": None,
                "transition_share": migration_row["source_out_share"],
                "public_related": None,
                "hybrid_score": migration_row["source_out_share"],
            }
        )
        migration_scores_by_artist[target_name].append(float(migration_row["source_out_share"]))

    for row in public_related_rows:
        target_name = str(row["target_artist"])
        adjacency_scores_by_artist[target_name].append(float(row["hybrid_score"]))
        edge_rows.append(row)

    seed_name_set = {_normalize_name(item) for item in seed_artists}
    seed_spotify_ids = {item.spotify_id for item in seed_metadata}
    candidate_node_names = {artist_labels[int(local_id)] for local_id in candidate_local_ids}
    candidate_node_names.update(metadata.name for metadata in metadata_lookup.values())

    nodes: list[dict[str, object]] = []
    for artist_name in sorted(candidate_node_names):
        norm_name = _normalize_name(artist_name)
        local_id = local_name_to_id.get(norm_name)
        metadata = metadata_lookup.get(norm_name)
        spotify_id = metadata.spotify_id if metadata is not None else None
        release_profile = release_profiles.get(spotify_id or "", {})
        if local_id is not None:
            scene_id = scene_map.get(int(local_id))
        elif spotify_id and external_scene_votes.get(spotify_id):
            scene_id = Counter(external_scene_votes[spotify_id]).most_common(1)[0][0]
        else:
            scene_id = None
        node = {
            "artist_name": metadata.name if metadata is not None else artist_name,
            "spotify_id": spotify_id,
            "spotify_url": metadata.spotify_url if metadata is not None else None,
            "image_url": metadata.image_url if metadata is not None else None,
            "genres": metadata.genres if metadata is not None else [],
            "followers_total": metadata.followers_total if metadata is not None else None,
            "public_popularity": metadata.popularity if metadata is not None else None,
            "local_artist": bool(local_id is not None),
            "local_artist_label": int(local_id) if local_id is not None else None,
            "seed": bool(
                norm_name in seed_name_set
                or (metadata is not None and metadata.spotify_id in seed_spotify_ids)
            ),
            "local_play_count": int(play_count_map.get(int(local_id), 0)) if local_id is not None else 0,
            "local_play_share": round(float(play_share_map.get(int(local_id), 0.0)), 6) if local_id is not None else 0.0,
            "scene_id": scene_id,
            "scene_name": "",
            "dominant_release_labels": release_profile.get("dominant_labels", []),
            "latest_release_date": release_profile.get("latest_release_date"),
            "days_since_latest_release": release_profile.get("days_since_latest_release"),
            "median_release_gap_days": release_profile.get("median_gap_days"),
            "release_whitespace_score": release_profile.get("release_whitespace_score", 0.0),
            "seed_adjacency_score": round(max(adjacency_scores_by_artist.get(artist_name, [0.0])), 4),
            "fan_migration_score": round(max(migration_scores_by_artist.get(artist_name, [0.0])), 4),
            "connected_seed_artists": sorted(set(external_seed_names.get(spotify_id or "", []))),
        }
        nodes.append(node)

    scene_rows: list[dict[str, object]] = []
    scene_names: dict[int, str] = {}
    for scene_id in sorted({row["scene_id"] for row in nodes if row["scene_id"] is not None}):
        scene_nodes = [row for row in nodes if row["scene_id"] == scene_id]
        genre_counter: Counter[str] = Counter()
        label_counter: Counter[str] = Counter()
        for row in scene_nodes:
            genre_counter.update(str(item) for item in row.get("genres", []) if str(item).strip())
            label_counter.update(str(item) for item in row.get("dominant_release_labels", []) if str(item).strip())
        top_genres = [item for item, _count in genre_counter.most_common(3)]
        top_labels = [item for item, _count in label_counter.most_common(3)]
        scene_name = " / ".join(top_genres[:2]) or " / ".join(top_labels[:2]) or f"scene-{int(scene_id) + 1}"
        scene_names[int(scene_id)] = scene_name
        popularity_values = [
            _safe_float(row.get("public_popularity"), default=float("nan"))
            for row in scene_nodes
            if row.get("public_popularity") is not None
        ]
        popularity_values = [value for value in popularity_values if np.isfinite(value)]
        scene_rows.append(
            {
                "scene_id": int(scene_id),
                "scene_name": scene_name,
                "artist_count": int(len(scene_nodes)),
                "local_artist_count": int(sum(1 for row in scene_nodes if row.get("local_artist"))),
                "seed_count": int(sum(1 for row in scene_nodes if row.get("seed"))),
                "dominant_genres": top_genres,
                "dominant_labels": top_labels,
                "scene_local_play_share": round(sum(_safe_float(row.get("local_play_share")) for row in scene_nodes), 6),
                "scene_avg_public_popularity": round(float(np.mean(popularity_values)), 2) if popularity_values else None,
                "scene_whitespace_artist_count": int(
                    sum(1 for row in scene_nodes if _safe_float(row.get("release_whitespace_score")) >= 1.0)
                ),
            }
        )

    for row in nodes:
        scene_id = row.get("scene_id")
        row["scene_name"] = scene_names.get(int(scene_id), "unmapped") if scene_id is not None else "unmapped"

    release_whitespace_rows = [
        {
            "artist_name": row["artist_name"],
            "scene_id": row["scene_id"],
            "scene_name": row["scene_name"],
            "latest_release_date": row["latest_release_date"],
            "days_since_latest_release": row["days_since_latest_release"],
            "median_release_gap_days": row["median_release_gap_days"],
            "release_whitespace_score": row["release_whitespace_score"],
            "dominant_release_labels": row["dominant_release_labels"],
        }
        for row in nodes
        if row.get("latest_release_date")
    ]
    release_whitespace_rows.sort(
        key=lambda row: (_safe_float(row.get("release_whitespace_score")), row.get("days_since_latest_release") or 0),
        reverse=True,
    )

    seed_play_reference = max([play_share_map.get(int(artist_id), 0.0) for artist_id in seed_local_ids] or [0.0])
    if seed_play_reference <= 0.0:
        seed_play_reference = max(play_share_map.values() or [1.0])

    opportunity_rows: list[dict[str, object]] = []
    for row in nodes:
        if row.get("seed"):
            continue
        popularity = row.get("public_popularity")
        popularity_tail = 1.0 - min(max(_safe_float(popularity, default=50.0), 0.0), 100.0) / 100.0
        local_play_share = _safe_float(row.get("local_play_share"))
        local_gap = 1.0 - min(local_play_share / max(seed_play_reference, 1e-6), 1.0)
        adjacency_score = _safe_float(row.get("seed_adjacency_score"))
        migration_score = _safe_float(row.get("fan_migration_score"))
        days_since_latest = row.get("days_since_latest_release")
        if days_since_latest is None:
            freshness_score = 0.0
        else:
            freshness_score = max(0.0, 1.0 - (float(days_since_latest) / 180.0))
        whitespace_score = min(_safe_float(row.get("release_whitespace_score")) / 2.0, 1.0)
        opportunity_score = (
            0.35 * adjacency_score
            + 0.20 * migration_score
            + 0.20 * freshness_score
            + 0.15 * local_gap
            + 0.10 * popularity_tail
        )
        rationale: list[str] = []
        if adjacency_score >= 0.45:
            rationale.append("strong adjacency to seed artists")
        if migration_score >= 0.15:
            rationale.append("visible fan migration from current listening routes")
        if freshness_score >= 0.35:
            rationale.append("recent release activity")
        if whitespace_score >= 0.7:
            rationale.append("release cadence suggests whitespace")
        if local_gap >= 0.5 and popularity_tail >= 0.25:
            rationale.append("under-penetrated long-tail fit")
        if not rationale:
            continue
        opportunity_rows.append(
            {
                "artist_name": row["artist_name"],
                "scene_id": row["scene_id"],
                "scene_name": row["scene_name"],
                "opportunity_score": round(float(opportunity_score), 4),
                "seed_adjacency_score": round(adjacency_score, 4),
                "fan_migration_score": round(migration_score, 4),
                "freshness_score": round(float(freshness_score), 4),
                "release_whitespace_score": round(_safe_float(row.get("release_whitespace_score")), 4),
                "local_play_share": round(local_play_share, 6),
                "public_popularity": popularity,
                "dominant_release_labels": row["dominant_release_labels"],
                "rationale": rationale,
            }
        )
    opportunity_rows.sort(key=lambda row: row["opportunity_score"], reverse=True)

    adjacency_rows.sort(key=lambda row: row["hybrid_score"], reverse=True)
    edge_rows.sort(
        key=lambda row: (
            0 if row["edge_type"] == "hybrid_adjacency" else 1,
            -_safe_float(row.get("hybrid_score")),
        )
    )

    return {
        "seed_artists": seed_artists,
        "artist_adjacency": adjacency_rows[:50],
        "nodes": nodes,
        "edges": edge_rows[:200],
        "scenes": scene_rows,
        "release_whitespace": release_whitespace_rows[:50],
        "fan_migration": fan_migration_rows[:50],
        "opportunities": opportunity_rows[:30],
        "graph_summary": {
            "seed_count": int(len(seed_artists)),
            "local_seed_count": int(len(seed_local_ids)),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edge_rows)),
            "scene_count": int(len(scene_rows)),
            "adjacency_count": int(len(adjacency_rows)),
            "fan_migration_count": int(len(fan_migration_rows)),
            "release_whitespace_count": int(len(release_whitespace_rows)),
            "opportunity_count": int(len(opportunity_rows)),
        },
    }
