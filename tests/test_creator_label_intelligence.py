from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from spotify.creator_label_intelligence import (
    _seed_transition_helpers,
    _transition_frame,
    build_creator_label_intelligence,
    prepare_creator_intelligence_inputs,
)
from spotify.multimodal import MultimodalArtistSpace
from spotify.public_catalog import SpotifyArtistMetadata


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


class _FakeCatalogClient:
    def __init__(self) -> None:
        self._artists = {
            "artista": SpotifyArtistMetadata(
                query="Artist A",
                spotify_id="artist_a",
                name="Artist A",
                spotify_url="https://open.spotify.com/artist/artist_a",
                genres=["indie pop", "alt z"],
                popularity=74,
                followers_total=500_000,
                image_url=None,
            ),
            "artistb": SpotifyArtistMetadata(
                query="Artist B",
                spotify_id="artist_b",
                name="Artist B",
                spotify_url="https://open.spotify.com/artist/artist_b",
                genres=["indie pop"],
                popularity=68,
                followers_total=310_000,
                image_url=None,
            ),
            "artistc": SpotifyArtistMetadata(
                query="Artist C",
                spotify_id="artist_c",
                name="Artist C",
                spotify_url="https://open.spotify.com/artist/artist_c",
                genres=["alt z", "bedroom pop"],
                popularity=43,
                followers_total=120_000,
                image_url=None,
            ),
            "artistd": SpotifyArtistMetadata(
                query="Artist D",
                spotify_id="artist_d",
                name="Artist D",
                spotify_url="https://open.spotify.com/artist/artist_d",
                genres=["trap", "rap"],
                popularity=61,
                followers_total=280_000,
                image_url=None,
            ),
            "emerginge": SpotifyArtistMetadata(
                query="Emerging E",
                spotify_id="artist_e",
                name="Emerging E",
                spotify_url="https://open.spotify.com/artist/artist_e",
                genres=["indie pop", "dream pop"],
                popularity=28,
                followers_total=48_000,
                image_url=None,
            ),
        }
        self._related = {
            "artist_a": [self._artists["artistc"], self._artists["emerginge"]],
            "artist_b": [self._artists["artistd"]],
            "artist_c": [],
            "artist_d": [],
            "artist_e": [],
        }
        self._albums = {
            "artist_a": [
                self._album("album_a2", "Signal Fire", "2026-02-01"),
                self._album("album_a1", "Open Roads", "2025-06-01"),
            ],
            "artist_b": [self._album("album_b1", "Blue Rooms", "2025-11-15")],
            "artist_c": [
                self._album("album_c3", "Afterlight", "2024-12-01"),
                self._album("album_c2", "Soft Static", "2024-01-01"),
                self._album("album_c1", "Halo City", "2023-01-01"),
            ],
            "artist_d": [self._album("album_d1", "Night Circuit", "2026-02-20")],
            "artist_e": [self._album("album_e1", "First Bloom", "2025-12-20")],
        }
        self._album_labels = {
            "album_a2": "Major House",
            "album_a1": "Major House",
            "album_b1": "Major House",
            "album_c3": "Indie Arc",
            "album_c2": "Indie Arc",
            "album_c1": "Indie Arc",
            "album_d1": "Trapline",
            "album_e1": "Indie Arc",
        }

    @staticmethod
    def _album(album_id: str, name: str, release_date: str) -> dict[str, object]:
        return {
            "id": album_id,
            "name": name,
            "album_type": "album",
            "release_date": release_date,
            "release_date_precision": "day",
            "total_tracks": 10,
            "external_urls": {"spotify": f"https://open.spotify.com/album/{album_id}"},
        }

    def search_artist(self, artist_name: str) -> SpotifyArtistMetadata | None:
        return self._artists.get("".join(char.lower() for char in artist_name if char.isalnum()))

    def get_related_artists(self, artist_id_or_uri: str, *, limit: int = 10) -> list[SpotifyArtistMetadata]:
        return self._related.get(artist_id_or_uri, [])[:limit]

    def get_artist_albums(
        self,
        artist_id_or_uri: str,
        *,
        include_groups: str = "album,single",
        limit: int = 50,
        market: str | None = None,
    ) -> list[dict[str, object]]:
        return self._albums.get(artist_id_or_uri, [])[:limit]

    def get_album(self, album_id_or_uri: str, *, market: str | None = None) -> dict[str, object]:
        return {"id": album_id_or_uri, "label": self._album_labels.get(album_id_or_uri, "")}


def _history_df() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-03-01T00:00:00Z",
            "2026-03-01T00:02:00Z",
            "2026-03-01T00:04:00Z",
            "2026-03-01T01:00:00Z",
            "2026-03-01T01:02:00Z",
            "2026-03-01T01:04:00Z",
            "2026-03-01T02:00:00Z",
            "2026-03-01T02:02:00Z",
            "2026-03-01T02:04:00Z",
            "2026-03-01T03:00:00Z",
            "2026-03-01T03:02:00Z",
            "2026-03-01T03:04:00Z",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "ts": timestamps,
            "session_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "artist_label": [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 3, 2],
            "master_metadata_album_artist_name": [
                "Artist A",
                "Artist B",
                "Artist C",
                "Artist A",
                "Artist B",
                "Artist D",
                "Artist A",
                "Artist C",
                "Artist D",
                "Artist B",
                "Artist D",
                "Artist C",
            ],
        }
    )


def _space() -> MultimodalArtistSpace:
    raw_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.88, 0.12],
            [0.0, 1.0],
        ],
        dtype="float32",
    )
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    embeddings = raw_embeddings / norms
    return MultimodalArtistSpace(
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        feature_names=["f0", "f1"],
        raw_features=np.zeros((4, 2), dtype="float32"),
        embeddings=embeddings.astype("float32"),
        popularity=np.asarray([0.35, 0.30, 0.20, 0.15], dtype="float32"),
        energy=np.zeros(4, dtype="float32"),
        danceability=np.zeros(4, dtype="float32"),
        tempo=np.zeros(4, dtype="float32"),
    )


def test_prepare_creator_intelligence_inputs_builds_derived_space() -> None:
    raw_history = pd.DataFrame(
        {
            "ts": [
                "2026-03-01T00:00:00Z",
                "2026-03-01T00:03:00Z",
                "2026-03-01T01:00:00Z",
                "2026-03-01T01:03:00Z",
            ],
            "master_metadata_album_artist_name": ["Artist A", "Artist B", "Artist A", "Artist C"],
            "platform": ["ios", "ios", "ios", "ios"],
            "reason_start": ["trackdone", "trackdone", "trackdone", "trackdone"],
            "reason_end": ["trackdone", "trackdone", "trackdone", "trackdone"],
        }
    )

    engineered_history, space, info = prepare_creator_intelligence_inputs(
        history_df=raw_history,
        logger=_logger("spotify.test.creator.prepare"),
        max_artists=10,
    )

    assert info["mode"] == "derived"
    assert info["artist_count"] == 3
    assert int(info["embedding_dim"]) >= 2
    assert {"artist_label", "session_id"}.issubset(engineered_history.columns)
    assert space.embeddings.shape[0] == 3


def test_build_creator_label_intelligence_fuses_adjacency_scenes_and_opportunities() -> None:
    payload = build_creator_label_intelligence(
        history_df=_history_df(),
        space=_space(),
        seed_artists=["Artist A", "Artist B"],
        client=_FakeCatalogClient(),
        market="US",
        related_limit=3,
        neighbor_k=3,
        release_limit=5,
        scene_count=2,
        now=pd.Timestamp("2026-03-15T00:00:00Z"),
    )

    assert payload["graph_summary"]["scene_count"] == 2
    assert any(row["target_artist"] == "Artist C" for row in payload["artist_adjacency"])
    assert any(
        row["artist_name"] == "Artist C" and float(row["release_whitespace_score"]) >= 1.0
        for row in payload["release_whitespace"]
    )
    assert any(
        row["source_artist"] == "Artist A" and row["target_artist"] == "Artist B"
        for row in payload["fan_migration"]
    )
    assert any(row["artist_name"] == "Emerging E" for row in payload["opportunities"])
    assert any(
        row["artist_name"] == "Emerging E" and "Indie Arc" in row["dominant_release_labels"]
        for row in payload["nodes"]
    )
    assert payload["ranking_rubric"]["version"] == "week8_creator_intelligence_v1"
    assert payload["opportunities"][0]["opportunity_rank"] == 1
    assert all(row["opportunity_band"] in {"priority_now", "watchlist", "explore"} for row in payload["opportunities"][:3])
    assert all(row["primary_driver"] for row in payload["opportunities"][:3])
    assert all(row["why_now"] for row in payload["opportunities"][:3])
    emerging_row = next(row for row in payload["opportunities"] if row["artist_name"] == "Emerging E")
    assert isinstance(emerging_row["connected_seed_artists"], list)
    assert emerging_row["connected_seed_artists"]
    assert emerging_row["adjacency_component"] > 0.0
    assert emerging_row["scene_momentum_component"] >= 0.0
    assert emerging_row["label_concentration_component"] >= 0.0
    assert emerging_row["local_gap_component"] >= 0.0
    assert emerging_row["why_now"]
    scene_row = next(row for row in payload["scenes"] if row["scene_name"])
    assert "scene_label_concentration" in scene_row
    assert "scene_release_pressure" in scene_row


def test_seed_transition_helpers_preserve_top_routes_and_shares() -> None:
    transitions = _transition_frame(_history_df(), artist_labels=_space().artist_labels)

    outgoing, incoming, share_lookup = _seed_transition_helpers(
        list(transitions.itertuples(index=False, name=None)),
        seed_local_ids=[0, 1],
        neighbor_k=1,
    )

    assert outgoing == {0: [1], 1: [3]}
    assert incoming == {1: [0]}
    assert share_lookup[(0, 1)] == pytest.approx(2.0 / 3.0)
    assert share_lookup[(1, 3)] == pytest.approx(2.0 / 3.0)
