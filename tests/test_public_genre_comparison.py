from __future__ import annotations

import json

import pytest

from spotify.public_genre_comparison import (
    ArtistTagCache,
    GenreDefinition,
    PublicGenreArtistRow,
    artist_tag_cache_key,
    build_genre_comparison,
)


def test_build_genre_comparison_reports_daily_shares_coverage_and_aliases() -> None:
    events = [
        {"timestamp": "2026-01-01T10:00:00Z", "artist_name": "MC One", "duration_ms": 100},
        {"timestamp": "2026-01-01T11:00:00Z", "artist_name": "Pop One", "duration_ms": 300},
        {"timestamp": "2026-01-02T10:00:00Z", "artist_name": "Unknown", "duration_ms": 100},
    ]
    result = build_genre_comparison(
        events,
        {
            "MC One": ["Rap"],
            "Pop One": ["pop"],
        },
        {
            "hip hop": [
                PublicGenreArtistRow("hip hop", 1, "Public Rapper"),
                PublicGenreArtistRow("hip hop", 2, "MC One"),
            ],
            "pop": [PublicGenreArtistRow("pop", 1, "Pop One")],
        },
        genre_definitions={
            "hip-hop": ("hip-hop", "hip hop", "rap"),
            "pop": ("pop",),
        },
    )

    aggregate = result["aggregate"]
    assert aggregate["coverage"]["tagged_event_share"] == pytest.approx(2 / 3)
    assert aggregate["coverage"]["classified_duration_share"] == pytest.approx(0.8)
    assert aggregate["confidence"]["overall_event_confidence"] == pytest.approx(2 / 3)

    genre_shares = {row["genre"]: row for row in aggregate["genre_shares"]}
    assert genre_shares["hip-hop"]["event_share"] == pytest.approx(1 / 3)
    assert genre_shares["pop"]["duration_share"] == pytest.approx(0.6)

    daily = {(row["date"], row["genre"]): row for row in result["daily"]}
    assert daily[("2026-01-01", "hip-hop")]["event_share"] == 0.5
    assert daily[("2026-01-02", "hip-hop")]["event_share"] == 0.0


def test_comparison_reports_overlap_rank_similarity_and_distinctive_artists() -> None:
    events = [
        {"ts": "2026-02-01", "artist": "Alpha", "plays": 3},
        {"ts": "2026-02-01", "artist": "Beta", "plays": 2},
        {"ts": "2026-02-02", "artist": "Personal Only", "plays": 1},
    ]
    public_rows = [
        {"tag": "rap", "rank": 1, "name": "Alpha"},
        {"tag": "rap", "rank": 2, "name": "Public Only"},
        {"tag": "rap", "rank": 3, "name": "Beta"},
    ]

    result = build_genre_comparison(
        events,
        {
            "Alpha": ["hip-hop"],
            "Beta": ["rap"],
            "Personal Only": ["rap"],
        },
        public_rows,
        genre_definitions=[GenreDefinition("hip-hop", ("rap", "hip hop"))],
        top_n=3,
    )

    comparison = result["genres"]["hip-hop"]
    assert comparison["metrics"]["shared_artist_count"] == 2
    assert comparison["metrics"]["personal_top_overlap"] == pytest.approx(2 / 3)
    assert 0.0 < comparison["metrics"]["rank_similarity"] < 1.0
    assert comparison["shared_artists"][0] == {
        "artist_name": "Alpha",
        "personal_rank": 1,
        "public_rank": 1,
        "event_weight": 3.0,
        "duration_ms": 0.0,
    }
    assert [row["artist_name"] for row in comparison["personal_distinctive_artists"]] == [
        "Personal Only"
    ]
    assert [row["artist_name"] for row in comparison["public_distinctive_artists"]] == [
        "Public Only"
    ]


def test_multi_genre_artist_is_fractionally_attributed_and_confidence_is_reduced() -> None:
    result = build_genre_comparison(
        [{"date": "2026-03-01", "artist_name": "Crossover", "duration_ms": 200}],
        {"Crossover": ["rap", "pop"]},
        {},
        genre_definitions={"hip-hop": ("rap",), "pop": ("pop",)},
    )

    shares = {row["genre"]: row for row in result["aggregate"]["genre_shares"]}
    assert shares["hip-hop"]["event_share"] == 0.5
    assert shares["pop"]["event_share"] == 0.5
    assert result["aggregate"]["confidence"]["classified_event_confidence"] == 0.5


def test_artist_tag_cache_has_deterministic_keys_and_ttl(tmp_path) -> None:
    now = [1_000.0]
    cache = ArtistTagCache(tmp_path, ttl_seconds=60, clock=lambda: now[0])

    assert artist_tag_cache_key("A Tribe Called Quest") == artist_tag_cache_key(
        "a-tribe called quest"
    )
    path = cache.set("A Tribe Called Quest", ["Hip-Hop", "hip hop", "Rap"])
    assert cache.get("a tribe called quest") == ["Hip-Hop", "Rap"]
    assert path == cache.path_for("A Tribe Called Quest")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["key"] == cache.key("A Tribe Called Quest")
    assert payload["cached_at"] == 1_000.0

    now[0] = 1_061.0
    assert cache.get("A Tribe Called Quest") is None


def test_genre_aliases_cannot_belong_to_multiple_definitions() -> None:
    with pytest.raises(ValueError, match="belongs to both"):
        build_genre_comparison(
            [],
            {},
            {},
            genre_definitions={"one": ("shared",), "two": ("shared",)},
        )
