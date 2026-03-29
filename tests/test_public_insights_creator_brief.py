from __future__ import annotations

from spotify.public_insights import (
    _creator_brief_executive_summary,
    _creator_brief_migration_watch,
    _creator_brief_priority_shortlist,
    _creator_brief_release_watch,
    _creator_brief_scene_comparison,
    _creator_brief_seed_comparison,
)


def _payload() -> dict[str, object]:
    return {
        "graph_summary": {
            "node_count": 6,
            "scene_count": 2,
            "opportunity_count": 3,
        },
        "scenes": [
            {
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "artist_count": 4,
                "seed_count": 2,
                "scene_local_play_share": 0.62,
            },
            {
                "scene_id": 1,
                "scene_name": "rap / trap",
                "artist_count": 2,
                "seed_count": 0,
                "scene_local_play_share": 0.18,
            },
        ],
        "artist_adjacency": [
            {
                "source_artist": "Artist A",
                "target_artist": "Artist C",
                "hybrid_score": 0.72,
                "transition_share": 0.28,
            },
            {
                "source_artist": "Artist B",
                "target_artist": "Emerging E",
                "hybrid_score": 0.69,
                "transition_share": 0.19,
            },
        ],
        "release_whitespace": [
            {
                "artist_name": "Artist C",
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "release_whitespace_score": 1.31,
            }
        ],
        "fan_migration": [
            {
                "source_artist": "Artist A",
                "target_artist": "Artist B",
                "source_out_share": 0.41,
            }
        ],
        "opportunities": [
            {
                "artist_name": "Emerging E",
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "opportunity_score": 0.77,
                "opportunity_rank": 1,
                "opportunity_band": "priority_now",
                "primary_driver": "seed_adjacency",
                "connected_seed_artists": ["Artist B"],
                "dominant_release_labels": ["Indie Arc"],
                "why_now": "Emerging E is the cleanest adjacency bridge out of Artist B inside `indie pop / alt z`.",
            },
            {
                "artist_name": "Artist C",
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "opportunity_score": 0.66,
                "opportunity_rank": 2,
                "opportunity_band": "priority_now",
                "primary_driver": "release_whitespace",
                "connected_seed_artists": ["Artist A"],
                "dominant_release_labels": ["Indie Arc"],
                "why_now": "Artist C shows whitespace in `indie pop / alt z`, which makes the lane feel under-served.",
            },
            {
                "artist_name": "Artist D",
                "scene_id": 1,
                "scene_name": "rap / trap",
                "opportunity_score": 0.44,
                "opportunity_rank": 3,
                "opportunity_band": "explore",
                "primary_driver": "fan_migration",
                "connected_seed_artists": ["Artist B"],
                "dominant_release_labels": ["Trapline"],
                "why_now": "Audience movement already points toward Artist D, making `rap / trap` the strongest migration lane.",
            },
        ],
    }


def test_creator_brief_scene_comparison_surfaces_top_scene_and_opportunity_counts() -> None:
    rows = _creator_brief_scene_comparison(_payload())

    assert rows[0]["scene_name"] == "indie pop / alt z"
    assert rows[0]["opportunity_count"] == 2
    assert rows[0]["top_opportunity_artist"] == "Emerging E"


def test_creator_brief_seed_comparison_surfaces_best_bridge_per_seed() -> None:
    rows = _creator_brief_seed_comparison(_payload())

    assert rows[0]["seed_artist"] == "Artist A"
    assert rows[0]["top_adjacent_artist"] == "Artist C"
    assert rows[1]["seed_artist"] == "Artist B"


def test_creator_brief_priority_shortlist_surfaces_rank_driver_and_seed_bridge() -> None:
    rows = _creator_brief_priority_shortlist(_payload())

    assert rows[0]["artist_name"] == "Emerging E"
    assert rows[0]["opportunity_rank"] == 1
    assert rows[0]["primary_driver"] == "seed_adjacency"
    assert rows[0]["connected_seed_artists"] == ["Artist B"]


def test_creator_brief_migration_watch_and_release_watch_surface_top_rows() -> None:
    migration_rows = _creator_brief_migration_watch(_payload())
    release_rows = _creator_brief_release_watch(_payload())

    assert migration_rows[0]["source_artist"] == "Artist A"
    assert migration_rows[0]["target_artist"] == "Artist B"
    assert release_rows[0]["artist_name"] == "Artist C"
    assert release_rows[0]["release_whitespace_score"] == 1.31


def test_creator_brief_executive_summary_calls_out_scene_opportunity_and_migration() -> None:
    lines = _creator_brief_executive_summary(_payload())

    assert any("6" in line and "2" in line for line in lines)
    assert any("Emerging E" in line for line in lines)
    assert any("Artist A -> Artist B" in line for line in lines)
