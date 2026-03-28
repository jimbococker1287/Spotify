from __future__ import annotations

import json
from pathlib import Path

from spotify.taste_os_showcase import (
    build_mode_comparison_rows,
    build_taste_os_showcase_payload,
    write_taste_os_showcase_artifacts,
)


def _payload(
    *,
    mode: str,
    description: str,
    top_artist: str,
    backup_artist: str,
    fallback_policy: str,
    safe_routed: bool,
    scenario: str = "steady",
) -> dict[str, object]:
    return {
        "request": {"mode": mode, "scenario": scenario, "top_k": 3},
        "mode": {"name": mode, "description": description, "planned_horizon": 4, "default_policy_name": fallback_policy},
        "top_candidates": [
            {
                "rank": 1,
                "artist_name": top_artist,
                "surface_score": 0.83,
                "continuity": 0.52,
                "freshness": 0.61,
                "transition_support": 0.08,
            },
            {
                "rank": 2,
                "artist_name": backup_artist,
                "surface_score": 0.77,
                "continuity": 0.47,
                "freshness": 0.54,
                "transition_support": 0.05,
            },
        ],
        "fallback_policy": {"active_policy_name": fallback_policy, "safe_routed": safe_routed},
        "adaptive_session": {"replan_count": 1, "safe_route_steps": 2},
    }


def test_build_mode_comparison_rows_orders_modes_and_surfaces_opening_summary() -> None:
    rows = build_mode_comparison_rows(
        [
            _payload(
                mode="discovery",
                description="Novelty-weighted lane.",
                top_artist="Artist C",
                backup_artist="Artist D",
                fallback_policy="novelty_boosted",
                safe_routed=False,
            ),
            _payload(
                mode="focus",
                description="Low-surprise lane.",
                top_artist="Artist A",
                backup_artist="Artist B",
                fallback_policy="comfort_policy",
                safe_routed=False,
            ),
        ]
    )

    assert [row["mode"] for row in rows] == ["focus", "discovery"]
    assert rows[0]["top_artist"] == "Artist A"
    assert "Artist A opens the mode" in rows[0]["opening_summary"]


def test_write_taste_os_showcase_artifacts_creates_showcase_and_comparison_files(tmp_path: Path) -> None:
    canonical_examples = [
        {
            "label": "Focus / Steady",
            "mode": "focus",
            "scenario": "steady",
            "story": "Baseline lane.",
            "story_outcome": "Focus stays coherent.",
            "top_artist": "Artist A",
            "backup_artist": "Artist B",
            "fallback_policy_name": "comfort_policy",
            "adaptive_replans": 0,
            "adaptive_safe_route_steps": 0,
            "demo_json_path": "/tmp/focus.json",
            "demo_md_path": "/tmp/focus.md",
        },
        {
            "label": "Discovery / Skip Recovery",
            "mode": "discovery",
            "scenario": "skip_recovery",
            "story": "Recovery lane.",
            "story_outcome": "Discovery explains the recovery.",
            "top_artist": "Artist C",
            "backup_artist": "Artist D",
            "fallback_policy_name": "novelty_boosted",
            "adaptive_replans": 1,
            "adaptive_safe_route_steps": 0,
            "demo_json_path": "/tmp/discovery.json",
            "demo_md_path": "/tmp/discovery.md",
        },
    ]
    mode_rows = build_mode_comparison_rows(
        [
            _payload(
                mode="focus",
                description="Low-surprise lane.",
                top_artist="Artist A",
                backup_artist="Artist B",
                fallback_policy="comfort_policy",
                safe_routed=False,
            ),
            _payload(
                mode="discovery",
                description="Novelty-weighted lane.",
                top_artist="Artist C",
                backup_artist="Artist D",
                fallback_policy="novelty_boosted",
                safe_routed=False,
            ),
        ]
    )
    payload = build_taste_os_showcase_payload(
        run_dir=tmp_path / "run_a",
        model_name="retrieval_reranker",
        model_type="retrieval_reranker",
        canonical_examples=canonical_examples,
        mode_comparison_rows=mode_rows,
        output_dir=tmp_path,
    )

    artifacts = write_taste_os_showcase_artifacts(payload, output_dir=tmp_path)

    assert artifacts["showcase_json"].exists()
    assert artifacts["showcase_md"].exists()
    assert artifacts["comparison_json"].exists()
    assert artifacts["comparison_md"].exists()
    assert "Focus / Steady" in artifacts["showcase_md"].read_text(encoding="utf-8")
    assert "Story Guardrails" in artifacts["showcase_md"].read_text(encoding="utf-8")
    assert "focus" in artifacts["comparison_md"].read_text(encoding="utf-8")
    comparison_payload = json.loads(artifacts["comparison_json"].read_text(encoding="utf-8"))
    assert comparison_payload["rows"][0]["mode"] == "focus"
