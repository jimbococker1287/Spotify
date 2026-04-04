from __future__ import annotations

import json
from pathlib import Path

from spotify.front_door import build_front_door_report, write_front_door_artifacts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sample_outputs(output_root: Path) -> None:
    commute_demo_md = output_root / "analysis" / "taste_os_demo" / "showcase" / "examples" / "commute.md"
    _write_text(commute_demo_md, "# Commute / Friction Spike\n")
    _write_json(
        output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.json",
        {
            "run_context": {
                "run_dir": str((output_root / "runs" / "run_anchor").resolve()),
            },
            "showcase_summary": {"canonical_example_count": 4, "mode_comparison_count": 4},
            "canonical_examples": [
                {
                    "label": "Commute / Friction Spike",
                    "mode": "commute",
                    "scenario": "friction_spike",
                    "story": "The system tightens when session friction spikes.",
                    "story_outcome": "A safer route takes over after the spike.",
                    "top_artist": "Kid Cudi",
                    "fallback_policy_name": "safe_bucket_normal_friction",
                    "adaptive_replans": 1,
                    "adaptive_safe_route_steps": 4,
                    "demo_md_path": str(commute_demo_md.resolve()),
                },
                {},
                {},
                {},
            ],
            "mode_comparison": {
                "rows": [
                    {"mode": "focus", "top_artist": "Tame Impala"},
                    {"mode": "commute", "top_artist": "Kid Cudi"},
                    {"mode": "discovery", "top_artist": "Arctic Monkeys"},
                ]
            },
        },
    )
    _write_text(output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.md", "# Showcase\n")
    _write_json(
        output_root / "analytics" / "control_room.json",
        {
            "latest_run": {"run_id": "run_anchor"},
            "ops_health": {"status": "healthy", "headline": "Operational review is healthy."},
            "operating_rhythm": {"overall_status": "healthy", "recommended_review_command": "make control-room"},
            "safety": {"robustness_max_top1_gap": 0.158, "test_jsd_target_drift": 0.218},
            "qoe": {"stress_worst_skip_risk": 0.439},
        },
    )
    _write_text(output_root / "analytics" / "control_room.md", "# Control Room\n")
    _write_text(output_root / "analytics" / "control_room_weekly_summary.md", "# Weekly\n")
    _write_text(output_root / "analytics" / "control_room_triage.md", "# Triage\n")

    creator_dir = output_root / "analysis" / "public_spotify" / "creator_label_intelligence"
    for name in ("creator.md", "scene_seed.md", "ranking.md", "scene.md", "seed.md"):
        _write_text(creator_dir / name, f"# {name}\n")
    _write_json(
        creator_dir / "creator_report_family.json",
        {
            "primary_report": str((creator_dir / "creator.md").resolve()),
            "comparison_view_markdown": {
                "ranking_comparison": str((creator_dir / "ranking.md").resolve()),
                "scene_comparison": str((creator_dir / "scene.md").resolve()),
                "seed_comparison": str((creator_dir / "seed.md").resolve()),
                "scene_seed_comparison": str((creator_dir / "scene_seed.md").resolve()),
            },
        },
    )

    _write_json(
        output_root / "analysis" / "research_claims" / "research_claims.json",
        {
            "run": {"run_id": "run_anchor"},
            "primary_claim": {
                "key": "shift_robustness",
                "title": "Failure concentration is measurable under drift",
                "status": "analysis_ready",
                "summary": "Repeated runs show the same supported slice breaking first.",
                "metrics": {
                    "worst_robustness_gap": 0.158,
                    "consistent_slice_run_count": 3,
                    "repeated_run_count": 3,
                    "consistent_slice_rate": 1.0,
                    "target_drift_jsd": 0.218,
                    "selective_risk": 0.439,
                    "abstention_rate": 0.463,
                    "stress_skip_risk": 0.439,
                },
                "missing_checks": ["Add a mitigation pass for repeat_from_prev=new."],
            },
            "backup_claim": {"key": "candidate_ranking", "status": "promising_but_unlocked", "summary": "Ranking still matters."},
            "benchmark_lock": {"benchmark_id": "smokebench-v3", "comparison_ready": True},
            "believable_submission_path": True,
        },
    )
    _write_text(output_root / "analysis" / "research_claims" / "research_claims.md", "# Research Claims\n")
    _write_text(output_root / "history" / "benchmark_lock_smokebench-v3_manifest.md", "# Benchmark\n")
    _write_json(output_root / "history" / "benchmark_lock_smokebench-v3_manifest.json", {"comparison_ready": True})


def test_front_door_builds_landing_report(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _sample_outputs(output_root)

    report = build_front_door_report(output_root)

    assert report["hero"]["flagship_label"] == "Commute / Friction Spike"
    assert report["hero"]["primary_claim_key"] == "shift_robustness"
    assert len(report["branch_cards"]) == 4


def test_front_door_writes_html_and_markdown(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _sample_outputs(output_root)

    report = build_front_door_report(output_root)
    paths = write_front_door_artifacts(report, output_dir=output_root)

    assert paths["json"].exists()
    assert paths["md"].exists()
    assert paths["html"].exists()
    html = paths["html"].read_text(encoding="utf-8")
    assert "Spotify Personal Taste OS" in html
    assert "Commute / Friction Spike" in html
    assert "Four Branches" in html
