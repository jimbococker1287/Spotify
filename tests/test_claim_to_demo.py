from __future__ import annotations

import json
from pathlib import Path

from spotify.claim_to_demo import build_claim_to_demo_report, write_claim_to_demo_artifacts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_sample_outputs(output_root: Path, *, aligned: bool) -> None:
    showcase_run_dir = output_root / "runs" / ("run_full_anchor" if aligned else "run_demo_only")
    control_room_run_id = "run_full_anchor"
    research_run_id = "run_full_anchor"
    showcase_examples_dir = output_root / "analysis" / "taste_os_demo" / "showcase" / "examples"

    commute_demo_md = showcase_examples_dir / "taste_os_demo_commute_friction-spike.md"
    discovery_demo_md = showcase_examples_dir / "taste_os_demo_discovery_skip-recovery.md"
    _write_text(commute_demo_md, "# Commute / Friction Spike\n")
    _write_text(discovery_demo_md, "# Discovery / Skip Recovery\n")

    _write_json(
        output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.json",
        {
            "run_context": {
                "run_dir": str(showcase_run_dir.resolve()),
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
            },
            "showcase_summary": {"canonical_example_count": 4, "mode_comparison_count": 4},
            "canonical_examples": [
                {
                    "label": "Commute / Friction Spike",
                    "mode": "commute",
                    "scenario": "friction_spike",
                    "story": "Friction rises and the route becomes more conservative.",
                    "story_outcome": "Kid Cudi opens and the policy becomes more defensive.",
                    "top_artist": "Kid Cudi",
                    "backup_artist": "Beyonce",
                    "fallback_policy_name": "safe_bucket_normal_friction",
                    "adaptive_replans": 1,
                    "adaptive_safe_route_steps": 4,
                    "demo_md_path": str(commute_demo_md.resolve()),
                },
                {
                    "label": "Discovery / Skip Recovery",
                    "mode": "discovery",
                    "scenario": "skip_recovery",
                    "story": "The system starts adventurous, then recovers after a rejection.",
                    "story_outcome": "Arctic Monkeys opens and then the plan pulls closer to taste.",
                    "top_artist": "Arctic Monkeys",
                    "backup_artist": "Kid Cudi",
                    "fallback_policy_name": "novelty_boosted",
                    "adaptive_replans": 1,
                    "adaptive_safe_route_steps": 0,
                    "demo_md_path": str(discovery_demo_md.resolve()),
                },
            ],
            "mode_comparison": {
                "rows": [
                    {"mode": "focus", "top_artist": "Tame Impala"},
                    {"mode": "workout", "top_artist": "Kid Cudi"},
                    {"mode": "commute", "top_artist": "Kid Cudi"},
                    {"mode": "discovery", "top_artist": "Arctic Monkeys"},
                ]
            },
        },
    )
    _write_text(output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.md", "# Taste OS\n")
    _write_json(
        output_root / "analytics" / "control_room.json",
        {
            "latest_run": {"run_id": control_room_run_id},
            "ops_health": {"headline": "Operational review is healthy.", "status": "healthy"},
            "safety": {
                "robustness_max_top1_gap": 0.158,
                "test_jsd_target_drift": 0.218,
                "test_selective_risk": 0.657,
                "test_abstention_rate": 0.137,
            },
            "qoe": {"stress_worst_skip_risk": 0.613},
        },
    )
    _write_text(output_root / "analytics" / "control_room.md", "# Control Room\n")

    _write_json(
        output_root / "analysis" / "research_claims" / "research_claims.json",
        {
            "run": {"run_id": research_run_id},
            "primary_claim": {
                "key": "shift_robustness",
                "title": "Failure concentration is measurable under drift and repeated-session regimes",
                "status": "analysis_ready",
                "summary": "The current full run shows actionable robustness and drift signals.",
                "metrics": {
                    "worst_robustness_gap": 0.158,
                    "target_drift_jsd": 0.218,
                    "selective_risk": 0.500,
                    "abstention_rate": 0.391,
                    "stress_skip_risk": 0.613,
                },
                "missing_checks": ["Repeat the slice analysis across more seeds."],
            },
            "backup_claim": {"key": "candidate_ranking", "status": "promising_but_unlocked", "summary": "Ranking is promising."},
        },
    )
    _write_text(output_root / "analysis" / "research_claims" / "research_claims.md", "# Research Claims\n")
    _write_text(output_root / "history" / "benchmark_lock_smokebench-v3_manifest.md", "# Benchmark\n")
    _write_json(
        output_root / "history" / "benchmark_lock_smokebench-v3_manifest.json",
        {"benchmark_id": "smokebench-v3", "comparison_ready": True},
    )


def test_claim_to_demo_prefers_safety_demo_for_shift_robustness(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _build_sample_outputs(output_root, aligned=False)

    report = build_claim_to_demo_report(output_root)

    assert report["flagship_demo"]["label"] == "Commute / Friction Spike"
    assert report["primary_claim"]["key"] == "shift_robustness"
    assert report["coherence"]["aligned"] is False
    assert any("Regenerate the Taste OS showcase" in action for action in report["next_actions"])


def test_write_claim_to_demo_artifacts_copies_review_assets(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _build_sample_outputs(output_root, aligned=True)

    report = build_claim_to_demo_report(output_root)
    paths = write_claim_to_demo_artifacts(report, output_dir=output_root)

    assert paths["json"].exists()
    assert paths["md"].exists()
    assert paths["talk_track_md"].exists()
    markdown = paths["md"].read_text(encoding="utf-8")
    assert "Claim To Demo Review Pack" in markdown
    assert "Commute / Friction Spike" in markdown
    assert "Worst supported robustness gap" in markdown
