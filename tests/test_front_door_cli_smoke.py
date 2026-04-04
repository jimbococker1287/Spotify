from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_front_door_cli_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    analysis = output_root / "analysis"
    analytics = output_root / "analytics"
    history = output_root / "history"
    creator = analysis / "public_spotify" / "creator_label_intelligence"
    showcase = analysis / "taste_os_demo" / "showcase"

    showcase.mkdir(parents=True, exist_ok=True)
    creator.mkdir(parents=True, exist_ok=True)
    analytics.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)

    (showcase / "taste_os_showcase.json").write_text(
        '{"run_context":{"run_dir":"'
        + str((output_root / "runs" / "run_anchor").resolve())
        + '"},"showcase_summary":{"canonical_example_count":4,"mode_comparison_count":4},"canonical_examples":[{"label":"Commute / Friction Spike","mode":"commute","scenario":"friction_spike","story":"demo","story_outcome":"outcome","top_artist":"Kid Cudi","fallback_policy_name":"safe_bucket_normal_friction","adaptive_replans":1,"adaptive_safe_route_steps":4,"demo_md_path":"'
        + str((showcase / "demo.md").resolve())
        + '"},{},{},{}],"mode_comparison":{"rows":[{"mode":"focus","top_artist":"Tame Impala"},{"mode":"commute","top_artist":"Kid Cudi"},{"mode":"discovery","top_artist":"Arctic Monkeys"}]}}',
        encoding="utf-8",
    )
    (showcase / "demo.md").write_text("# Demo\n", encoding="utf-8")
    (showcase / "taste_os_showcase.md").write_text("# Showcase\n", encoding="utf-8")
    (analytics / "control_room.json").write_text(
        '{"latest_run":{"run_id":"run_anchor"},"ops_health":{"status":"healthy","headline":"Operational review is healthy."},"operating_rhythm":{"overall_status":"healthy","recommended_review_command":"make control-room"},"safety":{"robustness_max_top1_gap":0.158,"test_jsd_target_drift":0.218},"qoe":{"stress_worst_skip_risk":0.439}}',
        encoding="utf-8",
    )
    (analytics / "control_room.md").write_text("# Control Room\n", encoding="utf-8")
    (analytics / "control_room_weekly_summary.md").write_text("# Weekly\n", encoding="utf-8")
    (analytics / "control_room_triage.md").write_text("# Triage\n", encoding="utf-8")
    (creator / "creator.md").write_text("# Creator\n", encoding="utf-8")
    (creator / "ranking.md").write_text("# Ranking\n", encoding="utf-8")
    (creator / "scene.md").write_text("# Scene\n", encoding="utf-8")
    (creator / "seed.md").write_text("# Seed\n", encoding="utf-8")
    (creator / "scene_seed.md").write_text("# Scene Seed\n", encoding="utf-8")
    (creator / "creator_report_family.json").write_text(
        '{"primary_report":"'
        + str((creator / "creator.md").resolve())
        + '","comparison_view_markdown":{"ranking_comparison":"'
        + str((creator / "ranking.md").resolve())
        + '","scene_comparison":"'
        + str((creator / "scene.md").resolve())
        + '","seed_comparison":"'
        + str((creator / "seed.md").resolve())
        + '","scene_seed_comparison":"'
        + str((creator / "scene_seed.md").resolve())
        + '"}}',
        encoding="utf-8",
    )
    (analysis / "research_claims" / "research_claims.json").parent.mkdir(parents=True, exist_ok=True)
    (analysis / "research_claims" / "research_claims.json").write_text(
        '{"run":{"run_id":"run_anchor"},"primary_claim":{"key":"shift_robustness","title":"Failure concentration","status":"analysis_ready","summary":"Repeated runs line up.","metrics":{"worst_robustness_gap":0.158,"consistent_slice_run_count":3,"repeated_run_count":3,"consistent_slice_rate":1.0,"target_drift_jsd":0.218,"selective_risk":0.439,"abstention_rate":0.463,"stress_skip_risk":0.439},"missing_checks":["mitigate repeat_from_prev=new"]},"backup_claim":{"key":"candidate_ranking","status":"promising_but_unlocked","summary":"Ranking is promising."},"benchmark_lock":{"benchmark_id":"smokebench-v3","comparison_ready":true},"believable_submission_path":true}',
        encoding="utf-8",
    )
    (analysis / "research_claims" / "research_claims.md").write_text("# Research Claims\n", encoding="utf-8")
    (history / "benchmark_lock_smokebench-v3_manifest.md").write_text("# Benchmark\n", encoding="utf-8")
    (history / "benchmark_lock_smokebench-v3_manifest.json").write_text('{"comparison_ready": true}', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "spotify.front_door", "--output-dir", str(output_root)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "front_door_html=" in result.stdout
