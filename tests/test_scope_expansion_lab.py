from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spotify.scope_expansion_lab import build_scope_expansion_lab


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_scope_expansion_lab_scores_four_branches_and_queue(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.scope_expansion_lab")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    output_dir = tmp_path / "outputs"

    warehouse_root = output_dir / "analytics" / "warehouse"
    _write_json(
        warehouse_root / "warehouse_manifest.json",
        {
            "quality": {
                "summary": {
                    "asset_count": 34,
                    "empty_asset_count": 2,
                    "row_count_anomaly_count": 0,
                    "branch_backed_freshness_status_counts": {"fresh": 5, "attention": 1},
                }
            },
            "lineage_graph": {"edges": [{"upstream_asset": "raw_streaming_history", "downstream_asset": "listener_daily_activity"}] * 18},
        },
    )
    _write_json(warehouse_root / "warehouse_lineage.json", {"summary": {"asset_count": 34}})
    _write_json(warehouse_root / "warehouse_verification.json", {"status": "pass"})

    quant_root = output_dir / "analysis" / "quant_decision_lab"
    _write_csv(
        quant_root / "model_decision_frontier.csv",
        [
            {"model_name": "retrieval_reranker", "utility_score": 0.78},
            {"model_name": "mlp", "utility_score": 0.52},
        ],
    )
    _write_csv(
        quant_root / "policy_decision_frontier.csv",
        [{"policy_name": "safe_global", "utility_score": 0.67}],
    )
    _write_csv(
        quant_root / "scenario_utility_simulation.csv",
        [
            {
                "model_name": "retrieval_reranker",
                "policy_name": "safe_global",
                "scenario": "baseline",
                "utility_score": 0.81,
                "high_drift_context": False,
                "high_skip_context": False,
            },
            {
                "model_name": "retrieval_reranker",
                "policy_name": "safe_routed",
                "scenario": "evening_drift",
                "utility_score": 0.74,
                "high_drift_context": True,
                "high_skip_context": True,
            },
        ],
    )
    _write_json(quant_root / "archetype_decision_bridge.json", {"status": "ok"})
    _write_json(quant_root / "quant_decision_brief.json", {"status": "ok"})

    creator_root = output_dir / "analysis" / "creator_market_intelligence"
    _write_json(
        creator_root / "creator_market_manifest.json",
        {
            "report_family_count": 3,
            "complete_report_family_count": 2,
            "partial_report_family_count": 1,
        },
    )
    _write_json(
        creator_root / "creator_market_brief.json",
        {"top_scene": {"scene_name": "scene-neon"}},
    )
    _write_csv(
        creator_root / "creator_market_trend_deltas.csv",
        [
            {"signal_type": "rising_scene", "severity": "high"},
            {"signal_type": "repeated_opportunity_lane", "severity": "medium"},
        ],
    )
    _write_csv(creator_root / "scene_market_pulse.csv", [{"scene_name": "scene-neon", "momentum_score": 0.71}])
    _write_csv(creator_root / "opportunity_lane_atlas.csv", [{"scene_name": "scene-neon"}])

    research_root = output_dir / "analysis" / "research_platform_lab"
    _write_json(
        research_root / "research_platform_maturity.json",
        {
            "claim_blocked_count": 2,
            "incomplete_benchmark_lock_count": 1,
            "stale_claim_artifact_count": 1,
            "stale_benchmark_count": 0,
            "top_blocker": "Finish repeated deep comparator coverage.",
        },
    )
    _write_json(
        research_root / "research_next_experiments.json",
        {
            "experiments": [
                {"claim_key": "candidate_ranking", "experiment_type": "deep_comparator_benchmark_coverage"},
                {"claim_key": "risk_aware_abstention", "experiment_type": "risk_coverage_tradeoff_evidence"},
            ]
        },
    )
    _write_csv(
        research_root / "research_claim_registry.csv",
        [
            {"claim_key": "candidate_ranking", "claim_readiness_status": "blocked"},
            {"claim_key": "shift_robustness", "claim_readiness_status": "ready"},
        ],
    )
    _write_csv(
        research_root / "benchmark_lock_atlas.csv",
        [
            {"benchmark_id": "smokebench", "comparison_status": "incomplete"},
            {"benchmark_id": "stablebench", "comparison_status": "ready"},
        ],
    )
    _write_csv(research_root / "run_research_registry.csv", [{"run_id": "run_a"}])

    paths = build_scope_expansion_lab(output_dir=output_dir, logger=logger)

    result_root = output_dir / "analysis" / "scope_expansion"
    scorecard = pd.read_csv(result_root / "branch_expansion_scorecard.csv")
    queue = pd.read_csv(result_root / "branch_expansion_implementation_queue.csv")
    strategy_cards = pd.read_csv(result_root / "branch_strategy_cards.csv")
    cockpit = json.loads((result_root / "branch_development_cockpit.json").read_text(encoding="utf-8"))
    manifest = json.loads((result_root / "scope_expansion_manifest.json").read_text(encoding="utf-8"))
    markdown = (result_root / "branch_expansion_scorecard.md").read_text(encoding="utf-8")
    strategy_markdown = (result_root / "branch_strategy_cards.md").read_text(encoding="utf-8")
    research_strategy_markdown = (result_root / "strategy_cards" / "research_platform.md").read_text(encoding="utf-8")
    cockpit_markdown = (result_root / "branch_development_cockpit.md").read_text(encoding="utf-8")

    assert result_root / "branch_expansion_scorecard.csv" in paths
    assert result_root / "branch_strategy_cards.csv" in paths
    assert result_root / "branch_development_cockpit.md" in paths
    assert set(scorecard["branch_key"]) == {
        "analytics_engineering",
        "data_science_quant",
        "creator_market_intelligence",
        "research_platform",
    }
    assert len(queue.index) == 4
    assert len(strategy_cards.index) == 4
    assert queue.iloc[0]["branch_key"] == "research_platform"
    assert queue["rank"].tolist() == [1, 2, 3, 4]
    assert manifest["branch_count"] == 4
    assert manifest["queue_count"] == 4
    assert manifest["strategy_card_count"] == 4
    assert manifest["tables"]["branch_strategy_cards"]["row_count"] == 4
    assert manifest["tables"]["branch_strategy_cards"]["individual_markdown_paths"]["research_platform"].endswith(
        "research_platform.md"
    )
    assert manifest["reports"]["branch_development_cockpit"]["markdown_path"].endswith(
        "branch_development_cockpit.md"
    )
    assert cockpit["summary"]["branch_count"] == 4
    assert cockpit["summary"]["top_queue_branch_key"] == "research_platform"
    assert cockpit["recommended_validation_sequence"][0].endswith(
        "tests/test_research_platform_lab.py tests/test_scope_expansion_lab.py"
    )
    assert cockpit["lanes"][0]["branch_key"] == "data_science_quant"
    research_lane = next(lane for lane in cockpit["lanes"] if lane["branch_key"] == "research_platform")
    assert research_lane["development_mode"] == "stabilize"
    assert "Scope Expansion Lab" in markdown
    assert "Data Science + Quant" in markdown
    assert "Branch Strategy Cards" in strategy_markdown
    assert "Research Platform Strategy Card" in research_strategy_markdown
    assert "Validation command" in research_strategy_markdown
    assert "Branch Development Cockpit" in cockpit_markdown
    assert "Recommended Validation Sequence" in cockpit_markdown
    assert "make benchmark-lock && make research-claims && make research-platform-lab" in cockpit_markdown
    assert "retrieval_reranker / safe_global / baseline" in scorecard.loc[
        scorecard["branch_key"].eq("data_science_quant"), "top_signal"
    ].iloc[0]
