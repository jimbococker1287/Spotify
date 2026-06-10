from __future__ import annotations

from spotify.creator_benchmark_contract import describe_creator_evidence_contract
from spotify.creator_benchmark_contract import evaluate_creator_opportunity_evidence


def _claim() -> str:
    return (
        "Saved creator artifacts provide a directional opportunity signal; "
        "this is not a forecast of audience or commercial outcomes."
    )


def test_creator_evidence_contract_exposes_conservative_publication_thresholds() -> None:
    contract = describe_creator_evidence_contract()

    assert contract["contract_version"] == "2026-06-v1"
    assert contract["minimum_support_count"] == 20
    assert contract["minimum_report_families"] == 2
    assert contract["verified_grade"] == "publishable"
    assert contract["evidence_grades"] == ["publishable", "watch_only", "suppress"]


def test_creator_evidence_contract_marks_fully_supported_recurrence_publishable() -> None:
    result = evaluate_creator_opportunity_evidence(
        artist_name="Artist Ready",
        occurrence_count=2,
        family_count=2,
        score_values=[0.41, 0.45],
        scene_values=["scene-a", "scene-a"],
        driver_values=["seed_adjacency", "seed_adjacency"],
        support_values=[80, 120],
        catalog_metadata_count=2,
        release_metadata_count=2,
        timestamped_occurrence_count=2,
        latest_source_age_days=12.0,
        claim_text=_claim(),
        limitations=["Saved observations do not establish causality."],
    )

    assert result["evidence_grade"] == "publishable"
    assert result["verified"] is True
    assert {gate["status"] for gate in result["gates"]} == {"pass"}


def test_creator_evidence_contract_downgrades_missing_and_failed_evidence() -> None:
    watch_result = evaluate_creator_opportunity_evidence(
        artist_name="Artist Watch",
        occurrence_count=1,
        family_count=1,
        score_values=[0.40],
        scene_values=["scene-a"],
        driver_values=["migration_capture"],
        support_values=[],
        catalog_metadata_count=1,
        release_metadata_count=1,
        timestamped_occurrence_count=1,
        latest_source_age_days=5.0,
        claim_text=_claim(),
        limitations=["Support counts are unavailable."],
    )
    suppress_result = evaluate_creator_opportunity_evidence(
        artist_name="Artist Suppress",
        occurrence_count=2,
        family_count=2,
        score_values=[0.40, 0.41],
        scene_values=["scene-a", "scene-a"],
        driver_values=["release_whitespace", "release_whitespace"],
        support_values=[5, 7],
        catalog_metadata_count=2,
        release_metadata_count=0,
        timestamped_occurrence_count=2,
        latest_source_age_days=5.0,
        claim_text="Strong opportunity.",
        limitations=[],
    )

    assert watch_result["evidence_grade"] == "watch_only"
    assert watch_result["verified"] is False
    watch_statuses = {gate["key"]: gate["status"] for gate in watch_result["gates"]}
    assert watch_statuses["minimum_support"] == "watch"
    assert watch_statuses["cross_family_recurrence"] == "watch"

    assert suppress_result["evidence_grade"] == "suppress"
    suppress_statuses = {gate["key"]: gate["status"] for gate in suppress_result["gates"]}
    assert suppress_statuses["minimum_support"] == "fail"
    assert suppress_statuses["release_metadata_coverage"] == "fail"
    assert suppress_statuses["claim_and_limitation_wording"] == "fail"
