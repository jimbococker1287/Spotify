from __future__ import annotations

import math
from typing import Iterable


_CONTRACT_VERSION = "2026-06-v1"
_MINIMUM_SUPPORT_COUNT = 20
_MINIMUM_REPORT_FAMILIES = 2
_MAXIMUM_SCORE_RANGE = 0.12
_MINIMUM_LABEL_AGREEMENT = 0.67
_MAXIMUM_SOURCE_AGE_DAYS = 90


def describe_creator_evidence_contract() -> dict[str, object]:
    return {
        "contract_version": _CONTRACT_VERSION,
        "evidence_grades": ["publishable", "watch_only", "suppress"],
        "verified_grade": "publishable",
        "minimum_support_count": _MINIMUM_SUPPORT_COUNT,
        "minimum_report_families": _MINIMUM_REPORT_FAMILIES,
        "maximum_opportunity_score_range": _MAXIMUM_SCORE_RANGE,
        "minimum_scene_driver_agreement": _MINIMUM_LABEL_AGREEMENT,
        "maximum_source_age_days": _MAXIMUM_SOURCE_AGE_DAYS,
        "publication_policy": (
            "Publishable requires every evidence gate to pass. Watch-only is used when evidence is incomplete "
            "or recurrence/stability is not yet established. Suppress is used for contradictory core evidence, "
            "observed support below the minimum, zero catalog or release coverage, or missing claim limitations."
        ),
        "claim_policy": (
            "Claims must identify saved-artifact evidence as directional and explicitly say the result is not a forecast."
        ),
    }


def _finite_values(values: Iterable[object]) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            output.append(numeric)
    return output


def _agreement_ratio(values: Iterable[object]) -> float | None:
    normalized = [
        str(value).strip().casefold()
        for value in values
        if value is not None
        and str(value).strip()
        and str(value).strip().casefold() not in {"nan", "none", "null"}
    ]
    if not normalized:
        return None
    counts: dict[str, int] = {}
    for value in normalized:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.values()) / len(normalized)


def _gate(
    key: str,
    status: str,
    detail: str,
    *,
    observed: object = None,
    threshold: object = None,
) -> dict[str, object]:
    return {
        "key": key,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def evaluate_creator_opportunity_evidence(
    *,
    artist_name: str,
    occurrence_count: int,
    family_count: int,
    score_values: Iterable[object],
    scene_values: Iterable[object],
    driver_values: Iterable[object],
    support_values: Iterable[object],
    catalog_metadata_count: int,
    release_metadata_count: int,
    timestamped_occurrence_count: int,
    latest_source_age_days: float | None,
    claim_text: str,
    limitations: Iterable[object],
) -> dict[str, object]:
    contract = describe_creator_evidence_contract()
    occurrence_count = max(0, int(occurrence_count))
    family_count = max(0, int(family_count))
    scores = _finite_values(score_values)
    support = _finite_values(support_values)
    limitation_texts = [str(value).strip() for value in limitations if str(value).strip()]
    gates: list[dict[str, object]] = []

    core_ready = bool(str(artist_name).strip()) and occurrence_count > 0 and len(scores) == occurrence_count
    gates.append(
        _gate(
            "core_opportunity_fields",
            "pass" if core_ready else "fail",
            (
                "Artist identity and a finite opportunity score are present for every saved occurrence."
                if core_ready
                else "Artist identity or one or more saved opportunity scores are missing."
            ),
            observed={"occurrences": occurrence_count, "finite_scores": len(scores)},
            threshold={"finite_scores": occurrence_count},
        )
    )

    recurrence_ready = family_count >= int(contract["minimum_report_families"])
    gates.append(
        _gate(
            "cross_family_recurrence",
            "pass" if recurrence_ready else "watch",
            (
                f"Opportunity recurs across {family_count} report families."
                if recurrence_ready
                else f"Opportunity appears in only {family_count} report family; recurrence is not established."
            ),
            observed=family_count,
            threshold=contract["minimum_report_families"],
        )
    )

    score_range = max(scores) - min(scores) if len(scores) >= 2 else None
    scene_agreement = _agreement_ratio(scene_values)
    driver_agreement = _agreement_ratio(driver_values)
    stability_ready = (
        recurrence_ready
        and score_range is not None
        and score_range <= float(contract["maximum_opportunity_score_range"])
        and scene_agreement is not None
        and scene_agreement >= float(contract["minimum_scene_driver_agreement"])
        and driver_agreement is not None
        and driver_agreement >= float(contract["minimum_scene_driver_agreement"])
    )
    gates.append(
        _gate(
            "cross_family_stability",
            "pass" if stability_ready else "watch",
            (
                "Opportunity score, scene, and primary driver are stable across recurring report families."
                if stability_ready
                else "Cross-family score, scene, or driver stability is unavailable or below the contract threshold."
            ),
            observed={
                "score_range": score_range,
                "scene_agreement": scene_agreement,
                "driver_agreement": driver_agreement,
            },
            threshold={
                "maximum_score_range": contract["maximum_opportunity_score_range"],
                "minimum_label_agreement": contract["minimum_scene_driver_agreement"],
            },
        )
    )

    if not support:
        support_status = "watch"
        support_detail = "No direct support count is available; the score is not treated as a substitute."
        minimum_support = None
    else:
        minimum_support = min(support)
        support_status = "pass" if minimum_support >= int(contract["minimum_support_count"]) else "fail"
        support_detail = (
            f"Minimum observed support is {minimum_support:.0f}."
            if support_status == "pass"
            else f"Minimum observed support is {minimum_support:.0f}, below the publication minimum."
        )
    gates.append(
        _gate(
            "minimum_support",
            support_status,
            support_detail,
            observed=minimum_support,
            threshold=contract["minimum_support_count"],
        )
    )

    catalog_coverage = catalog_metadata_count / occurrence_count if occurrence_count else 0.0
    catalog_status = "pass" if catalog_coverage >= 1.0 else ("fail" if catalog_coverage <= 0.0 else "watch")
    gates.append(
        _gate(
            "catalog_metadata_coverage",
            catalog_status,
            f"Catalog identity or public catalog metadata covers {catalog_metadata_count} of {occurrence_count} occurrences.",
            observed=catalog_coverage,
            threshold=1.0,
        )
    )

    release_coverage = release_metadata_count / occurrence_count if occurrence_count else 0.0
    release_status = "pass" if release_coverage >= 1.0 else ("fail" if release_coverage <= 0.0 else "watch")
    gates.append(
        _gate(
            "release_metadata_coverage",
            release_status,
            f"Release date or cadence metadata covers {release_metadata_count} of {occurrence_count} occurrences.",
            observed=release_coverage,
            threshold=1.0,
        )
    )

    timestamp_coverage = timestamped_occurrence_count / occurrence_count if occurrence_count else 0.0
    if latest_source_age_days is None or timestamp_coverage <= 0.0:
        freshness_status = "watch"
        freshness_detail = "Saved artifact timestamps are unavailable, so evidence freshness cannot be established."
    elif latest_source_age_days < 0:
        freshness_status = "fail"
        freshness_detail = "The latest saved artifact timestamp is after the evaluation anchor."
    elif (
        timestamp_coverage >= 1.0
        and latest_source_age_days <= float(contract["maximum_source_age_days"])
    ):
        freshness_status = "pass"
        freshness_detail = f"All occurrences are timestamped and the latest source is {latest_source_age_days:.1f} days old."
    else:
        freshness_status = "watch"
        freshness_detail = (
            f"Timestamp coverage is {timestamp_coverage:.3f} and the latest source is "
            f"{latest_source_age_days:.1f} days old."
        )
    gates.append(
        _gate(
            "evidence_freshness",
            freshness_status,
            freshness_detail,
            observed={
                "timestamp_coverage": timestamp_coverage,
                "latest_source_age_days": latest_source_age_days,
            },
            threshold={
                "timestamp_coverage": 1.0,
                "maximum_source_age_days": contract["maximum_source_age_days"],
            },
        )
    )

    normalized_claim = str(claim_text).strip().casefold()
    claim_ready = (
        bool(normalized_claim)
        and "saved" in normalized_claim
        and "directional" in normalized_claim
        and "not a forecast" in normalized_claim
        and bool(limitation_texts)
    )
    gates.append(
        _gate(
            "claim_and_limitation_wording",
            "pass" if claim_ready else "fail",
            (
                "Claim is explicitly directional, tied to saved evidence, and paired with limitations."
                if claim_ready
                else "Claim wording must cite saved evidence, say directional and not a forecast, and include limitations."
            ),
            observed={"claim_present": bool(normalized_claim), "limitation_count": len(limitation_texts)},
            threshold={"minimum_limitation_count": 1},
        )
    )

    statuses = {str(gate["status"]) for gate in gates}
    if "fail" in statuses:
        evidence_grade = "suppress"
    elif statuses == {"pass"}:
        evidence_grade = "publishable"
    else:
        evidence_grade = "watch_only"
    return {
        "contract_version": contract["contract_version"],
        "evidence_grade": evidence_grade,
        "verified": evidence_grade == contract["verified_grade"],
        "gates": gates,
    }
