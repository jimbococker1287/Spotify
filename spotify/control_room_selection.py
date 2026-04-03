from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Callable

from .control_room_history import _format_metric, _safe_float, _safe_int


def _manifest_sort_key(row: dict[str, object]) -> tuple[str, str]:
    return (
        str(row.get("timestamp", "")),
        str(row.get("run_id", "")),
    )


def _latest_manifest(manifests: list[dict[str, object]]) -> dict[str, object]:
    if not manifests:
        return {}
    return max(manifests, key=_manifest_sort_key)


def _is_promoted_manifest(manifest: dict[str, object]) -> bool:
    gate = manifest.get("champion_gate", {})
    if not isinstance(gate, dict):
        return False
    return bool(gate.get("promoted"))


def _latest_promoted_manifest(
    manifests: list[dict[str, object]],
    *,
    exclude_run_id: str | None = None,
) -> dict[str, object]:
    return max(
        (
            manifest
            for manifest in manifests
            if _is_promoted_manifest(manifest)
            and str(manifest.get("run_id", "")) != str(exclude_run_id or "")
        ),
        key=_manifest_sort_key,
        default={},
    )


def _profile_signal_rank(profile: str) -> int:
    normalized = str(profile).strip().lower()
    if normalized == "full":
        return 5
    if normalized in {"experimental", "core"}:
        return 4
    if normalized in {"small", "fast"}:
        return 3
    if normalized == "dev":
        return 1
    return 2


def _manifest_looks_like_smoke_run(manifest: dict[str, object]) -> bool:
    combined = " ".join(
        [
            str(manifest.get("run_name", "")).strip().lower(),
            str(manifest.get("run_id", "")).strip().lower(),
        ]
    )
    if not combined.strip():
        return False
    smoke_hints = ("check", "smoke", "probe", "debug", "verify")
    return any(hint in combined for hint in smoke_hints)


def _parse_manifest_timestamp(value: object) -> datetime | None:
    raw_value = str(value).strip()
    if not raw_value:
        return None
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_hours(timestamp: object, *, reference_time: datetime) -> float:
    parsed = _parse_manifest_timestamp(timestamp)
    if parsed is None:
        return float("nan")
    delta_hours = (reference_time - parsed).total_seconds() / 3600.0
    return max(delta_hours, 0.0)


def _freshness_rank(age_hours: float) -> int:
    if not math.isfinite(age_hours):
        return 0
    if age_hours <= 36:
        return 4
    if age_hours <= 24 * 7:
        return 3
    if age_hours <= 24 * 14:
        return 2
    if age_hours <= 24 * 30:
        return 1
    return 0


def _manifest_ops_signal(
    manifest: dict[str, object],
    snapshot: dict[str, dict[str, object]],
    *,
    reference_time: datetime,
) -> dict[str, object]:
    profile = str(manifest.get("profile", "")).strip().lower()
    ops_coverage = snapshot.get("ops_coverage", {})
    ops_coverage = ops_coverage if isinstance(ops_coverage, dict) else {}
    coverage_ratio = _safe_float(ops_coverage.get("coverage_ratio"))
    if not math.isfinite(coverage_ratio):
        coverage_ratio = -1.0
    backtest_rows = _safe_int(manifest.get("backtest_rows"))
    optuna_rows = _safe_int(manifest.get("optuna_rows"))
    smoke_like = _manifest_looks_like_smoke_run(manifest)
    production_profile = profile not in {"", "dev"}
    profile_rank = _profile_signal_rank(profile)
    coverage_ready = coverage_ratio >= 0.80
    promotion_status = str(
        (snapshot.get("run", {}) if isinstance(snapshot.get("run", {}), dict) else {}).get("promotion_status", "")
    )
    review_anchor_ready = bool(coverage_ready and not smoke_like and production_profile and promotion_status)
    age_hours = _age_hours(manifest.get("timestamp", ""), reference_time=reference_time)
    freshness_rank = _freshness_rank(age_hours)
    return {
        "run_id": str(manifest.get("run_id", "")),
        "run_name": str(manifest.get("run_name", "") or ""),
        "profile": profile,
        "timestamp": str(manifest.get("timestamp", "")),
        "smoke_like": bool(smoke_like),
        "production_profile": bool(production_profile),
        "profile_rank": int(profile_rank),
        "backtest_rows": int(backtest_rows),
        "optuna_rows": int(optuna_rows),
        "coverage_ratio": float(coverage_ratio),
        "available_summary_count": _safe_int(ops_coverage.get("available_summary_count"), default=0),
        "expected_summary_count": _safe_int(ops_coverage.get("expected_summary_count"), default=0),
        "coverage_ready": bool(coverage_ready),
        "review_anchor_ready": bool(review_anchor_ready),
        "age_hours": float(age_hours) if math.isfinite(age_hours) else float("nan"),
        "freshness_rank": int(freshness_rank),
        "sort_key": (
            int(coverage_ready),
            int(not smoke_like),
            int(freshness_rank),
            int(production_profile),
            int(backtest_rows > 0),
            int(profile_rank),
            float(coverage_ratio),
            int(backtest_rows),
            int(optuna_rows),
            str(manifest.get("timestamp", "")),
            str(manifest.get("run_id", "")),
        ),
    }


def _signal_ready_for_review_anchor(signal: dict[str, object]) -> bool:
    return bool(signal.get("review_anchor_ready"))


def _review_anchor_sort_key(signal: dict[str, object]) -> tuple[object, ...]:
    return (
        int(signal.get("profile_rank", 0)),
        int(signal.get("freshness_rank", 0)),
        float(signal.get("coverage_ratio", -1.0)),
        int(_safe_int(signal.get("backtest_rows"), default=0) > 0),
        str(signal.get("timestamp", "")),
        int(_safe_int(signal.get("backtest_rows"), default=0)),
        int(_safe_int(signal.get("optuna_rows"), default=0)),
        str(signal.get("run_id", "")),
    )


def _build_run_selection_summary(
    *,
    latest_observed_signal: dict[str, object],
    selected_signal: dict[str, object],
) -> dict[str, object]:
    observed_run_id = str(latest_observed_signal.get("run_id", ""))
    selected_run_id = str(selected_signal.get("run_id", ""))
    observed_matches_selected = observed_run_id == selected_run_id

    if observed_matches_selected:
        if _signal_ready_for_review_anchor(selected_signal):
            reason = "Latest observed run is a fully completed production run, so it becomes the review anchor."
        elif bool(selected_signal.get("coverage_ready")) and not bool(selected_signal.get("smoke_like")):
            reason = "Latest observed run already looks like the strongest ops candidate."
        elif not bool(selected_signal.get("smoke_like")) and bool(selected_signal.get("production_profile")):
            reason = "Latest observed run is still the strongest ops candidate, but its ops coverage is incomplete."
        else:
            reason = "No stronger ops candidate was available, so the control room stayed on the latest observed run."
    else:
        reasons: list[str] = []
        if bool(latest_observed_signal.get("smoke_like")):
            reasons.append("the latest observed run looks like a smoke/check run")
        if not bool(latest_observed_signal.get("production_profile")):
            reasons.append(
                f"the latest observed run uses the `{latest_observed_signal.get('profile', 'unknown')}` profile"
            )
        if not bool(latest_observed_signal.get("review_anchor_ready")):
            reasons.append("the latest observed run is not yet a fully completed review anchor")
        if _safe_int(selected_signal.get("profile_rank"), default=0) > _safe_int(
            latest_observed_signal.get("profile_rank"),
            default=0,
        ):
            reasons.append("the selected run uses a stronger production profile")
        if float(selected_signal.get("coverage_ratio", -1.0)) > float(
            latest_observed_signal.get("coverage_ratio", -1.0)
        ):
            reasons.append("the selected run has better ops artifact coverage")
        if _safe_int(selected_signal.get("backtest_rows"), default=0) > _safe_int(
            latest_observed_signal.get("backtest_rows"),
            default=0,
        ):
            reasons.append("the selected run has stronger backtest evidence")
        if not reasons:
            reasons.append("the selected run scored higher on the ops signal ranking")
        reason = (
            "Latest observed run was skipped because "
            + "; ".join(reasons)
            + f". Control room selected `{selected_run_id}` instead."
        )

    return {
        "selection_mode": "ops_signal_ranking",
        "observed_matches_selected": bool(observed_matches_selected),
        "selection_reason": reason,
        "latest_observed_run": latest_observed_signal,
        "selected_run": selected_signal,
    }


def _select_latest_control_room_candidate(
    manifests: list[dict[str, object]],
    *,
    reference_time: datetime,
    build_run_health_snapshot: Callable[[dict[str, object]], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    if not manifests:
        return {}, {"run": {}, "safety": {}, "qoe": {}, "ops_coverage": {}}, {
            "selection_mode": "ops_signal_ranking",
            "observed_matches_selected": True,
            "selection_reason": "No run manifests were available.",
            "latest_observed_run": {},
            "selected_run": {},
        }

    candidates: list[dict[str, object]] = []
    for manifest in manifests:
        snapshot = build_run_health_snapshot(manifest)
        signal = _manifest_ops_signal(manifest, snapshot, reference_time=reference_time)
        candidates.append(
            {
                "manifest": manifest,
                "snapshot": snapshot,
                "signal": signal,
            }
        )

    latest_observed = max(candidates, key=lambda item: _manifest_sort_key(item["manifest"]))
    ready_candidates = [item for item in candidates if _signal_ready_for_review_anchor(item["signal"])]
    if ready_candidates:
        selected = max(
            ready_candidates,
            key=lambda item: (_review_anchor_sort_key(item["signal"]), _manifest_sort_key(item["manifest"])),
        )
    else:
        selected = max(candidates, key=lambda item: (item["signal"]["sort_key"], _manifest_sort_key(item["manifest"])))
    selection = _build_run_selection_summary(
        latest_observed_signal=latest_observed["signal"],
        selected_signal=selected["signal"],
    )
    return selected["manifest"], selected["snapshot"], selection


def _status_rank(status: str) -> int:
    normalized = str(status).strip().lower()
    if normalized == "healthy":
        return 0
    if normalized == "attention":
        return 1
    if normalized == "stale":
        return 2
    if normalized == "missing":
        return 3
    if normalized == "blocked":
        return 4
    return 1


def _latest_manifest_for_profiles(
    manifests: list[dict[str, object]],
    *,
    profiles: set[str],
) -> dict[str, object]:
    candidates = [
        manifest
        for manifest in manifests
        if str(manifest.get("profile", "")).strip().lower() in profiles and not _manifest_looks_like_smoke_run(manifest)
    ]
    return max(candidates, key=_manifest_sort_key, default={})


def _build_cadence_lane(
    *,
    manifests: list[dict[str, object]],
    lane: str,
    profiles: set[str],
    target_interval_hours: int,
    reference_time: datetime,
    build_run_health_snapshot: Callable[[dict[str, object]], dict[str, dict[str, object]]],
) -> dict[str, object]:
    manifest = _latest_manifest_for_profiles(manifests, profiles=profiles)
    if not manifest:
        return {
            "lane": lane,
            "profiles": sorted(profiles),
            "target_interval_hours": int(target_interval_hours),
            "status": "missing",
            "latest_run": {},
            "hours_since_run": float("nan"),
            "overdue_hours": float("nan"),
            "recommended_command": f"make schedule-run MODE={lane}",
            "summary": f"No recent `{lane}`-lane run was found. Run `make schedule-run MODE={lane}` to restore cadence.",
        }

    snapshot = build_run_health_snapshot(manifest)
    signal = _manifest_ops_signal(manifest, snapshot, reference_time=reference_time)
    hours_since_run = _safe_float(signal.get("age_hours"))
    overdue_hours = max(hours_since_run - float(target_interval_hours), 0.0) if math.isfinite(hours_since_run) else float("nan")

    if not math.isfinite(hours_since_run):
        status = "attention"
        summary = f"The latest `{lane}`-lane run is missing a readable timestamp, so cadence could not be verified."
    elif hours_since_run > float(target_interval_hours) * 2.0:
        status = "stale"
        summary = (
            f"The `{lane}` lane is stale at `{hours_since_run:.1f}` hours since `{signal.get('run_id', '')}`. "
            f"Restore cadence with `make schedule-run MODE={lane}`."
        )
    elif hours_since_run > float(target_interval_hours):
        status = "attention"
        summary = (
            f"The `{lane}` lane is slipping at `{hours_since_run:.1f}` hours since `{signal.get('run_id', '')}`. "
            f"Plan `make schedule-run MODE={lane}` soon."
        )
    elif float(signal.get("coverage_ratio", 0.0)) < 0.8:
        status = "attention"
        summary = (
            f"The latest `{lane}`-lane run is recent, but ops coverage is only `{_format_metric(signal.get('coverage_ratio'))}`. "
            f"Backfill analysis before treating it as the cadence anchor."
        )
    else:
        status = "healthy"
        summary = (
            f"The `{lane}` lane is healthy with `{signal.get('run_id', '')}` at `{hours_since_run:.1f}` hours old "
            f"and coverage `{_format_metric(signal.get('coverage_ratio'))}`."
        )

    return {
        "lane": lane,
        "profiles": sorted(profiles),
        "target_interval_hours": int(target_interval_hours),
        "status": status,
        "latest_run": {
            "run_id": str(signal.get("run_id", "")),
            "profile": str(signal.get("profile", "")),
            "timestamp": str(signal.get("timestamp", "")),
            "coverage_ratio": _safe_float(signal.get("coverage_ratio")),
        },
        "hours_since_run": hours_since_run,
        "overdue_hours": overdue_hours,
        "recommended_command": f"make schedule-run MODE={lane}",
        "summary": summary,
    }


__all__ = [
    "_build_cadence_lane",
    "_freshness_rank",
    "_is_promoted_manifest",
    "_latest_manifest",
    "_latest_manifest_for_profiles",
    "_latest_promoted_manifest",
    "_manifest_looks_like_smoke_run",
    "_manifest_sort_key",
    "_parse_manifest_timestamp",
    "_select_latest_control_room_candidate",
    "_status_rank",
]
