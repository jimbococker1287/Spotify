from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
from statistics import median

from .run_artifacts import safe_read_json
from .track_expansion_tuning import normalize_track_expansion_model_name


TRACK_TUNING_PLAN_SCHEMA_VERSION = "track-tuning-plan-v1"

MODEL_FAMILIES: Mapping[str, str] = {
    "session_cooccurrence": "candidate_retrieval",
    "ease": "candidate_retrieval",
    "meantime": "sequence_encoder",
    "mmoe": "multitask",
    "ple": "multitask",
    "dcn_v2": "reranker",
}

FAMILY_MINIMUM_TOTAL_TRIALS: Mapping[str, int] = {
    "candidate_retrieval": 12,
    "sequence_encoder": 8,
    "multitask": 8,
    "reranker": 8,
    "unknown": 6,
}

FAMILY_MAX_INCREMENTAL_TRIALS: Mapping[str, int] = {
    "candidate_retrieval": 16,
    "sequence_encoder": 8,
    "multitask": 8,
    "reranker": 6,
    "unknown": 6,
}


@dataclass(frozen=True)
class TrackTuningPlanConfig:
    min_completed_trials: int = 3
    max_recommended_total_trials: int = 40
    cheap_trial_seconds: float = 2.0
    expensive_trial_seconds: float = 10.0
    good_metric_value: float = 0.2
    weak_metric_value: float = 0.05
    include_zero_budget_models: bool = True

    def validate(self) -> None:
        if self.min_completed_trials < 1:
            raise ValueError("min_completed_trials must be positive.")
        if self.max_recommended_total_trials < self.min_completed_trials:
            raise ValueError("max_recommended_total_trials must be at least min_completed_trials.")
        if self.cheap_trial_seconds <= 0 or self.expensive_trial_seconds <= 0:
            raise ValueError("runtime thresholds must be positive.")
        if self.cheap_trial_seconds > self.expensive_trial_seconds:
            raise ValueError("cheap_trial_seconds cannot exceed expensive_trial_seconds.")
        if self.weak_metric_value < 0 or self.good_metric_value < 0:
            raise ValueError("metric thresholds cannot be negative.")
        if self.weak_metric_value > self.good_metric_value:
            raise ValueError("weak_metric_value cannot exceed good_metric_value.")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_completed_trials": self.min_completed_trials,
            "max_recommended_total_trials": self.max_recommended_total_trials,
            "cheap_trial_seconds": self.cheap_trial_seconds,
            "expensive_trial_seconds": self.expensive_trial_seconds,
            "good_metric_value": self.good_metric_value,
            "weak_metric_value": self.weak_metric_value,
            "include_zero_budget_models": self.include_zero_budget_models,
        }


@dataclass(frozen=True)
class TrackTuningModelPlan:
    model_name: str
    model_family: str
    metric_name: str | None
    direction: str
    completed_trials: int
    total_trials: int
    failed_trials: int
    pruned_trials: int
    best_value: float | None
    median_trial_seconds: float | None
    estimated_incremental_seconds: float | None
    recommended_additional_trials: int
    recommended_total_trials: int
    priority: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "metric_name": self.metric_name,
            "direction": self.direction,
            "completed_trials": self.completed_trials,
            "total_trials": self.total_trials,
            "failed_trials": self.failed_trials,
            "pruned_trials": self.pruned_trials,
            "best_value": self.best_value,
            "median_trial_seconds": self.median_trial_seconds,
            "estimated_incremental_seconds": self.estimated_incremental_seconds,
            "recommended_additional_trials": self.recommended_additional_trials,
            "recommended_total_trials": self.recommended_total_trials,
            "priority": self.priority,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class TrackTuningCommand:
    model_names: tuple[str, ...]
    target_total_trials: int
    command: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_names": list(self.model_names),
            "target_total_trials": self.target_total_trials,
            "command": self.command,
        }


@dataclass(frozen=True)
class TrackTuningPlan:
    status: str
    generated_at: str
    schema_version: str
    next_pass_manifest_path: str | None
    optuna_summary_path: str | None
    source_status: str
    source_generated_at: str | None
    model_plans: tuple[TrackTuningModelPlan, ...]
    make_command: str | None
    make_commands: tuple[TrackTuningCommand, ...]
    config: TrackTuningPlanConfig
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
            "next_pass_manifest_path": self.next_pass_manifest_path,
            "optuna_summary_path": self.optuna_summary_path,
            "source_status": self.source_status,
            "source_generated_at": self.source_generated_at,
            "model_plans": [plan.to_dict() for plan in self.model_plans],
            "make_command": self.make_command,
            "make_commands": [command.to_dict() for command in self.make_commands],
            "config": self.config.to_dict(),
            "warnings": list(self.warnings),
        }


def build_track_tuning_plan(
    *,
    next_pass_manifest_path: Path | str | None = None,
    optuna_summary_path: Path | str | None = None,
    config: TrackTuningPlanConfig | None = None,
) -> TrackTuningPlan:
    plan_config = config or TrackTuningPlanConfig()
    plan_config.validate()
    manifest_path = Path(next_pass_manifest_path).expanduser() if next_pass_manifest_path is not None else None
    summary_path = Path(optuna_summary_path).expanduser() if optuna_summary_path is not None else None
    warnings: list[str] = []
    manifest: Mapping[str, object] = {}

    if manifest_path is not None:
        manifest_payload = safe_read_json(manifest_path, default={})
        if isinstance(manifest_payload, Mapping):
            manifest = manifest_payload
        else:
            warnings.append(f"Next-pass manifest was not a JSON object: {manifest_path}")

    if summary_path is None:
        summary_path = _summary_path_from_manifest(manifest, base_path=manifest_path)
    summary = _load_optuna_summary(summary_path, manifest, warnings)
    if not summary:
        return _empty_plan(
            config=plan_config,
            manifest_path=manifest_path,
            summary_path=summary_path,
            warnings=tuple(warnings or ("No Optuna summary was found.",)),
        )

    model_plans = tuple(
        _plan_study(study, config=plan_config)
        for study in _study_rows(summary)
    )
    visible_model_plans = tuple(
        model_plan
        for model_plan in model_plans
        if plan_config.include_zero_budget_models or model_plan.recommended_additional_trials > 0
    )
    commands = _make_commands(visible_model_plans)
    primary_command = commands[0].command if commands else None
    status = "ready" if any(plan.recommended_additional_trials > 0 for plan in model_plans) else "complete"

    return TrackTuningPlan(
        status=status,
        generated_at=_utc_now(),
        schema_version=TRACK_TUNING_PLAN_SCHEMA_VERSION,
        next_pass_manifest_path=str(manifest_path) if manifest_path is not None else None,
        optuna_summary_path=str(summary_path) if summary_path is not None else None,
        source_status=str(summary.get("status", "unknown")),
        source_generated_at=_optional_str(summary.get("generated_at")),
        model_plans=visible_model_plans,
        make_command=primary_command,
        make_commands=commands,
        config=plan_config,
        warnings=tuple(warnings),
    )


def write_track_tuning_plan(
    path: Path | str,
    *,
    next_pass_manifest_path: Path | str | None = None,
    optuna_summary_path: Path | str | None = None,
    config: TrackTuningPlanConfig | None = None,
) -> TrackTuningPlan:
    plan = build_track_tuning_plan(
        next_pass_manifest_path=next_pass_manifest_path,
        optuna_summary_path=optuna_summary_path,
        config=config,
    )
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def _empty_plan(
    *,
    config: TrackTuningPlanConfig,
    manifest_path: Path | None,
    summary_path: Path | None,
    warnings: tuple[str, ...],
) -> TrackTuningPlan:
    return TrackTuningPlan(
        status="blocked",
        generated_at=_utc_now(),
        schema_version=TRACK_TUNING_PLAN_SCHEMA_VERSION,
        next_pass_manifest_path=str(manifest_path) if manifest_path is not None else None,
        optuna_summary_path=str(summary_path) if summary_path is not None else None,
        source_status="missing",
        source_generated_at=None,
        model_plans=(),
        make_command=None,
        make_commands=(),
        config=config,
        warnings=warnings,
    )


def _summary_path_from_manifest(
    manifest: Mapping[str, object],
    *,
    base_path: Path | None,
) -> Path | None:
    optuna_stage = _mapping_at(manifest, "stages", "optuna_tuning")
    tuner_output = _mapping_at(optuna_stage, "tuners", "track_expansion", "output")
    summary_path = _optional_str(tuner_output.get("summary_path"))
    if summary_path:
        return _resolve_relative_path(summary_path, base_path)
    output_dir = _mapping_at(manifest, "config").get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        candidate = Path(output_dir).expanduser() / "analysis" / "recommender_expansion" / "next_pass"
        return candidate / "stages" / "optuna_track_expansion" / "track_expansion_optuna_summary.json"
    return None


def _load_optuna_summary(
    summary_path: Path | None,
    manifest: Mapping[str, object],
    warnings: list[str],
) -> Mapping[str, object]:
    if summary_path is not None:
        payload = safe_read_json(summary_path, default={})
        if isinstance(payload, Mapping) and payload:
            return payload
        warnings.append(f"Optuna summary was not found or was unreadable: {summary_path}")
    embedded = _mapping_at(manifest, "stages", "optuna_tuning", "tuners", "track_expansion", "output")
    if embedded:
        warnings.append("Using embedded Optuna summary from the next-pass manifest.")
    return embedded


def _plan_study(
    study: Mapping[str, object],
    *,
    config: TrackTuningPlanConfig,
) -> TrackTuningModelPlan:
    model_name = _normalize_model_name(study.get("model_name"))
    family = MODEL_FAMILIES.get(model_name, "unknown")
    best = study.get("best_trial") if isinstance(study.get("best_trial"), Mapping) else {}
    metric = study.get("metric") if isinstance(study.get("metric"), Mapping) else {}
    best_value = _finite_float(best.get("value") if isinstance(best, Mapping) else None)
    metric_name = _optional_str(best.get("metric_name") if isinstance(best, Mapping) else metric.get("name"))
    direction = str(best.get("direction") if isinstance(best, Mapping) else metric.get("direction", "maximize"))
    completed = _non_negative_int(study.get("completed_trials"))
    total = _non_negative_int(study.get("total_trials"))
    failed = _non_negative_int(study.get("failed_trials"))
    pruned = _non_negative_int(study.get("pruned_trials"))
    durations = _trial_durations(study.get("trials"))
    median_seconds = median(durations) if durations else None
    target_total, reasons = _recommended_total_trials(
        model_name=model_name,
        family=family,
        completed_trials=completed,
        total_trials=total,
        failed_trials=failed,
        pruned_trials=pruned,
        best_value=best_value,
        duration_seconds=median_seconds,
        config=config,
    )
    additional = max(0, target_total - total)
    estimated_seconds = float(additional * median_seconds) if median_seconds is not None else None
    priority = _priority(additional, target_total, total, family)
    return TrackTuningModelPlan(
        model_name=model_name,
        model_family=family,
        metric_name=metric_name,
        direction=direction if direction in {"maximize", "minimize"} else "maximize",
        completed_trials=completed,
        total_trials=total,
        failed_trials=failed,
        pruned_trials=pruned,
        best_value=best_value,
        median_trial_seconds=float(median_seconds) if median_seconds is not None else None,
        estimated_incremental_seconds=estimated_seconds,
        recommended_additional_trials=additional,
        recommended_total_trials=target_total,
        priority=priority,
        reasons=tuple(reasons),
    )


def _recommended_total_trials(
    *,
    model_name: str,
    family: str,
    completed_trials: int,
    total_trials: int,
    failed_trials: int,
    pruned_trials: int,
    best_value: float | None,
    duration_seconds: float | None,
    config: TrackTuningPlanConfig,
) -> tuple[int, list[str]]:
    reasons: list[str] = []
    minimum = max(config.min_completed_trials, FAMILY_MINIMUM_TOTAL_TRIALS.get(family, FAMILY_MINIMUM_TOTAL_TRIALS["unknown"]))
    target = minimum
    if completed_trials < config.min_completed_trials:
        target = max(target, total_trials + (config.min_completed_trials - completed_trials))
        reasons.append(f"only {completed_trials} completed trial(s); need a basic signal floor")
    if best_value is None:
        target += 3
        reasons.append("no completed best value yet")
    elif best_value < config.weak_metric_value:
        target += 4
        reasons.append(f"best value {best_value:.4f} is below weak-signal threshold")
    elif best_value >= config.good_metric_value:
        target = max(target - 2, config.min_completed_trials)
        reasons.append(f"best value {best_value:.4f} is already a stronger signal")
    else:
        reasons.append(f"best value {best_value:.4f} is promising but still early")
    if duration_seconds is not None:
        if duration_seconds <= config.cheap_trial_seconds:
            target += 4
            reasons.append(f"cheap median trial runtime ({duration_seconds:.2f}s)")
        elif duration_seconds >= config.expensive_trial_seconds:
            target = min(target, total_trials + FAMILY_MAX_INCREMENTAL_TRIALS.get(family, 6))
            reasons.append(f"expensive median trial runtime ({duration_seconds:.2f}s)")
        else:
            reasons.append(f"moderate median trial runtime ({duration_seconds:.2f}s)")
    else:
        reasons.append("missing per-trial runtime metadata")
    if failed_trials:
        target += min(3, failed_trials)
        reasons.append(f"{failed_trials} failed trial(s) need retry budget")
    if pruned_trials > completed_trials:
        target += 2
        reasons.append("pruning exceeded completed trials")
    if model_name == "dcn_v2" and best_value is not None and best_value >= config.good_metric_value:
        target += 2
        reasons.append("DCN-V2 is the current reranker candidate; add confirmation trials")
    max_increment = FAMILY_MAX_INCREMENTAL_TRIALS.get(family, FAMILY_MAX_INCREMENTAL_TRIALS["unknown"])
    target = min(target, total_trials + max_increment)
    target = min(target, config.max_recommended_total_trials)
    target = max(target, total_trials)
    return int(target), reasons


def _make_commands(model_plans: Sequence[TrackTuningModelPlan]) -> tuple[TrackTuningCommand, ...]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for plan in model_plans:
        if plan.recommended_additional_trials <= 0:
            continue
        grouped[plan.recommended_total_trials].append(plan.model_name)
    commands: list[TrackTuningCommand] = []
    for total_trials in sorted(grouped):
        models = tuple(sorted(grouped[total_trials]))
        extra_args = f"--tuning-models {','.join(models)} --tuning-trials {total_trials}"
        command = f"make train-recommender-next-pass EXTRA_ARGS={shlex.quote(extra_args)}"
        commands.append(
            TrackTuningCommand(
                model_names=models,
                target_total_trials=total_trials,
                command=command,
            )
        )
    return tuple(commands)


def _study_rows(summary: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    studies = summary.get("studies", ())
    if not isinstance(studies, Sequence) or isinstance(studies, (str, bytes)):
        return ()
    return tuple(study for study in studies if isinstance(study, Mapping))


def _trial_durations(trials: object) -> tuple[float, ...]:
    if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
        return ()
    durations: list[float] = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        value = _finite_float(trial.get("duration_seconds"))
        if value is not None and value >= 0.0:
            durations.append(value)
    return tuple(durations)


def _mapping_at(payload: Mapping[str, object], *keys: str) -> Mapping[str, object]:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _resolve_relative_path(value: str, base_path: Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or base_path is None:
        return path
    return base_path.parent / path


def _normalize_model_name(value: object) -> str:
    try:
        return normalize_track_expansion_model_name(str(value))
    except ValueError:
        return str(value or "unknown").strip().lower().replace(" ", "_") or "unknown"


def _priority(additional: int, target: int, total: int, family: str) -> str:
    if additional <= 0:
        return "none"
    if family == "reranker" or target - total >= 6:
        return "high"
    if target - total >= 3:
        return "medium"
    return "low"


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    return 0


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "FAMILY_MAX_INCREMENTAL_TRIALS",
    "FAMILY_MINIMUM_TOTAL_TRIALS",
    "MODEL_FAMILIES",
    "TRACK_TUNING_PLAN_SCHEMA_VERSION",
    "TrackTuningCommand",
    "TrackTuningModelPlan",
    "TrackTuningPlan",
    "TrackTuningPlanConfig",
    "build_track_tuning_plan",
    "write_track_tuning_plan",
]
