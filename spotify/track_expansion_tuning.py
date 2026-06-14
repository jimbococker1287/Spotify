from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .run_artifacts import write_json


TRACK_TUNING_SCHEMA_VERSION = "track-expansion-optuna-v1"
SUPPORTED_TRACK_EXPANSION_MODELS = (
    "session_cooccurrence",
    "ease",
    "meantime",
    "mmoe",
    "ple",
    "dcn_v2",
)


class TrialLike(Protocol):
    number: int

    def suggest_categorical(self, name: str, choices: Sequence[object]) -> object: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int: ...

    def report(self, value: float, step: int) -> None: ...

    def should_prune(self) -> bool: ...

    def set_user_attr(self, key: str, value: object) -> None: ...


@dataclass(frozen=True)
class SearchParameterContract:
    name: str
    kind: str
    low: int | float | None = None
    high: int | float | None = None
    choices: tuple[object, ...] = ()
    step: int | float | None = None
    log: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Search parameter names cannot be empty.")
        if self.kind not in {"int", "float", "categorical"}:
            raise ValueError(f"Unsupported search parameter kind: {self.kind}")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"Categorical parameter {self.name!r} needs choices.")
            if self.low is not None or self.high is not None or self.step is not None or self.log:
                raise ValueError(f"Categorical parameter {self.name!r} cannot declare numeric bounds.")
            return
        if self.low is None or self.high is None:
            raise ValueError(f"Numeric parameter {self.name!r} needs low and high bounds.")
        if self.low > self.high:
            raise ValueError(f"Parameter {self.name!r} has low greater than high.")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Parameter {self.name!r} must have a positive step.")
        if self.log and self.step is not None:
            raise ValueError(f"Parameter {self.name!r} cannot combine log sampling and a step.")
        if self.log and self.low <= 0:
            raise ValueError(f"Log parameter {self.name!r} must have a positive lower bound.")

    def suggest(self, trial: TrialLike) -> object:
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        if self.kind == "int":
            return trial.suggest_int(
                self.name,
                int(self.low),
                int(self.high),
                step=int(self.step or 1),
                log=self.log,
            )
        return trial.suggest_float(
            self.name,
            float(self.low),
            float(self.high),
            step=float(self.step) if self.step is not None else None,
            log=self.log,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "choices": [_json_value(choice) for choice in self.choices],
            "step": self.step,
            "log": self.log,
            "description": self.description,
        }


@dataclass(frozen=True)
class TemporalMetricContract:
    name: str
    direction: str = "maximize"
    split: str = "temporal_validation"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Metric names cannot be empty.")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("Metric direction must be 'maximize' or 'minimize'.")
        if self.split != "temporal_validation":
            raise ValueError("Track expansion tuning must select on temporal_validation.")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "direction": self.direction,
            "split": self.split,
            "description": self.description,
        }


@dataclass(frozen=True)
class TrackExpansionSearchSpace:
    model_name: str
    stage: str
    metric: TemporalMetricContract
    parameters: tuple[SearchParameterContract, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_TRACK_EXPANSION_MODELS:
            raise ValueError(f"Unsupported track expansion model: {self.model_name}")
        if not self.parameters:
            raise ValueError(f"{self.model_name} must declare at least one search parameter.")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.model_name} contains duplicate search parameter names.")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "stage": self.stage,
            "metric": self.metric.to_dict(),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "notes": self.notes,
        }


def _integer(
    name: str,
    low: int,
    high: int,
    *,
    step: int | None = None,
    description: str = "",
) -> SearchParameterContract:
    return SearchParameterContract(
        name=name,
        kind="int",
        low=low,
        high=high,
        step=step,
        description=description,
    )


def _float(
    name: str,
    low: float,
    high: float,
    *,
    step: float | None = None,
    log: bool = False,
    description: str = "",
) -> SearchParameterContract:
    return SearchParameterContract(
        name=name,
        kind="float",
        low=low,
        high=high,
        step=step,
        log=log,
        description=description,
    )


def _categorical(
    name: str,
    *choices: object,
    description: str = "",
) -> SearchParameterContract:
    return SearchParameterContract(
        name=name,
        kind="categorical",
        choices=tuple(choices),
        description=description,
    )


TRACK_EXPANSION_SEARCH_SPACES: Mapping[str, TrackExpansionSearchSpace] = MappingProxyType(
    {
        "session_cooccurrence": TrackExpansionSearchSpace(
            model_name="session_cooccurrence",
            stage="candidate_generation",
            metric=TemporalMetricContract(
                "validation_recall_at_100",
                description="Repeat-aware Recall@100 on the untouched temporal validation split.",
            ),
            parameters=(
                _integer("max_items", 500, 3_000, step=250),
                _float("shrinkage", 0.1, 100.0, log=True),
                _categorical("history_window", 16, 32, 64, 128, 256),
            ),
            notes="Fit session baskets on training sessions only; never rebuild them from validation histories.",
        ),
        "ease": TrackExpansionSearchSpace(
            model_name="ease",
            stage="candidate_generation",
            metric=TemporalMetricContract(
                "validation_recall_at_100",
                description="Repeat-aware Recall@100 on the untouched temporal validation split.",
            ),
            parameters=(
                _integer("max_items", 200, 1_000, step=100),
                _float("l2", 1.0, 10_000.0, log=True),
                _integer("min_item_support", 1, 10),
                _categorical("binary_interactions", True, False),
            ),
            notes="The item budget is intentionally bounded because EASE materializes a dense item matrix.",
        ),
        "meantime": TrackExpansionSearchSpace(
            model_name="meantime",
            stage="sequence_encoder",
            metric=TemporalMetricContract(
                "validation_ndcg_at_10",
                description="NDCG@10 from temporal next-track validation examples.",
            ),
            parameters=(
                _categorical("embedding_dim", 32, 64, 96, 128),
                _categorical("num_heads", 1, 2, 4, 8),
                _categorical("feed_forward_dim", 64, 128, 256, 384, 512),
                _integer("num_blocks", 1, 4),
                _categorical("num_time_buckets", 16, 32, 64, 96, 128),
                _categorical("context_dim", 24, 32, 64, 96, 128),
                _float("dropout_rate", 0.0, 0.4),
                _float("learning_rate", 1e-5, 3e-3, log=True),
                _categorical("batch_size", 64, 128, 256),
            ),
            notes="All embedding choices are divisible by every declared head count.",
        ),
        "mmoe": TrackExpansionSearchSpace(
            model_name="mmoe",
            stage="multitask_prediction",
            metric=TemporalMetricContract(
                "weighted_temporal_validation_task_score",
                description="Predeclared blend of next-item ranking and labeled auxiliary-head metrics.",
            ),
            parameters=(
                _categorical("embedding_dim", 32, 64, 96, 128),
                _categorical("sequence_encoder", "average", "gru"),
                _categorical("sequence_dim", 32, 64, 96, 128),
                _categorical("context_dim", 16, 24, 32, 64),
                _categorical("fusion_dim", 64, 96, 128, 192, 256),
                _integer("num_experts", 2, 8),
                _categorical("expert_units", 32, 48, 64, 96, 128, 192),
                _categorical("tower_units", 16, 24, 32, 64, 96),
                _float("dropout_rate", 0.0, 0.4),
                _float("learning_rate", 1e-5, 3e-3, log=True),
                _categorical("batch_size", 64, 128, 256),
            ),
            notes="Keep task-score weights fixed for the first architecture study to avoid metric gaming.",
        ),
        "ple": TrackExpansionSearchSpace(
            model_name="ple",
            stage="multitask_prediction",
            metric=TemporalMetricContract(
                "weighted_temporal_validation_task_score",
                description="Predeclared blend of next-item ranking and labeled auxiliary-head metrics.",
            ),
            parameters=(
                _categorical("embedding_dim", 32, 64, 96, 128),
                _categorical("sequence_encoder", "average", "gru"),
                _categorical("sequence_dim", 32, 64, 96, 128),
                _categorical("context_dim", 16, 24, 32, 64),
                _categorical("fusion_dim", 64, 96, 128, 192, 256),
                _integer("num_experts", 1, 6),
                _integer("task_experts", 1, 4),
                _categorical("expert_units", 32, 48, 64, 96, 128, 192),
                _categorical("tower_units", 16, 24, 32, 64, 96),
                _float("dropout_rate", 0.0, 0.4),
                _float("learning_rate", 1e-5, 3e-3, log=True),
                _categorical("batch_size", 64, 128, 256),
            ),
            notes="Keep task-score weights fixed for the first architecture study to avoid metric gaming.",
        ),
        "dcn_v2": TrackExpansionSearchSpace(
            model_name="dcn_v2",
            stage="reranking",
            metric=TemporalMetricContract(
                "validation_ndcg_at_10",
                description="NDCG@10 over temporally held-out retrieved candidate lists.",
            ),
            parameters=(
                _integer("cross_layers", 1, 6),
                _categorical("cross_parameterization", "matrix", "vector"),
                _categorical("architecture", "parallel", "stacked"),
                _categorical("deep_units", "64x32", "128x64", "256x128x64"),
                _float("dropout_rate", 0.0, 0.5),
                _float("l2_regularization", 1e-8, 1e-2, log=True),
                _float("learning_rate", 1e-5, 3e-3, log=True),
                _categorical("batch_size", 128, 256, 512),
            ),
            notes="Candidate lists must be generated without validation-label access and held fixed within a study.",
        ),
    }
)

_MODEL_ALIASES = {
    "session-cooccurrence": "session_cooccurrence",
    "session_cooc": "session_cooccurrence",
    "cooccurrence": "session_cooccurrence",
    "meantime_tisasrec": "meantime",
    "dcn-v2": "dcn_v2",
    "dcn_v2_reranker": "dcn_v2",
}


def normalize_track_expansion_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower().replace(" ", "_")
    normalized = _MODEL_ALIASES.get(normalized, normalized)
    if normalized not in TRACK_EXPANSION_SEARCH_SPACES:
        known = ", ".join(SUPPORTED_TRACK_EXPANSION_MODELS)
        raise ValueError(f"Unsupported track expansion model {model_name!r}. Known models: {known}")
    return normalized


def get_track_expansion_search_space(model_name: str) -> TrackExpansionSearchSpace:
    return TRACK_EXPANSION_SEARCH_SPACES[normalize_track_expansion_model_name(model_name)]


def suggest_track_expansion_params(trial: TrialLike, model_name: str) -> dict[str, object]:
    model_name = normalize_track_expansion_model_name(model_name)
    contract = TRACK_EXPANSION_SEARCH_SPACES[model_name]
    params = {parameter.name: parameter.suggest(trial) for parameter in contract.parameters}
    if model_name == "dcn_v2":
        encoded_units = str(params["deep_units"])
        params["deep_units"] = tuple(int(value) for value in encoded_units.split("x"))
    return params


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    intermediate_values: tuple[tuple[int, float], ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackObjectiveContext:
    model_name: str
    params: Mapping[str, object]
    contract: TrackExpansionSearchSpace
    trial: TrialLike


TrackObjectiveCallback = Callable[[TrackObjectiveContext], float | ObjectiveResult]


@dataclass(frozen=True)
class TrackExpansionTuningConfig:
    storage_path: Path
    output_dir: Path | None = None
    selected_models: tuple[str, ...] = ()
    trial_budgets: int | Mapping[str, int] = 20
    sampler_seed: int = 42
    pruner: str = "median"
    median_startup_trials: int = 3
    median_warmup_steps: int = 1
    timeout_seconds: float | None = None
    n_jobs: int = 1
    study_prefix: str = "track_expansion"


@dataclass(frozen=True)
class ValidatedTuningRequest:
    models: tuple[str, ...]
    trial_budgets: Mapping[str, int]
    storage_path: Path
    output_dir: Path


@dataclass(frozen=True)
class TrialSummary:
    number: int
    state: str
    value: float | None
    params: Mapping[str, object]
    intermediate_values: Mapping[str, float]
    duration_seconds: float | None
    datetime_start: str | None
    datetime_complete: str | None
    user_attrs: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "state": self.state,
            "value": self.value,
            "params": _json_value(self.params),
            "intermediate_values": _json_value(self.intermediate_values),
            "duration_seconds": self.duration_seconds,
            "datetime_start": self.datetime_start,
            "datetime_complete": self.datetime_complete,
            "user_attrs": _json_value(self.user_attrs),
        }


@dataclass(frozen=True)
class BestTrialSummary:
    number: int
    value: float
    params: Mapping[str, object]
    metric_name: str
    direction: str

    def to_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "value": self.value,
            "params": _json_value(self.params),
            "metric_name": self.metric_name,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class StudySummary:
    model_name: str
    study_name: str
    status: str
    metric: TemporalMetricContract
    trial_budget: int
    existing_trials: int
    executed_trials: int
    total_trials: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    best_trial: BestTrialSummary | None
    trials: tuple[TrialSummary, ...]
    storage_path: str
    trial_table_path: str
    best_trial_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "study_name": self.study_name,
            "status": self.status,
            "metric": self.metric.to_dict(),
            "trial_budget": self.trial_budget,
            "existing_trials": self.existing_trials,
            "executed_trials": self.executed_trials,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "pruned_trials": self.pruned_trials,
            "failed_trials": self.failed_trials,
            "best_trial": self.best_trial.to_dict() if self.best_trial is not None else None,
            "trials": [trial.to_dict() for trial in self.trials],
            "storage_path": self.storage_path,
            "trial_table_path": self.trial_table_path,
            "best_trial_path": self.best_trial_path,
        }


@dataclass(frozen=True)
class TrackExpansionTuningSummary:
    status: str
    generated_at: str
    schema_version: str
    sampler_seed: int
    pruner: str
    storage_path: str
    studies: tuple[StudySummary, ...]
    summary_path: str
    dependency_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
            "sampler_seed": self.sampler_seed,
            "pruner": self.pruner,
            "storage_path": self.storage_path,
            "studies": [study.to_dict() for study in self.studies],
            "summary_path": self.summary_path,
            "dependency_error": self.dependency_error,
        }


def validate_track_expansion_tuning(
    *,
    objectives: Mapping[str, TrackObjectiveCallback],
    config: TrackExpansionTuningConfig,
) -> ValidatedTuningRequest:
    if not isinstance(config.storage_path, Path):
        raise TypeError("storage_path must be a pathlib.Path.")
    storage_path = config.storage_path.expanduser().resolve()
    if storage_path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        raise ValueError("storage_path must end in .db, .sqlite, or .sqlite3.")
    if config.n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")
    if config.timeout_seconds is not None and config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when provided.")
    if config.median_startup_trials < 0 or config.median_warmup_steps < 0:
        raise ValueError("Median-pruner startup and warmup values cannot be negative.")
    if config.pruner not in {"none", "median", "successive_halving", "hyperband"}:
        raise ValueError("pruner must be one of: none, median, successive_halving, hyperband.")
    if not config.study_prefix.strip():
        raise ValueError("study_prefix cannot be empty.")

    normalized_objectives: dict[str, TrackObjectiveCallback] = {}
    for raw_name, callback in objectives.items():
        model_name = normalize_track_expansion_model_name(raw_name)
        if model_name in normalized_objectives:
            raise ValueError(f"Duplicate objective callback for {model_name}.")
        if not callable(callback):
            raise TypeError(f"Objective callback for {model_name} must be callable.")
        normalized_objectives[model_name] = callback

    if config.selected_models:
        models = tuple(normalize_track_expansion_model_name(model_name) for model_name in config.selected_models)
    else:
        models = tuple(normalized_objectives)
    if not models:
        raise ValueError("At least one track expansion model must be selected.")
    if len(models) != len(set(models)):
        raise ValueError("selected_models cannot contain duplicates.")

    if isinstance(config.trial_budgets, bool):
        raise TypeError("trial_budgets must be an integer or model-to-budget mapping.")
    if isinstance(config.trial_budgets, int):
        budgets = {model_name: config.trial_budgets for model_name in models}
    elif isinstance(config.trial_budgets, Mapping):
        normalized_budgets = {
            normalize_track_expansion_model_name(name): value for name, value in config.trial_budgets.items()
        }
        budgets = {}
        for model_name in models:
            if model_name not in normalized_budgets:
                raise ValueError(f"Missing trial budget for {model_name}.")
            budgets[model_name] = normalized_budgets[model_name]
    else:
        raise TypeError("trial_budgets must be an integer or model-to-budget mapping.")

    for model_name, budget in budgets.items():
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError(f"Trial budget for {model_name} must be a non-negative integer.")
        if budget > 0 and model_name not in normalized_objectives:
            raise ValueError(f"Missing objective callback for {model_name}.")

    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else storage_path.parent / f"{storage_path.stem}_summaries"
    )
    return ValidatedTuningRequest(
        models=models,
        trial_budgets=MappingProxyType(dict(budgets)),
        storage_path=storage_path,
        output_dir=output_dir,
    )


def _load_optuna():
    try:
        import optuna
    except ImportError:
        return None
    return optuna


def optuna_is_available() -> bool:
    return _load_optuna() is not None


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _contract_fingerprint(contract: TrackExpansionSearchSpace) -> str:
    payload = {
        "schema_version": TRACK_TUNING_SCHEMA_VERSION,
        "contract": contract.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_sampler_seed(base_seed: int, model_name: str) -> int:
    digest = hashlib.sha256(model_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (int(base_seed) + offset) % (2**32)


def _build_pruner(optuna: Any, config: TrackExpansionTuningConfig):
    if config.pruner == "none":
        return optuna.pruners.NopPruner()
    if config.pruner == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2)
    if config.pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3)
    return optuna.pruners.MedianPruner(
        n_startup_trials=config.median_startup_trials,
        n_warmup_steps=config.median_warmup_steps,
    )


def _normalize_objectives(
    objectives: Mapping[str, TrackObjectiveCallback],
) -> dict[str, TrackObjectiveCallback]:
    return {normalize_track_expansion_model_name(name): callback for name, callback in objectives.items()}


def _normalize_objective_result(value: float | ObjectiveResult) -> ObjectiveResult:
    result = value if isinstance(value, ObjectiveResult) else ObjectiveResult(value=float(value))
    metric_value = float(result.value)
    if not math.isfinite(metric_value):
        raise ValueError("Objective callbacks must return a finite metric value.")
    seen_steps: set[int] = set()
    normalized_intermediate: list[tuple[int, float]] = []
    for raw_step, raw_metric in result.intermediate_values:
        step = int(raw_step)
        metric = float(raw_metric)
        if step < 0:
            raise ValueError("Intermediate objective steps cannot be negative.")
        if step in seen_steps:
            raise ValueError(f"Duplicate intermediate objective step: {step}")
        if not math.isfinite(metric):
            raise ValueError("Intermediate objective values must be finite.")
        seen_steps.add(step)
        normalized_intermediate.append((step, metric))
    return ObjectiveResult(
        value=metric_value,
        intermediate_values=tuple(sorted(normalized_intermediate)),
        metadata=dict(result.metadata),
    )


def _trial_summary(trial: Any) -> TrialSummary:
    state = str(trial.state.name).lower()
    value = float(trial.value) if trial.value is not None and math.isfinite(float(trial.value)) else None
    duration = trial.duration.total_seconds() if trial.duration is not None else None
    return TrialSummary(
        number=int(trial.number),
        state=state,
        value=value,
        params=dict(trial.params),
        intermediate_values={
            str(step): float(value)
            for step, value in sorted(trial.intermediate_values.items())
            if math.isfinite(float(value))
        },
        duration_seconds=float(duration) if duration is not None else None,
        datetime_start=_isoformat(trial.datetime_start),
        datetime_complete=_isoformat(trial.datetime_complete),
        user_attrs=dict(trial.user_attrs),
    )


def _best_trial(
    trials: tuple[TrialSummary, ...],
    *,
    metric: TemporalMetricContract,
) -> BestTrialSummary | None:
    completed = [trial for trial in trials if trial.state == "complete" and trial.value is not None]
    if not completed:
        return None
    selector = max if metric.direction == "maximize" else min
    best = selector(completed, key=lambda trial: float(trial.value))
    return BestTrialSummary(
        number=best.number,
        value=float(best.value),
        params=dict(best.params),
        metric_name=metric.name,
        direction=metric.direction,
    )


def _summarize_study(
    study: Any,
    *,
    model_name: str,
    metric: TemporalMetricContract,
    budget: int,
    existing_trials: int,
    storage_path: Path,
    output_dir: Path,
    forced_status: str | None = None,
) -> StudySummary:
    trials = tuple(_trial_summary(trial) for trial in study.trials)
    state_counts = {state: sum(trial.state == state for trial in trials) for state in ("complete", "pruned", "fail")}
    total_trials = len(trials)
    status = forced_status or ("budget_complete" if total_trials >= budget else "partial")
    return StudySummary(
        model_name=model_name,
        study_name=str(study.study_name),
        status=status,
        metric=metric,
        trial_budget=budget,
        existing_trials=existing_trials,
        executed_trials=max(0, total_trials - existing_trials),
        total_trials=total_trials,
        completed_trials=state_counts["complete"],
        pruned_trials=state_counts["pruned"],
        failed_trials=state_counts["fail"],
        best_trial=_best_trial(trials, metric=metric),
        trials=trials,
        storage_path=str(storage_path),
        trial_table_path=str(output_dir / f"{model_name}_trials.json"),
        best_trial_path=str(output_dir / f"{model_name}_best_trial.json"),
    )


def _write_study_artifacts(summary: StudySummary) -> None:
    write_json(
        Path(summary.trial_table_path),
        {
            "model_name": summary.model_name,
            "study_name": summary.study_name,
            "metric": summary.metric.to_dict(),
            "trial_budget": summary.trial_budget,
            "trials": [trial.to_dict() for trial in summary.trials],
        },
        sort_keys=True,
    )
    write_json(
        Path(summary.best_trial_path),
        {
            "model_name": summary.model_name,
            "study_name": summary.study_name,
            "metric": summary.metric.to_dict(),
            "best_trial": (summary.best_trial.to_dict() if summary.best_trial is not None else None),
        },
        sort_keys=True,
    )


def _unavailable_study_summary(
    *,
    model_name: str,
    budget: int,
    storage_path: Path,
    output_dir: Path,
    study_prefix: str,
) -> StudySummary:
    return StudySummary(
        model_name=model_name,
        study_name=f"{study_prefix}_{model_name}",
        status="dependency_unavailable",
        metric=TRACK_EXPANSION_SEARCH_SPACES[model_name].metric,
        trial_budget=budget,
        existing_trials=0,
        executed_trials=0,
        total_trials=0,
        completed_trials=0,
        pruned_trials=0,
        failed_trials=0,
        best_trial=None,
        trials=(),
        storage_path=str(storage_path),
        trial_table_path=str(output_dir / f"{model_name}_trials.json"),
        best_trial_path=str(output_dir / f"{model_name}_best_trial.json"),
    )


def run_track_expansion_tuning(
    *,
    objectives: Mapping[str, TrackObjectiveCallback],
    config: TrackExpansionTuningConfig,
) -> TrackExpansionTuningSummary:
    request = validate_track_expansion_tuning(objectives=objectives, config=config)
    callbacks = _normalize_objectives(objectives)
    request.storage_path.parent.mkdir(parents=True, exist_ok=True)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = request.output_dir / "track_expansion_optuna_summary.json"
    optuna = _load_optuna()

    if optuna is None:
        studies = tuple(
            _unavailable_study_summary(
                model_name=model_name,
                budget=request.trial_budgets[model_name],
                storage_path=request.storage_path,
                output_dir=request.output_dir,
                study_prefix=config.study_prefix,
            )
            for model_name in request.models
        )
        for study_summary in studies:
            _write_study_artifacts(study_summary)
        summary = TrackExpansionTuningSummary(
            status="dependency_unavailable",
            generated_at=datetime.now(timezone.utc).isoformat(),
            schema_version=TRACK_TUNING_SCHEMA_VERSION,
            sampler_seed=config.sampler_seed,
            pruner=config.pruner,
            storage_path=str(request.storage_path),
            studies=studies,
            summary_path=str(summary_path),
            dependency_error="Optuna is not installed; validated the request without running trials.",
        )
        write_json(summary_path, summary.to_dict(), sort_keys=True)
        return summary

    storage_url = f"sqlite:///{request.storage_path.as_posix()}"
    study_summaries: list[StudySummary] = []
    for model_name in request.models:
        contract = TRACK_EXPANSION_SEARCH_SPACES[model_name]
        budget = request.trial_budgets[model_name]
        study = optuna.create_study(
            study_name=f"{config.study_prefix}_{model_name}",
            storage=storage_url,
            load_if_exists=True,
            direction=contract.metric.direction,
            sampler=optuna.samplers.TPESampler(seed=_model_sampler_seed(config.sampler_seed, model_name)),
            pruner=_build_pruner(optuna, config),
        )
        stored_direction = str(study.direction.name).lower()
        if stored_direction != contract.metric.direction:
            raise RuntimeError(
                f"Study {study.study_name!r} has direction {stored_direction!r}, "
                f"but {contract.metric.name!r} requires {contract.metric.direction!r}."
            )
        fingerprint = _contract_fingerprint(contract)
        stored_fingerprint = study.user_attrs.get("contract_fingerprint")
        if stored_fingerprint is not None and stored_fingerprint != fingerprint:
            raise RuntimeError(
                f"Study {study.study_name!r} uses a different search-space contract. "
                "Choose a new study_prefix to preserve the existing study."
            )
        study.set_user_attr("schema_version", TRACK_TUNING_SCHEMA_VERSION)
        study.set_user_attr("model_name", model_name)
        study.set_user_attr("contract_fingerprint", fingerprint)
        study.set_user_attr("metric", contract.metric.to_dict())
        study.set_user_attr("search_space", contract.to_dict())

        existing_trials = len(study.trials)
        remaining_trials = max(0, budget - existing_trials)

        def persist(current_study: Any, _trial: Any) -> None:
            current = _summarize_study(
                current_study,
                model_name=model_name,
                metric=contract.metric,
                budget=budget,
                existing_trials=existing_trials,
                storage_path=request.storage_path,
                output_dir=request.output_dir,
            )
            _write_study_artifacts(current)

        def objective(trial: TrialLike) -> float:
            params = suggest_track_expansion_params(trial, model_name)
            trial.set_user_attr("model_name", model_name)
            trial.set_user_attr("metric_name", contract.metric.name)
            trial.set_user_attr("evaluation_split", contract.metric.split)
            result = _normalize_objective_result(
                callbacks[model_name](
                    TrackObjectiveContext(
                        model_name=model_name,
                        params=MappingProxyType(dict(params)),
                        contract=contract,
                        trial=trial,
                    )
                )
            )
            for key, value in result.metadata.items():
                trial.set_user_attr(str(key), _json_value(value))
            for step, metric_value in result.intermediate_values:
                trial.report(metric_value, step)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"{contract.metric.name} pruned at step {step}: {metric_value:.6f}")
            return result.value

        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=config.timeout_seconds,
                n_jobs=config.n_jobs,
                callbacks=[persist],
                gc_after_trial=True,
                show_progress_bar=False,
            )

        study_summary = _summarize_study(
            study,
            model_name=model_name,
            metric=contract.metric,
            budget=budget,
            existing_trials=existing_trials,
            storage_path=request.storage_path,
            output_dir=request.output_dir,
        )
        _write_study_artifacts(study_summary)
        study_summaries.append(study_summary)

        partial_summary = TrackExpansionTuningSummary(
            status="complete" if len(study_summaries) == len(request.models) else "running",
            generated_at=datetime.now(timezone.utc).isoformat(),
            schema_version=TRACK_TUNING_SCHEMA_VERSION,
            sampler_seed=config.sampler_seed,
            pruner=config.pruner,
            storage_path=str(request.storage_path),
            studies=tuple(study_summaries),
            summary_path=str(summary_path),
        )
        write_json(summary_path, partial_summary.to_dict(), sort_keys=True)

    status = "complete" if all(summary.status == "budget_complete" for summary in study_summaries) else "partial"
    summary = TrackExpansionTuningSummary(
        status=status,
        generated_at=datetime.now(timezone.utc).isoformat(),
        schema_version=TRACK_TUNING_SCHEMA_VERSION,
        sampler_seed=config.sampler_seed,
        pruner=config.pruner,
        storage_path=str(request.storage_path),
        studies=tuple(study_summaries),
        summary_path=str(summary_path),
    )
    write_json(summary_path, summary.to_dict(), sort_keys=True)
    return summary


def run_track_expansion_study(
    *,
    model_name: str,
    objective: TrackObjectiveCallback,
    config: TrackExpansionTuningConfig,
) -> StudySummary:
    normalized = normalize_track_expansion_model_name(model_name)
    selected_config = TrackExpansionTuningConfig(
        storage_path=config.storage_path,
        output_dir=config.output_dir,
        selected_models=(normalized,),
        trial_budgets=config.trial_budgets,
        sampler_seed=config.sampler_seed,
        pruner=config.pruner,
        median_startup_trials=config.median_startup_trials,
        median_warmup_steps=config.median_warmup_steps,
        timeout_seconds=config.timeout_seconds,
        n_jobs=config.n_jobs,
        study_prefix=config.study_prefix,
    )
    summary = run_track_expansion_tuning(
        objectives={normalized: objective},
        config=selected_config,
    )
    return summary.studies[0]


__all__ = [
    "BestTrialSummary",
    "ObjectiveResult",
    "SUPPORTED_TRACK_EXPANSION_MODELS",
    "SearchParameterContract",
    "StudySummary",
    "TRACK_EXPANSION_SEARCH_SPACES",
    "TRACK_TUNING_SCHEMA_VERSION",
    "TemporalMetricContract",
    "TrackExpansionSearchSpace",
    "TrackExpansionTuningConfig",
    "TrackExpansionTuningSummary",
    "TrackObjectiveCallback",
    "TrackObjectiveContext",
    "TrialSummary",
    "ValidatedTuningRequest",
    "get_track_expansion_search_space",
    "normalize_track_expansion_model_name",
    "optuna_is_available",
    "run_track_expansion_study",
    "run_track_expansion_tuning",
    "suggest_track_expansion_params",
    "validate_track_expansion_tuning",
]
