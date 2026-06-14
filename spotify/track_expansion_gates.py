from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from .model_explainability import ExplainabilityCapability, resolve_explainer_capability


JsonMapping = Mapping[str, object]
ArtifactExists = Callable[[Path], bool]
GATE_STATUSES = ("pass", "warn", "fail")

_MODEL_NAME_KEYS = ("model_name", "name", "model", "best_model_name")
_METRIC_PRIORITY = (
    "ndcg_at_10",
    "recall_at_100",
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_10",
    "mrr_at_k",
    "hit_rate_at_10",
)
_UNAVAILABLE_STATUSES = {"unavailable", "not_available", "not_supported", "unsupported", "not_applicable"}
_PASS_STATUSES = {"pass", "passed", "complete", "completed", "available", "present", "ready", "validated"}
_FAIL_STATUSES = {"fail", "failed", "missing", "invalid", "blocked", "error"}


@dataclass(frozen=True)
class PromotionPolicy:
    """Thresholds and required evidence for track-model promotion."""

    primary_metric: str | None = None
    validation_metric_floor: float = 0.05
    test_metric_floor: float = 0.05
    max_popularity_regression: float = 0.01
    min_train_rows: int = 1
    min_validation_rows: int = 1
    min_test_rows: int = 1
    min_target_coverage: float = 0.25
    max_calibration_ece: float = 0.10
    max_drift_value: float = 0.20
    required_reproducibility_fields: tuple[str, ...] = (
        "config",
        "random_seed",
        "dataset_fingerprint",
        "temporal_split",
        "code_version",
    )

    def __post_init__(self) -> None:
        if self.validation_metric_floor < 0.0 or self.test_metric_floor < 0.0:
            raise ValueError("Metric floors must be non-negative.")
        if self.max_popularity_regression < 0.0:
            raise ValueError("max_popularity_regression must be non-negative.")
        if min(self.min_train_rows, self.min_validation_rows, self.min_test_rows) < 1:
            raise ValueError("Split row minimums must be positive.")
        if not 0.0 <= self.min_target_coverage <= 1.0:
            raise ValueError("min_target_coverage must be between zero and one.")
        if self.max_calibration_ece < 0.0 or self.max_drift_value < 0.0:
            raise ValueError("Calibration and drift thresholds must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GateCheck:
    """One auditable promotion requirement."""

    name: str
    status: str
    required: bool
    summary: str
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in GATE_STATUSES:
            raise ValueError(f"status must be one of {GATE_STATUSES}")

    @property
    def blocking(self) -> bool:
        return self.required and self.status != "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "blocking": self.blocking,
            "summary": self.summary,
            "evidence": _json_ready(self.evidence),
        }


@dataclass(frozen=True)
class ModelPromotionDecision:
    """Deterministic model-level readiness decision."""

    model_name: str
    model_family: str
    sources: tuple[str, ...]
    status: str
    promoted: bool
    primary_metric: str | None
    checks: tuple[GateCheck, ...]

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.blocking)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "sources": list(self.sources),
            "status": self.status,
            "promoted": self.promoted,
            "primary_metric": self.primary_metric,
            "blockers": list(self.blockers),
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_markdown(self) -> str:
        lines = [
            f"## {_markdown_text(self.model_name)}",
            "",
            f"- Family: `{_markdown_text(self.model_family)}`",
            f"- Sources: `{', '.join(_markdown_text(value) for value in self.sources)}`",
            f"- Status: `{self.status}`",
            f"- Promoted: `{str(self.promoted).lower()}`",
            f"- Primary metric: `{_markdown_text(self.primary_metric or 'unresolved')}`",
            "",
            "| Gate | Status | Required | Summary |",
            "|---|---|---:|---|",
        ]
        for check in self.checks:
            lines.append(
                f"| `{_markdown_text(check.name)}` | `{check.status}` | "
                f"`{str(check.required).lower()}` | {_markdown_text(check.summary)} |"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class PromotionGateReport:
    """JSON- and Markdown-friendly result for all discovered candidates."""

    status: str
    decisions: tuple[ModelPromotionDecision, ...]
    policy: PromotionPolicy

    @property
    def promoted_models(self) -> tuple[str, ...]:
        return tuple(decision.model_name for decision in self.decisions if decision.promoted)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "candidate_count": len(self.decisions),
            "promoted_count": len(self.promoted_models),
            "promoted_models": list(self.promoted_models),
            "policy": self.policy.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Track Expansion Promotion Gates",
            "",
            f"- Overall status: `{self.status}`",
            f"- Candidates: `{len(self.decisions)}`",
            f"- Promoted: `{len(self.promoted_models)}`",
        ]
        if self.promoted_models:
            lines.append(f"- Promoted models: `{', '.join(self.promoted_models)}`")
        for decision in self.decisions:
            lines.extend(("", decision.to_markdown()))
        return "\n".join(lines) + "\n"


@dataclass
class _CollectedCandidate:
    model_name: str
    payload: dict[str, object]
    contexts: list[JsonMapping]
    sources: list[str]


def evaluate_track_expansion_gates(
    *,
    baseline_manifest: JsonMapping,
    training_manifest: JsonMapping | None = None,
    tuning_manifest: JsonMapping | None = None,
    dcn_manifest: JsonMapping | None = None,
    public_pretraining_manifest: JsonMapping | None = None,
    policy: PromotionPolicy | None = None,
    artifact_base_dir: Path | None = None,
    artifact_exists: ArtifactExists | None = None,
) -> PromotionGateReport:
    """Evaluate every model found across track-expansion manifests.

    A required warning or failure blocks promotion. An explicit unavailable
    status is a warning; absent or malformed required evidence is a failure.
    """

    effective_policy = policy or PromotionPolicy()
    exists = artifact_exists or Path.exists
    candidates = _collect_candidates(
        training_manifest=training_manifest,
        tuning_manifest=tuning_manifest,
        dcn_manifest=dcn_manifest,
        public_pretraining_manifest=public_pretraining_manifest,
    )
    decisions = tuple(
        _evaluate_candidate(
            candidate,
            baseline_manifest=baseline_manifest,
            policy=effective_policy,
            artifact_base_dir=artifact_base_dir,
            artifact_exists=exists,
        )
        for candidate in sorted(candidates.values(), key=lambda value: value.model_name)
    )
    if not decisions:
        status = "fail"
    elif all(decision.status == "pass" for decision in decisions):
        status = "pass"
    elif any(decision.status == "fail" for decision in decisions):
        status = "fail" if not any(decision.promoted for decision in decisions) else "warn"
    else:
        status = "warn"
    return PromotionGateReport(status=status, decisions=decisions, policy=effective_policy)


def _collect_candidates(
    *,
    training_manifest: JsonMapping | None,
    tuning_manifest: JsonMapping | None,
    dcn_manifest: JsonMapping | None,
    public_pretraining_manifest: JsonMapping | None,
) -> dict[str, _CollectedCandidate]:
    candidates: dict[str, _CollectedCandidate] = {}
    manifests = (
        ("training", training_manifest, ("retrieval_results", "neural_results", "models", "results")),
        ("tuning", tuning_manifest, ("tuning_results", "best_models", "models", "results")),
        ("dcn", dcn_manifest, ("dcn_results", "models", "results")),
        (
            "public_pretraining",
            public_pretraining_manifest,
            ("transferred_models", "model_results", "models", "results"),
        ),
    )
    for source, manifest, result_keys in manifests:
        if not isinstance(manifest, Mapping):
            continue
        rows = _candidate_rows(manifest, result_keys)
        for row in rows:
            name = _model_name(row)
            if not name:
                continue
            existing = candidates.get(name)
            if existing is None:
                candidates[name] = _CollectedCandidate(
                    model_name=name,
                    payload=dict(row),
                    contexts=[manifest],
                    sources=[source],
                )
                continue
            existing.payload = _deep_merge(existing.payload, row)
            existing.contexts.append(manifest)
            if source not in existing.sources:
                existing.sources.append(source)
    return candidates


def _candidate_rows(manifest: JsonMapping, result_keys: Sequence[str]) -> tuple[JsonMapping, ...]:
    rows: list[JsonMapping] = []
    if _model_name(manifest):
        rows.append(manifest)
    for key in result_keys:
        value = manifest.get(key)
        if isinstance(value, Mapping):
            if _model_name(value):
                rows.append(value)
            else:
                for model_name, model_payload in sorted(value.items(), key=lambda item: str(item[0])):
                    if isinstance(model_payload, Mapping):
                        rows.append({"model_name": str(model_name), **model_payload})
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rows.extend(item for item in value if isinstance(item, Mapping))
    return tuple(rows)


def _evaluate_candidate(
    candidate: _CollectedCandidate,
    *,
    baseline_manifest: JsonMapping,
    policy: PromotionPolicy,
    artifact_base_dir: Path | None,
    artifact_exists: ArtifactExists,
) -> ModelPromotionDecision:
    payload = candidate.payload
    family = _model_family(payload, candidate.model_name, candidate.sources)
    primary_metric = _resolve_primary_metric(payload, policy)
    checks = (
        _artifact_check(payload, artifact_base_dir, artifact_exists),
        _temporal_metric_check(payload, primary_metric, policy),
        _popularity_regression_check(payload, baseline_manifest, primary_metric, policy),
        _split_coverage_check(payload, candidate.contexts, policy),
        _calibration_check(payload, candidate.contexts, policy),
        _explainability_check(payload, candidate.contexts, family, artifact_base_dir, artifact_exists),
        _drift_check(payload, candidate.contexts, policy),
        _reproducibility_check(payload, candidate.contexts, candidate.sources, policy),
        _public_provenance_check(payload, candidate.contexts, candidate.sources),
    )
    required_checks = tuple(check for check in checks if check.required)
    promoted = bool(required_checks) and all(check.status == "pass" for check in required_checks)
    if any(check.status == "fail" for check in required_checks):
        status = "fail"
    elif any(check.status == "warn" for check in checks):
        status = "warn"
    else:
        status = "pass"
    return ModelPromotionDecision(
        model_name=candidate.model_name,
        model_family=family,
        sources=tuple(candidate.sources),
        status=status,
        promoted=promoted,
        primary_metric=primary_metric,
        checks=checks,
    )


def _artifact_check(
    payload: JsonMapping,
    base_dir: Path | None,
    artifact_exists: ArtifactExists,
) -> GateCheck:
    explicit = _first_bool(payload, ("checkpoint_present", "artifact_present", "model_artifact_present"))
    paths = _artifact_paths(payload)
    resolved = tuple(_resolve_artifact_path(path, base_dir) for path in paths)
    present = explicit is True or any(artifact_exists(path) for path in resolved)
    if present:
        return _check(
            "model_artifact",
            "pass",
            "A model artifact or checkpoint is present.",
            paths=[str(path) for path in resolved],
            explicit_present=explicit,
        )
    return _check(
        "model_artifact",
        "fail",
        "No verifiable model artifact or checkpoint is present.",
        paths=[str(path) for path in resolved],
        explicit_present=explicit,
    )


def _temporal_metric_check(
    payload: JsonMapping,
    metric_name: str | None,
    policy: PromotionPolicy,
) -> GateCheck:
    validation = _mapping(payload.get("validation"))
    test = _mapping(payload.get("test"))
    validation_value = _finite_float(validation.get(metric_name)) if metric_name else None
    test_value = _finite_float(test.get(metric_name)) if metric_name else None
    evidence = {
        "metric": metric_name,
        "validation": validation_value,
        "test": test_value,
        "validation_floor": policy.validation_metric_floor,
        "test_floor": policy.test_metric_floor,
    }
    if metric_name is None or validation_value is None or test_value is None:
        return _check(
            "temporal_metrics",
            "fail",
            "Temporal validation and test metrics are both required.",
            **evidence,
        )
    if validation_value < policy.validation_metric_floor or test_value < policy.test_metric_floor:
        return _check(
            "temporal_metrics",
            "fail",
            "A temporal validation or test metric is below its floor.",
            **evidence,
        )
    return _check("temporal_metrics", "pass", "Temporal validation and test metric floors pass.", **evidence)


def _popularity_regression_check(
    payload: JsonMapping,
    baseline_manifest: JsonMapping,
    metric_name: str | None,
    policy: PromotionPolicy,
) -> GateCheck:
    local_baseline = _first_mapping(payload, ("popularity_baseline", "baseline"))
    baseline = local_baseline or baseline_manifest
    validation = _mapping(payload.get("validation"))
    test = _mapping(payload.get("test"))
    baseline_value = _finite_float(baseline.get(metric_name)) if metric_name else None
    if baseline_value is None and metric_name:
        baseline_value = _finite_float(_mapping(baseline.get("test")).get(metric_name))
    validation_value = _finite_float(validation.get(metric_name)) if metric_name else None
    test_value = _finite_float(test.get(metric_name)) if metric_name else None
    evidence = {
        "metric": metric_name,
        "popularity": baseline_value,
        "validation": validation_value,
        "test": test_value,
        "max_regression": policy.max_popularity_regression,
    }
    if baseline_value is None or validation_value is None or test_value is None:
        return _check(
            "popularity_regression",
            "fail",
            "Comparable popularity, validation, and test metrics are required.",
            **evidence,
        )
    validation_delta = validation_value - baseline_value
    test_delta = test_value - baseline_value
    evidence.update({"validation_delta": validation_delta, "test_delta": test_delta})
    if min(validation_delta, test_delta) < -policy.max_popularity_regression:
        return _check(
            "popularity_regression",
            "fail",
            "The candidate regresses beyond tolerance versus popularity.",
            **evidence,
        )
    return _check("popularity_regression", "pass", "Popularity regression tolerance passes.", **evidence)


def _split_coverage_check(
    payload: JsonMapping,
    contexts: Sequence[JsonMapping],
    policy: PromotionPolicy,
) -> GateCheck:
    coverage = _first_mapping(payload, ("coverage", "data_coverage", "split_coverage"))
    row_sources = [coverage]
    for context in contexts:
        row_sources.extend(
            (
                _mapping(context.get("tensor_summary")),
                _mapping(context.get("dataset")),
                _first_mapping(context, ("coverage", "data_coverage", "split_coverage")),
            )
        )
    train_rows = _first_number(row_sources, ("train_rows", "train_examples", "training_rows"))
    validation_rows = _first_number(
        row_sources,
        ("validation_rows", "validation_examples", "val_rows", "val_examples"),
    )
    test_rows = _first_number(row_sources, ("test_rows", "test_examples"))
    validation = _mapping(payload.get("validation"))
    test = _mapping(payload.get("test"))
    validation_coverage = _first_float(
        validation,
        ("target_vocabulary_coverage", "target_catalog_coverage", "coverage"),
    )
    test_coverage = _first_float(test, ("target_vocabulary_coverage", "target_catalog_coverage", "coverage"))
    evidence = {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "validation_target_coverage": validation_coverage,
        "test_target_coverage": test_coverage,
        "minimum_target_coverage": policy.min_target_coverage,
    }
    if train_rows is None or validation_rows is None or test_rows is None:
        return _check("split_coverage", "fail", "Train, validation, and test row coverage is required.", **evidence)
    if (
        train_rows < policy.min_train_rows
        or validation_rows < policy.min_validation_rows
        or test_rows < policy.min_test_rows
    ):
        return _check("split_coverage", "fail", "A required split has insufficient rows.", **evidence)
    available_coverages = tuple(value for value in (validation_coverage, test_coverage) if value is not None)
    if available_coverages and min(available_coverages) < policy.min_target_coverage:
        return _check(
            "split_coverage",
            "fail",
            "Validation or test target coverage is below its floor.",
            **evidence,
        )
    return _check("split_coverage", "pass", "Train, validation, and test coverage passes.", **evidence)


def _calibration_check(
    payload: JsonMapping,
    contexts: Sequence[JsonMapping],
    policy: PromotionPolicy,
) -> GateCheck:
    evidence = _evidence_mapping(payload, contexts, ("calibration", "calibration_metrics"))
    status = _normalized_status(evidence.get("status"))
    if status in _UNAVAILABLE_STATUSES:
        return _check(
            "calibration",
            "warn",
            "Calibration is explicitly unavailable and blocks promotion.",
            declared_status=status,
            reason=evidence.get("reason"),
        )
    ece = _first_float(evidence, ("ece", "test_ece", "expected_calibration_error"))
    within_threshold = _first_bool(evidence, ("within_threshold", "passes_threshold"))
    if not evidence:
        return _check("calibration", "fail", "Calibration evidence is missing.")
    if status in _FAIL_STATUSES:
        return _check("calibration", "fail", "Calibration evidence reports failure.", declared_status=status, ece=ece)
    if ece is None and within_threshold is not True:
        return _check(
            "calibration",
            "fail",
            "Calibration evidence lacks ECE or an explicit threshold result.",
            declared_status=status,
            ece=ece,
        )
    if ece is not None and ece > policy.max_calibration_ece:
        return _check(
            "calibration",
            "fail",
            "Expected calibration error exceeds the allowed maximum.",
            declared_status=status,
            ece=ece,
            maximum_ece=policy.max_calibration_ece,
        )
    return _check(
        "calibration",
        "pass",
        "Calibration evidence passes.",
        declared_status=status,
        ece=ece,
        maximum_ece=policy.max_calibration_ece,
    )


def _explainability_check(
    payload: JsonMapping,
    contexts: Sequence[JsonMapping],
    family: str,
    base_dir: Path | None,
    artifact_exists: ArtifactExists,
) -> GateCheck:
    try:
        capability = resolve_explainer_capability(family)
    except KeyError:
        return _check(
            "explainability",
            "fail",
            "The model family has no registered explainability capability.",
            family=family,
            shap_required=False,
            shap_artifact_status="unknown",
        )
    evidence = _evidence_mapping(payload, contexts, ("explainability", "explanations", "shap"))
    status = _normalized_status(evidence.get("status"))
    shap_required = capability.strategy == "shap_compatible"
    if status in _UNAVAILABLE_STATUSES:
        return _explainability_result(
            "warn",
            "Explainability is explicitly unavailable and blocks promotion.",
            capability,
            shap_required,
            "unavailable" if shap_required else "not_applicable",
            evidence,
        )
    if not evidence:
        return _explainability_result(
            "fail",
            "Required explainability evidence is missing.",
            capability,
            shap_required,
            "missing" if shap_required else "not_applicable",
            evidence,
        )
    method = str(evidence.get("method") or evidence.get("primary_method") or "").strip().lower()
    paths = _artifact_paths(evidence)
    resolved = tuple(_resolve_artifact_path(path, base_dir) for path in paths)
    explicit_present = _first_bool(evidence, ("artifact_present", "shap_artifact_present"))
    artifact_present = explicit_present is True or any(artifact_exists(path) for path in resolved)
    if status in _FAIL_STATUSES or not artifact_present:
        return _explainability_result(
            "fail",
            "The explanation artifact is missing or reports failure.",
            capability,
            shap_required,
            "missing" if shap_required else "not_applicable",
            evidence,
            method=method,
            paths=[str(path) for path in resolved],
        )
    if shap_required and "shap" not in method:
        return _explainability_result(
            "fail",
            "This model family requires a SHAP-compatible artifact.",
            capability,
            shap_required,
            "wrong_method",
            evidence,
            method=method,
            paths=[str(path) for path in resolved],
        )
    allowed_methods = {capability.primary_method, *capability.fallback_methods}
    if not shap_required and method not in allowed_methods:
        return _explainability_result(
            "fail",
            "The explanation method does not match the registered family capability.",
            capability,
            shap_required,
            "not_applicable",
            evidence,
            method=method,
            allowed_methods=sorted(allowed_methods),
        )
    return _explainability_result(
        "pass",
        "Family-appropriate explainability evidence is present.",
        capability,
        shap_required,
        "present" if shap_required else "not_applicable",
        evidence,
        method=method,
        paths=[str(path) for path in resolved],
    )


def _drift_check(
    payload: JsonMapping,
    contexts: Sequence[JsonMapping],
    policy: PromotionPolicy,
) -> GateCheck:
    evidence = _evidence_mapping(payload, contexts, ("drift", "drift_metrics", "data_drift"))
    status = _normalized_status(evidence.get("status"))
    if status in _UNAVAILABLE_STATUSES:
        return _check(
            "drift",
            "warn",
            "Drift evidence is explicitly unavailable and blocks promotion.",
            declared_status=status,
            reason=evidence.get("reason"),
        )
    values = _drift_values(evidence)
    within_threshold = _first_bool(evidence, ("within_threshold", "passes_threshold"))
    if not evidence:
        return _check("drift", "fail", "Drift evidence is missing.")
    if status in _FAIL_STATUSES:
        return _check("drift", "fail", "Drift evidence reports failure.", declared_status=status, metrics=values)
    if not values and within_threshold is not True:
        return _check(
            "drift",
            "fail",
            "Drift evidence lacks a supported metric or explicit threshold result.",
            declared_status=status,
        )
    maximum = max(values.values()) if values else None
    if maximum is not None and maximum > policy.max_drift_value:
        return _check(
            "drift",
            "fail",
            "A drift metric exceeds the allowed maximum.",
            declared_status=status,
            metrics=values,
            maximum_allowed=policy.max_drift_value,
        )
    return _check(
        "drift",
        "pass",
        "Drift evidence passes.",
        declared_status=status,
        metrics=values,
        maximum_allowed=policy.max_drift_value,
    )


def _reproducibility_check(
    payload: JsonMapping,
    contexts: Sequence[JsonMapping],
    sources: Sequence[str],
    policy: PromotionPolicy,
) -> GateCheck:
    reproducibility = _evidence_mapping(payload, contexts, ("reproducibility", "run_metadata"))
    configs = [_mapping(payload.get("config")), *(_mapping(context.get("config")) for context in contexts)]
    config = next((value for value in configs if value), {})
    combined_sources = [payload, reproducibility, config, *contexts]
    values: dict[str, object] = {
        "config": bool(config),
        "random_seed": _first_value(combined_sources, ("random_seed", "seed")),
        "dataset_fingerprint": _first_value(
            combined_sources,
            ("dataset_fingerprint", "data_fingerprint", "dataset_hash"),
        ),
        "temporal_split": _first_value(
            combined_sources,
            ("temporal_split", "split_strategy", "split_definition", "split_version"),
        ),
        "code_version": _first_value(combined_sources, ("code_version", "git_commit", "commit_sha", "package_version")),
    }
    if "tuning" in sources:
        values["tuning_trials"] = _first_value(
            [payload, reproducibility, *contexts],
            ("completed_trials", "trial_count", "n_trials", "trials"),
        )
        values["best_params"] = _first_value([payload, reproducibility, *contexts], ("best_params", "parameters"))
    missing = [
        field
        for field in policy.required_reproducibility_fields
        if not _has_reproducibility_value(values.get(field))
    ]
    if "tuning" in sources:
        if not _positive_trial_count(values.get("tuning_trials")):
            missing.append("tuning_trials")
        if not isinstance(values.get("best_params"), Mapping) or not values["best_params"]:
            missing.append("best_params")
    if missing:
        return _check(
            "reproducibility",
            "fail",
            "Required reproducibility metadata is missing.",
            missing=sorted(set(missing)),
            metadata=values,
        )
    return _check(
        "reproducibility",
        "pass",
        "Configuration and reproducibility metadata are complete.",
        metadata=values,
    )


def _public_provenance_check(
    payload: JsonMapping,
    contexts: Sequence[JsonMapping],
    sources: Sequence[str],
) -> GateCheck:
    transferred = "public_pretraining" in sources or _first_bool(
        payload,
        ("transferred", "public_pretrained", "pretrained_on_public_data"),
    )
    if not transferred:
        return _check(
            "public_license_provenance",
            "pass",
            "Public-license provenance is not required for this model.",
            required=False,
            transferred=False,
        )
    records = _public_dataset_records(payload, contexts)
    failures: list[str] = []
    if not records:
        failures.append("dataset_records")
    for index, record in enumerate(records):
        prefix = f"datasets[{index}]"
        if not _first_value([record], ("dataset", "dataset_name", "name", "source")):
            failures.append(f"{prefix}.dataset")
        if not _first_value([record], ("version", "dataset_version", "release")):
            failures.append(f"{prefix}.version")
        if not _first_value([record], ("license", "license_name", "spdx_id")):
            failures.append(f"{prefix}.license")
        if not _first_value([record], ("provenance", "source_url", "dataset_url", "citation")):
            failures.append(f"{prefix}.provenance")
        if not _first_value([record], ("allowed_use", "permitted_use", "usage_rights")):
            failures.append(f"{prefix}.allowed_use")
        if _first_bool(record, ("license_validated", "license_approved", "approved")) is not True:
            failures.append(f"{prefix}.license_validated")
        if _first_bool(record, ("spotify_platform_content", "spotify_api_content")) is not False:
            failures.append(f"{prefix}.spotify_platform_content")
    if failures:
        return _check(
            "public_license_provenance",
            "fail",
            "Transferred models require complete, validated public-license provenance.",
            required=True,
            record_count=len(records),
            missing_or_invalid=sorted(set(failures)),
        )
    return _check(
        "public_license_provenance",
        "pass",
        "Public dataset licenses and provenance are validated.",
        required=True,
        record_count=len(records),
    )


def _check(
    name: str,
    status: str,
    summary: str,
    *,
    required: bool = True,
    **evidence: object,
) -> GateCheck:
    return GateCheck(name=name, status=status, required=required, summary=summary, evidence=evidence)


def _explainability_result(
    status: str,
    summary: str,
    capability: ExplainabilityCapability,
    shap_required: bool,
    shap_artifact_status: str,
    raw_evidence: JsonMapping,
    **extra: object,
) -> GateCheck:
    return _check(
        "explainability",
        status,
        summary,
        family=capability.family,
        strategy=capability.strategy,
        primary_method=capability.primary_method,
        shap_required=shap_required,
        shap_artifact_status=shap_artifact_status,
        declared_status=_normalized_status(raw_evidence.get("status")),
        **extra,
    )


def _resolve_primary_metric(payload: JsonMapping, policy: PromotionPolicy) -> str | None:
    if policy.primary_metric:
        return policy.primary_metric
    declared = str(payload.get("primary_metric") or "").strip()
    if declared:
        return declared
    validation = _mapping(payload.get("validation"))
    test = _mapping(payload.get("test"))
    for metric in _METRIC_PRIORITY:
        if _finite_float(validation.get(metric)) is not None and _finite_float(test.get(metric)) is not None:
            return metric
    common = sorted(set(validation).intersection(test))
    return next(
        (
            key
            for key in common
            if isinstance(key, str)
            and _finite_float(validation.get(key)) is not None
            and _finite_float(test.get(key)) is not None
        ),
        None,
    )


def _model_family(payload: JsonMapping, model_name: str, sources: Sequence[str]) -> str:
    declared = str(payload.get("model_family") or payload.get("family") or "").strip().lower()
    if declared:
        return declared.replace(" ", "_")
    normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"session_cooccurrence", "ease", "implicit_als", "bpr", "two_tower", "dual_encoder"}:
        return "retrieval"
    if normalized in {"mmoe", "ple"}:
        return "multitask"
    if normalized in {"meantime", "tisasrec", "sasrec", "bert4rec"}:
        return "neural_sequence"
    if "dcn" in normalized or "dcn" in sources:
        return "dcn"
    return normalized


def _model_name(payload: JsonMapping) -> str:
    for key in _MODEL_NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _artifact_paths(payload: JsonMapping) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("checkpoint", "checkpoint_path", "model_artifact", "model_path", "artifact_path"):
        value = payload.get(key)
        if isinstance(value, (str, Path)) and str(value).strip():
            values.append(str(value))
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("checkpoint", "model", "model_artifact", "path"):
            value = artifacts.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                values.append(str(value))
    elif isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
        values.extend(str(value) for value in artifacts if isinstance(value, (str, Path)) and str(value).strip())
    return tuple(dict.fromkeys(values))


def _resolve_artifact_path(path: str, base_dir: Path | None) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute() or base_dir is None:
        return value
    based = base_dir / value
    if based.exists() or not value.exists():
        return based
    return value


def _evidence_mapping(
    payload: JsonMapping,
    contexts: Sequence[JsonMapping],
    keys: Sequence[str],
) -> JsonMapping:
    direct = _first_mapping(payload, keys)
    if direct:
        return direct
    model_name = _model_name(payload)
    for context in contexts:
        evidence = _first_mapping(context, keys)
        if evidence:
            model_evidence = evidence.get(model_name)
            if isinstance(model_evidence, Mapping):
                return model_evidence
            return evidence
    return {}


def _public_dataset_records(payload: JsonMapping, contexts: Sequence[JsonMapping]) -> tuple[JsonMapping, ...]:
    containers = [payload, *contexts]
    for container in containers:
        for key in ("public_datasets", "datasets", "license_provenance", "provenance_records"):
            value = container.get(key)
            if isinstance(value, Mapping):
                if any(field in value for field in ("license", "license_name", "dataset", "dataset_name")):
                    return (value,)
                records = tuple(record for record in value.values() if isinstance(record, Mapping))
                if records:
                    return records
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                records = tuple(record for record in value if isinstance(record, Mapping))
                if records:
                    return records
    return ()


def _drift_values(evidence: JsonMapping) -> dict[str, float]:
    values: dict[str, float] = {}

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                visit(f"{prefix}.{key}" if prefix else str(key), child)
            return
        key = prefix.lower()
        if any(token in key for token in ("jsd", "psi", "drift_value", "drift_score", "distance")):
            number = _finite_float(value)
            if number is not None:
                values[prefix] = number

    visit("", evidence)
    return values


def _normalized_status(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _first_mapping(payload: JsonMapping, keys: Sequence[str]) -> JsonMapping:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _mapping(value: object) -> JsonMapping:
    return value if isinstance(value, Mapping) else {}


def _first_float(payload: JsonMapping, keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _finite_float(payload.get(key))
        if value is not None:
            return value
    return None


def _first_bool(payload: JsonMapping, keys: Sequence[str]) -> bool | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            return value
    return None


def _first_number(sources: Sequence[JsonMapping], keys: Sequence[str]) -> int | None:
    value = _first_value(sources, keys)
    number = _finite_float(value)
    return int(number) if number is not None else None


def _first_value(sources: Sequence[JsonMapping], keys: Sequence[str]) -> object | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None and value != "":
                return value
    return None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _has_reproducibility_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) or isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return value is not None and str(value).strip() != ""


def _positive_trial_count(value: object) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    number = _finite_float(value)
    return number is not None and number > 0


def _deep_merge(left: JsonMapping, right: JsonMapping) -> dict[str, object]:
    merged = dict(left)
    for key, value in right.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple) or isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _markdown_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "GATE_STATUSES",
    "GateCheck",
    "ModelPromotionDecision",
    "PromotionGateReport",
    "PromotionPolicy",
    "evaluate_track_expansion_gates",
]
