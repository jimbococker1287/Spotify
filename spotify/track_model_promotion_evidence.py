from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from .run_artifacts import safe_read_json, write_json, write_markdown


JsonMapping = Mapping[str, object]

SUPPORTED_TRACK_PROMOTION_MODELS = (
    "session_cooccurrence",
    "ease",
    "meantime",
    "mmoe",
    "ple",
)

_RETRIEVAL_MODELS = {"session_cooccurrence", "ease"}
_NEURAL_SEQUENCE_MODELS = {"meantime"}
_MULTITASK_MODELS = {"mmoe", "ple"}


@dataclass(frozen=True)
class PromotionEvidenceConfig:
    """Configuration for converting track-model outputs into gate evidence."""

    artifact_base_dir: Path | None = None
    generated_at: str | None = None
    code_version: str | None = None
    dataset_fingerprint: str | None = None
    temporal_split: str = "track_level_chronological_train_validation_test"
    supported_models: tuple[str, ...] = SUPPORTED_TRACK_PROMOTION_MODELS


@dataclass(frozen=True)
class TrackModelEvidence:
    """One model row normalized for promotion-gate consumption."""

    model_name: str
    model_family: str
    source_section: str
    primary_metric: str | None
    artifact_paths: tuple[str, ...]
    validation: JsonMapping
    test: JsonMapping
    popularity_baseline: JsonMapping
    calibration: JsonMapping
    explainability: JsonMapping
    drift: JsonMapping
    reproducibility: JsonMapping
    coverage: JsonMapping
    tuning: JsonMapping
    notes: tuple[str, ...]
    raw: JsonMapping

    @property
    def status(self) -> str:
        if not self.test:
            return "blocked"
        if any(_is_unavailable(value) for value in (self.calibration, self.explainability, self.drift)):
            return "warn"
        return "ready"

    def to_gate_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "status": self.raw.get("status", "complete"),
            "source_section": self.source_section,
            "primary_metric": self.primary_metric,
            "validation": dict(self.validation),
            "test": dict(self.test),
            "popularity_baseline": dict(self.popularity_baseline),
            "calibration": dict(self.calibration),
            "explainability": dict(self.explainability),
            "drift": dict(self.drift),
            "reproducibility": dict(self.reproducibility),
            "coverage": dict(self.coverage),
            "evidence_notes": list(self.notes),
        }
        if self.artifact_paths:
            row["checkpoint"] = self.artifact_paths[0]
            row["artifacts"] = list(self.artifact_paths)
            row["artifact_present"] = True
        else:
            row["artifact_present"] = False
        row.update(_selected_passthrough_fields(self.raw))
        return row

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "source_section": self.source_section,
            "status": self.status,
            "primary_metric": self.primary_metric,
            "artifact_paths": list(self.artifact_paths),
            "validation": dict(self.validation),
            "test": dict(self.test),
            "popularity_baseline": dict(self.popularity_baseline),
            "calibration": dict(self.calibration),
            "explainability": dict(self.explainability),
            "drift": dict(self.drift),
            "reproducibility": dict(self.reproducibility),
            "coverage": dict(self.coverage),
            "tuning": dict(self.tuning),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TrackModelPromotionEvidence:
    """JSON- and Markdown-ready evidence bundle for the gate evaluator."""

    generated_at: str
    status: str
    models: tuple[TrackModelEvidence, ...]
    gate_training_manifest: JsonMapping
    gate_tuning_manifest: JsonMapping
    baseline_manifest: JsonMapping
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "model_count": len(self.models),
            "models": [model.to_dict() for model in self.models],
            "gate_training_manifest": _json_ready(self.gate_training_manifest),
            "gate_tuning_manifest": _json_ready(self.gate_tuning_manifest),
            "baseline_manifest": _json_ready(self.baseline_manifest),
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Track Model Promotion Evidence",
            "",
            f"- Status: `{self.status}`",
            f"- Generated: `{self.generated_at}`",
            f"- Models: `{len(self.models)}`",
            "",
            "| Model | Family | Source | Primary Metric | Validation | Test | Evidence Status |",
            "|---|---|---|---|---:|---:|---|",
        ]
        for model in self.models:
            metric = model.primary_metric or "unresolved"
            validation = _format_metric_value(model.validation.get(metric))
            test = _format_metric_value(model.test.get(metric))
            lines.append(
                f"| `{_md(model.model_name)}` | `{_md(model.model_family)}` | "
                f"`{_md(model.source_section)}` | `{_md(metric)}` | {validation} | {test} | "
                f"`{model.status}` |"
            )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {_md(note)}" for note in self.notes)
        for model in self.models:
            lines.extend(
                [
                    "",
                    f"## {_md(model.model_name)}",
                    "",
                    f"- Artifact paths: `{', '.join(model.artifact_paths) if model.artifact_paths else 'none'}`",
                    f"- Calibration: `{model.calibration.get('status', 'unknown')}`",
                    f"- Explainability: `{model.explainability.get('status', 'unknown')}`",
                    f"- Drift: `{model.drift.get('status', 'unknown')}`",
                ]
            )
            if model.notes:
                lines.append(f"- Notes: {_md('; '.join(model.notes))}")
        return "\n".join(lines) + "\n"


def build_track_model_promotion_evidence(
    *,
    training_manifest: JsonMapping,
    baseline_manifest: JsonMapping | None = None,
    tuning_manifest: JsonMapping | None = None,
    config: PromotionEvidenceConfig | None = None,
) -> TrackModelPromotionEvidence:
    """Convert track expansion outputs into promotion-gate-ready manifests.

    Missing calibration, explanation, and drift artifacts are represented as
    explicit unavailable evidence so the promotion gate blocks honestly instead
    of silently ignoring the gap.
    """

    effective_config = config or PromotionEvidenceConfig()
    supported = {name.lower() for name in effective_config.supported_models}
    baseline = _normalize_baseline(baseline_manifest or {})
    generated_at = effective_config.generated_at or datetime.now(timezone.utc).isoformat()
    code_version = effective_config.code_version or _manifest_value(training_manifest, "code_version") or _code_version()
    dataset_fingerprint = (
        effective_config.dataset_fingerprint
        or _manifest_value(training_manifest, "dataset_fingerprint")
        or _derived_dataset_fingerprint(training_manifest)
    )
    training_config = _mapping(training_manifest.get("config"))
    reproducibility_base = {
        "config": dict(training_config),
        "random_seed": _first_value((training_config, training_manifest), ("random_seed", "seed")),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_fingerprint_source": (
            "provided"
            if effective_config.dataset_fingerprint or _manifest_value(training_manifest, "dataset_fingerprint")
            else "derived_from_training_manifest_summary"
        ),
        "temporal_split": _manifest_value(training_manifest, "temporal_split") or effective_config.temporal_split,
        "code_version": code_version,
    }
    tensor_summary = _mapping(training_manifest.get("tensor_summary"))
    dataset_summary = _mapping(training_manifest.get("dataset"))
    tuning_by_model = _collect_tuning_rows(tuning_manifest or {}, supported)

    models: list[TrackModelEvidence] = []
    for section, rows in (
        ("retrieval_results", _sequence_of_mappings(training_manifest.get("retrieval_results"))),
        ("neural_results", _sequence_of_mappings(training_manifest.get("neural_results"))),
    ):
        for row in rows:
            model_name = str(row.get("model_name", "")).strip().lower()
            if model_name not in supported:
                continue
            models.append(
                _convert_model_row(
                    row=row,
                    source_section=section,
                    baseline=baseline,
                    tensor_summary=tensor_summary,
                    dataset_summary=dataset_summary,
                    reproducibility_base=reproducibility_base,
                    tuning=tuning_by_model.get(model_name, {}),
                    artifact_base_dir=effective_config.artifact_base_dir,
                )
            )

    ordered = tuple(sorted(models, key=lambda model: model.model_name))
    notes = _bundle_notes(ordered, training_manifest)
    gate_training_manifest = {
        "generated_at": generated_at,
        "status": "complete" if ordered else "blocked",
        "config": dict(training_config),
        "dataset": dict(dataset_summary),
        "tensor_summary": dict(tensor_summary),
        "dataset_fingerprint": dataset_fingerprint,
        "temporal_split": reproducibility_base["temporal_split"],
        "code_version": code_version,
        "retrieval_results": [model.to_gate_row() for model in ordered if model.source_section == "retrieval_results"],
        "neural_results": [model.to_gate_row() for model in ordered if model.source_section == "neural_results"],
    }
    gate_tuning_manifest = {
        "status": "complete" if tuning_by_model else "not_available",
        "tuning_results": [dict(value) for _name, value in sorted(tuning_by_model.items())],
    }
    status = "blocked" if not ordered else "warn" if any(model.status != "ready" for model in ordered) else "ready"
    return TrackModelPromotionEvidence(
        generated_at=generated_at,
        status=status,
        models=ordered,
        gate_training_manifest=gate_training_manifest,
        gate_tuning_manifest=gate_tuning_manifest,
        baseline_manifest=baseline,
        notes=notes,
    )


def load_track_model_promotion_evidence(
    *,
    training_manifest_path: Path,
    baseline_manifest_path: Path | None = None,
    tuning_manifest_path: Path | None = None,
    config: PromotionEvidenceConfig | None = None,
) -> TrackModelPromotionEvidence:
    """Read JSON manifests from disk and build normalized promotion evidence."""

    training = safe_read_json(training_manifest_path, default={})
    if not isinstance(training, Mapping):
        raise ValueError(f"Training manifest is not a JSON object: {training_manifest_path}")
    baseline = {}
    if baseline_manifest_path is not None:
        loaded_baseline = safe_read_json(baseline_manifest_path, default={})
        if isinstance(loaded_baseline, Mapping):
            baseline = loaded_baseline
    tuning = {}
    if tuning_manifest_path is not None:
        loaded_tuning = safe_read_json(tuning_manifest_path, default={})
        if isinstance(loaded_tuning, Mapping):
            tuning = loaded_tuning
    return build_track_model_promotion_evidence(
        training_manifest=training,
        baseline_manifest=baseline,
        tuning_manifest=tuning,
        config=config,
    )


def write_track_model_promotion_evidence(
    evidence: TrackModelPromotionEvidence,
    output_dir: Path,
    *,
    json_name: str = "track_model_promotion_evidence.json",
    markdown_name: str = "track_model_promotion_evidence.md",
) -> tuple[Path, Path]:
    """Write the normalized evidence bundle as JSON and Markdown."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / json_name, evidence.to_dict())
    markdown_path = write_markdown(output_dir / markdown_name, evidence.to_markdown())
    return json_path, markdown_path


def _convert_model_row(
    *,
    row: JsonMapping,
    source_section: str,
    baseline: JsonMapping,
    tensor_summary: JsonMapping,
    dataset_summary: JsonMapping,
    reproducibility_base: JsonMapping,
    tuning: JsonMapping,
    artifact_base_dir: Path | None,
) -> TrackModelEvidence:
    model_name = str(row.get("model_name", "")).strip().lower()
    family = _model_family(model_name)
    validation = _normalize_split_metrics(row, "validation")
    test = _normalize_split_metrics(row, "test")
    if source_section == "retrieval_results" and not validation:
        validation = _alias_metrics(row)
    notes: list[str] = []
    if source_section == "retrieval_results" and not test:
        notes.append("Retrieval benchmark output has no temporal test metrics; rerun with a test evaluation before promotion.")
    artifact_paths = _artifact_paths(row, artifact_base_dir)
    if source_section == "retrieval_results" and not artifact_paths:
        notes.append("Retrieval benchmark output did not persist a reusable model artifact.")
    calibration = _existing_or_unavailable(
        row,
        keys=("calibration", "calibration_metrics"),
        fallback_reason=f"{model_name} training output does not include calibration/ECE evidence.",
        method=None,
    )
    explainability = _existing_or_unavailable(
        row,
        keys=("explainability", "explanations", "shap"),
        fallback_reason=f"{model_name} training output does not include family-specific explanation artifacts.",
        method=_default_explanation_method(family),
    )
    drift = _existing_or_unavailable(
        row,
        keys=("drift", "drift_metrics", "data_drift"),
        fallback_reason=f"{model_name} training output does not include validation/test drift evidence.",
        method=None,
    )
    coverage = _coverage_payload(
        row=row,
        source_section=source_section,
        validation=validation,
        test=test,
        tensor_summary=tensor_summary,
        dataset_summary=dataset_summary,
    )
    primary_metric = _primary_metric(validation, test)
    popularity = _baseline_for_metric(baseline, primary_metric)
    reproducibility = {
        **dict(reproducibility_base),
        **dict(_mapping(row.get("reproducibility"))),
    }
    if tuning:
        reproducibility["completed_trials"] = tuning.get("completed_trials")
        reproducibility["best_params"] = tuning.get("best_params")
    return TrackModelEvidence(
        model_name=model_name,
        model_family=family,
        source_section=source_section,
        primary_metric=primary_metric,
        artifact_paths=artifact_paths,
        validation=validation,
        test=test,
        popularity_baseline=popularity,
        calibration=calibration,
        explainability=explainability,
        drift=drift,
        reproducibility=reproducibility,
        coverage=coverage,
        tuning=tuning,
        notes=tuple(notes),
        raw=row,
    )


def _normalize_split_metrics(row: JsonMapping, split_name: str) -> dict[str, object]:
    split = _mapping(row.get(split_name))
    return _alias_metrics(split)


def _alias_metrics(metrics: JsonMapping) -> dict[str, object]:
    payload = dict(metrics)
    k = _finite_int(payload.get("k"))
    if k is not None:
        for source, prefix in (
            ("recall_at_k", "recall_at"),
            ("ndcg_at_k", "ndcg_at"),
            ("mrr_at_k", "mrr_at"),
            ("hit_rate_at_k", "hit_rate_at"),
        ):
            value = payload.get(source)
            if _finite_float(value) is not None:
                payload[f"{prefix}_{k}"] = value
    if "recall_at_100" not in payload and _finite_float(payload.get("recall_at_k")) is not None:
        payload["recall_at_100"] = payload["recall_at_k"]
    if "ndcg_at_10" not in payload and _finite_float(payload.get("ndcg_at_k")) is not None and k == 10:
        payload["ndcg_at_10"] = payload["ndcg_at_k"]
    return payload


def _normalize_baseline(baseline: JsonMapping) -> dict[str, object]:
    payload = _alias_metrics(baseline)
    test = _mapping(payload.get("test"))
    if test:
        payload["test"] = _alias_metrics(test)
    return payload


def _baseline_for_metric(baseline: JsonMapping, primary_metric: str | None) -> dict[str, object]:
    payload = dict(baseline)
    if primary_metric and primary_metric not in payload:
        nested_test = _mapping(payload.get("test"))
        if primary_metric in nested_test:
            payload[primary_metric] = nested_test[primary_metric]
    return payload


def _primary_metric(validation: JsonMapping, test: JsonMapping) -> str | None:
    priority = ("ndcg_at_10", "recall_at_100", "recall_at_k", "ndcg_at_k", "mrr_at_100", "mrr_at_k")
    for name in priority:
        if _finite_float(validation.get(name)) is not None and _finite_float(test.get(name)) is not None:
            return name
    for name in priority:
        if _finite_float(validation.get(name)) is not None:
            return name
    return None


def _coverage_payload(
    *,
    row: JsonMapping,
    source_section: str,
    validation: JsonMapping,
    test: JsonMapping,
    tensor_summary: JsonMapping,
    dataset_summary: JsonMapping,
) -> dict[str, object]:
    if source_section == "retrieval_results":
        return {
            "train_rows": _finite_int(row.get("training_interactions")) or _finite_int(dataset_summary.get("train_examples")),
            "validation_rows": _finite_int(row.get("evaluated_examples")) or _finite_int(dataset_summary.get("validation_examples")),
            "test_rows": _finite_int(dataset_summary.get("test_examples")),
            "validation_target_coverage": _finite_float(validation.get("target_catalog_coverage")),
            "test_target_coverage": _finite_float(test.get("target_catalog_coverage")),
        }
    return {
        "train_rows": _finite_int(tensor_summary.get("train_rows")) or _finite_int(dataset_summary.get("train_examples")),
        "validation_rows": _finite_int(tensor_summary.get("validation_rows"))
        or _finite_int(dataset_summary.get("validation_examples")),
        "test_rows": _finite_int(tensor_summary.get("test_rows")) or _finite_int(dataset_summary.get("test_examples")),
        "validation_target_coverage": _finite_float(
            validation.get("target_vocabulary_coverage") or tensor_summary.get("validation_target_vocabulary_coverage")
        ),
        "test_target_coverage": _finite_float(
            test.get("target_vocabulary_coverage") or tensor_summary.get("test_target_vocabulary_coverage")
        ),
    }


def _collect_tuning_rows(tuning_manifest: JsonMapping, supported: set[str]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in _candidate_tuning_rows(tuning_manifest):
        model_name = str(row.get("model_name", "")).strip().lower()
        if model_name not in supported:
            continue
        best_trial = _mapping(row.get("best_trial"))
        params = _mapping(row.get("best_params")) or _mapping(row.get("parameters")) or _mapping(best_trial.get("params"))
        completed_trials = (
            _finite_int(row.get("completed_trials"))
            or _finite_int(row.get("trial_count"))
            or _finite_int(row.get("total_trials"))
        )
        rows[model_name] = {
            "model_name": model_name,
            "completed_trials": completed_trials,
            "trial_count": _finite_int(row.get("total_trials")) or completed_trials,
            "best_params": dict(params),
            "parameters": dict(params),
            "tuning_metric": row.get("tuning_metric") or row.get("metric_name") or best_trial.get("metric_name"),
            "tuning_value": row.get("tuning_value") or best_trial.get("value"),
            "storage_path": tuning_manifest.get("storage_path"),
        }
    return rows


def _candidate_tuning_rows(tuning_manifest: JsonMapping) -> tuple[JsonMapping, ...]:
    rows: list[JsonMapping] = []
    for key in ("tuning_results", "best_models", "models", "results", "studies"):
        value = tuning_manifest.get(key)
        if isinstance(value, Mapping):
            for model_name, payload in sorted(value.items(), key=lambda item: str(item[0])):
                if isinstance(payload, Mapping):
                    rows.append({"model_name": str(model_name), **payload})
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rows.extend(item for item in value if isinstance(item, Mapping))
    return tuple(rows)


def _existing_or_unavailable(
    row: JsonMapping,
    *,
    keys: Sequence[str],
    fallback_reason: str,
    method: str | None,
) -> dict[str, object]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    payload: dict[str, object] = {
        "status": "unavailable",
        "reason": fallback_reason,
        "promotion_impact": "blocks_promotion",
    }
    if method:
        payload["method"] = method
    return payload


def _model_family(model_name: str) -> str:
    if model_name in _RETRIEVAL_MODELS:
        return "retrieval"
    if model_name in _NEURAL_SEQUENCE_MODELS:
        return "neural_sequence"
    if model_name in _MULTITASK_MODELS:
        return "multitask"
    return model_name


def _default_explanation_method(model_family: str) -> str:
    if model_family == "retrieval":
        return "candidate_recall"
    if model_family == "multitask":
        return "per_head_integrated_gradients"
    return "integrated_gradients"


def _artifact_paths(row: JsonMapping, base_dir: Path | None) -> tuple[str, ...]:
    paths: list[str] = []
    for key in ("checkpoint", "checkpoint_path", "model_artifact", "model_path", "artifact_path"):
        value = row.get(key)
        if isinstance(value, (str, Path)) and str(value).strip():
            paths.append(_normalize_path(value, base_dir))
    artifacts = row.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("checkpoint", "model", "model_artifact", "path"):
            value = artifacts.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                paths.append(_normalize_path(value, base_dir))
    elif isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
        for value in artifacts:
            if isinstance(value, (str, Path)) and str(value).strip():
                paths.append(_normalize_path(value, base_dir))
    return tuple(dict.fromkeys(paths))


def _normalize_path(value: str | Path, base_dir: Path | None) -> str:
    path = Path(value).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if base_dir is not None:
            candidates.append(base_dir / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0])


def _selected_passthrough_fields(row: JsonMapping) -> dict[str, object]:
    allowed = (
        "fit_seconds",
        "epochs",
        "final_train_loss",
        "catalog_items",
        "training_sessions",
        "training_interactions",
        "weight_matrix_mb",
        "evaluated_examples",
        "candidate_catalog_coverage",
        "mean_candidate_count",
        "exclude_seen",
    )
    return {key: row[key] for key in allowed if key in row}


def _bundle_notes(models: Sequence[TrackModelEvidence], training_manifest: JsonMapping) -> tuple[str, ...]:
    notes: list[str] = []
    if not models:
        notes.append("No supported track promotion models were found in the training manifest.")
    if not _mapping(training_manifest.get("tensor_summary")):
        notes.append("Tensor summary is missing; neural split coverage may be incomplete.")
    if any(model.source_section == "retrieval_results" and not model.test for model in models):
        notes.append("At least one retrieval model lacks temporal test metrics.")
    if any(_is_unavailable(model.calibration) for model in models):
        notes.append("Calibration evidence is unavailable for at least one model.")
    if any(_is_unavailable(model.explainability) for model in models):
        notes.append("Explainability evidence is unavailable for at least one model.")
    if any(_is_unavailable(model.drift) for model in models):
        notes.append("Drift evidence is unavailable for at least one model.")
    return tuple(notes)


def _derived_dataset_fingerprint(training_manifest: JsonMapping) -> str:
    payload = {
        "dataset": _json_ready(_mapping(training_manifest.get("dataset"))),
        "tensor_summary": _json_ready(_mapping(training_manifest.get("tensor_summary"))),
        "config": _json_ready(_mapping(training_manifest.get("config"))),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _mapping(value: object) -> JsonMapping:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> tuple[JsonMapping, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _manifest_value(manifest: JsonMapping, key: str) -> object | None:
    value = manifest.get(key)
    if value is not None and value != "":
        return value
    metadata = _mapping(manifest.get("reproducibility")) or _mapping(manifest.get("run_metadata"))
    value = metadata.get(key)
    return value if value is not None and value != "" else None


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
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _finite_int(value: object) -> int | None:
    number = _finite_float(value)
    return int(number) if number is not None else None


def _is_unavailable(value: JsonMapping) -> bool:
    return str(value.get("status", "")).strip().lower().replace(" ", "_") in {
        "unavailable",
        "not_available",
        "not_supported",
        "unsupported",
        "not_applicable",
    }


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _format_metric_value(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "`missing`"
    return f"`{number:.6f}`"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "PromotionEvidenceConfig",
    "SUPPORTED_TRACK_PROMOTION_MODELS",
    "TrackModelEvidence",
    "TrackModelPromotionEvidence",
    "build_track_model_promotion_evidence",
    "load_track_model_promotion_evidence",
    "write_track_model_promotion_evidence",
]
