from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType


READINESS_LEVELS = (
    "component_ready",
    "integration_pending",
    "planned",
    "research_only",
)


@dataclass(frozen=True)
class SearchParameter:
    """Framework-neutral metadata for one Optuna-compatible parameter."""

    name: str
    kind: str
    low: int | float | None = None
    high: int | float | None = None
    choices: tuple[object, ...] = ()
    step: int | float | None = None
    log: bool = False
    condition: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OptunaSupport:
    supported: bool
    objective: str
    direction: str
    search_space: tuple[SearchParameter, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "objective": self.objective,
            "direction": self.direction,
            "search_space": [parameter.to_dict() for parameter in self.search_space],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ExpansionModelSpec:
    """Auditable contract for an expansion family."""

    key: str
    family: str
    stage: str
    summary: str
    implementation_modules: tuple[str, ...]
    required_data: tuple[str, ...]
    optional_data: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    guardrail_metrics: tuple[str, ...]
    explainability_family: str
    optuna: OptunaSupport
    readiness: str
    blockers: tuple[str, ...] = ()
    governance_requirements: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["optuna"] = self.optuna.to_dict()
        return payload


def _integer(name: str, low: int, high: int, *, step: int | None = None, condition: str = "") -> SearchParameter:
    return SearchParameter(name=name, kind="int", low=low, high=high, step=step, condition=condition)


def _float(
    name: str,
    low: float,
    high: float,
    *,
    step: float | None = None,
    log: bool = False,
    condition: str = "",
) -> SearchParameter:
    return SearchParameter(
        name=name,
        kind="float",
        low=low,
        high=high,
        step=step,
        log=log,
        condition=condition,
    )


def _categorical(name: str, *choices: object, condition: str = "") -> SearchParameter:
    return SearchParameter(name=name, kind="categorical", choices=tuple(choices), condition=condition)


_RANKING_GOVERNANCE = (
    "Use temporal train/validation/test boundaries and fit vocabularies on training data only.",
    "Report catalog coverage, popularity slices, repeat rate, and long-tail behavior with ranking metrics.",
    "Record the candidate source so retrieval misses are not attributed to the reranker.",
)
_MULTITASK_GOVERNANCE = (
    "Report every task head separately and preserve loss weights in artifacts.",
    "Audit missing labels and weak-label confidence; absence of a save is not an explicit negative.",
)
_PUBLIC_DATA_GOVERNANCE = (
    "Record dataset version, license, provenance, and allowed use before ingestion.",
    "Keep public identities and item namespaces separate from the private personal export.",
    "Do not train on Spotify Platform or Spotify API content.",
)
_MULTIMODAL_GOVERNANCE = (
    "Train only on audio, text, and images whose licenses permit the intended use.",
    "Track modality availability and compare against metadata-only and popularity baselines.",
)


EXPANSION_MODEL_REGISTRY: Mapping[str, ExpansionModelSpec] = MappingProxyType(
    {
        "track_level_retrieval": ExpansionModelSpec(
            key="track_level_retrieval",
            family="two_stage_retrieval",
            stage="candidate_generation",
            summary="Long-tail next-track retrieval using track URIs and leakage-safe session histories.",
            implementation_modules=("spotify.track_level_data", "spotify.track_retrieval", "spotify.retrieval"),
            required_data=("track_uri", "timestamp", "session_id", "ordered_history"),
            optional_data=("context_features", "saved_track_labels", "playlist_membership", "search_intent"),
            primary_metrics=("recall_at_k", "ndcg_at_k", "mrr_at_k"),
            guardrail_metrics=("catalog_coverage_at_k", "long_tail_recall_at_k", "repeat_rate_at_k"),
            explainability_family="retrieval",
            optuna=OptunaSupport(
                supported=True,
                objective="validation_recall_at_100",
                direction="maximize",
                search_space=(
                    _integer("embedding_dim", 32, 256, step=32),
                    _integer("candidate_k", 50, 500, step=50),
                    _float("learning_rate", 1e-5, 5e-3, log=True),
                    _integer("negative_samples", 5, 100, step=5),
                    _categorical("negative_sampler", "uniform", "popularity", "in_batch", "mixed"),
                ),
                notes="Tune on candidate recall before reranker metrics.",
            ),
            readiness="component_ready",
            blockers=("Clear temporal, explainability, calibration, and drift promotion gates.",),
            governance_requirements=_RANKING_GOVERNANCE,
        ),
        "dcn_v2_reranker": ExpansionModelSpec(
            key="dcn_v2_reranker",
            family="contextual_reranking",
            stage="reranking",
            summary="DCN-V2 feature-cross reranker over candidate, session, and context features.",
            implementation_modules=(
                "spotify.dcn_v2_model",
                "spotify.track_reranking_data",
                "spotify.track_dcn_training",
                "spotify.track_next_pass",
            ),
            required_data=("candidate_ids", "retrieval_scores", "context_features", "ranking_labels"),
            optional_data=("item_features", "time_gap_features", "technical_friction_features"),
            primary_metrics=("ndcg_at_10", "mrr_at_10", "hit_rate_at_10"),
            guardrail_metrics=("ece", "catalog_coverage_at_10", "long_tail_exposure_at_10"),
            explainability_family="dcn",
            optuna=OptunaSupport(
                supported=True,
                objective="validation_ndcg_at_10",
                direction="maximize",
                search_space=(
                    _integer("cross_layers", 1, 6),
                    _categorical("cross_parameterization", "matrix", "vector"),
                    _categorical("architecture", "parallel", "stacked"),
                    _categorical("deep_units", (64, 32), (128, 64), (256, 128, 64)),
                    _float("dropout_rate", 0.0, 0.5),
                    _float("l2_regularization", 1e-8, 1e-2, log=True),
                ),
            ),
            readiness="component_ready",
            blockers=("Run broader Optuna budgets and clear every promotion gate.",),
            governance_requirements=_RANKING_GOVERNANCE,
        ),
        "multitask_mmoe_ple": ExpansionModelSpec(
            key="multitask_mmoe_ple",
            family="multitask_ranking",
            stage="prediction",
            summary="MMoE or PLE shared encoder with next-item, skip, dwell, session-end, positive, and repeat heads.",
            implementation_modules=("spotify.multitask_model", "spotify.track_level_data"),
            required_data=("ordered_history", "next_item", "skip", "dwell", "session_end", "repeat"),
            optional_data=("saved_track_labels", "playlist_membership", "weak_label_confidence"),
            primary_metrics=("next_item_ndcg_at_10", "skip_auroc", "dwell_mae", "session_end_auroc"),
            guardrail_metrics=("per_head_ece", "negative_transfer_delta", "head_missing_label_rate"),
            explainability_family="multitask",
            optuna=OptunaSupport(
                supported=True,
                objective="weighted_validation_task_score",
                direction="maximize",
                search_space=(
                    _categorical("architecture", "mmoe", "ple"),
                    _integer("num_experts", 2, 8),
                    _integer("expert_units", 32, 256, step=32),
                    _integer("task_experts", 1, 4, condition="architecture == 'ple'"),
                    _categorical("tower_units", (32,), (64, 32), (128, 64)),
                    _float("dropout_rate", 0.0, 0.5),
                    _float("learning_rate", 1e-5, 3e-3, log=True),
                ),
                notes="Loss weights should be tuned in a constrained study after establishing stable per-head baselines.",
            ),
            readiness="component_ready",
            blockers=("Connect all six labels to training and establish missing-label masking.",),
            governance_requirements=_RANKING_GOVERNANCE + _MULTITASK_GOVERNANCE,
        ),
        "meantime_tisasrec": ExpansionModelSpec(
            key="meantime_tisasrec",
            family="time_aware_sequence",
            stage="sequence_encoder",
            summary="Temporal self-attention using absolute time, relative gaps, and causal sequence masking.",
            implementation_modules=("spotify.meantime_model",),
            required_data=("ordered_item_sequence", "timestamps", "time_gaps"),
            optional_data=("context_features", "session_boundaries"),
            primary_metrics=("ndcg_at_10", "mrr_at_10", "recall_at_10"),
            guardrail_metrics=("performance_by_gap_bucket", "catalog_coverage_at_10", "ece"),
            explainability_family="neural_sequence",
            optuna=OptunaSupport(
                supported=True,
                objective="validation_ndcg_at_10",
                direction="maximize",
                search_space=(
                    _integer("embedding_dim", 32, 256, step=32),
                    _integer("num_heads", 1, 8),
                    _integer("num_blocks", 1, 6),
                    _integer("relative_time_buckets", 16, 128, step=16),
                    _float("dropout_rate", 0.0, 0.5),
                    _float("learning_rate", 1e-5, 3e-3, log=True),
                ),
            ),
            readiness="component_ready",
            blockers=("Benchmark against SASRec with identical track-level splits and longer histories.",),
            governance_requirements=_RANKING_GOVERNANCE,
        ),
        "self_supervised_sequence": ExpansionModelSpec(
            key="self_supervised_sequence",
            family="s3rec_duorec",
            stage="pretraining",
            summary="S3-Rec or DuoRec-style pretraining over masked items, attributes, subsequences, and session views.",
            implementation_modules=("spotify.sequence_pretraining", "spotify.retrieval_pretraining"),
            required_data=("unlabeled_item_sequences", "session_boundaries"),
            optional_data=("item_attributes", "timestamps", "context_features"),
            primary_metrics=("fine_tuned_ndcg_at_10", "fine_tuned_recall_at_10", "label_efficiency"),
            guardrail_metrics=("augmentation_collision_rate", "representation_uniformity", "long_tail_recall_at_10"),
            explainability_family="neural_sequence",
            optuna=OptunaSupport(
                supported=True,
                objective="fine_tuned_validation_ndcg_at_10",
                direction="maximize",
                search_space=(
                    _categorical("objective", "masked_item", "s3rec", "duorec", "hybrid"),
                    _float("mask_rate", 0.05, 0.4),
                    _float("contrastive_temperature", 0.03, 0.5, log=True),
                    _float("contrastive_weight", 0.05, 2.0, log=True),
                    _integer("pretrain_epochs", 5, 100, step=5),
                ),
                notes="Study pretraining and fine-tuning budgets separately to avoid selecting on compute alone.",
            ),
            readiness="component_ready",
            blockers=("Wire pretraining batches into encoder training and compare fine-tuning gains.",),
            governance_requirements=_RANKING_GOVERNANCE,
            references=("S3-Rec", "DuoRec"),
        ),
        "session_linear_collaborative": ExpansionModelSpec(
            key="session_linear_collaborative",
            family="ease_als_bpr",
            stage="candidate_generation",
            summary="EASE, implicit ALS, and BPR baselines using sessions or playlists as pseudo-users.",
            implementation_modules=("spotify.track_retrieval",),
            required_data=("session_item_matrix",),
            optional_data=("playlist_item_matrix", "confidence_weights"),
            primary_metrics=("recall_at_k", "ndcg_at_k", "mrr_at_k"),
            guardrail_metrics=("catalog_coverage_at_k", "popularity_bias", "memory_mb"),
            explainability_family="retrieval",
            optuna=OptunaSupport(
                supported=True,
                objective="validation_recall_at_100",
                direction="maximize",
                search_space=(
                    _categorical("algorithm", "ease", "implicit_als", "bpr"),
                    _float("regularization", 1e-4, 1e4, log=True),
                    _integer("factors", 16, 256, step=16, condition="algorithm != 'ease'"),
                    _integer("iterations", 5, 100, step=5, condition="algorithm != 'ease'"),
                ),
            ),
            readiness="component_ready",
            blockers=("Add bounded sparse-matrix orchestration for full-catalog EASE, ALS, and BPR studies.",),
            governance_requirements=_RANKING_GOVERNANCE,
        ),
        "public_collaborative_transfer": ExpansionModelSpec(
            key="public_collaborative_transfer",
            family="lightgcn_recvae",
            stage="public_pretraining",
            summary="Multi-user collaborative benchmarks and transferable encoders trained on licensed public data.",
            implementation_modules=("spotify.public_training_data",),
            required_data=("licensed_public_user_item_interactions", "dataset_license_record"),
            optional_data=("timestamps", "public_item_metadata"),
            primary_metrics=("recall_at_k", "ndcg_at_k", "transfer_delta"),
            guardrail_metrics=("popularity_bias", "cold_item_recall", "private_fine_tune_regression"),
            explainability_family="retrieval",
            optuna=OptunaSupport(
                supported=True,
                objective="public_validation_ndcg_at_20",
                direction="maximize",
                search_space=(
                    _categorical("architecture", "lightgcn", "recvae"),
                    _integer("embedding_dim", 32, 256, step=32),
                    _integer("graph_layers", 1, 5, condition="architecture == 'lightgcn'"),
                    _float("learning_rate", 1e-5, 5e-3, log=True),
                    _float("regularization", 1e-8, 1e-2, log=True),
                ),
            ),
            readiness="integration_pending",
            blockers=(
                "Acquire approved local datasets and validate their manifests before enabling training.",
            ),
            governance_requirements=_RANKING_GOVERNANCE + _PUBLIC_DATA_GOVERNANCE,
            references=("LFM-1b", "Million Song Taste Profile"),
        ),
        "multimodal_content_tower": ExpansionModelSpec(
            key="multimodal_content_tower",
            family="multimodal_cold_start",
            stage="item_encoder",
            summary="Fuse licensed audio and metadata embeddings for cold-start retrieval and ranking.",
            implementation_modules=("spotify.multimodal_model", "spotify.public_training_data"),
            required_data=("licensed_audio_or_embeddings", "item_metadata", "item_identity_map"),
            optional_data=("album_art_embeddings", "artist_text_embeddings", "genre_labels"),
            primary_metrics=("cold_item_recall_at_k", "cold_item_ndcg_at_k", "retrieval_recall_at_k"),
            guardrail_metrics=("modality_missing_rate", "genre_slice_gap", "catalog_coverage_at_k"),
            explainability_family="retrieval",
            optuna=OptunaSupport(
                supported=True,
                objective="validation_cold_item_ndcg_at_10",
                direction="maximize",
                search_space=(
                    _integer("projection_dim", 32, 256, step=32),
                    _categorical("fusion", "concatenate", "gated", "attention"),
                    _float("audio_weight", 0.0, 1.0),
                    _float("modality_dropout", 0.0, 0.5),
                    _float("learning_rate", 1e-5, 3e-3, log=True),
                ),
            ),
            readiness="component_ready",
            blockers=("Acquire approved FMA/Music4All features and join them to dataset-local item identities.",),
            governance_requirements=_RANKING_GOVERNANCE + _PUBLIC_DATA_GOVERNANCE + _MULTIMODAL_GOVERNANCE,
            references=("FMA", "Music4All-Onion"),
        ),
        "long_sequence_state_space": ExpansionModelSpec(
            key="long_sequence_state_space",
            family="ss4rec_mamba4rec",
            stage="sequence_encoder",
            summary="Continuous-time or selective state-space encoders for histories much longer than 30 events.",
            implementation_modules=(),
            required_data=("ordered_item_sequence", "long_histories"),
            optional_data=("irregular_time_gaps", "context_features"),
            primary_metrics=("ndcg_at_10", "recall_at_10", "throughput_examples_per_second"),
            guardrail_metrics=("peak_memory_mb", "performance_by_history_length", "long_tail_recall_at_10"),
            explainability_family="neural_sequence",
            optuna=OptunaSupport(
                supported=True,
                objective="validation_ndcg_at_10",
                direction="maximize",
                search_space=(
                    _categorical("architecture", "ss4rec", "mamba4rec"),
                    _integer("state_dim", 16, 256, step=16),
                    _integer("num_blocks", 1, 8),
                    _integer("sequence_length", 128, 2048, step=128),
                    _float("dropout_rate", 0.0, 0.5),
                ),
                notes="Do not run this study until long-sequence data is materialized.",
            ),
            readiness="research_only",
            blockers=("Increase sequence length to at least 256 and establish compute-matched transformer baselines.",),
            governance_requirements=_RANKING_GOVERNANCE,
            references=("SS4Rec", "Mamba4Rec"),
        ),
        "semantic_id_generative_retrieval": ExpansionModelSpec(
            key="semantic_id_generative_retrieval",
            family="tiger_semantic_ids",
            stage="candidate_generation",
            summary="Generate hierarchical semantic item identifiers from content-aware track representations.",
            implementation_modules=(),
            required_data=("stable_item_content_embeddings", "semantic_id_codebook", "ordered_item_sequences"),
            optional_data=("multimodal_item_features", "public_pretraining"),
            primary_metrics=("recall_at_k", "ndcg_at_k", "codebook_utilization"),
            guardrail_metrics=("invalid_id_rate", "semantic_collision_rate", "catalog_coverage_at_k"),
            explainability_family="retrieval",
            optuna=OptunaSupport(
                supported=True,
                objective="validation_recall_at_100",
                direction="maximize",
                search_space=(
                    _integer("codebook_size", 64, 1024, step=64),
                    _integer("code_depth", 2, 8),
                    _integer("decoder_dim", 64, 512, step=64),
                    _float("commitment_weight", 0.01, 2.0, log=True),
                ),
            ),
            readiness="research_only",
            blockers=("Complete multimodal item embeddings and prove two-stage retrieval baselines first.",),
            governance_requirements=_RANKING_GOVERNANCE + _MULTIMODAL_GOVERNANCE,
            references=("TIGER"),
        ),
        "causal_policy_validation": ExpansionModelSpec(
            key="causal_policy_validation",
            family="off_policy_evaluation",
            stage="evaluation",
            summary="Validate propensity weighting, doubly robust estimators, and policy simulation on randomized datasets.",
            implementation_modules=("spotify.public_training_data", "spotify.safe_policy"),
            required_data=("logged_action", "reward", "propensity"),
            optional_data=("context", "multiple_feedback_signals", "randomized_exposure"),
            primary_metrics=("ips", "snips", "doubly_robust_value"),
            guardrail_metrics=("effective_sample_size", "weight_tail_quantiles", "estimator_variance"),
            explainability_family="multitask",
            optuna=OptunaSupport(
                supported=False,
                objective="not_applicable",
                direction="maximize",
                notes="This lane validates estimators and policies; it should not tune against the evaluation log.",
            ),
            readiness="integration_pending",
            blockers=("Connect canonical KuaiRand/Open Bandit logs to IPS, SNIPS, and doubly robust evaluation.",),
            governance_requirements=_PUBLIC_DATA_GOVERNANCE + (
                "Never report off-policy value without overlap and effective-sample-size diagnostics.",
            ),
            references=("KuaiRand", "Open Bandit Dataset"),
        ),
    }
)


def get_expansion_spec(key: str) -> ExpansionModelSpec:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return EXPANSION_MODEL_REGISTRY[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(EXPANSION_MODEL_REGISTRY))
        raise KeyError(f"Unknown expansion model family '{key}'. Known keys: {known}") from exc


def list_expansion_specs(*, readiness: str | None = None) -> tuple[ExpansionModelSpec, ...]:
    if readiness is not None and readiness not in READINESS_LEVELS:
        raise ValueError(f"readiness must be one of: {', '.join(READINESS_LEVELS)}")
    specs = (
        spec
        for key, spec in sorted(EXPANSION_MODEL_REGISTRY.items())
        if readiness is None or spec.readiness == readiness
    )
    return tuple(specs)


def expansion_registry_as_dict() -> dict[str, dict[str, object]]:
    return {key: spec.to_dict() for key, spec in sorted(EXPANSION_MODEL_REGISTRY.items())}


def validate_expansion_registry() -> tuple[str, ...]:
    """Return registry validation errors instead of failing at import time."""
    errors: list[str] = []
    valid_explainability = {"tree", "dcn", "neural_sequence", "retrieval", "multitask"}
    for key, spec in EXPANSION_MODEL_REGISTRY.items():
        if spec.key != key:
            errors.append(f"{key}: spec.key does not match registry key")
        if spec.readiness not in READINESS_LEVELS:
            errors.append(f"{key}: invalid readiness '{spec.readiness}'")
        if not spec.required_data:
            errors.append(f"{key}: required_data is empty")
        if not spec.primary_metrics:
            errors.append(f"{key}: primary_metrics is empty")
        if spec.explainability_family not in valid_explainability:
            errors.append(f"{key}: unknown explainability family '{spec.explainability_family}'")
        if spec.optuna.supported and not spec.optuna.search_space:
            errors.append(f"{key}: Optuna is supported but search_space is empty")
        if spec.optuna.direction not in {"maximize", "minimize"}:
            errors.append(f"{key}: invalid Optuna direction '{spec.optuna.direction}'")
        parameter_names = [parameter.name for parameter in spec.optuna.search_space]
        if len(parameter_names) != len(set(parameter_names)):
            errors.append(f"{key}: duplicate Optuna parameter names")
        for parameter in spec.optuna.search_space:
            if parameter.kind not in {"int", "float", "categorical"}:
                errors.append(f"{key}.{parameter.name}: invalid search parameter kind")
            if parameter.kind == "categorical" and not parameter.choices:
                errors.append(f"{key}.{parameter.name}: categorical parameter has no choices")
            if parameter.kind != "categorical" and (parameter.low is None or parameter.high is None):
                errors.append(f"{key}.{parameter.name}: numeric parameter is missing bounds")
    return tuple(errors)


__all__ = [
    "EXPANSION_MODEL_REGISTRY",
    "READINESS_LEVELS",
    "ExpansionModelSpec",
    "OptunaSupport",
    "SearchParameter",
    "expansion_registry_as_dict",
    "get_expansion_spec",
    "list_expansion_specs",
    "validate_expansion_registry",
]
