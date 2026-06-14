from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


ScoreFunction = Callable[[np.ndarray, Any], float]
PredictFunction = Callable[[np.ndarray], Any]
OutputSelector = str | int | Callable[[Any], Any] | None


BASE_GOVERNANCE_REQUIREMENTS = (
    "Record the model version, data fingerprint, split, random seed, and explainer configuration.",
    "Compute explanations on leakage-safe temporal holdouts rather than training rows alone.",
    "Report both aggregate and representative local explanations with the evaluated metric.",
    "Run stability checks across seeds or resamples before using explanations for promotion decisions.",
    "Treat explanations as diagnostic evidence, not proof of causality.",
    "Exclude or redact direct identifiers and sensitive raw event payloads from explanation artifacts.",
)


@dataclass(frozen=True)
class ExplainabilityCapability:
    """Declarative explainability contract for a model family."""

    family: str
    strategy: str
    primary_method: str
    fallback_methods: tuple[str, ...]
    artifact_scope: tuple[str, ...]
    optional_dependencies: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    governance_requirements: tuple[str, ...] = BASE_GOVERNANCE_REQUIREMENTS
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureImportance:
    """Metric change caused by perturbing one feature."""

    feature: str
    importance: float
    score_after: float
    standard_deviation: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationImportanceResult:
    """Serializable output shared by permutation and ablation utilities."""

    method: str
    baseline_score: float
    greater_is_better: bool
    random_state: int | None
    repeats: int
    importances: tuple[FeatureImportance, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "baseline_score": self.baseline_score,
            "greater_is_better": self.greater_is_better,
            "random_state": self.random_state,
            "repeats": self.repeats,
            "importances": [item.to_dict() for item in self.importances],
        }


_TREE_GOVERNANCE = BASE_GOVERNANCE_REQUIREMENTS + (
    "Preserve the background-sample selection rule for SHAP-compatible comparisons.",
    "Compare SHAP-compatible output with native or permutation importance when practical.",
)
_NEURAL_GOVERNANCE = BASE_GOVERNANCE_REQUIREMENTS + (
    "Declare the attribution baseline and target class or score.",
    "Pair gradient attribution with deterministic ablation to test faithfulness.",
)
_RETRIEVAL_GOVERNANCE = BASE_GOVERNANCE_REQUIREMENTS + (
    "Separate candidate-generation recall from reranker quality.",
    "Report neighborhood examples, popularity slices, and long-tail candidate coverage.",
)
_MULTITASK_GOVERNANCE = BASE_GOVERNANCE_REQUIREMENTS + (
    "Generate explanations per task head and record each head's loss weight.",
    "Do not aggregate task explanations until conflicting head-level effects have been reviewed.",
)


EXPLAINER_REGISTRY: Mapping[str, ExplainabilityCapability] = MappingProxyType(
    {
        "tree": ExplainabilityCapability(
            family="tree",
            strategy="shap_compatible",
            primary_method="tree_shap",
            fallback_methods=("native_feature_importance", "deterministic_permutation"),
            artifact_scope=("global", "local", "slice"),
            optional_dependencies=("shap",),
            required_inputs=("feature_matrix", "feature_names", "prediction_target"),
            governance_requirements=_TREE_GOVERNANCE,
            notes="Use native TreeSHAP where supported; the deterministic permutation utility is dependency-free.",
        ),
        "dcn": ExplainabilityCapability(
            family="dcn",
            strategy="shap_compatible",
            primary_method="gradient_shap",
            fallback_methods=("integrated_gradients", "deterministic_ablation"),
            artifact_scope=("global", "local", "feature_cross"),
            optional_dependencies=("shap", "tensorflow"),
            required_inputs=("context_features", "item_features", "ranking_score"),
            governance_requirements=_TREE_GOVERNANCE + _NEURAL_GOVERNANCE[-2:],
            notes="Explain the final ranking score and retain context/item feature namespaces.",
        ),
        "neural_sequence": ExplainabilityCapability(
            family="neural_sequence",
            strategy="gradient_and_ablation",
            primary_method="integrated_gradients",
            fallback_methods=("deterministic_ablation", "occlusion"),
            artifact_scope=("local", "position", "context", "global_aggregate"),
            optional_dependencies=("tensorflow",),
            required_inputs=("sequence", "attribution_baseline", "target_item_or_score"),
            governance_requirements=_NEURAL_GOVERNANCE,
            notes="Attention weights may be logged as diagnostics but are not the primary explanation.",
        ),
        "retrieval": ExplainabilityCapability(
            family="retrieval",
            strategy="candidate_diagnostics",
            primary_method="candidate_recall",
            fallback_methods=("embedding_neighborhood", "leave_one_event_out_ablation"),
            artifact_scope=("candidate_set", "neighborhood", "slice", "global"),
            required_inputs=("query_embedding", "candidate_ids", "candidate_scores", "ground_truth_item"),
            governance_requirements=_RETRIEVAL_GOVERNANCE,
            notes="Explain why an item entered the candidate set before explaining its final rank.",
        ),
        "multitask": ExplainabilityCapability(
            family="multitask",
            strategy="per_head",
            primary_method="per_head_integrated_gradients",
            fallback_methods=("per_head_deterministic_ablation", "gate_weight_summary"),
            artifact_scope=("task_head", "local", "global_aggregate", "conflict_report"),
            optional_dependencies=("tensorflow",),
            required_inputs=("task_head", "shared_inputs", "head_prediction", "loss_weights"),
            governance_requirements=_MULTITASK_GOVERNANCE,
            notes="Next-item, skip, dwell, session-end, explicit-positive, and repeat heads remain separate.",
        ),
    }
)


MODEL_FAMILY_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "tree": "tree",
        "classical_tree": "tree",
        "random_forest": "tree",
        "extra_trees": "tree",
        "hist_gradient_boosting": "tree",
        "xgboost": "tree",
        "lightgbm": "tree",
        "catboost": "tree",
        "dcn": "dcn",
        "dcn_v2": "dcn",
        "dcn-v2": "dcn",
        "neural_sequence": "neural_sequence",
        "sequence": "neural_sequence",
        "sasrec": "neural_sequence",
        "bert4rec": "neural_sequence",
        "meantime": "neural_sequence",
        "tisasrec": "neural_sequence",
        "ss4rec": "neural_sequence",
        "mamba4rec": "neural_sequence",
        "gru": "neural_sequence",
        "lstm": "neural_sequence",
        "transformer": "neural_sequence",
        "tcn": "neural_sequence",
        "retrieval": "retrieval",
        "dual_encoder": "retrieval",
        "two_tower": "retrieval",
        "lightgcn": "retrieval",
        "ease": "retrieval",
        "implicit_als": "retrieval",
        "bpr": "retrieval",
        "multitask": "multitask",
        "mmoe": "multitask",
        "ple": "multitask",
    }
)


def resolve_explainer_capability(model_family: str) -> ExplainabilityCapability:
    """Resolve a model name or family to its declared explanation strategy."""
    normalized = str(model_family).strip().lower().replace(" ", "_")
    canonical = MODEL_FAMILY_ALIASES.get(normalized, normalized)
    try:
        return EXPLAINER_REGISTRY[canonical]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_FAMILY_ALIASES))
        raise KeyError(f"Unknown explainability family '{model_family}'. Known families and aliases: {known}") from exc


def list_explainer_capabilities() -> tuple[ExplainabilityCapability, ...]:
    """Return capabilities in stable family order."""
    return tuple(EXPLAINER_REGISTRY[key] for key in sorted(EXPLAINER_REGISTRY))


def _validated_inputs(
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Sequence[str] | None,
    feature_axis: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    values = np.asarray(X)
    targets = np.asarray(y_true)
    if values.ndim < 2:
        raise ValueError("X must include a batch axis and at least one feature axis.")
    if len(values) == 0:
        raise ValueError("X must contain at least one row.")
    if len(targets) != len(values):
        raise ValueError("X and y_true must contain the same number of rows.")

    axis = int(feature_axis)
    if axis < 0:
        axis += values.ndim
    if axis <= 0 or axis >= values.ndim:
        raise ValueError("feature_axis must identify a non-batch axis.")

    width = int(values.shape[axis])
    if feature_names is None:
        names = tuple(f"feature_{index}" for index in range(width))
    else:
        names = tuple(str(name) for name in feature_names)
        if len(names) != width:
            raise ValueError(f"feature_names has {len(names)} entries but feature axis has width {width}.")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique.")
    return values, targets, names, axis


def _select_output(predictions: Any, selector: OutputSelector) -> Any:
    if selector is None:
        return predictions
    if callable(selector):
        return selector(predictions)
    if isinstance(selector, str):
        if not isinstance(predictions, Mapping):
            raise TypeError("A string output_selector requires mapping predictions.")
        return predictions[selector]
    if isinstance(selector, int):
        if not isinstance(predictions, (list, tuple)):
            raise TypeError("An integer output_selector requires list or tuple predictions.")
        return predictions[selector]
    raise TypeError("output_selector must be None, a key, an index, or a callable.")


def _score(
    predict_fn: PredictFunction,
    score_fn: ScoreFunction,
    values: np.ndarray,
    targets: np.ndarray,
    selector: OutputSelector,
) -> float:
    predictions = _select_output(predict_fn(values), selector)
    score = float(score_fn(targets, predictions))
    if not np.isfinite(score):
        raise ValueError("score_fn must return a finite score.")
    return score


def _importance_drop(baseline: float, perturbed: float, greater_is_better: bool) -> float:
    return baseline - perturbed if greater_is_better else perturbed - baseline


def _feature_index(ndim: int, axis: int, feature_index: int) -> tuple[object, ...]:
    index: list[object] = [slice(None)] * ndim
    index[axis] = feature_index
    return tuple(index)


def deterministic_permutation_importance(
    predict_fn: PredictFunction,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    score_fn: ScoreFunction,
    feature_names: Sequence[str] | None = None,
    feature_axis: int = -1,
    n_repeats: int = 5,
    random_state: int = 42,
    greater_is_better: bool = True,
    output_selector: OutputSelector = None,
) -> PerturbationImportanceResult:
    """Compute repeatable metric-drop importance without importing SHAP."""
    if isinstance(n_repeats, bool) or int(n_repeats) < 1:
        raise ValueError("n_repeats must be a positive integer.")
    repeats = int(n_repeats)
    values, targets, names, axis = _validated_inputs(X, y_true, feature_names, feature_axis)
    baseline_score = _score(predict_fn, score_fn, values, targets, output_selector)
    random_generator = np.random.default_rng(int(random_state))
    rows: list[FeatureImportance] = []

    for feature_index, feature_name in enumerate(names):
        index = _feature_index(values.ndim, axis, feature_index)
        scores: list[float] = []
        for _ in range(repeats):
            permuted = np.array(values, copy=True)
            row_order = random_generator.permutation(len(values))
            permuted[index] = values[index][row_order]
            scores.append(_score(predict_fn, score_fn, permuted, targets, output_selector))
        score_after = float(np.mean(scores))
        rows.append(
            FeatureImportance(
                feature=feature_name,
                importance=float(_importance_drop(baseline_score, score_after, greater_is_better)),
                score_after=score_after,
                standard_deviation=float(np.std(scores)),
            )
        )

    rows.sort(key=lambda item: (-item.importance, item.feature))
    return PerturbationImportanceResult(
        method="deterministic_permutation",
        baseline_score=baseline_score,
        greater_is_better=bool(greater_is_better),
        random_state=int(random_state),
        repeats=repeats,
        importances=tuple(rows),
    )


def deterministic_ablation_importance(
    predict_fn: PredictFunction,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    score_fn: ScoreFunction,
    feature_names: Sequence[str] | None = None,
    feature_axis: int = -1,
    baseline: str | float | np.ndarray = "mean",
    greater_is_better: bool = True,
    output_selector: OutputSelector = None,
) -> PerturbationImportanceResult:
    """Compute deterministic leave-one-feature-out metric changes."""
    values, targets, names, axis = _validated_inputs(X, y_true, feature_names, feature_axis)
    baseline_score = _score(predict_fn, score_fn, values, targets, output_selector)
    rows: list[FeatureImportance] = []

    if isinstance(baseline, str) and baseline not in {"mean", "median", "zero"}:
        raise ValueError("baseline must be 'mean', 'median', 'zero', a scalar, or an array.")

    for feature_index, feature_name in enumerate(names):
        index = _feature_index(values.ndim, axis, feature_index)
        selected = values[index]
        if isinstance(baseline, str):
            if baseline == "mean":
                replacement: object = np.mean(selected, axis=0)
            elif baseline == "median":
                replacement = np.median(selected, axis=0)
            else:
                replacement = 0
        else:
            baseline_values = np.asarray(baseline)
            replacement = (
                baseline_values[feature_index]
                if baseline_values.ndim == 1 and len(baseline_values) == len(names)
                else baseline_values
            )

        ablated = np.array(values, copy=True)
        try:
            ablated[index] = replacement
        except (TypeError, ValueError) as exc:
            raise ValueError(f"baseline cannot be broadcast to feature '{feature_name}'.") from exc
        score_after = _score(predict_fn, score_fn, ablated, targets, output_selector)
        rows.append(
            FeatureImportance(
                feature=feature_name,
                importance=float(_importance_drop(baseline_score, score_after, greater_is_better)),
                score_after=score_after,
            )
        )

    rows.sort(key=lambda item: (-item.importance, item.feature))
    return PerturbationImportanceResult(
        method="deterministic_ablation",
        baseline_score=baseline_score,
        greater_is_better=bool(greater_is_better),
        random_state=None,
        repeats=1,
        importances=tuple(rows),
    )


__all__ = [
    "BASE_GOVERNANCE_REQUIREMENTS",
    "EXPLAINER_REGISTRY",
    "MODEL_FAMILY_ALIASES",
    "ExplainabilityCapability",
    "FeatureImportance",
    "PerturbationImportanceResult",
    "deterministic_ablation_importance",
    "deterministic_permutation_importance",
    "list_explainer_capabilities",
    "resolve_explainer_capability",
]
