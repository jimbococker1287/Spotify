from __future__ import annotations

import json

import numpy as np
import pytest

from spotify.expansion_registry import (
    EXPANSION_MODEL_REGISTRY,
    expansion_registry_as_dict,
    get_expansion_spec,
    list_expansion_specs,
    validate_expansion_registry,
)
from spotify.model_explainability import (
    BASE_GOVERNANCE_REQUIREMENTS,
    deterministic_ablation_importance,
    deterministic_permutation_importance,
    list_explainer_capabilities,
    resolve_explainer_capability,
)


def _negative_mse(y_true: np.ndarray, predictions: np.ndarray) -> float:
    return -float(np.mean(np.square(np.asarray(y_true) - np.asarray(predictions))))


def _mse(y_true: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(y_true) - np.asarray(predictions))))


def test_explainer_routes_cover_requested_model_families() -> None:
    assert resolve_explainer_capability("xgboost").strategy == "shap_compatible"
    assert resolve_explainer_capability("DCN-V2").strategy == "shap_compatible"
    assert resolve_explainer_capability("MEANTIME").primary_method == "integrated_gradients"
    assert resolve_explainer_capability("dual encoder").primary_method == "candidate_recall"
    assert resolve_explainer_capability("MMoE").strategy == "per_head"


def test_explainer_capabilities_declare_governance_and_artifact_scope() -> None:
    capabilities = list_explainer_capabilities()

    assert {capability.family for capability in capabilities} == {
        "dcn",
        "multitask",
        "neural_sequence",
        "retrieval",
        "tree",
    }
    assert all(capability.artifact_scope for capability in capabilities)
    assert all(len(capability.governance_requirements) >= len(BASE_GOVERNANCE_REQUIREMENTS) for capability in capabilities)
    assert "conflict_report" in resolve_explainer_capability("ple").artifact_scope


def test_unknown_explainer_family_is_actionable() -> None:
    with pytest.raises(KeyError, match="Unknown explainability family"):
        resolve_explainer_capability("mystery-model")


def test_permutation_importance_is_deterministic_and_ranks_signal_first() -> None:
    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
            [5.0, 10.0],
        ],
        dtype="float64",
    )
    y = X[:, 0].copy()

    first = deterministic_permutation_importance(
        lambda values: values[:, 0],
        X,
        y,
        score_fn=_negative_mse,
        feature_names=("signal", "constant"),
        n_repeats=4,
        random_state=17,
    )
    second = deterministic_permutation_importance(
        lambda values: values[:, 0],
        X,
        y,
        score_fn=_negative_mse,
        feature_names=("signal", "constant"),
        n_repeats=4,
        random_state=17,
    )

    assert first == second
    assert first.importances[0].feature == "signal"
    assert first.importances[0].importance > 0.0
    assert first.importances[1].importance == pytest.approx(0.0)
    assert first.to_dict()["method"] == "deterministic_permutation"


def test_permutation_supports_multitask_output_selection() -> None:
    X = np.arange(12, dtype="float64").reshape(6, 2)
    y = X[:, 0].copy()

    result = deterministic_permutation_importance(
        lambda values: {
            "next_item_output": values[:, 0],
            "skip_output": np.zeros(len(values)),
        },
        X,
        y,
        score_fn=_negative_mse,
        output_selector="next_item_output",
        n_repeats=2,
    )

    assert result.importances[0].feature == "feature_0"


def test_ablation_importance_supports_lower_is_better_metrics() -> None:
    X = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    y = X[:, 0].copy()

    result = deterministic_ablation_importance(
        lambda values: values[:, 0],
        X,
        y,
        score_fn=_mse,
        feature_names=("signal", "constant"),
        baseline="zero",
        greater_is_better=False,
    )

    assert result.baseline_score == pytest.approx(0.0)
    assert result.importances[0].feature == "signal"
    assert result.importances[0].importance > 0.0
    assert result.random_state is None


def test_ablation_supports_feature_axis_and_vector_baselines() -> None:
    X = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[2.0, 10.0], [3.0, 20.0]],
            [[3.0, 10.0], [4.0, 20.0]],
        ]
    )
    y = X[:, :, 0].sum(axis=1)

    result = deterministic_ablation_importance(
        lambda values: values[:, :, 0].sum(axis=1),
        X,
        y,
        score_fn=_negative_mse,
        feature_names=("signal", "constant"),
        feature_axis=2,
        baseline=np.array([0.0, 99.0]),
    )

    assert result.importances[0].feature == "signal"
    assert result.importances[1].importance == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"feature_axis": 0}, "non-batch axis"),
        ({"feature_names": ("only_one",)}, "feature axis has width"),
        ({"n_repeats": 0}, "positive integer"),
    ],
)
def test_permutation_validation_is_actionable(kwargs: dict[str, object], message: str) -> None:
    X = np.ones((4, 2))
    y = np.ones(4)

    with pytest.raises(ValueError, match=message):
        deterministic_permutation_importance(
            lambda values: values[:, 0],
            X,
            y,
            score_fn=_negative_mse,
            **kwargs,
        )


def test_expansion_registry_is_complete_and_internally_valid() -> None:
    assert validate_expansion_registry() == ()
    assert len(EXPANSION_MODEL_REGISTRY) >= 10
    assert {
        "track_level_retrieval",
        "dcn_v2_reranker",
        "multitask_mmoe_ple",
        "meantime_tisasrec",
        "multimodal_content_tower",
        "causal_policy_validation",
    }.issubset(EXPANSION_MODEL_REGISTRY)
    assert all(spec.required_data for spec in EXPANSION_MODEL_REGISTRY.values())
    assert all(spec.primary_metrics for spec in EXPANSION_MODEL_REGISTRY.values())


def test_registry_optuna_metadata_is_specific_and_filterable() -> None:
    dcn = get_expansion_spec("DCN-V2-Reranker")
    component_ready = list_expansion_specs(readiness="component_ready")

    assert dcn.optuna.supported
    assert dcn.optuna.objective == "validation_ndcg_at_10"
    assert {parameter.name for parameter in dcn.optuna.search_space} >= {
        "cross_layers",
        "architecture",
        "dropout_rate",
    }
    assert dcn in component_ready
    assert all(spec.readiness == "component_ready" for spec in component_ready)


def test_registry_public_and_multimodal_lanes_declare_data_governance() -> None:
    public = get_expansion_spec("public_collaborative_transfer")
    multimodal = get_expansion_spec("multimodal_content_tower")

    assert any("license" in requirement.lower() for requirement in public.governance_requirements)
    assert any("Spotify Platform" in requirement for requirement in public.governance_requirements)
    assert any("audio" in requirement.lower() for requirement in multimodal.governance_requirements)
    assert public.references == ("LFM-1b", "Million Song Taste Profile")


def test_registry_payload_is_json_serializable() -> None:
    payload = expansion_registry_as_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert "track_level_retrieval" in serialized
    assert payload["causal_policy_validation"]["optuna"]["supported"] is False


def test_registry_rejects_unknown_filters_and_keys() -> None:
    with pytest.raises(ValueError, match="readiness must be one of"):
        list_expansion_specs(readiness="done")
    with pytest.raises(KeyError, match="Unknown expansion model family"):
        get_expansion_spec("unknown")
