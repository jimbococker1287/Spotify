from __future__ import annotations

import importlib
import importlib.util
import sys

import numpy as np
import pytest

from spotify.multimodal_model import (
    build_multimodal_cold_start_scorer,
    build_multimodal_track_encoder,
    fuse_track_representations,
    get_multimodal_custom_objects,
    infer_modality_mask,
    score_cold_start_tracks,
)


TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def test_tensorflow_import_is_lazy() -> None:
    sys.modules.pop("spotify.multimodal_model", None)
    module = importlib.import_module("spotify.multimodal_model")

    assert "tensorflow" not in module.__dict__


def test_infer_modality_mask_handles_absent_and_nonfinite_rows() -> None:
    audio = np.asarray([[1.0, 2.0], [np.nan, np.nan]], dtype=np.float32)
    metadata = np.asarray([[0.5], [1.0]], dtype=np.float32)

    mask = infer_modality_mask(audio, metadata, None)

    np.testing.assert_array_equal(
        mask,
        np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )


def test_numpy_fusion_projects_masks_and_normalizes_modalities() -> None:
    audio = np.asarray([[1.0, 0.0], [np.nan, np.nan]], dtype=np.float32)
    metadata = np.asarray([[1.0], [2.0]], dtype=np.float32)
    collaborative = np.asarray([[0.0, 1.0], [10.0, 10.0]], dtype=np.float32)

    fused = fuse_track_representations(
        audio,
        metadata,
        collaborative,
        projection_matrices={
            "metadata": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
        modality_mask=np.asarray(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        modality_weights={"audio": 2.0, "metadata": 1.0, "collaborative": 1.0},
    )

    assert fused.shape == (2, 2)
    np.testing.assert_allclose(
        np.linalg.norm(fused, axis=1),
        np.ones(2),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        fused[0],
        np.asarray([3.0, 1.0], dtype=np.float32) / np.sqrt(10.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(fused[1], np.asarray([1.0, 0.0]), atol=1e-6)


def test_numpy_fusion_returns_zero_for_fully_missing_track() -> None:
    fused = fuse_track_representations(
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        modality_mask=np.zeros((1, 3), dtype=np.float32),
    )

    np.testing.assert_array_equal(fused, np.zeros((1, 2), dtype=np.float32))


def test_numpy_fusion_validates_projection_and_output_dimensions() -> None:
    audio = np.ones((2, 3), dtype=np.float32)
    metadata = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="same output dimension"):
        fuse_track_representations(audio, metadata)
    with pytest.raises(ValueError, match="projection input dimension"):
        fuse_track_representations(
            audio,
            metadata,
            projection_matrices={"metadata": np.ones((3, 3), dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="equal rows"):
        fuse_track_representations(audio, np.ones((3, 2), dtype=np.float32))


def test_cold_start_scores_content_candidates_and_penalizes_missing_content() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    candidates = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    mask = np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    scores = score_cold_start_tracks(
        query,
        candidates,
        candidate_modality_mask=mask,
        missing_modality_penalty=0.2,
    )

    np.testing.assert_allclose(scores[:2], np.asarray([1.0, 0.9]), atol=1e-6)
    assert np.isneginf(scores[2])


def test_builder_validates_before_tensorflow_import() -> None:
    with pytest.raises(ValueError, match="audio_dim"):
        build_multimodal_track_encoder(0, 3, 4)
    with pytest.raises(ValueError, match="unknown multimodal parameters"):
        build_multimodal_track_encoder(2, 3, 4, params={"mystery": True})
    with pytest.raises(ValueError, match="modality_weights"):
        build_multimodal_track_encoder(
            2,
            3,
            4,
            params={"modality_weights": (0.0, 0.0, 0.0)},
        )


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow is not installed")
def test_keras_encoder_shapes_normalization_and_masking() -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(29)
    model = build_multimodal_track_encoder(
        audio_dim=3,
        metadata_dim=2,
        collaborative_dim=4,
        embedding_dim=5,
        params={"dropout_rate": 0.0},
    )
    audio = np.asarray([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32)
    metadata = np.asarray([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32)
    collaborative = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [-100.0, 80.0, 20.0, 7.0]],
        dtype=np.float32,
    )
    mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

    encoded = model(
        [audio, metadata, collaborative, mask],
        training=False,
    ).numpy()

    assert model.input_shape == [
        (None, 3),
        (None, 2),
        (None, 4),
        (None, 3),
    ]
    assert model.output_shape == (None, 5)
    np.testing.assert_allclose(encoded[0], encoded[1], atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(encoded, axis=1),
        np.ones(2),
        atol=1e-6,
    )


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow is not installed")
def test_projection_trainability_is_configurable() -> None:
    model = build_multimodal_track_encoder(
        3,
        2,
        4,
        embedding_dim=5,
        params={
            "audio_projection_trainable": False,
            "metadata_projection_trainable": True,
            "collaborative_projection_trainable": False,
        },
    )
    fusion = model.get_layer("masked_multimodal_fusion")

    assert fusion.projections[0].trainable is False
    assert fusion.projections[1].trainable is True
    assert fusion.projections[2].trainable is False


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow is not installed")
def test_keras_cold_start_scorer_returns_pairwise_scores() -> None:
    model = build_multimodal_cold_start_scorer(
        audio_dim=2,
        metadata_dim=2,
        collaborative_dim=2,
        embedding_dim=3,
        params={"dropout_rate": 0.0},
        missing_modality_penalty=0.3,
    )
    query = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    audio = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    metadata = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    collaborative = np.zeros((2, 2), dtype=np.float32)
    mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    scores = model(
        [query, audio, metadata, collaborative, mask],
        training=False,
    ).numpy()

    assert scores.shape == (2, 1)
    assert np.isfinite(scores).all()


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow is not installed")
def test_multimodal_encoder_round_trips_through_keras(tmp_path) -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(31)
    model = build_multimodal_track_encoder(
        3,
        2,
        4,
        embedding_dim=6,
        params={
            "activation": "tanh",
            "dropout_rate": 0.0,
            "modality_weights": (2.0, 1.0, 0.5),
            "audio_projection_trainable": False,
        },
    )
    inputs = [
        np.asarray([[1.0, 0.5, -0.25]], dtype=np.float32),
        np.asarray([[0.2, 0.8]], dtype=np.float32),
        np.asarray([[0.5, 0.1, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
    ]
    expected = model(inputs, training=False).numpy()
    path = tmp_path / "multimodal_encoder.keras"
    model.save(path)

    restored = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects=get_multimodal_custom_objects(),
    )
    actual = restored(inputs, training=False).numpy()
    fusion = restored.get_layer("masked_multimodal_fusion")

    assert restored.name == "multimodal_track_encoder"
    assert fusion.output_dim == 6
    assert fusion.projections[0].trainable is False
    assert fusion.modality_weights == (2.0, 1.0, 0.5)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
