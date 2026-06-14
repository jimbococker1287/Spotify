from __future__ import annotations

import json
from pathlib import Path
import zipfile


def _bert4rec_mask_token_from_artifact(model_path: str | Path | None) -> int | None:
    if model_path is None:
        return None
    try:
        with zipfile.ZipFile(Path(model_path)) as archive:
            payload = json.loads(archive.read("config.json"))
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        return None

    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            config = value.get("config")
            if value.get("class_name") == "Embedding" and isinstance(config, dict):
                if config.get("name") == "item_embedding":
                    return int(config["input_dim"]) - 1
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return None


def _model_dimensions_from_artifact(
    model_path: str | Path,
) -> tuple[int, int, int]:
    with zipfile.ZipFile(Path(model_path)) as archive:
        payload = json.loads(archive.read("config.json"))

    sequence_length = 0
    num_ctx = 0
    num_artists = 0
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            config = value.get("config")
            if isinstance(config, dict):
                name = config.get("name")
                batch_shape = config.get("batch_shape")
                if value.get("class_name") == "InputLayer" and isinstance(batch_shape, list):
                    if name == "seq_input":
                        sequence_length = int(batch_shape[-1])
                    elif name == "ctx_input":
                        num_ctx = int(batch_shape[-1])
                if value.get("class_name") == "Dense" and name == "artist_output":
                    num_artists = int(config["units"])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if min(sequence_length, num_ctx, num_artists) <= 0:
        raise ValueError(f"Could not recover model dimensions from {model_path}")
    return sequence_length, num_artists, num_ctx


def _bert4rec_params_from_artifact(model_path: str | Path) -> dict[str, object]:
    with zipfile.ZipFile(Path(model_path)) as archive:
        payload = json.loads(archive.read("config.json"))

    params: dict[str, object] = {}
    block_indices: set[int] = set()
    feed_forward_units = 0
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            config = value.get("config")
            if isinstance(config, dict):
                name = str(config.get("name", ""))
                if name == "item_embedding":
                    params["embedding_dim"] = int(config["output_dim"])
                elif name == "embedding_dropout":
                    params["dropout_rate"] = float(config["rate"])
                elif name.startswith("bidirectional_attention_"):
                    params["num_heads"] = int(config["num_heads"])
                    block_indices.add(int(name.rsplit("_", 1)[-1]))
                elif name == "feed_forward_expand_1":
                    feed_forward_units = int(config["units"])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if block_indices:
        params["num_blocks"] = max(block_indices)
    if feed_forward_units and "embedding_dim" in params:
        params["feed_forward_multiplier"] = (
            feed_forward_units // int(params["embedding_dim"])
        )
    return params


def _sasrec_params_from_artifact(model_path: str | Path) -> dict[str, object]:
    with zipfile.ZipFile(Path(model_path)) as archive:
        payload = json.loads(archive.read("config.json"))

    params: dict[str, object] = {}
    block_indices: set[int] = set()
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            config = value.get("config")
            if isinstance(config, dict):
                name = str(config.get("name", ""))
                if name == "item_embedding":
                    params["embedding_dim"] = int(config["output_dim"])
                elif name == "embedding_dropout":
                    params["dropout_rate"] = float(config["rate"])
                elif name.startswith("causal_self_attention_"):
                    params["num_heads"] = int(config["num_heads"])
                    block_indices.add(int(name.rsplit("_", 1)[-1]))
                elif name == "feed_forward_expand_1":
                    params["feed_forward_dim"] = int(config["units"])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if block_indices:
        params["num_blocks"] = max(block_indices)
    return params


def _artifact_contains_lambda(model_path: str | Path) -> bool:
    try:
        with zipfile.ZipFile(Path(model_path)) as archive:
            payload = json.loads(archive.read("config.json"))
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        return False
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get("class_name") == "Lambda":
                return True
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return False


def _rebuild_legacy_model_and_load_weights(
    model_path: str | Path,
    *,
    model_name: str,
):
    from .modeling import build_model_builders

    sequence_length, num_artists, num_ctx = _model_dimensions_from_artifact(model_path)
    model_params_by_name = None
    if model_name == "bert4rec":
        model_params_by_name = {
            "bert4rec": _bert4rec_params_from_artifact(model_path),
        }
    elif model_name == "sasrec":
        model_params_by_name = {
            "sasrec": _sasrec_params_from_artifact(model_path),
        }
    builders = dict(
        build_model_builders(
            sequence_length=sequence_length,
            num_artists=num_artists,
            num_ctx=num_ctx,
            selected_names=(model_name,),
            model_params_by_name=model_params_by_name,
        )
    )
    model = builders[model_name]()
    model.load_weights(model_path)
    return model


def keras_custom_objects_for_model(
    model_name: str,
    model_path: str | Path | None = None,
) -> dict[str, object]:
    normalized = str(model_name).strip().lower()
    objects: dict[str, object] = {}
    if normalized == "srgnn":
        from .srgnn_model import get_srgnn_custom_objects

        objects.update(get_srgnn_custom_objects())
    if normalized == "sasrec":
        from .sasrec_model import get_sasrec_custom_objects

        objects.update(get_sasrec_custom_objects())
    if normalized == "bert4rec":
        from .bert4rec_model import get_bert4rec_custom_objects

        objects.update(
            get_bert4rec_custom_objects(
                mask_token_id=_bert4rec_mask_token_from_artifact(model_path),
            )
        )
    if normalized in {"attention_rnn", "memory_net", "memory_net_artist"}:
        from .modeling import get_modeling_custom_objects

        objects.update(get_modeling_custom_objects())
    return objects


def load_trusted_keras_model(
    model_path: str | Path,
    *,
    model_name: str,
    compile: bool = False,
):
    """Load a checkpoint created by this project, including local Lambda layers."""
    import tensorflow as tf

    kwargs = {
        "compile": compile,
        "custom_objects": keras_custom_objects_for_model(model_name, model_path),
    }
    normalized = str(model_name).strip().lower()
    if normalized in {"bert4rec", "sasrec"} and _artifact_contains_lambda(model_path):
        return _rebuild_legacy_model_and_load_weights(
            model_path,
            model_name=normalized,
        )
    try:
        try:
            return tf.keras.models.load_model(model_path, safe_mode=False, **kwargs)
        except TypeError as exc:
            if "safe_mode" not in str(exc):
                raise
            return tf.keras.models.load_model(model_path, **kwargs)
    except (NotImplementedError, TypeError):
        if normalized not in {"memory_net", "memory_net_artist"}:
            raise
        return _rebuild_legacy_model_and_load_weights(
            model_path,
            model_name=normalized,
        )


__all__ = ["keras_custom_objects_for_model", "load_trusted_keras_model"]
