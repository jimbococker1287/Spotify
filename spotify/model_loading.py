from __future__ import annotations


def keras_custom_objects_for_model(model_name: str) -> dict[str, object]:
    if str(model_name).strip().lower() == "srgnn":
        from .srgnn_model import get_srgnn_custom_objects

        return get_srgnn_custom_objects()
    return {}


__all__ = ["keras_custom_objects_for_model"]
