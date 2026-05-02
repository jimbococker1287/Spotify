from __future__ import annotations


def analysis_prefix_for_model_type(model_type: str) -> str | None:
    normalized = str(model_type).strip().lower()
    if normalized == "deep":
        return "deep"
    if normalized in ("classical", "classical_tuned"):
        return "classical"
    if normalized in ("retrieval", "retrieval_reranker", "ensemble"):
        return normalized
    return None
