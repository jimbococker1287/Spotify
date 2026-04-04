from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib

from .probability_bundles import save_prediction_bundle
from .retrieval_artifact_payloads import build_retrieval_result_rows, build_retrieval_summary_payload
from .retrieval_runtime_eval import RetrievalBaselineEvaluation, RetrievalRerankerEvaluation


@dataclass
class RetrievalPersistedArtifacts:
    retrieval_bundle_path: Path
    retrieval_model_path: Path
    reranker_path: Path
    reranker_bundle_path: Path
    reranker_model_path: Path
    summary_path: Path


def persist_retrieval_outputs(
    *,
    artifact_paths: list[Path],
    baseline: RetrievalBaselineEvaluation,
    enable_self_supervised_pretraining: bool,
    fit_seconds: float,
    logger,
    objective_rows: list[dict[str, object]],
    output_dir: Path,
    pretrain_path: Path,
    pretrain_result,
    reranker: RetrievalRerankerEvaluation,
    retrieval_epochs: int,
    top_k: int,
) -> tuple[RetrievalPersistedArtifacts, list[dict[str, object]]]:
    prediction_dir = output_dir / "prediction_bundles"
    retrieval_dir = output_dir / "retrieval"

    retrieval_bundle_path = save_prediction_bundle(
        prediction_dir / "retrieval_dual_encoder.npz",
        val_proba=baseline.val_retrieval_scores,
        test_proba=baseline.test_retrieval_scores,
    )
    retrieval_model_path = retrieval_dir / "retrieval_dual_encoder.joblib"
    joblib.dump(baseline.retrieval_artifact, retrieval_model_path, compress=3)
    artifact_paths.extend([retrieval_bundle_path, retrieval_model_path])

    logger.info(
        "Retrieval baseline: val_top1=%.4f val_recall@%d=%.4f test_top1=%.4f",
        baseline.retrieval_metrics_val["top1"],
        top_k,
        baseline.val_candidate_hit,
        baseline.retrieval_metrics_test["top1"],
    )

    reranker_path = retrieval_dir / "retrieval_reranker_estimator.joblib"
    joblib.dump(reranker.reranker_estimator, reranker_path, compress=3)
    artifact_paths.append(reranker_path)

    reranker_bundle_path = save_prediction_bundle(
        prediction_dir / "retrieval_reranker.npz",
        val_proba=reranker.val_rerank_proba,
        test_proba=reranker.test_rerank_proba,
    )
    reranker_model_path = retrieval_dir / "retrieval_reranker.joblib"
    joblib.dump(reranker.reranker_artifact, reranker_model_path, compress=3)
    artifact_paths.extend([reranker_bundle_path, reranker_model_path])

    summary_path = retrieval_dir / "retrieval_summary.json"
    summary_path.write_text(
        json.dumps(
            build_retrieval_summary_payload(
                top_k=top_k,
                embedding_dim=int(baseline.retrieval_artifact.artist_embeddings.shape[1]),
                enable_self_supervised_pretraining=enable_self_supervised_pretraining,
                pretrain_result=pretrain_result,
                retrieval_epochs=retrieval_epochs,
                val_candidate_hit=baseline.val_candidate_hit,
                test_candidate_hit=baseline.test_candidate_hit,
                objective_rows=objective_rows,
                ann_metrics_val=baseline.ann_metrics_val,
                ann_metrics_test=baseline.ann_metrics_test,
                retrieval_metrics_val=baseline.retrieval_metrics_val,
                retrieval_metrics_test=baseline.retrieval_metrics_test,
                rerank_metrics_val=reranker.rerank_metrics_val,
                rerank_metrics_test=reranker.rerank_metrics_test,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_paths.append(summary_path)

    rows = build_retrieval_result_rows(
        top_k=top_k,
        fit_seconds=fit_seconds,
        retrieval_epochs=retrieval_epochs,
        pretrain_result=pretrain_result,
        pretrain_path=pretrain_path,
        retrieval_bundle_path=retrieval_bundle_path,
        retrieval_model_path=retrieval_model_path,
        reranker_bundle_path=reranker_bundle_path,
        reranker_model_path=reranker_model_path,
        reranker_path=reranker_path,
        retrieval_metrics_val=baseline.retrieval_metrics_val,
        retrieval_metrics_test=baseline.retrieval_metrics_test,
        rerank_metrics_val=reranker.rerank_metrics_val,
        rerank_metrics_test=reranker.rerank_metrics_test,
        val_candidate_hit=baseline.val_candidate_hit,
        test_candidate_hit=baseline.test_candidate_hit,
        ann_metrics_val=baseline.ann_metrics_val,
        ann_metrics_test=baseline.ann_metrics_test,
    )

    return (
        RetrievalPersistedArtifacts(
            retrieval_bundle_path=retrieval_bundle_path,
            retrieval_model_path=retrieval_model_path,
            reranker_path=reranker_path,
            reranker_bundle_path=reranker_bundle_path,
            reranker_model_path=reranker_model_path,
            summary_path=summary_path,
        ),
        rows,
    )


__all__ = ["RetrievalPersistedArtifacts", "persist_retrieval_outputs"]
