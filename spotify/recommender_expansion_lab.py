from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data import load_streaming_history
from .expansion_registry import list_expansion_specs, validate_expansion_registry
from .model_explainability import resolve_explainer_capability
from .run_artifacts import safe_read_json, write_csv_rows, write_json, write_markdown
from .track_level_data import (
    TrackLevelDataset,
    TrackLevelExample,
    build_track_level_dataset,
    split_track_level_examples,
)
from .track_retrieval import PopularityRetriever, candidate_diagnostics


@dataclass(frozen=True)
class ExpansionRunConfig:
    raw_data_dir: Path
    output_dir: Path
    max_history: int = 256
    min_history: int = 1
    session_gap_minutes: float = 30.0
    validation_fraction: float = 0.16
    test_fraction: float = 0.20
    evaluation_k: int = 100
    evaluation_limit: int = 20_000
    include_video: bool = True


def _examples_to_interactions(
    examples: Sequence[TrackLevelExample],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": [example.session_id for example in examples],
            "track_id": [example.target_track_uri for example in examples],
        }
    )


def _bounded_examples(
    examples: Sequence[TrackLevelExample],
    limit: int,
) -> tuple[TrackLevelExample, ...]:
    values = tuple(examples)
    if limit <= 0 or len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=int(limit), dtype="int64")
    return tuple(values[int(index)] for index in indices)


def _evaluate_popularity(
    train: Sequence[TrackLevelExample],
    validation: Sequence[TrackLevelExample],
    *,
    k: int,
    limit: int,
) -> dict[str, object]:
    if not train or not validation:
        return {
            "status": "not_run",
            "reason": "A non-empty train and validation split is required.",
        }

    retriever = PopularityRetriever().fit(_examples_to_interactions(train))
    ranked_catalog = retriever.recommend(
        (),
        k=len(retriever.catalog),
        exclude_seen=False,
    )
    selected = _bounded_examples(validation, max(1, int(limit)))
    predictions: dict[int, object] = {}
    truths: dict[int, str] = {}
    for example in selected:
        predictions[example.example_id] = ranked_catalog[:k]
        truths[example.example_id] = example.target_track_uri

    diagnostics = candidate_diagnostics(
        predictions,
        truths=truths,
        catalog=retriever.catalog,
        k=k,
    )
    return {
        "status": "complete",
        "model": "track_popularity",
        "evaluated_examples": len(selected),
        "exclude_seen": False,
        **diagnostics.as_dict(),
    }


def _capability_rows() -> list[dict[str, object]]:
    errors = validate_expansion_registry()
    if errors:
        raise RuntimeError("Invalid expansion registry: " + "; ".join(errors))

    rows: list[dict[str, object]] = []
    for spec in list_expansion_specs():
        explainer = resolve_explainer_capability(spec.explainability_family)
        rows.append(
            {
                "capability": spec.key,
                "module": " | ".join(spec.implementation_modules),
                "stage": spec.stage,
                "readiness": spec.readiness,
                "optuna_supported": spec.optuna.supported,
                "optuna_objective": spec.optuna.objective,
                "explainability_strategy": explainer.strategy,
                "purpose": spec.summary,
                "blockers": " | ".join(spec.blockers),
            }
        )
    return rows


def _dataset_summary(
    dataset: TrackLevelDataset,
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> dict[str, object]:
    timestamps = [example.target_timestamp for example in dataset.examples]
    return {
        "source_rows": dataset.source_row_count,
        "valid_track_rows": dataset.valid_track_row_count,
        "example_count": len(dataset.examples),
        "unique_tracks": dataset.unique_track_count,
        "session_count": dataset.session_count,
        "train_examples": train_count,
        "validation_examples": validation_count,
        "test_examples": test_count,
        "first_target_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_target_timestamp": max(timestamps).isoformat() if timestamps else None,
    }


def _continuation_markdown(
    summary: dict[str, object],
    baseline: dict[str, object],
    capabilities: list[dict[str, object]],
    training: dict[str, object] | None = None,
    next_pass: dict[str, object] | None = None,
) -> list[str]:
    lines = [
        "# Recommender Expansion Lab",
        "",
        "## Current Data Surface",
        "",
        f"- Track-level examples: `{summary['example_count']}`",
        f"- Unique tracks: `{summary['unique_tracks']}`",
        f"- Sessions: `{summary['session_count']}`",
        (
            f"- Temporal split: train `{summary['train_examples']}`, validation "
            f"`{summary['validation_examples']}`, test `{summary['test_examples']}`"
        ),
        "",
        "## Baseline",
        "",
        f"- Status: `{baseline.get('status', 'unknown')}`",
    ]
    if baseline.get("status") == "complete":
        lines.extend(
            [
                f"- Recall@{baseline.get('k')}: `{float(baseline.get('recall_at_k', 0.0)):.6f}`",
                f"- Catalog coverage: `{float(baseline.get('catalog_coverage', 0.0)):.6f}`",
                f"- Evaluated examples: `{baseline.get('evaluated_examples')}`",
            ]
        )
    else:
        lines.append(f"- Reason: {baseline.get('reason', 'not available')}")

    lines.extend(["", "## Implementation Lanes", ""])
    for row in capabilities:
        lines.append(
            f"- `{row['capability']}` ({row['readiness']}): {row['purpose']} "
            f"Module: `{row['module']}`."
        )

    if training and training.get("status") == "complete":
        lines.extend(["", "## Latest Training Pass", ""])
        retrieval = training.get("retrieval_results", [])
        if isinstance(retrieval, list):
            for row in retrieval:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"- `{row.get('model_name', 'retrieval')}`: Recall@"
                    f"{row.get('k', '?')} "
                    f"`{float(row.get('recall_at_k', 0.0)):.6f}`, target coverage "
                    f"`{float(row.get('target_catalog_coverage', 0.0)):.6f}`."
                )
        neural = training.get("neural_results", [])
        if isinstance(neural, list):
            for row in neural:
                if not isinstance(row, dict):
                    continue
                validation = row.get("validation", {})
                if not isinstance(validation, dict):
                    validation = {}
                lines.append(
                    f"- `{row.get('model_name', 'neural')}`: validation Recall@"
                    f"{validation.get('k', '?')} "
                    f"`{float(validation.get('recall_at_k', 0.0)):.6f}`."
                )
        lines.extend(
            [
                "- Full metrics and checkpoints: `training/training_manifest.json`.",
            ]
        )

    if next_pass and next_pass.get("status") in {"complete", "partial", "blocked"}:
        lines.extend(["", "## Latest Next Pass", ""])
        lines.append(f"- Overall status: `{next_pass.get('status')}`.")
        stages = next_pass.get("stages", {})
        if isinstance(stages, dict):
            for name in (
                "candidate_dataset",
                "dcn_training",
                "optuna_tuning",
                "public_pretraining",
                "promotion_gates",
            ):
                row = stages.get(name, {})
                if not isinstance(row, dict):
                    continue
                detail = row.get("reason")
                suffix = f" - {detail}" if detail else ""
                lines.append(f"- `{name}`: `{row.get('status', 'unknown')}`{suffix}")
        lines.append("- Full evidence: `next_pass/next_pass_manifest.json`.")

    lines.extend(
        [
            "",
            "## Next Training Pass",
            "",
        ]
    )
    if next_pass and next_pass.get("status") in {"complete", "partial"}:
        lines.extend(
            [
                "1. Review `next_pass/stages/promotion_gates/promotion_gate_report.md` and resolve blocking evidence.",
                "2. Increase resumable Optuna budgets only after reviewing trial runtime and validation stability.",
                "3. Add public-data pretraining only after an approved source manifest and canonical local records are available.",
                "4. Promote only models whose temporal, calibration, SHAP, drift, and reproducibility gates pass.",
            ]
        )
    elif training and training.get("status") == "complete":
        lines.extend(
            [
                "1. Train DCN-V2 on retrieved candidates using context, item, and multimodal features.",
                "2. Tune the completed retrieval and neural models with Optuna against temporal validation metrics.",
                "3. Add public-data pretraining only after source-license and provenance validation passes.",
                "4. Promote models only after temporal, calibration, explainability, and drift gates pass.",
            ]
        )
    else:
        lines.extend(
            [
                "1. Fit bounded EASE and session co-occurrence baselines, then record Recall@K and coverage.",
                "2. Train MEANTIME and MMoE/PLE on encoded track sequences and multi-task labels.",
                "3. Train DCN-V2 on retrieved candidates using context, item, and multimodal features.",
                "4. Add public-data pretraining only after source-license and provenance validation passes.",
                "5. Promote models only after temporal, calibration, explainability, and drift gates pass.",
            ]
        )
    return lines


def build_recommender_expansion_lab(
    *,
    config: ExpansionRunConfig,
    logger: logging.Logger,
) -> list[Path]:
    raw = load_streaming_history(
        config.raw_data_dir,
        include_video=config.include_video,
        logger=logger,
    )
    dataset = build_track_level_dataset(
        raw,
        session_gap_minutes=config.session_gap_minutes,
        max_history=config.max_history,
        min_history=config.min_history,
    )
    splits = split_track_level_examples(
        dataset,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    summary = _dataset_summary(
        dataset,
        train_count=len(splits.train),
        validation_count=len(splits.validation),
        test_count=len(splits.test),
    )
    baseline = _evaluate_popularity(
        splits.train,
        splits.validation,
        k=config.evaluation_k,
        limit=config.evaluation_limit,
    )
    capabilities = _capability_rows()

    root = config.output_dir / "analysis" / "recommender_expansion"
    root.mkdir(parents=True, exist_ok=True)
    training_payload = safe_read_json(
        root / "training" / "training_manifest.json",
        default={},
    )
    training = training_payload if isinstance(training_payload, dict) else {}
    next_pass_payload = safe_read_json(
        root / "next_pass" / "next_pass_manifest.json",
        default={},
    )
    next_pass = next_pass_payload if isinstance(next_pass_payload, dict) else {}
    config_payload = {
        **asdict(config),
        "raw_data_dir": str(config.raw_data_dir.resolve()),
        "output_dir": str(config.output_dir.resolve()),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "implementation_ready",
        "config": config_payload,
        "dataset": summary,
        "baseline": baseline,
        "capability_count": len(capabilities),
        "capabilities": capabilities,
        "training": training,
        "next_pass": next_pass,
    }

    paths = [
        write_json(root / "expansion_manifest.json", manifest),
        write_json(root / "track_dataset_summary.json", summary),
        write_json(root / "track_popularity_baseline.json", baseline),
        write_csv_rows(
            root / "expansion_capabilities.csv",
            capabilities,
            fieldnames=[
                "capability",
                "module",
                "stage",
                "readiness",
                "optuna_supported",
                "optuna_objective",
                "explainability_strategy",
                "purpose",
                "blockers",
            ],
        ),
        write_markdown(
            root / "CONTINUE_HERE.md",
            _continuation_markdown(
                summary,
                baseline,
                capabilities,
                training,
                next_pass,
            ),
        ),
    ]
    logger.info(
        "Recommender expansion lab ready: examples=%d tracks=%d sessions=%d",
        len(dataset.examples),
        dataset.unique_track_count,
        dataset.session_count,
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build track-level recommender expansion data and readiness artifacts."
    )
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-history", type=int, default=256)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--session-gap-minutes", type=float, default=30.0)
    parser.add_argument("--evaluation-k", type=int, default=100)
    parser.add_argument("--evaluation-limit", type=int, default=20_000)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("spotify.recommender_expansion_lab")
    paths = build_recommender_expansion_lab(
        config=ExpansionRunConfig(
            raw_data_dir=Path(args.raw_data_dir),
            output_dir=Path(args.output_dir),
            max_history=args.max_history,
            min_history=args.min_history,
            session_gap_minutes=args.session_gap_minutes,
            evaluation_k=args.evaluation_k,
            evaluation_limit=args.evaluation_limit,
            include_video=not args.no_video,
        ),
        logger=logger,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
