from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from spotify.artifact_cleanup import prune_existing_runs, prune_mlflow_artifacts, prune_old_auxiliary_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune bulky model artifacts from existing Spotify run directories.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Project output directory containing run artifacts.")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific run directory to prune. Repeat to prune multiple runs. Defaults to all runs under outputs/runs.",
    )
    parser.add_argument("--mode", type=str, default="light", help="Cleanup mode: off, light, or aggressive.")
    parser.add_argument("--min-size-mb", type=float, default=100.0, help="Only delete artifacts at or above this size.")
    parser.add_argument("--keep-full-runs", type=int, default=2, help="Keep this many most recent full runs intact.")
    parser.add_argument(
        "--no-prune-prediction-bundles",
        action="store_true",
        help="Do not remove old prediction bundle artifacts.",
    )
    parser.add_argument(
        "--no-prune-run-databases",
        action="store_true",
        help="Do not remove old run-local database files.",
    )
    parser.add_argument(
        "--no-prune-mlflow-artifacts",
        action="store_true",
        help="Do not prune mirrored MLflow artifact files from the MLflow artifact store.",
    )
    parser.add_argument(
        "--mlflow-artifact-mode",
        type=str,
        default="",
        help="MLflow artifact cleanup policy: metadata (default), all, or off.",
    )
    parser.add_argument(
        "--mlflow-max-artifact-mb",
        type=float,
        default=None,
        help="Max size for MLflow artifacts retained under metadata mode.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional path to write a JSON cleanup summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.prune_artifacts")

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dirs = [Path(item).expanduser().resolve() for item in args.run_dir]
    estimator_summary = prune_existing_runs(
        output_dir=output_dir,
        run_dirs=run_dirs or None,
        logger=logger,
        cleanup_mode=args.mode,
        min_size_mb=args.min_size_mb,
    )
    auxiliary_summary = prune_old_auxiliary_artifacts(
        output_dir=output_dir,
        current_run_dir=None,
        logger=logger,
        keep_last_full_runs=max(0, int(args.keep_full_runs)),
        prune_prediction_bundles=not bool(args.no_prune_prediction_bundles),
        prune_run_databases=not bool(args.no_prune_run_databases),
    )
    mlflow_summary = (
        {
            "enabled": False,
            "artifact_mode": "off",
            "max_artifact_mb": 0.0,
            "status": "disabled_by_flag",
            "artifact_dir_count": 0,
            "artifact_dirs": [],
            "deleted_file_count": 0,
            "deleted_files": [],
            "freed_bytes": 0,
        }
        if args.no_prune_mlflow_artifacts
        else prune_mlflow_artifacts(
            output_dir=output_dir,
            logger=logger,
            artifact_mode=(args.mlflow_artifact_mode or None),
            max_artifact_mb=args.mlflow_max_artifact_mb,
        )
    )
    summary = {
        "estimator_cleanup": estimator_summary,
        "auxiliary_cleanup": auxiliary_summary,
        "mlflow_artifact_cleanup": mlflow_summary,
        "deleted_file_count": int(estimator_summary.get("deleted_file_count", 0))
        + int(auxiliary_summary.get("deleted_file_count", 0))
        + int(mlflow_summary.get("deleted_file_count", 0)),
        "freed_bytes": int(estimator_summary.get("freed_bytes", 0))
        + int(auxiliary_summary.get("freed_bytes", 0))
        + int(mlflow_summary.get("freed_bytes", 0)),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
