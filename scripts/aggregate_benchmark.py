#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark-lock runs and compute confidence intervals.",
    )
    parser.add_argument("--benchmark-id", required=True, help="Benchmark lock id suffix.")
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help="Optional run_name prefix override. Defaults to benchmark-lock-<benchmark-id>.",
    )
    parser.add_argument(
        "--history-csv",
        default="outputs/history/experiment_history.csv",
        help="Path to experiment_history.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/history",
        help="Directory to write benchmark summary artifacts.",
    )
    return parser.parse_args()


def _safe_float(value: str | object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if np.isnan(numeric):
        return None
    return float(numeric)


def _metric_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    arr = np.array(values, dtype="float64")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"n": int(len(arr)), "mean": mean, "std": std, "ci95": ci95}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_benchmark_history(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "timestamp",
        "benchmark_id",
        "model_name",
        "model_type",
        "model_family",
        "runs",
        "val_top1_mean",
        "val_top1_std",
        "val_top1_ci95",
        "test_top1_mean",
        "test_top1_std",
        "test_top1_ci95",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_ci_chart(path: Path, summary_rows: list[dict[str, object]]) -> Path | None:
    if not summary_rows:
        return None
    labels = [str(row["model_name"]) for row in summary_rows]
    val_means = [float(row["val_top1_mean"]) for row in summary_rows]
    val_err = [float(row["val_top1_ci95"]) for row in summary_rows]
    test_means = [float(row["test_top1_mean"]) for row in summary_rows]
    test_err = [float(row["test_top1_ci95"]) for row in summary_rows]
    if not labels:
        return None

    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5))
    ax.bar(x - width / 2, val_means, width=width, yerr=val_err, capsize=4, label="Val Top-1 (mean ± CI)")
    ax.bar(x + width / 2, test_means, width=width, yerr=test_err, capsize=4, label="Test Top-1 (mean ± CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Benchmark Lock Confidence Intervals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _paired_significance(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        model_name = str(row.get("model_name", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        score = _safe_float(row.get("val_top1", ""))
        if not model_name or not run_id or score is None:
            continue
        grouped.setdefault(model_name, {})[run_id] = float(score)

    model_names = sorted(grouped)
    out: list[dict[str, object]] = []
    for idx, left in enumerate(model_names):
        for right in model_names[idx + 1 :]:
            common = sorted(set(grouped[left]) & set(grouped[right]))
            if not common:
                continue
            diffs = np.asarray([grouped[left][run_id] - grouped[right][run_id] for run_id in common], dtype="float64")
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            stderr = float(std_diff / math.sqrt(len(diffs))) if len(diffs) > 0 else float("nan")
            z_score = float(mean_diff / stderr) if stderr > 0 else float("inf" if mean_diff > 0 else 0.0)
            out.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "shared_runs": len(common),
                    "mean_diff_val_top1": mean_diff,
                    "ci95_diff_val_top1": float(1.96 * stderr) if stderr > 0 else 0.0,
                    "z_score": z_score,
                    "significant_at_95": int(abs(z_score) >= 1.96),
                }
            )
    return out


def main() -> int:
    from spotify.benchmark_contract import write_benchmark_lock_manifest

    args = _parse_args()
    history_csv = Path(args.history_csv).expanduser().resolve()
    if not history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {history_csv}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_dir / ".mplconfig"
    xdg_cache_dir = output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
    prefix = args.run_name_prefix or f"benchmark-lock-{args.benchmark_id}"

    with history_csv.open("r", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))

    filtered = [row for row in rows if str(row.get("run_name", "")).startswith(prefix)]
    if not filtered:
        raise RuntimeError(
            f"No rows found for run_name prefix '{prefix}' in {history_csv}."
        )

    raw_path = output_dir / f"benchmark_lock_{args.benchmark_id}_rows.csv"
    _write_csv(raw_path, fieldnames=list(filtered[0].keys()), rows=filtered)

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in filtered:
        key = (
            str(row.get("model_name", "")),
            str(row.get("model_type", "")),
            str(row.get("model_family", "")),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (model_name, model_type, model_family), model_rows in grouped.items():
        val_top1 = [_safe_float(row.get("val_top1", "")) for row in model_rows]
        test_top1 = [_safe_float(row.get("test_top1", "")) for row in model_rows]
        val_top5 = [_safe_float(row.get("val_top5", "")) for row in model_rows]
        test_top5 = [_safe_float(row.get("test_top5", "")) for row in model_rows]
        fit_seconds = [_safe_float(row.get("fit_seconds", "")) for row in model_rows]

        val_top1_stats = _metric_stats([value for value in val_top1 if value is not None])
        test_top1_stats = _metric_stats([value for value in test_top1 if value is not None])
        val_top5_stats = _metric_stats([value for value in val_top5 if value is not None])
        test_top5_stats = _metric_stats([value for value in test_top5 if value is not None])
        fit_stats = _metric_stats([value for value in fit_seconds if value is not None])

        run_ids = sorted({str(row.get("run_id", "")) for row in model_rows if str(row.get("run_id", ""))})
        summary_rows.append(
            {
                "benchmark_id": args.benchmark_id,
                "run_name_prefix": prefix,
                "run_count": len(run_ids),
                "model_name": model_name,
                "model_type": model_type,
                "model_family": model_family,
                "val_top1_mean": val_top1_stats["mean"],
                "val_top1_std": val_top1_stats["std"],
                "val_top1_ci95": val_top1_stats["ci95"],
                "test_top1_mean": test_top1_stats["mean"],
                "test_top1_std": test_top1_stats["std"],
                "test_top1_ci95": test_top1_stats["ci95"],
                "val_top5_mean": val_top5_stats["mean"],
                "test_top5_mean": test_top5_stats["mean"],
                "fit_seconds_mean": fit_stats["mean"],
            }
        )

    summary_rows.sort(key=lambda row: float(row.get("val_top1_mean", float("-inf"))), reverse=True)

    summary_csv = output_dir / f"benchmark_lock_{args.benchmark_id}_summary.csv"
    _write_csv(
        summary_csv,
        fieldnames=[
            "benchmark_id",
            "run_name_prefix",
            "run_count",
            "model_name",
            "model_type",
            "model_family",
            "val_top1_mean",
            "val_top1_std",
            "val_top1_ci95",
            "test_top1_mean",
            "test_top1_std",
            "test_top1_ci95",
            "val_top5_mean",
            "test_top5_mean",
            "fit_seconds_mean",
        ],
        rows=summary_rows,
    )

    summary_json = output_dir / f"benchmark_lock_{args.benchmark_id}_summary.json"
    with summary_json.open("w", encoding="utf-8") as outfile:
        json.dump(summary_rows, outfile, indent=2)

    ci_plot = output_dir / f"benchmark_lock_{args.benchmark_id}_ci95.png"
    _plot_ci_chart(ci_plot, summary_rows)
    significance_rows = _paired_significance(filtered)
    significance_csv = output_dir / f"benchmark_lock_{args.benchmark_id}_significance.csv"
    if significance_rows:
        _write_csv(
            significance_csv,
            [
                "left_model",
                "right_model",
                "shared_runs",
                "mean_diff_val_top1",
                "ci95_diff_val_top1",
                "z_score",
                "significant_at_95",
            ],
            significance_rows,
        )
    manifest_json, manifest_md = write_benchmark_lock_manifest(
        output_dir=output_dir,
        benchmark_id=args.benchmark_id,
        run_name_prefix=prefix,
        summary_rows=summary_rows,
        significance_rows=significance_rows,
        raw_rows=filtered,
    )

    history_append_rows = [
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "benchmark_id": args.benchmark_id,
            "model_name": row["model_name"],
            "model_type": row["model_type"],
            "model_family": row["model_family"],
            "runs": row["run_count"],
            "val_top1_mean": row["val_top1_mean"],
            "val_top1_std": row["val_top1_std"],
            "val_top1_ci95": row["val_top1_ci95"],
            "test_top1_mean": row["test_top1_mean"],
            "test_top1_std": row["test_top1_std"],
            "test_top1_ci95": row["test_top1_ci95"],
        }
        for row in summary_rows
    ]
    _append_benchmark_history(output_dir / "benchmark_history.csv", history_append_rows)

    print(f"benchmark_rows={raw_path}")
    print(f"benchmark_summary={summary_csv}")
    print(f"benchmark_ci_plot={ci_plot}")
    print(f"benchmark_manifest={manifest_json}")
    print(f"benchmark_manifest_md={manifest_md}")
    if significance_rows:
        print(f"benchmark_significance={significance_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
