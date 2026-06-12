from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import math
import re
from typing import Any

from .run_artifacts import safe_read_json, write_csv_rows, write_json, write_markdown


DEFAULT_DEEP_MODELS: tuple[str, ...] = (
    "sasrec",
    "bert4rec",
    "srgnn",
    "dense",
    "gru",
    "transformer",
    "lstm",
    "cnn",
    "tcn",
    "cnn_lstm",
    "attention_rnn",
    "tft",
    "transformer_xl",
    "memory_net",
    "graph_seq",
    "gru_artist",
    "memory_net_artist",
)

_TEST_LINE_RE = re.compile(
    r"\[TEST\]\s+(?P<model>[A-Za-z0-9_]+):\s+Top-1=(?P<top1>[0-9.]+)\s+\|\s+Top-5=(?P<top5>[0-9.]+)"
)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _metric_series(history: dict[str, Any], *keys: str) -> list[float]:
    for key in keys:
        values = history.get(key)
        if isinstance(values, list):
            return [_safe_float(value) for value in values]
    return []


def _last_finite(values: list[float]) -> float:
    for value in reversed(values):
        if math.isfinite(value):
            return value
    return float("nan")


def _best_finite(values: list[float]) -> tuple[float, int]:
    best_value = float("nan")
    best_epoch = 0
    for idx, value in enumerate(values, start=1):
        if not math.isfinite(value):
            continue
        if not math.isfinite(best_value) or value > best_value:
            best_value = value
            best_epoch = idx
    return best_value, best_epoch


def _min_finite(values: list[float]) -> tuple[float, int]:
    best_value = float("nan")
    best_epoch = 0
    for idx, value in enumerate(values, start=1):
        if not math.isfinite(value):
            continue
        if not math.isfinite(best_value) or value < best_value:
            best_value = value
            best_epoch = idx
    return best_value, best_epoch


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _median(values: list[float]) -> float:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return float("nan")
    midpoint = len(finite_values) // 2
    if len(finite_values) % 2:
        return finite_values[midpoint]
    return (finite_values[midpoint - 1] + finite_values[midpoint]) / 2.0


def _parse_test_metrics(train_log: Path) -> dict[str, dict[str, float]]:
    if not train_log.exists():
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _TEST_LINE_RE.search(line)
        if not match:
            continue
        metrics[match.group("model")] = {
            "test_top1": _safe_float(match.group("top1")),
            "test_top5": _safe_float(match.group("top5")),
        }
    return metrics


def _find_run_by_name(outputs_dir: Path, run_name: str) -> Path:
    run_name = run_name.strip()
    if not run_name:
        raise ValueError("run_name must be non-empty")
    candidates = [
        path
        for path in (outputs_dir / "runs").glob(f"*_{run_name}")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directory found for run name: {run_name}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _split_expected_models(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or DEFAULT_DEEP_MODELS


def _row_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        -_safe_float(row.get("best_val_top1")),
        -_safe_float(row.get("test_top1")),
        str(row.get("model_name", "")),
    )


def _display_metric(value: Any) -> str:
    metric = _safe_float(value)
    return f"{metric:.4f}" if math.isfinite(metric) else "n/a"


def build_deep_benchmark_summary(
    *,
    run_dir: Path,
    expected_models: tuple[str, ...] = DEFAULT_DEEP_MODELS,
) -> dict[str, Any]:
    histories = safe_read_json(run_dir / "histories.json", default={})
    if not isinstance(histories, dict) or not histories:
        raise FileNotFoundError(f"Missing or empty histories.json in {run_dir}")

    test_metrics = _parse_test_metrics(run_dir / "train.log")
    rows: list[dict[str, Any]] = []
    for model_name, raw_history in histories.items():
        if not isinstance(raw_history, dict):
            continue
        val_top1 = _metric_series(
            raw_history,
            "val_artist_output_sparse_categorical_accuracy",
            "val_sparse_categorical_accuracy",
        )
        val_top5 = _metric_series(raw_history, "val_artist_output_top_5", "val_top_5")
        train_top1 = _metric_series(
            raw_history,
            "artist_output_sparse_categorical_accuracy",
            "sparse_categorical_accuracy",
        )
        val_loss = _metric_series(raw_history, "val_loss")
        best_val_top1, best_epoch = _best_finite(val_top1)
        best_val_top5, best_top5_epoch = _best_finite(val_top5)
        min_val_loss, min_val_loss_epoch = _min_finite(val_loss)
        test_top1 = _safe_float(test_metrics.get(str(model_name), {}).get("test_top1"))
        test_top5 = _safe_float(test_metrics.get(str(model_name), {}).get("test_top5"))
        val_test_gap = best_val_top1 - test_top1 if _all_finite(best_val_top1, test_top1) else float("nan")
        test_to_val_ratio = (
            test_top1 / best_val_top1
            if _all_finite(best_val_top1, test_top1) and best_val_top1 > 0
            else float("nan")
        )
        artifact_path = run_dir / f"best_{model_name}.keras"
        row = {
            "model_name": str(model_name),
            "model_type": "deep",
            "epochs_recorded": int(max(len(val_top1), len(val_top5), len(val_loss))),
            "best_epoch": int(best_epoch),
            "best_val_top1": best_val_top1,
            "final_val_top1": _last_finite(val_top1),
            "best_val_top5": best_val_top5,
            "best_val_top5_epoch": int(best_top5_epoch),
            "final_val_top5": _last_finite(val_top5),
            "final_train_top1": _last_finite(train_top1),
            "min_val_loss": min_val_loss,
            "min_val_loss_epoch": int(min_val_loss_epoch),
            "test_top1": test_top1,
            "test_top5": test_top5,
            "val_test_top1_gap": val_test_gap,
            "test_to_val_top1_ratio": test_to_val_ratio,
            "artifact_path": str(artifact_path.resolve()) if artifact_path.exists() else "",
            "artifact_exists": artifact_path.exists(),
        }
        rows.append(row)

    rows = sorted(rows, key=_row_sort_key)
    expected_set = {str(model) for model in expected_models}
    observed_set = {str(row["model_name"]) for row in rows}
    missing_models = sorted(expected_set - observed_set)
    missing_artifacts = sorted(str(row["model_name"]) for row in rows if not row.get("artifact_exists"))
    missing_test_metrics = sorted(str(row["model_name"]) for row in rows if not math.isfinite(_safe_float(row.get("test_top1"))))
    best_by_val = rows[0] if rows else {}
    best_by_test = max(rows, key=lambda row: _safe_float(row.get("test_top1")), default={})
    gap_rows = [row for row in rows if math.isfinite(_safe_float(row.get("val_test_top1_gap")))]
    worst_gap = max(gap_rows, key=lambda row: _safe_float(row.get("val_test_top1_gap")), default={})
    large_gap_models = [
        str(row.get("model_name", ""))
        for row in rows
        if _safe_float(row.get("val_test_top1_gap")) >= 0.20
    ]
    full_manifest_exists = (run_dir / "run_manifest.json").exists()

    status = "deep_complete"
    if missing_models or missing_artifacts or missing_test_metrics:
        status = "deep_partial"
    if not full_manifest_exists:
        status = f"{status}_pipeline_incomplete"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "status": status,
        "pipeline_manifest_exists": full_manifest_exists,
        "expected_model_count": len(expected_models),
        "observed_model_count": len(rows),
        "artifact_model_count": sum(1 for row in rows if row.get("artifact_exists")),
        "test_metric_model_count": sum(1 for row in rows if math.isfinite(_safe_float(row.get("test_top1")))),
        "missing_models": missing_models,
        "missing_artifacts": missing_artifacts,
        "missing_test_metrics": missing_test_metrics,
        "best_by_val_top1": {
            "model_name": best_by_val.get("model_name", ""),
            "best_val_top1": best_by_val.get("best_val_top1"),
            "best_epoch": best_by_val.get("best_epoch"),
            "test_top1": best_by_val.get("test_top1"),
        },
        "best_by_test_top1": {
            "model_name": best_by_test.get("model_name", ""),
            "test_top1": best_by_test.get("test_top1"),
            "best_val_top1": best_by_test.get("best_val_top1"),
        },
        "generalization": {
            "median_val_test_top1_gap": _median([_safe_float(row.get("val_test_top1_gap")) for row in rows]),
            "worst_val_test_top1_gap": worst_gap.get("val_test_top1_gap"),
            "worst_gap_model_name": worst_gap.get("model_name", ""),
            "large_gap_threshold": 0.20,
            "large_gap_model_count": len(large_gap_models),
            "large_gap_models": large_gap_models,
        },
        "rows": rows,
    }


def _markdown_lines(summary: dict[str, Any]) -> list[str]:
    rows = list(summary.get("rows", []))
    lines = [
        "# Deep Benchmark Summary",
        "",
        f"- Run: `{summary.get('run_id', '')}`",
        f"- Status: `{summary.get('status', '')}`",
        f"- Models observed: `{summary.get('observed_model_count', 0)}/{summary.get('expected_model_count', 0)}`",
        f"- Models with artifacts: `{summary.get('artifact_model_count', 0)}`",
        f"- Models with test metrics: `{summary.get('test_metric_model_count', 0)}`",
        f"- Full pipeline manifest exists: `{bool(summary.get('pipeline_manifest_exists'))}`",
        "",
        "## Leaders",
        "",
        f"- Best validation top-1: `{(summary.get('best_by_val_top1') or {}).get('model_name', '')}` at `{_display_metric((summary.get('best_by_val_top1') or {}).get('best_val_top1'))}`",
        f"- Best test top-1: `{(summary.get('best_by_test_top1') or {}).get('model_name', '')}` at `{_display_metric((summary.get('best_by_test_top1') or {}).get('test_top1'))}`",
        "",
        "## Generalization",
        "",
        f"- Median validation-to-test top-1 gap: `{_display_metric((summary.get('generalization') or {}).get('median_val_test_top1_gap'))}`",
        f"- Worst validation-to-test top-1 gap: `{(summary.get('generalization') or {}).get('worst_gap_model_name', '')}` at `{_display_metric((summary.get('generalization') or {}).get('worst_val_test_top1_gap'))}`",
        f"- Models above `{_display_metric((summary.get('generalization') or {}).get('large_gap_threshold'))}` gap: `{(summary.get('generalization') or {}).get('large_gap_model_count', 0)}`",
        "",
    ]
    if summary.get("missing_models"):
        lines.extend(["## Missing Models", ""])
        for model_name in summary["missing_models"]:
            lines.append(f"- `{model_name}`")
        lines.append("")
    if summary.get("missing_test_metrics"):
        lines.extend(["## Missing Test Metrics", ""])
        for model_name in summary["missing_test_metrics"]:
            lines.append(f"- `{model_name}`")
        lines.append("")

    lines.extend(
        [
            "## Model Table",
            "",
            "| Model | Best Val Top-1 | Test Top-1 | Val-Test Gap | Test/Val Ratio | Best Val Top-5 | Test Top-5 | Epochs | Artifact |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        artifact = "yes" if row.get("artifact_exists") else "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('model_name', '')}`",
                    _display_metric(row.get("best_val_top1")),
                    _display_metric(row.get("test_top1")),
                    _display_metric(row.get("val_test_top1_gap")),
                    _display_metric(row.get("test_to_val_top1_ratio")),
                    _display_metric(row.get("best_val_top5")),
                    _display_metric(row.get("test_top5")),
                    str(row.get("epochs_recorded", "")),
                    artifact,
                ]
            )
            + " |"
        )
    return lines


def write_deep_benchmark_summary(
    *,
    run_dir: Path,
    outputs_dir: Path | None = None,
    expected_models: tuple[str, ...] = DEFAULT_DEEP_MODELS,
    mirror_latest: bool = True,
) -> list[Path]:
    summary = build_deep_benchmark_summary(run_dir=run_dir, expected_models=expected_models)
    summary_dir = run_dir / "analysis" / "deep_benchmark"
    summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = write_json(summary_dir / "deep_benchmark_summary.json", summary)
    csv_path = write_csv_rows(
        summary_dir / "deep_benchmark_summary.csv",
        list(summary.get("rows", [])),
        fieldnames=[
            "model_name",
            "model_type",
            "epochs_recorded",
            "best_epoch",
            "best_val_top1",
            "final_val_top1",
            "best_val_top5",
            "best_val_top5_epoch",
            "final_val_top5",
            "final_train_top1",
            "min_val_loss",
            "min_val_loss_epoch",
            "test_top1",
            "test_top5",
            "val_test_top1_gap",
            "test_to_val_top1_ratio",
            "artifact_exists",
            "artifact_path",
        ],
    )
    md_path = write_markdown(summary_dir / "deep_benchmark_summary.md", _markdown_lines(summary))
    written = [json_path, csv_path, md_path]

    if mirror_latest:
        resolved_outputs_dir = outputs_dir or run_dir.parent.parent
        latest_dir = resolved_outputs_dir / "analysis" / "deep_benchmark"
        latest_payload = dict(summary)
        latest_payload["source_summary_dir"] = str(summary_dir.resolve())
        written.append(write_json(latest_dir / "latest_deep_benchmark_summary.json", latest_payload))
        written.append(
            write_csv_rows(
                latest_dir / "latest_deep_benchmark_summary.csv",
                list(summary.get("rows", [])),
                fieldnames=[
                    "model_name",
                    "model_type",
                    "epochs_recorded",
                    "best_epoch",
                    "best_val_top1",
                    "final_val_top1",
                    "best_val_top5",
                    "best_val_top5_epoch",
                    "final_val_top5",
                    "final_train_top1",
                    "min_val_loss",
                    "min_val_loss_epoch",
                    "test_top1",
                    "test_top5",
                    "val_test_top1_gap",
                    "test_to_val_top1_ratio",
                    "artifact_exists",
                    "artifact_path",
                ],
            )
        )
        written.append(write_markdown(latest_dir / "latest_deep_benchmark_summary.md", _markdown_lines(summary)))

    return written


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Finalize a partial or completed all-deep benchmark run.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory to finalize.")
    parser.add_argument("--run-name", type=str, default="", help="Find the latest run directory for this run name.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Outputs root.")
    parser.add_argument(
        "--expected-models",
        type=str,
        default=",".join(DEFAULT_DEEP_MODELS),
        help="Comma-separated deep model set that this benchmark intended to train.",
    )
    parser.add_argument("--no-mirror-latest", action="store_true", help="Do not mirror artifacts under outputs/analysis/deep_benchmark.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _find_run_by_name(args.outputs_dir, args.run_name)
    run_dir = run_dir.expanduser().resolve()
    paths = write_deep_benchmark_summary(
        run_dir=run_dir,
        outputs_dir=args.outputs_dir.expanduser().resolve(),
        expected_models=_split_expected_models(args.expected_models),
        mirror_latest=not args.no_mirror_latest,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
