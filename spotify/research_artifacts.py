from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import math

import numpy as np

from .benchmark_contract import describe_canonical_benchmark_contract
from .run_artifacts import write_csv_rows


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _paired_significance_rows(backtest_rows: list[dict[str, object]], top_models: list[str]) -> list[dict[str, object]]:
    by_model_fold: dict[str, dict[int, float]] = {}
    for row in backtest_rows:
        model_name = str(row.get("model_name", "")).strip()
        if not model_name:
            continue
        fold = int(row.get("fold", 0) or 0)
        score = _safe_float(row.get("top1"))
        if math.isnan(score):
            continue
        by_model_fold.setdefault(model_name, {})[fold] = score

    rows: list[dict[str, object]] = []
    for idx, left in enumerate(top_models):
        for right in top_models[idx + 1 :]:
            left_scores = by_model_fold.get(left, {})
            right_scores = by_model_fold.get(right, {})
            common_folds = sorted(set(left_scores) & set(right_scores))
            if not common_folds:
                continue
            diffs = np.asarray([left_scores[fold] - right_scores[fold] for fold in common_folds], dtype="float64")
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            stderr = float(std_diff / math.sqrt(len(diffs))) if len(diffs) > 0 else float("nan")
            z_score = float(mean_diff / stderr) if stderr > 0 else float("inf" if mean_diff > 0 else 0.0)
            ci95 = float(1.96 * stderr) if stderr > 0 else 0.0
            rows.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "fold_count": int(len(common_folds)),
                    "mean_diff_top1": mean_diff,
                    "std_diff_top1": std_diff,
                    "stderr_diff_top1": stderr,
                    "ci95_diff_top1": ci95,
                    "z_score": z_score,
                    "significant_at_95": int(abs(z_score) >= 1.96),
                    "winner": left if mean_diff > 0 else right if mean_diff < 0 else "tie",
                }
            )
    return rows


def write_benchmark_protocol(
    *,
    output_dir: Path,
    run_id: str,
    profile: str,
    data,
    cache_info: dict[str, object],
    config,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {
        "train": int(len(data.X_seq_train)),
        "val": int(len(data.X_seq_val)),
        "test": int(len(data.X_seq_test)),
    }
    contract = describe_canonical_benchmark_contract()
    payload = {
        "run_id": str(run_id).strip(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": str(profile).strip(),
        "random_seed": int(config.random_seed),
        "sequence_length": int(config.sequence_length),
        "max_artists": int(config.max_artists),
        "num_artists": int(data.num_artists),
        "num_context_features": int(data.num_ctx),
        "split_rows": split_rows,
        "prepared_cache": dict(cache_info),
        "benchmark_contract": contract,
        "protocol": {
            "split_rule": "time_ordered_prefix_holdout_after_sequence_alignment",
            "primary_metrics": ["top1", "top5", "ndcg_at5", "mrr_at5", "coverage_at5", "diversity_at5"],
            "risk_metrics": ["ece", "brier", "abstention_rate", "selective_risk"],
            "temporal_backtest": {
                "enabled": bool(config.enable_temporal_backtest),
                "folds": int(config.temporal_backtest_folds),
                "models": list(config.temporal_backtest_model_names),
            },
            "reproducibility": {
                "data_fingerprint": str(cache_info.get("fingerprint", "")),
                "source_file_count": int(cache_info.get("source_file_count", 0) or 0),
            },
            "benchmark_lock": {
                "canonical_profile": str(contract.get("canonical_profile", "")),
                "comparison_mode": str(contract.get("comparison_mode", "")),
                "minimum_repeated_runs": int(contract.get("minimum_repeated_runs", 0) or 0),
                "required_run_artifacts": list(contract.get("required_run_artifacts", [])),
                "required_benchmark_lock_artifacts": list(contract.get("required_benchmark_lock_artifacts", [])),
                "significance_policy": dict(contract.get("significance_policy", {})),
                "stability_rules": list(contract.get("stability_rules", [])),
            },
        },
    }
    json_path = output_dir / "benchmark_protocol.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = output_dir / "benchmark_protocol.md"
    md_lines = [
        "# Benchmark Protocol",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Profile: `{payload['profile']}`",
        f"- Random seed: `{payload['random_seed']}`",
        f"- Sequence length: `{payload['sequence_length']}`",
        f"- Artist classes: `{payload['num_artists']}`",
        f"- Context features: `{payload['num_context_features']}`",
        f"- Split rows: train=`{split_rows['train']}` val=`{split_rows['val']}` test=`{split_rows['test']}`",
        f"- Data fingerprint: `{payload['protocol']['reproducibility']['data_fingerprint']}`",
        "",
        "## Reproducibility Checklist",
        "",
        "- Use the same raw export files and prepared-data fingerprint.",
        "- Keep `sequence_length`, `max_artists`, and `random_seed` fixed.",
        "- Compare models on the same holdout split and temporal backtest folds.",
        "- Report both ranking quality and risk metrics.",
        "",
        "## Benchmark Lock",
        "",
        f"- Contract version: `{contract['contract_version']}`",
        f"- Canonical profile: `{contract['canonical_profile']}`",
        f"- Comparison mode: `{contract['comparison_mode']}` with at least `{contract['minimum_repeated_runs']}` repeated runs.",
        f"- Significance metric: `{contract['significance_policy']['metric']}` at `{contract['significance_policy']['confidence_level']}` confidence with z >= `{contract['significance_policy']['z_threshold']}`.",
        "",
        "## Locked Artifact Set",
        "",
    ]
    for item in contract["required_run_artifacts"]:
        md_lines.append(f"- `{item}`")
    md_lines.extend(["", "## Stability Rules", ""])
    for item in contract["stability_rules"]:
        md_lines.append(f"- {item}")
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    return [json_path, md_path]


def write_experiment_registry(
    *,
    output_dir: Path,
    run_id: str,
    profile: str,
    results: list[dict[str, object]],
    backtest_rows: list[dict[str, object]],
    config,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_counts: dict[str, int] = {}
    for row in results:
        model_type = str(row.get("model_type", "")).strip() or "unknown"
        model_counts[model_type] = model_counts.get(model_type, 0) + 1

    payload = {
        "run_id": str(run_id).strip(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": str(profile).strip(),
        "tracks": {
            "uncertainty": bool(config.enable_conformal),
            "drift": True,
            "retrieval": bool(getattr(config, "enable_retrieval_stack", False)),
            "friction": bool(getattr(config, "enable_friction_analysis", False)),
            "moonshot_lab": bool(getattr(config, "enable_moonshot_lab", False)),
            "temporal_backtest": bool(config.enable_temporal_backtest),
        },
        "model_counts": model_counts,
        "result_row_count": int(len(results)),
        "backtest_row_count": int(len(backtest_rows)),
        "models": [
            {
                "model_name": str(row.get("model_name", "")),
                "model_type": str(row.get("model_type", "")),
                "model_family": str(row.get("model_family", "")),
                "val_top1": _safe_float(row.get("val_top1")),
                "test_top1": _safe_float(row.get("test_top1")),
            }
            for row in sorted(results, key=lambda item: _safe_float(item.get("val_top1")), reverse=True)
        ],
    }
    out_path = output_dir / "experiment_registry.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_ablation_summary(
    *,
    output_dir: Path,
    results: list[dict[str, object]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        return []

    ordered = sorted(results, key=lambda row: _safe_float(row.get("val_top1")), reverse=True)
    best_score = _safe_float(ordered[0].get("val_top1"))
    by_group: dict[tuple[str, str], dict[str, object]] = {}
    for row in ordered:
        key = (
            str(row.get("model_type", "")).strip(),
            str(row.get("model_family", "")).strip(),
        )
        if key not in by_group:
            by_group[key] = row

    rows: list[dict[str, object]] = []
    for (model_type, model_family), row in by_group.items():
        val_top1 = _safe_float(row.get("val_top1"))
        rows.append(
            {
                "group": f"{model_type}:{model_family or 'unspecified'}",
                "model_name": str(row.get("model_name", "")).strip(),
                "model_type": model_type,
                "model_family": model_family,
                "val_top1": val_top1,
                "test_top1": _safe_float(row.get("test_top1")),
                "delta_to_best_val_top1": float(best_score - val_top1) if not math.isnan(best_score) and not math.isnan(val_top1) else float("nan"),
                "fit_seconds": _safe_float(row.get("fit_seconds")),
            }
        )
    rows.sort(key=lambda row: _safe_float(row.get("val_top1")), reverse=True)

    csv_path = _write_csv(
        output_dir / "ablation_summary.csv",
        ["group", "model_name", "model_type", "model_family", "val_top1", "test_top1", "delta_to_best_val_top1", "fit_seconds"],
        rows,
    )
    json_path = output_dir / "ablation_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return [csv_path, json_path]


def write_significance_summary(
    *,
    output_dir: Path,
    results: list[dict[str, object]],
    backtest_rows: list[dict[str, object]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not backtest_rows or not results:
        return []

    top_models = [
        str(row.get("model_name", "")).strip()
        for row in sorted(results, key=lambda item: _safe_float(item.get("val_top1")), reverse=True)
        if str(row.get("model_name", "")).strip()
    ][:5]
    rows = _paired_significance_rows(backtest_rows, top_models=top_models)
    if not rows:
        return []

    csv_path = _write_csv(
        output_dir / "backtest_significance.csv",
        [
            "left_model",
            "right_model",
            "fold_count",
            "mean_diff_top1",
            "std_diff_top1",
            "stderr_diff_top1",
            "ci95_diff_top1",
            "z_score",
            "significant_at_95",
            "winner",
        ],
        rows,
    )
    json_path = output_dir / "backtest_significance.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return [csv_path, json_path]
