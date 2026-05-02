from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import json
import os
import time

import numpy as np

from .benchmarks import (
    ClassicalFeatureBundle,
    build_classical_estimator,
    build_classical_feature_bundle,
    sample_indices,
    collect_aligned_probabilities,
    evaluate_classical_estimator,
    resolve_classical_parallelism,
    validate_classical_models,
)
from .data import PreparedData
from .probability_bundles import save_prediction_bundle
from .run_artifacts import copy_file_if_changed, safe_read_json, write_json


OPTUNA_CACHE_SCHEMA_VERSION = "optuna-tuning-cache-v1"


@dataclass
class OptunaTuningResult:
    model_name: str
    base_model_name: str
    model_family: str
    fit_seconds: float
    val_top1: float
    val_top5: float
    val_ndcg_at5: float
    val_mrr_at5: float
    val_coverage_at5: float
    val_diversity_at5: float
    test_top1: float
    test_top5: float
    test_ndcg_at5: float
    test_mrr_at5: float
    test_coverage_at5: float
    test_diversity_at5: float
    n_trials: int
    best_params: dict[str, object]
    prediction_bundle_path: str = ""
    estimator_artifact_path: str = ""


@dataclass(frozen=True)
class OptunaModelCachePaths:
    cache_key: str
    cache_dir: Path
    result_path: Path
    metadata_path: Path
    trial_log_path: Path
    history_plot_path: Path
    estimator_artifact_path: Path
    prediction_bundle_path: Path


@dataclass(frozen=True)
class OptunaWarmStartCandidate:
    cache_fingerprint: str
    cache_key: str
    best_params: dict[str, object]
    val_top1: float
    modified_time: float


def _load_optuna():
    try:
        import optuna
    except Exception:
        return None
    return optuna


def _optuna_cache_enabled_from_env() -> bool:
    raw = os.getenv("SPOTIFY_CACHE_OPTUNA", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _optuna_warm_start_enabled_from_env() -> bool:
    raw = os.getenv("SPOTIFY_WARM_START_OPTUNA", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _suggest_params(trial, model_name: str) -> dict[str, object]:
    if model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 20.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 250, 900),
        }
    if model_name in ("random_forest", "extra_trees"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 600, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 120, 420, step=20),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        }
    if model_name == "hist_gbm":
        return {
            "max_iter": trial.suggest_int("max_iter", 80, 400, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        }
    if model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 60),
        }
    if model_name == "session_knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 24, 120, step=8),
            "candidate_cap": trial.suggest_int("candidate_cap", 128, 1024, step=128),
            "smoothing": trial.suggest_float("smoothing", 0.0, 3.0),
        }
    if model_name == "mlp":
        return {
            "hidden_1": trial.suggest_int("hidden_1", 64, 384, step=32),
            "hidden_2": trial.suggest_int("hidden_2", 32, 256, step=32),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 3e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "max_iter": trial.suggest_int("max_iter", 60, 180, step=10),
        }
    return {}


def _parse_positive_int(raw: str | None, fallback: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value > 0 else fallback
    except Exception:
        return fallback


def _parse_model_timeout_overrides(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    out: dict[str, int] = {}
    for chunk in str(raw).split(","):
        part = chunk.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        try:
            timeout_val = int(value.strip())
        except Exception:
            continue
        if key and timeout_val > 0:
            out[key] = timeout_val
    return out


def _parse_fidelity_schedule(raw: str | None) -> tuple[float, ...]:
    if not raw:
        return (0.25, 0.6, 1.0)
    values: list[float] = []
    for token in str(raw).split(","):
        try:
            value = float(token.strip())
        except Exception:
            continue
        if 0.0 < value <= 1.0:
            values.append(value)
    if not values:
        return (0.25, 0.6, 1.0)
    values = sorted(set(values))
    if values[-1] < 1.0:
        values.append(1.0)
    return tuple(values)


def _resolve_optuna_pruner_name(raw: str | None) -> str:
    pruner_name = str(raw or "median").strip().lower()
    if pruner_name in ("none", "off", "0"):
        return "none"
    if pruner_name in ("sha", "successive_halving", "halving"):
        return "successive_halving"
    return "median"


def _build_pruner(optuna):
    pruner_name = _resolve_optuna_pruner_name(os.getenv("SPOTIFY_OPTUNA_PRUNER", "median"))
    if pruner_name == "none":
        return optuna.pruners.NopPruner(), "none"
    if pruner_name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2), "successive_halving"
    startup_trials = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_STARTUP_TRIALS"), 5)
    warmup_steps = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_WARMUP_STEPS"), 1)
    return (
        optuna.pruners.MedianPruner(
            n_startup_trials=startup_trials,
            n_warmup_steps=warmup_steps,
        ),
        "median",
    )


def _resolve_optuna_worker_plan(
    model_count: int,
    requested_trial_jobs: int,
    requested_model_workers: int,
) -> tuple[int, int]:
    trial_jobs = max(1, requested_trial_jobs)
    model_workers = max(1, min(max(1, model_count), requested_model_workers))
    if model_workers <= 1:
        return 1, trial_jobs
    return model_workers, max(1, trial_jobs // model_workers)


def _build_optuna_cache_payload(
    *,
    cache_fingerprint: str,
    model_name: str,
    random_seed: int,
    trials: int,
    max_train_samples: int,
    max_eval_samples: int,
    model_timeout_seconds: int,
    per_trial_timeout_seconds: int,
    fidelity_schedule: tuple[float, ...],
    pruner_name: str,
) -> dict[str, object]:
    return {
        "schema_version": OPTUNA_CACHE_SCHEMA_VERSION,
        "prepared_fingerprint": str(cache_fingerprint).strip(),
        "model_name": model_name,
        "random_seed": int(random_seed),
        "trials": int(trials),
        "max_train_samples": int(max_train_samples),
        "max_eval_samples": int(max_eval_samples),
        "model_timeout_seconds": int(model_timeout_seconds),
        "per_trial_timeout_seconds": int(per_trial_timeout_seconds),
        "fidelity_schedule": [float(value) for value in fidelity_schedule],
        "pruner_name": pruner_name,
        "median_startup_trials": _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_STARTUP_TRIALS"), 5)
        if pruner_name == "median"
        else 0,
        "median_warmup_steps": _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_WARMUP_STEPS"), 1)
        if pruner_name == "median"
        else 0,
    }


def _build_optuna_cache_key(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _resolve_optuna_model_cache_paths(
    cache_root: Path,
    cache_fingerprint: str,
    model_name: str,
    cache_key: str,
) -> OptunaModelCachePaths:
    cache_dir = (cache_root / cache_fingerprint / model_name / cache_key).resolve()
    return OptunaModelCachePaths(
        cache_key=cache_key,
        cache_dir=cache_dir,
        result_path=cache_dir / "result.json",
        metadata_path=cache_dir / "cache_meta.json",
        trial_log_path=cache_dir / f"optuna_trials_{model_name}.csv",
        history_plot_path=cache_dir / f"optuna_history_{model_name}.png",
        estimator_artifact_path=cache_dir / "estimators" / f"classical_tuned_{model_name}.joblib",
        prediction_bundle_path=cache_dir / "prediction_bundles" / f"classical_tuned_{model_name}.npz",
    )


def _copy_optional_artifact(source: Path, destination: Path) -> str:
    if not source.exists():
        return ""
    return str(copy_file_if_changed(source, destination))


def _find_optuna_warm_start_candidate(
    *,
    cache_root: Path | None,
    current_cache_key: str,
    cache_fingerprint: str,
    model_name: str,
    max_train_samples: int,
    max_eval_samples: int,
    per_trial_timeout_seconds: int,
    fidelity_schedule: tuple[float, ...],
    pruner_name: str,
) -> OptunaWarmStartCandidate | None:
    if cache_root is None or not cache_root.exists() or not _optuna_warm_start_enabled_from_env():
        return None

    best_candidate: OptunaWarmStartCandidate | None = None
    for meta_path in cache_root.glob(f"*/{model_name}/*/cache_meta.json"):
        payload = safe_read_json(meta_path, default=None)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("schema_version", "")) != OPTUNA_CACHE_SCHEMA_VERSION:
            continue
        if str(payload.get("model_name", "")) != model_name:
            continue
        if int(payload.get("max_train_samples", -1)) != int(max_train_samples):
            continue
        if int(payload.get("max_eval_samples", -1)) != int(max_eval_samples):
            continue
        if int(payload.get("per_trial_timeout_seconds", -1)) != int(per_trial_timeout_seconds):
            continue
        payload_schedule = tuple(float(value) for value in payload.get("fidelity_schedule", []) or [])
        if payload_schedule != tuple(float(value) for value in fidelity_schedule):
            continue
        if str(payload.get("pruner_name", "")) != str(pruner_name):
            continue

        candidate_cache_key = str(payload.get("cache_key", "")).strip() or meta_path.parent.name
        if candidate_cache_key == current_cache_key:
            continue

        result_payload = safe_read_json(meta_path.parent / "result.json", default=None)
        if not isinstance(result_payload, dict):
            continue
        result = result_payload.get("result", result_payload)
        if not isinstance(result, dict):
            continue
        best_params = result.get("best_params")
        if not isinstance(best_params, dict) or not best_params:
            continue
        try:
            val_top1 = float(result.get("val_top1", float("nan")))
        except Exception:
            val_top1 = float("nan")
        try:
            modified_time = float((meta_path.parent / "result.json").stat().st_mtime)
        except Exception:
            modified_time = 0.0
        candidate = OptunaWarmStartCandidate(
            cache_fingerprint=str(payload.get("prepared_fingerprint", "")).strip(),
            cache_key=candidate_cache_key,
            best_params={str(key): value for key, value in best_params.items()},
            val_top1=val_top1,
            modified_time=modified_time,
        )
        if best_candidate is None:
            best_candidate = candidate
            continue
        candidate_rank = (
            int(candidate.cache_fingerprint == str(cache_fingerprint).strip()),
            int(np.isfinite(candidate.val_top1)),
            float(candidate.val_top1 if np.isfinite(candidate.val_top1) else float("-inf")),
            float(candidate.modified_time),
        )
        best_rank = (
            int(best_candidate.cache_fingerprint == str(cache_fingerprint).strip()),
            int(np.isfinite(best_candidate.val_top1)),
            float(best_candidate.val_top1 if np.isfinite(best_candidate.val_top1) else float("-inf")),
            float(best_candidate.modified_time),
        )
        if candidate_rank > best_rank:
            best_candidate = candidate
    return best_candidate


def _resolve_optuna_warm_start_trials(requested_trials: int) -> int:
    fraction_raw = os.getenv("SPOTIFY_OPTUNA_WARM_START_TRIAL_FRACTION", "0.60").strip()
    min_trials = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_WARM_START_MIN_TRIALS"), 4)
    try:
        fraction = float(fraction_raw)
    except Exception:
        fraction = 0.60
    fraction = min(1.0, max(0.10, fraction))
    reduced_trials = int(np.ceil(float(requested_trials) * fraction))
    return max(1, max(min_trials, reduced_trials))


def _load_cached_optuna_result(
    *,
    cache_paths: OptunaModelCachePaths,
    output_dir: Path,
    model_name: str,
    logger,
) -> OptunaTuningResult | None:
    payload = safe_read_json(cache_paths.result_path, default=None)
    if not isinstance(payload, dict):
        return None
    result_payload = payload.get("result", payload)
    if not isinstance(result_payload, dict):
        return None

    try:
        if cache_paths.trial_log_path.exists():
            copy_file_if_changed(cache_paths.trial_log_path, output_dir / f"optuna_trials_{model_name}.csv")
        if cache_paths.history_plot_path.exists():
            copy_file_if_changed(cache_paths.history_plot_path, output_dir / f"optuna_history_{model_name}.png")
        estimator_artifact_path = _copy_optional_artifact(
            cache_paths.estimator_artifact_path,
            output_dir / "estimators" / f"classical_tuned_{model_name}.joblib",
        )
        prediction_bundle_path = _copy_optional_artifact(
            cache_paths.prediction_bundle_path,
            output_dir / "prediction_bundles" / f"classical_tuned_{model_name}.npz",
        )
        hydrated_payload = dict(result_payload)
        hydrated_payload["estimator_artifact_path"] = estimator_artifact_path
        hydrated_payload["prediction_bundle_path"] = prediction_bundle_path
        return OptunaTuningResult(**hydrated_payload)
    except Exception as exc:
        logger.warning("Optuna cache load failed for %s (%s). Rebuilding.", model_name, exc)
        return None


def _save_optuna_result_to_cache(
    *,
    cache_paths: OptunaModelCachePaths,
    cache_payload: dict[str, object],
    result: OptunaTuningResult,
    output_dir: Path,
    model_name: str,
    logger,
) -> None:
    try:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        trial_log_path = output_dir / f"optuna_trials_{model_name}.csv"
        history_plot_path = output_dir / f"optuna_history_{model_name}.png"
        if trial_log_path.exists():
            copy_file_if_changed(trial_log_path, cache_paths.trial_log_path)
        if history_plot_path.exists():
            copy_file_if_changed(history_plot_path, cache_paths.history_plot_path)
        if result.estimator_artifact_path:
            source_estimator_path = Path(result.estimator_artifact_path)
            if source_estimator_path.exists():
                copy_file_if_changed(source_estimator_path, cache_paths.estimator_artifact_path)
        if result.prediction_bundle_path:
            source_bundle_path = Path(result.prediction_bundle_path)
            if source_bundle_path.exists():
                copy_file_if_changed(source_bundle_path, cache_paths.prediction_bundle_path)

        stored_result = asdict(result)
        stored_result["estimator_artifact_path"] = ""
        stored_result["prediction_bundle_path"] = ""
        write_json(
            cache_paths.result_path,
            {
                "cache_schema_version": OPTUNA_CACHE_SCHEMA_VERSION,
                "result": stored_result,
            },
            sort_keys=True,
        )
        write_json(
            cache_paths.metadata_path,
            {
                **cache_payload,
                "cache_key": cache_paths.cache_key,
            },
            sort_keys=True,
        )
    except Exception as exc:
        logger.warning("Optuna cache save failed for %s (%s).", model_name, exc)


def _plot_study_history(values: list[float], output_path: Path, title: str) -> None:
    if not values:
        return
    import matplotlib.pyplot as plt

    best_so_far: list[float] = []
    current_best = float("-inf")
    for value in values:
        current_best = max(current_best, value)
        best_so_far.append(current_best)

    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, values, marker="o", alpha=0.5, label="Trial score")
    ax.plot(x, best_so_far, linewidth=2, label="Best score")
    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation Top-1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_trial_log(study, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["trial", "state", "value", "duration_s", "params_json"])
        writer.writeheader()
        for trial in study.trials:
            writer.writerow(
                {
                    "trial": trial.number,
                    "state": str(trial.state),
                    "value": trial.value if trial.value is not None else "",
                    "duration_s": trial.duration.total_seconds() if trial.duration else "",
                    "params_json": json.dumps(trial.params, sort_keys=True),
                }
            )


def run_optuna_tuning(
    data: PreparedData,
    output_dir: Path,
    selected_models: tuple[str, ...],
    random_seed: int,
    trials: int,
    timeout_seconds: int,
    max_train_samples: int,
    max_eval_samples: int,
    logger,
    feature_bundle: ClassicalFeatureBundle | None = None,
    cache_root: Path | None = None,
    cache_fingerprint: str = "",
    cache_stats_out: dict[str, object] | None = None,
) -> list[OptunaTuningResult]:
    if trials <= 0:
        logger.info("Skipping Optuna tuning because trials <= 0.")
        return []

    validate_classical_models(selected_models, random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = tuple(selected_models)

    per_trial_timeout_seconds = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS"), 0)
    model_timeout_default = timeout_seconds if timeout_seconds > 0 else 0
    model_timeout_default = _parse_positive_int(
        os.getenv("SPOTIFY_OPTUNA_MODEL_TIMEOUT_SECONDS"),
        model_timeout_default,
    )
    model_timeout_overrides = _parse_model_timeout_overrides(os.getenv("SPOTIFY_OPTUNA_MODEL_TIMEOUTS"))
    fidelity_schedule = _parse_fidelity_schedule(os.getenv("SPOTIFY_OPTUNA_PRUNING_FIDELITIES"))
    pruner_name = _resolve_optuna_pruner_name(os.getenv("SPOTIFY_OPTUNA_PRUNER", "median"))

    cache_enabled = _optuna_cache_enabled_from_env() and cache_root is not None and bool(str(cache_fingerprint).strip())
    cache_contexts: dict[str, tuple[OptunaModelCachePaths, dict[str, object]]] = {}
    cached_results_by_model: dict[str, OptunaTuningResult] = {}
    uncached_models: list[str] = []

    for model_name in selected_models:
        model_timeout_seconds = model_timeout_overrides.get(model_name, model_timeout_default)
        if cache_enabled and cache_root is not None:
            cache_payload = _build_optuna_cache_payload(
                cache_fingerprint=cache_fingerprint,
                model_name=model_name,
                random_seed=random_seed,
                trials=trials,
                max_train_samples=max_train_samples,
                max_eval_samples=max_eval_samples,
                model_timeout_seconds=model_timeout_seconds,
                per_trial_timeout_seconds=per_trial_timeout_seconds,
                fidelity_schedule=fidelity_schedule,
                pruner_name=pruner_name,
            )
            cache_key = _build_optuna_cache_key(cache_payload)
            cache_paths = _resolve_optuna_model_cache_paths(
                cache_root=cache_root,
                cache_fingerprint=cache_fingerprint,
                model_name=model_name,
                cache_key=cache_key,
            )
            cache_contexts[model_name] = (cache_paths, cache_payload)
            cached_result = _load_cached_optuna_result(
                cache_paths=cache_paths,
                output_dir=output_dir,
                model_name=model_name,
                logger=logger,
            )
            if cached_result is not None:
                cached_results_by_model[model_name] = cached_result
                continue
        uncached_models.append(model_name)

    warm_start_candidates = {
        model_name: _find_optuna_warm_start_candidate(
            cache_root=cache_root,
            current_cache_key=cache_contexts[model_name][0].cache_key if model_name in cache_contexts else "",
            cache_fingerprint=cache_fingerprint,
            model_name=model_name,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            per_trial_timeout_seconds=per_trial_timeout_seconds,
            fidelity_schedule=fidelity_schedule,
            pruner_name=pruner_name,
        )
        for model_name in uncached_models
    }
    warm_start_model_names = [name for name, candidate in warm_start_candidates.items() if candidate is not None]

    if cache_stats_out is not None:
        cache_stats_out.clear()
        cache_stats_out.update(
            {
                "enabled": bool(cache_enabled),
                "fingerprint": (str(cache_fingerprint).strip() if cache_enabled else ""),
                "hit_model_names": [name for name in selected_models if name in cached_results_by_model],
                "miss_model_names": list(uncached_models),
            }
        )
        if warm_start_model_names:
            cache_stats_out["warm_start_model_names"] = list(warm_start_model_names)

    logger.info(
        "Optuna cache status: enabled=%s fingerprint=%s hits=%d misses=%d",
        cache_enabled,
        (cache_fingerprint if cache_enabled else "disabled"),
        len(cached_results_by_model),
        len(uncached_models),
    )
    if warm_start_model_names:
        logger.info("Optuna warm-start candidates: %s", ", ".join(warm_start_model_names))

    if not uncached_models:
        ordered_results = [cached_results_by_model[name] for name in selected_models if name in cached_results_by_model]
        write_json(output_dir / "optuna_results.json", [asdict(result) for result in ordered_results])
        return ordered_results

    optuna = _load_optuna()
    if optuna is None:
        logger.warning(
            "Optuna is not installed; skipping hyperparameter tuning for uncached models: %s",
            ", ".join(uncached_models),
        )
        ordered_results = [cached_results_by_model[name] for name in selected_models if name in cached_results_by_model]
        write_json(output_dir / "optuna_results.json", [asdict(result) for result in ordered_results])
        return ordered_results

    bundle = feature_bundle if feature_bundle is not None else build_classical_feature_bundle(data)
    X_train, X_val, X_test = bundle.X_train, bundle.X_val, bundle.X_test
    y_train, y_val, y_test = bundle.y_train, bundle.y_val, bundle.y_test
    X_val_full = X_val
    X_test_full = X_test
    X_train_seq = bundle.X_train_seq
    X_val_seq = bundle.X_val_seq
    X_test_seq = bundle.X_test_seq
    X_val_seq_full = X_val_seq
    X_test_seq_full = X_test_seq

    rng = np.random.default_rng(random_seed)
    train_idx = sample_indices(len(X_train), max_train_samples, rng)
    val_idx = sample_indices(len(X_val), max_eval_samples, rng)
    test_idx = sample_indices(len(X_test), max_eval_samples, rng)

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_val = X_val[val_idx]
    y_val = y_val[val_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]
    X_train_seq = X_train_seq[train_idx]
    X_val_seq = X_val_seq[val_idx]
    X_test_seq = X_test_seq[test_idx]

    logger.info(
        "Optuna tuning dataset sizes: train=%d, val=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    prediction_output_dir = output_dir / "prediction_bundles"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    estimator_output_dir = output_dir / "estimators"
    estimator_output_dir.mkdir(parents=True, exist_ok=True)
    workers, estimator_n_jobs = resolve_classical_parallelism()
    requested_optuna_jobs = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_JOBS"), 1)
    requested_model_workers = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_MODEL_WORKERS"), 1)
    model_workers, optuna_jobs = _resolve_optuna_worker_plan(
        len(uncached_models),
        requested_optuna_jobs,
        requested_model_workers,
    )
    if optuna_jobs > 1 or model_workers > 1:
        estimator_n_jobs = 1
    pruner, pruner_name = _build_pruner(optuna)

    logger.info(
        "Optuna parallelism: model_workers=%d trial_jobs=%d requested_trial_jobs=%d estimator_n_jobs=%d (classical_workers=%d) pruner=%s fidelity=%s trial_timeout_s=%d",
        model_workers,
        optuna_jobs,
        requested_optuna_jobs,
        estimator_n_jobs,
        workers,
        pruner_name,
        ",".join(f"{val:.2f}" for val in fidelity_schedule),
        per_trial_timeout_seconds,
    )

    def run_model_study(model_name: str) -> OptunaTuningResult | None:
        model_timeout = model_timeout_overrides.get(model_name, model_timeout_default)
        warm_start_candidate = warm_start_candidates.get(model_name)
        effective_trials = int(trials)
        if warm_start_candidate is not None:
            effective_trials = min(int(trials), _resolve_optuna_warm_start_trials(trials))
        logger.info(
            "Running Optuna tuning for %s (%d trials, timeout_s=%s warm_start=%s)",
            model_name,
            effective_trials,
            (str(model_timeout) if model_timeout > 0 else "none"),
            ("yes" if warm_start_candidate is not None else "no"),
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_seed),
            study_name=f"{model_name}_tuning",
            pruner=pruner,
        )
        if warm_start_candidate is not None:
            study.enqueue_trial(dict(warm_start_candidate.best_params))
            logger.info(
                "Warm-starting Optuna for %s from prior fingerprint=%s val_top1=%s",
                model_name,
                warm_start_candidate.cache_fingerprint or "unknown",
                (
                    f"{warm_start_candidate.val_top1:.4f}"
                    if np.isfinite(warm_start_candidate.val_top1)
                    else "unknown"
                ),
            )

        def objective(trial):
            params = _suggest_params(trial, model_name)
            trial_started = time.perf_counter()
            last_score = float("nan")
            X_train_model = X_train_seq if model_name == "session_knn" else X_train
            X_val_model = X_val_seq if model_name == "session_knn" else X_val
            for step_idx, fraction in enumerate(fidelity_schedule, start=1):
                if per_trial_timeout_seconds > 0:
                    elapsed = time.perf_counter() - trial_started
                    if elapsed > float(per_trial_timeout_seconds):
                        raise optuna.TrialPruned(f"trial timeout exceeded ({elapsed:.1f}s)")

                stage_rows = max(512, min(len(X_train_model), int(round(len(X_train_model) * fraction))))
                if stage_rows >= len(X_train_model):
                    X_stage = X_train_model
                    y_stage = y_train
                else:
                    # Use a rotating contiguous slice to avoid large per-trial copy buffers.
                    max_offset = max(0, len(X_train_model) - stage_rows)
                    offset = ((trial.number + 1) * 9973 + step_idx * 131) % (max_offset + 1)
                    X_stage = X_train_model[offset : offset + stage_rows]
                    y_stage = y_train[offset : offset + stage_rows]

                _, estimator = build_classical_estimator(
                    model_name,
                    random_seed,
                    params=params,
                    estimator_n_jobs=estimator_n_jobs,
                )
                try:
                    estimator.fit(X_stage, y_stage)
                except ValueError as exc:
                    # HistGradientBoosting can error on sampled subsets with extremely
                    # rare classes when it attempts stratified internal splits.
                    msg = str(exc).lower()
                    if "least populated classes" in msg or "minimum number of groups" in msg:
                        raise optuna.TrialPruned(f"insufficient class support in sampled stage: {exc}") from exc
                    raise
                val_pred = estimator.predict(X_val_model)
                last_score = float(np.mean(val_pred == y_val))
                trial.report(last_score, step=step_idx)
                if step_idx < len(fidelity_schedule) and trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at step={step_idx} score={last_score:.4f}")
            return last_score

        study.optimize(
            objective,
            n_trials=effective_trials,
            timeout=(None if model_timeout <= 0 else model_timeout),
            show_progress_bar=False,
            n_jobs=optuna_jobs,
        )

        complete_trials = [
            trial
            for trial in study.trials
            if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not complete_trials:
            _write_trial_log(study, output_dir / f"optuna_trials_{model_name}.csv")
            logger.warning(
                "No completed Optuna trials for %s (all pruned/failed). Skipping tuned fit.",
                model_name,
            )
            return None

        best_params = dict(study.best_trial.params)
        family, estimator = build_classical_estimator(
            model_name,
            random_seed,
            params=best_params,
            estimator_n_jobs=estimator_n_jobs,
        )
        started = time.perf_counter()
        X_train_model = X_train_seq if model_name == "session_knn" else X_train
        X_val_model = X_val_seq if model_name == "session_knn" else X_val
        X_test_model = X_test_seq if model_name == "session_knn" else X_test
        X_val_full_model = X_val_seq_full if model_name == "session_knn" else X_val_full
        X_test_full_model = X_test_seq_full if model_name == "session_knn" else X_test_full
        estimator.fit(X_train_model, y_train)
        fit_seconds = float(time.perf_counter() - started)
        estimator_artifact_path = ""
        try:
            import joblib

            estimator_path = estimator_output_dir / f"classical_tuned_{model_name}.joblib"
            joblib.dump(estimator, estimator_path, compress=3)
            estimator_artifact_path = str(estimator_path)
        except Exception:
            estimator_artifact_path = ""
        val_top1, val_top5, test_top1, test_top5, val_ranking, test_ranking = evaluate_classical_estimator(
            estimator,
            X_val_model,
            y_val,
            X_test_model,
            y_test,
        )
        prediction_bundle_path = ""
        aligned_probs = collect_aligned_probabilities(
            estimator,
            X_val_full_model,
            X_test_full_model,
            num_classes=data.num_artists,
        )
        if aligned_probs is not None:
            val_proba, test_proba = aligned_probs
            bundle_path = save_prediction_bundle(
                prediction_output_dir / f"classical_tuned_{model_name}.npz",
                val_proba=val_proba,
                test_proba=test_proba,
            )
            prediction_bundle_path = str(bundle_path)

        tuned_name = f"{model_name}_optuna"
        result = OptunaTuningResult(
            model_name=tuned_name,
            base_model_name=model_name,
            model_family=family,
            fit_seconds=fit_seconds,
            val_top1=val_top1,
            val_top5=val_top5,
            val_ndcg_at5=float(val_ranking["ndcg_at5"]),
            val_mrr_at5=float(val_ranking["mrr_at5"]),
            val_coverage_at5=float(val_ranking["coverage_at5"]),
            val_diversity_at5=float(val_ranking["diversity_at5"]),
            test_top1=test_top1,
            test_top5=test_top5,
            test_ndcg_at5=float(test_ranking["ndcg_at5"]),
            test_mrr_at5=float(test_ranking["mrr_at5"]),
            test_coverage_at5=float(test_ranking["coverage_at5"]),
            test_diversity_at5=float(test_ranking["diversity_at5"]),
            n_trials=len(study.trials),
            best_params=best_params,
            prediction_bundle_path=prediction_bundle_path,
            estimator_artifact_path=estimator_artifact_path,
        )
        _write_trial_log(study, output_dir / f"optuna_trials_{model_name}.csv")
        values = [float(t.value) for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
        _plot_study_history(
            values=values,
            output_path=output_dir / f"optuna_history_{model_name}.png",
            title=f"Optuna Search: {model_name}",
        )
        logger.info(
            "[OPTUNA] %s best val_top1=%.4f test_top1=%.4f",
            model_name,
            val_top1,
            test_top1,
        )
        if cache_enabled and model_name in cache_contexts:
            cache_paths, cache_payload = cache_contexts[model_name]
            _save_optuna_result_to_cache(
                cache_paths=cache_paths,
                cache_payload=cache_payload,
                result=result,
                output_dir=output_dir,
                model_name=model_name,
                logger=logger,
            )
        return result

    fresh_results_by_model: dict[str, OptunaTuningResult] = {}
    if model_workers > 1 and len(uncached_models) > 1:
        ordered_results: list[OptunaTuningResult | None] = [None] * len(uncached_models)
        with ThreadPoolExecutor(max_workers=model_workers) as executor:
            futures = {
                executor.submit(run_model_study, model_name): (idx, model_name)
                for idx, model_name in enumerate(uncached_models)
            }
            for future in as_completed(futures):
                idx, model_name = futures[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as exc:
                    logger.warning("Optuna tuning failed for %s: %s", model_name, exc)
        fresh_results_by_model = {
            model_name: result
            for model_name, result in zip(uncached_models, ordered_results)
            if result is not None
        }
    else:
        for model_name in uncached_models:
            result = run_model_study(model_name)
            if result is not None:
                fresh_results_by_model[model_name] = result

    results = [
        cached_results_by_model.get(model_name) or fresh_results_by_model.get(model_name)
        for model_name in selected_models
    ]
    ordered_results = [result for result in results if result is not None]
    summary_payload = [asdict(result) for result in ordered_results]

    write_json(output_dir / "optuna_results.json", summary_payload)

    return ordered_results
