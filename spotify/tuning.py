from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import os
import time

import numpy as np

from .benchmarks import (
    build_classical_estimator,
    build_tabular_features,
    evaluate_classical_estimator,
    resolve_classical_parallelism,
    sample_rows,
    validate_classical_models,
)
from .data import PreparedData


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


def _load_optuna():
    try:
        import optuna
    except Exception:
        return None
    return optuna


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
    if model_name == "mlp":
        return {
            "hidden_1": trial.suggest_int("hidden_1", 64, 384, step=32),
            "hidden_2": trial.suggest_int("hidden_2", 32, 256, step=32),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "max_iter": trial.suggest_int("max_iter", 40, 140, step=10),
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


def _build_pruner(optuna):
    pruner_name = os.getenv("SPOTIFY_OPTUNA_PRUNER", "median").strip().lower()
    if pruner_name in ("none", "off", "0"):
        return optuna.pruners.NopPruner(), "none"
    if pruner_name in ("sha", "successive_halving", "halving"):
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
) -> list[OptunaTuningResult]:
    if trials <= 0:
        logger.info("Skipping Optuna tuning because trials <= 0.")
        return []

    optuna = _load_optuna()
    if optuna is None:
        logger.warning("Optuna is not installed; skipping hyperparameter tuning.")
        return []

    validate_classical_models(selected_models, random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test = build_tabular_features(data)
    y_train = data.y_train.astype(int)
    y_val = data.y_val.astype(int)
    y_test = data.y_test.astype(int)

    rng = np.random.default_rng(random_seed)
    X_train, y_train = sample_rows(X_train, y_train, max_train_samples, rng)
    X_val, y_val = sample_rows(X_val, y_val, max_eval_samples, rng)
    X_test, y_test = sample_rows(X_test, y_test, max_eval_samples, rng)

    logger.info(
        "Optuna tuning dataset sizes: train=%d, val=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    results: list[OptunaTuningResult] = []
    summary_payload: list[dict[str, object]] = []
    workers, estimator_n_jobs = resolve_classical_parallelism()
    optuna_jobs = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_JOBS"), 1)
    if optuna_jobs > 1:
        estimator_n_jobs = 1
    per_trial_timeout_seconds = _parse_positive_int(os.getenv("SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS"), 0)
    model_timeout_default = timeout_seconds if timeout_seconds > 0 else 0
    model_timeout_default = _parse_positive_int(
        os.getenv("SPOTIFY_OPTUNA_MODEL_TIMEOUT_SECONDS"),
        model_timeout_default,
    )
    model_timeout_overrides = _parse_model_timeout_overrides(os.getenv("SPOTIFY_OPTUNA_MODEL_TIMEOUTS"))
    fidelity_schedule = _parse_fidelity_schedule(os.getenv("SPOTIFY_OPTUNA_PRUNING_FIDELITIES"))
    pruner, pruner_name = _build_pruner(optuna)

    logger.info(
        "Optuna parallelism: jobs=%d estimator_n_jobs=%d (classical_workers=%d) pruner=%s fidelity=%s trial_timeout_s=%d",
        optuna_jobs,
        estimator_n_jobs,
        workers,
        pruner_name,
        ",".join(f"{val:.2f}" for val in fidelity_schedule),
        per_trial_timeout_seconds,
    )

    for model_name in selected_models:
        model_timeout = model_timeout_overrides.get(model_name, model_timeout_default)
        logger.info(
            "Running Optuna tuning for %s (%d trials, timeout_s=%s)",
            model_name,
            trials,
            (str(model_timeout) if model_timeout > 0 else "none"),
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_seed),
            study_name=f"{model_name}_tuning",
            pruner=pruner,
        )

        def objective(trial):
            params = _suggest_params(trial, model_name)
            trial_started = time.perf_counter()
            last_score = float("nan")
            for step_idx, fraction in enumerate(fidelity_schedule, start=1):
                if per_trial_timeout_seconds > 0:
                    elapsed = time.perf_counter() - trial_started
                    if elapsed > float(per_trial_timeout_seconds):
                        raise optuna.TrialPruned(f"trial timeout exceeded ({elapsed:.1f}s)")

                stage_rows = max(512, min(len(X_train), int(round(len(X_train) * fraction))))
                if stage_rows >= len(X_train):
                    X_stage = X_train
                    y_stage = y_train
                else:
                    stage_rng = np.random.default_rng(random_seed + (trial.number + 1) * 9973 + step_idx * 131)
                    idx = stage_rng.choice(len(X_train), size=stage_rows, replace=False)
                    X_stage = X_train[idx]
                    y_stage = y_train[idx]

                _, estimator = build_classical_estimator(
                    model_name,
                    random_seed,
                    params=params,
                    estimator_n_jobs=estimator_n_jobs,
                )
                estimator.fit(X_stage, y_stage)
                val_pred = estimator.predict(X_val)
                last_score = float(np.mean(val_pred == y_val))
                trial.report(last_score, step=step_idx)
                if step_idx < len(fidelity_schedule) and trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at step={step_idx} score={last_score:.4f}")
            return last_score

        study.optimize(
            objective,
            n_trials=trials,
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
            continue

        best_params = dict(study.best_trial.params)
        family, estimator = build_classical_estimator(
            model_name,
            random_seed,
            params=best_params,
            estimator_n_jobs=estimator_n_jobs,
        )
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = float(time.perf_counter() - started)
        val_top1, val_top5, test_top1, test_top5, val_ranking, test_ranking = evaluate_classical_estimator(
            estimator,
            X_val,
            y_val,
            X_test,
            y_test,
        )

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
        )
        results.append(result)
        summary_payload.append(asdict(result))

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

    with (output_dir / "optuna_results.json").open("w", encoding="utf-8") as out:
        json.dump(summary_payload, out, indent=2)

    return results
