from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .run_artifacts import write_json


SUPPORTED_DEEP_OPTUNA_MODELS: tuple[str, ...] = ("sasrec", "bert4rec", "srgnn")


@dataclass(frozen=True)
class DeepOptunaResult:
    model_name: str
    best_value: float
    best_params: dict[str, object]
    n_trials: int
    fit_seconds: float


def suggest_deep_model_params(trial, model_name: str) -> dict[str, object]:
    if model_name == "sasrec":
        return {
            "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 96, 128]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
            "feed_forward_dim": trial.suggest_categorical("feed_forward_dim", [128, 256, 384]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.05, 0.3),
            "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        }
    if model_name == "bert4rec":
        return {
            "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 96, 128]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
            "feed_forward_multiplier": trial.suggest_int("feed_forward_multiplier", 2, 4),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.05, 0.3),
            "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        }
    if model_name == "srgnn":
        return {
            "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 96, 128]),
            "context_dim": trial.suggest_categorical("context_dim", [32, 64, 96]),
            "fusion_dim": trial.suggest_categorical("fusion_dim", [128, 192, 256]),
        }
    raise ValueError(
        f"Unsupported deep Optuna model: {model_name}. "
        f"Known models: {', '.join(SUPPORTED_DEEP_OPTUNA_MODELS)}"
    )


def build_tunable_deep_model(
    model_name: str,
    *,
    sequence_length: int,
    num_artists: int,
    num_ctx: int,
    params: dict[str, object],
):
    if model_name == "sasrec":
        from .sasrec_model import build_sasrec_model

        return build_sasrec_model(sequence_length, num_artists, num_ctx, params)
    if model_name == "bert4rec":
        from .bert4rec_model import build_bert4rec_model

        return build_bert4rec_model(sequence_length, num_artists, num_ctx, params)
    if model_name == "srgnn":
        from .srgnn_model import build_srgnn_model

        return build_srgnn_model(sequence_length, num_artists, num_ctx, params)
    raise ValueError(f"Unsupported deep Optuna model: {model_name}")


def run_deep_optuna_tuning(
    *,
    data,
    selected_models: tuple[str, ...],
    trials: int,
    epochs: int,
    max_train_rows: int,
    max_val_rows: int,
    random_seed: int,
    output_dir: Path,
    logger,
) -> list[DeepOptunaResult]:
    if trials <= 0 or not selected_models:
        return []
    unknown = [name for name in selected_models if name not in SUPPORTED_DEEP_OPTUNA_MODELS]
    if unknown:
        raise ValueError(f"Unsupported deep Optuna models: {', '.join(unknown)}")

    try:
        import optuna
        import tensorflow as tf
    except ImportError as exc:
        logger.warning("Deep Optuna tuning skipped because a dependency is unavailable: %s", exc)
        return []

    rng = np.random.default_rng(random_seed)

    def select_rows(total: int, maximum: int) -> np.ndarray:
        if maximum <= 0 or total <= maximum:
            return np.arange(total, dtype="int32")
        return np.sort(rng.choice(total, size=maximum, replace=False).astype("int32"))

    train_idx = select_rows(len(data.y_train), max_train_rows)
    val_idx = select_rows(len(data.y_val), max_val_rows)
    train_inputs = {
        "seq_input": np.asarray(data.X_seq_train[train_idx], dtype="int32"),
        "ctx_input": np.asarray(data.X_ctx_train[train_idx], dtype="float32"),
    }
    train_targets = np.asarray(data.y_train[train_idx], dtype="int32")
    val_inputs = {
        "seq_input": np.asarray(data.X_seq_val[val_idx], dtype="int32"),
        "ctx_input": np.asarray(data.X_ctx_val[val_idx], dtype="float32"),
    }
    val_targets = np.asarray(data.y_val[val_idx], dtype="int32")
    results: list[DeepOptunaResult] = []

    for model_name in selected_models:
        started = time.perf_counter()

        def objective(trial):
            tf.keras.backend.clear_session()
            params = suggest_deep_model_params(trial, model_name)
            model = build_tunable_deep_model(
                model_name,
                sequence_length=int(data.X_seq_train.shape[1]),
                num_artists=int(data.num_artists),
                num_ctx=int(data.num_ctx),
                params=params,
            )
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
                run_eagerly=True,
            )

            def train_epochs(inputs, targets, epoch_count: int, seed_offset: int) -> None:
                local_rng = np.random.default_rng(random_seed + int(trial.number) + seed_offset)
                for _epoch in range(max(1, int(epoch_count))):
                    order = local_rng.permutation(len(targets))
                    for start in range(0, len(order), 256):
                        selector = order[start : start + 256]
                        model.train_on_batch(
                            {
                                "seq_input": np.asarray(inputs["seq_input"])[selector],
                                "ctx_input": np.asarray(inputs["ctx_input"])[selector],
                            },
                            np.asarray(targets)[selector],
                        )

            if model_name == "bert4rec":
                from .bert4rec_model import build_cloze_pretraining_batch

                cloze = build_cloze_pretraining_batch(
                    train_inputs["seq_input"],
                    train_inputs["ctx_input"],
                    int(data.num_artists),
                    mask_probability=0.15,
                    seed=random_seed + int(trial.number),
                )
                train_epochs(
                    {"seq_input": cloze.seq_input, "ctx_input": cloze.ctx_input},
                    cloze.artist_output,
                    1,
                    101,
                )
            train_epochs(train_inputs, train_targets, epochs, 211)
            val_proba = np.asarray(model(val_inputs, training=False))
            value = float(np.mean(np.argmax(val_proba, axis=1) == val_targets))
            trial.report(value, step=max(1, int(epochs)))
            return value

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=min(3, trials)),
            study_name=f"deep_{model_name}",
        )
        study.optimize(objective, n_trials=int(trials), n_jobs=1)
        results.append(
            DeepOptunaResult(
                model_name=model_name,
                best_value=float(study.best_value),
                best_params=dict(study.best_params),
                n_trials=len(study.trials),
                fit_seconds=float(time.perf_counter() - started),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "deep_optuna_results.json",
        [
            {
                "model_name": result.model_name,
                "best_value": result.best_value,
                "best_params": result.best_params,
                "n_trials": result.n_trials,
                "fit_seconds": result.fit_seconds,
            }
            for result in results
        ],
    )
    return results


__all__ = [
    "DeepOptunaResult",
    "SUPPORTED_DEEP_OPTUNA_MODELS",
    "build_tunable_deep_model",
    "run_deep_optuna_tuning",
    "suggest_deep_model_params",
]
