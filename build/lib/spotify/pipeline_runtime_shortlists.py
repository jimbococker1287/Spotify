from __future__ import annotations

import os


def _resolve_shortlist_top_n(env_name: str) -> int:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return 0
    try:
        return max(0, int(raw_value))
    except Exception:
        return 0


def _shortlist_classical_model_names(
    candidate_names: tuple[str, ...],
    classical_results,
    *,
    top_n: int,
    logger,
    stage_label: str,
    passthrough_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if top_n <= 0 or len(candidate_names) <= top_n:
        return candidate_names
    if not classical_results:
        return candidate_names

    passthrough = {str(name).strip() for name in (passthrough_names or ()) if str(name).strip()}

    metrics_by_name: dict[str, tuple[float, float]] = {}
    for row in classical_results:
        model_name = str(getattr(row, "model_name", "")).strip()
        if not model_name:
            continue
        try:
            val_top1 = float(getattr(row, "val_top1"))
            fit_seconds = float(getattr(row, "fit_seconds"))
        except Exception:
            continue
        metrics_by_name[model_name] = (val_top1, fit_seconds)

    missing = [name for name in candidate_names if name not in metrics_by_name and name not in passthrough]
    if missing:
        logger.info(
            "Skipping %s shortlist because baseline metrics are missing for: %s",
            stage_label,
            ", ".join(missing),
        )
        return candidate_names

    eligible_names = tuple(name for name in candidate_names if name in metrics_by_name)
    if len(eligible_names) <= top_n:
        return candidate_names

    ranked_names = tuple(
        sorted(
            eligible_names,
            key=lambda name: (
                -metrics_by_name[name][0],
                metrics_by_name[name][1],
                name,
            ),
        )[:top_n]
    )
    dropped_names = tuple(name for name in eligible_names if name not in ranked_names)
    if not dropped_names:
        return candidate_names
    passthrough_selected = tuple(name for name in candidate_names if name in passthrough)
    selected_names = tuple((*ranked_names, *passthrough_selected))
    logger.info(
        "%s shortlist selected %s from %s using baseline val_top1 (top_n=%d); dropped=%s",
        stage_label,
        ",".join(selected_names),
        ",".join(candidate_names),
        top_n,
        ",".join(dropped_names),
    )
    return selected_names


def _tuned_backtest_specs(
    candidate_names: tuple[str, ...],
    tuned_results,
    *,
    logger,
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    if not candidate_names or not tuned_results:
        return candidate_names, {}

    augmented: list[str] = []
    specs: dict[str, dict[str, object]] = {}
    added_aliases: list[str] = []
    for candidate_name in candidate_names:
        augmented.append(candidate_name)
        for row in tuned_results:
            base_name = str(getattr(row, "base_model_name", "")).strip()
            tuned_name = str(getattr(row, "model_name", "")).strip()
            best_params = getattr(row, "best_params", {})
            if base_name != candidate_name or not tuned_name or tuned_name == candidate_name:
                continue
            if not isinstance(best_params, dict) or not best_params:
                continue
            if tuned_name not in augmented:
                augmented.append(tuned_name)
            specs[tuned_name] = {
                "base_model_name": base_name,
                "best_params": dict(best_params),
            }
            added_aliases.append(tuned_name)
            break
    if added_aliases:
        logger.info("Temporal backtest added tuned challenger variants: %s", ",".join(added_aliases))
    return tuple(augmented), specs


__all__ = [
    "_resolve_shortlist_top_n",
    "_shortlist_classical_model_names",
    "_tuned_backtest_specs",
]
