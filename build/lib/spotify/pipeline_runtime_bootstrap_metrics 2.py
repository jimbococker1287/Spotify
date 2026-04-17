from __future__ import annotations


def log_prepared_cache_status(*, cache_info, logger) -> None:
    if cache_info.enabled:
        logger.info(
            "Prepared cache status: %s (fingerprint=%s)",
            ("HIT" if cache_info.hit else "MISS"),
            cache_info.fingerprint,
        )


def log_prepared_baseline_metrics(*, prepared, logger, tracker, compute_baselines) -> None:
    baseline_metrics = compute_baselines(prepared, logger)
    tracker.log_params(
        {
            "data_records": len(prepared.df),
            "num_artists": prepared.num_artists,
            "num_context_features": prepared.num_ctx,
            **baseline_metrics,
        }
    )


__all__ = [
    "log_prepared_baseline_metrics",
    "log_prepared_cache_status",
]
