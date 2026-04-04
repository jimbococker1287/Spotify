from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

from .champion_alias import best_serveable_model, write_champion_alias
from .pipeline_helpers import _load_current_risk_metrics, _write_json_artifact


@dataclass
class ChampionGateOutcome:
    champion_gate: dict[str, object]
    champion_alias_payload: dict[str, object]
    champion_model: tuple[str, str] | None
    strict_gate_error: str | None


def _resolve_gate_settings() -> dict[str, object]:
    champion_gate_threshold_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_REGRESSION", "0.005").strip()
    try:
        champion_gate_threshold = max(0.0, float(champion_gate_threshold_raw))
    except Exception:
        champion_gate_threshold = 0.005
    champion_gate_metric = os.getenv("SPOTIFY_CHAMPION_GATE_METRIC", "backtest_top1").strip().lower()
    champion_gate_match_profile_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MATCH_PROFILE", "1").strip().lower()
    champion_gate_match_profile = champion_gate_match_profile_raw in ("1", "true", "yes", "on")
    champion_gate_significance_raw = os.getenv("SPOTIFY_CHAMPION_GATE_SIGNIFICANCE", "0").strip().lower()
    champion_gate_require_significance = champion_gate_significance_raw in ("1", "true", "yes", "on")
    champion_gate_significance_z_raw = os.getenv("SPOTIFY_CHAMPION_GATE_SIGNIFICANCE_Z", "1.96").strip()
    try:
        champion_gate_significance_z = max(0.0, float(champion_gate_significance_z_raw))
    except Exception:
        champion_gate_significance_z = 1.96
    gate_max_selective_risk_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK", "").strip()
    gate_max_abstention_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE", "").strip()
    try:
        gate_max_selective_risk = float(gate_max_selective_risk_raw) if gate_max_selective_risk_raw else None
    except Exception:
        gate_max_selective_risk = None
    try:
        gate_max_abstention_rate = float(gate_max_abstention_raw) if gate_max_abstention_raw else None
    except Exception:
        gate_max_abstention_rate = None
    return {
        "threshold": champion_gate_threshold,
        "metric": champion_gate_metric,
        "match_profile": champion_gate_match_profile,
        "require_significance": champion_gate_require_significance,
        "significance_z": champion_gate_significance_z,
        "max_selective_risk": gate_max_selective_risk,
        "max_abstention_rate": gate_max_abstention_rate,
    }


def run_champion_gate_and_alias(
    *,
    artifact_paths: list[Path],
    backtest_rows: list[dict[str, object]],
    config: Any,
    evaluate_champion_gate: Any,
    history_csv: Path,
    history_dir: Path,
    logger,
    phase_recorder,
    prepared,
    result_rows: list[dict[str, object]],
    run_dir: Path,
    run_id: str,
) -> ChampionGateOutcome:
    champion_alias_payload: dict[str, object] = {
        "updated": False,
        "alias_file": "",
        "run_id": "",
        "run_dir": "",
        "model_name": "",
        "reason": "gate_not_evaluated",
    }
    settings = _resolve_gate_settings()
    current_risk_metrics = _load_current_risk_metrics(
        run_dir,
        result_rows,
        val_y=prepared.y_val,
        test_y=prepared.y_test,
        conformal_alpha=config.conformal_alpha,
    )

    with phase_recorder.phase(
        "champion_gate_and_alias",
        metric_source=settings["metric"],
        require_profile_match=settings["match_profile"],
        require_significant_lift=settings["require_significance"],
    ) as phase:
        champion_gate = evaluate_champion_gate(
            history_csv=history_csv,
            current_run_id=run_id,
            current_results=result_rows,
            regression_threshold=settings["threshold"],
            backtest_history_csv=(history_dir / "backtest_history.csv"),
            current_backtest_rows=backtest_rows,
            metric_source=settings["metric"],
            current_profile=config.profile,
            require_profile_match=settings["match_profile"],
            require_significant_lift=settings["require_significance"],
            significance_z=settings["significance_z"],
            current_risk_metrics=current_risk_metrics,
            max_selective_risk=settings["max_selective_risk"],
            max_abstention_rate=settings["max_abstention_rate"],
        )
        _write_json_artifact(run_dir / "champion_gate.json", champion_gate, artifact_paths)
        logger.info(
            "Champion gate: source=%s promoted=%s regression=%.6f threshold=%.6f",
            str(champion_gate.get("metric_source", settings["metric"])),
            bool(champion_gate.get("promoted", False)),
            float(champion_gate.get("regression", 0.0)),
            float(champion_gate.get("threshold", settings["threshold"])),
        )
        strict_gate_raw = os.getenv("SPOTIFY_CHAMPION_GATE_STRICT", "0").strip().lower()
        strict_gate = strict_gate_raw in ("1", "true", "yes", "on")
        strict_gate_error: str | None = None
        if strict_gate and not bool(champion_gate.get("promoted", False)):
            strict_gate_error = (
                "Champion gate failed in strict mode: "
                f"regression={champion_gate.get('regression')} threshold={champion_gate.get('threshold')}"
            )

        champion_model: tuple[str, str] | None = None
        if bool(champion_gate.get("promoted", False)):
            champion_model = best_serveable_model(result_rows, run_dir=run_dir)
            if not champion_model:
                champion_alias_payload["reason"] = "no_serveable_models_in_promoted_run"
                logger.info("Skipping champion alias update: promoted run has no serveable models.")
            else:
                champion_model_name, champion_model_type = champion_model
                alias_file = write_champion_alias(
                    output_dir=config.output_dir,
                    run_id=run_id,
                    run_dir=run_dir,
                    model_name=champion_model_name,
                    model_type=champion_model_type,
                )
                champion_alias_payload = {
                    "updated": True,
                    "alias_file": str(alias_file),
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "model_name": champion_model_name,
                    "model_type": champion_model_type,
                    "reason": "promoted",
                }
                artifact_paths.append(alias_file)
                logger.info(
                    "Champion alias updated: %s -> run_id=%s model=%s type=%s",
                    alias_file,
                    run_id,
                    champion_model_name,
                    champion_model_type,
                )
        else:
            champion_alias_payload["reason"] = "gate_not_promoted"
        phase["promoted"] = bool(champion_gate.get("promoted", False))
        phase["champion_alias_updated"] = bool(champion_alias_payload.get("updated", False))
        phase["strict_gate_enabled"] = bool(strict_gate)

    return ChampionGateOutcome(
        champion_gate=champion_gate,
        champion_alias_payload=champion_alias_payload,
        champion_model=champion_model,
        strict_gate_error=strict_gate_error,
    )


__all__ = ["ChampionGateOutcome", "run_champion_gate_and_alias"]
