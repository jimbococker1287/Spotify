from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np

from .champion_alias import preferred_serveable_model, write_champion_alias
from .control_room import write_control_room_report
from .governance import evaluate_champion_gate
from .pipeline_helpers import _load_current_risk_metrics
from .run_artifacts import collect_run_manifests, safe_read_csv, safe_read_json, write_json


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _safe_optional_float(value) -> float | None:
    metric = _safe_float(value)
    if not math.isfinite(metric):
        return None
    return float(metric)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.champion_gate_refresh",
        description="Recompute champion_gate.json for an existing run without retraining.",
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit outputs/runs/<run_id> directory.")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs root used for histories and run lookup.")
    parser.add_argument(
        "--skip-control-room",
        action="store_true",
        help="Do not refresh outputs/analytics/control_room.{json,md} after rewriting the gate.",
    )
    return parser.parse_args()


def _resolve_run_dir(outputs_dir: Path, run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    manifests = collect_run_manifests(outputs_dir)
    if not manifests:
        raise FileNotFoundError(f"No run manifests found under {outputs_dir / 'runs'}")
    selected = max(
        manifests,
        key=lambda row: (
            str(row.get("timestamp", "")),
            str(row.get("run_id", "")),
        ),
    )
    run_id = str(selected.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("Latest run manifest is missing a run_id.")
    run_dir = outputs_dir / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Latest run directory not found: {run_dir}")
    return run_dir.resolve()


def _load_prepared_targets(manifest: dict[str, object]) -> tuple[np.ndarray | None, np.ndarray | None]:
    cache = manifest.get("cache", {})
    if not isinstance(cache, dict):
        return None, None
    cache_path_raw = str(cache.get("cache_path", "")).strip()
    if not cache_path_raw:
        return None, None
    cache_path = Path(cache_path_raw).expanduser().resolve()
    if not cache_path.exists():
        return None, None
    try:
        payload = joblib.load(cache_path)
    except Exception:
        return None, None

    prepared = payload.get("prepared") if isinstance(payload, dict) else payload
    y_val = getattr(prepared, "y_val", None)
    y_test = getattr(prepared, "y_test", None)
    if y_val is None or y_test is None:
        return None, None
    return np.asarray(y_val), np.asarray(y_test)


def _load_run_backtest_rows(outputs_dir: Path, run_id: str, *, run_dir: Path | None = None) -> list[dict[str, object]]:
    if run_dir is not None:
        local_backtest = safe_read_csv(run_dir / "backtest" / "temporal_backtest.csv")
        if not local_backtest.empty:
            return local_backtest.to_dict(orient="records")
    frame = safe_read_csv(outputs_dir / "history" / "backtest_history.csv")
    if frame.empty or "run_id" not in frame.columns:
        return []
    run_rows = frame.loc[frame["run_id"].fillna("").astype(str) == str(run_id)].copy()
    if run_rows.empty:
        return []
    return run_rows.to_dict(orient="records")


def refresh_champion_gate(
    *,
    outputs_dir: Path,
    run_dir: Path,
    refresh_control_room: bool = True,
) -> dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    gate_path = run_dir / "champion_gate.json"
    run_results_path = run_dir / "run_results.json"
    manifest = safe_read_json(manifest_path, default={})
    if not isinstance(manifest, dict) or not manifest:
        raise FileNotFoundError(f"run_manifest.json not found or unreadable: {manifest_path}")
    current_results = safe_read_json(run_results_path, default=[])
    if not isinstance(current_results, list) or not current_results:
        raise FileNotFoundError(f"run_results.json not found or unreadable: {run_results_path}")

    existing_gate = safe_read_json(gate_path, default={})
    if not isinstance(existing_gate, dict) or not existing_gate:
        gate_from_manifest = manifest.get("champion_gate", {})
        existing_gate = gate_from_manifest if isinstance(gate_from_manifest, dict) else {}

    threshold = _safe_optional_float(existing_gate.get("threshold"))
    metric_source = str(existing_gate.get("metric_source", "backtest_top1")).strip().lower() or "backtest_top1"
    profile_match = bool(existing_gate.get("profile_match", True))
    require_significant_lift = bool(existing_gate.get("require_significant_lift", False))
    significance_z = _safe_float(existing_gate.get("significance_z"))
    if not math.isfinite(significance_z):
        significance_z = 1.96
    max_selective_risk = _safe_optional_float(existing_gate.get("max_selective_risk"))
    max_abstention_rate = _safe_optional_float(existing_gate.get("max_abstention_rate"))
    conformal_alpha = _safe_float(existing_gate.get("conformal_alpha"))
    if not math.isfinite(conformal_alpha):
        conformal_alpha = 0.10

    y_val, y_test = _load_prepared_targets(manifest)
    current_risk_metrics = _load_current_risk_metrics(
        run_dir,
        [dict(row) for row in current_results if isinstance(row, dict)],
        val_y=y_val,
        test_y=y_test,
        conformal_alpha=conformal_alpha,
    )

    run_id = str(manifest.get("run_id", "")).strip() or run_dir.name
    refreshed_gate = evaluate_champion_gate(
        history_csv=outputs_dir / "history" / "experiment_history.csv",
        current_run_id=run_id,
        current_results=[dict(row) for row in current_results if isinstance(row, dict)],
        regression_threshold=threshold if threshold is not None else 0.005,
        backtest_history_csv=outputs_dir / "history" / "backtest_history.csv",
        current_backtest_rows=_load_run_backtest_rows(outputs_dir, run_id, run_dir=run_dir),
        metric_source=metric_source,
        current_profile=str(manifest.get("profile", "")).strip() or None,
        require_profile_match=profile_match,
        require_significant_lift=require_significant_lift,
        significance_z=significance_z,
        current_risk_metrics=current_risk_metrics,
        max_selective_risk=max_selective_risk,
        max_abstention_rate=max_abstention_rate,
    )

    champion_alias_payload: dict[str, object] = {
        "updated": False,
        "alias_file": "",
        "run_id": "",
        "run_dir": "",
        "model_name": "",
        "model_type": "",
        "reason": "gate_not_promoted",
    }
    if bool(refreshed_gate.get("promoted", False)):
        champion_model = preferred_serveable_model(
            [dict(row) for row in current_results if isinstance(row, dict)],
            run_dir=run_dir,
            preferred_model_name=str(refreshed_gate.get("challenger_model_name", "")),
        )
        if champion_model is None:
            champion_alias_payload["reason"] = "no_serveable_models_in_promoted_run"
        else:
            champion_model_name, champion_model_type = champion_model
            alias_file = write_champion_alias(
                output_dir=outputs_dir,
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

    write_json(gate_path, refreshed_gate)
    manifest["champion_gate"] = refreshed_gate
    manifest["champion_alias"] = champion_alias_payload
    write_json(manifest_path, manifest)

    control_room_json = None
    if refresh_control_room:
        control_room_json, _ = write_control_room_report(outputs_dir)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metric_source": str(refreshed_gate.get("metric_source", "")),
        "status": str(refreshed_gate.get("status", "")),
        "promoted": bool(refreshed_gate.get("promoted", False)),
        "regression": _safe_float(refreshed_gate.get("regression")),
        "threshold": _safe_float(refreshed_gate.get("threshold")),
        "challenger_model_name": str(refreshed_gate.get("challenger_model_name", "")),
        "challenger_selective_risk": _safe_float(refreshed_gate.get("challenger_selective_risk")),
        "challenger_abstention_rate": _safe_float(refreshed_gate.get("challenger_abstention_rate")),
        "risk_metric_model_count": int(len(current_risk_metrics)),
        "champion_alias_updated": bool(champion_alias_payload.get("updated", False)),
        "champion_alias_model_name": str(champion_alias_payload.get("model_name", "")),
        "control_room_refreshed": bool(control_room_json is not None),
        "control_room_json": str(control_room_json) if control_room_json is not None else "",
    }


def main() -> int:
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    run_dir = _resolve_run_dir(outputs_dir, args.run_dir)
    payload = refresh_champion_gate(
        outputs_dir=outputs_dir,
        run_dir=run_dir,
        refresh_control_room=not args.skip_control_room,
    )
    print(
        "run="
        f"{payload['run_id']} status={payload['status']} promoted={payload['promoted']} "
        f"metric={payload['metric_source']} regression={payload['regression']} threshold={payload['threshold']} "
        f"challenger={payload['challenger_model_name']} "
        f"challenger_selective_risk={payload['challenger_selective_risk']} "
        f"challenger_abstention_rate={payload['challenger_abstention_rate']} "
        f"risk_metric_model_count={payload['risk_metric_model_count']} "
        f"champion_alias_updated={payload['champion_alias_updated']} "
        f"champion_alias_model_name={payload['champion_alias_model_name']} "
        f"control_room_refreshed={payload['control_room_refreshed']}"
    )
    if payload["control_room_json"]:
        print(f"control_room_json={payload['control_room_json']}")
    return 0


__all__ = ["main", "refresh_champion_gate"]


if __name__ == "__main__":
    raise SystemExit(main())
