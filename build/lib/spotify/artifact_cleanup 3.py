from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sqlite3
from urllib.parse import unquote, urlparse

from .champion_alias import best_serveable_model, read_champion_alias
from .tracking import resolve_mlflow_artifact_policy, should_log_mlflow_artifact


def _normalize_cleanup_mode(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value in ("", "1", "true", "yes", "on", "light", "safe"):
        return "light"
    if value in ("0", "false", "no", "off", "none", "disabled"):
        return "off"
    if value in ("aggressive", "full"):
        return "aggressive"
    return "light"


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _resolve_existing_path(raw_path: str, *, fallbacks: list[Path]) -> Path | None:
    candidates: list[Path] = []
    if raw_path:
        candidates.append(Path(raw_path).expanduser())
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_path_from_uri(raw_uri: str) -> Path | None:
    value = str(raw_uri or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("", "file"):
        return None
    if parsed.scheme == "file":
        uri_path = unquote(parsed.path or "")
        if parsed.netloc:
            uri_path = f"/{parsed.netloc}{uri_path}"
        candidate = Path(uri_path).expanduser()
    else:
        candidate = Path(value).expanduser()
    return candidate.resolve()


def _load_mlflow_artifact_dirs(output_dir: Path) -> list[Path]:
    db_path = output_dir / "mlruns" / "mlflow.db"
    if not db_path.exists():
        return []

    artifact_dirs: list[Path] = []
    seen: set[Path] = set()
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT artifact_uri FROM runs WHERE artifact_uri IS NOT NULL AND artifact_uri != ''"
            ).fetchall()
    except sqlite3.Error:
        rows = []

    for (raw_uri,) in rows:
        artifact_dir = _resolve_path_from_uri(str(raw_uri or "").strip())
        if artifact_dir is None or artifact_dir in seen or not artifact_dir.exists() or not artifact_dir.is_dir():
            continue
        seen.add(artifact_dir)
        artifact_dirs.append(artifact_dir)
    return sorted(artifact_dirs)


def _collect_required_models(
    model_name: str,
    *,
    results_index: dict[str, dict[str, object]],
    required: set[str] | None = None,
) -> set[str]:
    required = required or set()
    normalized = str(model_name).strip()
    if not normalized or normalized in required:
        return required
    required.add(normalized)
    row = results_index.get(normalized)
    if row is None:
        return required
    if str(row.get("model_type", "")).strip().lower() == "ensemble":
        members = row.get("ensemble_members", [])
        if isinstance(members, list):
            for member_name in members:
                _collect_required_models(str(member_name).strip(), results_index=results_index, required=required)
    return required


def _sync_classical_results_json(
    *,
    run_dir: Path,
    deleted_models: set[str],
) -> None:
    path = run_dir / "classical_results.json"
    if not path.exists() or not deleted_models:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, list):
        return
    updated = False
    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("model_name", "")).strip() in deleted_models and row.get("estimator_artifact_path"):
            row["estimator_artifact_path"] = ""
            updated = True
    if updated:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sync_run_results_json(*, run_dir: Path, result_rows: list[dict[str, object]]) -> None:
    path = run_dir / "run_results.json"
    if not path.exists():
        return
    path.write_text(json.dumps(result_rows, indent=2), encoding="utf-8")


def _load_json_payload(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_deleted_prediction_paths(payload: object, *, deleted_paths: set[Path], deleted_dirs: set[Path]) -> bool:
    changed = False
    if isinstance(payload, dict):
        for key, value in list(payload.items()):
            if key == "prediction_bundle_path" and isinstance(value, str) and value.strip():
                candidate = Path(value).expanduser()
                resolved = candidate.resolve() if candidate.exists() else candidate
                if resolved in deleted_paths or any(parent == resolved or parent in resolved.parents for parent in deleted_dirs):
                    payload[key] = ""
                    changed = True
            else:
                changed = _clear_deleted_prediction_paths(value, deleted_paths=deleted_paths, deleted_dirs=deleted_dirs) or changed
    elif isinstance(payload, list):
        for item in payload:
            changed = _clear_deleted_prediction_paths(item, deleted_paths=deleted_paths, deleted_dirs=deleted_dirs) or changed
    return changed


def _sync_prediction_bundle_references(
    *,
    run_dir: Path,
    deleted_paths: set[Path],
    deleted_dirs: set[Path],
) -> None:
    json_candidates = [
        run_dir / "run_results.json",
        run_dir / "classical_results.json",
        run_dir / "optuna" / "optuna_results.json",
        run_dir / "analysis" / "ensemble_blended_ensemble_summary.json",
    ]
    for path in json_candidates:
        payload = _load_json_payload(path)
        if payload is None:
            continue
        if _clear_deleted_prediction_paths(payload, deleted_paths=deleted_paths, deleted_dirs=deleted_dirs):
            _write_json_payload(path, payload)


def _infer_run_profile(run_dir: Path) -> str | None:
    manifest = _load_json_payload(run_dir / "run_manifest.json")
    if isinstance(manifest, dict):
        profile = str(manifest.get("profile", "")).strip().lower()
        if profile:
            return profile
    name = run_dir.name.strip().lower()
    if "everything-" in name:
        return "full"
    return None


def _run_timestamp_key(run_dir: Path) -> tuple[int, str]:
    manifest = _load_json_payload(run_dir / "run_manifest.json")
    if isinstance(manifest, dict):
        raw = str(manifest.get("timestamp", "")).strip()
        if raw:
            try:
                stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return int(stamp.timestamp()), run_dir.name
            except Exception:
                pass
    prefix = run_dir.name.split("_", 2)
    if len(prefix) >= 2 and prefix[0].isdigit() and prefix[1].isdigit():
        try:
            stamp = datetime.strptime(f"{prefix[0]}_{prefix[1]}", "%Y%m%d_%H%M%S")
            return int(stamp.timestamp()), run_dir.name
        except Exception:
            pass
    return 0, run_dir.name


def retained_full_run_dirs(
    *,
    output_dir: Path,
    keep_last_n: int,
    current_run_dir: Path | None = None,
) -> set[Path]:
    if keep_last_n <= 0:
        return {current_run_dir.resolve()} if current_run_dir is not None else set()
    full_runs: list[tuple[tuple[int, str], Path]] = []
    for run_dir in sorted((output_dir / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        if _infer_run_profile(run_dir) == "full":
            full_runs.append((_run_timestamp_key(run_dir), run_dir.resolve()))
    full_runs.sort(key=lambda item: item[0], reverse=True)
    retained = {run_dir for _, run_dir in full_runs[: max(0, int(keep_last_n))]}
    if current_run_dir is not None:
        retained.add(current_run_dir.resolve())
    return retained


def load_result_rows_for_cleanup(run_dir: Path) -> list[dict[str, object]]:
    run_results_path = run_dir / "run_results.json"
    if run_results_path.exists():
        try:
            payload = json.loads(run_results_path.read_text(encoding="utf-8"))
        except Exception:
            payload = []
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]

    classical_results_path = run_dir / "classical_results.json"
    if classical_results_path.exists():
        try:
            payload = json.loads(classical_results_path.read_text(encoding="utf-8"))
        except Exception:
            payload = []
        rows: list[dict[str, object]] = []
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                upgraded = dict(row)
                upgraded.setdefault("model_type", "classical")
                upgraded.setdefault("model_family", "")
                upgraded.setdefault("prediction_bundle_path", "")
                rows.append(upgraded)
        return rows
    return []


def select_model_for_run_cleanup(
    *,
    run_dir: Path,
    output_dir: Path,
    result_rows: list[dict[str, object]],
) -> tuple[str, str] | None:
    alias = read_champion_alias(output_dir / "models" / "champion" / "alias.json")
    if alias is not None and alias.run_dir.resolve() == run_dir.resolve():
        return alias.model_name, alias.model_type

    selected = best_serveable_model(result_rows, run_dir=run_dir)
    if selected is not None:
        return selected

    best_row: dict[str, object] | None = None
    best_score = float("-inf")
    for row in result_rows:
        if str(row.get("model_type", "")).strip().lower() not in ("classical", "classical_tuned"):
            continue
        estimator_raw = str(row.get("estimator_artifact_path", "")).strip()
        if not estimator_raw:
            continue
        estimator_path = Path(estimator_raw).expanduser()
        if not estimator_path.exists():
            continue
        try:
            score = float(row.get("val_top1", float("-inf")))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return None
    return str(best_row.get("model_name", "")).strip(), str(best_row.get("model_type", "classical")).strip().lower()


def prune_run_artifacts(
    *,
    run_dir: Path,
    result_rows: list[dict[str, object]],
    selected_model: tuple[str, str] | None,
    logger,
    cleanup_mode: str,
    min_size_mb: float,
) -> dict[str, object]:
    mode = _normalize_cleanup_mode(cleanup_mode)
    min_size_bytes = int(max(0.0, float(min_size_mb)) * 1024 * 1024)
    summary: dict[str, object] = {
        "enabled": mode != "off",
        "mode": mode,
        "threshold_mb": float(min_size_mb),
        "status": "disabled" if mode == "off" else "pending",
        "selected_model_name": "",
        "selected_model_type": "",
        "kept_models": [],
        "deleted_files": [],
        "freed_bytes": 0,
    }
    if mode == "off":
        return summary

    resolved_selected = selected_model or best_serveable_model(result_rows, run_dir=run_dir)
    if not resolved_selected:
        summary["status"] = "skipped_no_serveable_model"
        return summary

    selected_model_name, selected_model_type = resolved_selected
    results_index = {
        str(row.get("model_name", "")).strip(): row
        for row in result_rows
        if str(row.get("model_name", "")).strip()
    }
    required_models = _collect_required_models(selected_model_name, results_index=results_index)
    summary["selected_model_name"] = selected_model_name
    summary["selected_model_type"] = selected_model_type
    summary["kept_models"] = sorted(required_models)

    deleted_files: list[str] = []
    deleted_models: set[str] = set()
    freed_bytes = 0

    for row in result_rows:
        model_name = str(row.get("model_name", "")).strip()
        model_type = str(row.get("model_type", "")).strip().lower()
        if not model_name or model_name in required_models:
            continue
        if model_type not in ("classical", "classical_tuned"):
            continue

        estimator_raw = str(row.get("estimator_artifact_path", "")).strip()
        fallback_dir = run_dir / "estimators" if model_type == "classical" else run_dir / "optuna" / "estimators"
        fallback_name = f"classical_{model_name}.joblib" if model_type == "classical" else f"classical_tuned_{str(row.get('base_model_name', '')).strip() or model_name}.joblib"
        estimator_path = _resolve_existing_path(
            estimator_raw,
            fallbacks=[fallback_dir / fallback_name],
        )
        if estimator_path is None:
            row["estimator_artifact_path"] = ""
            continue

        try:
            size_bytes = estimator_path.stat().st_size
        except OSError:
            size_bytes = 0
        if size_bytes < min_size_bytes:
            row["estimator_artifact_path"] = str(estimator_path)
            continue

        try:
            estimator_path.unlink()
        except OSError:
            row["estimator_artifact_path"] = str(estimator_path)
            continue

        row["estimator_artifact_path"] = ""
        deleted_models.add(model_name)
        deleted_files.append(str(estimator_path))
        freed_bytes += max(0, size_bytes)
        logger.info(
            "Pruned model artifact for %s: %s (%.1f MB)",
            model_name,
            estimator_path,
            size_bytes / (1024 * 1024),
        )

    referenced_paths: set[Path] = set()
    for row in result_rows:
        estimator_raw = str(row.get("estimator_artifact_path", "")).strip()
        if not estimator_raw:
            continue
        resolved = _resolve_existing_path(estimator_raw, fallbacks=[])
        if resolved is not None:
            referenced_paths.add(resolved)

    for estimator_dir in (run_dir / "estimators", run_dir / "optuna" / "estimators"):
        if not estimator_dir.exists():
            continue
        for estimator_path in estimator_dir.glob("*.joblib"):
            resolved_path = estimator_path.resolve()
            if resolved_path in referenced_paths:
                continue
            try:
                size_bytes = resolved_path.stat().st_size
            except OSError:
                size_bytes = 0
            if size_bytes < min_size_bytes:
                continue
            try:
                resolved_path.unlink()
            except OSError:
                continue
            deleted_files.append(str(resolved_path))
            freed_bytes += max(0, size_bytes)
            logger.info(
                "Pruned orphaned model artifact: %s (%.1f MB)",
                resolved_path,
                size_bytes / (1024 * 1024),
            )

    _sync_classical_results_json(run_dir=run_dir, deleted_models=deleted_models)
    _sync_run_results_json(run_dir=run_dir, result_rows=result_rows)

    summary["deleted_files"] = deleted_files
    summary["freed_bytes"] = int(freed_bytes)
    summary["status"] = "completed"
    if deleted_files:
        logger.info(
            "Artifact cleanup completed: pruned %d files and freed %.2f GB",
            len(deleted_files),
            freed_bytes / (1024 * 1024 * 1024),
        )
    else:
        logger.info("Artifact cleanup completed: nothing eligible for pruning.")
    return summary


def prune_existing_runs(
    *,
    output_dir: Path,
    run_dirs: list[Path] | None,
    logger,
    cleanup_mode: str,
    min_size_mb: float,
) -> dict[str, object]:
    target_dirs = [path.resolve() for path in run_dirs] if run_dirs else sorted((output_dir / "runs").glob("*"))
    summaries: list[dict[str, object]] = []
    total_freed = 0
    total_deleted = 0

    for run_dir in target_dirs:
        if not run_dir.exists() or not run_dir.is_dir():
            continue
        result_rows = load_result_rows_for_cleanup(run_dir)
        if not result_rows:
            continue
        selected_model = select_model_for_run_cleanup(run_dir=run_dir, output_dir=output_dir, result_rows=result_rows)
        summary = prune_run_artifacts(
            run_dir=run_dir,
            result_rows=result_rows,
            selected_model=selected_model,
            logger=logger,
            cleanup_mode=cleanup_mode,
            min_size_mb=min_size_mb,
        )
        summary["run_dir"] = str(run_dir)
        summaries.append(summary)
        total_freed += int(summary.get("freed_bytes", 0) or 0)
        total_deleted += len(summary.get("deleted_files", []))

    return {
        "run_count": len(summaries),
        "deleted_file_count": total_deleted,
        "freed_bytes": total_freed,
        "runs": summaries,
    }


def prune_old_auxiliary_artifacts(
    *,
    output_dir: Path,
    current_run_dir: Path | None,
    logger,
    keep_last_full_runs: int,
    prune_prediction_bundles: bool,
    prune_run_databases: bool,
) -> dict[str, object]:
    retained_runs = retained_full_run_dirs(
        output_dir=output_dir,
        keep_last_n=keep_last_full_runs,
        current_run_dir=current_run_dir,
    )
    summaries: list[dict[str, object]] = []
    total_freed = 0
    total_deleted = 0

    for run_dir in sorted((output_dir / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        resolved_run_dir = run_dir.resolve()
        summary = {
            "run_dir": str(resolved_run_dir),
            "retained": resolved_run_dir in retained_runs,
            "deleted_files": [],
            "freed_bytes": 0,
        }
        if resolved_run_dir in retained_runs:
            summaries.append(summary)
            continue

        deleted_paths: set[Path] = set()
        deleted_dirs: set[Path] = set()
        freed_bytes = 0

        if prune_prediction_bundles:
            bundle_dirs = [
                run_dir / "prediction_bundles",
                run_dir / "optuna" / "prediction_bundles",
            ]
            for bundle_dir in bundle_dirs:
                if not bundle_dir.exists():
                    continue
                for file_path in sorted(bundle_dir.rglob("*"), reverse=True):
                    if not file_path.is_file():
                        continue
                    size_bytes = _file_size(file_path)
                    try:
                        file_path.unlink()
                    except OSError:
                        continue
                    resolved_file = file_path.resolve()
                    deleted_paths.add(resolved_file)
                    summary["deleted_files"].append(str(resolved_file))
                    freed_bytes += size_bytes
                for path in sorted(bundle_dir.rglob("*"), reverse=True):
                    if path.is_dir():
                        try:
                            path.rmdir()
                        except OSError:
                            pass
                try:
                    bundle_dir.rmdir()
                except OSError:
                    pass
                deleted_dirs.add(bundle_dir.resolve())

        if prune_run_databases:
            for db_path in sorted(run_dir.glob("*.db")):
                size_bytes = _file_size(db_path)
                try:
                    db_path.unlink()
                except OSError:
                    continue
                resolved_db = db_path.resolve()
                deleted_paths.add(resolved_db)
                summary["deleted_files"].append(str(resolved_db))
                freed_bytes += size_bytes

        if deleted_paths or deleted_dirs:
            _sync_prediction_bundle_references(
                run_dir=run_dir,
                deleted_paths=deleted_paths,
                deleted_dirs=deleted_dirs,
            )

        summary["freed_bytes"] = int(freed_bytes)
        total_freed += int(freed_bytes)
        total_deleted += len(summary["deleted_files"])
        if summary["deleted_files"]:
            logger.info(
                "Pruned auxiliary artifacts for %s: %d files, freed %.2f GB",
                run_dir.name,
                len(summary["deleted_files"]),
                freed_bytes / (1024 * 1024 * 1024),
            )
        summaries.append(summary)

    return {
        "keep_last_full_runs": int(max(0, keep_last_full_runs)),
        "retained_runs": [str(path) for path in sorted(retained_runs)],
        "deleted_file_count": total_deleted,
        "freed_bytes": total_freed,
        "runs": summaries,
    }


def prune_mlflow_artifacts(
    *,
    output_dir: Path,
    logger,
    artifact_mode: str | None = None,
    max_artifact_mb: float | None = None,
) -> dict[str, object]:
    resolved_mode, resolved_max_artifact_mb = resolve_mlflow_artifact_policy(
        mode_raw=artifact_mode,
        max_artifact_mb_raw=max_artifact_mb,
    )
    summary: dict[str, object] = {
        "enabled": resolved_mode != "off",
        "artifact_mode": resolved_mode,
        "max_artifact_mb": float(resolved_max_artifact_mb),
        "status": "pending",
        "artifact_dir_count": 0,
        "artifact_dirs": [],
        "deleted_file_count": 0,
        "deleted_files": [],
        "freed_bytes": 0,
    }
    if resolved_mode in ("off", "all"):
        summary["status"] = "skipped_policy"
        return summary

    artifact_dirs = _load_mlflow_artifact_dirs(output_dir.resolve())
    summary["artifact_dir_count"] = len(artifact_dirs)
    summary["artifact_dirs"] = [str(path) for path in artifact_dirs]
    if not artifact_dirs:
        summary["status"] = "skipped_no_artifact_dirs"
        return summary

    deleted_files: list[str] = []
    freed_bytes = 0

    for artifact_dir in artifact_dirs:
        for file_path in sorted(artifact_dir.rglob("*"), reverse=True):
            if not file_path.is_file():
                continue
            if should_log_mlflow_artifact(
                file_path,
                mode=resolved_mode,
                max_artifact_mb=resolved_max_artifact_mb,
            ):
                continue
            size_bytes = _file_size(file_path)
            try:
                file_path.unlink()
            except OSError:
                continue
            deleted_files.append(str(file_path.resolve()))
            freed_bytes += size_bytes

        for path in sorted(artifact_dir.rglob("*"), reverse=True):
            if not path.is_dir():
                continue
            try:
                path.rmdir()
            except OSError:
                continue

    summary["status"] = "completed"
    summary["deleted_file_count"] = len(deleted_files)
    summary["deleted_files"] = deleted_files
    summary["freed_bytes"] = int(freed_bytes)
    if deleted_files:
        logger.info(
            "MLflow artifact cleanup completed: pruned %d files and freed %.2f GB",
            len(deleted_files),
            freed_bytes / (1024 * 1024 * 1024),
        )
    else:
        logger.info("MLflow artifact cleanup completed: nothing eligible for pruning.")
    return summary
