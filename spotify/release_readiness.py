from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlparse

from .champion_alias import read_champion_alias, resolve_prediction_run_dir
from .deployment_registry import DEFAULT_REGISTRY_ROOT
from .run_artifacts import safe_read_json, write_csv_rows, write_json, write_markdown


CHECK_COLUMNS = ["check_key", "status", "severity", "message", "path", "expected", "observed"]


@dataclass(frozen=True)
class ReadinessCheck:
    check_key: str
    status: str
    severity: str
    message: str
    path: str = ""
    expected: str = ""
    observed: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "check_key": self.check_key,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
            "expected": self.expected,
            "observed": self.observed,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    raw = str(value).strip()
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        return Path(unquote(parsed.path)).expanduser().resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _path_text(path: Path | None) -> str:
    return str(path.resolve()) if path is not None else ""


def _file_check(check_key: str, path: Path, *, severity: str = "required", label: str | None = None) -> ReadinessCheck:
    exists = path.exists()
    return ReadinessCheck(
        check_key=check_key,
        status="pass" if exists else ("warn" if severity == "advisory" else "fail"),
        severity=severity,
        message=f"{label or check_key} {'exists' if exists else 'is missing'}.",
        path=str(path),
        expected="exists",
        observed="exists" if exists else "missing",
    )


def _bool_check(
    check_key: str,
    condition: bool,
    *,
    severity: str = "required",
    pass_message: str,
    fail_message: str,
    path: Path | None = None,
    expected: str = "true",
    observed: object = "",
) -> ReadinessCheck:
    return ReadinessCheck(
        check_key=check_key,
        status="pass" if condition else ("warn" if severity == "advisory" else "fail"),
        severity=severity,
        message=pass_message if condition else fail_message,
        path=_path_text(path),
        expected=expected,
        observed=str(observed if observed != "" else condition).lower(),
    )


def _serving_bundle_rows(run_dir: Path) -> list[dict[str, object]]:
    serving_dir = run_dir / "analysis" / "serving"
    rows: list[dict[str, object]] = []
    for bundle_path in sorted(serving_dir.glob("*.joblib")):
        manifest_path = bundle_path.with_suffix(".manifest.json")
        rows.append(
            {
                "bundle_name": bundle_path.name,
                "bundle_path": str(bundle_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_exists": manifest_path.exists(),
                "size_bytes": bundle_path.stat().st_size if bundle_path.exists() else 0,
            }
        )
    return rows


def _summarize_checks(checks: list[ReadinessCheck]) -> dict[str, object]:
    pass_count = sum(1 for check in checks if check.status == "pass")
    warn_count = sum(1 for check in checks if check.status == "warn")
    fail_count = sum(1 for check in checks if check.status == "fail")
    status = "fail" if fail_count else "attention" if warn_count else "pass"
    return {
        "status": status,
        "release_ready": status == "pass",
        "check_count": len(checks),
        "pass_count": pass_count,
        "warning_count": warn_count,
        "fail_count": fail_count,
    }


def _alias_path_for(run_dir_arg: str | None, *, outputs_dir: Path) -> Path | None:
    if not run_dir_arg:
        return outputs_dir / "models" / "champion" / "alias.json"
    requested = Path(run_dir_arg).expanduser()
    if requested.is_dir() and (requested / "alias.json").exists():
        return (requested / "alias.json").resolve()
    return None


def _release_manifest_path(
    *,
    registry_root: Path,
    release_id: str,
    channel_payload: dict[str, object],
) -> Path:
    manifest_hint = str(channel_payload.get("release_manifest_path", "")).strip()
    channel_release_id = str(channel_payload.get("current_release_id", "")).strip()
    if manifest_hint and (not release_id or channel_release_id == release_id):
        hinted = Path(manifest_hint).expanduser()
        if hinted.exists():
            return hinted.resolve()
    return registry_root / "releases" / release_id / "deployment_release.json"


def _check_deploy_templates(project_root: Path) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    deploy_files = [
        project_root / "deploy" / "kubernetes" / "predict-deployment.yaml",
        project_root / "deploy" / "kubernetes" / "taste-os-deployment.yaml",
        project_root / "deploy" / "ecs" / "predict-task-definition.json",
        project_root / "deploy" / "ecs" / "taste-os-task-definition.json",
    ]
    for path in deploy_files:
        checks.append(_file_check(f"deploy_template_{path.stem}", path, label=f"Deployment template {path.name}"))

    for path in deploy_files[:2]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        checks.append(
            _bool_check(
                f"deploy_template_{path.stem}_readyz_probe",
                "/readyz" in text,
                pass_message=f"{path.name} points health checks at /readyz.",
                fail_message=f"{path.name} does not reference /readyz.",
                path=path,
                expected="/readyz",
                observed="/readyz" if "/readyz" in text else "missing",
            )
        )

    for path in deploy_files[2:]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        checks.append(
            _bool_check(
                f"deploy_template_{path.stem}_registry_channel",
                "/app/outputs/deployments/registry/channels/stable" in text,
                pass_message=f"{path.name} points services at the stable registry channel.",
                fail_message=f"{path.name} does not reference the stable registry channel.",
                path=path,
                expected="/app/outputs/deployments/registry/channels/stable",
                observed="present" if "/app/outputs/deployments/registry/channels/stable" in text else "missing",
            )
        )
    return checks


def _markdown(payload: dict[str, object]) -> list[str]:
    summary = payload.get("summary", {})
    summary_dict = summary if isinstance(summary, dict) else {}
    checks = payload.get("checks", [])
    check_rows = checks if isinstance(checks, list) else []
    failed_or_warned = [
        row
        for row in check_rows
        if isinstance(row, dict) and str(row.get("status", "")) in {"fail", "warn"}
    ]
    lines = [
        "# Release Readiness Smoke",
        "",
        f"Generated at: {payload.get('generated_at', '')}",
        f"Status: {str(summary_dict.get('status', '')).upper()}",
        f"Run: `{payload.get('run_dir', '')}`",
        f"Model: `{payload.get('model_name', '')}`",
        f"Channel: `{payload.get('channel', '')}`",
        f"Registry: `{payload.get('registry_root', '')}`",
        "",
        "## Summary",
        "",
        f"- Checks: {summary_dict.get('check_count', 0)}",
        f"- Passed: {summary_dict.get('pass_count', 0)}",
        f"- Warnings: {summary_dict.get('warning_count', 0)}",
        f"- Failed: {summary_dict.get('fail_count', 0)}",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Message |",
        "| --- | --- | --- | --- |",
    ]
    for row in check_rows:
        if not isinstance(row, dict):
            continue
        message = str(row.get("message", "")).replace("|", "/")
        lines.append(f"| {row.get('check_key', '')} | {row.get('status', '')} | {row.get('severity', '')} | {message} |")

    lines.extend(["", "## Follow-Up", ""])
    if not failed_or_warned:
        lines.append("- Release metadata, alias resolution, serving bundles, and deploy templates agree.")
    else:
        for row in failed_or_warned[:8]:
            lines.append(f"- {row.get('status', '')}: {row.get('message', '')}")
    return lines


def build_release_readiness_smoke(
    *,
    project_root: Path,
    outputs_dir: Path,
    run_dir: str | None = None,
    registry_root: str | Path | None = None,
    channel: str = "stable",
    release_id: str | None = None,
    output_dir: Path | None = None,
    require_registry: bool = False,
    require_serving_bundle: bool = True,
    allow_pending_channel: bool = False,
) -> dict[str, object]:
    project_root = project_root.expanduser().resolve()
    outputs_dir = _resolve_path(outputs_dir, base=project_root)
    registry_root_path = _resolve_path(registry_root or DEFAULT_REGISTRY_ROOT, base=project_root)
    output_root = (output_dir.expanduser().resolve() if output_dir is not None else outputs_dir / "analysis" / "release_readiness")
    normalized_channel = str(channel).strip().lower() or "stable"
    checks: list[ReadinessCheck] = []

    alias_path = _alias_path_for(run_dir, outputs_dir=outputs_dir)
    if alias_path is not None:
        checks.append(_file_check("alias_file_present", alias_path, label="Prediction alias file"))

    resolved_run_dir: Path | None = None
    alias_model_name: str | None = None
    model_name = ""
    model_type = ""
    try:
        resolved_run_dir, alias_model_name = resolve_prediction_run_dir(run_dir, project_root=project_root)
        checks.append(
            ReadinessCheck(
                check_key="alias_resolves_run",
                status="pass",
                severity="required",
                message="Run resolution succeeded.",
                path=str(resolved_run_dir),
                expected="resolvable run directory",
                observed=str(resolved_run_dir),
            )
        )
    except Exception as exc:
        checks.append(
            ReadinessCheck(
                check_key="alias_resolves_run",
                status="fail",
                severity="required",
                message=f"Run resolution failed: {exc}",
                expected="resolvable run directory",
                observed="unresolved",
            )
        )

    alias_payload = None
    if alias_path is not None and alias_path.exists():
        try:
            alias_payload = read_champion_alias(alias_path)
            if alias_payload is not None:
                model_name = alias_payload.model_name
                model_type = alias_payload.model_type
        except Exception as exc:
            checks.append(
                ReadinessCheck(
                    check_key="alias_payload_valid",
                    status="fail",
                    severity="required",
                    message=f"Alias payload is invalid: {exc}",
                    path=str(alias_path),
                )
            )

    if resolved_run_dir is not None:
        run_dir_path = resolved_run_dir
        checks.append(_file_check("run_manifest_present", run_dir_path / "run_manifest.json", label="Run manifest"))
        checks.append(_file_check("run_results_present", run_dir_path / "run_results.json", label="Run results"))
        checks.append(_file_check("feature_metadata_present", run_dir_path / "feature_metadata.json", label="Feature metadata"))
        checks.append(_file_check("champion_gate_present", run_dir_path / "champion_gate.json", label="Champion gate"))

        manifest_payload = safe_read_json(run_dir_path / "run_manifest.json", default={})
        manifest_payload = manifest_payload if isinstance(manifest_payload, dict) else {}
        manifest_alias = manifest_payload.get("champion_alias", {})
        if isinstance(manifest_alias, dict):
            model_name = model_name or str(alias_model_name or manifest_alias.get("model_name", "")).strip()
            model_type = model_type or str(manifest_alias.get("model_type", "")).strip().lower()
        gate_payload = safe_read_json(run_dir_path / "champion_gate.json", default={})
        gate_payload = gate_payload if isinstance(gate_payload, dict) else {}
        promoted = bool(gate_payload.get("promoted", False))
        checks.append(
            _bool_check(
                "champion_gate_promoted",
                promoted,
                pass_message="Champion gate is promoted.",
                fail_message="Champion gate is not promoted.",
                path=run_dir_path / "champion_gate.json",
                expected="promoted=true",
                observed=gate_payload.get("promoted", "missing"),
            )
        )
        checks.append(
            _bool_check(
                "model_identity_present",
                bool(model_name and model_type),
                pass_message="Model identity is available for serving.",
                fail_message="Model identity is missing model_name or model_type.",
                path=alias_path or run_dir_path / "run_manifest.json",
                expected="model_name and model_type",
                observed=f"{model_name}:{model_type}",
            )
        )
        global_alias_path = outputs_dir / "models" / "champion" / "alias.json"
        if global_alias_path.exists():
            try:
                global_alias = read_champion_alias(global_alias_path)
            except Exception as exc:
                global_alias = None
                checks.append(
                    ReadinessCheck(
                        check_key="global_champion_alias_valid",
                        status="fail",
                        severity="required",
                        message=f"Global champion alias is invalid: {exc}",
                        path=str(global_alias_path),
                    )
                )
            if global_alias is not None:
                checks.append(
                    _bool_check(
                        "global_champion_alias_matches_run",
                        global_alias.run_dir.resolve() == run_dir_path.resolve(),
                        pass_message="Global champion alias resolves to the release run.",
                        fail_message="Global champion alias resolves to a different run.",
                        path=global_alias_path,
                        expected=str(run_dir_path),
                        observed=str(global_alias.run_dir),
                    )
                )
                checks.append(
                    _bool_check(
                        "global_champion_alias_matches_model",
                        bool(model_name and global_alias.model_name == model_name),
                        pass_message="Global champion alias model matches the release model.",
                        fail_message="Global champion alias model does not match the release model.",
                        path=global_alias_path,
                        expected=model_name,
                        observed=global_alias.model_name,
                    )
                )

        serving_bundles = _serving_bundle_rows(run_dir_path)
        checks.append(
            _bool_check(
                "serving_bundle_present",
                bool(serving_bundles),
                severity="required" if require_serving_bundle else "advisory",
                pass_message="At least one materialized serving bundle is available.",
                fail_message="No materialized serving bundle is available under analysis/serving.",
                path=run_dir_path / "analysis" / "serving",
                expected="one or more *.joblib bundles",
                observed=str(len(serving_bundles)),
            )
        )
        missing_bundle_manifests = [row for row in serving_bundles if not bool(row.get("manifest_exists", False))]
        checks.append(
            _bool_check(
                "serving_bundle_manifests_present",
                not missing_bundle_manifests,
                severity="required" if serving_bundles else "advisory",
                pass_message="Every serving bundle has a manifest.",
                fail_message="One or more serving bundles are missing manifest JSON files.",
                path=run_dir_path / "analysis" / "serving",
                expected="manifest for every *.joblib",
                observed=str(len(missing_bundle_manifests)),
            )
        )
    else:
        run_dir_path = None
        serving_bundles = []

    channel_dir = registry_root_path / "channels" / normalized_channel
    channel_manifest_path = channel_dir / "deployment_channel.json"
    channel_alias_path = channel_dir / "alias.json"
    registry_severity = "required" if require_registry else "advisory"
    if allow_pending_channel:
        checks.append(
            ReadinessCheck(
                check_key="registry_channel_manifest_pending",
                status="pass",
                severity=registry_severity,
                message="Registry channel manifest is pending activation; checking the candidate release manifest instead.",
                path=str(channel_manifest_path),
                expected="pending activation allowed",
                observed="missing",
            )
        )
    else:
        checks.append(_file_check("registry_channel_manifest_present", channel_manifest_path, severity=registry_severity, label="Registry channel manifest"))
    if allow_pending_channel:
        checks.append(
            ReadinessCheck(
                check_key="registry_channel_alias_pending",
                status="pass",
                severity=registry_severity,
                message="Registry channel alias is pending activation; checking the candidate release manifest instead.",
                path=str(channel_alias_path),
                expected="pending activation allowed",
                observed="missing",
            )
        )
    else:
        checks.append(_file_check("registry_channel_alias_present", channel_alias_path, severity=registry_severity, label="Registry channel alias"))

    channel_payload = safe_read_json(channel_manifest_path, default={})
    channel_payload = channel_payload if isinstance(channel_payload, dict) else {}
    channel_release_id = str(channel_payload.get("current_release_id", "")).strip()
    effective_release_id = str(release_id or channel_release_id or (run_dir_path.name if run_dir_path is not None else "")).strip()
    checks.append(
        _bool_check(
            "registry_channel_release_id_present",
            bool(effective_release_id),
            severity=registry_severity,
            pass_message="A release id is available for registry verification.",
            fail_message="No release id is available from args, registry channel, or run directory.",
            path=channel_manifest_path,
            expected="release id",
            observed=effective_release_id or "missing",
        )
    )

    release_manifest_path = _release_manifest_path(
        registry_root=registry_root_path,
        release_id=effective_release_id,
        channel_payload=channel_payload,
    )
    checks.append(_file_check("registry_release_manifest_present", release_manifest_path, severity=registry_severity, label="Registry release manifest"))
    release_payload = safe_read_json(release_manifest_path, default={})
    release_payload = release_payload if isinstance(release_payload, dict) else {}
    release_model_name = str(release_payload.get("model_name", "")).strip()
    release_model_type = str(release_payload.get("model_type", "")).strip().lower()
    release_source_run_dir = Path(str(release_payload.get("source_run_dir", "")).strip()).expanduser() if release_payload.get("source_run_dir") else None
    release_bundles = release_payload.get("available_serving_bundles", [])
    release_bundles_list = release_bundles if isinstance(release_bundles, list) else []

    if release_payload:
        if run_dir_path is not None and release_source_run_dir is not None:
            try:
                release_matches_run = release_source_run_dir.resolve() == run_dir_path.resolve()
            except Exception:
                release_matches_run = False
            checks.append(
                _bool_check(
                    "registry_release_matches_resolved_run",
                    release_matches_run,
                    pass_message="Release manifest source run matches the resolved serving run.",
                    fail_message="Release manifest source run does not match the resolved serving run.",
                    path=release_manifest_path,
                    expected=str(run_dir_path),
                    observed=str(release_source_run_dir),
                )
            )
        checks.append(
            _bool_check(
                "registry_release_model_matches_alias",
                bool(model_name and release_model_name == model_name and (not model_type or release_model_type == model_type)),
                pass_message="Release model identity matches the alias.",
                fail_message="Release model identity does not match the alias.",
                path=release_manifest_path,
                expected=f"{model_name}:{model_type}",
                observed=f"{release_model_name}:{release_model_type}",
            )
        )
        checks.append(
            _bool_check(
                "registry_release_serving_bundles_present",
                bool(release_bundles_list),
                pass_message="Release manifest advertises serving bundles.",
                fail_message="Release manifest does not advertise serving bundles.",
                path=release_manifest_path,
                expected="one or more available_serving_bundles",
                observed=str(len(release_bundles_list)),
            )
        )

    if not allow_pending_channel and channel_alias_path.exists() and run_dir_path is not None:
        try:
            channel_run_dir, channel_model_name = resolve_prediction_run_dir(str(channel_dir), project_root=project_root)
            checks.append(
                _bool_check(
                    "registry_channel_alias_matches_run",
                    channel_run_dir.resolve() == run_dir_path.resolve(),
                    pass_message="Registry channel alias resolves to the same run.",
                    fail_message="Registry channel alias resolves to a different run.",
                    path=channel_alias_path,
                    expected=str(run_dir_path),
                    observed=str(channel_run_dir),
                )
            )
            checks.append(
                _bool_check(
                    "registry_channel_alias_matches_model",
                    bool(model_name and channel_model_name == model_name),
                    pass_message="Registry channel alias model matches the serving alias.",
                    fail_message="Registry channel alias model does not match the serving alias.",
                    path=channel_alias_path,
                    expected=model_name,
                    observed=channel_model_name or "",
                )
            )
        except Exception as exc:
            checks.append(
                ReadinessCheck(
                    check_key="registry_channel_alias_resolves",
                    status="fail" if require_registry else "warn",
                    severity=registry_severity,
                    message=f"Registry channel alias could not be resolved: {exc}",
                    path=str(channel_alias_path),
                )
            )

    checks.extend(_check_deploy_templates(project_root))
    summary = _summarize_checks(checks)
    generated_at = _utc_now_iso()
    payload: dict[str, object] = {
        "generated_at": generated_at,
        "project_root": str(project_root),
        "outputs_dir": str(outputs_dir),
        "run_dir": str(run_dir_path) if run_dir_path is not None else "",
        "model_name": model_name,
        "model_type": model_type,
        "channel": normalized_channel,
        "release_id": effective_release_id,
        "registry_root": str(registry_root_path),
        "release_manifest_path": str(release_manifest_path),
        "channel_manifest_path": str(channel_manifest_path),
        "require_registry": bool(require_registry),
        "require_serving_bundle": bool(require_serving_bundle),
        "allow_pending_channel": bool(allow_pending_channel),
        "serving_bundles": serving_bundles,
        "available_serving_bundle_count": len(release_bundles_list),
        "summary": summary,
        "checks": [check.as_dict() for check in checks],
    }
    paths = {
        "json": str(write_json(output_root / "release_readiness_smoke.json", payload, sort_keys=True)),
        "csv": str(write_csv_rows(output_root / "release_readiness_checks.csv", [check.as_dict() for check in checks], fieldnames=CHECK_COLUMNS)),
        "md": str(write_markdown(output_root / "release_readiness_smoke.md", _markdown(payload))),
    }
    manifest = {
        "generated_at": generated_at,
        "status": summary["status"],
        "release_ready": summary["release_ready"],
        "paths": paths,
    }
    paths["manifest_json"] = str(write_json(output_root / "release_readiness_manifest.json", manifest, sort_keys=True))
    payload["paths"] = paths
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.release_readiness",
        description="Write a release-readiness smoke bundle for serving aliases and deployment registry metadata.",
    )
    parser.add_argument("--project-root", type=str, default=".", help="Repository root containing deploy templates.")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs root used for champion aliases and reports.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory or alias directory to verify.")
    parser.add_argument(
        "--registry-root",
        type=str,
        default=os.getenv("SPOTIFY_DEPLOYMENT_REGISTRY_ROOT", DEFAULT_REGISTRY_ROOT),
        help="Deployment registry root to verify.",
    )
    parser.add_argument("--channel", type=str, default="stable", help="Registry channel to verify.")
    parser.add_argument("--release-id", type=str, default=None, help="Optional release id override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for readiness smoke artifacts.")
    parser.add_argument("--require-registry", action="store_true", help="Treat missing registry metadata as a failure.")
    parser.add_argument(
        "--no-require-serving-bundle",
        action="store_true",
        help="Downgrade missing serving bundles to an advisory warning.",
    )
    parser.add_argument("--strict", action="store_true", help="Return nonzero unless every readiness check passes.")
    parser.add_argument(
        "--allow-pending-channel",
        action="store_true",
        help="Allow missing channel alias/manifest while validating a candidate release before channel activation.",
    )
    parser.add_argument("--stdout-format", choices=("summary", "json"), default="summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_release_readiness_smoke(
        project_root=Path(args.project_root),
        outputs_dir=Path(args.outputs_dir),
        run_dir=args.run_dir,
        registry_root=args.registry_root,
        channel=str(args.channel),
        release_id=args.release_id,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        require_registry=bool(args.require_registry),
        require_serving_bundle=not bool(args.no_require_serving_bundle),
        allow_pending_channel=bool(args.allow_pending_channel),
    )
    summary = payload.get("summary", {})
    summary_dict = summary if isinstance(summary, dict) else {}
    paths = payload.get("paths", {})
    paths_dict = paths if isinstance(paths, dict) else {}
    if args.stdout_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"release_readiness_status={summary_dict.get('status', '')}")
        print(f"release_ready={summary_dict.get('release_ready', False)}")
        print(f"release_readiness_report={paths_dict.get('md', '')}")
    if args.strict and summary_dict.get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
