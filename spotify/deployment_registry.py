from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from urllib.parse import urlparse

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .run_artifacts import materialize_cached_file, safe_read_json, write_json
from .serving import resolve_model_row

DEFAULT_REGISTRY_ROOT = "outputs/deployments/registry"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_registry_root(value: str | Path) -> Path:
    if isinstance(value, Path):
        return value.expanduser().resolve()
    raw = str(value).strip()
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(
            f"Unsupported deployment registry root '{raw}'. Use a local path or file:// URI. "
            "Remote artifact roots can still be recorded with --artifact-base-uri."
        )
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    return Path(raw).expanduser().resolve()


def _artifact_uri_preview(*, artifact_base_uri: str | None, release_id: str, relative_path: str) -> str:
    base = str(artifact_base_uri or "").rstrip("/")
    if not base:
        return ""
    rel = relative_path.strip("/")
    parsed = urlparse(base)
    if parsed.scheme == "file":
        return ((Path(parsed.path).expanduser().resolve() / release_id / rel).resolve()).as_uri()
    if parsed.scheme:
        return f"{base}/{release_id}/{rel}"
    return ((Path(base).expanduser().resolve() / release_id / rel).resolve()).as_uri()


def _publish_file_artifact(*, source: Path, target_uri: str) -> str:
    parsed = urlparse(target_uri)
    if parsed.scheme == "file":
        destination = Path(parsed.path).expanduser().resolve()
    elif not parsed.scheme:
        destination = Path(target_uri).expanduser().resolve()
    else:
        raise ValueError(f"Unsupported file artifact URI '{target_uri}'.")
    materialize_cached_file(source, destination)
    return destination.as_uri()


def _publish_s3_artifact(*, source: Path, target_uri: str) -> str:
    parsed = urlparse(target_uri)
    bucket = parsed.netloc.strip()
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 artifact URI '{target_uri}'.")
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("boto3 is required to publish deployment artifacts to S3.") from exc
    client = boto3.client("s3")
    client.upload_file(str(source), bucket, key)
    return f"s3://{bucket}/{key}"


def _publish_gcs_artifact(*, source: Path, target_uri: str) -> str:
    parsed = urlparse(target_uri)
    bucket_name = parsed.netloc.strip()
    blob_name = parsed.path.lstrip("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS artifact URI '{target_uri}'.")
    try:
        from google.cloud import storage
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("google-cloud-storage is required to publish deployment artifacts to GCS.") from exc
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(source))
    return f"gs://{bucket_name}/{blob_name}"


def _publish_artifact(*, source: Path, target_uri: str) -> str:
    parsed = urlparse(target_uri)
    if parsed.scheme in {"", "file"}:
        return _publish_file_artifact(source=source, target_uri=target_uri)
    if parsed.scheme == "s3":
        return _publish_s3_artifact(source=source, target_uri=target_uri)
    if parsed.scheme == "gs":
        return _publish_gcs_artifact(source=source, target_uri=target_uri)
    raise ValueError(f"Unsupported artifact publish target '{target_uri}'.")


def _write_alias(*, alias_dir: Path, run_dir: Path, model_name: str, model_type: str) -> Path:
    alias_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": str(run_dir.name),
        "run_dir": str(run_dir.resolve()),
        "model_name": str(model_name).strip(),
        "model_type": str(model_type).strip().lower(),
        "promoted_at": _utc_now_iso(),
    }
    return write_json(alias_dir / "alias.json", payload, sort_keys=True)


def _materialize_release_artifacts(*, run_dir: Path, release_dir: Path) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    candidate_files = [
        run_dir / "run_manifest.json",
        run_dir / "champion_gate.json",
        run_dir / "run_results.json",
        run_dir / "feature_metadata.json",
        run_dir / "context_scaler.joblib",
    ]
    candidate_files.extend(sorted((run_dir / "analysis" / "serving").glob("*")) if (run_dir / "analysis" / "serving").exists() else [])
    for source in candidate_files:
        if not source.exists() or not source.is_file():
            continue
        relative = source.relative_to(run_dir)
        destination = release_dir / relative
        materialize_cached_file(source, destination)
        copied.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "source_path": str(source.resolve()),
                "registry_path": str(destination.resolve()),
                "size_bytes": int(source.stat().st_size),
            }
        )
    return copied


def _publish_release_artifacts(
    release_artifacts: list[dict[str, object]],
    *,
    artifact_base_uri: str | None,
    release_id: str,
    publish_artifacts: bool,
) -> tuple[list[dict[str, object]], int]:
    published_count = 0
    enriched: list[dict[str, object]] = []
    for item in release_artifacts:
        relative_path = str(item.get("relative_path", "")).strip()
        target_uri = _artifact_uri_preview(
            artifact_base_uri=artifact_base_uri,
            release_id=release_id,
            relative_path=relative_path,
        )
        entry = dict(item)
        entry["artifact_uri"] = target_uri
        entry["artifact_published"] = False
        if publish_artifacts and target_uri:
            published_uri = _publish_artifact(source=Path(str(item["registry_path"])), target_uri=target_uri)
            entry["artifact_uri"] = published_uri
            entry["artifact_published"] = True
            published_count += 1
        enriched.append(entry)
    return enriched, published_count


def _available_serving_bundles(release_artifacts: list[dict[str, object]], *, artifact_base_uri: str | None, release_id: str) -> list[dict[str, object]]:
    bundles: list[dict[str, object]] = []
    for item in release_artifacts:
        relative_path = str(item.get("relative_path", ""))
        if not relative_path.startswith("analysis/serving/") or not relative_path.endswith(".joblib"):
            continue
        include_video = "audio_video" in relative_path
        artifact_uri = str(item.get("artifact_uri", "")).strip()
        bundles.append(
            {
                "relative_path": relative_path,
                "include_video": include_video,
                "artifact_uri": artifact_uri
                or _artifact_uri_preview(
                    artifact_base_uri=artifact_base_uri,
                    release_id=release_id,
                    relative_path=relative_path,
                ),
            }
        )
    return bundles


def _update_registry_index(
    *,
    registry_root: Path,
    channel: str,
    channel_manifest: dict[str, object],
    release_manifest: dict[str, object],
) -> None:
    index_path = registry_root / "registry_index.json"
    index_payload = safe_read_json(index_path, default={})
    if not isinstance(index_payload, dict):
        index_payload = {}
    channels = index_payload.get("channels", {})
    channels = channels if isinstance(channels, dict) else {}
    releases = index_payload.get("releases", {})
    releases = releases if isinstance(releases, dict) else {}

    channels[channel] = {
        "current_release_id": str(channel_manifest.get("current_release_id", "")),
        "previous_release_id": str(channel_manifest.get("previous_release_id", "")),
        "updated_at": str(channel_manifest.get("updated_at", "")),
        "channel_manifest_path": str((registry_root / "channels" / channel / "deployment_channel.json").resolve()),
    }
    release_id = str(release_manifest.get("release_id", ""))
    releases[release_id] = {
        "published_at": str(release_manifest.get("published_at", "")),
        "source_run_dir": str(release_manifest.get("source_run_dir", "")),
        "model_name": str(release_manifest.get("model_name", "")),
        "model_type": str(release_manifest.get("model_type", "")),
        "release_manifest_path": str((registry_root / "releases" / release_id / "deployment_release.json").resolve()),
    }

    write_json(
        index_path,
        {
            "updated_at": _utc_now_iso(),
            "channels": channels,
            "releases": releases,
        },
        sort_keys=True,
    )


def publish_deployment_release(
    *,
    run_dir: Path,
    outputs_dir: Path,
    registry_root: Path,
    channel: str,
    explicit_model_name: str | None = None,
    artifact_base_uri: str | None = None,
    publish_artifacts: bool = False,
    allow_unpromoted: bool = False,
    activate_channel: bool = True,
) -> dict[str, object]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_outputs_dir = outputs_dir.expanduser().resolve()
    resolved_registry_root = registry_root.expanduser().resolve()
    resolved_registry_root.mkdir(parents=True, exist_ok=True)

    gate_payload = safe_read_json(resolved_run_dir / "champion_gate.json", default={})
    if not isinstance(gate_payload, dict):
        gate_payload = {}
    promoted = bool(gate_payload.get("promoted", False))
    if not allow_unpromoted and gate_payload and not promoted:
        raise ValueError(
            f"Run {resolved_run_dir.name} is not promoted. Pass --allow-unpromoted to publish it anyway."
        )

    manifest_payload = safe_read_json(resolved_run_dir / "run_manifest.json", default={})
    manifest_payload = manifest_payload if isinstance(manifest_payload, dict) else {}
    manifest_alias = manifest_payload.get("champion_alias", {})
    alias_model_name = str(manifest_alias.get("model_name", "")).strip() if isinstance(manifest_alias, dict) else ""
    explicit_name = str(explicit_model_name).strip() if isinstance(explicit_model_name, str) else ""
    model_row = resolve_model_row(
        resolved_run_dir,
        explicit_model_name=(explicit_name or None),
        alias_model_name=(alias_model_name or None),
    )
    model_name = str(model_row.get("model_name", "")).strip()
    model_type = str(model_row.get("model_type", "")).strip().lower()

    if publish_artifacts and not str(artifact_base_uri or "").strip():
        raise ValueError("Publishing deployment artifacts requires --artifact-base-uri or SPOTIFY_DEPLOYMENT_ARTIFACT_BASE_URI.")

    release_id = resolved_run_dir.name
    release_dir = resolved_registry_root / "releases" / release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    channel_dir = resolved_registry_root / "channels" / str(channel).strip().lower()
    channel_dir.mkdir(parents=True, exist_ok=True)

    previous_channel_payload = safe_read_json(channel_dir / "deployment_channel.json", default={})
    previous_channel_payload = previous_channel_payload if isinstance(previous_channel_payload, dict) else {}
    previous_release_id = str(previous_channel_payload.get("current_release_id", "")).strip()

    release_alias_path = _write_alias(alias_dir=release_dir, run_dir=resolved_run_dir, model_name=model_name, model_type=model_type)
    release_artifacts = _materialize_release_artifacts(run_dir=resolved_run_dir, release_dir=release_dir)
    release_artifacts, published_artifact_count = _publish_release_artifacts(
        release_artifacts,
        artifact_base_uri=artifact_base_uri,
        release_id=release_id,
        publish_artifacts=publish_artifacts,
    )
    release_manifest = {
        "release_id": release_id,
        "published_at": _utc_now_iso(),
        "channel": str(channel).strip().lower(),
        "source_run_id": release_id,
        "source_run_dir": str(resolved_run_dir),
        "source_outputs_dir": str(resolved_outputs_dir),
        "source_promoted": promoted,
        "source_gate_status": str(gate_payload.get("status", "")),
        "model_name": model_name,
        "model_type": model_type,
        "release_dir": str(release_dir.resolve()),
        "release_alias_path": str(release_alias_path.resolve()),
        "artifact_base_uri": str(artifact_base_uri or ""),
        "artifact_publish_enabled": bool(publish_artifacts),
        "published_artifact_count": int(published_artifact_count),
        "artifacts": release_artifacts,
        "available_serving_bundles": _available_serving_bundles(
            release_artifacts,
            artifact_base_uri=artifact_base_uri,
            release_id=release_id,
        ),
        "activate_channel": bool(activate_channel),
    }
    release_manifest_path = write_json(release_dir / "deployment_release.json", release_manifest, sort_keys=True)
    available_bundles_obj = release_manifest.get("available_serving_bundles")
    available_bundles: list[object] = available_bundles_obj if isinstance(available_bundles_obj, list) else []

    channel_alias_path = None
    channel_manifest_path = None
    if activate_channel:
        channel_alias_path = _write_alias(
            alias_dir=channel_dir,
            run_dir=resolved_run_dir,
            model_name=model_name,
            model_type=model_type,
        )
        channel_manifest = {
            "channel": str(channel).strip().lower(),
            "updated_at": _utc_now_iso(),
            "current_release_id": release_id,
            "current_model_name": model_name,
            "current_model_type": model_type,
            "previous_release_id": previous_release_id,
            "rollback_release_id": previous_release_id,
            "source_promoted": promoted,
            "release_manifest_path": str(release_manifest_path.resolve()),
            "channel_alias_path": str(channel_alias_path.resolve()),
        }
        channel_manifest_path = write_json(channel_dir / "deployment_channel.json", channel_manifest, sort_keys=True)
        write_json(channel_dir / "release.json", release_manifest, sort_keys=True)
        _update_registry_index(
            registry_root=resolved_registry_root,
            channel=str(channel).strip().lower(),
            channel_manifest=channel_manifest,
            release_manifest=release_manifest,
        )

    return {
        "release_id": release_id,
        "model_name": model_name,
        "model_type": model_type,
        "promoted": promoted,
        "channel": str(channel).strip().lower(),
        "release_manifest_path": str(release_manifest_path.resolve()),
        "channel_manifest_path": str(channel_manifest_path.resolve()) if channel_manifest_path is not None else "",
        "channel_alias_path": str(channel_alias_path.resolve()) if channel_alias_path is not None else "",
        "artifact_count": len(release_artifacts),
        "published_artifact_count": int(published_artifact_count),
        "available_bundle_count": len(available_bundles),
        "previous_release_id": previous_release_id,
        "registry_root": str(resolved_registry_root),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.deployment_registry",
        description="Publish promoted Spotify runs into a deployment registry with rollout channels.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory or deployment alias path. Defaults to the current champion alias.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Outputs root used to resolve run metadata and default aliases.",
    )
    parser.add_argument(
        "--registry-root",
        type=str,
        default=os.getenv("SPOTIFY_DEPLOYMENT_REGISTRY_ROOT", DEFAULT_REGISTRY_ROOT),
        help="Local deployment registry root path or file:// URI.",
    )
    parser.add_argument("--channel", type=str, default="stable", help="Rollout channel to activate.")
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument(
        "--artifact-base-uri",
        type=str,
        default=os.getenv("SPOTIFY_DEPLOYMENT_ARTIFACT_BASE_URI", ""),
        help="Optional remote artifact base URI recorded in release metadata and used for publish targets.",
    )
    parser.add_argument(
        "--publish-artifacts",
        action="store_true",
        default=str(os.getenv("SPOTIFY_DEPLOYMENT_PUBLISH_ARTIFACTS", "")).strip().lower() in {"1", "true", "yes", "on"},
        help="Copy release artifacts to the artifact base URI (supports file://, s3://, and gs://).",
    )
    parser.add_argument(
        "--allow-unpromoted",
        action="store_true",
        help="Allow publishing a run even when champion_gate.json is present and not promoted.",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Publish the release metadata without moving the rollout channel.",
    )
    return parser.parse_args()


def main() -> int:
    load_local_env()
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    run_dir, _ = resolve_prediction_run_dir(args.run_dir, project_root=outputs_dir.parent)
    result = publish_deployment_release(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        registry_root=_resolve_registry_root(args.registry_root),
        channel=str(args.channel).strip().lower() or "stable",
        explicit_model_name=(str(args.model_name).strip() if args.model_name is not None else None) or None,
        artifact_base_uri=(str(args.artifact_base_uri).strip() or None),
        publish_artifacts=bool(args.publish_artifacts),
        allow_unpromoted=bool(args.allow_unpromoted),
        activate_channel=not bool(args.no_activate),
    )
    print(f"release_id={result['release_id']} channel={result['channel']} promoted={result['promoted']}")
    print(f"release_manifest={result['release_manifest_path']}")
    if str(result.get("channel_manifest_path", "")).strip():
        print(f"channel_manifest={result['channel_manifest_path']}")
    if str(result.get("channel_alias_path", "")).strip():
        print(f"channel_alias={result['channel_alias_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
