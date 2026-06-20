from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

from .public_training_data import (
    CANONICAL_INTERACTION_COLUMNS,
    CanonicalSchemaError,
    DatasetSourceManifest,
    SourceManifestError,
    load_source_manifest,
    validate_canonical_interactions,
    validate_source_manifest,
)


DEFAULT_SEARCH_ROOTS: tuple[Path, ...] = (
    Path("data"),
    Path("outputs/analysis/recommender_expansion"),
)
DEFAULT_MANIFEST_PATTERNS: tuple[str, ...] = (
    "**/*source_manifest*.json",
    "**/*dataset_manifest*.json",
    "**/*public*manifest*.json",
    "**/*manifest*.json",
)
_MANIFEST_SHAPE_KEYS = frozenset(
    {
        "dataset_id",
        "display_name",
        "adapter",
        "version",
        "required_columns",
        "license_name",
        "license_url",
        "access_url",
        "files",
    }
)

PathLike = str | Path
ReadinessStatus = Literal["ready", "blocked"]


@dataclass(frozen=True)
class PublicRecordPreflightResult:
    path: str
    status: ReadinessStatus
    row_count: int = 0
    source_datasets: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()
    min_timestamp: str = ""
    max_timestamp: str = ""
    reason: str = ""
    next_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return _json_friendly(asdict(self))


@dataclass(frozen=True)
class PublicManifestPreflightResult:
    manifest_path: str
    status: ReadinessStatus
    dataset_id: str = ""
    display_name: str = ""
    adapter: str = ""
    license_name: str = ""
    manifest_digest: str = ""
    training_use_approved: bool = False
    checksums_verified: bool = False
    source_files: tuple[dict[str, object], ...] = ()
    record_checks: tuple[PublicRecordPreflightResult, ...] = ()
    reason: str = ""
    next_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return _json_friendly(asdict(self))


@dataclass(frozen=True)
class PublicPretrainingPreflightReport:
    status: ReadinessStatus
    search_roots: tuple[str, ...]
    manifest_count: int
    ready_manifest_count: int
    blocked_manifest_count: int
    manifests: tuple[PublicManifestPreflightResult, ...] = ()
    next_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return _json_friendly(asdict(self))


def discover_public_pretraining_manifests(
    search_roots: Sequence[PathLike] = DEFAULT_SEARCH_ROOTS,
    *,
    patterns: Sequence[str] = DEFAULT_MANIFEST_PATTERNS,
) -> tuple[Path, ...]:
    """Find local JSON files that look like DatasetSourceManifest documents."""

    discovered: dict[Path, None] = {}
    for root in search_roots:
        if _is_network_reference(root):
            continue
        candidate_root = Path(root).expanduser()
        if candidate_root.is_file():
            if _looks_like_dataset_manifest(candidate_root):
                discovered[candidate_root.resolve()] = None
            continue
        if not candidate_root.is_dir():
            continue
        for pattern in patterns:
            for path in candidate_root.glob(pattern):
                if path.is_file() and _looks_like_dataset_manifest(path):
                    discovered[path.resolve()] = None
    return tuple(sorted(discovered))


def validate_public_pretraining_records(
    records: PathLike | pd.DataFrame,
    *,
    manifest: DatasetSourceManifest | None = None,
    max_rows: int | None = 100_000,
) -> PublicRecordPreflightResult:
    """Validate canonical public interaction records without creating training tensors."""

    label = "<dataframe>" if isinstance(records, pd.DataFrame) else str(records)
    try:
        frame = _load_record_frame(records, max_rows=max_rows)
        if frame.empty:
            raise CanonicalSchemaError("Canonical interaction records must contain at least one row.")
        validate_canonical_interactions(frame)
        if manifest is not None:
            source_datasets = _string_values(frame["source_dataset"])
            if source_datasets != (manifest.dataset_id,):
                raise CanonicalSchemaError(
                    "Canonical record source_dataset values must match the manifest dataset_id "
                    f"{manifest.dataset_id!r}; found {source_datasets!r}."
                )
        return PublicRecordPreflightResult(
            path=label,
            status="ready",
            row_count=int(len(frame)),
            source_datasets=_string_values(frame["source_dataset"]),
            splits=_string_values(frame["split"]) if "split" in frame.columns else (),
            min_timestamp=_timestamp_bound(frame, minimum=True),
            max_timestamp=_timestamp_bound(frame, minimum=False),
        )
    except Exception as exc:
        return PublicRecordPreflightResult(
            path=label,
            status="blocked",
            reason=f"{type(exc).__name__}: {exc}",
            next_actions=_record_next_actions(exc),
        )


def validate_public_pretraining_manifest(
    manifest: DatasetSourceManifest | PathLike,
    *,
    record_paths: Sequence[PathLike] = (),
    verify_checksums: bool = True,
    max_record_rows: int | None = 100_000,
) -> PublicManifestPreflightResult:
    """Validate one DatasetSourceManifest and optional canonical records without training."""

    manifest_path = "<in-memory>"
    try:
        if isinstance(manifest, DatasetSourceManifest):
            source_manifest = manifest
        else:
            manifest_path = str(Path(manifest).expanduser())
            source_manifest = load_source_manifest(manifest)
        digest = _manifest_digest(source_manifest)
        validate_source_manifest(source_manifest, verify_checksums=verify_checksums)
    except Exception as exc:
        return PublicManifestPreflightResult(
            manifest_path=manifest_path,
            status="blocked",
            reason=f"{type(exc).__name__}: {exc}",
            next_actions=_manifest_next_actions(exc),
        )

    record_checks = tuple(
        validate_public_pretraining_records(
            path,
            manifest=source_manifest,
            max_rows=max_record_rows,
        )
        for path in record_paths
    )
    blocked_record_checks = tuple(check for check in record_checks if check.status == "blocked")
    if blocked_record_checks:
        return PublicManifestPreflightResult(
            manifest_path=manifest_path,
            status="blocked",
            dataset_id=source_manifest.dataset_id,
            display_name=source_manifest.display_name,
            adapter=source_manifest.adapter,
            license_name=source_manifest.license_name,
            manifest_digest=digest,
            training_use_approved=bool(source_manifest.training_use_approved),
            checksums_verified=True,
            source_files=_source_files(source_manifest),
            record_checks=record_checks,
            reason="One or more canonical record checks failed.",
            next_actions=_unique_actions(
                action
                for check in blocked_record_checks
                for action in check.next_actions
            ),
        )

    return PublicManifestPreflightResult(
        manifest_path=manifest_path,
        status="ready",
        dataset_id=source_manifest.dataset_id,
        display_name=source_manifest.display_name,
        adapter=source_manifest.adapter,
        license_name=source_manifest.license_name,
        manifest_digest=digest,
        training_use_approved=bool(source_manifest.training_use_approved),
        checksums_verified=True,
        source_files=_source_files(source_manifest),
        record_checks=record_checks,
    )


def run_public_pretraining_preflight(
    *,
    search_roots: Sequence[PathLike] = DEFAULT_SEARCH_ROOTS,
    manifest_paths: Sequence[PathLike] = (),
    records_by_manifest: Mapping[str, Sequence[PathLike]] | None = None,
    verify_checksums: bool = True,
    max_record_rows: int | None = 100_000,
    output_path: PathLike | None = None,
) -> PublicPretrainingPreflightReport:
    """Run a local-only readiness preflight for governed public pretraining."""

    explicit_paths = tuple(Path(path).expanduser() for path in manifest_paths)
    discovered_paths = discover_public_pretraining_manifests(search_roots)
    candidate_paths = _dedupe_paths((*explicit_paths, *discovered_paths))
    if not candidate_paths:
        report = PublicPretrainingPreflightReport(
            status="blocked",
            search_roots=tuple(str(Path(root).expanduser()) for root in search_roots),
            manifest_count=0,
            ready_manifest_count=0,
            blocked_manifest_count=0,
            next_actions=(
                "Create a DatasetSourceManifest JSON for an approved local public dataset.",
                "Keep source files local and record files[].path, files[].sha256, size_bytes, source_url, and acquired_at.",
                "Set training_use_approved, reviewed_by, and reviewed_at only after license/provenance review.",
            ),
        )
        if output_path is not None:
            write_public_pretraining_preflight_report(report, output_path)
        return report

    records_by_manifest = records_by_manifest or {}
    manifest_results: list[PublicManifestPreflightResult] = []
    for path in candidate_paths:
        manifest_for_lookup = _load_manifest_for_lookup(path)
        record_paths = _records_for_manifest(
            records_by_manifest,
            manifest_path=path,
            dataset_id=manifest_for_lookup.dataset_id if manifest_for_lookup is not None else "",
        )
        manifest_results.append(
            validate_public_pretraining_manifest(
                path,
                record_paths=record_paths,
                verify_checksums=verify_checksums,
                max_record_rows=max_record_rows,
            )
        )

    ready_count = sum(1 for result in manifest_results if result.status == "ready")
    blocked_count = len(manifest_results) - ready_count
    report = PublicPretrainingPreflightReport(
        status="ready" if ready_count else "blocked",
        search_roots=tuple(str(Path(root).expanduser()) for root in search_roots),
        manifest_count=len(manifest_results),
        ready_manifest_count=ready_count,
        blocked_manifest_count=blocked_count,
        manifests=tuple(manifest_results),
        next_actions=_report_next_actions(manifest_results),
    )
    if output_path is not None:
        write_public_pretraining_preflight_report(report, output_path)
    return report


def write_public_pretraining_preflight_report(
    report: PublicPretrainingPreflightReport,
    path: PathLike,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def _looks_like_dataset_manifest(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and _MANIFEST_SHAPE_KEYS.issubset(payload)


def _load_manifest_for_lookup(path: Path) -> DatasetSourceManifest | None:
    try:
        return load_source_manifest(path)
    except Exception:
        return None


def _dedupe_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.expanduser().resolve()] = None
    return tuple(sorted(unique))


def _records_for_manifest(
    records_by_manifest: Mapping[str, Sequence[PathLike]],
    *,
    manifest_path: Path,
    dataset_id: str,
) -> tuple[PathLike, ...]:
    keys = (
        str(manifest_path),
        str(manifest_path.expanduser()),
        str(manifest_path.expanduser().resolve()),
        manifest_path.name,
        dataset_id,
        "*",
    )
    records: list[PathLike] = []
    for key in keys:
        records.extend(records_by_manifest.get(key, ()))
    return tuple(records)


def _load_record_frame(records: PathLike | pd.DataFrame, *, max_rows: int | None) -> pd.DataFrame:
    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError("max_rows must be positive when supplied.")
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
        return frame if max_rows is None else frame.head(int(max_rows)).copy()

    path = _require_local_record_path(records)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet") or suffixes.endswith(".pq"):
        frame = pd.read_parquet(path)
        return frame if max_rows is None else frame.head(int(max_rows)).copy()
    if suffixes.endswith(".jsonl") or suffixes.endswith(".ndjson"):
        return pd.read_json(path, lines=True, nrows=max_rows)
    if suffixes.endswith(".json"):
        frame = pd.read_json(path)
        return frame if max_rows is None else frame.head(int(max_rows)).copy()
    separator = "\t" if suffixes.endswith(".tsv") else ","
    return pd.read_csv(path, sep=separator, nrows=max_rows)


def _require_local_record_path(path: PathLike) -> Path:
    if _is_network_reference(path):
        raise SourceManifestError("Network URLs are not accepted; supply a reviewed local record file.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SourceManifestError(f"Local record file does not exist: {resolved}.")
    return resolved


def _source_files(manifest: DatasetSourceManifest) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "path": record.path,
            "sha256": record.sha256,
            "size_bytes": int(record.size_bytes),
            "source_url": record.source_url,
            "acquired_at": record.acquired_at,
            "original_filename": record.original_filename,
        }
        for record in manifest.files
    )


def _manifest_digest(manifest: DatasetSourceManifest) -> str:
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _string_values(series: pd.Series) -> tuple[str, ...]:
    values = series.dropna().astype("string").str.strip()
    return tuple(sorted({str(value) for value in values if str(value)}))


def _timestamp_bound(frame: pd.DataFrame, *, minimum: bool) -> str:
    if "timestamp" not in frame.columns:
        return ""
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    valid = timestamps.dropna()
    if valid.empty:
        return ""
    value = valid.min() if minimum else valid.max()
    return value.isoformat()


def _manifest_next_actions(exc: Exception) -> tuple[str, ...]:
    reason = str(exc).casefold()
    actions: list[str] = []
    if "not approved" in reason:
        actions.append(
            "Complete license/provenance review, then set training_use_approved=true only if training is permitted."
        )
    if "reviewed_by" in reason or "reviewed_at" in reason:
        actions.append("Record reviewed_by and timezone-aware reviewed_at in the manifest.")
    if "sha-256 mismatch" in reason or "size mismatch" in reason or "invalid sha-256" in reason:
        actions.append(
            "Recompute files[].sha256 and size_bytes from the reviewed local file, or restore the exact approved file."
        )
    if "does not exist" in reason:
        actions.append("Place the approved source file locally and update files[].path to that local path.")
    if "network urls" in reason:
        actions.append("Do not point manifests at remote downloads; use reviewed local files only.")
    if "spotify" in reason:
        actions.append(
            "Remove Spotify Platform/API-derived identifiers, URLs, metadata, and source references from training inputs."
        )
    if "missing required fields" in reason:
        actions.append("Fill every required DatasetSourceManifest field before approval.")
    if "required source columns" in reason:
        actions.append("Declare the source dataset columns required by the adapter.")
    if not actions:
        actions.append("Open the manifest, fix the reported governance error, and rerun the preflight.")
    return tuple(actions)


def _record_next_actions(exc: Exception) -> tuple[str, ...]:
    reason = str(exc).casefold()
    actions: list[str] = []
    if "missing canonical interaction columns" in reason:
        actions.append(
            "Normalize records to the canonical interaction schema: "
            + ", ".join(CANONICAL_INTERACTION_COLUMNS)
            + "."
        )
    if "source_dataset" in reason:
        actions.append("Set every source_dataset value to the approved manifest dataset_id.")
    if "item_id" in reason:
        actions.append("Populate item_id for every canonical interaction row.")
    if "propensity" in reason:
        actions.append("Keep propensity values in the interval (0, 1].")
    if "dwell_ms" in reason:
        actions.append("Keep dwell_ms empty or non-negative.")
    if "spotify" in reason:
        actions.append(
            "Remove Spotify Platform/API-derived identifiers, URLs, metadata, and source references from records."
        )
    if "network urls" in reason:
        actions.append("Supply a reviewed local canonical record file; preflight will not download remote data.")
    if "does not exist" in reason:
        actions.append("Write the canonical interaction file locally and pass that local path to preflight.")
    if "at least one row" in reason:
        actions.append("Provide at least one canonical public interaction row.")
    if not actions:
        actions.append("Fix the canonical record validation error and rerun the preflight.")
    return tuple(actions)


def _report_next_actions(
    manifest_results: Sequence[PublicManifestPreflightResult],
) -> tuple[str, ...]:
    blocked = [result for result in manifest_results if result.status == "blocked"]
    if not blocked:
        return ()
    return _unique_actions(action for result in blocked for action in result.next_actions)


def _unique_actions(actions: Sequence[str] | object) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for action in actions:  # type: ignore[union-attr]
        text = str(action)
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return tuple(unique)


def _is_network_reference(value: object) -> bool:
    text = str(value).strip().lower()
    return "://" in text and not text.startswith("file://")


def _json_friendly(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_friendly(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_friendly(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "DEFAULT_MANIFEST_PATTERNS",
    "DEFAULT_SEARCH_ROOTS",
    "PublicManifestPreflightResult",
    "PublicPretrainingPreflightReport",
    "PublicRecordPreflightResult",
    "discover_public_pretraining_manifests",
    "run_public_pretraining_preflight",
    "validate_public_pretraining_manifest",
    "validate_public_pretraining_records",
    "write_public_pretraining_preflight_report",
]
