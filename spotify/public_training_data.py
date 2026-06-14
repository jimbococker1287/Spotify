from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_INTERACTION_COLUMNS: tuple[str, ...] = (
    "source_dataset",
    "user_id",
    "session_id",
    "item_id",
    "timestamp",
    "event_type",
    "interaction_value",
    "dwell_ms",
    "reward",
    "propensity",
    "position",
    "explicit_positive",
    "policy_id",
    "split",
    "context_json",
)

CANONICAL_ITEM_COLUMNS: tuple[str, ...] = (
    "source_dataset",
    "item_id",
    "artist_id",
    "album_id",
    "item_name",
    "artist_name",
    "album_name",
    "genres",
    "duration_ms",
    "audio_path",
    "content_license",
    "metadata_json",
)

_CHECKSUM_CHUNK_BYTES = 1024 * 1024
_SPOTIFY_COLUMN_PATTERN = re.compile(
    r"(?:^|_)(?:spotify|spotify_api)(?:_|$)|spotify_(?:uri|url|id|audio|content)",
    flags=re.IGNORECASE,
)
_SPOTIFY_VALUE_PATTERN = re.compile(
    r"(?:spotify:(?:track|album|artist|playlist|episode):|"
    r"https?://(?:api|open|accounts)\.spotify\.com(?:/|$))",
    flags=re.IGNORECASE,
)
_SPOTIFY_MANIFEST_PATTERN = re.compile(
    r"(?:\bspotify\b|api\.spotify\.com|open\.spotify\.com|accounts\.spotify\.com)",
    flags=re.IGNORECASE,
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PublicTrainingDataError(ValueError):
    """Base error for public training-data ingestion."""


class SourceManifestError(PublicTrainingDataError):
    """Raised when source provenance or approval is incomplete."""


class SpotifyTrainingContentError(PublicTrainingDataError):
    """Raised when Spotify Platform/API content is detected in training input."""


class CanonicalSchemaError(PublicTrainingDataError):
    """Raised when normalized rows violate the canonical schema."""


@dataclass(frozen=True)
class SourceFileProvenance:
    path: str
    sha256: str
    size_bytes: int
    source_url: str
    acquired_at: str
    original_filename: str = ""
    notes: str = ""

    @classmethod
    def from_local_file(
        cls,
        path: str | Path,
        *,
        source_url: str,
        acquired_at: str | None = None,
        original_filename: str | None = None,
        notes: str = "",
    ) -> SourceFileProvenance:
        resolved = _require_local_file(path)
        return cls(
            path=str(resolved),
            sha256=file_sha256(resolved),
            size_bytes=resolved.stat().st_size,
            source_url=str(source_url).strip(),
            acquired_at=acquired_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            original_filename=original_filename or resolved.name,
            notes=str(notes).strip(),
        )


@dataclass(frozen=True)
class DatasetSourceManifest:
    dataset_id: str
    display_name: str
    adapter: str
    version: str
    task_fit: tuple[str, ...]
    required_columns: tuple[str, ...]
    license_name: str
    license_url: str
    access_url: str
    access_caveats: tuple[str, ...]
    files: tuple[SourceFileProvenance, ...]
    training_use_approved: bool = False
    reviewed_by: str = ""
    reviewed_at: str = ""
    citation: str = ""
    provenance_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetSourceManifest:
        raw_files = payload.get("files", ())
        files = tuple(
            entry if isinstance(entry, SourceFileProvenance) else SourceFileProvenance(**dict(entry))
            for entry in raw_files  # type: ignore[union-attr]
        )
        return cls(
            dataset_id=str(payload.get("dataset_id", "")),
            display_name=str(payload.get("display_name", "")),
            adapter=str(payload.get("adapter", "")),
            version=str(payload.get("version", "")),
            task_fit=tuple(str(value) for value in payload.get("task_fit", ())),  # type: ignore[arg-type]
            required_columns=tuple(str(value) for value in payload.get("required_columns", ())),  # type: ignore[arg-type]
            license_name=str(payload.get("license_name", "")),
            license_url=str(payload.get("license_url", "")),
            access_url=str(payload.get("access_url", "")),
            access_caveats=tuple(str(value) for value in payload.get("access_caveats", ())),  # type: ignore[arg-type]
            files=files,
            training_use_approved=bool(payload.get("training_use_approved", False)),
            reviewed_by=str(payload.get("reviewed_by", "")),
            reviewed_at=str(payload.get("reviewed_at", "")),
            citation=str(payload.get("citation", "")),
            provenance_notes=tuple(str(value) for value in payload.get("provenance_notes", ())),  # type: ignore[arg-type]
        )


DATASET_MANIFEST_TEMPLATES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "million_song_taste_profile": MappingProxyType(
            {
                "display_name": "Million Song Dataset Taste Profile Subset",
                "adapter": "million_song_taste_profile",
                "task_fit": ("implicit collaborative filtering", "retrieval", "long-tail evaluation"),
                "required_columns": ("user_id", "song_id", "play_count"),
                "license_name": "Dataset-specific terms; review required",
                "license_url": "https://millionsongdataset.com/tasteprofile/",
                "access_url": "https://millionsongdataset.com/tasteprofile/",
                "access_caveats": (
                    "The Taste Profile and linked Million Song components may have different terms.",
                    "Confirm the terms attached to the exact local archive before approving training use.",
                    "The triplets contain play counts but no event timestamps.",
                ),
            }
        ),
        "lfm": MappingProxyType(
            {
                "display_name": "LFM-style timestamped listening log",
                "adapter": "lfm_listening_log",
                "task_fit": ("sequential recommendation", "temporal retrieval", "session modeling"),
                "required_columns": ("user_id", "track_id", "timestamp"),
                "license_name": "Release-specific terms; review required",
                "license_url": "https://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf",
                "access_url": "https://www.cp.jku.at/datasets/",
                "access_caveats": (
                    "Availability and redistribution rights have changed across LFM releases.",
                    "Do not rely on an unofficial mirror without documenting its provenance and terms.",
                    "Review Last.fm-derived data restrictions before approving a local copy.",
                ),
            }
        ),
        "music4all": MappingProxyType(
            {
                "display_name": "Music4All / Music4All-Onion",
                "adapter": "music4all",
                "task_fit": ("multimodal recommendation", "cold start", "content-aware retrieval"),
                "required_columns": ("user_id", "item_id"),
                "license_name": "Release/component-specific terms; review required",
                "license_url": "https://zenodo.org/records/6609677",
                "access_url": "https://zenodo.org/records/6609677",
                "access_caveats": (
                    "Interactions, metadata, audio, images, text, and derived features may have different terms.",
                    "Approve each downloaded component separately and retain its original license files.",
                    "Do not ingest Spotify identifiers or Spotify API-derived attributes if present in a derivative copy.",
                ),
            }
        ),
        "fma": MappingProxyType(
            {
                "display_name": "Free Music Archive",
                "adapter": "fma_metadata",
                "task_fit": ("audio representation learning", "genre classification", "cold start"),
                "required_columns": ("track_id",),
                "license_name": "Metadata CC BY 4.0; audio licenses vary by track",
                "license_url": "https://github.com/mdeff/fma",
                "access_url": "https://github.com/mdeff/fma",
                "access_caveats": (
                    "FMA metadata is CC BY 4.0, while each audio track retains its artist-selected license.",
                    "Filter audio by compatible per-track license before training or redistribution.",
                    "Preserve attribution and the original track-license field.",
                ),
            }
        ),
        "kuairand": MappingProxyType(
            {
                "display_name": "KuaiRand",
                "adapter": "kuairand_policy_log",
                "task_fit": ("exposure debiasing", "sequential recommendation", "policy evaluation"),
                "required_columns": ("user_id", "video_id"),
                "license_name": "CC BY-SA 4.0",
                "license_url": "https://github.com/chongminggao/KuaiRand/blob/main/LICENSE",
                "access_url": "https://github.com/chongminggao/KuaiRand",
                "access_caveats": (
                    "Attribution and ShareAlike obligations apply.",
                    "Keep randomized-exposure indicators and feedback definitions in provenance.",
                    "This is a video-domain policy dataset, not a music-domain personalization corpus.",
                ),
            }
        ),
        "open_bandit": MappingProxyType(
            {
                "display_name": "Open Bandit Dataset",
                "adapter": "open_bandit_policy_log",
                "task_fit": ("off-policy evaluation", "counterfactual learning", "bandit evaluation"),
                "required_columns": ("item_id", "position", "click", "propensity_score"),
                "license_name": "CC BY 4.0; confirm intended-use terms",
                "license_url": "https://research.zozo.com/data.html",
                "access_url": "https://research.zozo.com/data.html",
                "access_caveats": (
                    "Provide attribution and review the dataset site's current intended-use language.",
                    "Preserve policy, campaign, position, and propensity fields for valid OPE.",
                    "This is a fashion-domain evaluation corpus and should not be mixed into music identity embeddings.",
                ),
            }
        ),
    }
)


def file_sha256(path: str | Path) -> str:
    resolved = _require_local_file(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as infile:
        while block := infile.read(_CHECKSUM_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def write_source_manifest(path: str | Path, manifest: DatasetSourceManifest) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return destination


def load_source_manifest(path: str | Path) -> DatasetSourceManifest:
    source = _require_local_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SourceManifestError("A dataset source manifest must contain a JSON object.")
    return DatasetSourceManifest.from_dict(payload)


def validate_source_manifest(
    manifest: DatasetSourceManifest,
    *,
    source_paths: Sequence[str | Path] = (),
    verify_checksums: bool = True,
) -> None:
    required_text = {
        "dataset_id": manifest.dataset_id,
        "display_name": manifest.display_name,
        "adapter": manifest.adapter,
        "version": manifest.version,
        "license_name": manifest.license_name,
        "license_url": manifest.license_url,
        "access_url": manifest.access_url,
    }
    missing = [name for name, value in required_text.items() if not str(value).strip()]
    if missing:
        raise SourceManifestError(f"Source manifest is missing required fields: {', '.join(missing)}.")
    if not manifest.task_fit:
        raise SourceManifestError("Source manifest must declare at least one task fit.")
    if not manifest.required_columns:
        raise SourceManifestError("Source manifest must declare required source columns.")
    if not manifest.access_caveats:
        raise SourceManifestError("Source manifest must preserve license/access caveats.")
    if not manifest.training_use_approved:
        raise SourceManifestError("Training use is not approved in the source manifest.")
    if not manifest.reviewed_by.strip() or not manifest.reviewed_at.strip():
        raise SourceManifestError("Approved training data must record reviewed_by and reviewed_at.")
    _parse_timestamp(manifest.reviewed_at, field_name="reviewed_at")
    if not manifest.files:
        raise SourceManifestError("Source manifest must include at least one local file with provenance.")

    manifest_text = _manifest_origin_text(manifest)
    if _SPOTIFY_MANIFEST_PATTERN.search(manifest_text) or _SPOTIFY_VALUE_PATTERN.search(manifest_text):
        raise SpotifyTrainingContentError(
            "Spotify Platform/API content is not permitted in public training-data manifests."
        )

    records_by_path: dict[Path, SourceFileProvenance] = {}
    for record in manifest.files:
        local_path = _require_local_file(record.path)
        if not _SHA256_PATTERN.fullmatch(record.sha256.lower()):
            raise SourceManifestError(f"Invalid SHA-256 for {local_path}.")
        if record.size_bytes != local_path.stat().st_size:
            raise SourceManifestError(f"Size mismatch for {local_path}.")
        if not record.source_url.strip() or not record.acquired_at.strip():
            raise SourceManifestError(f"Provenance URL and acquisition time are required for {local_path}.")
        _parse_timestamp(record.acquired_at, field_name=f"acquired_at for {local_path.name}")
        if verify_checksums and file_sha256(local_path) != record.sha256.lower():
            raise SourceManifestError(f"SHA-256 mismatch for {local_path}.")
        records_by_path[local_path] = record

    for source_path in source_paths:
        resolved = _require_local_file(source_path)
        if resolved not in records_by_path:
            raise SourceManifestError(f"Input file is not declared in the source manifest: {resolved}.")


def validate_no_spotify_training_content(
    frame: pd.DataFrame,
    *,
    manifest: DatasetSourceManifest | None = None,
) -> None:
    blocked_columns = [str(column) for column in frame.columns if _SPOTIFY_COLUMN_PATTERN.search(str(column))]
    if blocked_columns:
        raise SpotifyTrainingContentError(
            "Spotify-derived training columns are blocked: " + ", ".join(sorted(blocked_columns))
        )

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype("string")
        if not values.empty and bool(values.str.contains(_SPOTIFY_VALUE_PATTERN, regex=True, na=False).any()):
            raise SpotifyTrainingContentError(
                f"Spotify URI/API URL detected in training values for column {column!r}."
            )

    if manifest is not None:
        manifest_text = _manifest_origin_text(manifest)
        if _SPOTIFY_MANIFEST_PATTERN.search(manifest_text) or _SPOTIFY_VALUE_PATTERN.search(manifest_text):
            raise SpotifyTrainingContentError(
                "Spotify Platform/API content is not permitted in public training-data manifests."
            )


def validate_canonical_interactions(frame: pd.DataFrame) -> None:
    _require_columns(frame, CANONICAL_INTERACTION_COLUMNS, label="canonical interaction")
    if frame["source_dataset"].isna().any() or (frame["source_dataset"].astype("string").str.strip() == "").any():
        raise CanonicalSchemaError("Canonical interactions require source_dataset on every row.")
    if frame["item_id"].isna().any() or (frame["item_id"].astype("string").str.strip() == "").any():
        raise CanonicalSchemaError("Canonical interactions require item_id on every row.")

    propensity = pd.to_numeric(frame["propensity"], errors="coerce")
    supplied_propensity = frame["propensity"].notna()
    invalid_propensity = supplied_propensity & ((propensity <= 0.0) | (propensity > 1.0) | propensity.isna())
    if bool(invalid_propensity.any()):
        raise CanonicalSchemaError("Propensity values must be in the interval (0, 1].")

    dwell_ms = pd.to_numeric(frame["dwell_ms"], errors="coerce")
    if bool((frame["dwell_ms"].notna() & ((dwell_ms < 0) | dwell_ms.isna())).any()):
        raise CanonicalSchemaError("dwell_ms values must be non-negative numbers.")

    validate_no_spotify_training_content(frame)


def validate_canonical_items(frame: pd.DataFrame) -> None:
    _require_columns(frame, CANONICAL_ITEM_COLUMNS, label="canonical item")
    if frame["source_dataset"].isna().any() or (frame["source_dataset"].astype("string").str.strip() == "").any():
        raise CanonicalSchemaError("Canonical items require source_dataset on every row.")
    if frame["item_id"].isna().any() or (frame["item_id"].astype("string").str.strip() == "").any():
        raise CanonicalSchemaError("Canonical items require item_id on every row.")
    duration_ms = pd.to_numeric(frame["duration_ms"], errors="coerce")
    if bool((frame["duration_ms"].notna() & ((duration_ms < 0) | duration_ms.isna())).any()):
        raise CanonicalSchemaError("duration_ms values must be non-negative numbers.")
    validate_no_spotify_training_content(frame)


def iter_million_song_taste_profile(
    path: str | Path,
    *,
    manifest: DatasetSourceManifest,
    chunksize: int = 100_000,
    has_header: bool = False,
    verify_checksum: bool = True,
) -> Iterator[pd.DataFrame]:
    source = _prepare_source(path, manifest, adapter="million_song_taste_profile", verify_checksum=verify_checksum)
    reader = pd.read_csv(
        source,
        sep="\t",
        header=0 if has_header else None,
        names=None if has_header else ["user_id", "song_id", "play_count"],
        chunksize=_valid_chunksize(chunksize),
        low_memory=False,
    )
    for raw in reader:
        validate_no_spotify_training_content(raw, manifest=manifest)
        columns = _resolve_columns(
            raw,
            {
                "user_id": ("user_id", "user", "user-id"),
                "item_id": ("song_id", "track_id", "item_id", "song"),
                "interaction_value": ("play_count", "count", "plays"),
            },
            required=("user_id", "item_id", "interaction_value"),
        )
        canonical = _interaction_frame(
            raw,
            manifest=manifest,
            columns=columns,
            event_type="play_count",
            explicit_positive=True,
        )
        yield canonical


def iter_lfm_listening_logs(
    path: str | Path,
    *,
    manifest: DatasetSourceManifest,
    chunksize: int = 100_000,
    sep: str = "\t",
    has_header: bool = True,
    column_map: Mapping[str, str] | None = None,
    verify_checksum: bool = True,
) -> Iterator[pd.DataFrame]:
    source = _prepare_source(path, manifest, adapter="lfm_listening_log", verify_checksum=verify_checksum)
    positional_names = ["user_id", "artist_id", "album_id", "track_id", "timestamp"]
    reader = pd.read_csv(
        source,
        sep=sep,
        header=0 if has_header else None,
        names=None if has_header else positional_names,
        chunksize=_valid_chunksize(chunksize),
        low_memory=False,
    )
    for raw in reader:
        validate_no_spotify_training_content(raw, manifest=manifest)
        columns = _resolve_columns(
            raw,
            {
                "user_id": ("user_id", "user", "user-id"),
                "item_id": ("track_id", "item_id", "track-id", "music_id"),
                "timestamp": ("timestamp", "ts", "time", "listened_at"),
                "artist_id": ("artist_id", "artist-id"),
                "album_id": ("album_id", "album-id"),
                "item_name": ("track_name", "track", "title"),
                "artist_name": ("artist_name", "artist"),
                "album_name": ("album_name", "album"),
            },
            required=("user_id", "item_id", "timestamp"),
            explicit=column_map,
        )
        yield _interaction_frame(
            raw,
            manifest=manifest,
            columns=columns,
            event_type="listen",
            explicit_positive=True,
        )


def iter_music4all_interactions(
    path: str | Path,
    *,
    manifest: DatasetSourceManifest,
    chunksize: int = 100_000,
    sep: str = "\t",
    column_map: Mapping[str, str] | None = None,
    verify_checksum: bool = True,
) -> Iterator[pd.DataFrame]:
    source = _prepare_source(path, manifest, adapter="music4all", verify_checksum=verify_checksum)
    reader = pd.read_csv(source, sep=sep, chunksize=_valid_chunksize(chunksize), low_memory=False)
    for raw in reader:
        validate_no_spotify_training_content(raw, manifest=manifest)
        columns = _resolve_columns(
            raw,
            {
                "user_id": ("user_id", "user", "userid"),
                "item_id": ("item_id", "music_id", "track_id", "song_id"),
                "timestamp": ("timestamp", "ts", "time"),
                "interaction_value": ("play_count", "count", "plays", "weight"),
                "split": ("split", "set", "partition"),
            },
            required=("user_id", "item_id"),
            explicit=column_map,
        )
        yield _interaction_frame(
            raw,
            manifest=manifest,
            columns=columns,
            event_type="listen",
            explicit_positive=True,
        )


def iter_music4all_metadata(
    path: str | Path,
    *,
    manifest: DatasetSourceManifest,
    chunksize: int = 50_000,
    sep: str = "\t",
    column_map: Mapping[str, str] | None = None,
    verify_checksum: bool = True,
) -> Iterator[pd.DataFrame]:
    source = _prepare_source(path, manifest, adapter="music4all", verify_checksum=verify_checksum)
    reader = pd.read_csv(source, sep=sep, chunksize=_valid_chunksize(chunksize), low_memory=False)
    for raw in reader:
        validate_no_spotify_training_content(raw, manifest=manifest)
        columns = _resolve_columns(
            raw,
            {
                "item_id": ("item_id", "music_id", "track_id", "song_id", "id"),
                "artist_id": ("artist_id", "artistid"),
                "album_id": ("album_id", "albumid"),
                "item_name": ("track_name", "song_name", "title", "name"),
                "artist_name": ("artist_name", "artist"),
                "album_name": ("album_name", "album"),
                "genres": ("genres", "genre", "tags"),
                "duration_ms": ("duration_ms", "length_ms"),
                "audio_path": ("audio_path", "path", "clip_path"),
                "content_license": ("license", "content_license"),
            },
            required=("item_id",),
            explicit=column_map,
        )
        yield _item_frame(raw, manifest=manifest, columns=columns)


def iter_fma_metadata(
    path: str | Path,
    *,
    manifest: DatasetSourceManifest,
    chunksize: int = 50_000,
    raw_tracks_csv: bool = True,
    column_map: Mapping[str, str] | None = None,
    verify_checksum: bool = True,
) -> Iterator[pd.DataFrame]:
    source = _prepare_source(path, manifest, adapter="fma_metadata", verify_checksum=verify_checksum)
    if raw_tracks_csv:
        reader = pd.read_csv(
            source,
            header=[0, 1],
            index_col=0,
            chunksize=_valid_chunksize(chunksize),
            low_memory=False,
        )
    else:
        reader = pd.read_csv(source, chunksize=_valid_chunksize(chunksize), low_memory=False)

    for raw in reader:
        if raw_tracks_csv:
            raw = raw.reset_index()
            raw.columns = [_flatten_column(column) for column in raw.columns]
        validate_no_spotify_training_content(raw, manifest=manifest)
        columns = _resolve_columns(
            raw,
            {
                "item_id": ("track_id", "trackid", "index", "track"),
                "artist_id": ("artist_id",),
                "album_id": ("album_id",),
                "item_name": ("track_title", "title", "track_name"),
                "artist_name": ("artist_name", "artist"),
                "album_name": ("album_title", "album_name", "album"),
                "genres": ("track_genres_all", "track_genres", "genres", "genre_top"),
                "duration_ms": ("duration_ms", "track_duration_ms"),
                "duration_seconds": ("track_duration", "duration"),
                "audio_path": ("audio_path", "path"),
                "content_license": ("track_license", "license"),
            },
            required=("item_id",),
            explicit=column_map,
        )
        yield _item_frame(raw, manifest=manifest, columns=columns)


def iter_kuairand_policy_logs(
    path: str | Path,
    *,
    manifest: DatasetSourceManifest,
    chunksize: int = 100_000,
    column_map: Mapping[str, str] | None = None,
    verify_checksum: bool = True,
) -> Iterator[pd.DataFrame]:
    source = _prepare_source(path, manifest, adapter="kuairand_policy_log", verify_checksum=verify_checksum)
    reader = pd.read_csv(source, chunksize=_valid_chunksize(chunksize), low_memory=False)
    for raw in reader:
        validate_no_spotify_training_content(raw, manifest=manifest)
        columns = _resolve_columns(
            raw,
            {
                "user_id": ("user_id", "userid"),
                "item_id": ("video_id", "item_id"),
                "timestamp": ("timestamp", "time_ms", "ts"),
                "dwell_ms": ("play_time_ms", "playtime_ms", "watch_time_ms"),
                "reward": ("is_click", "click", "long_view"),
                "propensity": ("propensity", "propensity_score", "pscore"),
                "position": ("position", "rank"),
                "policy_id": ("policy_id", "tab", "is_rand"),
                "session_id": ("session_id",),
            },
            required=("user_id", "item_id"),
            explicit=column_map,
        )
        yield _interaction_frame(
            raw,
            manifest=manifest,
            columns=columns,
            event_type="exposure",
            explicit_positive=None,
        )


def iter_open_bandit_policy_logs(
    path: str | Path,
    *,
    manifest: DatasetSourceManifest,
    chunksize: int = 100_000,
    column_map: Mapping[str, str] | None = None,
    verify_checksum: bool = True,
) -> Iterator[pd.DataFrame]:
    source = _prepare_source(path, manifest, adapter="open_bandit_policy_log", verify_checksum=verify_checksum)
    reader = pd.read_csv(source, chunksize=_valid_chunksize(chunksize), low_memory=False)
    for raw in reader:
        validate_no_spotify_training_content(raw, manifest=manifest)
        columns = _resolve_columns(
            raw,
            {
                "user_id": ("user_id", "userid"),
                "item_id": ("item_id", "action"),
                "timestamp": ("timestamp", "ts"),
                "reward": ("click", "reward"),
                "propensity": ("propensity_score", "pscore", "propensity"),
                "position": ("position", "rank"),
                "policy_id": ("policy", "policy_id"),
                "split": ("campaign", "split"),
            },
            required=("item_id", "reward", "propensity", "position"),
            explicit=column_map,
        )
        yield _interaction_frame(
            raw,
            manifest=manifest,
            columns=columns,
            event_type="impression",
            explicit_positive=None,
        )


def _prepare_source(
    path: str | Path,
    manifest: DatasetSourceManifest,
    *,
    adapter: str,
    verify_checksum: bool,
) -> Path:
    source = _require_local_file(path)
    if manifest.adapter != adapter:
        raise SourceManifestError(
            f"Manifest adapter {manifest.adapter!r} does not match requested adapter {adapter!r}."
        )
    validate_source_manifest(manifest, source_paths=(source,), verify_checksums=verify_checksum)
    return source


def _interaction_frame(
    raw: pd.DataFrame,
    *,
    manifest: DatasetSourceManifest,
    columns: Mapping[str, str],
    event_type: str,
    explicit_positive: bool | None,
) -> pd.DataFrame:
    promoted_fields = {
        "user_id",
        "session_id",
        "item_id",
        "timestamp",
        "interaction_value",
        "dwell_ms",
        "reward",
        "propensity",
        "position",
        "policy_id",
        "split",
    }
    frame = pd.DataFrame(index=raw.index)
    frame["source_dataset"] = manifest.dataset_id
    frame["user_id"] = _string_column(raw, columns.get("user_id"))
    frame["session_id"] = _string_column(raw, columns.get("session_id"))
    frame["item_id"] = _string_column(raw, columns.get("item_id"))
    frame["timestamp"] = _timestamp_column(raw, columns.get("timestamp"))
    frame["event_type"] = event_type
    frame["interaction_value"] = _numeric_column(raw, columns.get("interaction_value"))
    frame["dwell_ms"] = _numeric_column(raw, columns.get("dwell_ms"))
    frame["reward"] = _numeric_column(raw, columns.get("reward"))
    frame["propensity"] = _numeric_column(raw, columns.get("propensity"))
    frame["position"] = _integer_column(raw, columns.get("position"))
    frame["explicit_positive"] = (
        pd.Series(pd.array([explicit_positive] * len(raw), dtype="boolean"), index=raw.index)
        if explicit_positive is not None
        else pd.Series(pd.array([pd.NA] * len(raw), dtype="boolean"), index=raw.index)
    )
    frame["policy_id"] = _string_column(raw, columns.get("policy_id"))
    frame["split"] = _string_column(raw, columns.get("split"))
    frame["context_json"] = _extra_json(
        raw,
        used_columns={source for canonical, source in columns.items() if canonical in promoted_fields},
    )
    frame = frame.reset_index(drop=True).loc[:, CANONICAL_INTERACTION_COLUMNS]
    validate_canonical_interactions(frame)
    return frame


def _item_frame(
    raw: pd.DataFrame,
    *,
    manifest: DatasetSourceManifest,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw.index)
    frame["source_dataset"] = manifest.dataset_id
    frame["item_id"] = _string_column(raw, columns.get("item_id"))
    frame["artist_id"] = _string_column(raw, columns.get("artist_id"))
    frame["album_id"] = _string_column(raw, columns.get("album_id"))
    frame["item_name"] = _string_column(raw, columns.get("item_name"))
    frame["artist_name"] = _string_column(raw, columns.get("artist_name"))
    frame["album_name"] = _string_column(raw, columns.get("album_name"))
    frame["genres"] = _string_column(raw, columns.get("genres"))
    duration_ms = _numeric_column(raw, columns.get("duration_ms"))
    if columns.get("duration_ms") is None and columns.get("duration_seconds") is not None:
        duration_ms = _numeric_column(raw, columns["duration_seconds"]) * 1000.0
    frame["duration_ms"] = duration_ms
    frame["audio_path"] = _string_column(raw, columns.get("audio_path"))
    frame["content_license"] = _string_column(raw, columns.get("content_license"))
    frame["metadata_json"] = _extra_json(raw, used_columns=set(columns.values()))
    frame = frame.reset_index(drop=True).loc[:, CANONICAL_ITEM_COLUMNS]
    validate_canonical_items(frame)
    return frame


def _resolve_columns(
    frame: pd.DataFrame,
    aliases: Mapping[str, Sequence[str]],
    *,
    required: Sequence[str],
    explicit: Mapping[str, str] | None = None,
) -> dict[str, str]:
    normalized_lookup = {_normalize_column(column): str(column) for column in frame.columns}
    resolved: dict[str, str] = {}
    for canonical, source in (explicit or {}).items():
        if canonical not in aliases:
            raise CanonicalSchemaError(f"Unknown canonical column in column_map: {canonical!r}.")
        if source not in frame.columns:
            raise CanonicalSchemaError(f"Mapped source column is missing: {source!r}.")
        resolved[canonical] = source

    for canonical, candidates in aliases.items():
        if canonical in resolved:
            continue
        for candidate in candidates:
            match = normalized_lookup.get(_normalize_column(candidate))
            if match is not None:
                resolved[canonical] = match
                break

    missing = [column for column in required if column not in resolved]
    if missing:
        raise CanonicalSchemaError(
            "Source data is missing columns needed for canonical fields: " + ", ".join(missing)
        )
    return resolved


def _extra_json(frame: pd.DataFrame, *, used_columns: set[str]) -> pd.Series:
    extra_columns = [column for column in frame.columns if str(column) not in used_columns]
    if not extra_columns:
        return pd.Series(["{}"] * len(frame), index=frame.index, dtype="string")
    records = frame.loc[:, extra_columns].replace({np.nan: None}).to_dict(orient="records")
    values = [json.dumps(record, sort_keys=True, default=_json_default) for record in records]
    return pd.Series(values, index=frame.index, dtype="string")


def _string_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(pd.array([pd.NA] * len(frame), dtype="string"), index=frame.index)
    values = frame[column].astype("string").str.strip()
    return values.mask(values == "")


def _numeric_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _integer_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(pd.array([pd.NA] * len(frame), dtype="Int64"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").astype("Int64")


def _timestamp_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(values) else 0.0
    if numeric_ratio >= 0.9:
        magnitude = float(numeric.dropna().abs().median()) if numeric.notna().any() else 0.0
        unit = "ms" if magnitude >= 100_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(values, errors="coerce", utc=True)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CanonicalSchemaError(f"Missing {label} columns: {', '.join(missing)}.")


def _require_local_file(path: str | Path) -> Path:
    raw = str(path)
    if "://" in raw:
        raise SourceManifestError("Network URLs are not accepted; download and review the source file separately.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SourceManifestError(f"Local source file does not exist: {resolved}.")
    return resolved


def _valid_chunksize(chunksize: int) -> int:
    value = int(chunksize)
    if value <= 0:
        raise ValueError("chunksize must be a positive integer.")
    return value


def _normalize_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _flatten_column(column: object) -> str:
    if not isinstance(column, tuple):
        return _normalize_column(column)
    parts = [
        _normalize_column(part)
        for part in column
        if str(part).strip() and not str(part).lower().startswith("unnamed")
    ]
    return "_".join(part for part in parts if part)


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceManifestError(f"{field_name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise SourceManifestError(f"{field_name} must include a timezone.")
    return parsed


def _manifest_origin_text(manifest: DatasetSourceManifest) -> str:
    source_identity = {
        "dataset_id": manifest.dataset_id,
        "display_name": manifest.display_name,
        "adapter": manifest.adapter,
        "license_name": manifest.license_name,
        "license_url": manifest.license_url,
        "access_url": manifest.access_url,
        "files": [
            {
                "path": record.path,
                "source_url": record.source_url,
                "original_filename": record.original_filename,
            }
            for record in manifest.files
        ],
    }
    return json.dumps(source_identity, sort_keys=True)


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


__all__ = [
    "CANONICAL_INTERACTION_COLUMNS",
    "CANONICAL_ITEM_COLUMNS",
    "DATASET_MANIFEST_TEMPLATES",
    "CanonicalSchemaError",
    "DatasetSourceManifest",
    "PublicTrainingDataError",
    "SourceFileProvenance",
    "SourceManifestError",
    "SpotifyTrainingContentError",
    "file_sha256",
    "iter_fma_metadata",
    "iter_kuairand_policy_logs",
    "iter_lfm_listening_logs",
    "iter_million_song_taste_profile",
    "iter_music4all_interactions",
    "iter_music4all_metadata",
    "iter_open_bandit_policy_logs",
    "load_source_manifest",
    "validate_canonical_interactions",
    "validate_canonical_items",
    "validate_no_spotify_training_content",
    "validate_source_manifest",
    "write_source_manifest",
]
