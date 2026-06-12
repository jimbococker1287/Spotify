from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from spotify.public_listening_reference import (
    PublicListeningReference,
    PublicListeningRow,
    spotify_wrapped_reference,
)


PUBLIC_REFERENCE_SCHEMA_VERSION = 1
PUBLIC_REFERENCE_CATALOG_SCHEMA_VERSION = PUBLIC_REFERENCE_SCHEMA_VERSION

_REFERENCE_FIELDS = {
    "provider",
    "edition",
    "scope",
    "country",
    "published_date",
    "window_start",
    "window_end",
    "window_end_is_approximate",
    "source_url",
    "methodology_url",
    "dimensions",
}
_ROW_FIELDS = {"rank", "name", "artists"}


class PublicReferenceValidationError(ValueError):
    """Raised when a public listening reference does not match the supported schema."""


@dataclass(frozen=True)
class PublicReferenceCatalog:
    schema_version: int
    references: tuple[PublicListeningReference, ...]

    def find(
        self,
        *,
        edition: int,
        scope: str,
        country: str | None = None,
        provider: str | None = None,
    ) -> PublicListeningReference:
        matches = self.select(
            edition=edition,
            scope=scope,
            country=country,
            provider=provider,
        )
        if not matches:
            qualifier = f", country={country!r}" if country is not None else ""
            raise LookupError(f"No public listening reference for edition={edition}, scope={scope!r}{qualifier}.")
        if len(matches) > 1:
            raise LookupError(
                "Multiple public listening references matched; specify provider and/or country to disambiguate."
            )
        return matches[0]

    def select(
        self,
        *,
        edition: int | None = None,
        scope: str | None = None,
        country: str | None = None,
        provider: str | None = None,
    ) -> tuple[PublicListeningReference, ...]:
        normalized_scope = _lookup_value(scope)
        normalized_country = _lookup_value(country)
        normalized_provider = _lookup_value(provider)
        return tuple(
            reference
            for reference in self.references
            if (edition is None or reference.edition == edition)
            and (scope is None or _lookup_value(reference.scope) == normalized_scope)
            and (country is None or _lookup_value(reference.country) == normalized_country)
            and (provider is None or _lookup_value(reference.provider) == normalized_provider)
        )

    def get(
        self,
        *,
        edition: int,
        scope: str,
        country: str | None = None,
        provider: str | None = None,
    ) -> PublicListeningReference:
        return self.find(edition=edition, scope=scope, country=country, provider=provider)


def bundled_public_reference_catalog() -> PublicReferenceCatalog:
    return PublicReferenceCatalog(
        schema_version=PUBLIC_REFERENCE_SCHEMA_VERSION,
        references=(
            _immutable_reference(spotify_wrapped_reference(edition=2025, scope="global")),
            _immutable_reference(spotify_wrapped_reference(edition=2025, scope="country")),
        ),
    )


def load_public_reference_catalog(
    source: str | bytes | Path | Mapping[str, Any],
    *,
    include_bundled: bool = True,
) -> PublicReferenceCatalog:
    payload = _decode_source(source)
    schema_version = _schema_version(payload)

    if "references" in payload:
        _reject_unknown_fields(payload, {"schema_version", "references"}, path="$")
        raw_references = payload["references"]
        if not isinstance(raw_references, list):
            _fail("$.references", "must be an array")
        if not raw_references:
            _fail("$.references", "must contain at least one reference")
        custom_references = tuple(
            _parse_reference(raw_reference, path=f"$.references[{index}]")
            for index, raw_reference in enumerate(raw_references)
        )
    else:
        raw_reference = dict(payload)
        raw_reference.pop("schema_version", None)
        custom_references = (_parse_reference(raw_reference, path="$"),)

    references = (
        bundled_public_reference_catalog().references + custom_references
        if include_bundled
        else custom_references
    )
    _validate_unique_references(references)
    return PublicReferenceCatalog(schema_version=schema_version, references=references)


def load_public_listening_reference(
    source: str | bytes | Path | Mapping[str, Any],
) -> PublicListeningReference:
    catalog = load_public_reference_catalog(source, include_bundled=False)
    if len(catalog.references) != 1:
        raise PublicReferenceValidationError("Expected exactly one public listening reference.")
    return catalog.references[0]


def _decode_source(source: str | bytes | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source

    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PublicReferenceValidationError("Public reference JSON must be UTF-8 encoded.") from exc
    elif isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith(("{", "[")):
            text = source
        else:
            text = Path(source).expanduser().read_text(encoding="utf-8")
    else:
        raise TypeError("Public reference source must be a path, JSON string, bytes, or mapping.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PublicReferenceValidationError(
            f"Invalid public reference JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc
    if not isinstance(payload, dict):
        _fail("$", "must be a JSON object")
    return payload


def _schema_version(payload: Mapping[str, Any]) -> int:
    if "schema_version" not in payload:
        _fail("$.schema_version", "is required")
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        _fail("$.schema_version", "must be an integer")
    if version != PUBLIC_REFERENCE_SCHEMA_VERSION:
        _fail(
            "$.schema_version",
            f"unsupported version {version!r}; expected {PUBLIC_REFERENCE_SCHEMA_VERSION}",
        )
    return version


def _parse_reference(value: Any, *, path: str) -> PublicListeningReference:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    _reject_unknown_fields(value, _REFERENCE_FIELDS, path=path)
    missing = sorted(_REFERENCE_FIELDS - value.keys())
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")

    provider = _nonempty_string(value["provider"], path=f"{path}.provider")
    edition = _positive_integer(value["edition"], path=f"{path}.edition")
    scope = _nonempty_string(value["scope"], path=f"{path}.scope").lower()
    country = _optional_string(value["country"], path=f"{path}.country")
    if scope == "global" and country is not None:
        _fail(f"{path}.country", "must be null when scope is 'global'")
    if scope == "country" and country is None:
        _fail(f"{path}.country", "is required when scope is 'country'")

    published_date = _date_value(value["published_date"], path=f"{path}.published_date")
    window_start = _date_value(value["window_start"], path=f"{path}.window_start")
    window_end = _date_value(value["window_end"], path=f"{path}.window_end")
    if window_start > window_end:
        _fail(path, "window_start must be on or before window_end")
    if window_end > published_date:
        _fail(path, "window_end must be on or before published_date")

    approximate = value["window_end_is_approximate"]
    if not isinstance(approximate, bool):
        _fail(f"{path}.window_end_is_approximate", "must be a boolean")

    source_url = _url(value["source_url"], path=f"{path}.source_url")
    methodology_url = _url(value["methodology_url"], path=f"{path}.methodology_url")
    dimensions = _parse_dimensions(value["dimensions"], path=f"{path}.dimensions")

    return PublicListeningReference(
        provider=provider,
        edition=edition,
        scope=scope,
        country=country,
        published_date=published_date,
        window_start=window_start,
        window_end=window_end,
        window_end_is_approximate=approximate,
        source_url=source_url,
        methodology_url=methodology_url,
        dimensions=dimensions,  # type: ignore[arg-type]
    )


def _parse_dimensions(value: Any, *, path: str) -> Mapping[str, tuple[PublicListeningRow, ...]]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    if not value:
        _fail(path, "must contain at least one dimension")

    dimensions: dict[str, tuple[PublicListeningRow, ...]] = {}
    for raw_name, raw_rows in value.items():
        name = _nonempty_string(raw_name, path=f"{path} key").lower()
        if name in dimensions:
            _fail(path, f"contains duplicate normalized dimension {name!r}")
        if not isinstance(raw_rows, list):
            _fail(f"{path}.{name}", "must be an array")
        if not raw_rows:
            _fail(f"{path}.{name}", "must contain at least one row")

        rows = tuple(
            _parse_row(raw_row, path=f"{path}.{name}[{index}]")
            for index, raw_row in enumerate(raw_rows)
        )
        ranks = [row.rank for row in rows]
        duplicate_ranks = sorted(rank for rank in set(ranks) if ranks.count(rank) > 1)
        if duplicate_ranks:
            _fail(f"{path}.{name}", f"contains duplicate ranks: {duplicate_ranks}")
        dimensions[name] = tuple(sorted(rows, key=lambda row: row.rank))
    return MappingProxyType(dimensions)


def _parse_row(value: Any, *, path: str) -> PublicListeningRow:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    _reject_unknown_fields(value, _ROW_FIELDS, path=path)
    missing = sorted({"rank", "name"} - value.keys())
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")

    artists_value = value.get("artists", [])
    if not isinstance(artists_value, list):
        _fail(f"{path}.artists", "must be an array")
    artists = tuple(
        _nonempty_string(artist, path=f"{path}.artists[{index}]")
        for index, artist in enumerate(artists_value)
    )
    return PublicListeningRow(
        rank=_positive_integer(value["rank"], path=f"{path}.rank"),
        name=_nonempty_string(value["name"], path=f"{path}.name"),
        artists=artists,
    )


def _immutable_reference(reference: PublicListeningReference) -> PublicListeningReference:
    dimensions = MappingProxyType(
        {
            name: tuple(
                PublicListeningRow(rank=row.rank, name=row.name, artists=tuple(row.artists))
                for row in rows
            )
            for name, rows in reference.dimensions.items()
        }
    )
    return PublicListeningReference(
        provider=reference.provider,
        edition=reference.edition,
        scope=reference.scope,
        country=reference.country,
        published_date=reference.published_date,
        window_start=reference.window_start,
        window_end=reference.window_end,
        window_end_is_approximate=reference.window_end_is_approximate,
        source_url=reference.source_url,
        methodology_url=reference.methodology_url,
        dimensions=dimensions,  # type: ignore[arg-type]
    )


def _validate_unique_references(references: Iterable[PublicListeningReference]) -> None:
    seen: set[tuple[str, int, str, str | None]] = set()
    for reference in references:
        identity = (
            reference.provider.casefold(),
            reference.edition,
            reference.scope.casefold(),
            _lookup_value(reference.country),
        )
        if identity in seen:
            raise PublicReferenceValidationError(
                "Duplicate public listening reference for "
                f"provider={reference.provider!r}, edition={reference.edition}, "
                f"scope={reference.scope!r}, country={reference.country!r}."
            )
        seen.add(identity)


def _date_value(value: Any, *, path: str) -> date:
    if not isinstance(value, str):
        _fail(path, "must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PublicReferenceValidationError(f"{path}: must be a valid ISO date (YYYY-MM-DD).") from exc
    if value != parsed.isoformat():
        _fail(path, "must use YYYY-MM-DD format")
    return parsed


def _url(value: Any, *, path: str) -> str:
    url = _nonempty_string(value, path=path)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        _fail(path, "must be an absolute HTTP or HTTPS URL")
    return url


def _positive_integer(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(path, "must be a positive integer")
    return value


def _nonempty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, path=path)


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail(path, f"contains unknown fields: {', '.join(unknown)}")


def _lookup_value(value: str | None) -> str | None:
    return value.strip().casefold() if value is not None else None


def _fail(path: str, message: str) -> None:
    raise PublicReferenceValidationError(f"{path}: {message}.")


__all__ = [
    "PUBLIC_REFERENCE_CATALOG_SCHEMA_VERSION",
    "PUBLIC_REFERENCE_SCHEMA_VERSION",
    "PublicReferenceCatalog",
    "PublicReferenceValidationError",
    "bundled_public_reference_catalog",
    "load_public_listening_reference",
    "load_public_reference_catalog",
]
