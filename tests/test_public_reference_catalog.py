from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from types import MappingProxyType

import pytest

from spotify.public_listening_reference import PublicListeningReference, PublicListeningRow
from spotify.public_reference_catalog import (
    PUBLIC_REFERENCE_SCHEMA_VERSION,
    PublicReferenceValidationError,
    bundled_public_reference_catalog,
    load_public_listening_reference,
    load_public_reference_catalog,
)


def _reference_payload() -> dict[str, object]:
    return {
        "provider": "Example Charts",
        "edition": 2024,
        "scope": "city",
        "country": "United States",
        "published_date": "2024-12-10",
        "window_start": "2024-01-01",
        "window_end": "2024-11-30",
        "window_end_is_approximate": False,
        "source_url": "https://example.com/charts",
        "methodology_url": "https://example.com/methodology",
        "dimensions": {
            "albums": [
                {"rank": 2, "name": "Second Album", "artists": ["Second Artist"]},
                {"rank": 1, "name": "First Album", "artists": ["First Artist"]},
            ],
            "audiobooks": [{"rank": 1, "name": "A Book"}],
        },
    }


def test_bundled_catalog_exposes_existing_spotify_wrapped_references() -> None:
    catalog = bundled_public_reference_catalog()

    global_reference = catalog.find(edition=2025, scope="GLOBAL", provider="spotify wrapped")
    country_reference = catalog.get(edition=2025, scope="country", country="united states")

    assert global_reference.dimensions["artists"][0].name == "Bad Bunny"
    assert country_reference.dimensions["tracks"][0].name == "luther (with sza)"
    assert isinstance(global_reference.dimensions, MappingProxyType)
    with pytest.raises(TypeError):
        global_reference.dimensions["new"] = ()  # type: ignore[index]


def test_loads_versioned_json_catalog_with_additional_dimensions(tmp_path) -> None:
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": PUBLIC_REFERENCE_SCHEMA_VERSION,
                "references": [_reference_payload()],
            }
        ),
        encoding="utf-8",
    )

    catalog = load_public_reference_catalog(path)
    reference = catalog.find(edition=2024, scope="city")

    assert len(catalog.references) == 3
    assert isinstance(reference, PublicListeningReference)
    assert reference.dimensions["albums"] == (
        PublicListeningRow(1, "First Album", ("First Artist",)),
        PublicListeningRow(2, "Second Album", ("Second Artist",)),
    )
    assert reference.dimensions["audiobooks"][0].artists == ()
    with pytest.raises(FrozenInstanceError):
        reference.edition = 2023  # type: ignore[misc]


def test_loads_single_reference_document_from_json_text() -> None:
    payload = {"schema_version": PUBLIC_REFERENCE_SCHEMA_VERSION, **_reference_payload()}

    reference = load_public_listening_reference(json.dumps(payload))

    assert reference.edition == 2024
    assert reference.scope == "city"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["dimensions"]["albums"].append(  # type: ignore[index,union-attr]
                {"rank": 1, "name": "Duplicate"}
            ),
            "duplicate ranks",
        ),
        (
            lambda payload: payload.update(window_start="2024-12-01", window_end="2024-11-30"),
            "window_start must be on or before window_end",
        ),
        (
            lambda payload: payload.update(window_end="2024-12-11"),
            "window_end must be on or before published_date",
        ),
    ],
)
def test_rejects_invalid_ranks_and_date_windows(mutate, message: str) -> None:
    reference = _reference_payload()
    mutate(reference)
    payload = {"schema_version": PUBLIC_REFERENCE_SCHEMA_VERSION, "references": [reference]}

    with pytest.raises(PublicReferenceValidationError, match=message):
        load_public_reference_catalog(payload, include_bundled=False)


def test_rejects_unsupported_schema_and_malformed_rows() -> None:
    with pytest.raises(PublicReferenceValidationError, match="unsupported version"):
        load_public_reference_catalog(
            {"schema_version": 2, "references": [_reference_payload()]},
            include_bundled=False,
        )

    reference = _reference_payload()
    reference["dimensions"] = {"tracks": [{"rank": True, "name": "Not a valid rank"}]}
    with pytest.raises(PublicReferenceValidationError, match="positive integer"):
        load_public_reference_catalog(
            {"schema_version": 1, "references": [reference]},
            include_bundled=False,
        )
