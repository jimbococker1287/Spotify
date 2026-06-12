from __future__ import annotations

import pytest

from spotify.public_listening_reference import spotify_wrapped_reference


def test_spotify_wrapped_2025_reference_has_global_and_us_dimensions() -> None:
    global_reference = spotify_wrapped_reference(edition=2025, scope="global")
    us_reference = spotify_wrapped_reference(edition=2025, scope="country")

    assert global_reference.dimensions["artists"][0].name == "Bad Bunny"
    assert global_reference.dimensions["tracks"][0].name == "Die With A Smile"
    assert global_reference.dimensions["podcasts"][0].name == "The Joe Rogan Experience"
    assert us_reference.country == "United States"
    assert us_reference.dimensions["artists"][0].name == "Taylor Swift"
    assert us_reference.window_end_is_approximate is True


def test_spotify_wrapped_reference_rejects_unknown_edition() -> None:
    with pytest.raises(ValueError, match="supports edition 2025 only"):
        spotify_wrapped_reference(edition=2024, scope="global")
