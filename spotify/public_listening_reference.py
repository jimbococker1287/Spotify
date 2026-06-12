from __future__ import annotations

from dataclasses import dataclass
from datetime import date


SPOTIFY_2025_SOURCE_URL = (
    "https://newsroom.spotify.com/2025-12-03/wrapped-top-artists-songs-albums-podcasts-audiobooks/"
)
SPOTIFY_2025_METHOD_URL = "https://newsroom.spotify.com/2025-12-03/how-your-wrapped-is-made/"


@dataclass(frozen=True)
class PublicListeningRow:
    rank: int
    name: str
    artists: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicListeningReference:
    provider: str
    edition: int
    scope: str
    country: str | None
    published_date: date
    window_start: date
    window_end: date
    window_end_is_approximate: bool
    source_url: str
    methodology_url: str
    dimensions: dict[str, tuple[PublicListeningRow, ...]]


_GLOBAL_2025 = {
    "artists": (
        PublicListeningRow(1, "Bad Bunny"),
        PublicListeningRow(2, "Taylor Swift"),
        PublicListeningRow(3, "The Weeknd"),
        PublicListeningRow(4, "Drake"),
        PublicListeningRow(5, "Billie Eilish"),
        PublicListeningRow(6, "Kendrick Lamar"),
        PublicListeningRow(7, "Bruno Mars"),
        PublicListeningRow(8, "Ariana Grande"),
        PublicListeningRow(9, "Arijit Singh"),
        PublicListeningRow(10, "Fuerza Regida"),
    ),
    "tracks": (
        PublicListeningRow(1, "Die With A Smile", ("Lady Gaga", "Bruno Mars")),
        PublicListeningRow(2, "BIRDS OF A FEATHER", ("Billie Eilish",)),
        PublicListeningRow(3, "APT.", ("ROSE", "Bruno Mars")),
        PublicListeningRow(4, "Ordinary", ("Alex Warren",)),
        PublicListeningRow(5, "DtMF", ("Bad Bunny",)),
        PublicListeningRow(6, "back to friends", ("sombr",)),
        PublicListeningRow(
            7,
            "Golden",
            ("HUNTR/X", "EJAE", "AUDREY NUNA", "REI AMI", "KPop Demon Hunters Cast"),
        ),
        PublicListeningRow(8, "luther (with sza)", ("Kendrick Lamar", "SZA")),
        PublicListeningRow(9, "That's So True", ("Gracie Abrams",)),
        PublicListeningRow(10, "Wildflower", ("Billie Eilish",)),
    ),
    "podcasts": (
        PublicListeningRow(1, "The Joe Rogan Experience"),
        PublicListeningRow(2, "The Diary Of A CEO with Steven Bartlett"),
        PublicListeningRow(3, "The Mel Robbins Podcast"),
        PublicListeningRow(4, "Call Her Daddy"),
        PublicListeningRow(5, "This Past Weekend w/ Theo Von"),
        PublicListeningRow(6, "Huberman Lab"),
        PublicListeningRow(7, "Crime Junkie"),
        PublicListeningRow(8, "Modern Wisdom"),
        PublicListeningRow(9, "On Purpose with Jay Shetty"),
        PublicListeningRow(10, "The Tucker Carlson Show"),
    ),
}

_US_2025 = {
    "artists": (
        PublicListeningRow(1, "Taylor Swift"),
        PublicListeningRow(2, "Drake"),
        PublicListeningRow(3, "Morgan Wallen"),
        PublicListeningRow(4, "Kendrick Lamar"),
        PublicListeningRow(5, "Bad Bunny"),
        PublicListeningRow(6, "The Weeknd"),
        PublicListeningRow(7, "SZA"),
        PublicListeningRow(8, "Zach Bryan"),
        PublicListeningRow(9, "Tyler, The Creator"),
        PublicListeningRow(10, "Kanye West"),
    ),
    "tracks": (
        PublicListeningRow(1, "luther (with sza)", ("Kendrick Lamar", "SZA")),
        PublicListeningRow(2, "Die With A Smile", ("Lady Gaga", "Bruno Mars")),
        PublicListeningRow(3, "Ordinary", ("Alex Warren",)),
        PublicListeningRow(4, "BIRDS OF A FEATHER", ("Billie Eilish",)),
        PublicListeningRow(5, "tv off (feat. lefty gunplay)", ("Kendrick Lamar", "Lefty Gunplay")),
        PublicListeningRow(
            6,
            "Golden",
            ("HUNTR/X", "EJAE", "AUDREY NUNA", "REI AMI", "KPop Demon Hunters Cast"),
        ),
        PublicListeningRow(7, "back to friends", ("sombr",)),
        PublicListeningRow(8, "Pink Pony Club", ("Chappell Roan",)),
        PublicListeningRow(9, "Timeless (feat. Playboi Carti)", ("The Weeknd", "Playboi Carti")),
        PublicListeningRow(10, "No One Noticed", ("The Marias",)),
    ),
    "podcasts": (
        PublicListeningRow(1, "The Joe Rogan Experience"),
        PublicListeningRow(2, "This Past Weekend w/ Theo Von"),
        PublicListeningRow(3, "The Mel Robbins Podcast"),
        PublicListeningRow(4, "Call Her Daddy"),
        PublicListeningRow(5, "Crime Junkie"),
        PublicListeningRow(6, "The Shawn Ryan Show"),
        PublicListeningRow(7, "The Tucker Carlson Show"),
        PublicListeningRow(8, "The Daily"),
        PublicListeningRow(9, "Huberman Lab"),
        PublicListeningRow(10, "Good Hang with Amy Poehler"),
    ),
}


def spotify_wrapped_reference(*, edition: int, scope: str) -> PublicListeningReference:
    normalized_scope = scope.strip().lower()
    if edition != 2025:
        raise ValueError("The bundled Spotify Wrapped reference currently supports edition 2025 only.")
    if normalized_scope not in {"global", "country"}:
        raise ValueError("Spotify Wrapped scope must be 'global' or 'country'.")

    dimensions = _GLOBAL_2025 if normalized_scope == "global" else _US_2025
    return PublicListeningReference(
        provider="Spotify Wrapped",
        edition=edition,
        scope=normalized_scope,
        country=None if normalized_scope == "global" else "United States",
        published_date=date(2025, 12, 3),
        window_start=date(2025, 1, 1),
        # Spotify documents "mid November" rather than an exact cutoff.
        window_end=date(2025, 11, 15),
        window_end_is_approximate=True,
        source_url=SPOTIFY_2025_SOURCE_URL,
        methodology_url=SPOTIFY_2025_METHOD_URL,
        dimensions=dimensions,
    )
