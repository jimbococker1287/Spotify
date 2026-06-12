from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import unicodedata


DEFAULT_GENRE_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "hip-hop": ("hip-hop", "hip hop", "hiphop", "rap"),
    "r&b": ("r&b", "rnb", "rhythm and blues"),
    "electronic": ("electronic", "electronica", "edm", "dance"),
    "pop": ("pop",),
    "rock": ("rock",),
    "indie": ("indie", "indie rock", "indie pop"),
    "country": ("country",),
    "latin": ("latin", "latin pop", "reggaeton"),
    "jazz": ("jazz",),
    "classical": ("classical",),
}


@dataclass(frozen=True)
class GenreDefinition:
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicGenreArtistRow:
    genre: str
    rank: int
    artist_name: str


def normalize_label(value: object) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())


def artist_tag_cache_key(artist_name: str, *, namespace: str = "artist-tags-v1") -> str:
    normalized_artist = normalize_label(artist_name)
    if not normalized_artist:
        raise ValueError("Artist name must contain at least one letter or number.")
    payload = json.dumps(
        {"artist": normalized_artist, "namespace": str(namespace).strip()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ArtistTagCache:
    """Filesystem JSON cache for externally supplied artist tags."""

    def __init__(
        self,
        root: str | Path,
        *,
        ttl_seconds: float = 7 * 24 * 60 * 60,
        namespace: str = "artist-tags-v1",
        clock: Callable[[], float | datetime] = time.time,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative.")
        self.root = Path(root)
        self.ttl_seconds = float(ttl_seconds)
        self.namespace = str(namespace).strip()
        self._clock = clock

    def key(self, artist_name: str) -> str:
        return artist_tag_cache_key(artist_name, namespace=self.namespace)

    def path_for(self, artist_name: str) -> Path:
        return self.root / f"{self.key(artist_name)}.json"

    def get(self, artist_name: str) -> list[str] | None:
        path = self.path_for(artist_name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict) or payload.get("key") != self.key(artist_name):
            return None
        cached_at = payload.get("cached_at")
        tags = payload.get("tags")
        if not isinstance(cached_at, (int, float)) or not isinstance(tags, list):
            return None
        age_seconds = max(0.0, self._now() - float(cached_at))
        if age_seconds > self.ttl_seconds:
            return None
        return [str(tag) for tag in tags if str(tag).strip()]

    def set(self, artist_name: str, tags: Iterable[str]) -> Path:
        normalized_tags = _dedupe_text(tags)
        path = self.path_for(artist_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artist_name": str(artist_name).strip(),
            "cached_at": self._now(),
            "key": self.key(artist_name),
            "namespace": self.namespace,
            "tags": normalized_tags,
        }
        temporary_path = path.with_suffix(f".{os.getpid()}.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path

    def delete(self, artist_name: str) -> bool:
        try:
            self.path_for(artist_name).unlink()
        except FileNotFoundError:
            return False
        return True

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.timestamp()
        return float(value)


FilesystemArtistTagCache = ArtistTagCache


def build_genre_comparison(
    listening_events: Iterable[Mapping[str, Any] | object],
    artist_tags: Mapping[str, Iterable[str]],
    public_rows: Mapping[str, Iterable[Mapping[str, Any] | object]]
    | Iterable[Mapping[str, Any] | object],
    *,
    genre_definitions: Mapping[str, GenreDefinition | str | Iterable[str]]
    | Iterable[GenreDefinition]
    | None = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Compare personal genre/scene listening with public ranked artist rows.

    Events may use ``timestamp``/``ts``, ``artist_name``/Spotify's artist field,
    and ``duration_ms``/``ms_played``. Multi-genre events are split evenly so
    aggregate shares remain bounded.
    """

    definitions = _coerce_definitions(genre_definitions)
    alias_to_genre = _alias_index(definitions)
    tags_by_artist = {
        normalize_label(artist): _dedupe_text(tags)
        for artist, tags in artist_tags.items()
        if normalize_label(artist)
    }
    normalized_events = _normalize_events(listening_events)
    ranked_public = _normalize_public_rows(public_rows, alias_to_genre)
    limit = max(1, int(top_n))

    aggregate_accumulator = _new_accumulator()
    daily_accumulators: dict[str, dict[str, Any]] = {}
    artist_accumulators: dict[str, dict[str, dict[str, Any]]] = {
        definition.name: {} for definition in definitions
    }

    for event in normalized_events:
        artist_key = event["artist_key"]
        tags = tags_by_artist.get(artist_key, [])
        matched_genres = _matched_genres(tags, alias_to_genre)
        assignment_confidence = 1.0 / len(matched_genres) if matched_genres else 0.0
        daily = daily_accumulators.setdefault(event["date"], _new_accumulator())
        _record_coverage(aggregate_accumulator, event, bool(tags), bool(matched_genres), assignment_confidence)
        _record_coverage(daily, event, bool(tags), bool(matched_genres), assignment_confidence)

        for genre in matched_genres:
            fraction = 1.0 / len(matched_genres)
            event_weight = event["event_weight"] * fraction
            duration_ms = event["duration_ms"] * fraction
            _record_genre(aggregate_accumulator, genre, event_weight, duration_ms)
            _record_genre(daily, genre, event_weight, duration_ms)
            artist_row = artist_accumulators[genre].setdefault(
                artist_key,
                {
                    "artist_name": event["artist_name"],
                    "event_weight": 0.0,
                    "duration_ms": 0.0,
                },
            )
            artist_row["event_weight"] += event_weight
            artist_row["duration_ms"] += duration_ms

    daily_rows = _daily_rows(daily_accumulators, definitions)
    aggregate_genres: list[dict[str, object]] = []
    comparisons: dict[str, dict[str, object]] = {}
    total_events = float(aggregate_accumulator["total_event_weight"])
    total_duration = float(aggregate_accumulator["total_duration_ms"])

    for definition in definitions:
        genre = definition.name
        genre_totals = aggregate_accumulator["genres"].get(genre, {})
        personal_ranked = _rank_personal_artists(artist_accumulators[genre])
        public_ranked = ranked_public.get(genre, [])[:limit]
        comparison = _compare_rankings(personal_ranked[:limit], public_ranked, limit=limit)
        genre_row = {
            "genre": genre,
            "aliases": list(definition.aliases),
            "event_weight": float(genre_totals.get("event_weight", 0.0)),
            "event_share": _ratio(float(genre_totals.get("event_weight", 0.0)), total_events),
            "duration_ms": float(genre_totals.get("duration_ms", 0.0)),
            "duration_share": _ratio(float(genre_totals.get("duration_ms", 0.0)), total_duration),
            "artist_count": len(personal_ranked),
            **comparison["metrics"],
        }
        aggregate_genres.append(genre_row)
        comparisons[genre] = {
            "genre": genre,
            "personal_top_artists": comparison["personal_top_artists"],
            "public_top_artists": comparison["public_top_artists"],
            "shared_artists": comparison["shared_artists"],
            "personal_distinctive_artists": comparison["personal_distinctive_artists"],
            "public_distinctive_artists": comparison["public_distinctive_artists"],
            "metrics": comparison["metrics"],
        }

    coverage, confidence = _coverage_payload(aggregate_accumulator)
    return {
        "genre_definitions": [
            {"name": definition.name, "aliases": list(definition.aliases)}
            for definition in definitions
        ],
        "daily": daily_rows,
        "aggregate": {
            "event_count": total_events,
            "duration_ms": total_duration,
            "genre_shares": aggregate_genres,
            "coverage": coverage,
            "confidence": confidence,
        },
        "genres": comparisons,
    }


build_public_genre_comparison = build_genre_comparison
compare_genre_scenes = build_genre_comparison


def _coerce_definitions(
    definitions: Mapping[str, GenreDefinition | str | Iterable[str]]
    | Iterable[GenreDefinition]
    | None,
) -> list[GenreDefinition]:
    if definitions is None:
        definitions = DEFAULT_GENRE_DEFINITIONS
    rows: list[GenreDefinition] = []
    if isinstance(definitions, Mapping):
        for name, value in definitions.items():
            if isinstance(value, GenreDefinition):
                aliases = value.aliases
            elif isinstance(value, str):
                aliases = (value,)
            else:
                aliases = tuple(str(alias) for alias in value)
            rows.append(GenreDefinition(str(name).strip(), aliases))
    else:
        rows = list(definitions)

    normalized_rows: list[GenreDefinition] = []
    seen_names: set[str] = set()
    for row in rows:
        name = str(row.name).strip()
        name_key = normalize_label(name)
        if not name_key:
            raise ValueError("Genre definition names must not be empty.")
        if name_key in seen_names:
            raise ValueError(f"Duplicate genre definition: {name}")
        seen_names.add(name_key)
        aliases = tuple(_dedupe_text((name, *row.aliases)))
        normalized_rows.append(GenreDefinition(name=name, aliases=aliases))
    if not normalized_rows:
        raise ValueError("At least one genre definition is required.")
    return normalized_rows


def _alias_index(definitions: Iterable[GenreDefinition]) -> dict[str, str]:
    index: dict[str, str] = {}
    for definition in definitions:
        for alias in definition.aliases:
            alias_key = normalize_label(alias)
            existing = index.get(alias_key)
            if existing is not None and existing != definition.name:
                raise ValueError(f"Genre alias {alias!r} belongs to both {existing!r} and {definition.name!r}.")
            index[alias_key] = definition.name
    return index


def _normalize_events(events: Iterable[Mapping[str, Any] | object]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in events:
        artist_name = str(
            _field(row, "artist_name", "artist", "master_metadata_album_artist_name", default="")
        ).strip()
        artist_key = normalize_label(artist_name)
        if not artist_key:
            continue
        timestamp = _field(row, "timestamp", "ts", "played_at", "date", "listening_date")
        listening_date = _coerce_date(timestamp)
        event_weight = _coerce_non_negative(
            _field(row, "event_count", "plays", "count", "weight", default=1.0),
            default=1.0,
        )
        duration_ms = _coerce_non_negative(
            _field(row, "duration_ms", "ms_played", default=0.0),
            default=0.0,
        )
        if event_weight == 0.0 and duration_ms == 0.0:
            continue
        normalized.append(
            {
                "artist_name": artist_name,
                "artist_key": artist_key,
                "date": listening_date.isoformat(),
                "event_weight": event_weight,
                "duration_ms": duration_ms,
            }
        )
    return normalized


def _normalize_public_rows(
    rows: Mapping[str, Iterable[Mapping[str, Any] | object]]
    | Iterable[Mapping[str, Any] | object],
    alias_to_genre: Mapping[str, str],
) -> dict[str, list[dict[str, object]]]:
    flattened: list[tuple[str | None, Mapping[str, Any] | object]] = []
    if isinstance(rows, Mapping):
        for genre, genre_rows in rows.items():
            flattened.extend((str(genre), row) for row in genre_rows)
    else:
        flattened.extend((None, row) for row in rows)

    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for supplied_genre, row in flattened:
        genre_label = supplied_genre or str(_field(row, "genre", "tag", "scene", "category", default=""))
        genre = alias_to_genre.get(normalize_label(genre_label))
        artist_name = str(_field(row, "artist_name", "artist", "name", default="")).strip()
        artist_key = normalize_label(artist_name)
        if genre is None or not artist_key:
            continue
        rank = int(_coerce_non_negative(_field(row, "rank", "position", default=0), default=0.0))
        if rank < 1:
            continue
        existing = grouped[genre].get(artist_key)
        if existing is None or rank < int(existing["rank"]):
            grouped[genre][artist_key] = {
                "artist_name": artist_name,
                "artist_key": artist_key,
                "rank": rank,
            }
    return {
        genre: sorted(genre_rows.values(), key=lambda row: (int(row["rank"]), str(row["artist_name"]).casefold()))
        for genre, genre_rows in grouped.items()
    }


def _rank_personal_artists(rows: Mapping[str, dict[str, Any]]) -> list[dict[str, object]]:
    ranked = sorted(
        rows.values(),
        key=lambda row: (-float(row["event_weight"]), -float(row["duration_ms"]), str(row["artist_name"]).casefold()),
    )
    return [
        {
            "artist_name": row["artist_name"],
            "artist_key": normalize_label(row["artist_name"]),
            "rank": rank,
            "event_weight": float(row["event_weight"]),
            "duration_ms": float(row["duration_ms"]),
        }
        for rank, row in enumerate(ranked, start=1)
    ]


def _compare_rankings(
    personal_rows: list[dict[str, object]],
    public_rows: list[dict[str, object]],
    *,
    limit: int,
) -> dict[str, object]:
    personal_by_key = {str(row["artist_key"]): row for row in personal_rows}
    public_by_key = {str(row["artist_key"]): row for row in public_rows}
    shared_keys = set(personal_by_key).intersection(public_by_key)
    shared = [
        {
            "artist_name": personal_by_key[key]["artist_name"],
            "personal_rank": int(personal_by_key[key]["rank"]),
            "public_rank": int(public_by_key[key]["rank"]),
            "event_weight": float(personal_by_key[key]["event_weight"]),
            "duration_ms": float(personal_by_key[key]["duration_ms"]),
        }
        for key in shared_keys
    ]
    shared.sort(key=lambda row: (int(row["personal_rank"]), int(row["public_rank"])))
    similarity = _rank_weighted_jaccard(personal_by_key, public_by_key)
    personal_distinctive = [
        _without_key(row) for row in personal_rows if str(row["artist_key"]) not in public_by_key
    ]
    public_distinctive = [
        _without_key(row) for row in public_rows if str(row["artist_key"]) not in personal_by_key
    ]
    metrics = {
        "shared_artist_count": len(shared),
        "personal_top_overlap": _ratio(len(shared), len(personal_rows)),
        "public_top_overlap": _ratio(len(shared), len(public_rows)),
        "rank_similarity": similarity,
        "rank_similarity_method": "rank_weighted_jaccard",
    }
    return {
        "metrics": metrics,
        "personal_top_artists": [_without_key(row) for row in personal_rows[:limit]],
        "public_top_artists": [_without_key(row) for row in public_rows[:limit]],
        "shared_artists": shared,
        "personal_distinctive_artists": personal_distinctive[:limit],
        "public_distinctive_artists": public_distinctive[:limit],
    }


def _rank_weighted_jaccard(
    personal_by_key: Mapping[str, Mapping[str, object]],
    public_by_key: Mapping[str, Mapping[str, object]],
) -> float:
    union = set(personal_by_key).union(public_by_key)
    if not union:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for key in union:
        personal_weight = 1.0 / int(personal_by_key[key]["rank"]) if key in personal_by_key else 0.0
        public_weight = 1.0 / int(public_by_key[key]["rank"]) if key in public_by_key else 0.0
        numerator += min(personal_weight, public_weight)
        denominator += max(personal_weight, public_weight)
    return _ratio(numerator, denominator)


def _daily_rows(
    accumulators: Mapping[str, dict[str, Any]],
    definitions: Iterable[GenreDefinition],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for listening_date in sorted(accumulators):
        accumulator = accumulators[listening_date]
        coverage, confidence = _coverage_payload(accumulator)
        total_events = float(accumulator["total_event_weight"])
        total_duration = float(accumulator["total_duration_ms"])
        for definition in definitions:
            genre_values = accumulator["genres"].get(definition.name, {})
            rows.append(
                {
                    "date": listening_date,
                    "genre": definition.name,
                    "event_weight": float(genre_values.get("event_weight", 0.0)),
                    "event_share": _ratio(float(genre_values.get("event_weight", 0.0)), total_events),
                    "duration_ms": float(genre_values.get("duration_ms", 0.0)),
                    "duration_share": _ratio(float(genre_values.get("duration_ms", 0.0)), total_duration),
                    "coverage": coverage,
                    "confidence": confidence,
                }
            )
    return rows


def _new_accumulator() -> dict[str, Any]:
    return {
        "total_event_weight": 0.0,
        "total_duration_ms": 0.0,
        "tagged_event_weight": 0.0,
        "tagged_duration_ms": 0.0,
        "classified_event_weight": 0.0,
        "classified_duration_ms": 0.0,
        "confidence_event_weight": 0.0,
        "confidence_duration_ms": 0.0,
        "genres": {},
    }


def _record_coverage(
    accumulator: dict[str, Any],
    event: Mapping[str, Any],
    tagged: bool,
    classified: bool,
    assignment_confidence: float,
) -> None:
    event_weight = float(event["event_weight"])
    duration_ms = float(event["duration_ms"])
    accumulator["total_event_weight"] += event_weight
    accumulator["total_duration_ms"] += duration_ms
    if tagged:
        accumulator["tagged_event_weight"] += event_weight
        accumulator["tagged_duration_ms"] += duration_ms
    if classified:
        accumulator["classified_event_weight"] += event_weight
        accumulator["classified_duration_ms"] += duration_ms
        accumulator["confidence_event_weight"] += event_weight * assignment_confidence
        accumulator["confidence_duration_ms"] += duration_ms * assignment_confidence


def _record_genre(
    accumulator: dict[str, Any],
    genre: str,
    event_weight: float,
    duration_ms: float,
) -> None:
    values = accumulator["genres"].setdefault(genre, {"event_weight": 0.0, "duration_ms": 0.0})
    values["event_weight"] += event_weight
    values["duration_ms"] += duration_ms


def _coverage_payload(accumulator: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    total_events = float(accumulator["total_event_weight"])
    total_duration = float(accumulator["total_duration_ms"])
    classified_events = float(accumulator["classified_event_weight"])
    classified_duration = float(accumulator["classified_duration_ms"])
    coverage = {
        "tagged_event_share": _ratio(float(accumulator["tagged_event_weight"]), total_events),
        "tagged_duration_share": _ratio(float(accumulator["tagged_duration_ms"]), total_duration),
        "classified_event_share": _ratio(classified_events, total_events),
        "classified_duration_share": _ratio(classified_duration, total_duration),
    }
    confidence = {
        "classified_event_confidence": _ratio(
            float(accumulator["confidence_event_weight"]),
            classified_events,
        ),
        "classified_duration_confidence": _ratio(
            float(accumulator["confidence_duration_ms"]),
            classified_duration,
        ),
        "overall_event_confidence": _ratio(
            float(accumulator["confidence_event_weight"]),
            total_events,
        ),
        "overall_duration_confidence": _ratio(
            float(accumulator["confidence_duration_ms"]),
            total_duration,
        ),
    }
    return coverage, confidence


def _matched_genres(tags: Iterable[str], alias_to_genre: Mapping[str, str]) -> list[str]:
    return sorted(
        {
            alias_to_genre[tag_key]
            for tag in tags
            if (tag_key := normalize_label(tag)) in alias_to_genre
        }
    )


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Listening events require a timestamp or date.")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return date.fromisoformat(raw[:10])
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.date()


def _field(row: Mapping[str, Any] | object, *names: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
    else:
        for name in names:
            value = getattr(row, name, None)
            if value is not None:
                return value
    return default


def _coerce_non_negative(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0.0 else default


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _dedupe_text(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = normalize_label(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _without_key(row: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "artist_key"}


__all__ = [
    "ArtistTagCache",
    "DEFAULT_GENRE_DEFINITIONS",
    "FilesystemArtistTagCache",
    "GenreDefinition",
    "PublicGenreArtistRow",
    "artist_tag_cache_key",
    "build_genre_comparison",
    "build_public_genre_comparison",
    "compare_genre_scenes",
    "normalize_label",
]
