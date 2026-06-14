from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_SESSION_GAP_MINUTES = 30.0

_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y"})
_FALSE_VALUES = frozenset({"0", "false", "f", "no", "n"})
_SKIP_REASON_END_VALUES = frozenset({"fwdbtn", "backbtn"})
_COMPLETED_REASON_END_VALUES = frozenset({"trackdone"})


@dataclass(frozen=True)
class TrackLevelLabels:
    """Targets derived from one chronological streaming-history event."""

    next_track_uri: str
    skipped: bool | None
    listen_duration_ms: int | None
    session_end: bool
    repeat: bool

    @property
    def dwell_ms(self) -> int | None:
        return self.listen_duration_ms


@dataclass(frozen=True)
class TrackLevelExample:
    """A next-track example whose features contain only earlier session events."""

    example_id: int
    target_timestamp: pd.Timestamp
    session_id: int
    session_position: int
    history_track_uris: tuple[str, ...]
    history_time_gaps_seconds: tuple[float, ...]
    target_time_gap_seconds: float
    labels: TrackLevelLabels

    @property
    def target_track_uri(self) -> str:
        return self.labels.next_track_uri


@dataclass(frozen=True)
class TrackLevelDataset:
    examples: tuple[TrackLevelExample, ...]
    source_row_count: int
    valid_track_row_count: int
    unique_track_count: int
    session_count: int

    def to_frame(self) -> pd.DataFrame:
        return examples_to_frame(self.examples)


@dataclass(frozen=True)
class TrackLevelTemporalSplits:
    train: tuple[TrackLevelExample, ...]
    validation: tuple[TrackLevelExample, ...]
    test: tuple[TrackLevelExample, ...]

    @property
    def val(self) -> tuple[TrackLevelExample, ...]:
        return self.validation

    @property
    def all_examples(self) -> tuple[TrackLevelExample, ...]:
        return self.train + self.validation + self.test

    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "train": examples_to_frame(self.train),
            "validation": examples_to_frame(self.validation),
            "test": examples_to_frame(self.test),
        }


def _clean_track_uri(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    uri = str(value).strip()
    if not uri or uri.lower() in {"nan", "none", "null"}:
        return None
    return uri


def _optional_bool(value: object) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _skip_label(skipped: object, reason_end: object) -> bool | None:
    explicit = _optional_bool(skipped)
    if explicit is not None:
        return explicit
    normalized_reason = "" if reason_end is None or pd.isna(reason_end) else str(reason_end).strip().lower()
    if normalized_reason in _SKIP_REASON_END_VALUES:
        return True
    if normalized_reason in _COMPLETED_REASON_END_VALUES:
        return False
    return None


def _listen_duration_ms(value: object) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) < 0:
        return None
    return int(round(float(numeric)))


def _normalize_track_events(
    rows: pd.DataFrame,
    *,
    session_gap_minutes: float,
) -> pd.DataFrame:
    missing = [column for column in ("ts", "spotify_track_uri") if column not in rows.columns]
    if missing:
        raise ValueError(f"Track-level data requires columns: {', '.join(missing)}")
    if session_gap_minutes <= 0:
        raise ValueError("session_gap_minutes must be positive")

    frame = rows.copy()
    frame["_source_order"] = range(len(frame))
    frame["_target_timestamp"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame["_track_uri"] = frame["spotify_track_uri"].map(_clean_track_uri)
    frame = frame.loc[frame["_target_timestamp"].notna() & frame["_track_uri"].notna()].copy()
    if frame.empty:
        return frame

    frame = frame.sort_values(["_target_timestamp", "_source_order"], kind="mergesort").reset_index(drop=True)
    frame["_gap_seconds"] = (
        frame["_target_timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0).astype("float64")
    )
    threshold_seconds = float(session_gap_minutes) * 60.0
    frame["_session_id"] = (frame["_gap_seconds"] > threshold_seconds).cumsum().astype("int64")
    frame["_session_position"] = frame.groupby("_session_id", sort=False).cumcount().astype("int64")
    return frame


def _build_examples_from_events(
    events: pd.DataFrame,
    *,
    max_history: int | None,
    min_history: int,
) -> tuple[TrackLevelExample, ...]:
    if min_history < 0:
        raise ValueError("min_history must be non-negative")
    if max_history is not None and max_history <= 0:
        raise ValueError("max_history must be positive when provided")
    if max_history is not None and min_history > max_history:
        raise ValueError("min_history cannot exceed max_history")
    if events.empty:
        return ()

    skipped_values = events["skipped"] if "skipped" in events.columns else pd.Series(None, index=events.index)
    reason_end_values = (
        events["reason_end"] if "reason_end" in events.columns else pd.Series(None, index=events.index)
    )
    dwell_values = events["ms_played"] if "ms_played" in events.columns else pd.Series(None, index=events.index)

    examples: list[TrackLevelExample] = []
    next_example_id = 0
    for session_id, session in events.groupby("_session_id", sort=False):
        session = session.reset_index()
        track_uris = tuple(session["_track_uri"].astype(str))
        timestamps = tuple(session["_target_timestamp"])
        raw_gaps = tuple(float(value) for value in session["_gap_seconds"])
        source_indices = tuple(int(value) for value in session["index"])
        seen_tracks: set[str] = set()

        for position, target_uri in enumerate(track_uris):
            full_history = track_uris[:position]
            is_repeat = target_uri in seen_tracks
            seen_tracks.add(target_uri)
            if len(full_history) < min_history:
                continue

            history_start = 0 if max_history is None else max(0, position - max_history)
            history = full_history[history_start:]
            history_gaps = list(raw_gaps[history_start:position])
            if history_gaps:
                history_gaps[0] = 0.0

            source_index = source_indices[position]
            session_end = position == len(track_uris) - 1
            labels = TrackLevelLabels(
                next_track_uri=target_uri,
                skipped=_skip_label(skipped_values.loc[source_index], reason_end_values.loc[source_index]),
                listen_duration_ms=_listen_duration_ms(dwell_values.loc[source_index]),
                session_end=session_end,
                repeat=is_repeat,
            )
            examples.append(
                TrackLevelExample(
                    example_id=next_example_id,
                    target_timestamp=timestamps[position],
                    session_id=int(session_id),
                    session_position=position,
                    history_track_uris=tuple(history),
                    history_time_gaps_seconds=tuple(history_gaps),
                    target_time_gap_seconds=float(raw_gaps[position]),
                    labels=labels,
                )
            )
            next_example_id += 1

    return tuple(examples)


def build_track_level_dataset(
    rows: pd.DataFrame,
    *,
    session_gap_minutes: float = DEFAULT_SESSION_GAP_MINUTES,
    max_history: int | None = None,
    min_history: int = 1,
) -> TrackLevelDataset:
    """Build track-level examples without frequency filtering or future features."""

    events = _normalize_track_events(rows, session_gap_minutes=session_gap_minutes)
    examples = _build_examples_from_events(events, max_history=max_history, min_history=min_history)
    return TrackLevelDataset(
        examples=examples,
        source_row_count=len(rows),
        valid_track_row_count=len(events),
        unique_track_count=int(events["_track_uri"].nunique()) if not events.empty else 0,
        session_count=int(events["_session_id"].nunique()) if not events.empty else 0,
    )


def prepare_track_level_data(
    rows: pd.DataFrame,
    *,
    session_gap_minutes: float = DEFAULT_SESSION_GAP_MINUTES,
    max_history: int | None = None,
    min_history: int = 1,
) -> TrackLevelDataset:
    return build_track_level_dataset(
        rows,
        session_gap_minutes=session_gap_minutes,
        max_history=max_history,
        min_history=min_history,
    )


def build_track_level_examples(
    rows: pd.DataFrame,
    *,
    session_gap_minutes: float = DEFAULT_SESSION_GAP_MINUTES,
    max_history: int | None = None,
    min_history: int = 1,
) -> tuple[TrackLevelExample, ...]:
    return build_track_level_dataset(
        rows,
        session_gap_minutes=session_gap_minutes,
        max_history=max_history,
        min_history=min_history,
    ).examples


def _ordered_examples(
    examples: Sequence[TrackLevelExample] | TrackLevelDataset,
) -> tuple[TrackLevelExample, ...]:
    values = examples.examples if isinstance(examples, TrackLevelDataset) else tuple(examples)
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.target_timestamp.value,
                item.session_id,
                item.session_position,
                item.example_id,
            ),
        )
    )


def _closest_boundary(cumulative_counts: Sequence[int], candidates: Iterable[int], target: float) -> int:
    return min(candidates, key=lambda boundary: (abs(cumulative_counts[boundary - 1] - target), boundary))


def split_track_level_examples(
    examples: Sequence[TrackLevelExample] | TrackLevelDataset,
    *,
    validation_fraction: float = 0.16,
    test_fraction: float = 0.20,
) -> TrackLevelTemporalSplits:
    """Split chronologically at session boundaries to keep histories isolated."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    ordered = _ordered_examples(examples)
    if not ordered:
        return TrackLevelTemporalSplits(train=(), validation=(), test=())

    sessions: list[list[TrackLevelExample]] = []
    for example in ordered:
        if not sessions or sessions[-1][0].session_id != example.session_id:
            sessions.append([example])
        else:
            sessions[-1].append(example)

    if len(sessions) == 1:
        return TrackLevelTemporalSplits(train=ordered, validation=(), test=())
    if len(sessions) == 2:
        return TrackLevelTemporalSplits(train=tuple(sessions[0]), validation=(), test=tuple(sessions[1]))

    cumulative_counts: list[int] = []
    running_count = 0
    for session in sessions:
        running_count += len(session)
        cumulative_counts.append(running_count)

    train_fraction = 1.0 - validation_fraction - test_fraction
    train_boundary = _closest_boundary(
        cumulative_counts,
        range(1, len(sessions) - 1),
        len(ordered) * train_fraction,
    )
    validation_boundary = _closest_boundary(
        cumulative_counts,
        range(train_boundary + 1, len(sessions)),
        len(ordered) * (train_fraction + validation_fraction),
    )

    train = tuple(example for session in sessions[:train_boundary] for example in session)
    validation = tuple(
        example for session in sessions[train_boundary:validation_boundary] for example in session
    )
    test = tuple(example for session in sessions[validation_boundary:] for example in session)
    return TrackLevelTemporalSplits(train=train, validation=validation, test=test)


def temporal_train_val_test_split(
    examples: Sequence[TrackLevelExample] | TrackLevelDataset,
    *,
    validation_fraction: float = 0.16,
    test_fraction: float = 0.20,
) -> TrackLevelTemporalSplits:
    return split_track_level_examples(
        examples,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def examples_to_frame(examples: Sequence[TrackLevelExample]) -> pd.DataFrame:
    columns = [
        "example_id",
        "target_timestamp",
        "session_id",
        "session_position",
        "history_track_uris",
        "history_time_gaps_seconds",
        "target_time_gap_seconds",
        "next_track_uri",
        "skipped",
        "listen_duration_ms",
        "session_end",
        "repeat",
    ]
    rows = [
        {
            "example_id": example.example_id,
            "target_timestamp": example.target_timestamp,
            "session_id": example.session_id,
            "session_position": example.session_position,
            "history_track_uris": example.history_track_uris,
            "history_time_gaps_seconds": example.history_time_gaps_seconds,
            "target_time_gap_seconds": example.target_time_gap_seconds,
            "next_track_uri": example.labels.next_track_uri,
            "skipped": example.labels.skipped,
            "listen_duration_ms": example.labels.listen_duration_ms,
            "session_end": example.labels.session_end,
            "repeat": example.labels.repeat,
        }
        for example in examples
    ]
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "DEFAULT_SESSION_GAP_MINUTES",
    "TrackLevelDataset",
    "TrackLevelExample",
    "TrackLevelLabels",
    "TrackLevelTemporalSplits",
    "build_track_level_dataset",
    "build_track_level_examples",
    "examples_to_frame",
    "prepare_track_level_data",
    "split_track_level_examples",
    "temporal_train_val_test_split",
]
