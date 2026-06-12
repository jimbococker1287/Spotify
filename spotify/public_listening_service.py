from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Literal, Sequence

import duckdb
import pandas as pd

Dimension = Literal["artists", "tracks", "podcasts", "genres", "scenes"]
ReferenceAlignment = Literal["historical_projection", "date_aligned", "post_window_projection"]
ClosestScope = Literal["global", "united_states", "tie"]

_DIMENSIONS = frozenset({"artists", "tracks", "podcasts", "genres", "scenes"})
_REFERENCE_ALIGNMENTS = frozenset({"historical_projection", "date_aligned", "post_window_projection"})
_CLOSEST_SCOPES = frozenset({"global", "united_states", "tie"})
_MART_TABLE = "mart_public_listening_daily_similarity"
_TREND_VIEW = "public_listening_daily_trend"
_INJECTED_RELATION = "_public_listening_mart"
_REQUIRED_COLUMNS = frozenset(
    {
        "listening_date",
        "reference_edition",
        "reference_alignment",
        "dimension",
        "event_count",
        "duration_minutes",
        "unique_entity_count",
        "global_similarity",
        "united_states_similarity",
        "united_states_minus_global",
        "closer_scope",
        "global_event_share_on_public_top",
        "united_states_event_share_on_public_top",
        "global_duration_share_on_public_top",
        "united_states_duration_share_on_public_top",
        "personal_top_entity",
        "personal_top_entity_detail",
    }
)

__all__ = [
    "ClosestScope",
    "DailyListeningTrend",
    "Dimension",
    "ListeningDateRange",
    "ListeningQuery",
    "ListeningSummary",
    "PublicListeningService",
    "ReferenceAlignment",
    "query_closest_scopes",
    "query_daily_trend",
    "query_date_range",
    "query_dimensions",
    "query_reference_alignments",
    "query_summary_aggregates",
    "query_top_aligned_days",
    "query_top_distinctive_days",
]


@dataclass(frozen=True)
class ListeningQuery:
    start_date: date | str | None = None
    end_date: date | str | None = None
    dimension: Dimension | None = None
    reference_alignment: ReferenceAlignment | None = None
    closest_scope: ClosestScope | None = None
    reference_edition: int | None = None


@dataclass(frozen=True)
class ListeningDateRange:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class DailyListeningTrend:
    listening_date: date
    reference_edition: int
    reference_alignment: str
    dimension: str
    event_count: int
    duration_minutes: float
    unique_entity_count: int
    global_similarity: float
    united_states_similarity: float
    united_states_minus_global: float
    closer_scope: str
    global_event_share_on_public_top: float
    united_states_event_share_on_public_top: float
    global_duration_share_on_public_top: float
    united_states_duration_share_on_public_top: float
    personal_top_entity: str | None
    personal_top_entity_detail: str | None

    @property
    def best_scope_similarity(self) -> float:
        return max(self.global_similarity, self.united_states_similarity)


@dataclass(frozen=True)
class ListeningSummary:
    row_count: int
    active_day_count: int
    total_event_count: int
    total_duration_minutes: float
    average_global_similarity: float
    average_united_states_similarity: float
    average_united_states_minus_global: float
    average_best_scope_similarity: float
    global_closer_count: int
    united_states_closer_count: int
    tie_count: int


class PublicListeningService:
    """Read-only dashboard queries over the public-listening daily mart."""

    def __init__(
        self,
        *,
        duckdb_path: str | Path | None = None,
        parquet_path: str | Path | None = None,
        mart_df: pd.DataFrame | None = None,
    ) -> None:
        sources = sum(value is not None for value in (duckdb_path, parquet_path, mart_df))
        if sources != 1:
            raise ValueError("Provide exactly one of duckdb_path, parquet_path, or mart_df.")

        self._duckdb_path = Path(duckdb_path).expanduser() if duckdb_path is not None else None
        self._parquet_path = Path(parquet_path).expanduser() if parquet_path is not None else None
        self._mart_df = mart_df.copy(deep=True) if mart_df is not None else None

        if self._duckdb_path is not None and not self._duckdb_path.is_file():
            raise FileNotFoundError(f"DuckDB database does not exist: {self._duckdb_path}")
        if self._parquet_path is not None and not self._parquet_path.is_file():
            raise FileNotFoundError(f"Parquet mart does not exist: {self._parquet_path}")
        if self._mart_df is not None:
            self._validate_columns(self._mart_df.columns)

    def date_range(self, query: ListeningQuery | None = None) -> ListeningDateRange | None:
        where_sql, parameters = self._where_clause(query)
        sql = f"""
            SELECT
                MIN(CAST(listening_date AS DATE)) AS start_date,
                MAX(CAST(listening_date AS DATE)) AS end_date
            FROM {{source}}
            {where_sql}
        """
        row = self._fetchone(sql, parameters)
        if row is None or row[0] is None or row[1] is None:
            return None
        return ListeningDateRange(start_date=_as_date(row[0]), end_date=_as_date(row[1]))

    def dimensions(self, query: ListeningQuery | None = None) -> tuple[str, ...]:
        return self._distinct_values("dimension", query)

    def reference_alignments(self, query: ListeningQuery | None = None) -> tuple[str, ...]:
        return self._distinct_values("reference_alignment", query)

    def closest_scopes(self, query: ListeningQuery | None = None) -> tuple[str, ...]:
        return self._distinct_values("closer_scope", query)

    def daily_trend(
        self,
        query: ListeningQuery | None = None,
        *,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[DailyListeningTrend]:
        validated_limit = _validate_limit(limit, allow_none=True)
        where_sql, parameters = self._where_clause(query)
        direction = "DESC" if descending else "ASC"
        limit_sql = ""
        if validated_limit is not None:
            limit_sql = "LIMIT ?"
            parameters.append(validated_limit)
        sql = f"""
            SELECT
                CAST(listening_date AS DATE),
                CAST(reference_edition AS BIGINT),
                CAST(reference_alignment AS VARCHAR),
                CAST(dimension AS VARCHAR),
                CAST(event_count AS BIGINT),
                CAST(duration_minutes AS DOUBLE),
                CAST(unique_entity_count AS BIGINT),
                CAST(global_similarity AS DOUBLE),
                CAST(united_states_similarity AS DOUBLE),
                CAST(united_states_minus_global AS DOUBLE),
                CAST(closer_scope AS VARCHAR),
                CAST(global_event_share_on_public_top AS DOUBLE),
                CAST(united_states_event_share_on_public_top AS DOUBLE),
                CAST(global_duration_share_on_public_top AS DOUBLE),
                CAST(united_states_duration_share_on_public_top AS DOUBLE),
                CAST(personal_top_entity AS VARCHAR),
                CAST(personal_top_entity_detail AS VARCHAR)
            FROM {{source}}
            {where_sql}
            ORDER BY listening_date {direction}, dimension ASC, reference_edition ASC
            {limit_sql}
        """
        return [_trend_from_row(row) for row in self._fetchall(sql, parameters)]

    def summary_aggregates(self, query: ListeningQuery | None = None) -> ListeningSummary:
        where_sql, parameters = self._where_clause(query)
        sql = f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT CAST(listening_date AS DATE)) AS active_day_count,
                COALESCE(SUM(CAST(event_count AS BIGINT)), 0) AS total_event_count,
                COALESCE(SUM(CAST(duration_minutes AS DOUBLE)), 0.0) AS total_duration_minutes,
                COALESCE(AVG(CAST(global_similarity AS DOUBLE)), 0.0) AS average_global_similarity,
                COALESCE(AVG(CAST(united_states_similarity AS DOUBLE)), 0.0)
                    AS average_united_states_similarity,
                COALESCE(AVG(CAST(united_states_minus_global AS DOUBLE)), 0.0)
                    AS average_united_states_minus_global,
                COALESCE(
                    AVG(
                        GREATEST(
                            CAST(global_similarity AS DOUBLE),
                            CAST(united_states_similarity AS DOUBLE)
                        )
                    ),
                    0.0
                ) AS average_best_scope_similarity,
                COUNT(*) FILTER (WHERE closer_scope = 'global') AS global_closer_count,
                COUNT(*) FILTER (WHERE closer_scope = 'united_states') AS united_states_closer_count,
                COUNT(*) FILTER (WHERE closer_scope = 'tie') AS tie_count
            FROM {{source}}
            {where_sql}
        """
        row = self._fetchone(sql, parameters)
        assert row is not None
        return ListeningSummary(
            row_count=int(row[0]),
            active_day_count=int(row[1]),
            total_event_count=int(row[2]),
            total_duration_minutes=float(row[3]),
            average_global_similarity=float(row[4]),
            average_united_states_similarity=float(row[5]),
            average_united_states_minus_global=float(row[6]),
            average_best_scope_similarity=float(row[7]),
            global_closer_count=int(row[8]),
            united_states_closer_count=int(row[9]),
            tie_count=int(row[10]),
        )

    def top_aligned_days(
        self,
        query: ListeningQuery | None = None,
        *,
        limit: int = 10,
    ) -> list[DailyListeningTrend]:
        return self._notable_days(query, limit=limit, distinctive=False)

    def top_distinctive_days(
        self,
        query: ListeningQuery | None = None,
        *,
        limit: int = 10,
    ) -> list[DailyListeningTrend]:
        return self._notable_days(query, limit=limit, distinctive=True)

    def _notable_days(
        self,
        query: ListeningQuery | None,
        *,
        limit: int,
        distinctive: bool,
    ) -> list[DailyListeningTrend]:
        validated_limit = _validate_limit(limit)
        where_sql, parameters = self._where_clause(query)
        where_sql = (
            f"{where_sql} AND CAST(event_count AS BIGINT) > 0"
            if where_sql
            else "WHERE CAST(event_count AS BIGINT) > 0"
        )
        similarity_direction = "ASC" if distinctive else "DESC"
        parameters.append(validated_limit)
        sql = f"""
            SELECT
                CAST(listening_date AS DATE),
                CAST(reference_edition AS BIGINT),
                CAST(reference_alignment AS VARCHAR),
                CAST(dimension AS VARCHAR),
                CAST(event_count AS BIGINT),
                CAST(duration_minutes AS DOUBLE),
                CAST(unique_entity_count AS BIGINT),
                CAST(global_similarity AS DOUBLE),
                CAST(united_states_similarity AS DOUBLE),
                CAST(united_states_minus_global AS DOUBLE),
                CAST(closer_scope AS VARCHAR),
                CAST(global_event_share_on_public_top AS DOUBLE),
                CAST(united_states_event_share_on_public_top AS DOUBLE),
                CAST(global_duration_share_on_public_top AS DOUBLE),
                CAST(united_states_duration_share_on_public_top AS DOUBLE),
                CAST(personal_top_entity AS VARCHAR),
                CAST(personal_top_entity_detail AS VARCHAR)
            FROM {{source}}
            {where_sql}
            ORDER BY
                GREATEST(
                    CAST(global_similarity AS DOUBLE),
                    CAST(united_states_similarity AS DOUBLE)
                ) {similarity_direction},
                listening_date DESC,
                dimension ASC,
                reference_edition DESC
            LIMIT ?
        """
        return [_trend_from_row(row) for row in self._fetchall(sql, parameters)]

    def _distinct_values(self, column: str, query: ListeningQuery | None) -> tuple[str, ...]:
        if column not in {"dimension", "reference_alignment", "closer_scope"}:
            raise ValueError(f"Unsupported distinct-value column: {column}")
        where_sql, parameters = self._where_clause(query)
        sql = f"""
            SELECT DISTINCT CAST({column} AS VARCHAR)
            FROM {{source}}
            {where_sql}
            ORDER BY 1
        """
        return tuple(str(row[0]) for row in self._fetchall(sql, parameters) if row[0] is not None)

    def _where_clause(self, query: ListeningQuery | None) -> tuple[str, list[object]]:
        normalized = _normalize_query(query or ListeningQuery())
        predicates: list[str] = []
        parameters: list[object] = []
        if normalized.start_date is not None:
            predicates.append("CAST(listening_date AS DATE) >= ?")
            parameters.append(normalized.start_date)
        if normalized.end_date is not None:
            predicates.append("CAST(listening_date AS DATE) <= ?")
            parameters.append(normalized.end_date)
        if normalized.dimension is not None:
            predicates.append("dimension = ?")
            parameters.append(normalized.dimension)
        if normalized.reference_alignment is not None:
            predicates.append("reference_alignment = ?")
            parameters.append(normalized.reference_alignment)
        if normalized.closest_scope is not None:
            predicates.append("closer_scope = ?")
            parameters.append(normalized.closest_scope)
        if normalized.reference_edition is not None:
            predicates.append("CAST(reference_edition AS BIGINT) = ?")
            parameters.append(normalized.reference_edition)
        return ("WHERE " + " AND ".join(predicates) if predicates else ""), parameters

    def _fetchall(self, sql: str, parameters: Sequence[object]) -> list[tuple[object, ...]]:
        with self._connection() as (con, source):
            return con.execute(sql.format(source=source), list(parameters)).fetchall()

    def _fetchone(self, sql: str, parameters: Sequence[object]) -> tuple[object, ...] | None:
        with self._connection() as (con, source):
            return con.execute(sql.format(source=source), list(parameters)).fetchone()

    @contextmanager
    def _connection(self) -> Iterator[tuple[duckdb.DuckDBPyConnection, str]]:
        if self._duckdb_path is not None:
            con = duckdb.connect(str(self._duckdb_path), read_only=True)
            try:
                source = self._database_source(con)
                self._validate_relation(con, source)
                yield con, source
            finally:
                con.close()
            return

        con = duckdb.connect()
        try:
            if self._mart_df is not None:
                con.register(_INJECTED_RELATION, self._mart_df)
            else:
                assert self._parquet_path is not None
                con.from_parquet(str(self._parquet_path)).create_view(_INJECTED_RELATION)
            self._validate_relation(con, _INJECTED_RELATION)
            yield con, _INJECTED_RELATION
        finally:
            con.close()

    @staticmethod
    def _database_source(con: duckdb.DuckDBPyConnection) -> str:
        for relation in (_TREND_VIEW, _MART_TABLE):
            exists = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
                """,
                [relation],
            ).fetchone()
            if exists and int(exists[0]) > 0:
                return relation
        raise ValueError(
            f"DuckDB database must expose {_TREND_VIEW!r} or {_MART_TABLE!r}."
        )

    @classmethod
    def _validate_relation(cls, con: duckdb.DuckDBPyConnection, relation: str) -> None:
        columns = [str(row[0]) for row in con.execute(f"DESCRIBE {relation}").fetchall()]
        cls._validate_columns(columns)

    @staticmethod
    def _validate_columns(columns: Sequence[object]) -> None:
        available = {str(column) for column in columns}
        missing = sorted(_REQUIRED_COLUMNS - available)
        if missing:
            raise ValueError(f"Public-listening mart is missing required columns: {', '.join(missing)}")


def query_date_range(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
) -> ListeningDateRange | None:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).date_range(query)


def query_daily_trend(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
    limit: int | None = None,
    descending: bool = False,
) -> list[DailyListeningTrend]:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).daily_trend(query, limit=limit, descending=descending)


def query_dimensions(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
) -> tuple[str, ...]:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).dimensions(query)


def query_reference_alignments(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
) -> tuple[str, ...]:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).reference_alignments(query)


def query_closest_scopes(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
) -> tuple[str, ...]:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).closest_scopes(query)


def query_summary_aggregates(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
) -> ListeningSummary:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).summary_aggregates(query)


def query_top_aligned_days(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
    limit: int = 10,
) -> list[DailyListeningTrend]:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).top_aligned_days(query, limit=limit)


def query_top_distinctive_days(
    *,
    duckdb_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    mart_df: pd.DataFrame | None = None,
    query: ListeningQuery | None = None,
    limit: int = 10,
) -> list[DailyListeningTrend]:
    return PublicListeningService(
        duckdb_path=duckdb_path,
        parquet_path=parquet_path,
        mart_df=mart_df,
    ).top_distinctive_days(query, limit=limit)


def _normalize_query(query: ListeningQuery) -> ListeningQuery:
    start_date = _parse_date(query.start_date, name="start_date")
    end_date = _parse_date(query.end_date, name="end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")
    if query.dimension is not None and query.dimension not in _DIMENSIONS:
        raise ValueError(f"dimension must be one of: {', '.join(sorted(_DIMENSIONS))}")
    if query.reference_alignment is not None and query.reference_alignment not in _REFERENCE_ALIGNMENTS:
        raise ValueError(
            f"reference_alignment must be one of: {', '.join(sorted(_REFERENCE_ALIGNMENTS))}"
        )
    if query.closest_scope is not None and query.closest_scope not in _CLOSEST_SCOPES:
        raise ValueError(f"closest_scope must be one of: {', '.join(sorted(_CLOSEST_SCOPES))}")
    if query.reference_edition is not None:
        if (
            isinstance(query.reference_edition, bool)
            or not isinstance(query.reference_edition, int)
            or query.reference_edition <= 0
        ):
            raise ValueError("reference_edition must be a positive integer.")
    return ListeningQuery(
        start_date=start_date,
        end_date=end_date,
        dimension=query.dimension,
        reference_alignment=query.reference_alignment,
        closest_scope=query.closest_scope,
        reference_edition=int(query.reference_edition) if query.reference_edition is not None else None,
    )


def _parse_date(value: date | str | None, *, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO date.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date in YYYY-MM-DD format.") from exc


def _validate_limit(limit: int | None, *, allow_none: bool = False) -> int | None:
    if limit is None and allow_none:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer.")
    return limit


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _trend_from_row(row: Sequence[object]) -> DailyListeningTrend:
    return DailyListeningTrend(
        listening_date=_as_date(row[0]),
        reference_edition=int(row[1]),
        reference_alignment=str(row[2]),
        dimension=str(row[3]),
        event_count=int(row[4]),
        duration_minutes=float(row[5]),
        unique_entity_count=int(row[6]),
        global_similarity=float(row[7]),
        united_states_similarity=float(row[8]),
        united_states_minus_global=float(row[9]),
        closer_scope=str(row[10]),
        global_event_share_on_public_top=float(row[11]),
        united_states_event_share_on_public_top=float(row[12]),
        global_duration_share_on_public_top=float(row[13]),
        united_states_duration_share_on_public_top=float(row[14]),
        personal_top_entity=_optional_string(row[15]),
        personal_top_entity_detail=_optional_string(row[16]),
    )
