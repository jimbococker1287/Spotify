from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"


class LastFmError(RuntimeError):
    pass


@dataclass(frozen=True)
class LastFmArtistChartRow:
    rank: int
    name: str
    playcount: int | None
    listeners: int | None
    url: str


class LastFmClient:
    def __init__(self, api_key: str, *, timeout_seconds: float = 10.0) -> None:
        self._api_key = api_key.strip()
        self._timeout_seconds = float(timeout_seconds)
        if not self._api_key:
            raise ValueError("A Last.fm API key is required.")

    @classmethod
    def from_env(cls, *, timeout_seconds: float = 10.0) -> LastFmClient | None:
        api_key = os.getenv("LASTFM_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(api_key=api_key, timeout_seconds=timeout_seconds)

    def get_top_artists(
        self,
        *,
        limit: int = 50,
        page: int = 1,
        country: str | None = None,
    ) -> list[LastFmArtistChartRow]:
        params: dict[str, Any] = {
            "limit": max(1, int(limit)),
            "page": max(1, int(page)),
        }
        method = "chart.gettopartists"
        if country and country.strip():
            method = "geo.gettopartists"
            params["country"] = country.strip()

        payload = self._call(method, params)
        root = payload.get("topartists") if method.startswith("geo.") else payload.get("artists")
        if not isinstance(root, dict):
            raise LastFmError("Last.fm response did not include an artist chart payload.")
        rows = root.get("artist", [])
        if not isinstance(rows, list):
            raise LastFmError("Last.fm response did not include an artist list.")

        parsed: list[LastFmArtistChartRow] = []
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            attr_rank = row.get("@attr", {}).get("rank") if isinstance(row.get("@attr"), dict) else None
            parsed.append(
                LastFmArtistChartRow(
                    rank=self._optional_int(attr_rank) or idx,
                    name=name,
                    playcount=self._optional_int(row.get("playcount")),
                    listeners=self._optional_int(row.get("listeners")),
                    url=str(row.get("url", "")).strip(),
                )
            )
        return parsed

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode(
            {
                "method": method,
                "api_key": self._api_key,
                "format": "json",
                **params,
            }
        )
        request = Request(f"{LASTFM_API_URL}?{query}", headers={"Accept": "application/json"}, method="GET")
        return self._request_json(request)

    def _request_json(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else exc.reason
            raise LastFmError(f"Last.fm API request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LastFmError(f"Last.fm API request failed: {exc.reason}") from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LastFmError("Last.fm API returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise LastFmError("Last.fm API returned an unexpected response shape.")
        if "error" in payload:
            message = str(payload.get("message", "Unknown error")).strip() or "Unknown error"
            raise LastFmError(f"Last.fm API error: {message}")
        return payload

    def _optional_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
