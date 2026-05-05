from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re

import pandas as pd

from .data import load_streaming_history
from .env import load_local_env
from .lastfm import LastFmArtistChartRow, LastFmClient, LastFmError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.compare_public",
        description="Compare your Spotify listening history to public Last.fm artist charts.",
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for comparison artifacts.")
    parser.add_argument("--lookback-days", type=int, default=180, help="Recent listening window to compare.")
    parser.add_argument("--top-n", type=int, default=50, help="How many top artists to compare.")
    parser.add_argument(
        "--scope",
        choices=("country", "global"),
        default="country",
        help="Compare against a country-specific or global public chart.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="United States",
        help="Country name for the public chart when --scope country is used.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video history files while rebuilding the listening summary.",
    )
    return parser.parse_args()


def _normalize_artist_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _history_artist_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ts", "master_metadata_album_artist_name"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise RuntimeError(f"Streaming history is missing required columns: {', '.join(sorted(missing))}")

    frame = df[["ts", "master_metadata_album_artist_name"]].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    frame["artist_name"] = frame["master_metadata_album_artist_name"].fillna("").astype(str).str.strip()
    frame = frame[frame["ts"].notna()]
    frame = frame[frame["artist_name"] != ""]
    frame["artist_key"] = frame["artist_name"].map(_normalize_artist_name)
    frame = frame[frame["artist_key"] != ""]
    return frame.sort_values("ts").reset_index(drop=True)


def build_public_comparison(
    history_df: pd.DataFrame,
    public_rows: list[LastFmArtistChartRow],
    *,
    lookback_days: int,
    top_n: int,
    scope: str,
    country: str | None,
) -> dict[str, object]:
    if not public_rows:
        raise RuntimeError("No public artist rows were provided for comparison.")

    history = _history_artist_frame(history_df)
    if history.empty:
        raise RuntimeError("No artist listening events were found in the streaming history.")

    history_end_ts = history["ts"].max()
    if pd.isna(history_end_ts):
        raise RuntimeError("Could not determine the latest timestamp in streaming history.")
    cutoff_ts = history_end_ts - pd.Timedelta(days=max(1, int(lookback_days)))
    recent = history[history["ts"] >= cutoff_ts].copy()
    if recent.empty:
        raise RuntimeError("No listening events fall inside the selected recent window.")

    public_rows = public_rows[: max(1, int(top_n))]
    public_by_key = {_normalize_artist_name(row.name): row for row in public_rows}
    public_keys = set(public_by_key)

    recent_artist_counts = recent["artist_name"].value_counts()
    recent_unique_artist_count = int(recent["artist_key"].nunique())
    recent_top_artists = recent_artist_counts.head(max(1, int(top_n)))
    recent_top_rows: list[dict[str, object]] = []
    for artist_name, plays in recent_top_artists.items():
        artist_key = _normalize_artist_name(str(artist_name))
        recent_top_rows.append(
            {
                "artist_name": str(artist_name),
                "artist_key": artist_key,
                "plays": int(plays),
            }
        )

    overlap_rows: list[dict[str, object]] = []
    distinctive_rows: list[dict[str, object]] = []
    for row in recent_top_rows:
        public_row = public_by_key.get(str(row["artist_key"]))
        if public_row is None:
            distinctive_rows.append(
                {
                    "artist_name": row["artist_name"],
                    "your_plays": row["plays"],
                }
            )
            continue
        overlap_rows.append(
            {
                "artist_name": row["artist_name"],
                "your_plays": row["plays"],
                "public_rank": int(public_row.rank),
                "public_listeners": public_row.listeners,
                "public_playcount": public_row.playcount,
                "public_url": public_row.url,
            }
        )

    recent_public_mask = recent["artist_key"].isin(public_keys)
    public_artists_new_to_you: list[dict[str, object]] = []
    heard_artist_keys = set(recent["artist_key"].unique().tolist())
    for public_row in public_rows:
        artist_key = _normalize_artist_name(public_row.name)
        if artist_key in heard_artist_keys:
            continue
        public_artists_new_to_you.append(
            {
                "artist_name": public_row.name,
                "public_rank": int(public_row.rank),
                "public_listeners": public_row.listeners,
                "public_playcount": public_row.playcount,
                "public_url": public_row.url,
            }
        )

    summary = {
        "baseline": {
            "provider": "Last.fm",
            "scope": scope,
            "country": country if scope == "country" else None,
            "top_n": len(public_rows),
        },
        "history_window": {
            "lookback_days": max(1, int(lookback_days)),
            "end_ts": history_end_ts.isoformat(),
            "cutoff_ts": cutoff_ts.isoformat(),
            "recent_streams": int(len(recent)),
            "recent_unique_artists": recent_unique_artist_count,
        },
        "overlap": {
            "shared_artist_count": int(len(overlap_rows)),
            "your_top_artist_overlap_ratio": float(len(overlap_rows) / max(1, len(recent_top_rows))),
            "public_top_artist_overlap_ratio": float(len(overlap_rows) / max(1, len(public_rows))),
            "recent_play_share_on_public_artists": float(recent_public_mask.mean()),
        },
        "shared_artists": overlap_rows,
        "your_distinctive_artists": distinctive_rows[:10],
        "public_artists_new_to_you": public_artists_new_to_you[:10],
    }
    return summary


def write_public_comparison(
    summary: dict[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    baseline = summary["baseline"]
    window = summary["history_window"]
    overlap = summary["overlap"]
    scope = str(baseline["scope"])
    country = str(baseline.get("country") or "global").strip().lower().replace(" ", "-")
    lookback_days = int(window["lookback_days"])

    analysis_dir = output_dir / "analysis" / "public_compare"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lastfm_{scope}_{country}_{lookback_days}d"
    json_path = analysis_dir / f"{stem}.json"
    md_path = analysis_dir / f"{stem}.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Public Listening Comparison",
        "",
        f"- Provider: `{baseline['provider']}`",
        f"- Scope: `{scope}`",
        f"- Country: `{baseline.get('country') or 'global'}`",
        f"- Recent window: last `{lookback_days}` days ending `{window['end_ts']}`",
        f"- Recent streams: `{window['recent_streams']}`",
        f"- Recent unique artists: `{window['recent_unique_artists']}`",
        "",
        "## Similarity Snapshot",
        "",
        f"- Shared artists between your recent top list and the public top list: `{overlap['shared_artist_count']}`",
        f"- Overlap ratio across your recent top artists: `{float(overlap['your_top_artist_overlap_ratio']):.1%}`",
        f"- Overlap ratio across the public top artists: `{float(overlap['public_top_artist_overlap_ratio']):.1%}`",
        f"- Share of your recent plays spent on public-top artists: `{float(overlap['recent_play_share_on_public_artists']):.1%}`",
        "",
        "## Shared Artists",
        "",
    ]

    shared = summary.get("shared_artists", [])
    if isinstance(shared, list) and shared:
        for row in shared[:10]:
            if not isinstance(row, dict):
                continue
            listeners = row.get("public_listeners")
            lines.append(
                f"- {row.get('artist_name')}: your plays `{row.get('your_plays')}`, public rank "
                f"`#{row.get('public_rank')}`, listeners `{listeners if listeners is not None else 'n/a'}`"
            )
    else:
        lines.append("- No shared artists in the selected top lists.")

    lines.extend(
        [
            "",
            "## Your Distinctive Artists",
            "",
        ]
    )

    distinctive = summary.get("your_distinctive_artists", [])
    if isinstance(distinctive, list) and distinctive:
        for row in distinctive[:10]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('artist_name')}: your plays `{row.get('your_plays')}`")
    else:
        lines.append("- Your recent top artists were fully represented in the public chart.")

    lines.extend(
        [
            "",
            "## Public Artists New To You",
            "",
        ]
    )
    new_to_you = summary.get("public_artists_new_to_you", [])
    if isinstance(new_to_you, list) and new_to_you:
        for row in new_to_you[:10]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('artist_name')}: public rank `#{row.get('public_rank')}`, "
                f"listeners `{row.get('public_listeners') if row.get('public_listeners') is not None else 'n/a'}`"
            )
    else:
        lines.append("- You have already listened to every artist in the selected public chart slice.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.compare_public")

    client = LastFmClient.from_env()
    if client is None:
        logger.error("LASTFM_API_KEY is not set. Add it to .env or .env.local to compare against public charts.")
        return 1

    scope = str(args.scope)
    country = str(args.country).strip() if scope == "country" else None
    try:
        public_rows = client.get_top_artists(
            limit=max(1, int(args.top_n)),
            country=country,
        )
    except LastFmError as exc:
        logger.error("Could not load public chart data: %s", exc)
        return 1

    history_df = load_streaming_history(
        Path(args.data_dir).expanduser().resolve(),
        include_video=bool(args.include_video),
        logger=logger,
    )
    summary = build_public_comparison(
        history_df=history_df,
        public_rows=public_rows,
        lookback_days=max(1, int(args.lookback_days)),
        top_n=max(1, int(args.top_n)),
        scope=scope,
        country=country,
    )
    json_path, md_path = write_public_comparison(
        summary,
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )

    overlap = summary["overlap"]
    print(f"comparison_json={json_path}")
    print(f"comparison_md={md_path}")
    print(f"shared_artists={overlap['shared_artist_count']}")
    print(f"recent_play_share_on_public_artists={float(overlap['recent_play_share_on_public_artists']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
