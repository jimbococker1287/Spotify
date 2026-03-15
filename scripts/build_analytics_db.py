from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spotify.analytics_db import refresh_analytics_database


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the DuckDB analytics database for the Spotify project.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw Spotify export directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing run artifacts.")
    parser.add_argument("--no-video", action="store_true", help="Exclude video history files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.analytics_db")

    db_path = refresh_analytics_database(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        include_video=not bool(args.no_video),
        logger=logger,
    )
    if db_path is None:
        return 1
    print(db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
