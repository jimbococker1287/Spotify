from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spotify.analytics_db import refresh_analytics_database
from spotify.analytics_warehouse import build_analytics_warehouse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the local DuckDB analytics database and analytics warehouse for the Spotify project."
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw Spotify export directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing run artifacts.")
    parser.add_argument("--no-video", action="store_true", help="Exclude video history files.")
    parser.add_argument(
        "--warehouse-only",
        action="store_true",
        help="Build the local analytics warehouse without refreshing DuckDB.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.analytics_db")

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.warehouse_only:
        warehouse_root = build_analytics_warehouse(
            data_dir=data_dir,
            output_dir=output_dir,
            include_video=not bool(args.no_video),
            logger=logger,
        )
        print(warehouse_root)
        return 0

    db_path = refresh_analytics_database(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=not bool(args.no_video),
        logger=logger,
    )
    if db_path is None:
        return 1
    print(db_path)
    print(output_dir / "analytics" / "warehouse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
