from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from spotify.aws_athena import export_athena_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a curated Athena-ready bundle for the Spotify project.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw Spotify export directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Project outputs directory.")
    parser.add_argument(
        "--export-dir",
        type=str,
        default="outputs/aws_athena_bundle",
        help="Local directory where the Athena bundle will be written.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        required=True,
        help="Destination S3 prefix, for example s3://my-bucket/spotify-athena.",
    )
    parser.add_argument(
        "--database-name",
        type=str,
        default="spotify_taste_os",
        help="Athena database name to embed in the generated SQL.",
    )
    parser.add_argument("--no-video", action="store_true", help="Exclude video history files from the curated export.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.aws_athena")

    report = export_athena_bundle(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        export_dir=Path(args.export_dir).expanduser().resolve(),
        include_video=not bool(args.no_video),
        s3_prefix=args.s3_prefix,
        database_name=args.database_name.strip() or "spotify_taste_os",
        logger=logger,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
