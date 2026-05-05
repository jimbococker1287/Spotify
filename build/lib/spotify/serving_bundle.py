from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .predict_next import build_prediction_serving_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.serving_bundle",
        description="Materialize production serving bundles for prediction services.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to outputs/runs/<run_id> or outputs/models/champion. Defaults to champion alias.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to raw Streaming_History JSON files used once to build the serving bundle.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Build the audio+video bundle instead of the audio-only bundle.",
    )
    parser.add_argument(
        "--all-contexts",
        action="store_true",
        help="Build both audio-only and audio+video bundles.",
    )
    return parser.parse_args()


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.serving_bundle")

    run_dir, _ = resolve_prediction_run_dir(args.run_dir)
    data_dir = Path(args.data_dir).expanduser().resolve()
    include_video_values = [False, True] if bool(args.all_contexts) else [bool(args.include_video)]

    for include_video in include_video_values:
        bundle_path = build_prediction_serving_bundle(
            run_dir=run_dir,
            data_dir=data_dir,
            include_video=include_video,
            logger=logger,
        )
        print(f"serving_bundle={bundle_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
