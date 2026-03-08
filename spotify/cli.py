from __future__ import annotations

import argparse
from typing import Sequence

from .config import add_cli_arguments, build_config
from .env import load_local_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spotify",
        description="Train and compare deep models on Spotify extended streaming history.",
    )
    add_cli_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = build_config(args)

    from .pipeline import run_pipeline

    run_pipeline(config)
    return 0
