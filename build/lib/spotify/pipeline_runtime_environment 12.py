from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

from .config import PipelineConfig


def configure_pipeline_runtime_environment(*, config: PipelineConfig, run_dir: Path) -> None:
    isolate_mpl_cache = os.getenv("SPOTIFY_ISOLATE_MPL_CACHE", "0").strip().lower() in ("1", "true", "yes", "on")
    if isolate_mpl_cache:
        mpl_config_dir = run_dir / ".mplconfig"
        xdg_cache_dir = run_dir / ".cache"
    else:
        mpl_config_dir = config.output_dir / ".mplconfig"
        xdg_cache_dir = config.output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    np.random.seed(config.random_seed)
    random.seed(config.random_seed)


__all__ = ["configure_pipeline_runtime_environment"]
