# Recommender Safety Platform

Week 9 turns the safety layer into a reusable platform surface instead of leaving it implied inside the Spotify training pipeline.

Week 10 then freezes the comparison side of that platform with the benchmark contract in `docs/benchmark_contract.md`, so safety claims and benchmark claims use the same stable operating language.

## Platform Split

The reusable core lives in:

- `spotify/recommender_safety.py`
- `spotify/safety_platform.py`

The Spotify-specific wrappers live in:

- `spotify/backtesting.py`
- `spotify/drift.py`
- `spotify/governance.py`
- `spotify/evaluation.py`

That split is the key Week 9 framing rule: keep the platform primitives generic, then let Spotify modules adapt them to the repository's run artifacts and product language.

## Minimum Public API

The minimum public API is organized into four groups:

1. `temporal_backtest`
   `build_temporal_backtest_windows`, `run_temporal_backtest_benchmark`, `summarize_backtest_rows`, `write_temporal_backtest_artifacts`
2. `drift`
   `SequenceSplitSnapshot`, `compute_context_feature_drift_rows`, `compute_segment_share_shift_rows`, `compute_target_distribution_drift`
3. `promotion_gate`
   `evaluate_promotion_gate`
4. `abstention`
   `build_conformal_abstention_summary`

The curated Week 9 export surface is in `spotify/safety_platform.py`, which re-exports those primitives and describes how the Spotify wrappers map onto them.

## Spotify Integration Map

- `spotify/backtesting.py:run_temporal_backtest`
  wraps the generic temporal-window utilities for the repo's classical and deep benchmark runners.
- `spotify/drift.py:run_drift_diagnostics`
  builds `SequenceSplitSnapshot` objects from `PreparedData`, then uses the generic drift helpers.
- `spotify/governance.py:evaluate_champion_gate`
  maps the generic promotion payload into the repo's champion-gate shape.
- `spotify/evaluation.py:run_extended_evaluation`
  uses the conformal abstention helper to attach calibrated risk summaries to model evaluation.

## How To Reuse Outside Spotify

You can treat this as a small recommender-safety SDK even if your project has no Spotify concepts at all.

```python
from pathlib import Path

import numpy as np
import pandas as pd

from spotify.safety_platform import (
    SequenceSplitSnapshot,
    build_conformal_abstention_summary,
    compute_context_feature_drift_rows,
    compute_segment_share_shift_rows,
    compute_target_distribution_drift,
    evaluate_promotion_gate,
    run_temporal_backtest_benchmark,
)


backtest_rows = run_temporal_backtest_benchmark(
    n_rows=12000,
    folds=4,
    metric_name="watch_time",
    output_dir=Path("outputs/safety/backtest"),
    evaluators={
        "retriever": lambda window: {"watch_time": 0.41 + (0.01 * window.fold)},
        "reranker": lambda window: {"watch_time": 0.46 + (0.02 * window.fold)},
    },
)

baseline = SequenceSplitSnapshot(
    name="baseline",
    context=np.array([[0.0, 1.0], [1.0, 2.0]], dtype="float32"),
    targets=np.array([0, 1], dtype="int32"),
    frame=pd.DataFrame({"device": ["mobile", "desktop"]}),
)
canary = SequenceSplitSnapshot(
    name="canary",
    context=np.array([[2.0, 4.0], [3.0, 5.0]], dtype="float32"),
    targets=np.array([1, 1], dtype="int32"),
    frame=pd.DataFrame({"device": ["desktop", "desktop"]}),
)

context_drift = compute_context_feature_drift_rows(
    feature_names=["recency", "depth"],
    reference_split=baseline,
    comparison_splits=[canary],
)
segment_drift = compute_segment_share_shift_rows(
    reference_split=baseline,
    comparison_splits=[canary],
    segment_extractors={"device_type": lambda frame: frame["device"].to_numpy()},
)
target_drift = compute_target_distribution_drift(reference_split=baseline, comparison_splits=[canary])

gate = evaluate_promotion_gate(
    history_csv=Path("outputs/safety/history.csv"),
    current_run_id="run_2026_03_29",
    current_rows=[{"model_name": "candidate", "watch_time": 0.48}],
    metric_name="watch_time",
    regression_threshold=0.01,
)

abstention = build_conformal_abstention_summary(
    tag="video_home_feed",
    val_proba=np.array([[0.8, 0.2], [0.3, 0.7]], dtype="float32"),
    val_y=np.array([0, 1], dtype="int32"),
)
```

## Remaining Spotify-Coupled Assumptions

The safety primitives are mostly generic, but a few assumptions still matter:

- Promotion gating currently expects CSV history rows with `run_id`, `profile`, and group-level score columns.
- Segment-share drift requires pandas frames when you want named segment extractors; pure context drift does not.
- Conformal abstention assumes a classification-style recommender that can emit probability distributions over discrete outcomes.
- Plot writing in the artifact helpers depends on `matplotlib`, so headless deployments may prefer using the row-level outputs only.
- The richer product and operator language still lives in Spotify wrappers such as `control_room.py`, not in the platform layer.

## Week 9 Exit Check

Week 9 is complete when someone can:

- read this document and understand the reusable surface without opening `spotify/pipeline.py`
- import from `spotify.safety_platform` and see the minimum public API
- tell where Spotify-specific wrappers begin and generic safety primitives end
