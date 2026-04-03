# Refactor Optimization Log

## 2026-03-28

### Creator Label Intelligence

Scope:
- Reduced repeated transition filtering in `build_creator_label_intelligence` by precomputing seed transition lookups once.
- Replaced per-row release-date parsing with a fast numeric parser and an `lru_cache` fast path.
- Added a crossover threshold in `_top_similarity_neighbors` so small artist spaces keep the cheaper full-sort path while larger spaces use `argpartition`.
- Added a regression test for the transition-helper ordering and share math.

Cleanup:
- Diff stat for this pass: `2 files changed, 233 insertions(+), 60 deletions(-)`.
- Runtime code: `spotify/creator_label_intelligence.py` `+211/-59`.
- Tests: `tests/test_creator_label_intelligence.py` `+22/-1`.

Benchmark method:
- Baseline: committed `HEAD` version of `spotify/creator_label_intelligence.py`.
- Current: working-tree version after this refactor pass.
- Workload: synthetic `150,000`-row listening history, `250` artists, `30,000` sessions, `6` seed artists, local fake catalog client.
- Validation gate: baseline and refactored outputs matched for `graph_summary` and the first `20` rows of `artist_adjacency`, `fan_migration`, `release_whitespace`, and `opportunities`.

Measured impact:
- End-to-end cold build (`build_creator_label_intelligence`, release-date cache cleared before each current run):
  - Baseline median: `0.06077s`
  - Current median: `0.02626s`
  - Improvement: `2.31x` faster, `56.8%` lower median runtime
- End-to-end warm build (`build_creator_label_intelligence`, cache left hot across runs):
  - Baseline median: `0.04563s`
  - Current median: `0.02060s`
  - Improvement: `2.21x` faster, `54.8%` lower median runtime
- cProfile single-build comparison on the same workload:
  - Baseline: `311,729` function calls in `0.180s`
  - Current: `226,347` function calls in `0.109s`
  - Change: about `27.4%` fewer function calls
- `_top_similarity_neighbors` microbenchmark:
  - `128` artists: roughly neutral (`0.99x`)
  - `256` artists: `1.52x` faster
  - `512` artists: `2.63x` faster
  - `1024` artists: `1.88x` faster

Interpretation:
- This pass should make creator-intelligence payload generation noticeably faster, especially in repeated local/demo/server flows where the same process builds multiple payloads.
- If a run is dominated by live Spotify API latency instead of local computation, the user-visible speedup will be smaller than the CPU-only measurements above.
- These numbers describe the creator-intelligence surface, not the entire project runtime.

### Inference Top-K And Safety Drift

Scope:
- Added a shared fast 1D top-k helper and routed `predict_next`, `predict_service`, and the retrieval ANN recall path through it.
- Refactored `compute_segment_share_shift_rows` to precompute segment bucket counts instead of rebuilding the same reference counts for every comparison split.
- Added targeted regression coverage for the shared top-k helper and multi-split safety drift behavior.

Cleanup:
- Diff stat for this pass: `7 files changed, 85 insertions(+), 19 deletions(-)`.
- Runtime code:
  - `spotify/ranking.py` `+15/-0`
  - `spotify/predict_next.py` `+2/-1`
  - `spotify/predict_service.py` `+2/-1`
  - `spotify/retrieval.py` `+3/-3`
  - `spotify/recommender_safety.py` `+22/-13`
- Tests:
  - `tests/test_ranking_metrics.py` `+10/-1`
  - `tests/test_recommender_safety_platform.py` `+31/-0`

Benchmark method:
- Baseline: committed `HEAD` version of the affected modules, with the old prediction path represented by full `np.argsort(...)[::-1][:k]`.
- Current: working-tree version after this refactor pass.
- Validation gate:
  - `topk_indices_1d` matched the old full-sort ordering exactly.
  - `compute_segment_share_shift_rows` matched the baseline rows exactly.

Measured impact:
- `topk_indices_1d` on a `2,000`-class probability vector with `top_k=20`:
  - Baseline median: `18.79us`
  - Current median: `6.04us`
  - Improvement: `3.11x` faster, `67.9%` lower median runtime
- `topk_indices_1d` on a `10,000`-class probability vector with `top_k=20`:
  - Baseline median: `162.21us`
  - Current median: `11.54us`
  - Improvement: `14.05x` faster, `92.9%` lower median runtime
- `compute_segment_share_shift_rows` with `50,000` rows, `6` comparison splits, and `4` segment extractors:
  - Baseline median: `0.67775s`
  - Current median: `0.41453s`
  - Improvement: `1.63x` faster, `38.8%` lower median runtime

Interpretation:
- Prediction-serving paths should benefit directly when the label space is large, because they now avoid fully sorting every probability vector just to return a small `top_k`.
- Safety drift analysis gets more efficient as the number of comparison splits grows, because the reference split work is now reused instead of recomputed.
- These gains are local to prediction ranking and safety analysis; they improve hot paths, but they are not a claim about the entire project runtime.

### Public Insights, Evaluation, And Reporting

Scope:
- Reworked `public_insights._top_tracks_from_history` to avoid `groupby(...).agg(... lambda values.mode())` and instead derive canonical metadata from grouped counts.
- Replaced `public_insights._parse_release_date` with a fast numeric parser plus `lru_cache`, which also speeds release sorting in the release-tracker and discography flows.
- Replaced `evaluation._build_label_lookup` row-wise `iterrows()` usage with column-wise iteration.
- Refactored `reporting.persist_to_sqlite` to stream SQLite row inserts via helper iterators instead of building large intermediate Python lists.
- Added direct regression coverage for top-track canonicalization, label lookup behavior, and SQLite persistence output.

Cleanup:
- Tracked diff stat for this pass: `4 files changed, 153 insertions(+), 57 deletions(-)`.
- Runtime code:
  - `spotify/public_insights.py` `+47/-12`
  - `spotify/evaluation.py` `+5/-2`
  - `spotify/reporting.py` `+61/-43`
- Tests:
  - `tests/test_public_insights.py` `+40/-0`
  - New file: `tests/test_reporting_and_evaluation.py` `+71`

Benchmark method:
- Baseline: committed `HEAD` version of the affected modules.
- Current: working-tree version after this refactor pass.
- Validation gate:
  - `public_insights._top_tracks_from_history` matched the baseline output exactly on a `120,000`-row synthetic history.
  - `public_insights._parse_release_date` matched the baseline parsed timestamps exactly.
  - `evaluation._build_label_lookup` matched the baseline lookup exactly.
  - `reporting.persist_to_sqlite` produced the same table row counts as the baseline.

Measured impact:
- `public_insights._top_tracks_from_history` on a `120,000`-row synthetic history:
  - Baseline median: `0.36444s`
  - Current median: `0.06780s`
  - Improvement: `5.38x` faster, `81.4%` lower median runtime
- `public_insights._parse_release_date` over `23,790` calls:
  - Baseline median: `1.95778s`
  - Current cold median: `0.00194s`
  - Current warm median: `0.00092s`
  - Improvement: `1008.09x` faster cold, `2124.51x` faster warm
- `evaluation._build_label_lookup` on a `200,000`-row frame:
  - Baseline median: `0.04858s`
  - Current median: `0.00181s`
  - Improvement: `26.84x` faster, `96.3%` lower median runtime
- `reporting.persist_to_sqlite` with `18` models and `180` epochs per model:
  - Baseline median: `0.00503s`
  - Current median: `0.00480s`
  - Improvement: `1.05x` faster, `4.7%` lower median runtime

Interpretation:
- Public insights should feel materially snappier in the “top tracks” and release-oriented report flows, especially when working over larger history windows or release lists.
- The reporting change is a smaller speedup, but it also lowers peak Python list allocation during SQLite persistence, which should scale better as model counts and epoch histories grow.
- These gains affect public-insight generation, evaluation helpers, and reporting persistence; they are not a blanket claim about every project command.

### Data Quality, Control Room, And Taste OS Demo

Scope:
- Tightened `data_quality` boolean validation so float-coded values like `0.0` and `1.0` are rejected, while trimmed case-insensitive string values such as `" false "` still pass.
- Reworked the boolean gate helper to use an exact-value fast path before doing string normalization work.
- Reduced `control_room` row materialization in the weekly summary writer and trend snapshot lookup by switching the hot paths to tuple iteration instead of repeated dict conversion.
- Reworked `taste_os_demo` candidate reranking and journey-plan selection so large artist spaces avoid full descending sorts unless the shortlist cannot satisfy the repeat-avoidance policy.
- Added targeted regression coverage for the stricter data-quality boolean handling and for the reranker’s shortlist/union behavior.

Cleanup:
- Current cumulative diff in the touched files relative to `HEAD`: `5 files changed, 332 insertions(+), 95 deletions(-)`.
- Runtime code:
  - `spotify/data_quality.py` `+20/-3`
  - `spotify/control_room.py` `+47/-27`
  - `spotify/taste_os_demo.py` `+162/-64`
- Tests:
  - `tests/test_data_quality_gate.py` `+13/-0`
  - `tests/test_taste_os_demo.py` `+90/-1`
- Note: `control_room.py` and `taste_os_demo.py` were already part of earlier uncommitted refactor passes in this worktree, so these counts are cumulative for the touched files rather than a clean per-pass patch size.

Benchmark method:
- Baseline: committed `HEAD` version of the affected modules.
- Current: working-tree version after this refactor pass.
- Validation gate:
  - `data_quality._bool_like_invalid_count` matched the baseline count on a `240,000`-row mixed object series using canonical boolean-like values, and a targeted regression test separately verifies the intentional stricter handling for float-coded values like `0.0`.
  - `data_quality.evaluate_data_quality` matched the baseline report structure on the same `240,000`-row synthetic frame aside from the new float strictness covered in tests.
  - `control_room._build_ops_trends` and `_write_weekly_ops_summary` matched the baseline output exactly on an `18,000`-snapshot synthetic history.
  - `taste_os_demo._surface_reranked_indices`, `_journey_plan_rows`, and `build_taste_os_demo_payload` matched the baseline candidate and journey ordering exactly on a `2,048`-artist synthetic workload.

Measured impact:
- `data_quality._bool_like_invalid_count` on a `240,000`-row mixed object series:
  - Baseline median: `0.04411s`
  - Current median: `0.03819s`
  - Improvement: `1.15x` faster, `13.4%` lower median runtime
- `data_quality.evaluate_data_quality` on a `240,000`-row synthetic gate frame:
  - Baseline median: `0.23385s`
  - Current median: `0.07843s`
  - Improvement: `2.98x` faster, `66.5%` lower median runtime
- `control_room._build_ops_trends` on an `18,000`-snapshot history:
  - Baseline median: `0.02250s`
  - Current median: `0.02257s`
  - Improvement: effectively neutral (`1.00x`)
- `control_room._write_weekly_ops_summary` on the same `18,000`-snapshot history:
  - Baseline median: `0.02492s`
  - Current median: `0.02395s`
  - Improvement: `1.04x` faster, `3.9%` lower median runtime
- `taste_os_demo._surface_reranked_indices` with `2,048` artists and `top_k=8`:
  - Baseline median: `0.00010s`
  - Current median: `0.00007s`
  - Improvement: `1.44x` faster, `30.4%` lower median runtime
- `taste_os_demo._journey_plan_rows` on a `2,048`-artist planning workload:
  - Baseline median: `0.00103s`
  - Current median: `0.00056s`
  - Improvement: `1.84x` faster, `45.7%` lower median runtime
- `taste_os_demo.build_taste_os_demo_payload` on the same `2,048`-artist synthetic setup:
  - Baseline median: `0.00345s`
  - Current median: `0.00235s`
  - Improvement: `1.47x` faster, `32.0%` lower median runtime

Interpretation:
- This pass meaningfully improves the end-to-end data-quality gate and the Taste OS demo generation path, with the biggest wins coming from avoiding repeated per-value normalization and repeated full-sort work.
- The control-room weekly summary path is a smaller improvement, and the ops-trend helper is effectively neutral in the current benchmark; the value there is mostly cleaner row extraction and less intermediate Python object churn.
- These measurements describe the specific analytics/demo/reporting paths touched here, not the entire project runtime. They do support a real claim that the project is faster in these surfaces, especially when running data-quality gates or generating Taste OS demo payloads repeatedly.

## 2026-03-29

### Policy Eval, Multimodal, Journey Planner, And Uncertainty

Scope:
- Added a shared `ranking.topk_indices_2d` helper so wide probability matrices can return exact top-k rows without fully sorting every class dimension.
- Routed `policy_eval` through the shared 2D top-k helper, vectorized the synthetic session-repeat utility construction, and replaced the per-row discounted-reward loop with array math.
- Reworked `multimodal._transition_features` to compute smoothed transition entropy without building a dense artist-by-artist count matrix, and switched neighbor extraction to the shared 2D top-k helper.
- Refactored `journey_planner` beam expansion so it batches similarity/novelty/repeat/energy scoring per beam instead of recomputing those values candidate-by-candidate, while still preserving the selected path.
- Added a small fast path in `uncertainty.conformal_prediction_sets` for the very common singleton/two-item prediction-set cases.
- Added regression coverage for shared 2D top-k behavior, smoothed transition-entropy math, multimodal neighbor ordering, journey-plan artifact generation, and small conformal prediction-set ordering.

Cleanup:
- Current cumulative diff in the touched files relative to `HEAD`: `8 files changed, 272 insertions(+), 56 deletions(-)`.
- Runtime code:
  - `spotify/ranking.py` `+29/-7`
  - `spotify/policy_eval.py` `+14/-11`
  - `spotify/multimodal.py` `+12/-6`
  - `spotify/journey_planner.py` `+64/-30`
  - `spotify/uncertainty.py` `+5/-1`
- Tests:
  - `tests/test_ranking_metrics.py` `+25/-1`
  - `tests/test_uncertainty.py` `+17/-0`
  - `tests/test_moonshot_lab.py` `+106/-0`

Benchmark method:
- Baseline: committed `HEAD` version of the affected modules, with the shared 2D top-k baseline represented by full `np.argsort(..., axis=1)[:, ::-1][:, :k]`.
- Current: working-tree version after this refactor pass.
- Validation gate:
  - `ranking.topk_indices_2d` matched the old full-sort top-k output exactly on a `128 x 5,000` matrix.
  - `policy_eval.run_policy_simulation` matched the baseline JSON output exactly on a synthetic `2,048`-artist bundle workload.
  - `multimodal._transition_features` and `_top_neighbors` matched the baseline outputs exactly on synthetic transition and embedding workloads.
  - `journey_planner.build_journey_plans` matched the baseline selected artist path exactly; only the accumulated `plan_score` showed tiny floating-point differences within `1e-5`.
  - `uncertainty.conformal_prediction_sets` matched the baseline prediction sets exactly on a `4,096 x 2,048` sparse-threshold workload.

Measured impact:
- `ranking.topk_indices_2d` on a `128 x 5,000` probability matrix with `top_k=20`:
  - Baseline median: `0.02452s`
  - Current median: `0.00389s`
  - Improvement: `6.31x` faster, `84.1%` lower median runtime
- `policy_eval.run_policy_simulation` on a `2,048`-artist synthetic bundle workload with `k=20`:
  - Baseline median: `0.14953s`
  - Current median: `0.05244s`
  - Improvement: `2.85x` faster, `64.9%` lower median runtime
- `multimodal._transition_features` on a `180,000`-row, `512`-artist history:
  - Baseline median: `0.00894s`
  - Current median: `0.00770s`
  - Improvement: `1.16x` faster, `13.9%` lower median runtime
- `multimodal._top_neighbors` on a `1,200`-artist embedding space:
  - Baseline median: `0.04883s`
  - Current median: `0.01139s`
  - Improvement: `4.29x` faster, `76.7%` lower median runtime
- `journey_planner.build_journey_plans` on a `2,048`-artist synthetic planning workload:
  - Baseline median: `0.01644s`
  - Current median: `0.00607s`
  - Improvement: `2.71x` faster, `63.1%` lower median runtime
- `uncertainty.conformal_prediction_sets` on a `4,096 x 2,048` sparse-threshold workload:
  - Baseline median: `0.01272s`
  - Current median: `0.00825s`
  - Improvement: `1.54x` faster, `35.2%` lower median runtime

Interpretation:
- This pass materially improves offline evaluation and moonshot analysis paths, especially when they operate over wide artist spaces where full sorts used to dominate runtime.
- The strongest end-to-end gains are in policy simulation and journey planning, while multimodal neighbor lookup also benefits substantially from the shared 2D top-k helper.
- `multimodal._transition_features` is a smaller speedup in this benchmark, but it removes the dense `num_artists x num_artists` intermediate allocation, which should scale more safely as artist catalogs grow.
- These numbers describe the targeted research/moonshot/uncertainty surfaces, not every project command. They do support a concrete claim that these subsystems now run faster.

### Digital Twin, Stress Test, Safe Policy, And Run Startup

Scope:
- Added a batched digital-twin rollout path that caches static transition/similarity tensors, avoids the large broadcasted repeat-penalty allocation, and reuses serving features within each rollout step.
- Switched the stress-test lab to execute rollouts in batches when the default simulator is in use, while keeping the old per-session path available for monkeypatched tests.
- Refactored safe-policy learning to score whole validation buckets with vectorized reward math instead of looping row-by-row through single-session rollouts.
- Deferred the `pandas` import inside `run_artifacts.py` so `pipeline.py` can import `write_json` and related helpers without violating the repo's TensorFlow-first import ordering on macOS.
- Added regression coverage for batched digital-twin rollouts and reran the moonshot, stress-test, and run-artifact tests.

Cleanup:
- Current cumulative diff in the touched runtime files relative to `HEAD`: `4 files changed, 364 insertions(+), 104 deletions(-)`.
- Runtime code:
  - `spotify/digital_twin.py` `+224/-23`
  - `spotify/stress_test.py` `+94/-38`
  - `spotify/safe_policy.py` `+34/-39`
  - `spotify/run_artifacts.py` `+12/-4`
- Tests:
  - `tests/test_digital_twin.py` new file, `99` lines

Benchmark method:
- Baseline: committed `HEAD` version of the affected modules in a detached worktree rooted at `/tmp/spotify-head-current.drnj2I`.
- Current: working-tree version after the batching, vectorization, and lazy-import changes.
- Targeted hotspot benchmark:
  - Synthetic `run_stress_test_lab` workload with `3,200` held-out sessions, `30`-track histories, `120` artists, a causal skip artifact, and the default `2,500`-session sampling cap.
  - Each side ran `3` repeats with identical seeded inputs.
- Bounded full-script benchmark:
  - Real `scripts/run_everything.sh` flow using the local Spotify export under `/Users/akashponugoti/Documents/Spotify/data/raw`.
  - Shared bounded settings to keep the comparison tractable: `EPOCHS=2`, `OPTUNA_TRIALS=1`, `BACKTEST_FOLDS=2`, `CLASSICAL_MAX_TRAIN_SAMPLES=5000`, `CLASSICAL_MAX_EVAL_SAMPLES=2500`, `SPOTIFY_STRESS_TEST_PROGRESS_EVERY=0`, plus CLI overrides `--models dense --classical-models logreg --optuna-models logreg --backtest-models logreg --max-artists 120 --no-shap --no-mlflow`.
  - Baseline run dir: `/tmp/spotify-fullbench-baseline.iGnqgJ/runs/20260329_180709_bench-baseline`
  - Current run dir: `/tmp/spotify-fullbench-current.hFw9Go/runs/20260329_181426_bench-current`
- Validation gate:
  - `simulate_rollout_batch_summary` matched the single-session rollout outputs exactly in a deterministic no-early-end regression test.
  - `tests/test_digital_twin.py`, `tests/test_stress_test.py`, `tests/test_moonshot_lab.py`, and `tests/test_run_artifacts.py` all passed.
  - `ruff` passed on the touched modules and tests.

Measured impact:
- `run_stress_test_lab` targeted benchmark:
  - Baseline median: `3.0513s`
  - Current median: `0.0514s`
  - Improvement: `59.34x` faster, `98.3%` lower median runtime
- Bounded `scripts/run_everything.sh` benchmark:
  - Baseline total from run log: `144.700s`
  - Current total from run log / phase timing: `78.783s`
  - Improvement: `1.84x` faster, `45.6%` lower end-to-end runtime
- Moonshot tail within the bounded full run, measured from the first post-journey-planning log to moonshot completion:
  - Baseline: `67.421s`
  - Current: `0.369s`
  - Improvement: `182.7x` faster, `99.5%` lower runtime
- Safe-policy learning inside the full run:
  - Baseline: about `11.22s`
  - Current: about `0.05s`
  - Improvement: about `229x` faster, `99.6%` lower runtime
- Stress-test scenarios inside the full run:
  - Baseline: about `56.2s` total across `10` scenario/policy evaluations
  - Current: about `0.31s` total
  - Improvement: about `181x` faster, `99.4%` lower runtime

Interpretation:
- This pass turns the moonshot lab from a major full-run bottleneck into a comparatively small tail stage, and that change is large enough to move the bounded end-to-end launcher runtime by nearly half.
- The biggest raw speedup comes from batch execution in the stress-test lab, but vectorized safe-policy scoring and cached digital-twin rollout inputs also contribute materially.
- The lazy `pandas` import in `run_artifacts.py` is not just a cleanup; it preserves the repository's TensorFlow-before-pandas/sklearn startup rule and prevents the current pipeline from deadlocking during the first training epoch on macOS.
- These results come from the bounded full-script benchmark configuration above, not the unconstrained default full launcher, but they are still real end-to-end measurements on the actual local export data and launcher script.

### Retrieval Stack Acceleration

Scope:
- Vectorized the self-supervised retrieval pretraining loop in `spotify/retrieval.py` by replacing the per-example Python embedding updates with batched scatter updates via `np.add.at`.
- Switched retrieval softmax and sigmoid math to stay in `float32`, which reduces conversion overhead inside the inner training loops.
- Reused cached validation and test retrieval scores/session vectors instead of rescoring the same splits multiple times for retrieval metrics, ANN diagnostics, and reranker evaluation.
- Added a default cap for ANN recall/latency diagnostics so the summary metrics evaluate a deterministic sample instead of the full validation and test splits every run.

Cleanup:
- Current cumulative diff in the touched runtime file relative to `HEAD`: `1 file changed, 67 insertions(+), 35 deletions(-)`.
- Runtime code:
  - `spotify/retrieval.py` `+67/-35`

Benchmark method:
- Main-worktree caveat:
  - The shared workspace currently has an unrelated deletion of `spotify/data.py`, which prevents imports like `spotify.data` from resolving cleanly.
  - To avoid stepping on that in-progress refactor, validation and benchmarking for this pass were run in clean detached worktrees rather than the dirty main worktree.
- Retrieval hotspot benchmark:
  - Baseline worktree: `/tmp/spotify-head-current.drnj2I`
  - Candidate worktree with only the retrieval patch: `/tmp/spotify-retrieval-cand.wrmz1z`
  - Shared dataset: the prepared-data cache bundle at `/tmp/spotify-fullbench-current.hFw9Go/cache/prepared_data/161ca5021d30d1f4dda7ae51/prepared_bundle.joblib`
  - Command shape: direct `train_retrieval_stack(...)` and direct `train_self_supervised_artist_embeddings(..., objective_name="cooccurrence")` timing with the same random seed and default retrieval env settings.
- Bounded launcher benchmark for the retrieval patch in isolation:
  - Baseline run dir: `/tmp/spotify-retrieval-full-baseline.aCR7v6/runs/20260402_175700_bench-retrieval-baseline`
  - Candidate run dir: `/tmp/spotify-retrieval-full-candidate.XYFAaC/runs/20260402_180010_bench-retrieval-candidate`
  - Shared bounded settings: `EPOCHS=2`, `OPTUNA_TRIALS=1`, `BACKTEST_FOLDS=2`, `CLASSICAL_MAX_TRAIN_SAMPLES=5000`, `CLASSICAL_MAX_EVAL_SAMPLES=2500`, `SPOTIFY_STRESS_TEST_PROGRESS_EVERY=0`, plus CLI overrides `--models dense --classical-models logreg --optuna-models logreg --backtest-models logreg --max-artists 120 --no-shap --no-mlflow`.
- Validation gate:
  - `ruff` passed on `spotify/retrieval.py` and `tests/test_retrieval_and_friction.py`.
  - `tests/test_retrieval_and_friction.py` passed in the clean candidate worktree.

Measured impact:
- Direct `train_retrieval_stack(...)` benchmark on the shared prepared bundle:
  - Baseline: `103.96s`
  - Candidate: `27.04s`
  - Improvement: `3.85x` faster, `74.0%` lower runtime
- Direct `train_self_supervised_artist_embeddings(..., "cooccurrence")` benchmark on the same bundle:
  - Baseline: `24.79s`
  - Candidate: `3.13s`
  - Improvement: `7.93x` faster, `87.4%` lower runtime
- Bounded `scripts/run_everything.sh` benchmark for this retrieval patch in isolation:
  - Baseline total from run log: `150.42s`
  - Candidate total from run log: `120.94s`
  - Improvement: `1.24x` faster, `19.6%` lower end-to-end runtime
- Retrieval model fit time reported in the bounded launcher logs:
  - Baseline `retrieval_dual_encoder` / `retrieval_reranker`: `36.04s`
  - Candidate `retrieval_dual_encoder` / `retrieval_reranker`: `7.77s`
  - Improvement: `4.64x` faster, `78.4%` lower retrieval fit time in the full launcher

Interpretation:
- This pass materially accelerates the retrieval stack itself; the dominant win comes from eliminating the inner Python update loop in self-supervised pretraining.
- Even without the separate moonshot/stress-test optimizations that only exist in the dirty workspace, the retrieval patch alone trims about one-fifth off the bounded full launcher on clean `HEAD`.
- Using the last instrumented dirty-worktree phase timings together with the measured retrieval-fit reduction, the likely top remaining bottlenecks in that optimized tree are now:
  - `tensorflow_runtime_init` at about `10.09s`
  - `retrieval_stack` at about `7.34s` (inferred from the measured retrieval-fit reduction)
  - `temporal_backtest` at about `5.83s`
  - `deep_model_training` at about `3.13s`
  - `robustness_slice_evaluation` at about `2.33s`
- Because the dirty main workspace currently has the unrelated `spotify/data.py` deletion, that final bottleneck ranking is an informed inference from the last instrumented dirty run rather than a freshly rerun dirty-workspace measurement.

### Temporal Backtest Fold Reuse + Auto Parallelism

Scope:
- Added a dedicated single-pass classical backtest scorer in `spotify/backtesting.py` so temporal backtesting no longer calls the full benchmark evaluator with the same test matrix twice.
- Refactored temporal backtest window resolution around reusable capped slices, which removes repeated tail-cap array churn and makes the classical and deep fold loops share the same fold bookkeeping.
- Deferred sequence/context/skip-array concatenation until a selected backtest model actually needs those modalities.
- Added a small auto-worker heuristic for classical temporal backtests: when `SPOTIFY_BACKTEST_WORKERS` is unset and there is more than one independent classical job, temporal backtesting now defaults to up to `2` workers while still forcing estimator-internal `n_jobs=1` to avoid oversubscription.

Cleanup:
- Runtime/test files touched for this pass:
  - `spotify/backtesting.py`
  - `tests/test_drift_and_backtesting.py`
- Added regression coverage for:
  - single-pass classical eval scoring
  - auto backtest worker resolution

Benchmark method:
- Baseline worktree: `/tmp/spotify-backtest-base.jOt1QM`
- Candidate worktree: `/tmp/spotify-backtest-cand.UUJka4`
- Shared dataset: `/tmp/spotify-fullbench-current.hFw9Go/cache/prepared_data/161ca5021d30d1f4dda7ae51/prepared_bundle.joblib`
- Shared workload: direct `run_temporal_backtest(...)` on the real prepared bundle with `selected_models=("logreg",)`, `folds=2`, `feature_bundle=build_classical_feature_bundle(prepared)`, and BLAS/OpenMP thread env vars pinned to `1` for repeatability.

Measured impact:
- Baseline `HEAD` direct backtest median:
  - `30.94s`
- Candidate median after the first backtest refactor, with `SPOTIFY_BACKTEST_WORKERS=1`:
  - `29.17s`
  - Improvement vs baseline: `1.06x` faster, `5.7%` lower runtime
- Candidate median after enabling the new default auto-worker heuristic:
  - `26.79s`
  - Improvement vs baseline: `1.16x` faster, `13.4%` lower runtime
  - Improvement vs the single-worker candidate: `1.09x` faster, `8.2%` lower runtime

Validation gate:
- `ruff` passed on `spotify/backtesting.py`, `spotify/robustness.py`, `tests/test_drift_and_backtesting.py`, and `tests/test_research_platform.py`.
- `tests/test_drift_and_backtesting.py` and `tests/test_research_platform.py` passed in the main workspace.

Interpretation:
- The first backtest cleanup removed measurable waste, but the larger win came from auto-parallelizing the independent classical folds.
- This should lower the instrumented `temporal_backtest` phase from roughly `5.83s` to about `5.04s` on the previously measured bounded full run, based on the direct benchmark ratio.
- `tensorflow_runtime_init` and `retrieval_stack` remain the largest likely end-to-end bottlenecks after this pass.

### Rejected Robustness Slice Refactor

Scope:
- Prototyped a `spotify/robustness.py` refactor that precomputed split bucket maps and reused top-k hit masks across models.

Measured impact:
- Direct replay of `run_robustness_slice_evaluation(...)` on the real cached run artifacts was slightly worse:
  - Baseline median: `1.106s`
  - Prototype median: `1.133s`
  - Result: about `2.4%` slower

Outcome:
- Reverted that prototype instead of keeping a cleanup that did not pay for itself.
