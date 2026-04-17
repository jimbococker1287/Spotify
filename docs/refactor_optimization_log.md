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

### Full Launcher Budgeting From Real Run

Scope:
- Added baseline-driven shortlist helpers in `spotify/pipeline_runtime.py` so the launcher can cap Optuna and temporal-backtest candidate sets using the already-computed untuned classical benchmark results.
- Tightened the default `scripts/run_everything.sh` research footprint:
  - deep default now uses a small core set unless `SPOTIFY_ENABLE_RESEARCH_DEEP_MODELS=1`
  - classical default drops `random_forest` and `hist_gbm`
  - Optuna default narrows from `logreg,extra_trees,mlp` to `logreg,mlp`
  - Optuna default trial/timeout budget drops from `18/1200s` to `10/600s`
  - Optuna/backtest shortlist env defaults are now both `2`

Motivation from the real 2026-04-02 overnight run:
- Source run: `outputs/runs/20260402_181212_everything-20260402-181212`
- Total measured runtime: `4075.04s` (`67m 55s`)
- Largest phases:
  - `optuna_tuning`: `1926.40s`
  - `deep_model_training`: `1458.08s`
  - `classical_benchmarks`: `534.14s`
  - `temporal_backtest`: `89.72s`
- Optuna wall-clock breakdown from `train.log`:
  - `logreg`: about `340s`
  - `extra_trees`: about `1405s`
  - `mlp`: about `181s`
- Deep-model fit breakdown from `run_results.json`:
  - all `14` deep models combined: `1173.72s` fit time
  - retained default core (`dense,gru,transformer`): `121.93s` fit time

Projected impact:
- Removing `extra_trees` from the default Optuna sweep and dropping to `10` trials should save about `27` wall-clock minutes on runs similar to the measured overnight run.
- Dropping `random_forest` and `hist_gbm` from the default classical sweep should save about `7.5` minutes on that same run shape.
- Restricting the default deep sweep to `dense,gru,transformer` should save roughly `17` to `22` minutes, depending on how much non-fit overhead remains fixed.
- Total projected improvement for the default overnight launcher: roughly `45` to `55` minutes saved on runs shaped like `20260402_181212`, with the largest share coming from Optuna and deep-model budgeting.

Validation gate:
- `ruff` passed on `spotify/pipeline_runtime.py`, `tests/test_pipeline_runtime_shortlists.py`, and `tests/test_config_smoke.py`.
- `compileall` passed for the touched Python runtime/test files.
- `tests/test_pipeline_runtime_shortlists.py` and `tests/test_config_smoke.py` passed.
- `bash -n scripts/run_everything.sh` passed.

### Persistent Optuna Cache Reuse

Scope:
- Added a persistent Optuna cache in `spotify/tuning.py` keyed by the prepared-data fingerprint plus the effective tuning budget for each model.
- Wired the pipeline to pass the prepared-data fingerprint and cache root into `run_optuna_tuning(...)` from `spotify/pipeline_runtime.py`.
- Added per-run phase metadata for cache hits and misses, and made the launcher explicitly opt into the new cache with `SPOTIFY_CACHE_OPTUNA=1`.

What the cache reuses:
- tuned estimator artifact
- tuned prediction bundle
- per-model Optuna trial log
- per-model Optuna history plot
- tuned metrics / params summary

Behavioral goal:
- if the prepared-data fingerprint and tuning settings have not changed, the next run should skip the Optuna search entirely for cached models and simply hydrate the current run directory from cache
- this reuse happens before Optuna import and before classical feature-bundle construction, so the hot path avoids both tuning and most setup overhead

Measured impact:
- Direct live benchmark on a real `run_optuna_tuning(...)` call with `selected_models=("logreg",)` and `trials=4`:
  - cold run (populate cache): `0.6904s`
  - warm run (cache hit): `0.0011s`
  - speedup: about `610.5x`
- That micro-benchmark is intentionally tiny, but it confirms the hot path is effectively just artifact copy + JSON load.
- Tiny end-to-end pipeline smoke on the real local dataset with `--profile dev --classical-only --classical-models logreg --optuna --optuna-models logreg --optuna-trials 2`:
  - first run `optuna_tuning` phase: `8.7018s`
  - second identical run `optuna_tuning` phase: `0.0030s`
  - effective phase speedup: about `2900x`
  - recorded cache stats on the second run: `cache_hit_count=1`, `cache_miss_count=0`

Projected impact on the real overnight run:
- Source run: `outputs/runs/20260402_181212_everything-20260402-181212`
- Real measured `optuna_tuning` phase there: `1926.40s` (`32m 06s`)
- On stable input data, a full Optuna cache hit would remove almost all of that phase, so the run would project from about `67m 55s` down to about `35m 49s` before accounting for the other launcher reductions already made.
- In practice the savings will be slightly less than the raw phase time because of artifact hydration overhead, but that overhead is tiny relative to the original tuning cost.

Validation gate:
- `ruff` passed on `spotify/tuning.py`, `spotify/pipeline_runtime.py`, `tests/test_tuning_cache.py`, `tests/test_tuning_helpers.py`, and `tests/test_pipeline_runtime_shortlists.py`.
- `compileall` passed on the touched runtime/test files.
- `tests/test_tuning_cache.py`, `tests/test_tuning_helpers.py`, `tests/test_pipeline_runtime_shortlists.py`, and `tests/test_config_smoke.py` passed.
- `bash -n scripts/run_everything.sh` passed.

### Persistent Deep Training + SHAP Cache Reuse

Scope:
- Added a persistent per-model deep-training cache in `spotify/training.py` keyed by:
  - prepared-data fingerprint
  - model name
  - random seed, batch size, epochs
  - sequence length, artist count, context width
  - selected runtime knobs such as eager/mixed-precision/distribution mode
  - a source digest from `training.py` + `modeling.py`
- Wired the pipeline to pass the prepared-data fingerprint into deep training from `spotify/pipeline_runtime.py` and record deep cache hit/miss metadata in the `deep_model_training` phase.
- Added persistent SHAP cache reuse in `spotify/explainability.py`, keyed by:
  - prepared-data fingerprint
  - selected best-model name
  - best-model checkpoint digest
  - explainer source digest
  - background/explain sample geometry
- Updated the launchers so `SPOTIFY_CACHE_DEEP=1` and `SPOTIFY_CACHE_SHAP=1` are on by default.

What gets reused:
- `best_<model>.keras` checkpoints
- deep prediction bundles in `prediction_bundles/deep_<model>.npz`
- deep history/metric summaries used by reporting and evaluation
- `shap_values.pkl` for the best deep model

Validation and cache behavior:
- Added `tests/test_training_cache.py` to prove deep cache hits can replay artifacts without importing TensorFlow.
- Added an explainability cache regression in `tests/test_explainability.py` to prove SHAP cache hits return before importing `shap` or TensorFlow.

Measured impact:
- Real artifact-size deep-cache replay benchmark using the full 14-model artifact set from `outputs/runs/20260402_181212_everything-20260402-181212`:
  - warm replay time for all 14 deep models: `0.4197s`
- Real artifact-size SHAP-cache replay benchmark using the same run’s `shap_values.pkl`:
  - warm replay time: `0.0048s`

Impact grounded in the actual 2026-04-02 overnight run:
- Actual `deep_model_training` phase: `1458.081s` (`24m 18s`)
- Sum of deep-model fit times from `run_results.json`: `1173.716s`
- Non-fit deep-phase overhead: `284.365s`
- SHAP alone inside that phase, from `train.log`: about `186.385s`
- Residual non-fit/non-SHAP overhead: about `97.980s`

Projected identical-rerun deep-phase runtime:
- residual plotting/sqlite/reporting overhead: `97.980s`
- deep artifact replay: `0.4197s`
- SHAP artifact replay: `0.0048s`
- projected deep phase total on an identical rerun: about `98.405s` (`1m 38s`)

Projected savings versus the measured overnight run:
- deep phase improvement: about `14.8x` faster
- deep phase reduction: about `93.3%`
- deep phase wall-clock saved: about `22m 40s`

Interpretation:
- The big win is not making TensorFlow train 5% faster; it is avoiding repeated fitting and repeated SHAP work entirely when the prepared fingerprint and deep configuration have not changed.
- After this pass, the remaining deep-stage cost on identical reruns is mostly plot regeneration, SQLite persistence, and other reporting work, not model fitting.

Validation gate:
- `compileall` passed on `spotify/training.py`, `spotify/explainability.py`, `spotify/pipeline_runtime.py`, `tests/test_training_cache.py`, and `tests/test_explainability.py`.
- `ruff` passed on the touched runtime and test files.
- `tests/test_training_cache.py`, `tests/test_training_helpers.py`, and `tests/test_explainability.py` passed.
- `bash -n scripts/run_everything.sh` and `bash -n scripts/run_fast.sh` passed.

### Persistent Classical Benchmark Cache Reuse

Scope:
- Added a persistent per-model classical benchmark cache in `spotify/benchmarks.py` keyed by:
  - prepared-data fingerprint
  - model name
  - random seed
  - train/eval sampling caps
  - sequence length, artist count, context width
  - a source digest from `benchmarks.py`
- Wired the runtime to pass the prepared-data fingerprint into `run_classical_benchmarks(...)` from `spotify/pipeline_runtime_runner.py` and record classical cache hit/miss metadata in the `classical_benchmarks` phase.
- Enabled the cache by default in `scripts/run_everything.sh` and `scripts/run_fast.sh` with `SPOTIFY_CACHE_CLASSICAL=1`.
- Added regression coverage for the classical cache replay path and for the already-added deep reporting cache round trip.

What gets reused:
- classical estimator artifacts in `estimators/classical_<model>.joblib`
- classical prediction bundles in `prediction_bundles/classical_<model>.npz`
- per-model benchmark metrics
- the assembled `classical_results.json` manifest for the current run

Validation and cache behavior:
- Added `tests/test_benchmarks_features.py` coverage proving a full classical cache hit replays results without rebuilding classical features or refitting estimators.
- Added `tests/test_reporting_and_evaluation.py` coverage proving deep reporting artifacts can be saved to cache and restored into a fresh run directory.

Measured impact:
- Real prepared-data benchmark using `outputs/cache/prepared_data/07eb728b1f7b45a58b263785/prepared_bundle.joblib`
- Model sweep: `logreg,extra_trees,knn,gaussian_nb,mlp`
- Sampling caps: `max_train_samples=30000`, `max_eval_samples=12000`
- Cold classical sweep: `71.221s`
- Warm cache replay of the same sweep: `0.305s`
- Speedup on identical reruns: about `233.3x` faster

Projected pipeline impact:
- On repeated nightlies where the prepared fingerprint is unchanged, the classical benchmark phase should now collapse from “fit every estimator again” to “copy cached artifacts into the new run dir”.
- That does not make the whole pipeline `233x` faster, but it removes another minute-scale repeated-work block after the Optuna and deep-cache passes.

Validation gate:
- `compileall` passed on `spotify/benchmarks.py`, `spotify/pipeline_runtime_runner.py`, `spotify/reporting.py`, `tests/test_benchmarks_features.py`, and `tests/test_reporting_and_evaluation.py`.
- `ruff` passed on the touched Python files.
- `bash -n scripts/run_everything.sh scripts/run_fast.sh` passed.
- `tests/test_benchmarks_features.py`, `tests/test_reporting_and_evaluation.py`, `tests/test_training_cache.py`, `tests/test_explainability.py`, `tests/test_pipeline_runtime_shortlists.py`, and `tests/test_tuning_cache.py` passed.

### Persistent Temporal Backtest Cache Reuse

Scope:
- Added a persistent phase-level temporal backtest cache in `spotify/backtesting.py` keyed by:
  - prepared-data fingerprint
  - selected backtest models
  - folds, train/eval sample caps, adaptation mode
  - tuned challenger specs
  - sequence length, artist count, context width, total row count
  - deep/retrieval backtest runtime knobs that affect results
  - a source digest across the backtesting stack
- Wired the experiment runtime to pass the prepared-data fingerprint into `run_temporal_backtest(...)` from `spotify/pipeline_runtime_experiments.py` and record cache metadata in the `temporal_backtest` phase.
- Enabled the cache by default in `scripts/run_everything.sh` and `scripts/run_fast.sh` with `SPOTIFY_CACHE_BACKTEST=1`.

What gets reused:
- the full backtest row payload
- `temporal_backtest.csv`
- `temporal_backtest.json`
- `temporal_backtest_summary.csv`
- `temporal_backtest_summary.json`
- the generated metric plot such as `temporal_backtest_top1.png`

Validation and cache behavior:
- Added `tests/test_drift_and_backtesting.py` coverage proving a cache hit restores the phase artifacts and returns cached rows without rebuilding the full dataset or resolving deep builders.

Measured impact:
- Real prepared-data benchmark using `outputs/cache/prepared_data/07eb728b1f7b45a58b263785/prepared_bundle.joblib`
- Bounded workload A: `selected_models=(logreg, mlp)`, `folds=1`, `max_train_samples=12000`, `max_eval_samples=6000`
  - cold backtest run: `32.8805s`
  - warm cache replay: `0.0046s`
  - speedup: about `7168.0x`
- Larger representative workload B: `selected_models=(logreg, extra_trees, mlp)`, `folds=2`, `max_train_samples=30000`, `max_eval_samples=12000`
  - cold backtest run: `191.9846s`
  - warm cache replay: `0.0128s`
  - speedup: about `14972.5x`

Interpretation:
- This does not mean the whole pipeline is `7000x` faster.
- It means unchanged-data reruns can now collapse the temporal backtest phase to artifact replay instead of refitting every fold again.
- Pipeline-level savings will be slightly smaller if TensorFlow was already initialized earlier for another phase, but the backtest compute itself is now effectively removed on stable reruns.

Validation gate:
- `compileall` passed on `spotify/backtesting.py`, `spotify/pipeline_runtime_experiments.py`, and `tests/test_drift_and_backtesting.py`.
- `ruff` passed on the touched Python files.
- `bash -n scripts/run_everything.sh scripts/run_fast.sh` passed.
- `tests/test_drift_and_backtesting.py`, `tests/test_research_platform.py`, and `tests/test_recommender_safety_platform.py` passed.

### Retrieval Phase Replay And Lazy TensorFlow Init

Scope:
- Added a persistent phase-level retrieval cache in `spotify/retrieval_runtime.py` keyed by:
  - prepared-data fingerprint
  - random seed and resolved candidate-k
  - sequence length, artist count, context width, and split sizes
  - self-supervised pretraining toggle
  - retrieval, pretraining, ANN, and reranker runtime knobs
  - a source digest across the retrieval stack
- The retrieval cache replays the full retrieval phase into a fresh run directory, including:
  - retrieval prediction bundles
  - retrieval and reranker serving artifacts
  - the reranker estimator artifact
  - the retrieval summary JSON
  - the selected pretraining artifact
- Wired retrieval cache metadata into the `retrieval_stack` phase in `spotify/pipeline_runtime_search_stage.py`.
- Enabled retrieval cache by default in `scripts/run_everything.sh` and `scripts/run_fast.sh` with `SPOTIFY_CACHE_RETRIEVAL=1`.

Lazy TensorFlow init:
- Reordered `spotify/pipeline_runtime_experiments.py` so classical benchmarks, Optuna, and retrieval run before TensorFlow startup.
- Added `inspect_temporal_backtest_cache(...)` in `spotify/backtesting.py` so the runtime can decide whether the selected deep temporal backtest models will be served from cache before it tries to initialize TensorFlow.
- Updated `spotify/pipeline_runtime_tensorflow_stage.py` so TensorFlow initializes only when:
  - there are uncached deep training models, or
  - the selected deep temporal backtest models are uncached
- Stopped passing partial deep-builder lists from deep training into temporal backtesting; the backtest phase now resolves deep builders lazily only on real cache misses.

Validation and cache behavior:
- Added `tests/test_retrieval_and_friction.py` coverage proving a retrieval cache hit restores all phase artifacts and rows without running pretraining, baseline scoring, or reranker training.
- Added `tests/test_pipeline_runtime_experiments.py` coverage proving `run_experiment_stages(...)` skips the `tensorflow_runtime_init` phase entirely when deep training is fully cached and the selected deep temporal backtest is already cached.

Measured impact:
- Real prepared-data benchmark using `/tmp/spotify-cache-bench.6BCLOP/cache/prepared_data/e66160c1b6edbadf099d85ed/prepared_bundle.joblib`
- Retrieval workload: `candidate_k=30`, `enable_self_supervised_pretraining=False`, same prepared fingerprint and random seed
  - cold retrieval run: `10.0572s`
  - warm retrieval cache replay: `0.0039s`
  - speedup: about `2571.9x`

Interpretation:
- This does not mean the whole pipeline is `2571x` faster.
- It means the retrieval phase is now effectively removed on unchanged reruns, just like Optuna, classical benchmarks, deep training, deep reporting, SHAP, and temporal backtest.
- The TensorFlow startup change is a structural warm-path optimization: on cache-hit reruns, the pipeline can now skip the entire `tensorflow_runtime_init` phase instead of paying that startup just to discover there is no uncached deep work left.
- I attempted a tiny live deep-training smoke on April 8, 2026 to attach a wall-clock number to the TensorFlow skip, but the local Python 3.13 CPU-only TensorFlow environment stalled after `Epoch 1/1 started`, so I am not claiming a measured end-to-end warm-run speedup from that smoke.

Validation gate:
- `compileall` passed on `spotify/retrieval_runtime.py`, `spotify/backtesting.py`, `spotify/pipeline_runtime_dependency_types.py`, `spotify/pipeline_runtime_experiment_types.py`, `spotify/pipeline_runtime_dependency_bundle.py`, `spotify/pipeline_runtime_tensorflow_stage.py`, `spotify/pipeline_runtime_deep_training.py`, `spotify/pipeline_runtime_search_stage.py`, `spotify/pipeline_runtime_experiments.py`, `tests/test_retrieval_and_friction.py`, and `tests/test_pipeline_runtime_experiments.py`.
- `ruff` passed on the touched Python files.
- `bash -n scripts/run_everything.sh scripts/run_fast.sh` passed.
- `tests/test_retrieval_and_friction.py`, `tests/test_drift_and_backtesting.py`, `tests/test_pipeline_runtime_experiments.py`, `tests/test_research_platform.py`, and `tests/test_recommender_safety_platform.py` passed.

### Cold-Path Deep Training And Warm-Start Pass

Scope:
- Added deep-model warm-start discovery in `spotify/training.py` so changed-data runs can reuse the best compatible prior checkpoint for the same:
  - model name
  - sequence length
  - artist vocabulary size
  - context width
  - deep-training source digest
- Deep cache writes now persist a dedicated weights artifact (`warm_start_<model>.weights.h5`) alongside the existing `.keras` checkpoint so future changed-data runs can load weights directly instead of requiring a full-model restore.
- Added deep screening in `spotify/training.py`:
  - when a large uncached deep sweep is requested, the pipeline runs a short probe fit on each uncached model
  - ranks models by probe `val_top1`, `val_top5`, then probe time
  - fully trains only the shortlisted top-N models
- Added Optuna warm-start in `spotify/tuning.py`:
  - changed-data runs now enqueue the best compatible prior hyperparameters first
  - the live study budget is reduced using `SPOTIFY_OPTUNA_WARM_START_TRIAL_FRACTION`
- Added Apple Silicon Python 3.13 runtime protection in `spotify/runtime.py` plus launcher auto-routing in `scripts/run_everything.sh` and `scripts/run_fast.sh`:
  - if `.venv-metal/bin/python` is available and the default deep runtime would otherwise run on Python 3.13, the launcher now routes there automatically
  - if a risky Python 3.13 deep runtime still slips through, TensorFlow now fails fast with a clear remediation message instead of stalling deep phases

Default efficiency policy now enabled by the launchers:
- `SPOTIFY_WARM_START_DEEP=1`
- `SPOTIFY_WARM_START_OPTUNA=1`
- `SPOTIFY_DEEP_SCREENING=auto`
- `SPOTIFY_DEEP_SCREENING_TOP_N=3` for `run_everything.sh`
- `SPOTIFY_DEEP_SCREENING_TOP_N=2` for `run_fast.sh`
- `SPOTIFY_OPTUNA_WARM_START_TRIAL_FRACTION=0.60`
- `SPOTIFY_FAIL_FAST_PY313_DEEP=1`

Measured / deterministic impact:
- Optuna changed-data budget reduction is deterministic:
  - requested `10` trials now becomes `6` on warm-started changed-data runs
  - requested `18` trials now becomes `11`
- Deep-screening budget reduction is workload dependent, but on the old 14-model research sweep with `12` target epochs and `1` probe epoch:
  - before: `14 x 12 = 168` full-epoch equivalents
  - after shortlist-to-3: `14 x 1 + 3 x 12 = 50`
  - projected training-budget reduction: about `70.2%`

Interpretation:
- The Optuna reduction is a real budget cut for changed-data runs; it should lower wall-clock time whenever prior studies exist.
- The deep-screening number is a projected epoch-budget reduction, not a claimed end-to-end wall-clock benchmark.
- The launcher/runtime work is primarily a reliability and startup-protection fix: it avoids wasting time in unstable Python 3.13 deep-runtime paths on Apple Silicon.

Validation gate:
- `python3 -m py_compile` passed on the touched Python modules and the new tests.
- `bash -n scripts/run_everything.sh scripts/run_fast.sh` passed.
- `.venv/bin/ruff check` passed on the touched Python files and tests.
- `PYTHONPATH=/Users/akashponugoti/Documents/Documents - Akash’s MacBook Pro/Spotify .venv/bin/pytest -q tests/test_training_cache.py tests/test_tuning_cache.py tests/test_runtime_acceleration.py` passed.

### Analysis Artifact Replay And Safe Postrun Gating

Scope:
- Added a shared run-relative artifact replay layer in `spotify/pipeline_artifact_cache.py`.
- Wired `spotify/pipeline_runtime_analysis_artifacts.py` to cache and replay these run-specific analysis phases when the prepared-data fingerprint and semantic result rows match:
  - `extended_evaluation`
  - `drift_diagnostics`
  - `robustness_slice_evaluation`
  - `policy_simulation`
  - `friction_proxy_analysis`
  - `moonshot_lab`
- The cache keys deliberately normalize result rows to ignore run-specific absolute artifact paths and transient timing fields so stable reruns can actually hit.
- Added safe postrun reuse in `spotify/pipeline_postrun_reporting.py` for the run-independent research summaries under `analysis/`:
  - `ablation_summary`
  - `backtest_significance`
- Kept run-id-bearing postrun artifacts live:
  - `benchmark_protocol`
  - `experiment_registry`
  - `run_report`
  - `control_room_report`
  This avoids restoring stale `run_id` / timestamp content into a fresh run directory.

Measured / deterministic impact:
- The phase-elimination behavior is deterministic on unchanged reruns:
  - second-run analysis replay now executes `0/5` of the expensive enabled analysis builders in the cache test harness
  - second-run postrun replay now executes `0/2` of the cached research-summary writers in the cache test harness
- This pass is intentionally about removing repeated work rather than claiming a synthetic wall-clock speedup number that would understate or overstate the real value. The real savings scale directly with how expensive those diagnostics and summaries are in your actual pipeline runs.

Interpretation:
- Stable reruns no longer need to recompute the heavy diagnostics tail just because they are writing into a new run directory.
- Postrun is now partially fingerprint-gated in a safe way: only the summaries that do not embed fresh run identity are replayed.
- The remaining live postrun steps are the ones that either append to shared history or intentionally reflect the current run identity.

Validation gate:
- `python3 -m py_compile` passed on `spotify/pipeline_artifact_cache.py`, `spotify/pipeline_runtime_analysis_artifacts.py`, `spotify/pipeline_postrun_reporting.py`, `spotify/pipeline_postrun_stages.py`, and the new tests.
- `.venv/bin/ruff check` passed on the touched Python files and tests.
- `PYTHONPATH=/Users/akashponugoti/Documents/Documents - Akash’s MacBook Pro/Spotify .venv/bin/pytest -q tests/test_pipeline_runtime_analysis_artifacts.py tests/test_reporting_and_evaluation.py tests/test_pipeline_runtime_experiments.py tests/test_pipeline_runtime_shortlists.py` passed.
