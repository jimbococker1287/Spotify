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
