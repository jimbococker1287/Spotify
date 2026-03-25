# Spotify Personal Taste OS

This repository packages a Personal Taste OS on top of Spotify extended streaming history. The central idea is to turn raw listening logs into session-aware recommendations, explanations, and policy-safe public insights that can eventually power product modes like focus sessions, workout arcs, commute mode, discovery mode, "why this next", and adaptive playlist steering.

The repo is broad, but the parts fit together around four clear surfaces:

- Taste engine: training, benchmarking, retrieval, uncertainty, and champion gating
- Session planner: digital twin, causal friction, journey planning, safe policy routing, and stress tests
- Personal insights CLI: policy-safe explainers, release tracking, playlist diffing, and catalog exploration
- Serving layer: prediction CLI, HTTP service, and promoted-model aliases

The current system already includes:

- Deep model training
- Classical model benchmarking
- Optuna hyperparameter tuning
- Temporal backtesting
- MLflow tracking
- Prepared-data fingerprint caching
- Ranking metrics (`NDCG@5`, `MRR@5`, coverage, diversity)
- Champion/challenger gating
- Recommender safety platform primitives for governance, drift, backtesting, and conformal abstention across arbitrary sequence recommenders
- Drift diagnostics across train/validation/test regimes
- Self-supervised sequence pretraining for artist embeddings
- Two-stage retrieval plus reranking recommenders
- Friction proxy analysis for skip-risk counterfactuals
- ANN candidate retrieval diagnostics with latency/recall reporting
- Robustness slice evaluation across friction, repeat, platform, and session regimes
- Semi-synthetic policy simulation for offline policy comparison
- Benchmark protocol, experiment registry, ablation, and significance artifacts
- Moonshot lab with multimodal artist space, listener digital twin, causal skip decomposition, journey planning, safe policy routing, group Auto-DJ planning, and stress tests
- Pre-train data quality gate with fail-fast checks
- Champion aliasing (`outputs/models/champion/alias.json`) for no-run-id serving
- Per-run Markdown auto-report
- Persistent cross-run history and charts
- Benchmark lock runs with seed-based confidence intervals
- Prediction CLI for serving top-k next-artist recommendations
- Conformal uncertainty diagnostics with abstaining prediction support

## Product Direction

The strongest big-scope bet in this codebase is `Personal Taste OS`: use the digital twin, journey planner, safe policy layer, and public metadata surface to actively shape listening sessions instead of only scoring the next artist. A short product framing and build map lives in `docs/personal_taste_os.md`.

## Recommender Safety Platform

`spotify/recommender_safety.py` now exposes the reusable safety surface behind the Spotify pipeline:

- Generic temporal backtest windows, callback-driven backtest benchmarking, and artifact writers
- Split drift utilities over arbitrary sequence/context/target snapshots plus pluggable segment extractors
- Promotion gating for arbitrary leaderboard metrics and history tables, including selective-risk and abstention caps
- Conformal abstention summary builders that can be reused by any sequence recommender with class-probability outputs

The Spotify-specific modules (`spotify/backtesting.py`, `spotify/drift.py`, `spotify/governance.py`, and the conformal path in `spotify/evaluation.py`) sit on top of that shared layer, so the same safety logic can be reused as a B2B-style recommender safety SDK.

## Project Threads

The six new project threads added in the latest scope expansion now clear the same requirement bar:

- a clear thesis
- implementation anchors in code
- verification anchors in tests or integration coverage
- a user-facing surface or an intentional integration-only role
- named artifacts or outputs
- top-level documentation

All six threads meet that bar today. The detailed audit and expansion map lives in `docs/project_threads.md`, and the short version is:

- Personal Taste OS: the umbrella product thesis tying training, planning, explanation, and safe delivery into one system.
- Recommender Safety Platform: the reusable safety SDK layer behind drift, backtesting, governance, and abstention.
- Control Room: the operator-facing summary that turns run artifacts into a portfolio dashboard.
- Creator Label Intelligence: the public-insights graph for adjacency, migration, scenes, and release whitespace.
- Group Auto-DJ: the shared-session planner for household, party, car, and ambient group listening.
- Doctorate-Level Roadmap: the research program that turns the repository into publication-grade work.

## Project Layout

- `spotify/`: pipeline source code
- `tests/`: smoke/config tests
- `data/raw/`: Spotify `Streaming_History_*.json` files, either flat or inside the exported `Spotify Extended Streaming History/` folder
- `outputs/runs/<run_id>/`: per-run artifacts, logs, charts, model files
- `outputs/history/experiment_history.csv`: cumulative model leaderboard history
- `outputs/history/optuna_history.csv`: cumulative tuned-model history
- `outputs/history/backtest_history.csv`: cumulative temporal backtest history
- `outputs/history/benchmark_history.csv`: benchmark-lock aggregate history with CI stats
- `outputs/mlruns/mlflow.db`: local MLflow tracking DB (default)
- `mlruns/`: MLflow artifact store for local runs; storage reporting and pruning now include it alongside `outputs/`

## Storage Controls

MLflow still records params, metrics, and lightweight reports by default, but it now skips mirroring heavyweight local artifacts like model binaries, bundles, and databases unless you opt back in.

- `SPOTIFY_MLFLOW_ARTIFACT_MODE=metadata|all|off`
- `SPOTIFY_MLFLOW_ARTIFACT_MAX_MB=25`
- `PYTHONPATH=. .venv/bin/python scripts/prune_artifacts.py` to clean retained run artifacts plus mirrored MLflow artifacts
- `PYTHONPATH=. .venv/bin/python scripts/storage_report.py --output-dir outputs` to report both `outputs/` and external MLflow artifact roots

## Profiles

- `dev`: fastest smoke profile (no Optuna, no backtest, no MLflow)
- `fast`: daily iteration profile (smaller model set + Optuna + backtest + MLflow)
- `small`: balanced profile (deep + classical + Optuna + backtest + MLflow)
- `full`: broad profile (full model suite + deeper tuning/backtesting)

CLI flags always override profile defaults.

## Quickstart

```bash
make setup
make train PROFILE=dev
make test
make qa
```

`make test` now installs the package in editable mode first, so tests do not require manual `PYTHONPATH=.` exports.

If you activate `.venv`, the package also installs named entry points for the main public-facing surfaces:

```bash
source .venv/bin/activate
spotify-lab --help
spotify-predict --help
spotify-serve --help
spotify-public-insights --help
spotify-compare-public --help
spotify-control-room --help
spotify-taste-os-demo --help
```

The existing `python -m spotify...` module entry points still work too.

## Quality Tooling

```bash
make lint
make typecheck
make qa
```

Optional local hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Training Commands

Run deep + classical with profile defaults:

```bash
make train PROFILE=small
```

Run fast iteration mode:

```bash
bash scripts/run_fast.sh
```

Run full exhaustive mode:

```bash
bash scripts/run_full.sh
```

Run classical only:

```bash
make train-classical PROFILE=small
```

Run deep only:

```bash
make train-deep PROFILE=small
```

Run elite stack:

```bash
make train-elite
```

Run everything (all deep + all classical + full Optuna + full temporal backtest + MLflow):

```bash
bash scripts/run_everything.sh
```

Run everything with more aggressive CPU parallelism for the classical / Optuna / backtest stages:

```bash
bash scripts/run_everything_cpu_boost.sh
```

Run everything with the Apple Silicon Metal GPU environment and the more aggressive CPU profile for classical / Optuna / backtest work:

```bash
bash scripts/run_everything_gpu.sh
```

That launcher is tuned for fastest wall-clock runs on Apple Silicon: it skips SHAP by default, keeps Metal enabled for deep models, and uses slightly safer classical parallelism on 16-18 GB unified-memory machines. To re-enable SHAP on that path, run `SPOTIFY_ENABLE_SHAP=1 bash scripts/run_everything_gpu.sh`.

Temporal backtesting can now include deep model names in `--backtest-models` alongside classical ones. For lightweight rolling deep evaluation, the main knobs are:

- `SPOTIFY_DEEP_BACKTEST_EPOCHS` (default `3`)
- `SPOTIFY_DEEP_BACKTEST_BATCH_SIZE` (default `256`)

Re-run classical + backtest only (useful after metric fixes to refresh temporal artifacts):

```bash
bash scripts/rerun_classical_backtest.sh
```

Run a canonical 3-seed benchmark lock and produce confidence intervals:

```bash
bash scripts/run_benchmark_lock.sh
```

Run the regression guard (hang + artifact checks):

```bash
python scripts/regression_guard.py
```

Inspect the current environment's CPU / GPU acceleration state:

```bash
python scripts/check_acceleration.py
```

Build the dedicated Apple Silicon Metal environment:

```bash
bash scripts/setup_metal_venv.sh
```

Or via Makefile:

```bash
make train-everything RUN_NAME=full-e2e
```

GPU setup / run via Make:

```bash
make setup-metal
make train-everything-gpu RUN_NAME=metal-e2e
```

Benchmark lock via Make:

```bash
make benchmark-lock RUN_NAME=nightly-benchmark
```

Regression guard via Make:

```bash
make regression-guard
```

Custom run:

```bash
python -m spotify \
  --profile full \
  --run-name nightly-01 \
  --models lstm,gru,transformer \
  --classical-models logreg,random_forest,extra_trees \
  --optuna --optuna-trials 20 \
  --temporal-backtest --backtest-folds 5 \
  --mlflow
```

The `scripts/run_everything.sh` launcher also supports environment overrides:

- `EPOCHS` (default `12`)
- `OPTUNA_TRIALS` (default `18`)
- `OPTUNA_TIMEOUT_SECONDS` (default `1200`)
- `BACKTEST_FOLDS` (default `4`)
- `CLASSICAL_MAX_TRAIN_SAMPLES` (default `50000`)
- `CLASSICAL_MAX_EVAL_SAMPLES` (default `25000`)
- `PYTHON_BIN` (override Python executable)
- `SPOTIFY_FORCE_CPU` (default `0` in launcher; set `1` to force CPU-only)
- `SPOTIFY_MIXED_PRECISION` (`auto`, `on`, or `off`)
- `SPOTIFY_RUN_EAGER` (default `0` in launcher for faster graph execution)
- `SPOTIFY_STEPS_PER_EXECUTION` (default `64` in launcher)
- `SPOTIFY_BATCH_LOG_INTERVAL` (default `100`, reduces logging overhead)
- `SPOTIFY_DISABLE_MONITOR` (`auto` by default; monitor disabled automatically on macOS)
- `TF_NUM_INTRAOP_THREADS` (launcher defaults to logical CPU count)
- `TF_NUM_INTEROP_THREADS` (launcher defaults based on CPU count)
- `SPOTIFY_CLASSICAL_MODEL_WORKERS` (parallel classical model workers; launcher auto-sets with RAM-aware cap)
- `SPOTIFY_MAX_CLASSICAL_WORKERS` (`auto` by default; set explicit hard cap)
- `SPOTIFY_BACKTEST_WORKERS` (parallel temporal backtest workers; launcher auto-sets, capped for memory)
- `SPOTIFY_OPTUNA_JOBS` (parallel Optuna trial workers inside a study; launcher auto-sets)
- `SPOTIFY_OPTUNA_MODEL_WORKERS` (parallel Optuna studies across model families; launcher auto-sets to `2` on roomier machines)
- `SPOTIFY_TF_DATA_CACHE` (default `auto`; only caches batches when memory headroom is sufficient)
- `SPOTIFY_TF_DATA_CACHE_FRACTION` (default `0.40`; auto-cache headroom fraction of currently available RAM)
- `SPOTIFY_TF_DATA_THREADPOOL` (optional private `tf.data` threadpool size)
- `SPOTIFY_TF_PREFETCH` (default `auto`; TensorFlow prefetch buffer)
- `SPOTIFY_DISTRIBUTION_STRATEGY` (`auto`, `mirrored`, `default`)
- `SPOTIFY_ISOLATE_MPL_CACHE` (default `0`; shared matplotlib cache for faster startup)
- `SPOTIFY_CACHE_PREPARED` (default `1`; reuses preprocessed arrays when raw files/config fingerprint is unchanged)
- `SPOTIFY_OPTUNA_PRUNER` (default `median`; use `none` to disable pruning)
- `SPOTIFY_OPTUNA_PRUNING_FIDELITIES` (default `0.25,0.60,1.0`)
- `SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS` (default `120`; per-trial budget)
- `SPOTIFY_OPTUNA_MODEL_TIMEOUT_SECONDS` (optional; override per-model tuning timeout)
- `SPOTIFY_OPTUNA_MODEL_TIMEOUTS` (launcher default `logreg=300,random_forest=900,extra_trees=600,hist_gbm=900,knn=180,gaussian_nb=120,mlp=600`)
- `SPOTIFY_CHAMPION_GATE_MAX_REGRESSION` (default `0.005`; max allowed drop in gate metric vs previous champion)
- `SPOTIFY_CHAMPION_GATE_METRIC` (default `backtest_top1`; alternatives: `val_top1`)
- `SPOTIFY_CHAMPION_GATE_MATCH_PROFILE` (default `1`; compare challengers only against prior runs of the same profile)
- `SPOTIFY_CHAMPION_GATE_STRICT` (default `0`; set `1` to fail run when gate fails)
- `SPOTIFY_BACKTEST_ADAPTATION_MODE` (`cold`, `warm`, `continual`; default `cold`)
- `SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK` (optional absolute ceiling for conformal selective risk)
- `SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE` (optional absolute ceiling for abstention rate)
- `SPOTIFY_PRETRAIN_OBJECTIVES` (comma-separated self-supervised objectives; default `cooccurrence,masked_tail,contrastive_session`)
- `SPOTIFY_RETRIEVAL_ANN_BITS` (default `10`; random-projection ANN hash width)
- `SPOTIFY_MOONSHOT_PLAN_HORIZON` (default `8`; journey-planner rollout horizon)
- `SPOTIFY_MOONSHOT_PLAN_BEAM` (default `4`; journey-planner beam width)

The launcher still includes all deep and classical model families, but runs lighter deep models first so progress appears sooner.

For memory-constrained devices, keep the same model/training-set definitions but reduce peak RAM:

```bash
SPOTIFY_TF_DATA_CACHE=off \
SPOTIFY_CLASSICAL_MODEL_WORKERS=1 \
SPOTIFY_BACKTEST_WORKERS=1 \
SPOTIFY_OPTUNA_JOBS=1 \
bash scripts/run_everything.sh
```

For machines where the classical / Optuna / backtest stages are underusing CPU, try:

```bash
bash scripts/run_everything_cpu_boost.sh
```

That launcher raises the classical worker cap, gives Optuna and temporal backtesting more parallelism, and allows a slightly larger `tf.data` cache / threadpool budget.

`bash scripts/run_everything_gpu.sh` now layers the Metal Python 3.11 environment on top of that same CPU-boost profile, skips SHAP by default, and adds a bit more memory headroom for unified-memory Macs, so it is the fastest full-run entrypoint on Apple Silicon when `.venv-metal` is available.

## Acceleration

The default `.venv` on this machine is currently `Python 3.13` with `tensorflow 2.20.0`, and TensorFlow is not seeing any GPU devices. On Apple Silicon, `python scripts/check_acceleration.py` will print the active TensorFlow packages, visible GPU devices, and setup guidance.

Local verification on this repo showed:

- the current Python `3.13` environment could not install `tensorflow-metal`
- a dedicated `Python 3.11` environment with `tensorflow-macos==2.16.2` and `tensorflow-metal==1.2.0` can see `GPU:0` on this Apple Silicon Mac
- stale user-site TensorFlow Metal plugins under `~/Library/Python/<version>/.../tensorflow-plugins` can interfere with isolated virtualenv checks, so the setup script backs that folder up before creating `.venv-metal`

The easiest repo-local path is:

```bash
bash scripts/setup_metal_venv.sh
PYTHONNOUSERSITE=1 .venv-metal/bin/python scripts/check_acceleration.py
bash scripts/run_everything_gpu.sh
```

If you want to build it manually, the working stack from local verification was:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv-metal
PYTHONNOUSERSITE=1 .venv-metal/bin/python -m pip install --upgrade pip setuptools wheel
PYTHONNOUSERSITE=1 .venv-metal/bin/python -m pip install tensorflow-macos==2.16.2 tensorflow-metal==1.2.0
PYTHONNOUSERSITE=1 .venv-metal/bin/python scripts/check_acceleration.py
```

`scripts/run_benchmark_lock.sh` supports:

- `BENCHMARK_SEEDS` (default `11 42 77`)
- `DEEP_MODELS` (default `dense,gru_artist,lstm`)
- `CLASSICAL_MODELS` (default `logreg,random_forest,extra_trees`)
- `EPOCHS` (default `6`)

## Elite Flags

- `--mlflow` / `--no-mlflow`
- `--mlflow-tracking-uri <uri>`
- `--mlflow-experiment <name>`
- `--conformal-alpha <float>`
- `--no-conformal`
- `--optuna` / `--no-optuna`
- `--optuna-trials <n>`
- `--optuna-timeout-seconds <n>`
- `--optuna-models logreg,random_forest,...`
- `--temporal-backtest` / `--no-temporal-backtest`
- `--backtest-folds <n>`
- `--backtest-models logreg,random_forest,...`
- `--retrieval-stack` / `--no-retrieval-stack`
- `--self-supervised-pretrain` / `--no-self-supervised-pretrain`
- `--friction-analysis` / `--no-friction-analysis`
- `--moonshot-lab` / `--no-moonshot-lab`
- `--retrieval-candidates <n>`

## Outputs

Per run (`outputs/runs/<run_id>/`), typical files include:

- `train.log`
- `run_manifest.json`
- `benchmark_protocol.json` + `benchmark_protocol.md`
- `experiment_registry.json`
- `run_results.json`
- `run_report.md` (auto-generated run summary with metrics, speed, and trend links)
- `data_quality_report.json` (schema/null/range gate report, generated before training)
- `champion_gate.json` (promotion decision vs prior champion)
- `feature_metadata.json` (artist label map + context feature schema)
- `run_leaderboard.png`
- `classical_results.json`
- `histories.json` and deep-model learning curves
- `optuna/optuna_results.json`
- `optuna/optuna_trials_<model>.csv`
- `optuna/optuna_history_<model>.png`
- `backtest/temporal_backtest.csv`
- `backtest/temporal_backtest_top1.png`
- `backtest/temporal_backtest_summary.csv` + `backtest/temporal_backtest_summary.json`
- `analysis/*_confidence_summary.json` (ECE/Brier/top-1 confidence summary)
- `analysis/*_conformal_summary.json` (split-conformal threshold, coverage, abstention stats)
- `analysis/data_drift_summary.json` (target drift and largest context/segment shifts)
- `analysis/context_feature_drift.csv` + `analysis/segment_drift.csv`
- `analysis/context_feature_drift.png` + `analysis/segment_drift.png`
- `analysis/friction_proxy_summary.json` (proxy counterfactual gap between observed and healthier technical context)
- `analysis/friction_feature_coefficients.csv` + `analysis/friction_counterfactual_delta.csv`
- `analysis/robustness_slices.csv` + `analysis/robustness_summary.json`
- `analysis/policy_simulation_summary.csv` + `analysis/policy_simulation_summary.json`
- `analysis/ablation_summary.csv` + `analysis/backtest_significance.csv`
- `analysis/moonshot_summary.json`
- `analysis/multimodal/*` (artist-space summary, neighbors, and embedding artifact)
- `analysis/causal/*` (causal skip decomposition summary and per-row uplift estimates)
- `analysis/digital_twin/*` (listener digital twin artifact and end-of-session diagnostics)
- `analysis/journey_planner/*` (multi-step listening journey plans)
- `analysis/safe_policy/*` (safe bandit policy candidates and selected routing map)
- `analysis/group_auto_dj/*` (shared-space cohort plans for household, party, car, and ambient listening)
- `analysis/stress_test/*` (policy stress scenarios across friction and drift regimes)
- `retrieval/retrieval_summary.json` + retrieval model artifacts
- `analysis/*_reliability.png` (calibration curves)
- `analysis/*_segment_metrics.csv` (segment-level top-1 and confidence)
- `analysis/*_top_errors.csv` (most frequent true/prediction confusions)

Global history (`outputs/history/`):

- `experiment_history.csv` + `history_best_runs.png`
- `optuna_history.csv` + `history_optuna_best_runs.png`
- `backtest_history.csv` + `history_backtest_mean_top1.png`
- `benchmark_lock_<id>_rows.csv` + `benchmark_lock_<id>_summary.csv`
- `benchmark_lock_<id>_ci95.png`
- `benchmark_history.csv`

Prepared-data cache:

- `outputs/cache/prepared_data/<fingerprint>/prepared_bundle.joblib`
- `outputs/cache/prepared_data/<fingerprint>/cache_meta.json`

Champion alias:

- `outputs/models/champion/alias.json` (latest promoted run pointer + default serving model)

Control room:

- `outputs/analytics/control_room.json`
- `outputs/analytics/control_room.md`

Generate the control-room summary:

```bash
make control-room
```

## Taste OS Demo

Run the first unified `Personal Taste OS` demo contract against the current champion run:

```bash
python -m spotify.taste_os_demo --mode focus --top-k 5
```

Or via the console script:

```bash
spotify-taste-os-demo --mode discovery --top-k 5
```

The contract and acceptance checklist for this demo live in `docs/taste_os_demo_contract.md`.

## Prediction CLI

Load the best deep model from the latest promoted champion run and print top-5 predictions:

```bash
python -m spotify.predict_next --top-k 5
```

Or target a specific run directory:

```bash
python -m spotify.predict_next \
  --run-dir outputs/runs/<run_id> \
  --top-k 5
```

Use a specific model checkpoint:

```bash
python -m spotify.predict_next \
  --run-dir outputs/runs/<run_id> \
  --model-name gru_artist \
  --top-k 5
```

Override sequence input with your own recent artists:

```bash
python -m spotify.predict_next \
  --run-dir outputs/runs/<run_id> \
  --recent-artists "Artist A|Artist B|Artist C|..."
```

Optionally enrich printed predictions with Spotify public artist metadata:

```bash
SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=... \
python -m spotify.predict_next \
  --top-k 5 \
  --spotify-public-metadata \
  --spotify-market US
```

## Prediction Service (HTTP)

Serve the latest promoted champion run via a lightweight HTTP API:

```bash
python -m spotify.predict_service \
  --host 127.0.0.1 \
  --port 8000
```

Or pin a specific run:

```bash
python -m spotify.predict_service \
  --run-dir outputs/runs/<run_id> \
  --host 127.0.0.1 \
  --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"top_k":5,"recent_artists":["Artist A","Artist B","Artist C"]}'
```

Optional token auth + request limits:

- Set `SPOTIFY_PREDICT_AUTH_TOKEN` (or pass `--auth-token`) to require `Authorization: Bearer <token>` or `X-API-Key`.
- Set `--max-top-k` (or `SPOTIFY_PREDICT_MAX_TOP_K`) to cap request `top_k`.
- Errors return structured payloads: `{"error":{"code","message","details"}}`.
- When conformal diagnostics exist for the served model, requests can set `allow_abstain` and `return_prediction_set` to get risk-aware responses.

## Public Comparison

Spotify's developer policy does not allow building derived listenership metrics or benchmarking from Spotify public content, so the "compare my habits to the public" workflow uses Last.fm public charts instead.

Compare your recent listening to public artist charts:

```bash
LASTFM_API_KEY=... \
python -m spotify.compare_public \
  --scope country \
  --country "United States" \
  --lookback-days 180 \
  --top-n 50
```

Or use the Make target:

```bash
make compare-public EXTRA_ARGS='--scope global --lookback-days 90 --top-n 50'
```

Outputs are written under `outputs/analysis/public_compare/` as both JSON and Markdown summaries. The report highlights:

- shared artists between your recent top artists and the public chart
- how much of your recent listening is concentrated on public-top artists
- your most distinctive recent artists
- public-chart artists that are new to you

## Spotify Public Insights

You can also use Spotify's public Web API for policy-safe metadata workflows that do not benchmark users or train models on Spotify content.

All commands below require:

```bash
export SPOTIFY_CLIENT_ID=...
export SPOTIFY_CLIENT_SECRET=...
```

Artist explainer for your recent top artists:

```bash
python -m spotify.public_insights explain-artists --top-n 5 --lookback-days 180
```

Release tracker for favorite artists:

```bash
python -m spotify.public_insights release-tracker --top-n 10 --since-days 120
```

Market availability check for your recent top tracks:

```bash
python -m spotify.public_insights market-check --markets US,GB,IN --top-n 20
```

Discography timeline:

```bash
python -m spotify.public_insights discography --top-n 5 --album-limit 20
```

Public playlist viewer:

```bash
python -m spotify.public_insights playlist-view --playlist https://open.spotify.com/playlist/<id>
```

Discovery console:

```bash
python -m spotify.public_insights discovery-search --query "genre:indie year:2024-2026 tag:hipster" --types artist,album,track
```

Catalog link-outs:

```bash
python -m spotify.public_insights catalog-linkouts --top-artists 10 --top-tracks 20
```

Artist graph:

```bash
python -m spotify.public_insights artist-graph --top-n 5 --related-limit 10
```

Creator / label intelligence graph:

```bash
python -m spotify.public_insights creator-label-intelligence --top-n 8 --lookback-days 365 --neighbor-k 5
```

Release inbox:

```bash
python -m spotify.public_insights release-inbox --top-n 10 --since-days 120
```

Playlist diff tracker:

```bash
python -m spotify.public_insights playlist-diff --playlist https://open.spotify.com/playlist/<id>
```

Market gap finder:

```bash
python -m spotify.public_insights market-gap --top-n 20
```

Playlist archive:

```bash
python -m spotify.public_insights playlist-archive --playlist https://open.spotify.com/playlist/<id>
```

ISRC / UPC / EAN crosswalk:

```bash
python -m spotify.public_insights catalog-crosswalk --top-n 20
```

Podcast / audiobook explorer:

```bash
python -m spotify.public_insights media-explorer --media-type show --query "indie music"
```

Cross-media taste graph:

```bash
python -m spotify.public_insights cross-media-taste-graph --lookback-days 180 --bridge-limit 6
```

Or use the Make target:

```bash
make public-insights EXTRA_ARGS='explain-artists --top-n 5'
```

Artifacts are written under `outputs/analysis/public_spotify/`.

## Roadmaps

See `docs/personal_taste_os.md` for the product thesis, `docs/taste_os_demo_contract.md` for the Week-1 demo contract, `docs/project_threads.md` for the six-thread audit and expansion map, `docs/90_day_roadmap.md` for a concrete 90-day execution plan, and `docs/doctorate_roadmap.md` for the dissertation-scale research plan.

## Docker (Prediction Service)

Build:

```bash
docker build -t spotify-predict-service .
```

Run (mount project outputs and raw data):

```bash
docker run --rm -p 8000:8000 \
  -e RUN_DIR=/app/outputs/models/champion \
  -e DATA_DIR=/app/data/raw \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/data/raw:/app/data/raw:ro" \
  spotify-predict-service
```

The container includes a `/health` Docker healthcheck.

## CI

GitHub Actions CI is defined in `.github/workflows/ci.yml` and runs on push/PR:

- `ruff`
- `mypy`
- `pytest`

## Scheduling + Alerts

Run scheduled-style jobs (fast/full) and alert on champion-gate regression:

```bash
bash scripts/run_scheduled.sh fast
```

Check latest run gate manually:

```bash
python scripts/regression_alert.py
```

Optional webhook alerts:

```bash
SPOTIFY_ALERT_WEBHOOK_URL="https://example.com/webhook" python scripts/regression_alert.py
```

## Spotify API Credentials

The training pipeline no longer fetches Spotify audio features. Spotify's `audio-features` Web API endpoint is deprecated, and Spotify's current developer policy does not allow Spotify content to train ML/AI models.

Existing `.env` and `.env.local` loading remains in place for compatibility, but training now zero-fills `danceability`, `energy`, and `tempo` instead of calling Spotify.

You can still use `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` for opt-in prediction-time enrichment via `python -m spotify.predict_next --spotify-public-metadata`. That lookup only fetches public artist metadata for display and does not feed Spotify content into training.

## Feature Upgrades

The training context now includes additional recency/frequency and session-transition features:

- `artist_play_count_24h`, `artist_play_count_7d`
- `artist_freq_smooth` (Laplace-smoothed per-artist frequency)
- `plays_since_last_artist`
- `artist_session_play_count`
- `session_elapsed_seconds`
- `session_skip_rate_so_far`
- `hours_since_last_artist`
- `session_unique_artists_so_far`
- `is_artist_repeat_from_prev`
- `transition_repeat_count`
- `artist_skip_rate_hist`
- `artist_skip_rate_smooth`

If `data/raw/Spotify Technical Log Information/` is present, preprocessing also joins recent device/network/playback-health context from the export, including connection churn, playback errors, stutters, track-not-played events, cloud-playback stalls, and the latest known bitrate / downgrade settings for the current device family.

## Notes

- Spotify audio feature columns are zero-filled during training.
- `orjson` is used automatically (when installed) for faster raw JSON loading.
- Matplotlib cache is shared under `outputs/.mplconfig` by default for faster repeat runs (`SPOTIFY_ISOLATE_MPL_CACHE=1` restores per-run isolation).
- `make clean` keeps historical outputs.
- `make clean-all` removes all outputs.
