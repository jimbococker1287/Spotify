# Spotify Experiment Lab

This project is an end-to-end experiment system for Spotify extended streaming history with:

- Deep model training
- Classical model benchmarking
- Optuna hyperparameter tuning
- Temporal backtesting
- MLflow tracking
- Prepared-data fingerprint caching
- Ranking metrics (`NDCG@5`, `MRR@5`, coverage, diversity)
- Champion/challenger gating
- Per-run Markdown auto-report
- Persistent cross-run history and charts
- Benchmark lock runs with seed-based confidence intervals
- Prediction CLI for serving top-k next-artist recommendations

## Project Layout

- `spotify/`: pipeline source code
- `tests/`: smoke/config tests
- `data/raw/`: Spotify `Streaming_History_*.json`
- `outputs/runs/<run_id>/`: per-run artifacts, logs, charts, model files
- `outputs/history/experiment_history.csv`: cumulative model leaderboard history
- `outputs/history/optuna_history.csv`: cumulative tuned-model history
- `outputs/history/backtest_history.csv`: cumulative temporal backtest history
- `outputs/history/benchmark_history.csv`: benchmark-lock aggregate history with CI stats
- `outputs/mlruns/mlflow.db`: local MLflow tracking DB (default)

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

Or via Makefile:

```bash
make train-everything RUN_NAME=full-e2e
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
- `OPTUNA_TRIALS` (default `30`)
- `OPTUNA_TIMEOUT_SECONDS` (default `1800`)
- `BACKTEST_FOLDS` (default `5`)
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
- `SPOTIFY_OPTUNA_JOBS` (parallel Optuna trial workers; launcher auto-sets, capped for memory)
- `SPOTIFY_TF_DATA_CACHE` (default `auto`; only caches batches when memory headroom is sufficient)
- `SPOTIFY_TF_PREFETCH` (default `auto`; TensorFlow prefetch buffer)
- `SPOTIFY_DISTRIBUTION_STRATEGY` (`auto`, `mirrored`, `default`)
- `SPOTIFY_ISOLATE_MPL_CACHE` (default `0`; shared matplotlib cache for faster startup)
- `SPOTIFY_CACHE_PREPARED` (default `1`; reuses preprocessed arrays when raw files/config fingerprint is unchanged)
- `SPOTIFY_OPTUNA_PRUNER` (default `median`; use `none` to disable pruning)
- `SPOTIFY_OPTUNA_PRUNING_FIDELITIES` (default `0.25,0.60,1.0`)
- `SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS` (default `120`; per-trial budget)
- `SPOTIFY_OPTUNA_MODEL_TIMEOUT_SECONDS` (optional; override per-model tuning timeout)
- `SPOTIFY_OPTUNA_MODEL_TIMEOUTS` (optional, e.g. `logreg=90,random_forest=300`)
- `SPOTIFY_CHAMPION_GATE_MAX_REGRESSION` (default `0.005`; max allowed drop in gate metric vs previous champion)
- `SPOTIFY_CHAMPION_GATE_METRIC` (default `backtest_top1`; alternatives: `val_top1`)
- `SPOTIFY_CHAMPION_GATE_MATCH_PROFILE` (default `1`; compare challengers only against prior runs of the same profile)
- `SPOTIFY_CHAMPION_GATE_STRICT` (default `0`; set `1` to fail run when gate fails)

The launcher still includes all deep and classical model families, but runs lighter deep models first so progress appears sooner.

For memory-constrained devices, keep the same model/training-set definitions but reduce peak RAM:

```bash
SPOTIFY_TF_DATA_CACHE=off \
SPOTIFY_CLASSICAL_MODEL_WORKERS=1 \
SPOTIFY_BACKTEST_WORKERS=1 \
SPOTIFY_OPTUNA_JOBS=1 \
bash scripts/run_everything.sh
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
- `--optuna` / `--no-optuna`
- `--optuna-trials <n>`
- `--optuna-timeout-seconds <n>`
- `--optuna-models logreg,random_forest,...`
- `--temporal-backtest` / `--no-temporal-backtest`
- `--backtest-folds <n>`
- `--backtest-models logreg,random_forest,...`

## Outputs

Per run (`outputs/runs/<run_id>/`), typical files include:

- `train.log`
- `run_manifest.json`
- `run_results.json`
- `run_report.md` (auto-generated run summary with metrics, speed, and trend links)
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
- `analysis/*_confidence_summary.json` (ECE/Brier/top-1 confidence summary)
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

## Prediction CLI

Load the best deep model from a run and print top-5 next-artist predictions:

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

## Prediction Service (HTTP)

Serve the best deep model from a run via a lightweight HTTP API:

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

## Spotipy Setup

Copy the env template and add your Spotify API credentials:

```bash
cp .env.example .env
```

Set:

- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`

The project now auto-loads `.env` and `.env.local` for:

- `python -m spotify`
- `python -m spotify.predict_next`
- `python -m spotify.predict_service`

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

## Notes

- Set `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET` to enrich with Spotify audio features.
- If Spotipy credentials are unavailable, audio features are zero-filled.
- `orjson` is used automatically (when installed) for faster raw JSON loading.
- Matplotlib cache is shared under `outputs/.mplconfig` by default for faster repeat runs (`SPOTIFY_ISOLATE_MPL_CACHE=1` restores per-run isolation).
- `make clean` keeps historical outputs.
- `make clean-all` removes all outputs.
