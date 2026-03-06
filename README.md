# Spotify Experiment Lab

This project is an end-to-end experiment system for Spotify extended streaming history with:

- Deep model training
- Classical model benchmarking
- Optuna hyperparameter tuning
- Temporal backtesting
- MLflow tracking
- Persistent cross-run history and charts

## Project Layout

- `spotify/`: pipeline source code
- `tests/`: smoke/config tests
- `data/raw/`: Spotify `Streaming_History_*.json`
- `outputs/runs/<run_id>/`: per-run artifacts, logs, charts, model files
- `outputs/history/experiment_history.csv`: cumulative model leaderboard history
- `outputs/history/optuna_history.csv`: cumulative tuned-model history
- `outputs/history/backtest_history.csv`: cumulative temporal backtest history
- `outputs/mlruns/mlflow.db`: local MLflow tracking DB (default)

## Profiles

- `dev`: fastest smoke profile (no Optuna, no backtest, no MLflow)
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

Or via Makefile:

```bash
make train-everything RUN_NAME=full-e2e
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
- `SPOTIFY_FORCE_CPU` (default `1` in launcher; set `0` to allow GPU)
- `SPOTIFY_MIXED_PRECISION` (`auto`, `on`, or `off`)
- `SPOTIFY_RUN_EAGER` (default `1` in launcher for macOS stability)
- `SPOTIFY_DISABLE_MONITOR` (`auto` by default; monitor disabled automatically on macOS)
- `TF_NUM_INTRAOP_THREADS` (optional manual override; default uses TensorFlow auto threading)
- `TF_NUM_INTEROP_THREADS` (optional manual override; default uses TensorFlow auto threading)

The launcher still includes all deep and classical model families, but runs lighter deep models first so progress appears sooner.

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
- `run_leaderboard.png`
- `classical_results.json`
- `histories.json` and deep-model learning curves
- `optuna/optuna_results.json`
- `optuna/optuna_trials_<model>.csv`
- `optuna/optuna_history_<model>.png`
- `backtest/temporal_backtest.csv`
- `backtest/temporal_backtest_top1.png`

Global history (`outputs/history/`):

- `experiment_history.csv` + `history_best_runs.png`
- `optuna_history.csv` + `history_optuna_best_runs.png`
- `backtest_history.csv` + `history_backtest_mean_top1.png`

## Notes

- Set `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET` to enrich with Spotify audio features.
- If Spotipy credentials are unavailable, audio features are zero-filled.
- Matplotlib cache is isolated per run under the run folder for macOS sandbox compatibility.
- `make clean` keeps historical outputs.
- `make clean-all` removes all outputs.
