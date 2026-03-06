# Spotify Pipeline

Clean baseline for training Spotify next-artist models from exported streaming history.

## Layout

- `spotify/`: pipeline source code
- `tests/`: smoke and config tests
- `data/raw/`: place `Streaming_History_*.json` files
- `outputs/`: generated checkpoints, plots, logs, JSON, and SQLite DB
- `Makefile`: standard local workflow

## Run Profiles

Profiles are selected with `--profile` (or `make train PROFILE=...`).

- `dev`: fastest smoke run
: `epochs=2`, `batch=256`, `max_artists=40`, `models=dense,lstm`, `no_video`, `no_spotify_features`, `no_shap`
- `small`: medium experiment
: `epochs=10`, `batch=512`, `max_artists=120`, `models=dense,gru,lstm,transformer`, `no_spotify_features`, `no_shap`
- `full`: full training preset
: `epochs=50`, `batch=1024`, `max_artists=200`, all models enabled, video+spotipy+shap enabled

CLI flags override profile values.

## Makefile Workflow

```bash
make setup
make train PROFILE=dev
make train PROFILE=small EXTRA_ARGS="--epochs 3 --models dense,lstm"
make test
make clean
```

## Direct CLI

```bash
python -m spotify --profile dev
python -m spotify --profile small --epochs 3
python -m spotify --profile full --data-dir /path/to/raw --output-dir /path/to/outputs
```

## Defaults

- Data dir: `data/raw`
- Output dir: `outputs`
- DB: `outputs/spotify_training.db`
- Scaler: `outputs/context_scaler.joblib`
- Log: `outputs/train.log`

## Notes

- For Spotify audio feature enrichment, set `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET`.
- If unavailable, audio features fall back to zeros.
