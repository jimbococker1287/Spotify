from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .predict_next import _prepare_inputs, load_prediction_input_context, prediction_signature_fingerprint
from .serving import load_predictor, resolve_model_row
from .taste_os_demo_core import MODE_CONFIGS
from .taste_os_demo_core import SCENARIOS
from .taste_os_demo_core import _percentile_ranks
from .taste_os_demo_core import _surface_reranked_indices
from .taste_os_demo_core import _target_alignment
from .taste_os_demo_core import build_taste_os_demo_payload
from .taste_os_demo_core import write_taste_os_demo_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.taste_os_demo",
        description="Run the Personal Taste OS demo with adaptive steering and written artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to outputs/runs/<run_id> or outputs/models/champion. Defaults to champion alias.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument(
        "--mode",
        type=str,
        default="focus",
        choices=sorted(MODE_CONFIGS),
        help="Taste OS mode to simulate.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="steady",
        choices=sorted(SCENARIOS),
        help="Adaptive scenario to simulate after the first plan is built.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to surface.")
    parser.add_argument(
        "--recent-artists",
        type=str,
        default=None,
        help="Optional pipe-separated artist names to use as the recent sequence override.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video history files while rebuilding the latest context.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis/taste_os_demo",
        help="Directory to write the JSON and Markdown demo artifacts.",
    )
    parser.add_argument(
        "--stdout-format",
        type=str,
        default="summary",
        choices=("summary", "json"),
        help="Whether to print a short summary or the full JSON payload to stdout.",
    )
    return parser.parse_args()


def _load_artifact(path: Path, *, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} artifact: {path}")
    return joblib.load(path)


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.taste_os_demo")

    run_dir, champion_alias_model_name = resolve_prediction_run_dir(args.run_dir)
    model_row = resolve_model_row(
        run_dir,
        explicit_model_name=args.model_name,
        alias_model_name=champion_alias_model_name,
    )
    prediction_context = load_prediction_input_context(
        run_dir=run_dir,
        data_dir=Path(args.data_dir),
        include_video=bool(args.include_video),
        logger=logger,
    )

    recent_artists = None
    if args.recent_artists:
        recent_artists = [part.strip() for part in args.recent_artists.split("|") if part.strip()]

    seq_batch, ctx_batch, sequence_names = _prepare_inputs(
        run_dir=run_dir,
        data_dir=Path(args.data_dir),
        recent_artists=recent_artists,
        include_video=bool(args.include_video),
        logger=logger,
        context=prediction_context,
    )

    predictor = load_predictor(
        run_dir=run_dir,
        row=model_row,
        artist_labels=list(prediction_context.artist_labels),
    )

    artifact_paths = {
        "multimodal_space": str((run_dir / "analysis" / "multimodal" / "multimodal_artist_space.joblib").resolve()),
        "digital_twin": str((run_dir / "analysis" / "digital_twin" / "listener_digital_twin.joblib").resolve()),
        "safe_policy": str((run_dir / "analysis" / "safe_policy" / "safe_bandit_policy.joblib").resolve()),
    }
    multimodal_space = _load_artifact(Path(artifact_paths["multimodal_space"]), label="multimodal artist space")
    digital_twin = _load_artifact(Path(artifact_paths["digital_twin"]), label="listener digital twin")
    safe_policy = _load_artifact(Path(artifact_paths["safe_policy"]), label="safe policy")
    context_fingerprint = prediction_signature_fingerprint(prediction_context.source_signature)

    payload = build_taste_os_demo_payload(
        predictor=predictor,
        artist_labels=list(prediction_context.artist_labels),
        sequence_labels=np.asarray(seq_batch[0], dtype="int32"),
        sequence_names=sequence_names,
        context_batch=np.asarray(ctx_batch, dtype="float32"),
        context_raw_batch=prediction_context.context_raw,
        context_features=list(prediction_context.context_features or []),
        friction_reference=prediction_context.friction_reference,
        scaler_mean=prediction_context.scaler_mean,
        scaler_scale=prediction_context.scaler_scale,
        digital_twin=digital_twin,
        multimodal_space=multimodal_space,
        safe_policy=safe_policy,
        mode_name=str(args.mode),
        scenario_name=str(args.scenario),
        top_k=max(1, int(args.top_k)),
        artifact_paths=artifact_paths,
        run_dir=run_dir,
        context_fingerprint=context_fingerprint,
    )
    json_path, md_path = write_taste_os_demo_artifacts(
        payload,
        output_dir=Path(args.output_dir),
    )

    if args.stdout_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"taste_os_demo_json={json_path}")
        print(f"taste_os_demo_md={md_path}")
        print(f"mode={payload['mode']['name']}")
        print(f"scenario={payload['adaptive_session']['scenario']}")
        print(f"top_artist={payload['demo_summary']['top_artist']}")
        print(f"adaptive_replans={payload['demo_summary']['adaptive_replans']}")
        print(f"adaptive_safe_route_steps={payload['demo_summary']['adaptive_safe_route_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
