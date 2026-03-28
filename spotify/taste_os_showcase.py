from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .predict_next import _prepare_inputs, load_prediction_input_context
from .serving import load_predictor, resolve_model_row
from .taste_os_demo import _load_artifact, build_taste_os_demo_payload, write_taste_os_demo_artifacts


@dataclass(frozen=True)
class ShowcaseExample:
    label: str
    mode: str
    scenario: str
    story: str


CANONICAL_SHOWCASE_EXAMPLES: tuple[ShowcaseExample, ...] = (
    ShowcaseExample(
        label="Focus / Steady",
        mode="focus",
        scenario="steady",
        story="Show the stable Taste OS baseline: a low-surprise opening and a coherent working-session arc.",
    ),
    ShowcaseExample(
        label="Discovery / Skip Recovery",
        mode="discovery",
        scenario="skip_recovery",
        story="Show that the system can start adventurous, then explain why it pulls closer to taste after a rejection.",
    ),
    ShowcaseExample(
        label="Commute / Friction Spike",
        mode="commute",
        scenario="friction_spike",
        story="Show the safety story: playback friction rises, the planner says why, and the route becomes more conservative.",
    ),
    ShowcaseExample(
        label="Workout / Repeat Request",
        mode="workout",
        scenario="repeat_request",
        story="Show that the planner can respect a comfort signal without collapsing into a one-artist loop.",
    ),
)

STEADY_MODE_COMPARISON_ORDER: tuple[str, ...] = ("focus", "workout", "commute", "discovery")

NARRATIVE_GUARDRAILS: tuple[str, ...] = (
    "Do not add UI or infrastructure work unless it makes the mode, explanation, or adaptive steering story clearer.",
    "Do not add more model families to the demo surface unless they improve the opening choice or the recovery transcript.",
    "Do not add creator, control-room, or research material into the Taste OS share pack unless it directly strengthens the product narrative.",
    "Prefer one clear comparison artifact over many raw run dumps; readability matters more than exhaustiveness here.",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.taste_os_showcase",
        description="Build the Week 3-4 Taste OS showcase pack with canonical adaptive demos and mode comparison artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to outputs/runs/<run_id> or outputs/models/champion. Defaults to champion alias.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to capture in each showcase run.")
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
        default="outputs/analysis/taste_os_demo/showcase",
        help="Directory to write the showcase pack and canonical demo artifacts.",
    )
    parser.add_argument(
        "--stdout-format",
        type=str,
        default="summary",
        choices=("summary", "json"),
        help="Whether to print a short summary or the full JSON payload to stdout.",
    )
    return parser.parse_args()


def _recent_artists_from_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in str(raw).split("|") if part.strip()]
    return values or None


def _top_candidate_summary(payload: dict[str, object]) -> dict[str, object]:
    top_rows = payload.get("top_candidates", []) if isinstance(payload, dict) else []
    first = top_rows[0] if isinstance(top_rows, list) and top_rows else {}
    second = top_rows[1] if isinstance(top_rows, list) and len(top_rows) > 1 else {}
    fallback_policy = payload.get("fallback_policy", {}) if isinstance(payload, dict) else {}
    adaptive_session = payload.get("adaptive_session", {}) if isinstance(payload, dict) else {}
    return {
        "top_artist": str(first.get("artist_name", "")),
        "backup_artist": str(second.get("artist_name", "")),
        "surface_score": float(first.get("surface_score", 0.0)),
        "continuity": float(first.get("continuity", 0.0)),
        "freshness": float(first.get("freshness", 0.0)),
        "transition_support": float(first.get("transition_support", 0.0)),
        "safe_routed": bool(fallback_policy.get("safe_routed", False)),
        "fallback_policy_name": str(fallback_policy.get("active_policy_name", "")),
        "adaptive_replans": int(adaptive_session.get("replan_count", 0)),
        "adaptive_safe_route_steps": int(adaptive_session.get("safe_route_steps", 0)),
    }


def build_mode_comparison_rows(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        mode_block = payload.get("mode", {})
        if not isinstance(mode_block, dict):
            continue
        summary = _top_candidate_summary(payload)
        top_artist = str(summary["top_artist"])
        backup_artist = str(summary["backup_artist"])
        rows.append(
            {
                "mode": str(mode_block.get("name", "")),
                "description": str(mode_block.get("description", "")),
                "top_artist": top_artist,
                "backup_artist": backup_artist,
                "fallback_policy_name": str(summary["fallback_policy_name"]),
                "safe_routed": bool(summary["safe_routed"]),
                "opening_surface_score": round(float(summary["surface_score"]), 4),
                "opening_continuity": round(float(summary["continuity"]), 4),
                "opening_freshness": round(float(summary["freshness"]), 4),
                "opening_transition_support": round(float(summary["transition_support"]), 4),
                "opening_summary": (
                    f"{top_artist} opens the mode with backup {backup_artist or 'n/a'} "
                    f"via {summary['fallback_policy_name']}."
                ),
            }
        )
    preferred_order = {name: idx for idx, name in enumerate(STEADY_MODE_COMPARISON_ORDER)}
    rows.sort(key=lambda row: preferred_order.get(str(row.get("mode", "")), len(preferred_order)))
    return rows


def build_taste_os_showcase_payload(
    *,
    run_dir: Path,
    model_name: str,
    model_type: str,
    canonical_examples: list[dict[str, object]],
    mode_comparison_rows: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    return {
        "showcase_summary": {
            "canonical_example_count": int(len(canonical_examples)),
            "mode_comparison_count": int(len(mode_comparison_rows)),
            "review_goal": "Explain Taste OS in under five minutes with one adaptive share pack and one steady-mode comparison.",
        },
        "run_context": {
            "run_dir": str(run_dir.resolve()),
            "model_name": model_name,
            "model_type": model_type,
            "output_dir": str(output_dir.resolve()),
        },
        "review_order": [example["label"] for example in canonical_examples if isinstance(example, dict)],
        "canonical_examples": canonical_examples,
        "mode_comparison": {
            "scenario": "steady",
            "rows": mode_comparison_rows,
        },
        "narrative_guardrails": list(NARRATIVE_GUARDRAILS),
    }


def write_taste_os_showcase_artifacts(
    payload: dict[str, object],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    showcase_json = output_root / "taste_os_showcase.json"
    showcase_md = output_root / "taste_os_showcase.md"
    comparison_json = output_root / "taste_os_mode_comparison.json"
    comparison_md = output_root / "taste_os_mode_comparison.md"

    canonical_examples = payload.get("canonical_examples", []) if isinstance(payload, dict) else []
    comparison_rows = (
        (payload.get("mode_comparison", {}) if isinstance(payload, dict) else {}).get("rows", [])
        if isinstance(payload, dict)
        else []
    )
    run_context = payload.get("run_context", {}) if isinstance(payload, dict) else {}

    showcase_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    comparison_json.write_text(
        json.dumps(
            {
                "scenario": "steady",
                "rows": comparison_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    showcase_lines = [
        "# Taste OS Showcase",
        "",
        f"- Run dir: `{run_context.get('run_dir', '')}`",
        f"- Model: `{run_context.get('model_name', '')}` [{run_context.get('model_type', '')}]",
        f"- Canonical examples: `{len(canonical_examples)}`",
        f"- Steady mode comparison rows: `{len(comparison_rows)}`",
        "",
        "## Canonical Share Pack",
        "",
    ]
    for example in canonical_examples if isinstance(canonical_examples, list) else []:
        if not isinstance(example, dict):
            continue
        showcase_lines.extend(
            [
                f"### {example.get('label', '')}",
                "",
                f"- Story: {example.get('story', '')}",
                f"- Top artist: `{example.get('top_artist', '')}`",
                f"- Backup artist: `{example.get('backup_artist', '')}`",
                f"- Fallback policy: `{example.get('fallback_policy_name', '')}`",
                f"- Replans: `{example.get('adaptive_replans', '')}`",
                f"- Safe-route steps: `{example.get('adaptive_safe_route_steps', '')}`",
                f"- Why this run matters: {example.get('story_outcome', '')}",
                f"- Demo JSON: `{example.get('demo_json_path', '')}`",
                f"- Demo Markdown: `{example.get('demo_md_path', '')}`",
                "",
            ]
        )

    showcase_lines.extend(
        [
            "## Story Guardrails",
            "",
        ]
    )
    for item in payload.get("narrative_guardrails", []) if isinstance(payload, dict) else []:
        showcase_lines.append(f"- {item}")

    showcase_md.write_text("\n".join(showcase_lines) + "\n", encoding="utf-8")

    comparison_lines = [
        "# Taste OS Mode Comparison",
        "",
        "## Steady Openings",
        "",
    ]
    for row in comparison_rows if isinstance(comparison_rows, list) else []:
        if not isinstance(row, dict):
            continue
        comparison_lines.append(
            f"- `{row.get('mode', '')}` -> `{row.get('top_artist', '')}` "
            f"(backup `{row.get('backup_artist', '')}`, policy `{row.get('fallback_policy_name', '')}`, "
            f"surface=`{row.get('opening_surface_score', '')}`, freshness=`{row.get('opening_freshness', '')}`)"
        )
        comparison_lines.append(str(row.get("opening_summary", "")))

    comparison_md.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    return {
        "showcase_json": showcase_json,
        "showcase_md": showcase_md,
        "comparison_json": comparison_json,
        "comparison_md": comparison_md,
    }


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.taste_os_showcase")

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

    recent_artists = _recent_artists_from_arg(args.recent_artists)
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

    showcase_root = Path(args.output_dir)
    examples_dir = showcase_root / "examples"

    canonical_examples: list[dict[str, object]] = []
    steady_mode_payloads: list[dict[str, object]] = []
    steady_payload_by_mode: dict[str, dict[str, object]] = {}

    common_kwargs = {
        "predictor": predictor,
        "artist_labels": list(prediction_context.artist_labels),
        "sequence_labels": np.asarray(seq_batch[0], dtype="int32"),
        "sequence_names": sequence_names,
        "context_batch": np.asarray(ctx_batch, dtype="float32"),
        "context_raw_batch": prediction_context.context_raw,
        "context_features": list(prediction_context.context_features or []),
        "friction_reference": prediction_context.friction_reference,
        "scaler_mean": prediction_context.scaler_mean,
        "scaler_scale": prediction_context.scaler_scale,
        "digital_twin": digital_twin,
        "multimodal_space": multimodal_space,
        "safe_policy": safe_policy,
        "top_k": max(1, int(args.top_k)),
        "artifact_paths": artifact_paths,
    }

    for example in CANONICAL_SHOWCASE_EXAMPLES:
        payload = build_taste_os_demo_payload(
            mode_name=example.mode,
            scenario_name=example.scenario,
            **common_kwargs,
        )
        demo_json, demo_md = write_taste_os_demo_artifacts(payload, output_dir=examples_dir)
        summary = _top_candidate_summary(payload)
        canonical_examples.append(
            {
                "label": example.label,
                "mode": example.mode,
                "scenario": example.scenario,
                "story": example.story,
                "story_outcome": (
                    f"{summary['top_artist']} opens the lane while {summary['fallback_policy_name']} "
                    f"handles replans={summary['adaptive_replans']} safe_steps={summary['adaptive_safe_route_steps']}."
                ),
                "top_artist": str(summary["top_artist"]),
                "backup_artist": str(summary["backup_artist"]),
                "fallback_policy_name": str(summary["fallback_policy_name"]),
                "adaptive_replans": int(summary["adaptive_replans"]),
                "adaptive_safe_route_steps": int(summary["adaptive_safe_route_steps"]),
                "demo_json_path": str(demo_json),
                "demo_md_path": str(demo_md),
            }
        )
        if example.scenario == "steady":
            steady_payload_by_mode[example.mode] = payload

    for mode_name in STEADY_MODE_COMPARISON_ORDER:
        payload = steady_payload_by_mode.get(mode_name)
        if payload is None:
            payload = build_taste_os_demo_payload(
                mode_name=mode_name,
                scenario_name="steady",
                **common_kwargs,
            )
        steady_mode_payloads.append(payload)

    mode_comparison_rows = build_mode_comparison_rows(steady_mode_payloads)
    payload = build_taste_os_showcase_payload(
        run_dir=run_dir,
        model_name=str(predictor.model_name),
        model_type=str(predictor.model_type),
        canonical_examples=canonical_examples,
        mode_comparison_rows=mode_comparison_rows,
        output_dir=showcase_root,
    )
    artifact_map = write_taste_os_showcase_artifacts(payload, output_dir=showcase_root)

    if args.stdout_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"taste_os_showcase_json={artifact_map['showcase_json']}")
        print(f"taste_os_showcase_md={artifact_map['showcase_md']}")
        print(f"taste_os_mode_comparison_json={artifact_map['comparison_json']}")
        print(f"taste_os_mode_comparison_md={artifact_map['comparison_md']}")
        print(f"canonical_examples={len(canonical_examples)}")
        print(f"steady_mode_rows={len(mode_comparison_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
