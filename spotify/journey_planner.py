from __future__ import annotations

from heapq import nlargest
import os
from pathlib import Path
import csv
import json

import numpy as np

from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact
from .multimodal import MultimodalArtistSpace
from .ranking import topk_indices_1d


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _expand_beam(
    *,
    beam: dict[str, object],
    beam_width: int,
    multimodal_space: MultimodalArtistSpace,
    digital_twin: ListenerDigitalTwinArtifact,
) -> list[dict[str, object]]:
    sequence = np.asarray(beam["sequence"], dtype="int32")
    last_artist = int(sequence[-1])
    transition = np.asarray(digital_twin.transition_matrix[last_artist], dtype="float32")
    candidate_ids = topk_indices_1d(transition, max(int(beam_width) * 2, 8))
    if candidate_ids.size == 0:
        return []

    last_embedding = np.asarray(multimodal_space.embeddings[last_artist], dtype="float32")
    similarities = np.asarray(
        np.asarray(multimodal_space.embeddings[candidate_ids], dtype="float32") @ last_embedding,
        dtype="float64",
    )
    novelty = np.asarray(1.0 - multimodal_space.popularity[candidate_ids], dtype="float64")
    repeat_penalty = np.asarray(np.isin(candidate_ids, sequence), dtype="float64")
    energy_delta = np.abs(np.asarray(multimodal_space.energy[candidate_ids], dtype="float64") - float(multimodal_space.energy[last_artist]))
    candidate_scores = (
        float(beam["score"])
        + np.log(np.clip(np.asarray(transition[candidate_ids], dtype="float64"), 1e-6, 1.0))
        + (0.30 * novelty)
        + (0.20 * similarities)
        - (0.35 * repeat_penalty)
        - (0.10 * energy_delta)
    )

    artists = tuple(int(artist_id) for artist_id in beam.get("artists", ()))
    shifted_prefix = sequence[1:]
    next_beams: list[dict[str, object]] = []
    for pos, candidate_id in enumerate(candidate_ids.tolist()):
        next_sequence = np.empty_like(sequence)
        next_sequence[:-1] = shifted_prefix
        next_sequence[-1] = int(candidate_id)
        next_beams.append(
            {
                "sequence": next_sequence,
                "score": float(candidate_scores[pos]),
                "artists": artists + (int(candidate_id),),
                "expected_skip": float(beam["expected_skip"]) + (float(repeat_penalty[pos]) * 0.1),
                "expected_end": float(beam["expected_end"]) + (float(energy_delta[pos]) * 0.02),
            }
        )
    return next_beams


def build_journey_plans(
    *,
    data: PreparedData,
    artist_labels: list[str],
    multimodal_space: MultimodalArtistSpace,
    digital_twin: ListenerDigitalTwinArtifact,
    output_dir: Path,
    logger,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = max(4, int(os.environ.get("SPOTIFY_MOONSHOT_PLAN_HORIZON", "8")))
    beam_width = max(2, int(os.environ.get("SPOTIFY_MOONSHOT_PLAN_BEAM", "4")))

    seed_sequences = np.asarray(data.X_seq_test, dtype="int32")
    if len(seed_sequences) == 0:
        return []

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for seed_idx in range(min(3, len(seed_sequences))):
        beams = [
            {
                "sequence": seed_sequences[seed_idx].copy(),
                "score": 0.0,
                "artists": (),
                "expected_skip": 0.0,
                "expected_end": 0.0,
            }
        ]
        for _ in range(horizon):
            next_beams: list[dict[str, object]] = []
            for beam in beams:
                next_beams.extend(
                    _expand_beam(
                        beam=beam,
                        beam_width=beam_width,
                        multimodal_space=multimodal_space,
                        digital_twin=digital_twin,
                    )
                )
            beams = nlargest(beam_width, next_beams, key=lambda item: float(item["score"]))

        if not beams:
            continue
        best = beams[0]
        for step, artist_id in enumerate(best["artists"], start=1):
            rows.append(
                {
                    "seed_index": seed_idx,
                    "step": step,
                    "artist_label": int(artist_id),
                    "artist_name": artist_labels[int(artist_id)],
                    "plan_score": float(best["score"]),
                    "expected_skip_penalty": float(best["expected_skip"]),
                    "expected_end_penalty": float(best["expected_end"]),
                }
            )
        summary_rows.append(
            {
                "seed_index": seed_idx,
                "planned_horizon": int(len(best["artists"])),
                "plan_score": float(best["score"]),
                "first_artist": artist_labels[int(best["artists"][0])] if best["artists"] else "",
                "last_artist": artist_labels[int(best["artists"][-1])] if best["artists"] else "",
            }
        )

    if not rows:
        return []

    plan_csv = _write_csv(
        output_dir / "journey_plans.csv",
        rows,
        ["seed_index", "step", "artist_label", "artist_name", "plan_score", "expected_skip_penalty", "expected_end_penalty"],
    )
    summary_path = output_dir / "journey_plans_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    logger.info("Built journey plans for %d seeds with horizon=%d", len(summary_rows), horizon)
    return [plan_csv, summary_path]
