from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from .champion_alias import resolve_prediction_run_dir
from .digital_twin import ListenerDigitalTwinArtifact
from .env import load_local_env
from .multimodal import MultimodalArtistSpace
from .predict_next import (
    PredictionInputContext,
    _prepare_inputs,
    load_prediction_input_context,
    prediction_source_signature,
)
from .safe_policy import SafeBanditPolicyArtifact
from .serving import load_predictor, resolve_model_row
from .taste_os_demo import MODE_CONFIGS, SCENARIOS, _load_artifact, build_taste_os_demo_payload, write_taste_os_demo_artifacts
from .taste_os_http import (
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_MAX_TOP_K,
    RequestValidationError,
    append_jsonl as _append_jsonl,
    artist_key as _artist_key,
    build_taste_os_handler,
    is_authorized_request,
    load_json_document as _load_json_document,
    normalize_taste_os_feedback_payload,
    normalize_taste_os_payload,
    parse_taste_os_args,
    read_jsonl_tail as _read_jsonl_tail,
    utc_now_iso as _utc_now_iso,
)
from .taste_os_page import render_taste_os_page_html

@dataclass(frozen=True)
class _TasteOSContextCacheEntry:
    signature: tuple[tuple[str, int, int], ...]
    context: PredictionInputContext


def _taste_os_page_html(service: "TasteOSService") -> str:
    return render_taste_os_page_html(service)


class TasteOSService:
    def __init__(
        self,
        *,
        run_dir: Path,
        data_dir: Path,
        output_dir: Path,
        model_name: str,
        include_video: bool,
        max_top_k: int,
        auth_token: str | None,
        logger: logging.Logger,
        digital_twin: ListenerDigitalTwinArtifact | None = None,
        multimodal_space: MultimodalArtistSpace | None = None,
        safe_policy: SafeBanditPolicyArtifact | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_video = include_video
        self.max_top_k = max(1, int(max_top_k))
        self.auth_token = auth_token
        self.logger = logger
        self._context_lock = threading.Lock()
        self._plan_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._context_cache: dict[bool, _TasteOSContextCacheEntry] = {}
        self.feedback_memory_path = self.output_dir / "feedback_memory.json"
        self.feedback_event_log_path = self.output_dir / "feedback_events.jsonl"
        self.session_history_path = self.output_dir / "session_history.jsonl"

        metadata = json.loads((run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
        self.artist_labels = list(metadata.get("artist_labels", []))
        model_row = resolve_model_row(
            run_dir,
            explicit_model_name=model_name,
            alias_model_name=model_name,
        )
        self.predictor = load_predictor(
            run_dir=run_dir,
            row=model_row,
            artist_labels=self.artist_labels,
        )
        self.model_name = self.predictor.model_name
        self.model_type = self.predictor.model_type
        self.artifact_paths = {
            "multimodal_space": str((run_dir / "analysis" / "multimodal" / "multimodal_artist_space.joblib").resolve()),
            "digital_twin": str((run_dir / "analysis" / "digital_twin" / "listener_digital_twin.joblib").resolve()),
            "safe_policy": str((run_dir / "analysis" / "safe_policy" / "safe_bandit_policy.joblib").resolve()),
        }
        self.multimodal_space = (
            multimodal_space
            if multimodal_space is not None
            else _load_artifact(Path(self.artifact_paths["multimodal_space"]), label="multimodal artist space")
        )
        self.digital_twin = (
            digital_twin
            if digital_twin is not None
            else _load_artifact(Path(self.artifact_paths["digital_twin"]), label="listener digital twin")
        )
        self.safe_policy = (
            safe_policy
            if safe_policy is not None
            else _load_artifact(Path(self.artifact_paths["safe_policy"]), label="safe policy")
        )

        try:
            self._get_prediction_context(include_video=self.include_video)
        except Exception as exc:
            logger.info("Taste OS context warm-up skipped: %s", exc)
        logger.info(
            "Loaded Taste OS service: model=%s type=%s modes=%d scenarios=%d",
            self.model_name,
            self.model_type,
            len(MODE_CONFIGS),
            len(SCENARIOS),
        )

    def _empty_feedback_store(self) -> dict[str, object]:
        return {
            "artists": {},
            "event_count": 0,
            "updated_at": "",
        }

    def _load_feedback_store(self) -> dict[str, object]:
        default_store = self._empty_feedback_store()
        store = _load_json_document(self.feedback_memory_path, default=default_store)
        artists = store.get("artists")
        if not isinstance(artists, dict):
            store["artists"] = {}
        if not isinstance(store.get("event_count"), int):
            store["event_count"] = 0
        if not isinstance(store.get("updated_at"), str):
            store["updated_at"] = ""
        return store

    def _write_feedback_store(self, store: dict[str, object]) -> None:
        self.feedback_memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_memory_path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")

    def _feedback_artist_summary(self, artist_name: str, stats: dict[str, object]) -> dict[str, object]:
        like_count = int(stats.get("like_count", 0))
        repeat_count = int(stats.get("repeat_count", 0))
        skip_count = int(stats.get("skip_count", 0))
        dislike_count = int(stats.get("dislike_count", 0))
        positive_score = float(like_count) + float(repeat_count) * 1.25
        negative_score = float(dislike_count) * 1.3 + float(skip_count) * 0.7
        net_score = round(positive_score - negative_score, 3)
        if net_score >= 1.0:
            memory_state = "favored"
        elif net_score <= -0.75:
            memory_state = "avoid"
        elif like_count or repeat_count or skip_count or dislike_count:
            memory_state = "mixed"
        else:
            memory_state = "neutral"
        return {
            "artist_name": artist_name,
            "like_count": like_count,
            "repeat_count": repeat_count,
            "skip_count": skip_count,
            "dislike_count": dislike_count,
            "net_score": net_score,
            "memory_state": memory_state,
            "last_signal": str(stats.get("last_signal", "")),
            "last_session_id": str(stats.get("last_session_id", "")),
            "updated_at": str(stats.get("updated_at", "")),
        }

    def _feedback_seed_artists(self, *, store: dict[str, object], limit: int) -> list[str]:
        artists = store.get("artists", {})
        if not isinstance(artists, dict):
            return []
        candidates: list[tuple[float, str, str]] = []
        for key, raw_stats in artists.items():
            if not isinstance(raw_stats, dict):
                continue
            artist_name = str(raw_stats.get("artist_name") or key).strip()
            if not artist_name:
                continue
            summary = self._feedback_artist_summary(artist_name, raw_stats)
            if str(summary["memory_state"]) == "avoid":
                continue
            if float(summary["net_score"]) <= 0.0:
                continue
            candidates.append((float(summary["net_score"]), str(summary["updated_at"]), artist_name))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [artist_name for _, _, artist_name in candidates[: max(0, int(limit))]]

    def _feedback_summary(
        self,
        *,
        store: dict[str, object] | None = None,
        limit: int = 5,
        effective_recent_artists: list[str] | None = None,
    ) -> dict[str, object]:
        active_store = store if store is not None else self._load_feedback_store()
        artists = active_store.get("artists", {})
        summaries: list[dict[str, object]] = []
        if isinstance(artists, dict):
            for key, raw_stats in artists.items():
                if not isinstance(raw_stats, dict):
                    continue
                artist_name = str(raw_stats.get("artist_name") or key).strip()
                if not artist_name:
                    continue
                summaries.append(self._feedback_artist_summary(artist_name, raw_stats))
        summaries.sort(key=lambda item: (float(item["net_score"]), str(item["updated_at"])), reverse=True)
        top_affinities = [item for item in summaries if float(item["net_score"]) > 0][:limit]
        avoid_artists = [item for item in sorted(summaries, key=lambda item: (float(item["net_score"]), str(item["updated_at"]))) if float(item["net_score"]) < 0][:limit]
        seed_artists = self._feedback_seed_artists(store=active_store, limit=max(limit, len(effective_recent_artists or [])))
        return {
            "event_count": int(active_store.get("event_count", 0)),
            "artist_count": len(summaries),
            "updated_at": str(active_store.get("updated_at", "")),
            "seed_artists": seed_artists,
            "effective_recent_artists": list(effective_recent_artists or []),
            "top_affinities": top_affinities,
            "avoid_artists": avoid_artists,
            "recent_feedback": _read_jsonl_tail(self.feedback_event_log_path, limit=limit),
        }

    def _annotate_feedback_rows(self, rows: object, *, store: dict[str, object]) -> None:
        if not isinstance(rows, list):
            return
        artists = store.get("artists", {})
        if not isinstance(artists, dict):
            artists = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            artist_name = str(row.get("artist_name", "")).strip()
            if not artist_name:
                continue
            stats = artists.get(_artist_key(artist_name), {})
            if not isinstance(stats, dict):
                stats = {}
            summary = self._feedback_artist_summary(artist_name, stats)
            row["memory_state"] = summary["memory_state"]
            row["memory_net_score"] = summary["net_score"]
            row["memory_like_count"] = summary["like_count"]
            row["memory_dislike_count"] = summary["dislike_count"]

    def _session_history_row(
        self,
        *,
        result: dict[str, object],
        session_id: str,
        created_at: str,
        effective_recent_artists: list[str],
        used_feedback_memory: bool,
    ) -> dict[str, object]:
        request = result.get("request", {}) if isinstance(result, dict) else {}
        summary = result.get("demo_summary", {}) if isinstance(result, dict) else {}
        service = result.get("service", {}) if isinstance(result, dict) else {}
        return {
            "session_id": session_id,
            "created_at": created_at,
            "mode": str(request.get("mode", "")),
            "scenario": str(request.get("scenario", "")),
            "top_artist": str(summary.get("top_artist", "")),
            "adaptive_replans": int(summary.get("adaptive_replans", 0)),
            "safe_route_steps": int(summary.get("adaptive_safe_route_steps", 0)),
            "effective_recent_artists": list(effective_recent_artists),
            "used_feedback_memory": bool(used_feedback_memory),
            "persisted": bool(service.get("persisted", False)),
        }

    def _record_session_history(
        self,
        *,
        result: dict[str, object],
        session_id: str,
        created_at: str,
        effective_recent_artists: list[str],
        used_feedback_memory: bool,
    ) -> None:
        _append_jsonl(
            self.session_history_path,
            self._session_history_row(
                result=result,
                session_id=session_id,
                created_at=created_at,
                effective_recent_artists=effective_recent_artists,
                used_feedback_memory=used_feedback_memory,
            ),
        )

    def history_snapshot(self, *, limit: int = DEFAULT_HISTORY_LIMIT) -> dict[str, object]:
        return {
            "recent_sessions": _read_jsonl_tail(self.session_history_path, limit=limit),
            "feedback_memory": self._feedback_summary(limit=limit),
        }

    def record_feedback(
        self,
        *,
        session_id: str,
        artist_name: str,
        signal: str,
        notes: str | None = None,
    ) -> dict[str, object]:
        timestamp = _utc_now_iso()
        normalized_artist_name = str(artist_name).strip()
        normalized_signal = str(signal).strip().lower()
        with self._memory_lock:
            store = self._load_feedback_store()
            artists = store.setdefault("artists", {})
            if not isinstance(artists, dict):
                artists = {}
                store["artists"] = artists
            key = _artist_key(normalized_artist_name)
            row = artists.get(key)
            if not isinstance(row, dict):
                row = {
                    "artist_name": normalized_artist_name,
                    "like_count": 0,
                    "repeat_count": 0,
                    "skip_count": 0,
                    "dislike_count": 0,
                    "last_signal": "",
                    "last_session_id": "",
                    "updated_at": "",
                }
            row["artist_name"] = normalized_artist_name
            counter_name = f"{normalized_signal}_count"
            row[counter_name] = int(row.get(counter_name, 0)) + 1
            row["last_signal"] = normalized_signal
            row["last_session_id"] = session_id
            row["updated_at"] = timestamp
            artists[key] = row
            store["event_count"] = int(store.get("event_count", 0)) + 1
            store["updated_at"] = timestamp
            self._write_feedback_store(store)
            event = {
                "timestamp": timestamp,
                "session_id": session_id,
                "artist_name": normalized_artist_name,
                "signal": normalized_signal,
                "notes": notes or "",
            }
            _append_jsonl(self.feedback_event_log_path, event)
            summary = self._feedback_summary(store=store)
        return {
            "recorded": True,
            "feedback": event,
            "feedback_memory": summary,
        }

    def artifact_url(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.output_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact path is outside the configured output dir: {path}") from exc
        encoded = "/".join(quote(part) for part in relative.parts)
        return f"/taste-os/artifacts/{encoded}"

    def resolve_artifact_path(self, relative_path: str) -> Path:
        candidate = (self.output_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.output_dir.resolve())
        except ValueError as exc:
            raise FileNotFoundError("Artifact path escapes the Taste OS output directory.") from exc
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Artifact not found: {relative_path}")
        return candidate

    def _prediction_source_signature(self, *, include_video: bool) -> tuple[tuple[str, int, int], ...]:
        return prediction_source_signature(
            run_dir=self.run_dir,
            data_dir=self.data_dir,
            include_video=include_video,
        )

    def _get_prediction_context(self, *, include_video: bool) -> PredictionInputContext:
        signature = self._prediction_source_signature(include_video=include_video)
        cached = self._context_cache.get(include_video)
        if cached is not None and cached.signature == signature:
            return cached.context

        with self._context_lock:
            cached = self._context_cache.get(include_video)
            if cached is not None and cached.signature == signature:
                return cached.context

            context = load_prediction_input_context(
                run_dir=self.run_dir,
                data_dir=self.data_dir,
                include_video=include_video,
                logger=self.logger,
            )
            self._context_cache[include_video] = _TasteOSContextCacheEntry(
                signature=signature,
                context=context,
            )
            return context

    def plan_session(
        self,
        *,
        mode: str,
        scenario: str,
        top_k: int,
        recent_artists: list[str] | None,
        include_video: bool,
        persist_artifacts: bool,
        use_feedback_memory: bool,
    ) -> dict[str, object]:
        context = self._get_prediction_context(include_video=include_video)
        feedback_store = self._load_feedback_store()
        requested_recent_artists = list(recent_artists or [])
        effective_recent_artists = list(requested_recent_artists)
        if not effective_recent_artists and use_feedback_memory:
            effective_recent_artists = self._feedback_seed_artists(
                store=feedback_store,
                limit=max(1, int(context.sequence_length)),
            )
        seq_batch, ctx_batch, sequence_names = _prepare_inputs(
            run_dir=self.run_dir,
            data_dir=self.data_dir,
            recent_artists=effective_recent_artists or None,
            include_video=include_video,
            logger=self.logger,
            context=context,
        )

        with self._plan_lock:
            payload = build_taste_os_demo_payload(
                predictor=self.predictor,
                artist_labels=list(context.artist_labels),
                sequence_labels=np.asarray(seq_batch[0], dtype="int32"),
                sequence_names=sequence_names,
                context_batch=np.asarray(ctx_batch, dtype="float32"),
                context_raw_batch=context.context_raw,
                context_features=list(context.context_features or []),
                friction_reference=context.friction_reference,
                scaler_mean=context.scaler_mean,
                scaler_scale=context.scaler_scale,
                digital_twin=self.digital_twin,
                multimodal_space=self.multimodal_space,
                safe_policy=self.safe_policy,
                mode_name=mode,
                scenario_name=scenario,
                top_k=top_k,
                artifact_paths=self.artifact_paths,
            )

        result = dict(payload)
        request = result.get("request")
        if isinstance(request, dict):
            request["requested_recent_artists"] = requested_recent_artists
            request["effective_recent_artists"] = effective_recent_artists
            request["used_feedback_memory"] = bool(use_feedback_memory and not requested_recent_artists and effective_recent_artists)

        self._annotate_feedback_rows(result.get("top_candidates"), store=feedback_store)
        self._annotate_feedback_rows(result.get("journey_plan"), store=feedback_store)
        result["memory_summary"] = self._feedback_summary(
            store=feedback_store,
            effective_recent_artists=effective_recent_artists,
        )

        session_id = f"taste-os-{secrets.token_hex(6)}"
        created_at = _utc_now_iso()
        result["service"] = {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "persisted": False,
            "session_id": session_id,
            "created_at": created_at,
        }
        if persist_artifacts:
            json_path, md_path = write_taste_os_demo_artifacts(result, output_dir=self.output_dir)
            result["service"] = {
                "run_dir": str(self.run_dir),
                "output_dir": str(self.output_dir),
                "persisted": True,
                "artifact_json": str(json_path),
                "artifact_md": str(md_path),
                "artifact_json_url": self.artifact_url(json_path),
                "artifact_md_url": self.artifact_url(md_path),
                "session_id": session_id,
                "created_at": created_at,
            }
        self._record_session_history(
            result=result,
            session_id=session_id,
            created_at=created_at,
            effective_recent_artists=effective_recent_artists,
            used_feedback_memory=bool(use_feedback_memory and not requested_recent_artists and effective_recent_artists),
        )
        return result


def _build_handler(service: "TasteOSService"):
    return build_taste_os_handler(service, page_renderer=_taste_os_page_html)


def main() -> int:
    load_local_env()
    args = parse_taste_os_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.taste_os_service")

    run_dir, champion_alias_model_name = resolve_prediction_run_dir(args.run_dir)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_row = resolve_model_row(
        run_dir,
        explicit_model_name=args.model_name,
        alias_model_name=champion_alias_model_name,
    )
    model_name = str(model_row.get("model_name", "")).strip()

    service = TasteOSService(
        run_dir=run_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        include_video=bool(args.include_video),
        max_top_k=max(1, int(args.max_top_k)),
        auth_token=(str(args.auth_token).strip() if args.auth_token else None),
        logger=logger,
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), _build_handler(service))
    logger.info("Taste OS service listening on http://%s:%d", args.host, int(args.port))
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
