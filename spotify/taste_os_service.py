from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import secrets
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .champion_alias import resolve_prediction_run_dir
from .digital_twin import ListenerDigitalTwinArtifact
from .env import load_local_env
from .multimodal import MultimodalArtistSpace
from .predict_next import (
    PredictionInputContext,
    _prediction_serving_bundle_path,
    _prepare_inputs,
    load_prediction_input_context,
    prediction_source_signature,
)
from .safe_policy import SafeBanditPolicyArtifact
from .serving import load_predictor, resolve_model_row
from .taste_os_state import (
    ActiveSessionNotFoundError,
    SessionEventConflictError,
    StaleSessionVersionError,
    TasteOSStateStore,
)
from .taste_os_demo import MODE_CONFIGS, SCENARIOS, _load_artifact, build_taste_os_demo_payload, write_taste_os_demo_artifacts
from .taste_os_http import (
    DEFAULT_HISTORY_LIMIT,
    RequestValidationError,
    artist_key as _artist_key,
    build_taste_os_handler,
    normalize_taste_os_feedback_payload,
    normalize_taste_os_payload,
    normalize_taste_os_session_event_payload,
    parse_taste_os_args,
    utc_now_iso as _utc_now_iso,
)
from .taste_os_page import render_taste_os_page_html

__all__ = [
    "RequestValidationError",
    "TasteOSService",
    "normalize_taste_os_feedback_payload",
    "normalize_taste_os_payload",
    "normalize_taste_os_session_event_payload",
    "_taste_os_page_html",
]

@dataclass(frozen=True)
class _TasteOSContextCacheEntry:
    signature: tuple[tuple[str, int, int], ...]
    context: PredictionInputContext


def _taste_os_page_html(service: "TasteOSService") -> str:
    return render_taste_os_page_html(service)


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


class TasteOSService:
    def __init__(
        self,
        *,
        run_dir: Path,
        data_dir: Path | None,
        output_dir: Path,
        model_name: str,
        include_video: bool,
        max_top_k: int,
        auth_token: str | None,
        logger: logging.Logger,
        state_db_path: Path | None = None,
        state_database_url: str | None = None,
        require_serving_bundle: bool = False,
        digital_twin: ListenerDigitalTwinArtifact | None = None,
        multimodal_space: MultimodalArtistSpace | None = None,
        safe_policy: SafeBanditPolicyArtifact | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.data_dir = data_dir.expanduser().resolve() if isinstance(data_dir, Path) else None
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_video = include_video
        self.max_top_k = max(1, int(max_top_k))
        self.auth_token = auth_token
        self.logger = logger
        self._context_lock = threading.Lock()
        self._plan_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._session_event_lock = threading.Lock()
        self._context_cache: dict[bool, _TasteOSContextCacheEntry] = {}
        self.feedback_memory_path = self.output_dir / "feedback_memory.json"
        self.feedback_event_log_path = self.output_dir / "feedback_events.jsonl"
        self.session_history_path = self.output_dir / "session_history.jsonl"
        self.state_db_path = (state_db_path or (self.output_dir / "taste_os_state.sqlite3")).expanduser().resolve()
        self.state_database_url = str(state_database_url or "").strip() or None
        self.require_serving_bundle = bool(require_serving_bundle)
        self.state_store = TasteOSStateStore(
            db_path=(None if self.state_database_url else self.state_db_path),
            logger=logger,
            database_url=self.state_database_url,
            legacy_feedback_memory_path=self.feedback_memory_path,
            legacy_feedback_event_log_path=self.feedback_event_log_path,
            legacy_session_history_path=self.session_history_path,
        )

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
            "Loaded Taste OS service: model=%s type=%s modes=%d scenarios=%d state_db=%s bundle_required=%s",
            self.model_name,
            self.model_type,
            len(MODE_CONFIGS),
            len(SCENARIOS),
            self.state_db_path,
            self.require_serving_bundle,
        )

    def _empty_feedback_store(self) -> dict[str, object]:
        return {
            "artists": {},
            "event_count": 0,
            "updated_at": "",
        }

    def _load_feedback_store(self) -> dict[str, object]:
        store = self.state_store.feedback_store()
        artists = store.get("artists")
        if not isinstance(artists, dict):
            store["artists"] = {}
        if not isinstance(store.get("event_count"), int):
            store["event_count"] = 0
        if not isinstance(store.get("updated_at"), str):
            store["updated_at"] = ""
        return store

    def _write_feedback_store(self, store: dict[str, object]) -> None:
        _ = store
        self.state_store.sync_legacy_snapshots()

    def _feedback_artist_summary(self, artist_name: str, stats: dict[str, object]) -> dict[str, object]:
        like_count = _coerce_int(stats.get("like_count", 0))
        repeat_count = _coerce_int(stats.get("repeat_count", 0))
        skip_count = _coerce_int(stats.get("skip_count", 0))
        dislike_count = _coerce_int(stats.get("dislike_count", 0))
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
            if _coerce_float(summary["net_score"]) <= 0.0:
                continue
            candidates.append((_coerce_float(summary["net_score"]), str(summary["updated_at"]), artist_name))
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
        summaries.sort(key=lambda item: (_coerce_float(item.get("net_score")), str(item["updated_at"])), reverse=True)
        top_affinities = [item for item in summaries if _coerce_float(item.get("net_score")) > 0][:limit]
        avoid_artists = [
            item
            for item in sorted(summaries, key=lambda item: (_coerce_float(item.get("net_score")), str(item["updated_at"])))
            if _coerce_float(item.get("net_score")) < 0
        ][:limit]
        seed_artists = self._feedback_seed_artists(store=active_store, limit=max(limit, len(effective_recent_artists or [])))
        return {
            "event_count": _coerce_int(active_store.get("event_count", 0)),
            "artist_count": len(summaries),
            "updated_at": str(active_store.get("updated_at", "")),
            "seed_artists": seed_artists,
            "effective_recent_artists": list(effective_recent_artists or []),
            "top_affinities": top_affinities,
            "avoid_artists": avoid_artists,
            "recent_feedback": self.state_store.recent_feedback(limit=limit),
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
        request = _mapping(result.get("request", {}) if isinstance(result, dict) else {})
        summary = _mapping(result.get("demo_summary", {}) if isinstance(result, dict) else {})
        service = _mapping(result.get("service", {}) if isinstance(result, dict) else {})
        return {
            "session_id": session_id,
            "created_at": created_at,
            "mode": str(request.get("mode", "")),
            "scenario": str(request.get("scenario", "")),
            "top_artist": str(summary.get("top_artist", "")),
            "adaptive_replans": _coerce_int(summary.get("adaptive_replans", 0)),
            "safe_route_steps": _coerce_int(summary.get("adaptive_safe_route_steps", 0)),
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
        self.state_store.record_session_history(
            self._session_history_row(
                result=result,
                session_id=session_id,
                created_at=created_at,
                effective_recent_artists=effective_recent_artists,
                used_feedback_memory=used_feedback_memory,
            )
        )

    def history_snapshot(self, *, limit: int = DEFAULT_HISTORY_LIMIT) -> dict[str, object]:
        return {
            "recent_sessions": self.state_store.recent_sessions(limit=limit),
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
            event = self.state_store.record_feedback(
                timestamp=timestamp,
                session_id=session_id,
                artist_name=normalized_artist_name,
                signal=normalized_signal,
                notes=notes,
            )
            store = self.state_store.feedback_store()
            summary = self._feedback_summary(store=store)
        return {
            "recorded": True,
            "feedback": event,
            "feedback_memory": summary,
        }

    @staticmethod
    def _mark_idempotent_replay(response: dict[str, object]) -> dict[str, object]:
        replay = dict(response)
        event = _mapping(replay.get("session_event", {}))
        replay["session_event"] = {
            **event,
            "idempotent_replay": True,
        }
        return replay

    @staticmethod
    def _event_scenario(event_type: str) -> str:
        return "repeat_request" if event_type in {"repeat", "like"} else "skip_recovery"

    @staticmethod
    def _why_session_changed(*, event_type: str, artist_name: str, durable_feedback: bool) -> str:
        if event_type == "skip":
            return (
                f"You skipped {artist_name}, so this session moved away from that choice "
                "and replanned toward nearby alternatives."
            )
        if event_type == "repeat":
            return (
                f"You repeated {artist_name}, so this session leaned into continuity "
                "and more familiar choices."
            )
        if event_type == "like":
            suffix = " The like was also saved to your durable taste feedback." if durable_feedback else ""
            return f"You liked {artist_name}, so this session now uses it as a stronger anchor.{suffix}"
        suffix = " The dislike was also saved to your durable taste feedback." if durable_feedback else ""
        return f"You disliked {artist_name}, so this session moved away from it and replanned.{suffix}"

    def _adapt_session_recent_artists(
        self,
        *,
        active_session: dict[str, object],
        event_type: str,
        artist_name: str,
        include_video: bool,
    ) -> list[str]:
        adaptation = _mapping(active_session.get("adaptation", {}))
        current = adaptation.get("effective_recent_artists", [])
        recent = [str(item).strip() for item in current if str(item).strip()] if isinstance(current, list) else []
        artist_name_key = _artist_key(artist_name)
        recent = [item for item in recent if _artist_key(item) != artist_name_key]

        if event_type in {"repeat", "like"}:
            recent.append(artist_name)
        else:
            previous_plan = _mapping(active_session.get("plan", {}))
            adaptive = _mapping(previous_plan.get("adaptive_session", {}))
            candidate_sources = [
                adaptive.get("final_sequence_tail", []),
                previous_plan.get("journey_plan", []),
                previous_plan.get("top_candidates", []),
            ]
            for source in candidate_sources:
                if not isinstance(source, list):
                    continue
                for item in source:
                    candidate = str(item.get("artist_name", "")).strip() if isinstance(item, dict) else str(item).strip()
                    if not candidate or _artist_key(candidate) == artist_name_key:
                        continue
                    if any(_artist_key(existing) == _artist_key(candidate) for existing in recent):
                        continue
                    recent.append(candidate)

        context = self._get_prediction_context(include_video=include_video)
        sequence_length = max(1, int(context.sequence_length))
        return recent[-sequence_length:]

    def apply_session_event(
        self,
        *,
        session_id: str,
        event_id: str,
        event_type: str,
        artist_name: str,
        expected_version: int,
    ) -> dict[str, object]:
        event_request = normalize_taste_os_session_event_payload(
            {
                "session_id": session_id,
                "event_id": event_id,
                "event_type": event_type,
                "artist_name": artist_name,
                "expected_version": expected_version,
            }
        )
        session_id = event_request["session_id"]
        event_id = event_request["event_id"]
        event_type = event_request["event_type"]
        artist_name = event_request["artist_name"]
        expected_version = event_request["expected_version"]
        with self._session_event_lock:
            existing_event = self.state_store.session_event(session_id=session_id, event_id=event_id)
            if existing_event is not None:
                if existing_event["request"] != event_request:
                    raise RequestValidationError(
                        status_code=409,
                        code="event_id_conflict",
                        message=f"event_id {event_id!r} was already used with a different payload.",
                    )
                return self._mark_idempotent_replay(_mapping(existing_event.get("response", {})))

            try:
                active_session = self.state_store.active_session(session_id)
            except ActiveSessionNotFoundError as exc:
                raise RequestValidationError(
                    status_code=404,
                    code="session_not_found",
                    message=str(exc),
                ) from exc

            current_version = _coerce_int(active_session.get("version", 0))
            if current_version != int(expected_version):
                raise RequestValidationError(
                    status_code=409,
                    code="stale_session_version",
                    message=f"Session version {expected_version} is stale; current version is {current_version}.",
                    details={
                        "expected_version": int(expected_version),
                        "current_version": current_version,
                    },
                )

            active_request = _mapping(active_session.get("request", {}))
            include_video = bool(active_request.get("include_video", self.include_video))
            adapted_recent_artists = self._adapt_session_recent_artists(
                active_session=active_session,
                event_type=event_type,
                artist_name=artist_name,
                include_video=include_video,
            )
            scenario = self._event_scenario(event_type)
            resulting_version = int(expected_version) + 1
            result = self.plan_session(
                mode=str(active_request.get("mode", "focus")),
                scenario=scenario,
                top_k=max(1, _coerce_int(active_request.get("top_k", 5))),
                recent_artists=adapted_recent_artists,
                include_video=include_video,
                persist_artifacts=bool(active_request.get("persist_artifacts", False)),
                use_feedback_memory=False,
                _session_id=session_id,
                _created_at=str(active_session.get("created_at", "")),
                _session_version=resulting_version,
                _record_state=False,
            )
            durable_feedback = event_type in {"like", "dislike"}
            why_this_changed = self._why_session_changed(
                event_type=event_type,
                artist_name=artist_name,
                durable_feedback=durable_feedback,
            )
            result["why_this_changed"] = why_this_changed
            result["session_event"] = {
                "event_id": event_id,
                "event_type": event_type,
                "artist_name": artist_name,
                "expected_version": int(expected_version),
                "resulting_version": resulting_version,
                "durable_feedback": durable_feedback,
                "idempotent_replay": False,
            }
            service_payload = _mapping(result.get("service", {}))
            updated_at = _utc_now_iso()
            result["service"] = {
                **service_payload,
                "session_id": session_id,
                "created_at": str(active_session.get("created_at", "")),
                "updated_at": updated_at,
                "version": resulting_version,
            }
            updated_request = {
                **active_request,
                "scenario": scenario,
                "recent_artists": adapted_recent_artists,
                "use_feedback_memory": False,
            }
            previous_adaptation = _mapping(active_session.get("adaptation", {}))
            previous_events = previous_adaptation.get("events", [])
            adaptation_events = list(previous_events) if isinstance(previous_events, list) else []
            adaptation_events.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "artist_name": artist_name,
                    "version": resulting_version,
                    "why_this_changed": why_this_changed,
                }
            )
            adaptation: dict[str, object] = {
                "effective_recent_artists": adapted_recent_artists,
                "events": adaptation_events,
            }
            try:
                stored_result, idempotent_replay = self.state_store.apply_session_event(
                    timestamp=updated_at,
                    session_id=session_id,
                    event_id=event_id,
                    event_type=event_type,
                    artist_name=artist_name,
                    expected_version=expected_version,
                    request=event_request,
                    response=result,
                    updated_request=updated_request,
                    adaptation=adaptation,
                    durable_feedback=durable_feedback,
                )
            except ActiveSessionNotFoundError as exc:
                raise RequestValidationError(
                    status_code=404,
                    code="session_not_found",
                    message=str(exc),
                ) from exc
            except StaleSessionVersionError as exc:
                concurrent_event = self.state_store.session_event(session_id=session_id, event_id=event_id)
                if concurrent_event is not None and concurrent_event["request"] == event_request:
                    return self._mark_idempotent_replay(_mapping(concurrent_event.get("response", {})))
                raise RequestValidationError(
                    status_code=409,
                    code="stale_session_version",
                    message=str(exc),
                    details={
                        "expected_version": exc.expected_version,
                        "current_version": exc.current_version,
                    },
                ) from exc
            except SessionEventConflictError as exc:
                raise RequestValidationError(
                    status_code=409,
                    code="event_id_conflict",
                    message=str(exc),
                ) from exc

            if idempotent_replay:
                return self._mark_idempotent_replay(stored_result)
            self._record_session_history(
                result=stored_result,
                session_id=session_id,
                created_at=str(active_session.get("created_at", "")),
                effective_recent_artists=adapted_recent_artists,
                used_feedback_memory=False,
            )
            return stored_result

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

    def health_payload(self) -> dict[str, object]:
        bundle_path = _prediction_serving_bundle_path(self.run_dir, include_video=self.include_video)
        state_health = self.state_store.health_payload()
        return {
            "status": "ok",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "state_db_path": str(self.state_db_path),
            "state_database_url": str(state_health.get("database_url", "")),
            "state_backend": str(state_health.get("backend", "")),
            "state_reachable": bool(state_health.get("reachable", False)),
            "max_top_k": self.max_top_k,
            "requires_auth": bool(self.auth_token),
            "data_dir_configured": bool(self.data_dir),
            "require_serving_bundle": self.require_serving_bundle,
            "serving_bundle_path": str(bundle_path),
            "serving_bundle_present": bundle_path.exists(),
            "input_source_mode": (
                "serving_bundle"
                if bundle_path.exists()
                else ("raw_history_fallback" if self.data_dir is not None else "bundle_required")
            ),
            "mode_count": len(MODE_CONFIGS),
            "scenario_count": len(SCENARIOS),
        }

    def _prediction_source_signature(self, *, include_video: bool) -> tuple[tuple[str, int, int], ...]:
        return prediction_source_signature(
            run_dir=self.run_dir,
            data_dir=self.data_dir,
            include_video=include_video,
            prefer_serving_bundle=True,
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
                prefer_serving_bundle=True,
                require_serving_bundle=self.require_serving_bundle,
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
        _session_id: str | None = None,
        _created_at: str | None = None,
        _session_version: int = 0,
        _record_state: bool = True,
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

        session_id = str(_session_id or f"taste-os-{secrets.token_hex(6)}")
        created_at = str(_created_at or _utc_now_iso())
        result["service"] = {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "persisted": False,
            "session_id": session_id,
            "created_at": created_at,
            "version": int(_session_version),
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
                "version": int(_session_version),
            }
        if _record_state:
            active_request = {
                "mode": mode,
                "scenario": scenario,
                "top_k": int(top_k),
                "recent_artists": requested_recent_artists,
                "include_video": bool(include_video),
                "persist_artifacts": bool(persist_artifacts),
                "use_feedback_memory": bool(use_feedback_memory),
            }
            self.state_store.create_active_session(
                session_id=session_id,
                created_at=created_at,
                request=active_request,
                plan=result,
                adaptation={
                    "effective_recent_artists": effective_recent_artists,
                    "events": [],
                },
            )
            self._record_session_history(
                result=result,
                session_id=session_id,
                created_at=created_at,
                effective_recent_artists=effective_recent_artists,
                used_feedback_memory=bool(
                    use_feedback_memory and not requested_recent_artists and effective_recent_artists
                ),
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
    data_dir = Path(args.data_dir).expanduser().resolve() if str(args.data_dir).strip() else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    state_db_path = Path(args.state_db).expanduser().resolve() if str(args.state_db).strip() else None
    state_database_url = str(getattr(args, "state_db_url", "") or "").strip() or None
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
        state_db_path=state_db_path,
        state_database_url=state_database_url,
        require_serving_bundle=bool(args.require_serving_bundle),
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
