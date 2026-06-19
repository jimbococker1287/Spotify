from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from importlib import import_module
import inspect
import logging
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol, Sequence

from .run_artifacts import write_json, write_markdown


MANIFEST_SCHEMA_VERSION = 1
STAGE_ORDER = (
    "candidate_dataset",
    "dcn_training",
    "optuna_tuning",
    "public_pretraining",
    "promotion_gates",
)
SUCCESS_STATUSES = frozenset({"complete", "skipped"})
TERMINAL_STATUSES = frozenset(
    {"complete", "partial", "failed", "blocked", "skipped"}
)


@dataclass(frozen=True)
class NextPassConfig:
    output_dir: Path
    enable_public_pretraining: bool = False
    stage_options: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    @property
    def artifact_root(self) -> Path:
        return (
            self.output_dir
            / "analysis"
            / "recommender_expansion"
            / "next_pass"
        )


@dataclass(frozen=True)
class StageRequest:
    stage_name: str
    artifact_dir: Path
    manifest_path: Path
    config: NextPassConfig
    options: Mapping[str, object]
    upstream: Mapping[str, object]


class StageCallable(Protocol):
    def __call__(self, request: StageRequest) -> object: ...


@dataclass(frozen=True)
class NextPassAdapters:
    candidate_dataset_builder: StageCallable
    dcn_trainer: StageCallable
    optuna_tuners: Mapping[str, StageCallable]
    promotion_gates: StageCallable
    public_pretrainer: StageCallable | None = None


class AdapterUnavailableError(RuntimeError):
    """Raised when no compatible implementation is available for a stage."""


class AdapterContractError(TypeError):
    """Raised when a discovered callable does not accept the adapter contract."""


@dataclass(frozen=True)
class LazyImportAdapter:
    module_names: tuple[str, ...]
    callable_names: tuple[str, ...]

    def __call__(self, request: StageRequest) -> object:
        failures: list[str] = []
        for module_name in self.module_names:
            try:
                module = import_module(module_name)
            except Exception as exc:
                failures.append(
                    f"{module_name}: {type(exc).__name__}: {exc}"
                )
                continue
            for callable_name in self.callable_names:
                implementation = getattr(module, callable_name, None)
                if callable(implementation):
                    return _invoke_discovered_callable(implementation, request)
            failures.append(
                f"{module_name}: none of {', '.join(self.callable_names)} found"
            )
        detail = "; ".join(failures) or "no module candidates configured"
        raise AdapterUnavailableError(
            f"No implementation available for {request.stage_name}. {detail}"
        )


def default_next_pass_adapters() -> NextPassAdapters:
    """Create lazy adapters without importing optional worker modules."""
    return NextPassAdapters(
        candidate_dataset_builder=LazyImportAdapter(
            module_names=(
                "spotify.recommender_candidates",
                "spotify.retrieval_reranking",
                "spotify.dcn_candidate_data",
            ),
            callable_names=(
                "build_candidate_dataset",
                "build_reranking_dataset",
                "build_dcn_candidate_dataset",
            ),
        ),
        dcn_trainer=LazyImportAdapter(
            module_names=(
                "spotify.dcn_v2_training",
                "spotify.dcn_v2_model",
            ),
            callable_names=(
                "train_dcn_v2",
                "run_dcn_v2_training",
                "train_dcn",
            ),
        ),
        optuna_tuners={
            "recommender_models": LazyImportAdapter(
                module_names=(
                    "spotify.recommender_optuna",
                    "spotify.optuna_training",
                    "spotify.model_tuning",
                ),
                callable_names=(
                    "run_optuna_tuning",
                    "tune_recommender_models",
                    "tune_models",
                ),
            )
        },
        public_pretrainer=LazyImportAdapter(
            module_names=(
                "spotify.public_pretraining",
                "spotify.public_training_data",
            ),
            callable_names=(
                "run_governed_public_pretraining",
                "train_with_public_data",
                "run_public_pretraining",
            ),
        ),
        promotion_gates=LazyImportAdapter(
            module_names=(
                "spotify.recommender_promotion",
                "spotify.promotion_gates",
                "spotify.recommender_safety",
            ),
            callable_names=(
                "evaluate_promotion_gates",
                "run_promotion_gates",
                "promote_if_eligible",
            ),
        ),
    )


def _invoke_discovered_callable(
    implementation: Callable[..., object],
    request: StageRequest,
) -> object:
    signature = inspect.signature(implementation)
    parameters = signature.parameters
    if not parameters:
        return implementation()

    available: dict[str, object] = {
        "request": request,
        "stage_request": request,
        "config": request.config,
        "output_dir": request.artifact_dir,
        "artifact_dir": request.artifact_dir,
        "manifest_path": request.manifest_path,
        "options": request.options,
        "upstream": request.upstream,
        "stage_name": request.stage_name,
    }
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return implementation(**available)

    positional_only = [
        parameter
        for parameter in parameters.values()
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only:
        if len(parameters) == 1:
            return implementation(request)
        raise AdapterContractError(
            f"{implementation!r} has unsupported positional-only parameters. "
            "Inject a small wrapper accepting StageRequest."
        )

    kwargs = {
        name: available[name]
        for name in parameters
        if name in available
    }
    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        and name not in kwargs
    ]
    if missing:
        if len(parameters) == 1:
            return implementation(request)
        raise AdapterContractError(
            f"{implementation!r} requires unsupported arguments "
            f"{', '.join(missing)}. Inject a StageRequest wrapper."
        )
    return implementation(**kwargs)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value.resolve())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_jsonable(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _jsonable(item_method())
        except Exception:
            pass
    return str(value)


def _normalize_result(result: object) -> tuple[str, object]:
    payload = _jsonable(result)
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {"result": payload}
    status = str(payload.get("status", "complete")).strip().lower()
    if status not in TERMINAL_STATUSES:
        status = "complete"
    return status, dict(payload)


def _error_payload(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _stage_record(
    name: str,
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "name": name,
        "status": status,
    }
    if reason:
        record["reason"] = reason
    return record


def _upstream_payload(
    stages: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        name: {
            "status": record.get("status"),
            "output": record.get("output", {}),
        }
        for name, record in stages.items()
        if record.get("status") in SUCCESS_STATUSES
    }


def _request_for(
    config: NextPassConfig,
    manifest_path: Path,
    stages: Mapping[str, Mapping[str, object]],
    stage_name: str,
    *,
    artifact_name: str | None = None,
) -> StageRequest:
    option_key = stage_name.split(":", maxsplit=1)[0]
    options = config.stage_options.get(
        stage_name,
        config.stage_options.get(option_key, {}),
    )
    artifact_dir = (
        config.artifact_root
        / "stages"
        / (artifact_name or stage_name.replace(":", "_"))
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return StageRequest(
        stage_name=stage_name,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        config=config,
        options=options,
        upstream=_upstream_payload(stages),
    )


def _run_stage(
    name: str,
    adapter: StageCallable,
    request: StageRequest,
) -> dict[str, object]:
    started_at = _utc_now()
    started = time.monotonic()
    try:
        status, output = _normalize_result(adapter(request))
        record = _stage_record(name, status=status)
        record["output"] = output
        if status not in SUCCESS_STATUSES and output.get("reason"):
            record["reason"] = str(output["reason"])
    except Exception as exc:
        record = _stage_record(name, status="failed")
        record["error"] = _error_payload(exc)
    record["started_at"] = started_at
    record["finished_at"] = _utc_now()
    record["duration_seconds"] = round(time.monotonic() - started, 6)
    return record


def _dependencies_complete(
    stages: Mapping[str, Mapping[str, object]],
    dependencies: Sequence[str],
) -> bool:
    return all(
        stages.get(name, {}).get("status") in SUCCESS_STATUSES
        for name in dependencies
    )


def _dependency_reason(
    stages: Mapping[str, Mapping[str, object]],
    dependencies: Sequence[str],
) -> str:
    unavailable = [
        f"{name}={stages.get(name, {}).get('status', 'missing')}"
        for name in dependencies
        if stages.get(name, {}).get("status") not in SUCCESS_STATUSES
    ]
    return "Blocked by " + ", ".join(unavailable) + "."


def _overall_status(stages: Mapping[str, Mapping[str, object]]) -> str:
    statuses = [str(stages.get(name, {}).get("status", "blocked")) for name in STAGE_ORDER]
    required = [
        status
        for name, status in zip(STAGE_ORDER, statuses)
        if name != "public_pretraining" or status != "skipped"
    ]
    if all(status == "complete" for status in required):
        return "complete"
    if any(status == "complete" for status in statuses):
        return "partial"
    return "blocked"


def _continuation_lines(manifest: Mapping[str, object]) -> list[str]:
    stages = manifest.get("stages", {})
    stage_records = stages if isinstance(stages, Mapping) else {}
    lines = [
        "# Recommender Next Pass",
        "",
        f"- Overall status: `{manifest.get('status', 'unknown')}`",
        f"- Updated: `{manifest.get('updated_at', '')}`",
        "",
        "## Stages",
        "",
    ]
    for name in STAGE_ORDER:
        record = stage_records.get(name, {})
        if not isinstance(record, Mapping):
            record = {}
        line = f"- `{name}`: `{record.get('status', 'pending')}`"
        reason = record.get("reason")
        error = record.get("error")
        if reason:
            line += f" - {reason}"
        elif isinstance(error, Mapping) and error.get("message"):
            line += f" - {error['message']}"
        lines.append(line)

    incomplete = [
        name
        for name in STAGE_ORDER
        if not isinstance(stage_records.get(name), Mapping)
        or stage_records[name].get("status") not in SUCCESS_STATUSES
    ]
    lines.extend(["", "## Continue", ""])
    if incomplete:
        lines.append(
            "Resume from the first unresolved stage: "
            f"`{incomplete[0]}`. Completed stage outputs remain in "
            "`next_pass_manifest.json`."
        )
    else:
        lines.append(
            "All configured stages completed. Review promotion-gate output "
            "before changing the serving champion."
        )
    return lines


def _checkpoint(
    manifest_path: Path,
    continuation_path: Path,
    manifest: dict[str, object],
) -> None:
    manifest["updated_at"] = _utc_now()
    write_json(manifest_path, manifest)
    write_markdown(continuation_path, _continuation_lines(manifest))


def _run_optuna_stage(
    *,
    config: NextPassConfig,
    adapters: NextPassAdapters,
    manifest: dict[str, object],
    manifest_path: Path,
    continuation_path: Path,
) -> dict[str, object]:
    stages = manifest["stages"]
    assert isinstance(stages, dict)
    started_at = _utc_now()
    started = time.monotonic()
    tuner_records: dict[str, dict[str, object]] = {}
    record: dict[str, object] = {
        "name": "optuna_tuning",
        "status": "running",
        "started_at": started_at,
        "tuners": tuner_records,
    }
    stages["optuna_tuning"] = record
    _checkpoint(manifest_path, continuation_path, manifest)

    if not adapters.optuna_tuners:
        record.update(
            _stage_record(
                "optuna_tuning",
                status="blocked",
                reason="No Optuna tuner adapters were configured.",
            )
        )
    else:
        for tuner_name, tuner in adapters.optuna_tuners.items():
            tuner_request = _request_for(
                config,
                manifest_path,
                stages,
                f"optuna_tuning:{tuner_name}",
                artifact_name=f"optuna_{tuner_name}",
            )
            tuner_records[tuner_name] = _run_stage(
                tuner_name,
                tuner,
                tuner_request,
            )
            _checkpoint(manifest_path, continuation_path, manifest)

        statuses = [
            str(tuner_record.get("status", "failed"))
            for tuner_record in tuner_records.values()
        ]
        if statuses and all(status == "complete" for status in statuses):
            record["status"] = "complete"
        elif any(status == "complete" for status in statuses):
            record["status"] = "partial"
        elif statuses and all(status == "blocked" for status in statuses):
            record["status"] = "blocked"
        else:
            record["status"] = "failed"

    record["finished_at"] = _utc_now()
    record["duration_seconds"] = round(time.monotonic() - started, 6)
    return record


def run_recommender_next_pass(
    *,
    config: NextPassConfig,
    adapters: NextPassAdapters | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    """Run and checkpoint the dependency-ordered recommender next pass."""
    resolved_adapters = adapters or default_next_pass_adapters()
    resolved_logger = logger or logging.getLogger(__name__)
    root = config.artifact_root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "next_pass_manifest.json"
    continuation_path = root / "CONTINUE_NEXT_PASS.md"
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "running",
        "config": {
            "output_dir": str(config.output_dir.resolve()),
            "enable_public_pretraining": config.enable_public_pretraining,
            "stage_options": _jsonable(config.stage_options),
        },
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "continuation": str(continuation_path.resolve()),
        },
        "stage_order": list(STAGE_ORDER),
        "stages": {},
    }
    stages = manifest["stages"]
    assert isinstance(stages, dict)
    _checkpoint(manifest_path, continuation_path, manifest)

    candidate_request = _request_for(
        config,
        manifest_path,
        stages,
        "candidate_dataset",
    )
    stages["candidate_dataset"] = _run_stage(
        "candidate_dataset",
        resolved_adapters.candidate_dataset_builder,
        candidate_request,
    )
    _checkpoint(manifest_path, continuation_path, manifest)

    dcn_dependencies = ("candidate_dataset",)
    if _dependencies_complete(stages, dcn_dependencies):
        dcn_request = _request_for(
            config,
            manifest_path,
            stages,
            "dcn_training",
        )
        stages["dcn_training"] = _run_stage(
            "dcn_training",
            resolved_adapters.dcn_trainer,
            dcn_request,
        )
    else:
        stages["dcn_training"] = _stage_record(
            "dcn_training",
            status="blocked",
            reason=_dependency_reason(stages, dcn_dependencies),
        )
    _checkpoint(manifest_path, continuation_path, manifest)

    optuna_dependencies = ("dcn_training",)
    if _dependencies_complete(stages, optuna_dependencies):
        stages["optuna_tuning"] = _run_optuna_stage(
            config=config,
            adapters=resolved_adapters,
            manifest=manifest,
            manifest_path=manifest_path,
            continuation_path=continuation_path,
        )
    else:
        stages["optuna_tuning"] = _stage_record(
            "optuna_tuning",
            status="blocked",
            reason=_dependency_reason(stages, optuna_dependencies),
        )
    _checkpoint(manifest_path, continuation_path, manifest)

    public_dependencies = ("optuna_tuning",)
    if not config.enable_public_pretraining:
        stages["public_pretraining"] = _stage_record(
            "public_pretraining",
            status="skipped",
            reason="Governed public pretraining was not enabled.",
        )
    elif not _dependencies_complete(stages, public_dependencies):
        stages["public_pretraining"] = _stage_record(
            "public_pretraining",
            status="blocked",
            reason=_dependency_reason(stages, public_dependencies),
        )
    elif resolved_adapters.public_pretrainer is None:
        stages["public_pretraining"] = _stage_record(
            "public_pretraining",
            status="blocked",
            reason="Public pretraining is enabled but no adapter was configured.",
        )
    else:
        public_request = _request_for(
            config,
            manifest_path,
            stages,
            "public_pretraining",
        )
        stages["public_pretraining"] = _run_stage(
            "public_pretraining",
            resolved_adapters.public_pretrainer,
            public_request,
        )
    _checkpoint(manifest_path, continuation_path, manifest)

    promotion_dependencies = [
        "candidate_dataset",
        "dcn_training",
        "optuna_tuning",
    ]
    if config.enable_public_pretraining:
        promotion_dependencies.append("public_pretraining")
    if _dependencies_complete(stages, promotion_dependencies):
        promotion_request = _request_for(
            config,
            manifest_path,
            stages,
            "promotion_gates",
        )
        stages["promotion_gates"] = _run_stage(
            "promotion_gates",
            resolved_adapters.promotion_gates,
            promotion_request,
        )
    else:
        stages["promotion_gates"] = _stage_record(
            "promotion_gates",
            status="blocked",
            reason=_dependency_reason(stages, promotion_dependencies),
        )

    manifest["status"] = _overall_status(stages)
    _checkpoint(manifest_path, continuation_path, manifest)
    resolved_logger.info(
        "Recommender next pass finished with status=%s; manifest=%s",
        manifest["status"],
        manifest_path,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the dependency-ordered recommender next pass."
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--enable-public-pretraining",
        action="store_true",
        help="Run public-data pretraining after its governance checks.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = run_recommender_next_pass(
        config=NextPassConfig(
            output_dir=Path(args.output_dir),
            enable_public_pretraining=args.enable_public_pretraining,
        )
    )
    print(manifest["artifacts"]["manifest"])
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
