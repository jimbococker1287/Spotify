from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .run_artifacts import safe_read_json


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _latest_path(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    best_path: Path | None = None
    best_key: tuple[int, int, str] | None = None
    for path in directory.glob(pattern):
        if not path.is_file():
            continue
        stat = path.stat()
        key = (int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))), int(stat.st_size), path.name)
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
    return best_path


@dataclass(frozen=True)
class PortfolioArtifactBundle:
    output_root: Path
    taste_os_showcase_json: Path
    taste_os_showcase_md: Path
    taste_os_showcase_payload: dict[str, Any]
    control_room_json: Path
    control_room_md: Path
    control_room_payload: dict[str, Any]
    creator_manifest_paths: tuple[Path, ...]
    creator_manifest_path: Path | None
    creator_manifest: dict[str, Any]
    creator_primary_report_path: Path | None
    creator_comparison_markdown_paths: dict[str, Path]
    creator_comparison_csv_paths: dict[str, Path]
    research_claims_json: Path
    research_claims_md: Path
    research_claims_payload: dict[str, Any]
    benchmark_manifest_json: Path | None
    benchmark_manifest_md: Path | None
    benchmark_manifest_payload: dict[str, Any]


@lru_cache(maxsize=8)
def _load_portfolio_artifact_bundle_cached(output_root_str: str) -> PortfolioArtifactBundle:
    output_root = Path(output_root_str).expanduser().resolve()

    taste_os_showcase_json = output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.json"
    taste_os_showcase_md = output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.md"
    control_room_json = output_root / "analytics" / "control_room.json"
    control_room_md = output_root / "analytics" / "control_room.md"
    research_claims_json = output_root / "analysis" / "research_claims" / "research_claims.json"
    research_claims_md = output_root / "analysis" / "research_claims" / "research_claims.md"

    creator_dir = output_root / "analysis" / "public_spotify" / "creator_label_intelligence"
    creator_manifest_paths = tuple(sorted(path for path in creator_dir.glob("*_report_family.json") if path.is_file()))
    creator_manifest_path = _latest_path(creator_dir, "*_report_family.json")
    creator_manifest = _coerce_dict(safe_read_json(creator_manifest_path, default={})) if creator_manifest_path else {}
    creator_primary_report_path = (
        Path(str(creator_manifest.get("primary_report", "")).strip())
        if str(creator_manifest.get("primary_report", "")).strip()
        else None
    )
    creator_comparison_markdown_paths = {
        str(key): Path(str(value))
        for key, value in _coerce_dict(creator_manifest.get("comparison_view_markdown")).items()
        if str(value).strip()
    }
    creator_comparison_csv_paths = {
        str(key): Path(str(value))
        for key, value in _coerce_dict(creator_manifest.get("comparison_view_csv")).items()
        if str(value).strip()
    }

    history_dir = output_root / "history"
    benchmark_manifest_json = _latest_path(history_dir, "benchmark_lock_*_manifest.json")
    benchmark_manifest_md = _latest_path(history_dir, "benchmark_lock_*_manifest.md")
    benchmark_manifest_payload = (
        _coerce_dict(safe_read_json(benchmark_manifest_json, default={})) if benchmark_manifest_json else {}
    )

    return PortfolioArtifactBundle(
        output_root=output_root,
        taste_os_showcase_json=taste_os_showcase_json,
        taste_os_showcase_md=taste_os_showcase_md,
        taste_os_showcase_payload=_coerce_dict(safe_read_json(taste_os_showcase_json, default={})),
        control_room_json=control_room_json,
        control_room_md=control_room_md,
        control_room_payload=_coerce_dict(safe_read_json(control_room_json, default={})),
        creator_manifest_paths=creator_manifest_paths,
        creator_manifest_path=creator_manifest_path,
        creator_manifest=creator_manifest,
        creator_primary_report_path=creator_primary_report_path,
        creator_comparison_markdown_paths=creator_comparison_markdown_paths,
        creator_comparison_csv_paths=creator_comparison_csv_paths,
        research_claims_json=research_claims_json,
        research_claims_md=research_claims_md,
        research_claims_payload=_coerce_dict(safe_read_json(research_claims_json, default={})),
        benchmark_manifest_json=benchmark_manifest_json,
        benchmark_manifest_md=benchmark_manifest_md,
        benchmark_manifest_payload=benchmark_manifest_payload,
    )


def load_portfolio_artifact_bundle(output_dir: Path | str = "outputs", *, refresh: bool = False) -> PortfolioArtifactBundle:
    output_root = Path(output_dir).expanduser().resolve()
    if refresh:
        _load_portfolio_artifact_bundle_cached.cache_clear()
    return _load_portfolio_artifact_bundle_cached(str(output_root))


def clear_portfolio_artifact_bundle_cache() -> None:
    _load_portfolio_artifact_bundle_cached.cache_clear()
