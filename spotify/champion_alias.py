from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class ChampionAlias:
    run_id: str
    run_dir: Path
    model_name: str
    promoted_at: str


def _to_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def best_deep_model_name(result_rows: list[dict[str, object]]) -> str | None:
    best_name = ""
    best_score = float("-inf")
    for row in result_rows:
        if str(row.get("model_type", "")).strip().lower() != "deep":
            continue
        model_name = str(row.get("model_name", "")).strip()
        if not model_name:
            continue
        score = _to_float(row.get("val_top1"))
        if math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_name = model_name
    return best_name or None


def champion_alias_file(output_dir: Path) -> Path:
    return output_dir / "models" / "champion" / "alias.json"


def default_champion_alias_file(project_root: Path | None = None) -> Path:
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    return champion_alias_file(root / "outputs")


def write_champion_alias(
    *,
    output_dir: Path,
    run_id: str,
    run_dir: Path,
    model_name: str,
) -> Path:
    alias_file = champion_alias_file(output_dir)
    alias_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": str(run_id).strip(),
        "run_dir": str(run_dir.expanduser().resolve()),
        "model_name": str(model_name).strip(),
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    alias_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return alias_file


def read_champion_alias(alias_file: Path) -> ChampionAlias | None:
    path = alias_file.expanduser().resolve()
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Champion alias file is not valid JSON: {path}") from exc

    run_id = str(payload.get("run_id", "")).strip()
    run_dir_raw = str(payload.get("run_dir", "")).strip()
    model_name = str(payload.get("model_name", "")).strip()
    promoted_at = str(payload.get("promoted_at", "")).strip()
    if not run_id or not run_dir_raw or not model_name:
        raise RuntimeError(f"Champion alias file is missing required fields: {path}")

    return ChampionAlias(
        run_id=run_id,
        run_dir=Path(run_dir_raw).expanduser().resolve(),
        model_name=model_name,
        promoted_at=promoted_at,
    )


def resolve_prediction_run_dir(
    run_dir_arg: str | None,
    *,
    project_root: Path | None = None,
) -> tuple[Path, str | None]:
    if run_dir_arg:
        requested = Path(run_dir_arg).expanduser().resolve()
        if not requested.exists():
            raise FileNotFoundError(f"Run directory does not exist: {requested}")
        alias = read_champion_alias(requested / "alias.json") if requested.is_dir() else None
        if alias is None:
            return requested, None
        if not alias.run_dir.exists():
            raise FileNotFoundError(
                f"Champion alias points to a missing run directory: {alias.run_dir}"
            )
        return alias.run_dir, alias.model_name

    alias_file = default_champion_alias_file(project_root)
    alias = read_champion_alias(alias_file)
    if alias is None:
        raise FileNotFoundError(
            "Champion alias not found. Pass --run-dir explicitly or run a promoted training run first."
        )
    if not alias.run_dir.exists():
        raise FileNotFoundError(
            f"Champion alias points to a missing run directory: {alias.run_dir}"
        )
    return alias.run_dir, alias.model_name
