from __future__ import annotations

from pathlib import Path
import os


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_dotenv_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if value and len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    else:
        # Support inline comments for unquoted values.
        hash_pos = value.find(" #")
        if hash_pos >= 0:
            value = value[:hash_pos].rstrip()

    return key, value


def load_local_env(
    *,
    project_root: Path | None = None,
    filenames: tuple[str, ...] = (".env", ".env.local"),
    override: bool = False,
) -> dict[str, str]:
    root = (project_root or _default_project_root()).expanduser().resolve()
    loaded: dict[str, str] = {}

    for name in filenames:
        env_path = root / name
        if not env_path.exists() or not env_path.is_file():
            continue

        with env_path.open("r", encoding="utf-8") as infile:
            for raw_line in infile:
                parsed = _parse_dotenv_line(raw_line)
                if parsed is None:
                    continue
                key, value = parsed
                if override or key not in os.environ:
                    os.environ[key] = value
                    loaded[key] = value

    return loaded
