from __future__ import annotations

from spotify.env import load_local_env


def test_load_local_env_reads_dotenv_files(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "# comment",
                "SPOTIPY_CLIENT_ID=abc123",
                "SPOTIPY_CLIENT_SECRET='shh-secret'",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "export EXTRA_FLAG=yes",
                "SPOTIPY_CLIENT_ID=override-id",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("SPOTIPY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIPY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("EXTRA_FLAG", raising=False)

    loaded = load_local_env(project_root=tmp_path)

    assert loaded["SPOTIPY_CLIENT_ID"] == "abc123"
    assert loaded["SPOTIPY_CLIENT_SECRET"] == "shh-secret"
    assert loaded["EXTRA_FLAG"] == "yes"


def test_load_local_env_respects_existing_env_by_default(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("SPOTIPY_CLIENT_ID=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SPOTIPY_CLIENT_ID", "from-shell")

    loaded = load_local_env(project_root=tmp_path)

    assert "SPOTIPY_CLIENT_ID" not in loaded
