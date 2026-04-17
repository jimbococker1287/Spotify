from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self) -> None:
        spotify_build_dir = Path(self.build_lib) / "spotify"
        if spotify_build_dir.exists():
            shutil.rmtree(spotify_build_dir)
        super().run()


setup(cmdclass={"build_py": build_py})
