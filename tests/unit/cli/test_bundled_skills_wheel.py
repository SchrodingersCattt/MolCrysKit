"""Assert the wheel packages the full canonical skills tree."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = REPO_ROOT / "skills"


def _canonical_skill_files() -> set[str]:
    files: set[str] = set()
    for path in SKILLS_ROOT.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(SKILLS_ROOT).as_posix()
        files.add(relative)
    return files


@pytest.mark.timeout(240)
def test_wheel_contains_full_bundled_skills_tree() -> None:
    """Installed wheels must mirror skills/** under molcrys_kit/_bundled_skills."""
    pytest.importorskip("build")
    canonical = _canonical_skill_files()
    assert canonical, "canonical skills tree is empty"

    with tempfile.TemporaryDirectory() as tmp:
        dist = Path(tmp) / "dist"
        dist.mkdir()
        # Keep the source checkout clean: build from a copy so interrupted
        # packaging cannot leave skill copies under molcrys_kit/_bundled_skills.
        work = Path(tmp) / "src"
        shutil.copytree(
            REPO_ROOT,
            work,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "venv",
                "build",
                "dist",
                "*.egg-info",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                "molcrys_kit/_version.py",
            ),
            symlinks=True,
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        wheels = list(dist.glob("molcrys_kit-*.whl"))
        assert len(wheels) == 1, wheels
        prefix = "molcrys_kit/_bundled_skills/"
        with zipfile.ZipFile(wheels[0]) as archive:
            packaged = {
                name[len(prefix) :]
                for name in archive.namelist()
                if name.startswith(prefix)
                and not name.endswith("/")
                and not name.endswith("__init__.py")
            }
        assert packaged == canonical, sorted(canonical.symmetric_difference(packaged))

        # The build must not leave skill trees in the package source directory.
        leftover = work / "molcrys_kit" / "_bundled_skills"
        assert leftover.is_dir()
        assert sorted(p.name for p in leftover.iterdir()) == ["__init__.py"]
