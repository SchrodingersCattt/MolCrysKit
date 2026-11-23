"""Assert the wheel packages the full canonical skills tree."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = REPO_ROOT / "skills"


def _assert_import_resolves_under_site(
    site: Path, env: dict[str, str], cwd: Path
) -> Path:
    """Fail unless ``import molcrys_kit`` loads from the wheel ``--target`` tree."""
    probe = subprocess.run(
        [sys.executable, "-c", "import molcrys_kit; print(molcrys_kit.__file__)"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=cwd,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    resolved = Path(probe.stdout.strip()).resolve()
    site_root = site.resolve()
    assert site_root in resolved.parents, (
        f"molcrys_kit loaded from {resolved}, expected under install site {site_root}"
    )
    checkout_init = (REPO_ROOT / "molcrys_kit" / "__init__.py").resolve()
    assert resolved != checkout_init, "wheel smoke must not resolve the source checkout"
    return resolved


def _canonical_skill_files() -> set[str]:
    files: set[str] = set()
    for path in SKILLS_ROOT.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(SKILLS_ROOT).as_posix()
        files.add(relative)
    return files


@pytest.mark.timeout(240)
def test_wheel_contains_full_bundled_skills_tree() -> None:
    """Installed wheels must mirror skills/** under molcrys_kit/_bundled_skills."""
    # Require the real PyPA builder. A bare ``import build`` can succeed against a
    # local setuptools ``build/`` artifact directory that has no ``__main__``.
    pytest.importorskip("build.__main__")
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
                "*.pyo",
                ".pytest_cache",
                "molcrys_kit/_version.py",
            ),
            symlinks=True,
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        # Editable installs put the project root on sys.path. A leftover
        # setuptools ``build/`` directory there shadows PyPA ``build`` for
        # ``python -m build``. Drop those path entries before launching.
        launcher = textwrap.dedent(
            f"""
            import sys
            from pathlib import Path

            cleaned = []
            for entry in sys.path:
                if entry:
                    local_build = Path(entry) / "build"
                    if local_build.is_dir() and not (local_build / "__main__.py").is_file():
                        continue
                cleaned.append(entry)
            sys.path[:] = cleaned
            sys.argv = ["build", "--wheel", "--outdir", {str(dist)!r}]
            from build.__main__ import entrypoint
            entrypoint()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", launcher],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        wheels = sorted(dist.glob("*.whl"))
        assert len(wheels) == 1, list(dist.iterdir())
        prefix = "molcrys_kit/_bundled_skills/"
        with zipfile.ZipFile(wheels[0]) as archive:
            names = archive.namelist()
            packaged = {
                name[len(prefix) :]
                for name in names
                if name.startswith(prefix)
                and not name.endswith("/")
                and not name.endswith("__init__.py")
            }
            bytecode = [
                name
                for name in names
                if "__pycache__" in name or name.endswith((".pyc", ".pyo"))
            ]
        assert not bytecode, bytecode
        assert packaged == canonical, sorted(canonical.symmetric_difference(packaged))

        # The build must not leave skill trees in the package source directory.
        leftover = work / "molcrys_kit" / "_bundled_skills"
        assert leftover.is_dir()
        assert sorted(p.name for p in leftover.iterdir()) == ["__init__.py"]

        site = Path(tmp) / "site"
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                str(site),
                str(wheels[0]),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert install.returncode == 0, install.stdout + install.stderr
        # Host site-packages supplies core runtime deps; only the wheel payload
        # lands in ``site``. Run CLI probes outside the repo so ``sys.path[0]``
        # cannot pick up ``./molcrys_kit`` from the checkout.
        isolated_env = dict(env)
        isolated_env["PYTHONPATH"] = str(site)
        cli_cwd = Path(tmp) / "cli_cwd"
        cli_cwd.mkdir()
        _assert_import_resolves_under_site(site, isolated_env, cli_cwd)

        listed = subprocess.run(
            [sys.executable, "-m", "molcrys_kit", "skills", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
            env=isolated_env,
            cwd=cli_cwd,
        )
        assert listed.returncode == 0, listed.stdout + listed.stderr
        names = [item["name"] for item in json.loads(listed.stdout)["skills"]]

        missing = subprocess.run(
            [sys.executable, "-m", "molcrys_kit", "skills", "tree"],
            capture_output=True,
            text=True,
            check=False,
            env=isolated_env,
            cwd=cli_cwd,
        )
        assert missing.returncode == 2, missing.stdout + missing.stderr
        assert "SKILL_NAME or --all" in missing.stderr
        assert "Missing argument" not in missing.stderr

        tree = subprocess.run(
            [sys.executable, "-m", "molcrys_kit", "skills", "tree", "--all"],
            capture_output=True,
            text=True,
            check=False,
            env=isolated_env,
            cwd=cli_cwd,
        )
        assert tree.returncode == 0, tree.stdout + tree.stderr
        blocks = [b for b in tree.stdout.strip().split("\n\n") if b.strip()]
        assert [block.splitlines()[0] for block in blocks] == names
        assert "SKILL.md (markdown)" in tree.stdout
