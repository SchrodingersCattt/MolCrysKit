"""Setuptools hook that copies the canonical skills tree into the wheel."""

from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py


def _copy_skill_tree(source: Path, target: Path) -> None:
    """Copy a skill directory without following symlinks into regular files.

    Symlinks are skipped so a link that escapes the skill root cannot become a
    concrete file in the wheel (source checkouts reject such escapes at read time).
    Bytecode caches are omitted so a prior local import cannot pollute the wheel.
    """
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if (
            item.is_symlink()
            or item.name == "__pycache__"
            or item.name.endswith((".pyc", ".pyo"))
        ):
            continue
        destination = target / item.name
        if item.is_dir():
            _copy_skill_tree(item, destination)
        elif item.is_file():
            copy2(item, destination)


class BuildBundledSkills(build_py):
    def run(self):
        super().run()
        source_root = Path(__file__).parent / "skills"
        package_root = Path(self.build_lib) / "molcrys_kit" / "_bundled_skills"
        if not source_root.is_dir():
            return
        for source in sorted(source_root.iterdir()):
            if source.is_symlink() or not source.is_dir():
                continue
            if not (source / "SKILL.md").is_file():
                continue
            _copy_skill_tree(source, package_root / source.name)


setup(cmdclass={"build_py": BuildBundledSkills})
