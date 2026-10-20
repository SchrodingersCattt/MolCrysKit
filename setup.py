"""Setuptools hook that copies the canonical skills tree into the wheel."""

from pathlib import Path
from shutil import copytree, rmtree

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildBundledSkills(build_py):
    def build_package_data(self):
        source_root = Path(__file__).parent / "skills"
        package_root = Path(self.get_package_dir("molcrys_kit._bundled_skills"))
        generated: list[Path] = []
        for source in sorted(source_root.iterdir()):
            if not source.is_dir() or not (source / "SKILL.md").is_file():
                continue
            target = package_root / source.name
            copytree(source, target, dirs_exist_ok=True)
            generated.append(target)
        try:
            super().build_package_data()
        finally:
            for target in generated:
                rmtree(target, ignore_errors=True)


setup(cmdclass={"build_py": BuildBundledSkills})
