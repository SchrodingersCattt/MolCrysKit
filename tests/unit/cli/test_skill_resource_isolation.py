from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def test_source_checkout_wins_over_foreign_top_level_skills_package(tmp_path: Path) -> None:
    project_root = tmp_path / "molcryskit-source"
    foreign_root = tmp_path / "foreign-source"
    shutil.copytree(ROOT / "molcrys_kit", project_root / "molcrys_kit")
    shutil.copytree(ROOT / "skills", project_root / "skills")
    (project_root / "skills" / "__init__.py").write_text("", encoding="utf-8")
    (foreign_root / "skills" / "foreign").mkdir(parents=True)
    (foreign_root / "skills" / "__init__.py").write_text("", encoding="utf-8")
    (foreign_root / "skills" / "foreign" / "SKILL.md").write_text(
        "---\nname: foreign\ndescription: foreign\n---\n", encoding="utf-8"
    )

    script = """
import sys
sys.argv = ['mck', 'skills', 'list']
from molcrys_kit.cli.skill_viewer import list_skills
names = [item.name for item in list_skills()]
assert names == ['analyze-molecular-crystal', 'operate-molecular-crystal', 'query-csd-structures'], names
"""
    for roots in ((project_root, foreign_root), (foreign_root, project_root)):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(root) for root in roots)
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
