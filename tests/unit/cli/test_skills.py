from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from click.testing import CliRunner
import pytest

from molcrys_kit.cli import main
from molcrys_kit.cli import skill_viewer


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_skills_list_json() -> None:
    result = CliRunner().invoke(main, ["skills", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "mck.skills.list/v1"
    assert [item["name"] for item in payload["skills"]] == [
        "analyze-molecular-crystal",
        "operate-molecular-crystal",
        "query-csd-structures",
    ]


def test_skills_show_is_raw_and_file_is_explicit() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["skills", "show", "analyze-molecular-crystal"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("---\nname: analyze-molecular-crystal\n")
    assert "# Analyze Molecular Crystals" in result.output
    assert "# Operate Molecular Crystals" not in result.output

    result = runner.invoke(
        main,
        [
            "skills",
            "show",
            "analyze-molecular-crystal",
            "--file",
            "references/cli-analyses.md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "# CLI Analyses" in result.output


def test_skills_tree_has_links_without_bodies() -> None:
    result = CliRunner().invoke(
        main, ["skills", "tree", "query-csd-structures", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "mck.skills.tree/v1"
    assert any(
        resource["path"] == "scripts/demo_csd_disorder_workflow.py"
        for resource in payload["resources"]
    )
    assert all("content" not in resource for resource in payload["resources"])


def test_skills_reject_escape() -> None:
    result = CliRunner().invoke(
        main,
        ["skills", "show", "analyze-molecular-crystal", "--file", "../SKILL.md"],
    )
    assert result.exit_code != 0
    assert "escapes" in result.output


def test_skills_json_errors_are_machine_readable() -> None:
    result = CliRunner().invoke(
        main, ["skills", "show", "missing-skill", "--json"]
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload == {
        "error": "unknown skill: 'missing-skill'",
        "ok": False,
        "schema": "mck.error/v1",
    }


def test_skills_argument_does_not_hide_existing_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "molcrys_kit", "io", "info", "skills"],
        cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert "No such command 'info'" not in result.stdout + result.stderr


def test_skill_links_are_nested_deduplicated_and_cycle_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "skills"
    skill_root = root / "demo"
    (skill_root / "references" / "nested").mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo\n---\n"
        "[one](references/one.md#part) [again](references/one.md) "
        "[script](scripts/tool.py)\n",
        encoding="utf-8",
    )
    (skill_root / "references/one.md").write_text(
        "[nested](nested/two.md) [cycle](../SKILL.md)\n", encoding="utf-8"
    )
    (skill_root / "references/nested/two.md").write_text("# Nested\n", encoding="utf-8")
    (skill_root / "scripts").mkdir()
    (skill_root / "scripts/tool.py").write_text("raise SystemExit\n", encoding="utf-8")
    monkeypatch.setattr(skill_viewer, "_skills_root", lambda: root)

    skill = skill_viewer.get_skill("demo")
    assert [row["path"] for row in skill_viewer.tree(skill)] == [
        "SKILL.md",
        "references/one.md",
        "references/nested/two.md",
        "scripts/tool.py",
    ]
    assert [document["path"] for document in skill_viewer.all_documents(skill)] == [
        "SKILL.md",
        "references/one.md",
        "references/nested/two.md",
    ]


def test_skill_tree_rejects_broken_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "skills"
    skill_root = root / "demo"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text("[missing](references/nope.md)\n", encoding="utf-8")
    monkeypatch.setattr(skill_viewer, "_skills_root", lambda: root)
    with pytest.raises(skill_viewer.SkillError, match="resource not found"):
        skill_viewer.tree(skill_viewer.get_skill("demo"))
