from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from click.testing import CliRunner
import pytest

from molcrys_kit.cli import main
from molcrys_kit.cli import skill_viewer


REPO_ROOT = Path(__file__).resolve().parents[3]


def _bundled_skill_names(runner: CliRunner) -> list[str]:
    listed = runner.invoke(main, ["skills", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    return [item["name"] for item in json.loads(listed.output)["skills"]]


def _aggregate_tree_blocks(text: str) -> list[str]:
    return [block for block in text.strip().split("\n\n") if block.strip()]


def _first_markdown_heading_from_show(runner: CliRunner, name: str) -> str:
    shown = runner.invoke(main, ["skills", "show", name])
    assert shown.exit_code == 0, shown.output
    body = shown.output.split("---", 2)[-1]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped
    pytest.fail(f"no markdown heading in bundled skill {name!r}")


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


def test_skills_tree_without_name_reports_usage_with_choices() -> None:
    runner = CliRunner()
    names = _bundled_skill_names(runner)

    text = runner.invoke(main, ["skills", "tree"])
    assert text.exit_code == 2, text.output
    assert "skills tree requires SKILL_NAME or --all" in text.output
    assert "Missing argument" not in text.output
    for name in names:
        assert name in text.output
    assert "mck skills tree --all" in text.output
    assert "SKILL.md (markdown)" not in text.output

    err_json = runner.invoke(main, ["skills", "tree", "--json"])
    assert err_json.exit_code == 2
    payload = json.loads(err_json.stdout)
    assert payload == {
        "choices": names,
        "error": "skills tree requires SKILL_NAME or --all",
        "ok": False,
        "schema": "mck.error/v1",
    }


def test_skills_tree_all_shows_aggregate() -> None:
    runner = CliRunner()
    names = _bundled_skill_names(runner)
    listed = json.loads(runner.invoke(main, ["skills", "list", "--json"]).output)

    text = runner.invoke(main, ["skills", "tree", "--all"])
    assert text.exit_code == 0, text.output
    blocks = _aggregate_tree_blocks(text.output)
    assert [block.splitlines()[0] for block in blocks] == names
    for block in blocks:
        indented = [line for line in block.splitlines()[1:] if line.startswith("  ")]
        assert indented, block
    for item in listed["skills"]:
        assert item["description"] not in text.output
    for name in names:
        assert _first_markdown_heading_from_show(runner, name) not in text.output

    payload = json.loads(runner.invoke(main, ["skills", "tree", "--all", "--json"]).output)
    assert payload["schema"] == "mck.skills.trees/v1"
    assert [entry["skill"] for entry in payload["trees"]] == names
    for entry in payload["trees"]:
        assert any(r["path"] == "SKILL.md" for r in entry["resources"])
        assert all("content" not in resource for resource in entry["resources"])


def test_skills_tree_rejects_name_with_all() -> None:
    runner = CliRunner()
    text = runner.invoke(main, ["skills", "tree", "analyze-molecular-crystal", "--all"])
    assert text.exit_code == 2
    assert "not both" in text.output

    err_json = runner.invoke(
        main, ["skills", "tree", "analyze-molecular-crystal", "--all", "--json"]
    )
    assert err_json.exit_code == 2
    assert json.loads(err_json.stdout) == {
        "error": "skills tree accepts SKILL_NAME or --all, not both",
        "ok": False,
        "schema": "mck.error/v1",
    }


def test_skills_group_help_documents_tree_invocation() -> None:
    result = CliRunner().invoke(main, ["skills", "--help"])
    assert result.exit_code == 0, result.output
    assert "tree --all" in result.output
    assert "skill name" in result.output.lower()


def test_skills_tree_help_documents_all_option() -> None:
    result = CliRunner().invoke(main, ["skills", "tree", "--help"])
    assert result.exit_code == 0, result.output
    assert "--all" in result.output
    assert "SKILL_NAME" in result.output
    assert "aggregate" in result.output.lower() or "every" in result.output.lower()


def test_skills_tree_unknown_skill_still_fails() -> None:
    runner = CliRunner()
    text = runner.invoke(main, ["skills", "tree", "missing-skill"])
    assert text.exit_code == 2
    assert "unknown skill" in text.output

    err_json = runner.invoke(main, ["skills", "tree", "missing-skill", "--json"])
    assert err_json.exit_code == 2
    assert json.loads(err_json.stdout) == {
        "error": "unknown skill: 'missing-skill'",
        "ok": False,
        "schema": "mck.error/v1",
    }


def test_skills_reject_escape() -> None:
    result = CliRunner().invoke(
        main,
        ["skills", "show", "analyze-molecular-crystal", "--file", "../SKILL.md"],
    )
    assert result.exit_code == 2
    assert "escapes" in result.output


def test_skills_json_errors_are_machine_readable() -> None:
    result = CliRunner().invoke(main, ["skills", "show", "missing-skill", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload == {
        "error": "unknown skill: 'missing-skill'",
        "ok": False,
        "schema": "mck.error/v1",
    }


def test_skills_text_errors_also_exit_two() -> None:
    result = CliRunner().invoke(main, ["skills", "show", "missing-skill"])
    assert result.exit_code == 2
    assert "unknown skill" in result.output


def test_unknown_subcommand_suggests_without_traceback() -> None:
    runner = CliRunner()
    nested = runner.invoke(main, ["io", "inf"])
    assert nested.exit_code == 2, nested.output
    assert "No such command 'inf'" in nested.output
    assert "Did you mean 'info'" in nested.output
    assert "AttributeError" not in nested.output
    assert "Traceback" not in nested.output

    root = runner.invoke(main, ["nosuch"])
    assert root.exit_code == 2, root.output
    assert "No such command 'nosuch'" in root.output
    assert "AttributeError" not in root.output
    assert "Traceback" not in root.output


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


def _run_isolated_cli(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )


def test_host_skills_argv_does_not_drop_structure_commands() -> None:
    """Importing under a host argv of ``skills`` must not hide structure commands.

    The documentation path stays free of chemistry imports until a structure
    command is actually looked up.
    """
    result = _run_isolated_cli(
        textwrap.dedent(
            """
            import sys
            sys.argv = ["host.py", "skills"]
            from click.testing import CliRunner
            from molcrys_kit.cli import io_group, main

            heavy = (
                "molcrys_kit.cli.io_cmd",
                "molcrys_kit.cli.operate_cmd",
                "molcrys_kit.cli.analyze_cmd",
                "molcrys_kit.cli.periodic_chain_cmd",
            )
            assert all(name not in sys.modules for name in heavy)

            listed = CliRunner().invoke(main, ["skills", "list"])
            assert listed.exit_code == 0, listed.output
            skills_help = CliRunner().invoke(main, ["skills", "--help"])
            assert skills_help.exit_code == 0, skills_help.output
            assert "list" in skills_help.output
            assert all(name not in sys.modules for name in heavy), sorted(sys.modules)

            names = sorted(io_group.commands)
            assert names == ["convert", "extract-molecule", "info", "molecules"], names
            assert "molcrys_kit.cli.io_cmd" in sys.modules

            help_result = CliRunner().invoke(main, ["io", "--help"])
            assert help_result.exit_code == 0, help_result.output
            for name in names:
                assert name in help_result.output, help_result.output

            root_help = CliRunner().invoke(main, ["--help"])
            assert root_help.exit_code == 0, root_help.output
            assert "build" in root_help.output
            assert "validate-periodic-bundle" in root_help.output
            """
        )
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_host_option_skills_token_does_not_drop_structure_commands() -> None:
    """An option value of ``skills`` in the host argv is not the mck subcommand."""
    result = _run_isolated_cli(
        textwrap.dedent(
            """
            import sys
            sys.argv = ["host.py", "--opt", "skills", "io"]
            from click.testing import CliRunner
            from molcrys_kit.cli import main

            result = CliRunner().invoke(main, ["io", "--help"])
            assert result.exit_code == 0, result.output
            for name in ("convert", "extract-molecule", "info", "molecules"):
                assert name in result.output, result.output
            """
        )
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    (skill_root / "SKILL.md").write_text(
        "[missing](references/nope.md)\n", encoding="utf-8"
    )
    monkeypatch.setattr(skill_viewer, "_skills_root", lambda: root)
    with pytest.raises(skill_viewer.SkillError, match="resource not found"):
        skill_viewer.tree(skill_viewer.get_skill("demo"))


def test_skill_show_ignores_code_links_and_allows_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "skills"
    skill_root = root / "demo"
    (skill_root / "references").mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo\n---\n"
        "See [refs](references/) and `not-a-link](missing.md)`.\n"
        "```\n[example](references/missing.md)\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_viewer, "_skills_root", lambda: root)

    skill = skill_viewer.get_skill("demo")
    skill_viewer.emit_show(skill, "SKILL.md", all_files=False, as_json=False)
    rows = skill_viewer.tree(skill)
    assert any(row["path"] == "references" and row["kind"] == "directory" for row in rows)
    assert all(row["path"] != "references/missing.md" for row in rows)
