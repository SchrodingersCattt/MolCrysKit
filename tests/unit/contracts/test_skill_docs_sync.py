"""Docs-as-code contracts for bundled agent skills.

These tests fail when skill examples refer to removed CLI options, import names
outside the documented public API, or contain broken/local orphan assets.
"""

from __future__ import annotations

import ast
import importlib
import re
import shlex
from collections.abc import Iterable
from pathlib import Path

import click
import pytest

from molcrys_kit.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = REPO_ROOT / "skills"
DOCS_CLI = REPO_ROOT / "docs" / "cli.md"


def _markdown_files() -> list[Path]:
    return sorted(SKILLS_ROOT.glob("**/*.md"))


def _fenced_blocks(text: str, languages: set[str]) -> Iterable[str]:
    pattern = re.compile(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(text):
        if match.group(1).lower() in languages:
            yield match.group(2)


def _shell_commands(block: str) -> Iterable[list[str]]:
    logical_lines: list[str] = []
    pending = ""
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        logical_lines.append(pending + line)
        pending = ""
    if pending:
        logical_lines.append(pending)

    for line in logical_lines:
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError as exc:
            pytest.fail(f"Invalid shell example {line!r}: {exc}")
        if tokens and tokens[0] == "mck":
            yield tokens


def _click_command(tokens: list[str]) -> tuple[click.Command, list[str], str]:
    command: click.Command = cli_main
    consumed = ["mck"]
    index = 1
    while isinstance(command, click.Group) and index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            break
        ctx = click.Context(command)
        child = command.get_command(ctx, token)
        if child is None:
            break
        command = child
        consumed.append(token)
        index += 1
    return command, tokens[index:], " ".join(consumed)


def _validate_option_tokens(command: click.Command, args: list[str], path: str) -> None:
    option_map: dict[str, click.Option] = {}
    for param in command.params:
        if isinstance(param, click.Option):
            for option in (*param.opts, *param.secondary_opts):
                option_map[option] = param

    index = 0
    while index < len(args):
        token = args[index]
        if token in {"--help", "-h"}:
            index += 1
            continue
        if token == "--":
            return
        if not token.startswith("-") or token == "-":
            index += 1
            continue

        option_name, separator, _inline_value = token.partition("=")
        option = option_map.get(option_name)
        assert option is not None, f"Unknown option {option_name!r} for {path}"
        if separator or option.is_flag:
            index += 1
            continue
        index += 1 + option.nargs


def _module_all(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    exported = getattr(module, "__all__", None)
    assert (
        exported is not None
    ), f"Skill examples may import from {module_name} only when it defines __all__"
    return set(exported)


def test_skill_frontmatter_links_assets_and_orphans() -> None:
    assert SKILLS_ROOT.is_dir(), "skills/ directory is missing"
    for skill_file in sorted(SKILLS_ROOT.glob("*/SKILL.md")):
        skill_dir = skill_file.parent
        text = skill_file.read_text(encoding="utf-8")
        frontmatter = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        assert frontmatter, f"Missing YAML frontmatter: {skill_file}"
        assert re.search(
            rf"(?m)^name:\s*{re.escape(skill_dir.name)}\s*$",
            frontmatter.group(1),
        ), f"Skill name must match directory: {skill_file}"
        assert re.search(r"(?m)^description:\s*.+$", frontmatter.group(1))

        linked_files: set[Path] = set()
        for markdown_file in sorted(skill_dir.glob("**/*.md")):
            markdown = markdown_file.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown):
                if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith(
                    "#"
                ):
                    continue
                relative_target = target.split("#", 1)[0]
                resolved = (markdown_file.parent / relative_target).resolve()
                assert resolved.is_relative_to(
                    skill_dir.resolve()
                ), f"Skill link escapes its installable directory: {markdown_file} -> {target}"
                assert (
                    resolved.exists()
                ), f"Broken skill link: {markdown_file} -> {target}"
                if resolved.is_file():
                    linked_files.add(resolved)

        expected_assets = {
            path.resolve()
            for pattern in ("references/*", "scripts/*")
            for path in skill_dir.glob(pattern)
            if path.is_file()
        }
        assert expected_assets <= linked_files, (
            f"Orphan skill assets in {skill_dir}: "
            f"{sorted(str(path) for path in expected_assets - linked_files)}"
        )


def test_molecular_crystal_skill_routes_are_exclusive() -> None:
    operate = (SKILLS_ROOT / "operate-molecular-crystal" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    analyze = (SKILLS_ROOT / "analyze-molecular-crystal" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "Not for analysis or rendering" in operate
    assert "Inspect once" not in operate
    assert "mck analyze summary INPUT --json" not in operate
    assert "Read exactly one operation page" in operate
    assert "Never for drawings, trajectories, or vibrations" in analyze
    assert "full audit" in analyze
    assert len(operate.splitlines()) <= 45
    assert len(analyze.splitlines()) <= 45


def test_skill_mck_examples_match_click_contract() -> None:
    for markdown_file in _markdown_files():
        for block in _fenced_blocks(
            markdown_file.read_text(encoding="utf-8"), {"bash", "sh", "shell"}
        ):
            for tokens in _shell_commands(block):
                command, args, path = _click_command(tokens)
                if path == "mck":
                    assert args in (
                        ["--help"],
                        ["-h"],
                        ["--version"],
                    ), f"Unknown mck command in {markdown_file}: {tokens}"
                    continue
                if isinstance(command, click.Group):
                    assert args in (
                        ["--help"],
                        ["-h"],
                    ), f"Incomplete mck command in {markdown_file}: {tokens}"
                    continue
                _validate_option_tokens(command, args, path)


def test_docs_cli_lists_full_paths_and_options() -> None:
    text = DOCS_CLI.read_text(encoding="utf-8")

    def walk(
        group: click.Group, prefix: tuple[str, ...]
    ) -> Iterable[tuple[str, click.Command]]:
        ctx = click.Context(group)
        for name in group.list_commands(ctx):
            command = group.get_command(ctx, name)
            assert command is not None
            path = (*prefix, name)
            if isinstance(command, click.Group):
                yield from walk(command, path)
            else:
                yield " ".join(path), command

    for path, command in walk(cli_main, ("mck",)):
        assert path in text, f"Full CLI path missing from docs/cli.md: {path}"
        for param in command.params:
            if not isinstance(param, click.Option):
                continue
            long_options = [
                option
                for option in (*param.opts, *param.secondary_opts)
                if option.startswith("--")
            ]
            for option in long_options:
                assert (
                    option in text
                ), f"CLI option missing from docs/cli.md: {path} {option}"


def test_skill_python_imports_use_public_api() -> None:
    for markdown_file in _markdown_files():
        for block in _fenced_blocks(
            markdown_file.read_text(encoding="utf-8"), {"python", "py"}
        ):
            tree = ast.parse(block, filename=str(markdown_file))
            for node in ast.walk(tree):
                if (
                    not isinstance(node, ast.ImportFrom)
                    or node.level
                    or not node.module
                ):
                    continue
                if node.module != "molcrys_kit" and not node.module.startswith(
                    "molcrys_kit."
                ):
                    continue
                assert not any(
                    part.startswith("_") for part in node.module.split(".")
                ), f"Private module imported in {markdown_file}: {node.module}"
                exported = _module_all(node.module)
                for alias in node.names:
                    assert alias.name != "*", f"Wildcard import in {markdown_file}"
                    assert alias.name in exported, (
                        f"Non-public API in {markdown_file}: "
                        f"from {node.module} import {alias.name}"
                    )


def test_skill_python_scripts_compile() -> None:
    for script in sorted(SKILLS_ROOT.glob("*/scripts/*.py")):
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
