"""Contracts for the warn-only docs ship-with-code check.

The check must warn when public modules change without docs and without a
visible ``Docs: N/A`` declaration, and it must exit 0 in every case.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "docs_ship_with_code.py"
TEMPLATE = REPO_ROOT / ".github" / "pull_request_template.md"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "unit-tests.yml"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("docs_ship_with_code", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warns_when_package_changes_without_docs_or_declaration(checker) -> None:
    assert checker.should_warn(["molcrys_kit/io/cif.py"], body="")
    assert checker.should_warn(["./molcrys_kit/cli.py"], body=None)
    assert checker.should_warn(
        ["molcrys_kit\\structures\\crystal.py"], body="see notes"
    )


def test_docs_change_or_declaration_suppresses_warning(checker) -> None:
    package = ["molcrys_kit/operations/surface.py"]
    assert not checker.should_warn(package + ["docs/api.md"], body="")
    assert not checker.should_warn(package, body="Docs: N/A — private helper rename.")
    assert not checker.should_warn(
        package,
        body="Intro\nDocs: N/A — test-only follow-up stays out of scope.\n",
    )


def test_declaration_inside_comment_or_fence_does_not_count(checker) -> None:
    package = ["molcrys_kit/__init__.py"]
    commented = "<!-- Docs: N/A — hidden -->\n- Docs paths touched:\n"
    fenced = "Example:\n```\nDocs: N/A — one-line reason\n```\n- Docs paths touched:\n"
    assert checker.should_warn(package, body=commented)
    assert checker.should_warn(package, body=fenced)
    visible = fenced + "\nDocs: N/A — pure refactor with identical public behavior.\n"
    assert not checker.should_warn(package, body=visible)


def test_non_package_paths_do_not_warn(checker) -> None:
    assert not checker.should_warn(["tests/unit/contracts/test_docs_api_sync.py"], "")
    assert not checker.should_warn(["docs/cli.md"], "")
    assert not checker.should_warn(["skills/example/SKILL.md"], None)
    assert not checker.should_warn([], "Docs: N/A — nothing changed")


def test_template_example_does_not_self_satisfy(checker) -> None:
    raw = TEMPLATE.read_text(encoding="utf-8")
    assert "## Summary" in raw
    assert "## How to verify" in raw
    assert "## Documentation" in raw
    assert "Docs paths touched:" in raw
    assert (
        "Public API / CLI / CIF-I/O contracts / crystallographic invariants changed (yes/no):"
        in raw
    )
    assert (
        "Agent guide (`AGENTS.md`) updated if a standing rule changed (yes/no):" in raw
    )
    assert "Project-agnostic:" in raw
    assert "internal project names" in raw
    assert "private paths" in raw
    assert "unpublished data or CIF files" in raw
    assert checker.NA_DECLARATION in raw
    assert checker.NA_DECLARATION not in checker.visible_pull_request_body(raw)


def test_workflow_job_is_warn_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "docs-ship-with-code:" in text
    assert "Docs ship with code (warn)" in text
    assert "continue-on-error: true" in text
    assert "docs_ship_with_code.py" in text
    job = text.split("docs-ship-with-code:", 1)[1].split("\n  unit-tests:", 1)[0]
    assert "continue-on-error: true" in job
    assert "python3 .github/scripts/docs_ship_with_code.py" in job


def test_main_always_exits_zero(
    checker, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = tmp_path / "files.txt"
    files.write_text("molcrys_kit/io/cif.py\n", encoding="utf-8")
    body = tmp_path / "body.md"
    body.write_text("No declaration here.\n", encoding="utf-8")

    warned = checker.main(["--files", str(files), "--body-file", str(body)])
    captured = capsys.readouterr()
    assert warned == 0
    assert "::warning::" in captured.out
    assert "does not fail the build" in captured.out

    body.write_text("Docs: N/A — CI-only helper.\n", encoding="utf-8")
    quiet = checker.main(["--files", str(files), "--body-file", str(body)])
    captured = capsys.readouterr()
    assert quiet == 0
    assert "::warning::" not in captured.out
    assert "no warning" in captured.out


def test_missing_revision_does_not_fail(
    checker, capsys: pytest.CaptureFixture[str]
) -> None:
    code = checker.main(["--base", "not-a-rev", "--head", "HEAD"])
    captured = capsys.readouterr()
    assert code == 0
    assert "::notice::" in captured.out
    assert "::warning::" not in captured.out


def test_script_entrypoint_exits_zero(tmp_path: Path) -> None:
    files = tmp_path / "files.txt"
    files.write_text("tests/unit/test_bonds.py\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--files", str(files)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "no warning" in completed.stdout
