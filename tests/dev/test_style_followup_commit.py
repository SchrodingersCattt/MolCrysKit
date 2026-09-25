"""Integration tests for the optional Ruff follow-up commit."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "scripts" / "style_followup_commit.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    pytest.importorskip("ruff")
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Style Test")
    git(tmp_path, "config", "user.email", "style-test@example.com")
    return tmp_path


def commit(repo: Path, message: str) -> None:
    git(repo, "add", ".")
    git(repo, "commit", "-qm", message)


def run_hook(repo: Path) -> str:
    return subprocess.check_output((sys.executable, str(HOOK)), cwd=repo, text=True)


def test_unformatted_python_gets_separate_commit(repo: Path) -> None:
    source = repo / "package" / "module.py"
    source.parent.mkdir()
    source.write_text("value = dict( first=1, second=2 )\n", encoding="utf-8")
    commit(repo, "feat: add module")

    run_hook(repo)

    assert git(repo, "log", "-2", "--format=%s").splitlines() == [
        "style: format Python from previous commit",
        "feat: add module",
    ]
    assert source.read_text(encoding="utf-8") == "value = dict(first=1, second=2)\n"
    assert git(repo, "status", "--porcelain") == ""


def test_already_formatted_commit_gets_no_extra_commit(repo: Path) -> None:
    (repo / "module.py").write_text("value = 1\n", encoding="utf-8")
    commit(repo, "feat: add formatted module")

    run_hook(repo)

    assert git(repo, "rev-list", "--count", "HEAD") == "1"


def test_only_python_files_in_latest_commit_are_formatted(repo: Path) -> None:
    legacy = repo / "legacy.py"
    legacy.write_text("value = dict( first=1 )\n", encoding="utf-8")
    commit(repo, "feat: add legacy module")
    (repo / "new.py").write_text("value = dict( second=2 )\n", encoding="utf-8")
    commit(repo, "feat: add new module")

    run_hook(repo)

    assert legacy.read_text(encoding="utf-8") == "value = dict( first=1 )\n"
    assert (repo / "new.py").read_text(encoding="utf-8") == "value = dict(second=2)\n"


def test_dirty_tree_is_not_formatted_or_committed(repo: Path) -> None:
    source = repo / "module.py"
    source.write_text("value = dict( first=1, second=2 )\n", encoding="utf-8")
    commit(repo, "feat: add module")
    source.write_text(
        "value = dict( first=1, second=2 )\n# my pending edit\n", encoding="utf-8"
    )

    assert "skipped" in run_hook(repo)

    assert "# my pending edit" in source.read_text(encoding="utf-8")
    assert git(repo, "rev-list", "--count", "HEAD") == "1"


def install_test_hook(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    script = repo / "scripts" / "style_followup_commit.py"
    script.parent.mkdir()
    shutil.copyfile(HOOK, script)
    shutil.copyfile(
        HOOK.parent / "install_style_hook.py",
        repo / "scripts" / "install_style_hook.py",
    )
    hook_dir = repo / ".githooks"
    hook_dir.mkdir()
    shutil.copyfile(
        HOOK.parents[1] / ".githooks" / "post-commit", hook_dir / "post-commit"
    )
    subprocess.run(
        (sys.executable, str(repo / "scripts" / "install_style_hook.py")),
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_installed_post_commit_hook_creates_followup(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_test_hook(repo, monkeypatch)
    (repo / "module.py").write_text("value = dict( first=1 )\n", encoding="utf-8")

    commit(repo, "feat: add module")

    assert git(repo, "log", "-2", "--format=%s").splitlines() == [
        "style: format Python from previous commit",
        "feat: add module",
    ]
    assert git(repo, "status", "--porcelain") == ""


def test_installed_hook_preserves_unstaged_changes(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_test_hook(repo, monkeypatch)
    source = repo / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    commit(repo, "feat: initial module")
    previous_commits = int(git(repo, "rev-list", "--count", "HEAD"))
    source.write_text("value = dict( first=1 )\n", encoding="utf-8")
    git(repo, "add", "module.py")
    source.write_text("value = dict( first=1 )\n# pending edit\n", encoding="utf-8")

    git(repo, "commit", "-qm", "feat: change module")

    assert int(git(repo, "rev-list", "--count", "HEAD")) == previous_commits + 1
    assert source.read_text(encoding="utf-8").endswith("# pending edit\n")


def test_installer_does_not_replace_existing_hook(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_test_hook(repo, monkeypatch)
    installed_hook = repo / ".git" / "hooks" / "post-commit"
    original = installed_hook.read_bytes()

    result = subprocess.run(
        (sys.executable, str(repo / "scripts" / "install_style_hook.py")),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not replaced" in result.stderr
    assert installed_hook.read_bytes() == original


def test_installer_respects_custom_hooks_path(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(repo, "config", "core.hooksPath", "custom-hooks")
    script = repo / "scripts" / "install_style_hook.py"
    script.parent.mkdir()
    shutil.copyfile(HOOK.parent / "install_style_hook.py", script)

    result = subprocess.run(
        (sys.executable, str(script)),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Custom core.hooksPath" in result.stderr
    assert not (repo / "custom-hooks" / "post-commit").exists()


def test_installer_does_not_replace_dangling_hook_symlink(repo: Path) -> None:
    script = repo / "scripts" / "install_style_hook.py"
    script.parent.mkdir()
    shutil.copyfile(HOOK.parent / "install_style_hook.py", script)
    target = repo / ".git" / "hooks" / "post-commit"
    try:
        target.symlink_to(repo / "nonexistent-hook")
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this system")

    result = subprocess.run(
        (sys.executable, str(script)),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not replaced" in result.stderr
    assert target.is_symlink()
