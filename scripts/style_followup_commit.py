"""Format Python files from the last commit in a separate, optional commit."""

import os
import subprocess
import sys
from pathlib import Path


def git(*args: str) -> bytes:
    return subprocess.check_output(("git", *args))


def main() -> int:
    if os.environ.get("MCK_STYLE_FOLLOWUP"):
        return 0

    # Never incorporate or overwrite unrelated staged or working-tree changes.
    if git("status", "--porcelain", "--untracked-files=no"):
        print("Style follow-up skipped: tracked files have uncommitted changes.")
        return 0

    paths = [
        Path(os.fsdecode(path))
        for path in git(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "-z", "HEAD"
        ).split(b"\0")
        if path and path.endswith(b".py")
    ]
    paths = [path for path in paths if path.is_file() and not path.is_symlink()]
    if not paths:
        return 0

    subprocess.run(
        (sys.executable, "-m", "ruff", "format", "--", *(str(path) for path in paths)),
        check=True,
    )
    if not git("diff", "--name-only", "-z", "--", *(str(path) for path in paths)):
        return 0

    # The pre-commit checks still run on the follow-up; the guard prevents recursion.
    env = {**os.environ, "MCK_STYLE_FOLLOWUP": "1"}
    subprocess.run(
        ("git", "add", "--", *(str(path) for path in paths)), check=True, env=env
    )
    subprocess.run(
        ("git", "commit", "-m", "style: format Python from previous commit"),
        check=True,
        env=env,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
