"""Install the optional native Git hook without replacing an existing hook."""

import shutil
import stat
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    custom_hooks = subprocess.run(
        ("git", "config", "--get", "core.hooksPath"),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if custom_hooks.returncode == 0:
        raise SystemExit("Custom core.hooksPath is set; install the hook manually.")

    target = Path(
        subprocess.check_output(
            ("git", "rev-parse", "--git-path", "hooks/post-commit"),
            cwd=root,
            text=True,
        ).strip()
    )
    if not target.is_absolute():
        target = root / target
    if target.exists() or target.is_symlink():
        raise SystemExit(f"Existing post-commit hook not replaced: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / ".githooks" / "post-commit", target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed post-commit hook: {target}")


if __name__ == "__main__":
    main()
