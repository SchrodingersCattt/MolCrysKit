#!/usr/bin/env python3
"""Warn when molcrys_kit changes without docs or a Docs: N/A declaration.

Warn-only: this process always exits 0. A missing declaration prints a
GitHub Actions warning annotation and does not fail the build.

The pull request template keeps an example declaration inside a fenced block.
HTML comments and fenced code blocks are ignored so that example does not
satisfy the check.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

NA_DECLARATION = "Docs: N/A"
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def visible_pull_request_body(body: str | None) -> str:
    """Drop HTML comments and fenced code blocks from a pull request body."""
    if not body:
        return ""
    without_comments = _COMMENT_RE.sub("", body)
    return _FENCE_RE.sub("", without_comments)


def has_docs_na_declaration(body: str | None) -> bool:
    """Return whether the visible body contains the out-of-scope declaration."""
    return NA_DECLARATION in visible_pull_request_body(body)


def normalize_path(path: str) -> str:
    """Normalize a git or API path to a repo-relative forward-slash path."""
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def is_public_module_path(path: str) -> bool:
    """Return whether the path is the public package tree under molcrys_kit/."""
    normalized = normalize_path(path)
    return normalized == "molcrys_kit" or normalized.startswith("molcrys_kit/")


def is_docs_path(path: str) -> bool:
    """Return whether the path is under the stable docs/ tree."""
    normalized = normalize_path(path)
    return normalized == "docs" or normalized.startswith("docs/")


def should_warn(paths: Iterable[str], body: str | None) -> bool:
    """Return whether this change set should produce a non-blocking warning.

    Warn only when a public module changed, no docs file changed, and the
    pull request body lacks ``Docs: N/A`` outside comments and code fences.
    """
    public_changed = False
    docs_changed = False
    for path in paths:
        if is_public_module_path(path):
            public_changed = True
        if is_docs_path(path):
            docs_changed = True
    return public_changed and not docs_changed and not has_docs_na_declaration(body)


def warning_message(paths: Sequence[str]) -> str:
    """Build a single-line warning. Paths are limited so logs stay readable."""
    changed = [normalize_path(path) for path in paths if is_public_module_path(path)]
    shown = ", ".join(changed[:20])
    if len(changed) > 20:
        shown = f"{shown}, ..."
    return (
        "molcrys_kit/ changed, no docs/** file changed, and the pull request "
        "body has no Docs: N/A declaration outside comments and code fences. "
        "In-scope public API, CLI, CIF/I-O, or crystallographic-invariant "
        "changes need a stable docs update in this pull request "
        "(docs/api.md, docs/cli.md, docs/architecture.md, docs/bonds.md, "
        "docs/tutorials.md, or a focused design note). "
        "Out-of-scope changes should add a Docs: N/A line and a one-line reason. "
        "See CONTRIBUTING.md. This check does not fail the build. "
        f"Changed: {shown}."
    )


def emit_annotation(level: str, message: str) -> None:
    """Print a GitHub Actions annotation plus the same text as a plain line."""
    flat = " ".join(message.split())
    print(f"::{level}::{flat}")
    print(flat)


def changed_files_between(base: str, head: str) -> list[str]:
    """List paths that differ between two git revisions."""
    completed = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _read_paths(args: argparse.Namespace) -> list[str]:
    if args.files is not None:
        text = Path(args.files).read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]
    return changed_files_between(args.base, args.head)


def _read_body(args: argparse.Namespace) -> str | None:
    if args.body_file is not None:
        return Path(args.body_file).read_text(encoding="utf-8")
    return os.environ.get("PR_BODY")


def run_check(args: argparse.Namespace) -> None:
    """Evaluate one pull request and annotate when the policy heuristic hits."""
    if args.files is None and not (args.base and args.head):
        print("Docs ship-with-code check skipped: no changed-file input.")
        return
    paths = _read_paths(args)
    body = _read_body(args)
    if should_warn(paths, body):
        emit_annotation("warning", warning_message(paths))
        return
    print("Docs ship-with-code check: no warning.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        type=Path,
        help="Newline-separated changed paths. Skips git.",
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        help="Pull request body. Overrides the PR_BODY environment variable.",
    )
    parser.add_argument("--base", help="Git revision for the start of the diff.")
    parser.add_argument("--head", help="Git revision for the end of the diff.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the check. Always returns 0 so the workflow job stays green."""
    args = build_parser().parse_args(argv)
    try:
        run_check(args)
    except Exception as exc:
        emit_annotation("notice", f"Docs ship-with-code check skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
