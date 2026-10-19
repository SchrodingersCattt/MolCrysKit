"""Command line interface for MolCrysKit."""

from __future__ import annotations

import logging
import sys

import click

from .skill_viewer import (
    SkillError,
    emit_list,
    emit_show,
    emit_tree,
    get_skill,
    json_payload,
)

try:
    from molcrys_kit._version import version as __version__
except Exception:  # pragma: no cover - setuptools_scm fallback edge case
    __version__ = "0.0.0+unknown"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="mck")
@click.option("--verbose", is_flag=True, help="Show debug logging.")
@click.option("--quiet", is_flag=True, help="Only show warnings and errors.")
def main(verbose: bool = False, quiet: bool = False) -> None:
    """MolCrysKit command line tools."""
    if verbose and quiet:
        raise click.UsageError("Use at most one of --verbose and --quiet.")
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@main.group(name="io")
def io_group() -> None:
    """Read, summarize, and convert structures."""


@main.group(name="operate")
def operate_group() -> None:
    """Generate modified structures."""


@main.group(name="analyze")
def analyze_group() -> None:
    """Analyze crystals and print reports."""


def _skill_error(exc: SkillError, *, as_json: bool) -> None:
    if as_json:
        click.echo(f"Error: {exc}", err=True)
        click.echo(json_payload({"schema": "mck.error/v1", "ok": False, "error": str(exc)}))
        raise click.exceptions.Exit(2)
    raise click.ClickException(str(exc)) from exc


@main.group(name="skills")
def skills_group() -> None:
    """Discover and read bundled Agent Skills."""


@skills_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def skills_list(as_json: bool) -> None:
    """List bundled skills and frontmatter descriptions."""
    try:
        emit_list(as_json=as_json)
    except SkillError as exc:
        _skill_error(exc, as_json=as_json)


@skills_group.command(name="show")
@click.argument("name", metavar="SKILL_NAME")
@click.option("--file", "relative_path", default="SKILL.md", show_default=True,
              metavar="RELATIVE_PATH", help="Resource path relative to the skill root.")
@click.option("--all", "all_files", is_flag=True,
              help="Expand reachable Markdown references in link order.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def skills_show(name: str, relative_path: str, all_files: bool, as_json: bool) -> None:
    """Print one bundled skill resource, or explicitly expand its Markdown links."""
    try:
        emit_show(get_skill(name), relative_path, all_files=all_files, as_json=as_json)
    except SkillError as exc:
        _skill_error(exc, as_json=as_json)


@skills_group.command(name="tree")
@click.argument("name", metavar="SKILL_NAME")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def skills_tree(name: str, as_json: bool) -> None:
    """Show linked local resources without outputting document bodies."""
    try:
        emit_tree(get_skill(name), as_json=as_json)
    except SkillError as exc:
        _skill_error(exc, as_json=as_json)


# Register structure command modules after the Click groups exist.  The
# documentation-only path stays on the standard-library resource viewer and
# does not import chemistry, analysis, or periodic-geometry modules.  Inspect
# the root command only: a positional value such as ``mck io info skills``
# must not hide the existing structure commands.
_root_command = next(
    (argument for argument in sys.argv[1:] if not argument.startswith("-")),
    None,
)
if _root_command != "skills":
    from .io_cmd import register_io_commands  # noqa: E402
    from .operate_cmd import register_operate_commands  # noqa: E402
    from .analyze_cmd import register_analyze_commands  # noqa: E402
    from .periodic_chain_cmd import register_periodic_chain_commands  # noqa: E402

    register_io_commands(io_group)
    register_operate_commands(operate_group)
    register_analyze_commands(analyze_group)
    register_periodic_chain_commands(main)


__all__ = ["main"]
