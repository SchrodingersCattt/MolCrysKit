"""Command line interface for MolCrysKit."""

from __future__ import annotations

import logging

import click

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


# Register command modules after the Click groups exist.
from .io_cmd import register_io_commands  # noqa: E402
from .operate_cmd import register_operate_commands  # noqa: E402
from .analyze_cmd import register_analyze_commands  # noqa: E402

register_io_commands(io_group)
register_operate_commands(operate_group)
register_analyze_commands(analyze_group)


__all__ = ["main"]
