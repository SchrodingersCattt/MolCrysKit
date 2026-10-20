"""Command line interface for MolCrysKit."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import logging

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


# Structure commands import chemistry, analysis, and periodic geometry.
# Registration is deferred until a caller actually looks one up.  The decision
# uses the command Click resolved, not the host process ``sys.argv``: a wrapper
# whose own first argument is ``skills`` must still expose the structure CLI
# when it invokes ``main(["io", ...])`` or reads a structure group's commands.
_structure_commands_loaded = False
_structure_commands_loading = False


class _LoadingCommandMap(dict[str, click.Command]):
    """Command registry that imports structure commands on the first read.

    ``lightweight`` names are already registered (today, only ``skills``) and
    can be fetched without importing chemistry modules.  Inserts never load,
    so group construction can populate the map before any lookup.
    """

    def __init__(self, lightweight: frozenset[str] = frozenset()) -> None:
        super().__init__()
        self._lightweight = lightweight

    def _load(self, key: object | None = None) -> None:
        if key in self._lightweight:
            return
        _ensure_structure_commands()

    def __getitem__(self, key: str) -> click.Command:
        self._load(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._load(key)
        return super().__contains__(key)

    def __iter__(self) -> Iterator[str]:
        self._load()
        return super().__iter__()

    def __len__(self) -> int:
        self._load()
        return super().__len__()

    def __bool__(self) -> bool:
        self._load()
        return super().__bool__()

    def get(
        self, key: str, default: click.Command | None = None
    ) -> click.Command | None:
        self._load(key)
        return super().get(key, default)

    def keys(self) -> Iterable[str]:  # type: ignore[override]
        self._load()
        return super().keys()

    def items(self) -> Iterable[tuple[str, click.Command]]:  # type: ignore[override]
        self._load()
        return super().items()

    def values(self) -> Iterable[click.Command]:  # type: ignore[override]
        self._load()
        return super().values()


class _StructureCommandGroup(click.Group):
    """Load chemistry CLI modules the first time a structure command is needed."""

    defer_skills = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        lightweight = frozenset({"skills"}) if self.defer_skills else frozenset()
        self.commands = _LoadingCommandMap(lightweight)

    def list_commands(self, ctx: click.Context) -> list[str]:
        _ensure_structure_commands()
        return super().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if not (self.defer_skills and cmd_name == "skills"):
            _ensure_structure_commands()
        return super().get_command(ctx, cmd_name)


class _RootCommandGroup(_StructureCommandGroup):
    """Root group.  ``skills`` stays import-light until another command is used."""

    defer_skills = True


@click.group(
    cls=_RootCommandGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="mck")
@click.option("--verbose", is_flag=True, help="Show debug logging.")
@click.option("--quiet", is_flag=True, help="Only show warnings and errors.")
def main(verbose: bool = False, quiet: bool = False) -> None:
    """MolCrysKit command line tools."""
    if verbose and quiet:
        raise click.UsageError("Use at most one of --verbose and --quiet.")
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@main.group(name="io", cls=_StructureCommandGroup)
def io_group() -> None:
    """Read, summarize, and convert structures."""


@main.group(name="operate", cls=_StructureCommandGroup)
def operate_group() -> None:
    """Generate modified structures."""


@main.group(name="analyze", cls=_StructureCommandGroup)
def analyze_group() -> None:
    """Analyze crystals and print reports."""


def _skill_error(exc: SkillError, *, as_json: bool) -> None:
    if as_json:
        click.echo(f"Error: {exc}", err=True)
        click.echo(
            json_payload({"schema": "mck.error/v1", "ok": False, "error": str(exc)})
        )
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
@click.option(
    "--file",
    "relative_path",
    default="SKILL.md",
    show_default=True,
    metavar="RELATIVE_PATH",
    help="Resource path relative to the skill root.",
)
@click.option(
    "--all",
    "all_files",
    is_flag=True,
    help="Expand reachable Markdown references in link order.",
)
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


def _ensure_structure_commands() -> None:
    """Import and register chemistry commands once.

    Resolving ``skills`` does not call this.  Help listings, other command
    lookups, and reads of a structure group's command map do.  A failed import
    leaves the registry unloaded so a later lookup can retry.
    """
    global _structure_commands_loaded, _structure_commands_loading
    if _structure_commands_loaded or _structure_commands_loading:
        return
    _structure_commands_loading = True
    try:
        from .analyze_cmd import register_analyze_commands
        from .io_cmd import register_io_commands
        from .operate_cmd import register_operate_commands
        from .periodic_chain_cmd import register_periodic_chain_commands

        register_io_commands(io_group)
        register_operate_commands(operate_group)
        register_analyze_commands(analyze_group)
        register_periodic_chain_commands(main)
        _structure_commands_loaded = True
    finally:
        _structure_commands_loading = False


__all__ = ["main"]
