"""Analysis commands for the MolCrysKit CLI."""

from __future__ import annotations

from pathlib import Path

import click

from molcrys_kit.analysis import enumerate_bfdh_facets, find_polyhedra
from molcrys_kit.analysis.interactions import interaction_profile

from ._common import load_crystal, rows_to_json


def _miller_text(value) -> str:
    return "(" + " ".join(str(int(x)) for x in value) + ")"


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-index", type=int, default=2, show_default=True)
@click.option("--top-n", type=int, default=10, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def bfdh(input: Path, max_index: int, top_n: int, as_json: bool) -> None:
    """Rank low-index facets using BFDH morphology."""
    facets = enumerate_bfdh_facets(load_crystal(input), max_index=max_index, top_n=top_n)
    if as_json:
        click.echo(rows_to_json(facets))
        return

    click.echo("miller       d_hkl      importance")
    click.echo("-----------------------------------")
    for facet in facets:
        importance = getattr(facet, "morphological_importance", getattr(facet, "importance", ""))
        click.echo(f"{_miller_text(facet.miller_index):<12} {facet.d_hkl:>8.4f} {importance}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def interactions(input: Path, as_json: bool) -> None:
    """Summarize weak interactions and continuous scores."""
    profile = interaction_profile(load_crystal(input))
    if as_json:
        click.echo(rows_to_json(profile))
        return

    click.echo("Interaction profile")
    for key, value in profile.__dict__.items():
        if isinstance(value, (int, float, str, bool)):
            click.echo(f"  {key}: {value}")
        elif isinstance(value, list):
            click.echo(f"  {key}: {len(value)}")
        else:
            click.echo(f"  {key}: {value}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--central", required=True, help="Central atom symbol or moiety string.")
@click.option("--ligand", required=True, help="Ligand atom symbol or moiety string.")
@click.option("--level", type=click.Choice(["atom", "molecule"]), default="atom", show_default=True)
@click.option("--cutoff", type=float, default=None, help="Coordination cutoff in Angstrom.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def polyhedra(input: Path, central: str, ligand: str, level: str, cutoff: float | None, as_json: bool) -> None:
    """Enumerate coordination polyhedra."""
    rows = find_polyhedra(load_crystal(input), central=central, ligand=ligand, level=level, cutoff=cutoff)
    if as_json:
        click.echo(rows_to_json(rows))
        return

    click.echo(f"Found {len(rows)} polyhedra")
    for idx, row in enumerate(rows):
        center = row.get("center_index", row.get("central_index", idx))
        cn = row.get("coordination_number", row.get("cn", "?"))
        shape = row.get("shape", row.get("best_shape", ""))
        click.echo(f"  [{idx}] center={center} cn={cn} shape={shape}")


def register_analyze_commands(group: click.Group) -> None:
    group.add_command(bfdh)
    group.add_command(interactions)
    group.add_command(polyhedra)
