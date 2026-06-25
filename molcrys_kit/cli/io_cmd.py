"""I/O commands for the MolCrysKit CLI."""

from __future__ import annotations

from pathlib import Path

import click

from ._common import load_crystal, write_structure


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(input: Path) -> None:
    """Print a concise molecular-crystal summary."""
    crystal = load_crystal(input)
    click.echo(crystal.summary().rstrip())
    try:
        a, b, c, alpha, beta, gamma = crystal.get_lattice_parameters()
        click.echo(
            "  Cell: "
            f"a={a:.4f} b={b:.4f} c={c:.4f} "
            f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f}"
        )
    except Exception:
        pass
    click.echo("  Molecules:")
    for idx, molecule in enumerate(crystal.molecules):
        formula = molecule.get_chemical_formula()
        click.echo(f"    [{idx}] {formula} atoms={len(molecule)}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output path; extension selects the format.")
def convert(input: Path, output: Path) -> None:
    """Convert a structure file by output extension."""
    crystal = load_crystal(input)
    write_structure(crystal, output)
    click.echo(f"Wrote {output}")


def register_io_commands(group: click.Group) -> None:
    group.add_command(info)
    group.add_command(convert)
