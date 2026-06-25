"""I/O commands for the MolCrysKit CLI."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import click

from molcrys_kit.analysis.molecular_identity import ChemicalIdentity

from ._common import load_crystal, rows_to_json, write_structure


logger = logging.getLogger(__name__)


def _round_triplet(values) -> list[float]:
    """Return a JSON-friendly coordinate triplet."""
    return [float(value) for value in values]


def _species_ids(crystal) -> list[str]:
    """Assign stable formula/topology species labels to all molecules."""
    identities = [
        ChemicalIdentity.from_molecule(molecule, idx, crystal=crystal)
        for idx, molecule in enumerate(crystal.molecules)
    ]
    species_by_key: dict[tuple[str, str | None], str] = {}
    counters: dict[str, int] = defaultdict(int)
    labels: list[str] = []

    for identity in identities:
        key = (identity.formula, identity.topo_signature)
        if key not in species_by_key:
            counters[identity.formula] += 1
            species_by_key[key] = f"{identity.formula}_{counters[identity.formula]}"
        labels.append(species_by_key[key])
    return labels


def _molecule_inventory(crystal) -> list[dict[str, Any]]:
    """Build JSON-friendly molecule inventory rows for a crystal."""
    labels = _species_ids(crystal)
    rows: list[dict[str, Any]] = []
    atom_offset = 0

    for idx, molecule in enumerate(crystal.molecules):
        identity = ChemicalIdentity.from_molecule(
            molecule,
            idx,
            crystal=crystal,
            species_id=labels[idx],
        )
        row = identity.to_dict()
        row["index"] = idx
        row["atom_count"] = len(molecule)
        row["centroid"] = _round_triplet(molecule.get_centroid())
        try:
            row["centroid_frac"] = _round_triplet(molecule.get_centroid_frac())
        except (AttributeError, ValueError):
            row["centroid_frac"] = None
        atom_indices = getattr(molecule, "info", {}).get("atom_indices")
        if atom_indices is None:
            atom_indices = list(range(atom_offset, atom_offset + len(molecule)))
        row["atom_indices"] = [int(atom_index) for atom_index in atom_indices]
        rows.append(row)
        atom_offset += len(molecule)
    return rows


def _format_coord(values: list[float] | None) -> str:
    if values is None:
        return "-"
    return "(" + ", ".join(f"{value:.3f}" for value in values) + ")"


def _selector_count(
    index: int | None,
    formula: str | None,
    species_id: str | None,
    largest: bool,
    all_molecules: bool,
) -> int:
    return sum(
        [
            index is not None,
            formula is not None,
            species_id is not None,
            largest,
            all_molecules,
        ]
    )


def _selected_molecule_indices(
    rows: list[dict[str, Any]],
    *,
    index: int | None,
    formula: str | None,
    species_id: str | None,
    largest: bool,
    all_molecules: bool,
) -> list[int]:
    """Resolve CLI selector options to one or more molecule indices."""
    selector_count = _selector_count(index, formula, species_id, largest, all_molecules)
    if selector_count > 1:
        raise click.UsageError(
            "Use only one molecule selector: --index, --formula, --species-id, --largest, or --all."
        )
    if not rows:
        raise click.ClickException("No molecules found in input structure.")
    if selector_count == 0:
        index = 0

    if all_molecules:
        return [int(row["index"]) for row in rows]
    if largest:
        return [int(max(rows, key=lambda row: row["atom_count"])["index"])]
    if index is not None:
        if index < 0 or index >= len(rows):
            raise click.ClickException(f"Molecule index {index} out of range 0..{len(rows) - 1}.")
        return [int(index)]
    if formula is not None:
        for row in rows:
            if row["formula"] == formula or row.get("hill_formula") == formula:
                return [int(row["index"])]
        raise click.ClickException(f"No molecule with formula {formula!r} found.")
    if species_id is not None:
        for row in rows:
            if row["species_id"] == species_id:
                return [int(row["index"])]
        raise click.ClickException(f"No molecule with species id {species_id!r} found.")

    raise click.ClickException("Could not resolve molecule selector.")


def _format_output_path(template: Path, row: dict[str, Any], *, multi: bool) -> Path:
    """Resolve an output template for a selected molecule."""
    values = {
        "index": row["index"],
        "molecule_index": row["index"],
        "formula": row["formula"],
        "species_id": row["species_id"],
    }
    text = str(template)
    if "{" in text and "}" in text:
        return Path(text.format(**values))
    if multi:
        return template.with_name(f"{template.stem}_{row['index']}{template.suffix}")
    return template


def _write_sidecar(path: Path, row: dict[str, Any], output_path: Path) -> None:
    payload = dict(row)
    payload["output"] = str(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    except (ArithmeticError, ValueError) as exc:
        logger.debug("Could not compute lattice parameters for %s: %s", input, exc)
    click.echo("  Molecules:")
    for idx, molecule in enumerate(crystal.molecules):
        formula = molecule.get_chemical_formula()
        click.echo(f"    [{idx}] {formula} atoms={len(molecule)}")


@click.command(name="molecules")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def molecules(input: Path, as_json: bool) -> None:
    """List molecule inventory for a crystal."""
    crystal = load_crystal(input)
    rows = _molecule_inventory(crystal)
    if as_json:
        click.echo(rows_to_json(rows))
        return

    click.echo("index  formula         atoms  species_id      centroid")
    click.echo("-----  --------------  -----  --------------  ------------------------------")
    for row in rows:
        click.echo(
            f"{row['index']:>5}  "
            f"{row['formula']:<14}  "
            f"{row['atom_count']:>5}  "
            f"{row['species_id']:<14}  "
            f"{_format_coord(row['centroid'])}"
        )


@click.command(name="extract-molecule")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output path or template. Use {index}, {formula}, or {species_id} with --all.")
@click.option("--index", type=int, help="0-based molecule index to extract. Defaults to 0 if no selector is given.")
@click.option("--formula", help="Extract the first molecule matching this formula or Hill formula.")
@click.option("--species-id", help="Extract the first molecule matching this computed species id.")
@click.option("--largest", is_flag=True, help="Extract the molecule with the largest atom count.")
@click.option("--all", "all_molecules", is_flag=True, help="Extract every molecule; output becomes a template.")
@click.option("--center-vacuum", type=float, help="Center each molecule in a box with this vacuum padding in angstrom.")
@click.option("--pbc", type=click.BOOL, default=False, show_default=True, help="Set periodic boundary conditions on the extracted molecule.")
@click.option("--json-sidecar", type=click.Path(dir_okay=False, path_type=Path), help="Write JSON metadata for the extracted molecule. With --all, this also acts as a template.")
def extract_molecule(
    input: Path,
    output: Path,
    index: int | None,
    formula: str | None,
    species_id: str | None,
    largest: bool,
    all_molecules: bool,
    center_vacuum: float | None,
    pbc: bool,
    json_sidecar: Path | None,
) -> None:
    """Extract one or more molecules from a crystal."""
    if center_vacuum is not None and center_vacuum < 0:
        raise click.UsageError("--center-vacuum must be non-negative.")

    crystal = load_crystal(input)
    rows = _molecule_inventory(crystal)
    selected = _selected_molecule_indices(
        rows,
        index=index,
        formula=formula,
        species_id=species_id,
        largest=largest,
        all_molecules=all_molecules,
    )
    rows_by_index = {int(row["index"]): row for row in rows}
    multi = len(selected) > 1
    written: list[Path] = []

    for molecule_index in selected:
        row = rows_by_index[molecule_index]
        molecule = crystal.molecules[molecule_index].copy()
        molecule.set_pbc((bool(pbc), bool(pbc), bool(pbc)))
        if center_vacuum is not None:
            molecule.center(vacuum=center_vacuum, axis=(0, 1, 2))

        output_path = _format_output_path(output, row, multi=multi)
        write_structure(molecule, output_path)
        written.append(output_path)

        if json_sidecar is not None:
            sidecar_path = _format_output_path(json_sidecar, row, multi=multi)
            _write_sidecar(sidecar_path, row, output_path)

    for path in written:
        click.echo(f"Wrote {path}")


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
    group.add_command(molecules)
    group.add_command(extract_molecule)
    group.add_command(convert)
