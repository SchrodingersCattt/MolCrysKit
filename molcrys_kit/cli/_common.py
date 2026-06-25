"""Shared helpers for the MolCrysKit command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import ase.io
import click

from molcrys_kit.io import (
    read_extxyz,
    read_mol_crystal,
    read_poscar,
    write_cif,
    write_extxyz,
    write_poscar,
    write_xyz,
)
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


CRYSTAL_INPUT_EXTENSIONS = {".cif", ".vasp", ".poscar", ".contcar", ".extxyz"}


def load_crystal(path: str | Path) -> MolecularCrystal:
    """Load a MolecularCrystal from a supported structure file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()

    if suffix == ".cif":
        return read_mol_crystal(str(file_path))
    if suffix in {".vasp", ".poscar"} or name in {"poscar", "contcar"}:
        return read_poscar(str(file_path))
    if suffix == ".extxyz":
        crystal = read_extxyz(str(file_path))
        if isinstance(crystal, list):
            if not crystal:
                raise click.ClickException(f"No frames found in {file_path}")
            return crystal[-1]
        return crystal

    raise click.ClickException(
        f"Unsupported input format for {file_path!s}; expected CIF, POSCAR/CONTCAR, or ExtXYZ."
    )


def write_structure(obj: MolecularCrystal | CrystalMolecule | Iterable[MolecularCrystal], path: str | Path) -> None:
    """Write a crystal, molecule, or frame sequence using the output suffix."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".cif":
        if isinstance(obj, MolecularCrystal):
            write_cif(obj, str(file_path))
            return
        raise click.ClickException("CIF output requires a MolecularCrystal")
    if suffix in {".vasp", ".poscar"} or name in {"poscar", "contcar"}:
        if isinstance(obj, MolecularCrystal):
            write_poscar(obj, str(file_path))
            return
        raise click.ClickException("POSCAR output requires a MolecularCrystal")
    if suffix == ".xyz":
        if isinstance(obj, MolecularCrystal):
            # Whole-crystal XYZ output is a flattened ASE Atoms view.  The
            # project writer below is intentionally molecule/cluster oriented.
            ase.io.write(str(file_path), obj.to_ase(), format="xyz")
            return
        if isinstance(obj, CrystalMolecule):
            write_xyz(obj, str(file_path))
            return
        raise click.ClickException("XYZ output requires a MolecularCrystal or CrystalMolecule")
    if suffix == ".extxyz":
        write_extxyz(obj, str(file_path))
        return

    raise click.ClickException(
        f"Unsupported output format for {file_path!s}; use .cif, .vasp/.poscar, .xyz, or .extxyz."
    )


def write_crystal_sequence(crystals: list[MolecularCrystal], output: str | Path) -> list[Path]:
    """Write one or more crystal frames to a file or a numbered file set."""
    output_path = Path(output)
    if len(crystals) == 1:
        write_structure(crystals[0], output_path)
        return [output_path]

    if output_path.suffix.lower() == ".extxyz":
        write_structure(crystals, output_path)
        return [output_path]

    stem = output_path.with_suffix("")
    suffix = output_path.suffix or ".cif"
    written: list[Path] = []
    for idx, crystal in enumerate(crystals):
        path = stem.parent / f"{stem.name}_replica{idx}{suffix}"
        write_structure(crystal, path)
        written.append(path)
    return written


def rows_to_json(rows: Any) -> str:
    """Return pretty JSON using dataclass/object __dict__ fallback."""
    def default(obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.dumps(rows, indent=2, default=default)


def echo_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        click.echo(f"Wrote {path}")
