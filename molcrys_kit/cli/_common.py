"""Shared helpers for the MolCrysKit command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import click

if TYPE_CHECKING:
    from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


CRYSTAL_INPUT_EXTENSIONS = {".cif", ".vasp", ".poscar", ".contcar", ".extxyz", ".mryx"}


STRICT_FORMULA = click.option(
    "--strict-formula",
    is_flag=True,
    help=(
        "Require formula moiety and Z and reject mismatched candidates "
        "during disorder resolution."
    ),
)
INDEPENDENT_ASSEMBLIES = click.option(
    "--independent-assemblies",
    is_flag=True,
    help=(
        "Treat complete multi-group assemblies as independent during disorder "
        "resolution; verify source semantics first."
    ),
)


def load_crystal(
    path: str | Path,
    *,
    resolve_disorder: bool = False,
    bond_scale: float = 1.0,
    strict_formula: bool = False,
    independent_assemblies: bool = False,
) -> MolecularCrystal:
    """Load a MolecularCrystal from a supported structure file.

    ``strict_formula`` and ``independent_assemblies`` are forwarded to CIF
    disorder resolution. Both default to false. They apply only when
    ``resolve_disorder`` is true, so callers that omit them keep legacy
    coupling, geometry heuristics, and soft formula filtering.
    """
    from molcrys_kit.io import read_extxyz, read_mol_crystal, read_poscar, read_mryx

    if (strict_formula or independent_assemblies) and not resolve_disorder:
        raise click.UsageError(
            "strict_formula and independent_assemblies apply only when "
            "resolve_disorder is true."
        )

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()

    if suffix == ".mryx":
        if resolve_disorder:
            raise click.UsageError("MRYX disorder resolution is not supported; raw groups are retained as metadata.")
        return read_mryx(file_path, bond_scale=bond_scale)

    if suffix == ".cif":
        try:
            return read_mol_crystal(
                str(file_path),
                resolve_disorder=resolve_disorder,
                bond_scale=bond_scale,
                strict_formula=strict_formula,
                independent_assemblies=independent_assemblies,
            )
        except ValueError as exc:
            # Opt-in flags should fail as a CLI error, matching ``disorder``.
            # Callers that omit them still see the original exception.
            if strict_formula or independent_assemblies:
                raise click.ClickException(str(exc)) from exc
            raise
    if suffix in {".vasp", ".poscar"} or name in {"poscar", "contcar"}:
        return read_poscar(str(file_path), bond_scale=bond_scale)
    if suffix == ".extxyz":
        crystal = read_extxyz(str(file_path), bond_scale=bond_scale)
        if isinstance(crystal, list):
            if not crystal:
                raise click.ClickException(f"No frames found in {file_path}")
            return crystal[-1]
        return crystal

    raise click.ClickException(
        f"Unsupported input format for {file_path!s}; expected CIF, POSCAR/CONTCAR, ExtXYZ, or MRYX."
    )


def write_structure(obj: MolecularCrystal | CrystalMolecule | Iterable[MolecularCrystal], path: str | Path) -> None:
    """Write a crystal, molecule, or frame sequence using the output suffix."""
    import ase.io
    import numpy as np

    from molcrys_kit.io import write_cif, write_extxyz, write_poscar, write_xyz
    from molcrys_kit.structures import CrystalMolecule, MolecularCrystal

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".cif":
        if isinstance(obj, MolecularCrystal):
            # CIF supports a small MolCrysKit side-table for transform
            # provenance; pass crystal metadata so it survives this export.
            write_cif(obj, str(file_path), metadata=obj.metadata)
            return
        if isinstance(obj, CrystalMolecule):
            lattice = np.asarray(obj.get_cell().array, dtype=float)
            if np.allclose(lattice, 0) or abs(float(np.linalg.det(lattice))) < 1e-8:
                raise click.ClickException(
                    "CIF output for a molecule requires a non-zero cell; use --center-vacuum."
                )
            crystal = MolecularCrystal(
                lattice=lattice,
                molecules=[obj],
                pbc=tuple(bool(flag) for flag in obj.get_pbc()),
            )
            write_cif(crystal, str(file_path))
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
        if isinstance(obj, CrystalMolecule):
            ase.io.write(str(file_path), obj.to_ase(), format="extxyz")
            return
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
