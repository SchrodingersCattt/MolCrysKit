"""Stable, machine-readable summaries for supported structure files."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from molcrys_kit.structures import MolecularCrystal


def summarize_structure(
    crystal: MolecularCrystal,
    *,
    source: str | None = None,
    symprec: float = 0.1,
) -> dict[str, Any]:
    """Return composition, cell, symmetry, and disorder facts."""
    if symprec <= 0:
        raise ValueError("symprec must be positive")

    atoms = crystal.to_ase()
    counts = Counter(atoms.get_chemical_symbols())
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    pbc = [bool(value) for value in atoms.pbc]
    site_records = crystal.get_site_records()

    partial_occupancy_sites = sum(
        record.occupancy < 1.0 - 1e-8 for record in site_records
    )
    disorder_groups = sorted(
        {record.disorder_group for record in site_records if record.disorder_group}
    )
    disorder_assemblies = sorted(
        {
            record.disorder_assembly
            for record in site_records
            if record.disorder_assembly is not None
        }
    )

    report: dict[str, Any] = {
        "source": source,
        "formula": Composition(dict(counts)).reduced_formula,
        "n_atoms": len(atoms),
        "species_counts": dict(sorted(counts.items())),
        "pbc": pbc,
        "periodic": any(pbc),
        "cell": {
            "lengths_A": [float(value) for value in lengths],
            "angles_deg": [float(value) for value in angles],
            "volume_A3": float(abs(np.linalg.det(np.asarray(atoms.cell)))),
        },
        "disorder": {
            "has_disorder": bool(
                partial_occupancy_sites or disorder_groups or disorder_assemblies
            ),
            "partial_occupancy_sites": partial_occupancy_sites,
            "groups": disorder_groups,
            "assemblies": disorder_assemblies,
        },
        "symmetry": None,
    }

    if all(pbc) and report["cell"]["volume_A3"] > 1e-8:
        try:
            structure = AseAtomsAdaptor.get_structure(atoms)
            analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
            symmetrized = analyzer.get_symmetrized_structure()
            wyckoff_sites = []
            for symbol, indices in zip(
                symmetrized.wyckoff_symbols,
                symmetrized.equivalent_indices,
                strict=True,
            ):
                species = sorted({structure[index].species_string for index in indices})
                wyckoff_sites.append(
                    {
                        "symbol": symbol,
                        "species": species,
                        "multiplicity": len(indices),
                    }
                )
            report["symmetry"] = {
                "status": "ok",
                "symprec_A": symprec,
                "space_group_symbol": analyzer.get_space_group_symbol(),
                "space_group_number": analyzer.get_space_group_number(),
                "crystal_system": analyzer.get_crystal_system(),
                "wyckoff_sites": wyckoff_sites,
            }
        except Exception as exc:  # pragma: no cover - parser-specific edge cases
            report["symmetry"] = {
                "status": "unavailable",
                "symprec_A": symprec,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    return report


__all__ = ["summarize_structure"]
