"""Structure ↔ MolecularCrystal conversion for the pymatgen addon."""

from __future__ import annotations

from pymatgen.core import Structure

from molcrys_kit.structures import MolecularCrystal


def as_molecular_crystal(structure: Structure | MolecularCrystal, **kwargs) -> MolecularCrystal:
    """Convert a pymatgen ``Structure`` to ``MolecularCrystal``.

    Uses MolCrysKit's occupancy-preserving bridge.  Passes through an existing
    ``MolecularCrystal`` unchanged.
    """
    if isinstance(structure, MolecularCrystal):
        return structure
    if not isinstance(structure, Structure):
        raise TypeError(
            "expected a pymatgen Structure or MolecularCrystal, "
            f"got {type(structure).__name__}"
        )
    return MolecularCrystal.from_pymatgen(structure, **kwargs)


def as_structure(crystal: MolecularCrystal | Structure) -> Structure:
    """Convert a ``MolecularCrystal`` to a pymatgen ``Structure``.

    Passes through an existing ``Structure`` unchanged.
    """
    if isinstance(crystal, MolecularCrystal):
        return crystal.to_pymatgen()
    if isinstance(crystal, Structure):
        return crystal
    raise TypeError(
        "expected a MolecularCrystal or pymatgen Structure, "
        f"got {type(crystal).__name__}"
    )
