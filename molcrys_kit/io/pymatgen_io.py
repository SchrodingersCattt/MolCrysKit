"""Bidirectional MolecularCrystal ↔ pymatgen Structure conversion.

``AseAtomsAdaptor`` rejects partially occupied structures
(``ValueError: ASE Atoms only supports ordered structures``).  This module
bridges the two representations while preserving per-site occupancy as
MolCrysKit ``occupancy`` arrays and pymatgen composition amounts.

Occupancy contract
------------------
- ``to_pymatgen``: each MolCrysKit site becomes one ``PeriodicSite`` whose
  composition is ``{symbol: occupancy}`` (default occupancy ``1.0``).
- ``from_pymatgen``: each ``(element, amount)`` pair on a site becomes one
  atom at that site's Cartesian position with ``occupancy = amount``.
  Multi-species (mixed) sites therefore expand to multiple atoms that share
  coordinates — matching how disordered CIF rows are stored in MolCrysKit.
- Site properties ``molecule_index``, ``disorder_group``, ``assembly``, and
  ``label`` round-trip when present; missing values use MolCrysKit defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure

from molcrys_kit.constants.config import (
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_LABEL,
    KEY_OCCUPANCY,
)
from molcrys_kit.structures.crystal import MolecularCrystal

if TYPE_CHECKING:
    from ase import Atoms

_PROP_MOLECULE_INDEX = "molecule_index"
_PROP_DISORDER_GROUP = "disorder_group"
_PROP_ASSEMBLY = "assembly"
_PROP_LABEL = "label"

__all__ = [
    "from_pymatgen_structure",
    "to_pymatgen_structure",
]


def _element_symbol(species: Any) -> str:
    """Return a chemical-symbol string from a pymatgen Element/Species key."""
    symbol = getattr(species, "symbol", None)
    if symbol is not None:
        return str(symbol)
    return str(species)


def to_pymatgen_structure(crystal: MolecularCrystal) -> Structure:
    """Convert a :class:`MolecularCrystal` to a pymatgen ``Structure``.

    Occupancy is taken from each site's ``occupancy`` array (default ``1.0``)
    and written as the composition amount on that site.  Partial occupancies
    are preserved; the result may be ``not structure.is_ordered``.

    Parameters
    ----------
    crystal:
        Source molecular crystal.

    Returns
    -------
    Structure
        Periodic structure with one site per MolCrysKit atom, in
        :meth:`MolecularCrystal.to_ase` / :meth:`get_site_records` order.
    """
    records = crystal.get_site_records()
    if not records:
        lattice = Lattice(np.asarray(crystal.lattice, dtype=float), pbc=tuple(crystal.pbc))
        return Structure(lattice, [], [], coords_are_cartesian=False)

    species: List[dict[str, float]] = []
    frac_coords: List[Tuple[float, float, float]] = []
    molecule_index: List[int] = []
    disorder_group: List[int] = []
    assembly: List[str] = []
    label: List[str] = []

    for record in records:
        occupancy = float(record.occupancy)
        if occupancy < 0.0:
            raise ValueError(
                f"site {record.global_index} has negative occupancy {occupancy}"
            )
        species.append({record.symbol: occupancy})
        frac_coords.append(tuple(float(v) for v in record.fractional_position))
        molecule_index.append(int(record.molecule_index))
        disorder_group.append(int(record.disorder_group))
        assembly.append("" if record.disorder_assembly is None else str(record.disorder_assembly))
        label.append(str(record.label))

    lattice = Lattice(np.asarray(crystal.lattice, dtype=float), pbc=tuple(bool(v) for v in crystal.pbc))
    return Structure(
        lattice,
        species,
        frac_coords,
        coords_are_cartesian=False,
        site_properties={
            _PROP_MOLECULE_INDEX: molecule_index,
            _PROP_DISORDER_GROUP: disorder_group,
            _PROP_ASSEMBLY: assembly,
            _PROP_LABEL: label,
        },
    )


def _expand_structure_sites(
    structure: Structure,
) -> Tuple[List[str], np.ndarray, np.ndarray, dict[str, list]]:
    """Expand Structure sites into flat atom lists with occupancy arrays."""
    symbols: List[str] = []
    positions: List[Sequence[float]] = []
    occupancies: List[float] = []
    molecule_index: List[int] = []
    disorder_group: List[int] = []
    assembly: List[str] = []
    label: List[str] = []
    has_molecule_index = True

    for site_index, site in enumerate(structure):
        props = site.properties or {}
        raw_mol = props.get(_PROP_MOLECULE_INDEX)
        if raw_mol is None:
            has_molecule_index = False
            mol_value = -1
        else:
            mol_value = int(raw_mol)
            if mol_value < 0:
                has_molecule_index = False

        raw_group = props.get(_PROP_DISORDER_GROUP, 0)
        group_value = int(raw_group) if raw_group is not None else 0
        raw_assembly = props.get(_PROP_ASSEMBLY, "")
        assembly_value = "" if raw_assembly is None else str(raw_assembly)
        raw_label = props.get(_PROP_LABEL)

        el_amt = site.species.get_el_amt_dict()
        if not el_amt:
            raise ValueError(f"structure site {site_index} has an empty composition")

        for element_symbol, amount in el_amt.items():
            occupancy = float(amount)
            if occupancy == 0.0:
                continue
            if occupancy < 0.0:
                raise ValueError(
                    f"structure site {site_index} has negative occupancy "
                    f"for {element_symbol}: {occupancy}"
                )
            symbols.append(str(element_symbol))
            positions.append([float(v) for v in site.coords])
            occupancies.append(occupancy)
            molecule_index.append(mol_value)
            disorder_group.append(group_value)
            assembly.append(assembly_value)
            if raw_label is None:
                label.append(str(element_symbol))
            else:
                label.append(str(raw_label))

    if not symbols:
        raise ValueError("structure expands to zero atoms after dropping zero-occupancy species")

    prop_lists: dict[str, list] = {
        KEY_OCCUPANCY: occupancies,
        KEY_DISORDER_GROUP: disorder_group,
        KEY_ASSEMBLY: assembly,
        KEY_LABEL: label,
    }
    if has_molecule_index:
        prop_lists["molecule_index"] = molecule_index
    return symbols, np.asarray(positions, dtype=float), np.asarray(occupancies, dtype=float), prop_lists


def _structure_to_atoms(structure: Structure) -> "Atoms":
    """Build an ASE Atoms object that carries occupancy metadata arrays."""
    from ase import Atoms

    symbols, positions, _occupancies, prop_lists = _expand_structure_sites(structure)
    cell = np.asarray(structure.lattice.matrix, dtype=float)
    pbc = tuple(bool(v) for v in structure.lattice.pbc)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    for key, values in prop_lists.items():
        atoms.set_array(key, np.asarray(values))
    return atoms


def from_pymatgen_structure(
    structure: Structure,
    *,
    bond_scale: float = 1.0,
    bond_thresholds=None,
    max_atoms: Optional[int] = None,
) -> MolecularCrystal:
    """Create a :class:`MolecularCrystal` from a pymatgen ``Structure``.

    Unlike ``AseAtomsAdaptor.get_atoms``, partial and mixed occupancies are
    accepted.  Each composition component becomes one atom with the component
    amount stored in the ``occupancy`` array.

    If every site carries a non-negative ``molecule_index`` property, the
    molecule partition is restored via :meth:`MolecularCrystal.from_ase_atoms`.
    Otherwise molecules are identified with :meth:`MolecularCrystal.from_ase`.

    Parameters
    ----------
    structure:
        Periodic pymatgen structure.
    bond_scale, bond_thresholds, max_atoms:
        Forwarded to molecule identification when ``molecule_index`` is absent.

    Returns
    -------
    MolecularCrystal
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            f"structure must be a pymatgen Structure, got {type(structure).__name__}"
        )

    atoms = _structure_to_atoms(structure)
    if "molecule_index" in atoms.arrays:
        return MolecularCrystal.from_ase_atoms(atoms, bond_scale=bond_scale)
    return MolecularCrystal.from_ase(
        atoms,
        bond_thresholds=bond_thresholds,
        max_atoms=max_atoms,
        bond_scale=bond_scale,
    )
