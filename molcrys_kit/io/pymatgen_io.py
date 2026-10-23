"""Bidirectional MolecularCrystal ↔ pymatgen Structure conversion.

``AseAtomsAdaptor`` rejects partially occupied structures
(``ValueError: ASE Atoms only supports ordered structures``).  This module
bridges the two representations while preserving per-site occupancy as
MolCrysKit ``occupancy`` arrays and pymatgen composition amounts.

Occupancy contract
------------------
- ``to_pymatgen``: each MolCrysKit site becomes one ``PeriodicSite`` whose
  composition is ``{species: occupancy}`` (default occupancy ``1.0``).
  Formal charge and hydrogen-isotope mass numbers are mapped onto pymatgen
  ``Species`` / ``Element`` keys when present.
- ``from_pymatgen``: each composition component from ``site.species.items()``
  becomes one atom at that site's Cartesian position with
  ``occupancy = amount``.  Components are **not** collapsed by element
  symbol (so ``Fe2+/Fe3+`` and ``H/D`` stay separate).  Oxidation state maps
  to ``formal_charge``; hydrogen isotopes ``D``/``T`` map to ``isotope``.
- Multi-species (mixed) sites therefore expand to multiple atoms that share
  coordinates — matching how disordered CIF rows are stored in MolCrysKit.
- Site properties ``molecule_index``, ``disorder_group``, ``assembly``, and
  ``label`` round-trip when present; missing values use MolCrysKit defaults.
- An empty ``Structure`` (zero sites) becomes an empty ``MolecularCrystal``
  with the same lattice / PBC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element, Species

from molcrys_kit.constants.config import (
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_FORMAL_CHARGE,
    KEY_FORMAL_CHARGE_KNOWN,
    KEY_ISOTOPE,
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
    """Return the chemical-element symbol (H for D/T, Fe for Fe2+)."""
    symbol = getattr(species, "symbol", None)
    if symbol is not None:
        return str(symbol)
    return str(species)


def _species_name(species: Any) -> str:
    """Return pymatgen ``name`` when present (D/T for hydrogen isotopes)."""
    name = getattr(species, "name", None)
    if name is not None:
        return str(name)
    return _element_symbol(species)


def _isotope_mass_number(species: Any) -> int:
    """Return mass number for named isotopes; else ``0`` (unspecified)."""
    name = _species_name(species)
    symbol = _element_symbol(species)
    raw = getattr(species, "A", None)
    if raw is None:
        raw = getattr(species, "atomic_mass_number", None)
    if raw is None:
        return 0
    try:
        value = int(round(float(raw)))
    except (TypeError, ValueError):
        return 0
    # Named hydrogen isotopes, or any Element whose name differs from symbol.
    if name in {"D", "T"} or (name and name != symbol):
        return value
    return 0


def _oxidation_state(species: Any) -> Optional[int]:
    oxi = getattr(species, "oxi_state", None)
    if oxi is None:
        return None
    try:
        return int(oxi)
    except (TypeError, ValueError):
        return None


def _validate_occupancy(occupancy: float, *, label: str) -> float:
    """Return occupancy in ``[0, 1]`` or raise a clear MolCrysKit error."""
    value = float(occupancy)
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"{label} has occupancy {value}; MolCrysKit requires occupancy in [0, 1]"
        )
    return value


def _make_lattice(matrix: Any, pbc: Sequence[bool]) -> Lattice:
    """Build a Lattice, tolerating older pymatgen without ``pbc=``."""
    matrix_arr = np.asarray(matrix, dtype=float)
    pbc_t = tuple(bool(v) for v in pbc)
    try:
        return Lattice(matrix_arr, pbc=pbc_t)
    except TypeError:
        lattice = Lattice(matrix_arr)
        setter = getattr(lattice, "__setattr__", None)
        if setter is not None and hasattr(lattice, "pbc"):
            try:
                lattice.pbc = pbc_t  # type: ignore[attr-defined]
            except Exception:
                pass
        return lattice


def _composition_key_for_record(record) -> Any:
    """Choose a pymatgen composition key that preserves charge / isotope."""
    symbol = str(record.symbol)
    charge = record.formal_charge
    isotope = record.isotope

    if symbol == "H" and isotope in {2, 3}:
        isotope_name = "D" if isotope == 2 else "T"
        if charge is not None:
            return Species(isotope_name, charge)
        return Element(isotope_name)

    if charge is not None:
        return Species(symbol, charge)
    return symbol


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
    lattice = _make_lattice(crystal.lattice, crystal.pbc)
    if not records:
        return Structure(lattice, [], [], coords_are_cartesian=False)

    species: List[Any] = []
    frac_coords: List[Tuple[float, float, float]] = []
    molecule_index: List[int] = []
    disorder_group: List[int] = []
    assembly: List[str] = []
    label: List[str] = []

    for record in records:
        occupancy = _validate_occupancy(
            record.occupancy,
            label=f"site {record.global_index}",
        )
        species.append({_composition_key_for_record(record): occupancy})
        frac_coords.append(tuple(float(v) for v in record.fractional_position))
        molecule_index.append(int(record.molecule_index))
        disorder_group.append(int(record.disorder_group))
        assembly.append("" if record.disorder_assembly is None else str(record.disorder_assembly))
        label.append(str(record.label))

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
) -> Tuple[List[str], np.ndarray, dict[str, list]]:
    """Expand Structure sites into flat atom lists with occupancy arrays.

    Iterates ``site.species.items()`` so same-element components with distinct
    oxidation state or isotope identity are not merged.
    """
    symbols: List[str] = []
    positions: List[Sequence[float]] = []
    occupancies: List[float] = []
    molecule_index: List[int] = []
    disorder_group: List[int] = []
    assembly: List[str] = []
    label: List[str] = []
    isotopes: List[int] = []
    formal_charges: List[int] = []
    formal_charge_known: List[bool] = []
    has_molecule_index = True
    has_isotope = False
    has_formal_charge = False

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

        components = list(site.species.items())
        if not components:
            raise ValueError(f"structure site {site_index} has an empty composition")

        for species_key, amount in components:
            occupancy = float(amount)
            if occupancy == 0.0:
                continue
            occupancy = _validate_occupancy(
                occupancy,
                label=(
                    f"structure site {site_index} component "
                    f"{_species_name(species_key)}"
                ),
            )
            symbol = _element_symbol(species_key)
            isotope = _isotope_mass_number(species_key)
            oxi = _oxidation_state(species_key)
            if isotope > 0:
                has_isotope = True
            if oxi is not None:
                has_formal_charge = True

            symbols.append(symbol)
            positions.append([float(v) for v in site.coords])
            occupancies.append(occupancy)
            molecule_index.append(mol_value)
            disorder_group.append(group_value)
            assembly.append(assembly_value)
            isotopes.append(isotope)
            if oxi is None:
                formal_charges.append(0)
                formal_charge_known.append(False)
            else:
                formal_charges.append(int(oxi))
                formal_charge_known.append(True)
            if raw_label is None:
                label.append(_species_name(species_key))
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
    if has_isotope:
        prop_lists[KEY_ISOTOPE] = isotopes
    if has_formal_charge:
        prop_lists[KEY_FORMAL_CHARGE] = formal_charges
        prop_lists[KEY_FORMAL_CHARGE_KNOWN] = formal_charge_known
    return symbols, np.asarray(positions, dtype=float), prop_lists


def _structure_to_atoms(structure: Structure) -> "Atoms":
    """Build an ASE Atoms object that carries occupancy metadata arrays."""
    from ase import Atoms

    if len(structure) == 0:
        cell = np.asarray(structure.lattice.matrix, dtype=float)
        pbc = tuple(bool(v) for v in getattr(structure.lattice, "pbc", (True, True, True)))
        return Atoms(cell=cell, pbc=pbc)

    symbols, positions, prop_lists = _expand_structure_sites(structure)
    cell = np.asarray(structure.lattice.matrix, dtype=float)
    pbc = tuple(bool(v) for v in getattr(structure.lattice, "pbc", (True, True, True)))
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
    accepted.  Each composition component from ``site.species.items()`` becomes
    one atom with the component amount stored in the ``occupancy`` array.
    Same-element components that differ by oxidation state or isotope are kept
    separate.

    If every site carries a non-negative ``molecule_index`` property, the
    molecule partition is restored via :meth:`MolecularCrystal.from_ase_atoms`.
    Otherwise molecules are identified with :meth:`MolecularCrystal.from_ase`.
    An empty structure yields an empty crystal with the same lattice.

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

    if len(structure) == 0:
        pbc = tuple(bool(v) for v in getattr(structure.lattice, "pbc", (True, True, True)))
        return MolecularCrystal(
            np.asarray(structure.lattice.matrix, dtype=float),
            [],
            pbc=pbc,
        )

    atoms = _structure_to_atoms(structure)
    if len(atoms) == 0:
        pbc = tuple(bool(v) for v in atoms.get_pbc())
        return MolecularCrystal(
            np.asarray(atoms.get_cell(), dtype=float),
            [],
            pbc=pbc,
        )
    if "molecule_index" in atoms.arrays:
        return MolecularCrystal.from_ase_atoms(atoms, bond_scale=bond_scale)
    return MolecularCrystal.from_ase(
        atoms,
        bond_thresholds=bond_thresholds,
        max_atoms=max_atoms,
        bond_scale=bond_scale,
    )
