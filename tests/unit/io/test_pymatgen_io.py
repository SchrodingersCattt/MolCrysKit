"""Occupancy-preserving MolecularCrystal ↔ pymatgen Structure bridge."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.ase import AseAtomsAdaptor

from molcrys_kit.constants.config import (
    KEY_DISORDER_GROUP,
    KEY_FORMAL_CHARGE,
    KEY_FORMAL_CHARGE_KNOWN,
    KEY_ISOTOPE,
    KEY_OCCUPANCY,
)
from molcrys_kit.io.pymatgen_io import from_pymatgen_structure, to_pymatgen_structure
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


def _water_crystal_with_occupancy(
    occupancies: list[float],
    *,
    disorder_groups: list[int] | None = None,
) -> MolecularCrystal:
    """Single water molecule crystal with explicit occupancy metadata."""
    atoms = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.asarray(occupancies, dtype=float))
    if disorder_groups is not None:
        atoms.set_array(KEY_DISORDER_GROUP, np.asarray(disorder_groups, dtype=int))
    molecule = CrystalMolecule(atoms, crystal=None, check_pbc=False)
    molecule.info["atom_indices"] = [0, 1, 2]
    return MolecularCrystal(np.eye(3) * 8.0, [molecule], pbc=(True, True, True))


def test_to_pymatgen_preserves_partial_occupancy():
    crystal = _water_crystal_with_occupancy([1.0, 0.7, 0.3])
    structure = to_pymatgen_structure(crystal)

    assert not structure.is_ordered
    amounts = [site.species.get_el_amt_dict() for site in structure]
    assert amounts[0] == {"O": 1.0}
    assert amounts[1] == {"H": 0.7}
    assert amounts[2] == {"H": 0.3}


def test_ase_atoms_adaptor_rejects_partial_occupancy_that_mck_accepts():
    """Guard the reason this bridge exists."""
    crystal = _water_crystal_with_occupancy([1.0, 0.5, 0.5])
    structure = to_pymatgen_structure(crystal)
    with pytest.raises(ValueError, match="ordered structures"):
        AseAtomsAdaptor.get_atoms(structure)


def test_round_trip_preserves_occupancy_and_symbols():
    crystal = _water_crystal_with_occupancy(
        [1.0, 0.6, 0.4],
        disorder_groups=[0, 1, 2],
    )
    restored = from_pymatgen_structure(to_pymatgen_structure(crystal))

    original_occ = np.concatenate(
        [mol.arrays[KEY_OCCUPANCY] for mol in crystal.molecules]
    )
    restored_occ = np.concatenate(
        [mol.arrays[KEY_OCCUPANCY] for mol in restored.molecules]
    )
    assert restored_occ == pytest.approx(original_occ)

    original_symbols = [
        sym for mol in crystal.molecules for sym in mol.get_chemical_symbols()
    ]
    restored_symbols = [
        sym for mol in restored.molecules for sym in mol.get_chemical_symbols()
    ]
    assert restored_symbols == original_symbols

    original_groups = np.concatenate(
        [mol.arrays[KEY_DISORDER_GROUP] for mol in crystal.molecules]
    )
    restored_groups = np.concatenate(
        [mol.arrays[KEY_DISORDER_GROUP] for mol in restored.molecules]
    )
    assert restored_groups.tolist() == original_groups.tolist()


def test_molecular_crystal_methods_match_module_functions():
    crystal = _water_crystal_with_occupancy([0.9, 1.0, 1.0])
    structure = crystal.to_pymatgen()
    assert structure[0].species.get_el_amt_dict()["O"] == pytest.approx(0.9)

    restored = MolecularCrystal.from_pymatgen(structure)
    occ = restored.molecules[0].arrays[KEY_OCCUPANCY]
    assert float(occ[0]) == pytest.approx(0.9)


def test_from_pymatgen_expands_mixed_composition_site():
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        [{"C": 0.5, "N": 0.5}, "O"],
        [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]],
    )
    crystal = from_pymatgen_structure(structure)
    symbols = [sym for mol in crystal.molecules for sym in mol.get_chemical_symbols()]
    occupancies = np.concatenate(
        [mol.arrays[KEY_OCCUPANCY] for mol in crystal.molecules]
    )

    assert sorted(symbols) == ["C", "N", "O"]
    assert sorted(float(v) for v in occupancies) == pytest.approx([0.5, 0.5, 1.0])


def test_from_pymatgen_skips_zero_occupancy_components():
    lattice = Lattice.cubic(4.0)
    structure = Structure(
        lattice,
        [{"C": 0.0, "N": 1.0}],
        [[0.1, 0.2, 0.3]],
    )
    crystal = from_pymatgen_structure(structure)
    symbols = [sym for mol in crystal.molecules for sym in mol.get_chemical_symbols()]
    occupancies = [
        float(v)
        for mol in crystal.molecules
        for v in mol.arrays[KEY_OCCUPANCY]
    ]
    assert symbols == ["N"]
    assert occupancies == pytest.approx([1.0])


def test_full_occupancy_round_trip_is_ordered():
    crystal = _water_crystal_with_occupancy([1.0, 1.0, 1.0])
    structure = to_pymatgen_structure(crystal)
    assert structure.is_ordered
    restored = from_pymatgen_structure(structure)
    assert len(restored.to_ase()) == 3


def test_site_records_occupancy_matches_structure():
    crystal = _water_crystal_with_occupancy([1.0, 0.25, 0.75])
    structure = to_pymatgen_structure(crystal)
    records = crystal.get_site_records()
    for record, site in zip(records, structure, strict=True):
        amount = site.species.get_el_amt_dict()[record.symbol]
        assert float(amount) == pytest.approx(record.occupancy)


def test_from_pymatgen_rejects_non_structure():
    with pytest.raises(TypeError, match="pymatgen Structure"):
        from_pymatgen_structure(object())  # type: ignore[arg-type]


def test_from_pymatgen_keeps_fe2_fe3_as_separate_components():
    """get_el_amt_dict would merge these into Fe@1.0; items() must not."""
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        [{Species("Fe", 2): 0.5, Species("Fe", 3): 0.5}],
        [[0.0, 0.0, 0.0]],
    )
    # Guard the regression target: element-amount collapse looks "ordered".
    assert structure[0].species.get_el_amt_dict() == {"Fe": 1.0}

    crystal = from_pymatgen_structure(structure)
    symbols = [sym for mol in crystal.molecules for sym in mol.get_chemical_symbols()]
    occupancies = [
        float(v)
        for mol in crystal.molecules
        for v in mol.arrays[KEY_OCCUPANCY]
    ]
    charges = [
        int(v)
        for mol in crystal.molecules
        for v in mol.arrays[KEY_FORMAL_CHARGE]
    ]
    known = [
        bool(v)
        for mol in crystal.molecules
        for v in mol.arrays[KEY_FORMAL_CHARGE_KNOWN]
    ]

    assert symbols == ["Fe", "Fe"]
    assert occupancies == pytest.approx([0.5, 0.5])
    assert sorted(charges) == [2, 3]
    assert known == [True, True]


def test_from_pymatgen_keeps_h_and_d_as_separate_components():
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        [{Element("H"): 0.5, Element("D"): 0.5}],
        [[0.1, 0.2, 0.3]],
    )
    assert structure[0].species.get_el_amt_dict() == {"H": 1.0}

    crystal = from_pymatgen_structure(structure)
    symbols = [sym for mol in crystal.molecules for sym in mol.get_chemical_symbols()]
    occupancies = [
        float(v)
        for mol in crystal.molecules
        for v in mol.arrays[KEY_OCCUPANCY]
    ]
    isotopes = [
        int(v)
        for mol in crystal.molecules
        for v in mol.arrays[KEY_ISOTOPE]
    ]

    assert symbols == ["H", "H"]
    assert occupancies == pytest.approx([0.5, 0.5])
    assert sorted(isotopes) == [0, 2]


def test_to_pymatgen_rejects_occupancy_above_one():
    crystal = _water_crystal_with_occupancy([1.0, 1.2, 1.0])
    with pytest.raises(ValueError, match="occupancy in \\[0, 1\\]"):
        to_pymatgen_structure(crystal)


def test_empty_structure_round_trips_to_empty_crystal():
    lattice = Lattice.cubic(4.0, pbc=(True, True, False))
    structure = Structure(lattice, [], [])
    crystal = from_pymatgen_structure(structure)
    assert crystal.molecules == []
    assert tuple(crystal.pbc) == (True, True, False)
    restored = to_pymatgen_structure(crystal)
    assert len(restored) == 0


def test_to_pymatgen_emits_species_for_charge_and_deuterium():
    atoms = Atoms(
        "FeH",
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    atoms.set_array(KEY_OCCUPANCY, np.array([0.5, 0.5], dtype=float))
    atoms.set_array(KEY_FORMAL_CHARGE, np.array([2, 0], dtype=int))
    atoms.set_array(KEY_FORMAL_CHARGE_KNOWN, np.array([True, False], dtype=bool))
    atoms.set_array(KEY_ISOTOPE, np.array([0, 2], dtype=int))
    molecule = CrystalMolecule(atoms, crystal=None, check_pbc=False)
    molecule.info["atom_indices"] = [0, 1]
    crystal = MolecularCrystal(np.eye(3) * 6.0, [molecule], pbc=(True, True, True))

    structure = to_pymatgen_structure(crystal)
    keys = [list(site.species.keys())[0] for site in structure]
    assert isinstance(keys[0], Species)
    assert keys[0].symbol == "Fe"
    assert int(keys[0].oxi_state) == 2
    assert _species_is_deuterium(keys[1])


def _species_is_deuterium(species) -> bool:
    return str(getattr(species, "name", "")) == "D" or str(species) in {"D", "Element D"}
