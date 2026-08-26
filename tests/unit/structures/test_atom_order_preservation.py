"""
Regression tests for preserving original atom order across MolecularCrystal.
"""

import numpy as np
from ase import Atoms

from molcrys_kit.io.cif import identify_molecule_indices, identify_molecules
from molcrys_kit.operations.molecule_manipulation import (
    replace_molecule,
    rotate_molecule,
    translate_molecule,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule


def _two_oh_atoms():
    return Atoms(
        symbols=["O", "O", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [5.96, 0.0, 0.0],
        ],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )


def _molecule_concat_symbols(crystal):
    symbols = []
    for molecule in crystal.molecules:
        symbols.extend(molecule.get_chemical_symbols())
    return symbols


def test_from_ase_to_ase_preserves_symbol_order():
    atoms = _two_oh_atoms()

    round_tripped = MolecularCrystal.from_ase(atoms).to_ase()

    assert round_tripped.get_chemical_symbols() == atoms.get_chemical_symbols()


def test_from_ase_to_ase_preserves_positions():
    atoms = _two_oh_atoms()

    round_tripped = MolecularCrystal.from_ase(atoms).to_ase()

    assert np.allclose(round_tripped.get_positions(), atoms.get_positions(), atol=1e-9)


def test_pbc_and_cell_preserved():
    atoms = _two_oh_atoms()

    round_tripped = MolecularCrystal.from_ase(atoms).to_ase()

    assert np.allclose(round_tripped.cell.array, atoms.cell.array)
    assert tuple(round_tripped.get_pbc()) == tuple(atoms.get_pbc())


def test_to_ase_fallback_when_no_atom_indices():
    molecule_a = CrystalMolecule(Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]))
    molecule_b = CrystalMolecule(Atoms("N", positions=[[5.0, 0.0, 0.0]]))
    crystal = MolecularCrystal(np.eye(3) * 10.0, [molecule_a, molecule_b])

    atoms = crystal.to_ase()

    assert len(atoms) == 3
    assert atoms.get_chemical_symbols() == _molecule_concat_symbols(crystal)


def test_to_ase_fallback_when_partial_atom_indices():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())
    crystal.molecules[1].info.pop("atom_indices")

    atoms = crystal.to_ase()

    assert len(atoms) == 4
    assert atoms.get_chemical_symbols() == _molecule_concat_symbols(crystal)


def test_to_ase_fallback_when_indices_not_contiguous():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())
    crystal.molecules[0].info["atom_indices"] = [0, 4]

    atoms = crystal.to_ase()

    assert len(atoms) == 4
    assert atoms.get_chemical_symbols() == _molecule_concat_symbols(crystal)


def test_to_ase_fallback_when_indices_overlap():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())
    crystal.molecules[1].info["atom_indices"] = [0, 3]

    atoms = crystal.to_ase()

    assert len(atoms) == 4
    assert atoms.get_chemical_symbols() == _molecule_concat_symbols(crystal)


def test_rotate_then_to_ase_keeps_global_order():
    atoms = _two_oh_atoms()
    crystal = MolecularCrystal.from_ase(atoms)

    rotated = rotate_molecule(crystal, 0, axis=np.array([0.0, 0.0, 1.0]), angle=180.0)
    out = rotated.to_ase()

    assert out.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert np.allclose(out.positions[[1, 3]], atoms.positions[[1, 3]])
    assert not np.allclose(out.positions[[0, 2]], atoms.positions[[0, 2]])


def test_translate_then_to_ase_keeps_global_order():
    atoms = _two_oh_atoms()
    crystal = MolecularCrystal.from_ase(atoms)
    vector = np.array([1.0, 2.0, 3.0])

    translated = translate_molecule(crystal, 1, vector)
    out = translated.to_ase()

    assert out.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert np.allclose(out.positions[[0, 2]], atoms.positions[[0, 2]])
    assert np.allclose(out.positions[[1, 3]], atoms.positions[[1, 3]] + vector)


def test_replace_molecule_invalidates_atom_indices_or_falls_back():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())
    replacement = CrystalMolecule(Atoms("He", positions=[[0.0, 0.0, 0.0]]))

    replaced = replace_molecule(
        crystal,
        0,
        replacement,
        clash_threshold=0.0,
        max_rotation_attempts=0,
    )
    atoms = replaced.to_ase()

    assert len(atoms) == 3
    assert atoms.get_chemical_symbols() == _molecule_concat_symbols(replaced)


def test_supercell_clears_stale_atom_indices():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())

    supercell = crystal.get_supercell(2, 1, 1)
    atoms = supercell.to_ase()

    assert len(atoms) == 8
    assert all("atom_indices" not in molecule.info for molecule in supercell.molecules)


def test_identify_molecule_indices_basic():
    groups = identify_molecule_indices(_two_oh_atoms())

    assert sorted(len(group) for group in groups) == [2, 2]
    assert all(group == sorted(group) for group in groups)
    assert {index for group in groups for index in group} == set(range(4))


def test_identify_molecule_indices_no_double_assignment():
    groups = identify_molecule_indices(_two_oh_atoms())
    flat = [index for group in groups for index in group]

    assert len(flat) == len(set(flat))


def test_identify_molecule_indices_matches_identify_molecules():
    atoms = _two_oh_atoms()

    groups = identify_molecule_indices(atoms)
    molecule_groups = [molecule.info["atom_indices"] for molecule in identify_molecules(atoms)]

    assert sorted(groups) == sorted(molecule_groups)


def test_identify_molecule_indices_respects_bond_thresholds():
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )

    default_groups = identify_molecule_indices(atoms)
    split_groups = identify_molecule_indices(atoms, bond_thresholds={("O", "H"): 0.1})

    assert len(default_groups) == 1
    assert sorted(split_groups) == [[0], [1]]


def test_identify_molecule_indices_respects_exclude_indices():
    atoms = Atoms(
        symbols=["H", "O", "Cl"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [5.0, 5.0, 5.0],
        ],
    )

    groups = identify_molecule_indices(atoms, exclude_indices={1})

    assert sorted(groups) == [[0], [2]]
