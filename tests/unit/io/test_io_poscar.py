"""
Unit tests for VASP POSCAR/CONTCAR input and output.
"""

import warnings

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.constants.config import (
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_LABEL,
    KEY_OCCUPANCY,
)
from molcrys_kit.io import read_poscar, write_poscar
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule


def _write_temp_poscar(tmp_path, content):
    path = tmp_path / "POSCAR"
    path.write_text(content, encoding="utf-8")
    return str(path)


def _all_molecule_symbols(crystal):
    symbols = []
    for molecule in crystal.molecules:
        symbols.extend(molecule.get_chemical_symbols())
    return symbols


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


def _coordinate_rows(poscar):
    lines = poscar.splitlines()
    coord_header_index = next(
        i for i, line in enumerate(lines) if line.strip() in {"Direct", "Cartesian"}
    )
    return lines[coord_header_index + 1 :]


def test_round_trip_string_match(crystal_single_water, tmp_path):
    poscar_path = tmp_path / "POSCAR"
    poscar = write_poscar(crystal_single_water, filename=str(poscar_path))

    parsed = read_poscar(str(poscar_path))

    assert poscar.startswith("MolCrysKit")
    assert np.allclose(parsed.lattice, crystal_single_water.lattice)
    assert len(_all_molecule_symbols(parsed)) == crystal_single_water.get_total_nodes()
    assert sorted(_all_molecule_symbols(parsed)) == sorted(["O", "H", "H"])


def test_direct_vs_cartesian(crystal_single_water):
    direct = write_poscar(crystal_single_water, direct=True)
    cartesian = write_poscar(crystal_single_water, direct=False)

    assert "Direct" in direct.splitlines()
    assert "Cartesian" in cartesian.splitlines()


def test_species_sorted(crystal_single_water):
    poscar = write_poscar(crystal_single_water, sort=True)
    lines = poscar.splitlines()

    assert lines[5].split() == ["H", "O"]
    assert lines[6].split() == ["2", "1"]


def test_round_trip_preserves_order(tmp_path):
    atoms = _two_oh_atoms()
    crystal = MolecularCrystal.from_ase(atoms)
    poscar_path = tmp_path / "POSCAR"

    write_poscar(crystal, filename=str(poscar_path))
    parsed = read_poscar(str(poscar_path))

    assert parsed.to_ase().get_chemical_symbols() == atoms.get_chemical_symbols()


def test_write_poscar_default_sort_is_false():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())

    poscar = write_poscar(crystal)
    lines = poscar.splitlines()

    assert lines[5].split() == ["O", "H"]
    assert lines[6].split() == ["2", "2"]


def test_write_poscar_sort_true_still_alphabetises():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())

    poscar = write_poscar(crystal, sort=True)
    lines = poscar.splitlines()

    assert lines[5].split() == ["H", "O"]
    assert lines[6].split() == ["2", "2"]


def test_lossy_occupancy_warning_and_comment(cubic_lattice_10, water_atoms):
    molecule = CrystalMolecule(water_atoms)
    molecule.set_array(KEY_OCCUPANCY, np.array([1.0, 0.5, 1.0]))
    crystal = MolecularCrystal(cubic_lattice_10, [molecule])

    with pytest.warns(UserWarning, match="POSCAR cannot represent"):
        poscar = write_poscar(crystal)

    first_line = poscar.splitlines()[0]
    assert "lossy export" in first_line
    assert "occupancy" in first_line


def test_lossy_occupancy_warning_with_reorder():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())
    for molecule in crystal.molecules:
        occupancies = np.ones(len(molecule))
        if 1 in molecule.info["atom_indices"]:
            occupancies[molecule.info["atom_indices"].index(1)] = 0.5
        molecule.set_array(KEY_OCCUPANCY, occupancies)

    with pytest.warns(UserWarning, match="POSCAR cannot represent"):
        poscar = write_poscar(crystal)

    assert "lossy export" in poscar.splitlines()[0]
    assert "occupancy" in poscar.splitlines()[0]


def test_lossy_label_silent_when_default(cubic_lattice_10, water_atoms):
    molecule = CrystalMolecule(water_atoms)
    molecule.set_array(KEY_LABEL, np.array(["O1", "H2", "H3"]))
    crystal = MolecularCrystal(cubic_lattice_10, [molecule])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        poscar = write_poscar(crystal)

    assert not record
    assert "lossy export" not in poscar.splitlines()[0]


def test_selective_dynamics_round_trip(crystal_single_water):
    selective_dynamics = np.array(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
        ]
    )

    poscar = write_poscar(
        crystal_single_water,
        selective_dynamics=selective_dynamics,
        sort=False,
    )

    lines = poscar.splitlines()
    assert "Selective dynamics" in lines
    flags = [row.split()[-3:] for row in _coordinate_rows(poscar)]
    assert flags == [["T", "T", "F"], ["F", "T", "T"], ["T", "F", "T"]]


def test_selective_dynamics_aligns_with_global_indices():
    crystal = MolecularCrystal.from_ase(_two_oh_atoms())
    selective_dynamics = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, True],
        ]
    )

    poscar = write_poscar(crystal, selective_dynamics=selective_dynamics)

    flags = [row.split()[-3:] for row in _coordinate_rows(poscar)]
    assert flags == [["T", "F", "F"], ["F", "T", "F"], ["F", "F", "T"], ["T", "T", "T"]]


def test_default_metadata_passthrough_when_no_atom_indices(crystal_single_water):
    poscar = write_poscar(crystal_single_water)

    assert "lossy export" not in poscar.splitlines()[0]
    assert "Direct" in poscar.splitlines()


def test_wrap_keeps_inside_cell(cubic_lattice_10, water_atoms):
    shifted = water_atoms.copy()
    shifted.positions += np.array([12.0, -3.0, 10.0])
    crystal = MolecularCrystal(cubic_lattice_10, [CrystalMolecule(shifted)])

    poscar = write_poscar(crystal, direct=True, wrap=True, sort=False)
    coords = np.array(
        [[float(value) for value in row.split()[:3]] for row in _coordinate_rows(poscar)]
    )

    assert np.all(coords >= 0.0)
    assert np.all(coords < 1.0)


def test_read_poscar_defaults_metadata(tmp_path):
    poscar_path = _write_temp_poscar(
        tmp_path,
        """Minimal POSCAR
1.0
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
O H
1 2
Direct
0.0 0.0 0.0
0.0757 0.0586 0.0
0.9243 0.0586 0.0
""",
    )

    crystal = read_poscar(poscar_path)

    for molecule in crystal.molecules:
        assert np.allclose(molecule.arrays[KEY_OCCUPANCY], 1.0)
        assert np.all(molecule.arrays[KEY_DISORDER_GROUP] == 0)
        assert all(value == "" for value in molecule.arrays[KEY_ASSEMBLY])
        assert list(molecule.arrays[KEY_LABEL]) == molecule.get_chemical_symbols()
