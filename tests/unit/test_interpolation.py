"""Tests for molecular-crystal interpolation paths."""

import numpy as np
from ase import Atoms

from molcrys_kit.io import read_extxyz, write_cif_sequence, write_poscar_sequence, write_trajectory
from molcrys_kit.operations.interpolation import (
    InterpolationMethod,
    best_atom_mapping,
    find_flipping_molecules,
    interpolate_crystal,
    interpolate_molecule,
    match_molecules,
)
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.utils.geometry import get_rotation_matrix


def _water_molecule(positions=None):
    if positions is None:
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.95, 0.0, 0.0],
                [-0.24, 0.92, 0.0],
            ]
        )
    return CrystalMolecule(
        Atoms("OHH", positions=np.asarray(positions, dtype=float), pbc=False),
        check_pbc=False,
    )


def _single_water_crystals():
    lattice = np.eye(3) * 10.0
    mol_a = _water_molecule()
    com_a = mol_a.get_center_of_mass()
    rotation = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    translation = np.array([2.0, 1.0, 0.5])
    positions_b = (mol_a.get_positions() - com_a) @ rotation.T + com_a + translation
    crystal_a = MolecularCrystal(lattice, [mol_a], pbc=(True, True, True))
    crystal_b = MolecularCrystal(lattice, [_water_molecule(positions_b)], pbc=(True, True, True))
    return crystal_a, crystal_b, translation


def _two_water_crystals():
    lattice = np.eye(3) * 20.0
    mol0 = _water_molecule()
    mol1 = _water_molecule(_water_molecule().get_positions() + np.array([6.0, 0.0, 0.0]))
    crystal_a = MolecularCrystal(lattice, [mol0, mol1], pbc=(True, True, True))

    rotation = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    com0 = mol0.get_center_of_mass()
    positions0_b = (mol0.get_positions() - com0) @ rotation.T + com0 + np.array([1.0, 0.0, 0.0])
    crystal_b = MolecularCrystal(lattice, [_water_molecule(positions0_b), mol1], pbc=(True, True, True))
    return crystal_a, crystal_b


def test_best_atom_mapping_handles_permuted_atoms():
    mol_a = _water_molecule()
    positions = mol_a.get_positions()[[0, 2, 1]]
    mol_b = CrystalMolecule(Atoms(["O", "H", "H"], positions=positions, pbc=False), check_pbc=False)
    mapping = best_atom_mapping(mol_a, mol_b)
    assert list(mapping) in ([0, 1, 2], [0, 2, 1])
    assert mol_b.get_chemical_symbols()[mapping[0]] == "O"


def test_match_molecules_decomposes_translation_and_rotation():
    crystal_a, crystal_b, translation = _single_water_crystals()
    match = match_molecules(crystal_a, crystal_b)[0]
    np.testing.assert_allclose(match.com_translation, translation, atol=1e-10)
    assert abs(match.angle_deg - 90.0) < 1e-8
    assert match.fit_rmsd < 1e-10


def test_interpolation_methods_preserve_endpoints():
    crystal_a, crystal_b, _ = _single_water_crystals()
    target_positions = crystal_b.molecules[0].get_positions()
    for method in InterpolationMethod:
        frames = interpolate_crystal(crystal_a, crystal_b, method=method, n_images=5)
        assert len(frames) == 5
        np.testing.assert_allclose(
            frames[0].molecules[0].get_positions(),
            crystal_a.molecules[0].get_positions(),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            frames[-1].molecules[0].get_positions(),
            target_positions,
            atol=1e-8,
        )


def test_interpolate_molecule_keeps_unselected_molecules_fixed():
    crystal_a, crystal_b = _two_water_crystals()
    frames = interpolate_molecule(
        crystal_a,
        crystal_b,
        0,
        method="com_alignment",
        n_images=3,
    )
    np.testing.assert_allclose(
        frames[1].molecules[1].get_positions(),
        crystal_a.molecules[1].get_positions(),
        atol=1e-10,
    )
    assert not np.allclose(
        frames[1].molecules[0].get_positions(),
        crystal_a.molecules[0].get_positions(),
    )


def test_find_flipping_molecules_reports_changed_pose_only():
    crystal_a, crystal_b = _two_water_crystals()
    selected = find_flipping_molecules(
        crystal_a,
        crystal_b,
        rmsd_threshold=0.1,
        angle_threshold=5.0,
    )
    assert selected == [0]


def test_sequence_writers_create_expected_files(tmp_path):
    crystal_a, crystal_b, _ = _single_water_crystals()
    frames = interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=2)

    poscar_paths = write_poscar_sequence(frames, str(tmp_path / "poscars"))
    cif_paths = write_cif_sequence(frames, str(tmp_path / "cifs"))

    assert len(poscar_paths) == 2
    assert len(cif_paths) == 2
    assert (tmp_path / "poscars" / "00" / "POSCAR").exists()
    assert (tmp_path / "poscars" / "01" / "POSCAR").exists()
    assert (tmp_path / "cifs" / "frame_000.cif").exists()
    assert (tmp_path / "cifs" / "frame_001.cif").exists()


def test_write_trajectory_extxyz_round_trips_interpolation_frames(tmp_path):
    crystal_a, crystal_b, _ = _single_water_crystals()
    frames = interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=3)
    path = tmp_path / "path.extxyz"

    written = write_trajectory(
        frames,
        str(path),
        format="extxyz",
        info=[{"lambda_index": i} for i in range(len(frames))],
    )
    restored = read_extxyz(written, index=":")

    assert len(restored) == 3
    assert restored[1].metadata["lambda_index"] == 1
    np.testing.assert_allclose(restored[-1].lattice, crystal_b.lattice)


def test_write_trajectory_xyz_creates_multiframe_file(tmp_path):
    crystal_a, crystal_b, _ = _single_water_crystals()
    frames = interpolate_crystal(crystal_a, crystal_b, method="com_so3", n_images=2)
    path = tmp_path / "path.xyz"

    written = write_trajectory(frames, str(path), format="xyz")

    assert written.endswith("path.xyz")
    assert path.exists()
    assert path.read_text().count("\n3\n") == 1 or path.read_text().startswith("3\n")
