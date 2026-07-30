"""Tests for molecular-crystal interpolation paths."""

import warnings

import numpy as np
from ase import Atoms

from molcrys_kit.io import read_extxyz, write_cif_sequence, write_poscar_sequence, write_trajectory
from molcrys_kit.operations.interpolation import (
    InterpolationMethod,
    NonRigidInterpolationWarning,
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


# ---------------------------------------------------------------------------
# Exact endpoint and NonRigidInterpolationWarning tests
# ---------------------------------------------------------------------------


def _non_rigid_water_crystals():
    """Create two crystals where molecule B has distorted internal geometry.

    The O-H bond lengths are changed, so Kabsch fit_rmsd > 0.
    This means the rigid-body interpolation cannot exactly reach B
    unless exact endpoint placement is used.
    """
    lattice = np.eye(3) * 10.0
    # A: standard water
    mol_a = _water_molecule()
    crystal_a = MolecularCrystal(lattice, [mol_a], pbc=(True, True, True))

    # B: distorted water (stretched bonds) with COM translation
    positions_b = np.array([
        [2.0, 1.0, 0.5],   # O
        [3.2, 1.0, 0.5],   # H (stretched from 0.95 to 1.2)
        [1.7, 2.1, 0.5],   # H (different angle and length)
    ])
    mol_b = _water_molecule(positions_b)
    crystal_b = MolecularCrystal(lattice, [mol_b], pbc=(True, True, True))
    return crystal_a, crystal_b


def test_exact_endpoint_with_non_rigid_target():
    """Lambda=1 frame must exactly match target B, even when fit_rmsd > 0."""
    crystal_a, crystal_b = _non_rigid_water_crystals()

    # Verify this is indeed a non-rigid case
    from molcrys_kit.operations.interpolation import match_molecules
    matches = match_molecules(crystal_a, crystal_b)
    assert matches[0].fit_rmsd > 0.01, "Test setup: target must be non-rigid"

    for method in InterpolationMethod:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NonRigidInterpolationWarning)
            frames = interpolate_crystal(
                crystal_a, crystal_b, method=method, n_images=5
            )
        # First frame is exact A
        np.testing.assert_allclose(
            frames[0].molecules[0].get_positions(),
            crystal_a.molecules[0].get_positions(),
            atol=1e-12,
        )
        # Last frame is exact B (reordered by atom_mapping into A order)
        match = matches[0]
        target_positions = crystal_b.molecules[match.idx_b].get_positions()[match.atom_mapping]
        np.testing.assert_allclose(
            frames[-1].molecules[0].get_positions(),
            target_positions,
            atol=1e-12,
            err_msg=f"method={method}: last frame != exact mapped target B",
        )


def test_exact_endpoint_with_permuted_atoms():
    """Exact endpoint works when B has atoms in a different order."""
    lattice = np.eye(3) * 10.0
    mol_a = _water_molecule()
    crystal_a = MolecularCrystal(lattice, [mol_a], pbc=(True, True, True))

    # B: same geometry but atoms permuted (H, O, H) instead of (O, H, H)
    pos_a = mol_a.get_positions()
    positions_b = pos_a[[1, 0, 2]] + np.array([2.0, 0.0, 0.0])
    mol_b = CrystalMolecule(
        Atoms(["H", "O", "H"], positions=positions_b, pbc=False), check_pbc=False
    )
    crystal_b = MolecularCrystal(lattice, [mol_b], pbc=(True, True, True))

    frames = interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=3)

    # Last frame should have positions geometrically equal to B
    # (reordered to A's atom order via atom_mapping)
    last_pos = frames[-1].molecules[0].get_positions()
    # The O atom in A is index 0; in B it's index 1. After mapping, frame[-1]
    # should have O at B's position for index 1 (mapped back to slot 0).
    b_o_pos = positions_b[1]  # O is at index 1 in B
    np.testing.assert_allclose(last_pos[0], b_o_pos, atol=1e-10)


def test_non_rigid_warning_emitted():
    """NonRigidInterpolationWarning is raised for non-rigid targets."""
    crystal_a, crystal_b = _non_rigid_water_crystals()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=3)
    non_rigid_warnings = [x for x in w if issubclass(x.category, NonRigidInterpolationWarning)]
    assert len(non_rigid_warnings) == 1
    assert "fit_rmsd" in str(non_rigid_warnings[0].message)


def test_no_warning_for_rigid_target():
    """No NonRigidInterpolationWarning for a purely rigid transformation."""
    crystal_a, crystal_b, _ = _single_water_crystals()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=3)
    non_rigid_warnings = [x for x in w if issubclass(x.category, NonRigidInterpolationWarning)]
    assert len(non_rigid_warnings) == 0


def test_partial_interpolation_endpoint_semantics():
    """With molecule_indices, selected molecules reach B; others stay at A."""
    crystal_a, crystal_b = _two_water_crystals()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonRigidInterpolationWarning)
        frames = interpolate_crystal(
            crystal_a, crystal_b, method="se3_screw", n_images=3,
            molecule_indices=[0],
        )

    # Molecule 0 at last frame should match crystal_b molecule 0
    target_0 = crystal_b.molecules[0].get_positions()
    np.testing.assert_allclose(
        frames[-1].molecules[0].get_positions(), target_0, atol=1e-8
    )
    # Molecule 1 (unselected) should remain at crystal_a positions
    np.testing.assert_allclose(
        frames[-1].molecules[1].get_positions(),
        crystal_a.molecules[1].get_positions(),
        atol=1e-12,
    )


def test_include_endpoints_false_unchanged():
    """When include_endpoints=False, behavior is unchanged (no exact endpoints)."""
    crystal_a, crystal_b, _ = _single_water_crystals()
    frames = interpolate_crystal(
        crystal_a, crystal_b, method="se3_screw", n_images=3,
        include_endpoints=False,
    )
    assert len(frames) == 3
    # No frame should exactly equal A or B
    pos_a = crystal_a.molecules[0].get_positions()
    pos_b = crystal_b.molecules[0].get_positions()
    for frame in frames:
        assert not np.allclose(frame.molecules[0].get_positions(), pos_a, atol=1e-8)
        assert not np.allclose(frame.molecules[0].get_positions(), pos_b, atol=1e-8)


def test_pbc_crossing_endpoint_continuous():
    """When B molecule crosses PBC, endpoint must be in the same image as interior frames."""
    lattice = np.eye(3) * 10.0
    # A: molecule near top of cell (COM x ~ 9.5)
    pos_a = np.array([[9.5, 5.0, 5.0], [9.5, 5.5, 5.0], [10.0, 5.0, 5.0]])
    mol_a = _water_molecule(pos_a)
    crystal_a = MolecularCrystal(lattice, [mol_a], pbc=(True, True, True))

    # B: molecule near bottom of cell (COM x ~ 0.5) — shortest path crosses PBC
    pos_b = np.array([[0.5, 5.0, 5.0], [0.5, 5.5, 5.0], [1.0, 5.0, 5.0]])
    mol_b = _water_molecule(pos_b)
    crystal_b = MolecularCrystal(lattice, [mol_b], pbc=(True, True, True))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonRigidInterpolationWarning)
        frames = interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=5)

    # Check consecutive COM displacements are bounded (no full-cell jump)
    coms = [np.mean(f.molecules[0].get_positions(), axis=0) for f in frames]
    for i in range(len(coms) - 1):
        displacement = np.linalg.norm(coms[i + 1] - coms[i])
        assert displacement < 2.0, (
            f"Frame {i}->{i+1} COM displacement {displacement:.2f} Å exceeds 2 Å"
        )


def test_no_warning_for_endpoints_only():
    """n_images=2 with include_endpoints=True has no interior frames; no warning."""
    crystal_a, crystal_b = _non_rigid_water_crystals()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames = interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=2)
    non_rigid = [x for x in w if issubclass(x.category, NonRigidInterpolationWarning)]
    assert len(non_rigid) == 0
    assert len(frames) == 2


def test_no_warning_for_single_image():
    """n_images=1 with include_endpoints=True returns only start; no warning."""
    crystal_a, crystal_b = _non_rigid_water_crystals()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames = interpolate_crystal(crystal_a, crystal_b, method="se3_screw", n_images=1)
    non_rigid = [x for x in w if issubclass(x.category, NonRigidInterpolationWarning)]
    assert len(non_rigid) == 0
    assert len(frames) == 1
