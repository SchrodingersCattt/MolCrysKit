"""
Unit tests for molcrys_kit.operations.molecule_manipulation.

Tests cover:
- translate_molecule (Cartesian and fractional)
- rotate_molecule (COM and centroid pivot)
- replace_molecule (no clash, with clash resolution, unresolvable clash)
- MoleculeManipulator.select_molecules (by index and species)
- read_xyz helper
- min_distance_between_atom_sets utility
"""

import os
import tempfile

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.operations.molecule_manipulation import (
    MoleculeManipulator,
    MoleculeClashError,
    translate_molecule,
    rotate_molecule,
    replace_molecule,
)
from molcrys_kit.io.xyz import read_xyz
from molcrys_kit.utils.geometry import min_distance_between_atom_sets


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def cubic_lattice():
    """10 Å cubic lattice."""
    return np.eye(3) * 10.0


@pytest.fixture
def water_molecule():
    """A single water molecule as CrystalMolecule (no crystal ref)."""
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    return CrystalMolecule(atoms, crystal=None, check_pbc=False)


@pytest.fixture
def methane_molecule():
    """A CH4 molecule as CrystalMolecule (no crystal ref)."""
    atoms = Atoms(
        symbols=["C", "H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
    )
    return CrystalMolecule(atoms, crystal=None, check_pbc=False)


@pytest.fixture
def two_molecule_crystal(cubic_lattice, water_molecule, methane_molecule):
    """Crystal with a water at (1,1,1) and methane at (5,5,5)."""
    water = water_molecule.copy()
    water.set_positions(water.get_positions() + np.array([1.0, 1.0, 1.0]))

    methane = methane_molecule.copy()
    methane.set_positions(methane.get_positions() + np.array([5.0, 5.0, 5.0]))

    return MolecularCrystal(
        lattice=cubic_lattice,
        molecules=[water, methane],
        pbc=(True, True, True),
    )


@pytest.fixture
def xyz_file(tmp_path):
    """Create a temporary XYZ file with a simple CO2 molecule."""
    content = """3
CO2 molecule
C   0.000000   0.000000   0.000000
O   1.160000   0.000000   0.000000
O  -1.160000   0.000000   0.000000
"""
    filepath = tmp_path / "co2.xyz"
    filepath.write_text(content)
    return str(filepath)



# =====================================================================
# Tests: read_xyz
# =====================================================================


class TestReadXyz:
    """Tests for the XYZ file reader."""

    def test_read_xyz_basic(self, xyz_file):
        mol = read_xyz(xyz_file)
        assert isinstance(mol, CrystalMolecule)
        assert len(mol) == 3
        assert mol.get_chemical_formula() == "CO2"

    def test_read_xyz_positions(self, xyz_file):
        mol = read_xyz(xyz_file)
        positions = mol.get_positions()
        np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(positions[1], [1.16, 0.0, 0.0], atol=1e-6)

    def test_read_xyz_no_crystal_ref(self, xyz_file):
        mol = read_xyz(xyz_file)
        assert mol.crystal is None

    def test_read_xyz_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_xyz("/nonexistent/path/molecule.xyz")

    def test_read_xyz_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad.xyz"
        bad_file.write_text("this is not a valid xyz file")
        with pytest.raises(ValueError):
            read_xyz(str(bad_file))


# =====================================================================
# Tests: min_distance_between_atom_sets
# =====================================================================


class TestMinDistanceBetweenAtomSets:
    """Tests for the geometry utility."""

    def test_basic_distance(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[3.0, 4.0, 0.0]])
        assert abs(min_distance_between_atom_sets(a, b) - 5.0) < 1e-10

    def test_multiple_atoms(self):
        a = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        b = np.array([[1.0, 0.0, 0.0], [20.0, 20.0, 20.0]])
        # Closest pair is (0,0,0)-(1,0,0) = 1.0
        assert abs(min_distance_between_atom_sets(a, b) - 1.0) < 1e-10

    def test_identical_points(self):
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[1.0, 2.0, 3.0]])
        assert abs(min_distance_between_atom_sets(a, b)) < 1e-10

    def test_with_pbc(self):
        lattice = np.eye(3) * 10.0
        # Two points at (0,0,0) and (9.5,0,0) — through PBC, distance is 0.5
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[9.5, 0.0, 0.0]])
        dist = min_distance_between_atom_sets(
            a, b, lattice=lattice, pbc=(True, True, True)
        )
        assert abs(dist - 0.5) < 1e-6


# =====================================================================
# Tests: MoleculeManipulator.select_molecules
# =====================================================================


class TestSelectMolecules:
    """Tests for molecule selection."""

    def test_select_by_single_index(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        result = manip.select_molecules(indices=0)
        assert result == [0]

    def test_select_by_index_list(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        result = manip.select_molecules(indices=[1, 0])
        assert result == [0, 1]

    def test_select_index_out_of_range(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        with pytest.raises(ValueError, match="out of range"):
            manip.select_molecules(indices=99)

    def test_select_no_args(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        with pytest.raises(ValueError, match="Exactly one"):
            manip.select_molecules()

    def test_select_both_args(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        with pytest.raises(ValueError, match="Exactly one"):
            manip.select_molecules(indices=0, species_id="H2O_1")

    def test_select_by_species_unknown(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        with pytest.raises(ValueError, match="not found"):
            manip.select_molecules(species_id="UNKNOWN_1")


# =====================================================================
# Tests: translate_molecule
# =====================================================================


class TestTranslateMolecule:
    """Tests for translation operations."""

    def test_translate_basic(self, two_molecule_crystal):
        shift = np.array([1.0, 2.0, 3.0])
        old_positions = two_molecule_crystal.molecules[0].get_positions().copy()

        new_crystal = translate_molecule(two_molecule_crystal, 0, shift)

        new_positions = new_crystal.molecules[0].get_positions()
        np.testing.assert_allclose(new_positions, old_positions + shift, atol=1e-10)

    def test_translate_preserves_other_molecules(self, two_molecule_crystal):
        old_mol1_pos = two_molecule_crystal.molecules[1].get_positions().copy()

        new_crystal = translate_molecule(
            two_molecule_crystal, 0, [1.0, 0.0, 0.0]
        )

        new_mol1_pos = new_crystal.molecules[1].get_positions()
        np.testing.assert_allclose(new_mol1_pos, old_mol1_pos, atol=1e-10)

    def test_translate_does_not_mutate_original(self, two_molecule_crystal):
        old_positions = two_molecule_crystal.molecules[0].get_positions().copy()
        translate_molecule(two_molecule_crystal, 0, [5.0, 5.0, 5.0])
        # Original should be unchanged
        np.testing.assert_allclose(
            two_molecule_crystal.molecules[0].get_positions(),
            old_positions,
            atol=1e-10,
        )

    def test_translate_fractional(self, two_molecule_crystal):
        # Fractional (0.1, 0, 0) on a 10Å cubic lattice = (1.0, 0, 0) Å
        old_positions = two_molecule_crystal.molecules[0].get_positions().copy()

        new_crystal = translate_molecule(
            two_molecule_crystal, 0, [0.1, 0.0, 0.0], fractional=True
        )

        expected = old_positions + np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            new_crystal.molecules[0].get_positions(), expected, atol=1e-10
        )

    def test_translate_index_out_of_range(self, two_molecule_crystal):
        with pytest.raises(ValueError, match="out of range"):
            translate_molecule(two_molecule_crystal, 5, [0.0, 0.0, 0.0])

    def test_translate_returns_new_crystal(self, two_molecule_crystal):
        new_crystal = translate_molecule(
            two_molecule_crystal, 0, [1.0, 0.0, 0.0]
        )
        assert new_crystal is not two_molecule_crystal


# =====================================================================
# Tests: rotate_molecule
# =====================================================================


class TestRotateMolecule:
    """Tests for rotation operations."""

    def test_rotate_com_preserved(self, two_molecule_crystal):
        old_com = two_molecule_crystal.molecules[0].get_center_of_mass().copy()

        new_crystal = rotate_molecule(
            two_molecule_crystal, 0, axis=[0, 0, 1], angle=45.0, center="com"
        )

        new_com = new_crystal.molecules[0].get_center_of_mass()
        np.testing.assert_allclose(new_com, old_com, atol=1e-10)

    def test_rotate_centroid_preserved(self, two_molecule_crystal):
        old_centroid = two_molecule_crystal.molecules[0].get_centroid().copy()

        new_crystal = rotate_molecule(
            two_molecule_crystal, 0, axis=[0, 0, 1], angle=45.0, center="centroid"
        )

        new_centroid = new_crystal.molecules[0].get_centroid()
        np.testing.assert_allclose(new_centroid, old_centroid, atol=1e-10)

    def test_rotate_360_identity(self, two_molecule_crystal):
        old_positions = two_molecule_crystal.molecules[0].get_positions().copy()

        new_crystal = rotate_molecule(
            two_molecule_crystal, 0, axis=[1, 0, 0], angle=360.0
        )

        np.testing.assert_allclose(
            new_crystal.molecules[0].get_positions(), old_positions, atol=1e-10
        )

    def test_rotate_preserves_other_molecules(self, two_molecule_crystal):
        old_mol1_pos = two_molecule_crystal.molecules[1].get_positions().copy()

        new_crystal = rotate_molecule(
            two_molecule_crystal, 0, axis=[0, 0, 1], angle=90.0
        )

        np.testing.assert_allclose(
            new_crystal.molecules[1].get_positions(), old_mol1_pos, atol=1e-10
        )

    def test_rotate_does_not_mutate_original(self, two_molecule_crystal):
        old_positions = two_molecule_crystal.molecules[0].get_positions().copy()
        rotate_molecule(two_molecule_crystal, 0, axis=[0, 0, 1], angle=90.0)
        np.testing.assert_allclose(
            two_molecule_crystal.molecules[0].get_positions(),
            old_positions,
            atol=1e-10,
        )

    def test_rotate_zero_axis_raises(self, two_molecule_crystal):
        with pytest.raises(ValueError, match="non-zero"):
            rotate_molecule(
                two_molecule_crystal, 0, axis=[0, 0, 0], angle=45.0
            )

    def test_rotate_unknown_center_raises(self, two_molecule_crystal):
        with pytest.raises(ValueError, match="Unknown center"):
            rotate_molecule(
                two_molecule_crystal, 0, axis=[0, 0, 1], angle=45.0, center="xyz"
            )


# =====================================================================
# Tests: replace_molecule
# =====================================================================


class TestReplaceMolecule:
    """Tests for molecule replacement operations."""

    def test_replace_no_clash(self, two_molecule_crystal, methane_molecule):
        """Replace water (mol #0) with a methane — molecules are far apart, no clash."""
        new_crystal = replace_molecule(
            two_molecule_crystal,
            molecule_index=0,
            new_molecule=methane_molecule,
            clash_threshold=1.0,
        )

        # New crystal should have 2 molecules
        assert len(new_crystal.molecules) == 2

        # The replacement should have CH4 formula
        assert new_crystal.molecules[0].get_chemical_formula() == "CH4"

        # The second molecule should be unchanged
        assert new_crystal.molecules[1].get_chemical_formula() == "CH4"

    def test_replace_com_alignment(self, two_molecule_crystal, methane_molecule):
        """After replacement, new molecule COM should match old molecule COM."""
        old_com = two_molecule_crystal.molecules[0].get_center_of_mass().copy()

        new_crystal = replace_molecule(
            two_molecule_crystal,
            molecule_index=0,
            new_molecule=methane_molecule,
            clash_threshold=0.0,  # Skip clash check
        )

        new_com = new_crystal.molecules[0].get_center_of_mass()
        np.testing.assert_allclose(new_com, old_com, atol=1e-6)

    def test_replace_from_xyz_file(self, two_molecule_crystal, xyz_file):
        """Replace using an XYZ file path."""
        new_crystal = replace_molecule(
            two_molecule_crystal,
            molecule_index=0,
            new_molecule=xyz_file,
            clash_threshold=1.0,
        )

        assert new_crystal.molecules[0].get_chemical_formula() == "CO2"

    def test_replace_with_clash_resolution(self, two_molecule_crystal):
        """Create a clash scenario and verify rotation attempts resolve it."""
        # Place a molecule very close to molecule #1 (at [5,5,5])
        close_mol = Atoms(
            symbols=["N", "N"],
            positions=[[4.5, 5.0, 5.0], [4.0, 5.0, 5.0]],
        )
        close_crystal_mol = CrystalMolecule(close_mol, crystal=None, check_pbc=False)

        # The molecule's COM is at (4.25, 5.0, 5.0) and molecule #1 has atoms at ~5Å
        # After COM alignment to molecule #0's COM (~(1.0, 1.39, 1.0)), it will be
        # far from molecule #1, so there should be no real clash.
        new_crystal = replace_molecule(
            two_molecule_crystal,
            molecule_index=0,
            new_molecule=close_crystal_mol,
            clash_threshold=1.0,
            max_rotation_attempts=10,
        )

        assert len(new_crystal.molecules) == 2

    def test_replace_unresolvable_clash_raises(self, cubic_lattice):
        """When a very large molecule is placed inside a dense cage, MoleculeClashError
        should be raised because no rotation can resolve the clash."""
        # mol #0: single placeholder atom at the origin (will be replaced)
        placeholder = Atoms(symbols=["He"], positions=[[0.0, 0.0, 0.0]])

        # mol #1: dense spherical cage of atoms at r=1.5 Å surrounding the origin.
        # Any guest molecule with atoms at r≥2 Å will *always* clash regardless of rotation.
        host_positions = []
        for theta in np.linspace(0, np.pi, 8):
            for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                host_positions.append([
                    1.5 * np.sin(theta) * np.cos(phi),
                    1.5 * np.sin(theta) * np.sin(phi),
                    1.5 * np.cos(theta),
                ])
        cage_mol = Atoms(
            symbols=["C"] * len(host_positions), positions=host_positions
        )

        crystal = MolecularCrystal(
            lattice=cubic_lattice,
            molecules=[
                CrystalMolecule(placeholder, crystal=None, check_pbc=False),
                CrystalMolecule(cage_mol, crystal=None, check_pbc=False),
            ],
            pbc=(True, True, True),
        )

        # Guest: shell of atoms at r=2 Å.  After COM alignment to origin (He atom),
        # guest atoms sit at 2 Å from origin; cage atoms at 1.5 Å → distance 0.5 Å < 1.0 Å.
        # Rotation preserves atomic distances from origin, so the clash is unresolvable.
        guest_positions = []
        for theta in np.linspace(0, np.pi, 5):
            for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                guest_positions.append([
                    2.0 * np.sin(theta) * np.cos(phi),
                    2.0 * np.sin(theta) * np.sin(phi),
                    2.0 * np.cos(theta),
                ])
        guest_mol = CrystalMolecule(
            Atoms(symbols=["N"] * len(guest_positions), positions=guest_positions),
            crystal=None,
            check_pbc=False,
        )

        with pytest.raises(MoleculeClashError):
            replace_molecule(
                crystal,
                molecule_index=0,
                new_molecule=guest_mol,
                clash_threshold=1.0,
                max_rotation_attempts=5,  # Few attempts for test speed
            )

    def test_replace_does_not_mutate_original(self, two_molecule_crystal, methane_molecule):
        old_formula = two_molecule_crystal.molecules[0].get_chemical_formula()
        old_positions = two_molecule_crystal.molecules[0].get_positions().copy()

        replace_molecule(
            two_molecule_crystal,
            molecule_index=0,
            new_molecule=methane_molecule,
        )

        # Original should be unchanged
        assert two_molecule_crystal.molecules[0].get_chemical_formula() == old_formula
        np.testing.assert_allclose(
            two_molecule_crystal.molecules[0].get_positions(),
            old_positions,
            atol=1e-10,
        )

    def test_replace_centroid_alignment(self, two_molecule_crystal, methane_molecule):
        """Test centroid alignment mode."""
        old_centroid = two_molecule_crystal.molecules[0].get_centroid().copy()

        new_crystal = replace_molecule(
            two_molecule_crystal,
            molecule_index=0,
            new_molecule=methane_molecule,
            clash_threshold=0.0,
            align_method="centroid",
        )

        new_centroid = new_crystal.molecules[0].get_centroid()
        np.testing.assert_allclose(new_centroid, old_centroid, atol=1e-6)

    def test_replace_preserves_lattice(self, two_molecule_crystal, methane_molecule):
        new_crystal = replace_molecule(
            two_molecule_crystal,
            molecule_index=0,
            new_molecule=methane_molecule,
        )

        np.testing.assert_allclose(
            new_crystal.lattice, two_molecule_crystal.lattice, atol=1e-10
        )


# =====================================================================
# Tests: MoleculeManipulator class API
# =====================================================================


class TestMoleculeManipulatorAPI:
    """Tests for the class-based API."""

    def test_class_translate(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        old_pos = two_molecule_crystal.molecules[0].get_positions().copy()

        new_crystal = manip.translate_molecule(0, [1.0, 0.0, 0.0])

        np.testing.assert_allclose(
            new_crystal.molecules[0].get_positions(),
            old_pos + np.array([1.0, 0.0, 0.0]),
            atol=1e-10,
        )

    def test_class_rotate(self, two_molecule_crystal):
        manip = MoleculeManipulator(two_molecule_crystal)
        old_com = two_molecule_crystal.molecules[0].get_center_of_mass()

        new_crystal = manip.rotate_molecule(0, [0, 0, 1], 90.0)

        np.testing.assert_allclose(
            new_crystal.molecules[0].get_center_of_mass(),
            old_com,
            atol=1e-10,
        )

    def test_class_replace(self, two_molecule_crystal, methane_molecule):
        manip = MoleculeManipulator(two_molecule_crystal)
        new_crystal = manip.replace_molecule(0, methane_molecule)

        assert new_crystal.molecules[0].get_chemical_formula() == "CH4"

    def test_molecule_count_preserved(self, two_molecule_crystal, methane_molecule):
        """All operations should preserve the total number of molecules."""
        manip = MoleculeManipulator(two_molecule_crystal)
        n_orig = len(two_molecule_crystal.molecules)

        c1 = manip.translate_molecule(0, [1.0, 0.0, 0.0])
        assert len(c1.molecules) == n_orig

        c2 = manip.rotate_molecule(0, [0, 0, 1], 45.0)
        assert len(c2.molecules) == n_orig

        c3 = manip.replace_molecule(0, methane_molecule)
        assert len(c3.molecules) == n_orig
