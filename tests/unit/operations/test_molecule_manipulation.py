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

import copy
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.operations.molecule_manipulation import (
    MoleculeManipulator,
    MoleculeClashError,
    translate_atom,
    translate_molecule,
    rotate_molecule,
    replace_molecule,
)
from molcrys_kit.io import read_mol_crystal
from molcrys_kit.io.xyz import read_xyz
from molcrys_kit.utils.geometry import min_distance_between_atom_sets
from molcrys_kit.constants.config import KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z


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


@pytest.mark.parametrize("operation", ["translate", "rotate"])
def test_rigid_molecule_edit_preserves_frame_metadata(two_molecule_crystal, operation):
    crystal = two_molecule_crystal
    crystal.metadata = {"frame_id": 7, "nested": {"labels": ["source"]}}
    crystal.formula_moiety = "H2 O1, C1 H4"
    crystal.disorder_provenance = {"selection": [0, 1]}
    crystal.extra_arrays = {
        "site_id": np.arange(8),
        KEY_FRAC_X: np.zeros(8),
        KEY_FRAC_Y: np.zeros(8),
        KEY_FRAC_Z: np.zeros(8),
    }
    crystal._calc_results = {"energy": -1.0}
    crystal.molecules[0].get_graph()

    if operation == "translate":
        result = translate_molecule(crystal, 0, [0.1, 0.2, 0.3])
    else:
        result = rotate_molecule(crystal, 0, [0, 0, 1], 45)

    assert result.formula_moiety == crystal.formula_moiety
    assert result.metadata == crystal.metadata
    assert result.metadata is not crystal.metadata
    assert result.disorder_provenance == crystal.disorder_provenance
    np.testing.assert_array_equal(result.extra_arrays["site_id"], np.arange(8))
    assert not any(key in result.extra_arrays for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z))
    assert result._calc_results is None
    for left, right, edge in result.molecules[0].get_graph().edges(data=True):
        np.testing.assert_allclose(
            edge["vector"],
            result.molecules[0].positions[right] - result.molecules[0].positions[left],
        )
    result.metadata["nested"]["labels"].append("edited")
    result.disorder_provenance["selection"].append(2)
    assert crystal.metadata["nested"]["labels"] == ["source"]
    assert crystal.disorder_provenance["selection"] == [0, 1]


def test_translate_atom_preserves_source_and_topology(two_molecule_crystal):
    crystal = two_molecule_crystal
    before = crystal.molecules[0].get_positions().copy()
    edges = set(crystal.molecules[0].get_graph().edges())
    result = translate_atom(crystal, 0, 1, [0.2, 0.1, 0.0])
    np.testing.assert_allclose(crystal.molecules[0].positions, before)
    np.testing.assert_allclose(result.molecules[0].positions[1], before[1] + [0.2, 0.1, 0.0])
    np.testing.assert_allclose(result.molecules[0].positions[[0, 2]], before[[0, 2]])
    assert set(result.molecules[0].get_graph().edges()) == edges
    for left, right, data in result.molecules[0].get_graph().edges(data=True):
        np.testing.assert_allclose(
            data["vector"], result.molecules[0].positions[right] - result.molecules[0].positions[left]
        )


@pytest.mark.parametrize("index, vector", [(-1, [0, 0, 0]), (True, [0, 0, 0]), (0, [1, 2]), (0, [0, np.nan, 0])])
def test_translate_atom_rejects_invalid_input(two_molecule_crystal, index, vector):
    with pytest.raises(ValueError):
        translate_atom(two_molecule_crystal, 0, index, vector)


def test_translate_atom_fractional_displacement_invalidates_metadata(two_molecule_crystal):
    """Fractional edits move one atom and drop coordinate-derived frame data."""
    crystal = two_molecule_crystal
    crystal.metadata = {"frame_id": 3, "nested": {"labels": ["source"]}}
    crystal.formula_moiety = "H2 O1, C1 H4"
    crystal.disorder_provenance = {"selection": [0, 1]}
    crystal.extra_arrays = {
        "site_id": np.arange(8),
        KEY_FRAC_X: np.zeros(8),
        KEY_FRAC_Y: np.zeros(8),
        KEY_FRAC_Z: np.zeros(8),
    }
    crystal._calc_results = {"energy": -2.0}
    before = crystal.molecules[0].get_positions().copy()

    result = translate_atom(crystal, 0, 1, [0.1, -0.2, 0.3], fractional=True)

    np.testing.assert_allclose(crystal.molecules[0].positions, before)
    np.testing.assert_allclose(
        result.molecules[0].positions[1],
        before[1] + np.array([1.0, -2.0, 3.0]),
    )
    np.testing.assert_allclose(result.molecules[0].positions[[0, 2]], before[[0, 2]])
    assert result.formula_moiety == crystal.formula_moiety
    assert result.metadata == {"frame_id": 3, "nested": {"labels": ["source"]}}
    assert result.metadata is not crystal.metadata
    assert result.disorder_provenance == {"selection": [0, 1]}
    assert result.disorder_provenance is not crystal.disorder_provenance
    np.testing.assert_array_equal(result.extra_arrays["site_id"], np.arange(8))
    assert not any(key in result.extra_arrays for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z))
    assert result._calc_results is None
    assert crystal._calc_results == {"energy": -2.0}


# Helium is a spectator so the periodic carbon pair is not molecule 0.
# Both C–C images are inside the 1.9 Å cutoff (direct 1 Å and the a-axis image).
_PARALLEL_SECOND_MOLECULE_CIF = """\
data_parallel_second
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 2.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
He1 He 0.0 0.5 0.5
C1 C 0.0 0.0 0.0
C2 C 0.5 0.0 0.0
"""


def _contact_identity(bond):
    return (
        bond.left_global_index,
        bond.right_global_index,
        bond.right_image_shift,
    )


def _stored_contacts(molecule):
    indices = {int(index) for index in molecule.info["atom_indices"]}
    contacts = []
    for record in molecule.info["bond_records"]:
        left = int(record["left"])
        right = int(record["right"])
        assert {left, right} <= indices
        contacts.append(
            (
                left,
                right,
                tuple(int(value) for value in record["right_image_shift"]),
                [float(value) for value in record["vector"]],
            )
        )
    return contacts


def _assert_public_contacts_follow_positions(source, edited, molecule_index):
    """Bond identity stays fixed and each vector tracks the edited positions."""
    source_bonds = [
        bond
        for bond in source.get_bond_records()
        if bond.molecule_index == molecule_index
    ]
    edited_bonds = [
        bond
        for bond in edited.get_bond_records()
        if bond.molecule_index == molecule_index
    ]
    assert [_contact_identity(bond) for bond in edited_bonds] == [
        _contact_identity(bond) for bond in source_bonds
    ]
    assert source_bonds
    source_molecule = source.molecules[molecule_index]
    edited_molecule = edited.molecules[molecule_index]
    for previous, bond in zip(source_bonds, edited_bonds):
        old_delta = (
            source_molecule.positions[previous.right_local_index]
            - source_molecule.positions[previous.left_local_index]
        )
        new_delta = (
            edited_molecule.positions[bond.right_local_index]
            - edited_molecule.positions[bond.left_local_index]
        )
        correction = np.asarray(previous.vector_A, dtype=float) - old_delta
        np.testing.assert_allclose(bond.vector_A, new_delta + correction, atol=1e-8)


def _edit_molecule(crystal, molecule_index, operation):
    if operation == "translate_atom":
        return translate_atom(crystal, molecule_index, 1, [0.05, 0.0, 0.0])
    if operation == "translate_molecule":
        return translate_molecule(crystal, molecule_index, [0.05, 0.02, -0.03])
    if operation == "rotate_molecule":
        return rotate_molecule(crystal, molecule_index, [0.0, 0.0, 1.0], 90.0)
    raise AssertionError(operation)


@pytest.mark.parametrize(
    "operation",
    ["translate_atom", "translate_molecule", "rotate_molecule"],
)
def test_cif_second_molecule_keeps_parallel_bond_records(operation):
    """A non-first CIF molecule keeps every signed periodic contact."""
    crystal = read_mol_crystal(
        cif_text=_PARALLEL_SECOND_MOLECULE_CIF,
        attach_chemistry=False,
    )
    molecule = crystal.molecules[1]
    assert molecule.info["atom_indices"] == [1, 2]
    assert molecule.info["atom_indices"] != list(range(len(molecule)))
    source_contacts = _stored_contacts(molecule)
    identities = {(left, right, shift) for left, right, shift, _vector in source_contacts}
    assert identities == {
        (1, 2, (0, 0, 0)),
        (1, 2, (-1, 0, 0)),
    }
    source_records = copy.deepcopy(molecule.info["bond_records"])

    edited = _edit_molecule(crystal, 1, operation)

    assert crystal.molecules[1].info["bond_records"] == source_records
    edited_contacts = _stored_contacts(edited.molecules[1])
    assert [(left, right, shift) for left, right, shift, _vector in edited_contacts] == [
        (left, right, shift) for left, right, shift, _vector in source_contacts
    ]
    _assert_public_contacts_follow_positions(crystal, edited, 1)
    public = [
        (
            bond.left_global_index,
            bond.right_global_index,
            bond.right_image_shift,
            bond.vector_A,
        )
        for bond in edited.get_bond_records()
        if bond.molecule_index == 1
    ]
    assert [(left, right, shift) for left, right, shift, _vector in public] == [
        (1, 2, (-1, 0, 0)),
        (1, 2, (0, 0, 0)),
    ]


@pytest.mark.parametrize(
    "operation",
    ["translate_atom", "translate_molecule", "rotate_molecule"],
)
def test_methane_cif_second_molecule_bond_records_follow_coordinates(operation):
    """The synthetic methane fixture's second molecule keeps global endpoints."""
    path = Path(__file__).resolve().parents[2] / "data" / "cif" / "rigid_motion_methane.cif"
    crystal = read_mol_crystal(filepath=str(path), attach_chemistry=False)
    molecule = crystal.molecules[1]
    assert molecule.get_chemical_formula() == "CH4"
    assert [int(index) for index in molecule.info["atom_indices"]] != list(range(len(molecule)))
    assert molecule.info["bond_records"]
    source_contacts = _stored_contacts(molecule)
    source_records = copy.deepcopy(molecule.info["bond_records"])

    edited = _edit_molecule(crystal, 1, operation)

    assert crystal.molecules[1].info["bond_records"] == source_records
    edited_contacts = _stored_contacts(edited.molecules[1])
    assert [(left, right, shift) for left, right, shift, _vector in edited_contacts] == [
        (left, right, shift) for left, right, shift, _vector in source_contacts
    ]
    _assert_public_contacts_follow_positions(crystal, edited, 1)


def test_lazy_graph_rotation_refreshes_explicit_bond_vector():
    """A 90° rotation updates explicit records without building a graph first."""
    lattice = np.diag([10.0, 10.0, 10.0])
    spectator = CrystalMolecule(
        Atoms("He", positions=[[1.0, 1.0, 1.0]], cell=lattice, pbc=True),
        check_pbc=False,
    )
    spectator.info["atom_indices"] = [0]
    bonded = CrystalMolecule(
        Atoms(
            "CC",
            positions=[[4.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            cell=lattice,
            pbc=True,
        ),
        check_pbc=False,
    )
    bonded.info["atom_indices"] = [1, 2]
    bonded.info["bond_records"] = [
        {
            "left": 1,
            "right": 2,
            "right_image_shift": [0, 0, 0],
            "vector": [1.0, 0.0, 0.0],
            "role": "covalent",
        },
        {
            "left": 1,
            "right": 2,
            "right_image_shift": [1, 0, 0],
            "vector": [11.0, 0.0, 0.0],
            "role": "covalent",
        },
    ]
    assert bonded._graph is None
    crystal = MolecularCrystal(lattice, [spectator, bonded], pbc=(True, True, True))
    molecule = crystal.molecules[1]
    assert molecule._graph is None
    assert molecule.info["atom_indices"] == [1, 2]
    source_records = copy.deepcopy(molecule.info["bond_records"])

    edited = rotate_molecule(crystal, 1, [0.0, 0.0, 1.0], 90.0)

    assert crystal.molecules[1]._graph is None
    assert crystal.molecules[1].info["bond_records"] == source_records
    edited_molecule = edited.molecules[1]
    assert [int(index) for index in edited_molecule.info["atom_indices"]] == [1, 2]
    stored = {
        tuple(int(value) for value in record["right_image_shift"]): np.asarray(
            record["vector"], dtype=float
        )
        for record in edited_molecule.info["bond_records"]
    }
    assert set(stored) == {(0, 0, 0), (1, 0, 0)}
    np.testing.assert_allclose(stored[(0, 0, 0)], [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(stored[(1, 0, 0)], [10.0, 1.0, 0.0], atol=1e-8)
    bonds = [bond for bond in edited.get_bond_records() if bond.molecule_index == 1]
    assert [(bond.left_global_index, bond.right_global_index, bond.right_image_shift) for bond in bonds] == [
        (1, 2, (0, 0, 0)),
        (1, 2, (1, 0, 0)),
    ]
    for bond in bonds:
        delta = (
            edited_molecule.positions[bond.right_local_index]
            - edited_molecule.positions[bond.left_local_index]
        )
        expected = delta + np.asarray(bond.right_image_shift, dtype=float) @ lattice
        np.testing.assert_allclose(bond.vector_A, expected, atol=1e-8)


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
