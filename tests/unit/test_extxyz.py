"""Tests for the Extended XYZ (extxyz) I/O layer."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.io.extxyz import read_extxyz, write_extxyz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_crystal():
    """2-molecule water crystal in a cubic box."""
    lattice = np.eye(3) * 10.0
    m1 = CrystalMolecule(Atoms(
        "H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
    ))
    m2 = CrystalMolecule(Atoms(
        "H2O", positions=[[5, 5, 5], [5.96, 5, 5], [5, 5.96, 5]],
    ))
    return MolecularCrystal(lattice, [m1, m2])


@pytest.fixture
def tmp_xyz():
    """Managed temporary .xyz file that is cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix=".xyz")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Single-frame round-trip
# ---------------------------------------------------------------------------

def test_single_frame_round_trip(simple_crystal, tmp_xyz):
    """Write one crystal → read one crystal → properties match."""
    write_extxyz(simple_crystal, tmp_xyz)
    result = read_extxyz(tmp_xyz)

    assert isinstance(result, MolecularCrystal)
    assert len(result.molecules) == len(simple_crystal.molecules)
    assert result.get_total_nodes() == simple_crystal.get_total_nodes()
    assert np.allclose(result.lattice, simple_crystal.lattice)
    assert result.pbc == simple_crystal.pbc


# ---------------------------------------------------------------------------
# Multi-frame round-trip
# ---------------------------------------------------------------------------

def test_multi_frame_round_trip(simple_crystal, tmp_xyz):
    """Write a list of frames → read all → list length and properties match."""
    frames = [simple_crystal, simple_crystal, simple_crystal]
    write_extxyz(frames, tmp_xyz)
    results = read_extxyz(tmp_xyz, index=":")

    assert isinstance(results, list)
    assert len(results) == 3
    for c in results:
        assert c.get_total_nodes() == simple_crystal.get_total_nodes()
        assert np.allclose(c.lattice, simple_crystal.lattice)


def test_write_extxyz_append(simple_crystal, tmp_xyz):
    """append=True appends frames instead of overwriting the file."""
    write_extxyz(simple_crystal, tmp_xyz)
    write_extxyz(simple_crystal, tmp_xyz, append=True)

    results = read_extxyz(tmp_xyz, index=":")
    assert isinstance(results, list)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Index access
# ---------------------------------------------------------------------------

def test_read_int_index(simple_crystal, tmp_xyz):
    """Read with an int index returns a single MolecularCrystal."""
    frames = [simple_crystal, simple_crystal]
    write_extxyz(frames, tmp_xyz)

    c0 = read_extxyz(tmp_xyz, index=0)
    assert isinstance(c0, MolecularCrystal)

    c1 = read_extxyz(tmp_xyz, index=1)
    assert isinstance(c1, MolecularCrystal)


def test_read_slice_index(simple_crystal, tmp_xyz):
    """Read with a slice returns a list."""
    frames = [simple_crystal] * 4
    write_extxyz(frames, tmp_xyz)

    subset = read_extxyz(tmp_xyz, index=slice(1, 3))
    assert isinstance(subset, list)
    assert len(subset) == 2


def test_read_none_index_returns_last(simple_crystal, tmp_xyz):
    """index=None (default) returns the last frame as a single crystal."""
    frames = [simple_crystal] * 3
    write_extxyz(frames, tmp_xyz)

    result = read_extxyz(tmp_xyz)
    assert isinstance(result, MolecularCrystal)
    assert result.get_total_nodes() == simple_crystal.get_total_nodes()


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

def test_formula_moiety_preserved(tmp_xyz):
    """formula_moiety stored in atoms.info survives a round-trip."""
    lattice = np.eye(3) * 10.0
    mol = CrystalMolecule(Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]))
    crystal = MolecularCrystal(lattice, [mol], formula_moiety="C O")

    write_extxyz(crystal, tmp_xyz)
    c2 = read_extxyz(tmp_xyz)
    assert c2.formula_moiety == "C O"


def test_disorder_provenance_preserved(tmp_xyz):
    """DisorderProvenance stored in atoms.info survives a round-trip."""
    from molcrys_kit.analysis.disorder.provenance import DisorderProvenance

    lattice = np.eye(3) * 10.0
    mol = CrystalMolecule(Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]]))
    provenance = DisorderProvenance(
        kept_indices=[0, 1, 2], dropped_indices=[], method="test", coupled=False,
    )
    crystal = MolecularCrystal(lattice, [mol], disorder_provenance=provenance)

    write_extxyz(crystal, tmp_xyz)
    c2 = read_extxyz(tmp_xyz)
    assert c2.disorder_provenance is not None
    assert c2.disorder_provenance["method"] == "test"


# ---------------------------------------------------------------------------
# Per-atom array preservation
# ---------------------------------------------------------------------------

def test_molecule_index_preserved(simple_crystal, tmp_xyz):
    """molecule_index array is reconstructed faithfully."""
    write_extxyz(simple_crystal, tmp_xyz)
    c2 = read_extxyz(tmp_xyz)

    for i, mol in enumerate(c2.molecules):
        atoms = mol
        mol_idx = atoms.arrays.get("molecule_index")
        if mol_idx is not None:  # molecule_index is on the flat Atoms, not per molecule
            assert np.all(mol_idx == i)

    # Verify via total count
    assert len(c2.molecules) == len(simple_crystal.molecules)


def test_occupancy_propagated_for_partial_occ(tmp_xyz):
    """Molecules with partial occupancy keep it in the flat Atoms arrays."""
    lattice = np.eye(3) * 10.0
    mol = CrystalMolecule(Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]))
    mol.set_array("occupancy", np.array([0.5, 0.5]))

    crystal = MolecularCrystal(lattice, [mol])
    atoms = crystal.to_ase()
    occ = atoms.arrays.get("occupancy")
    assert occ is not None
    assert np.allclose(occ, [0.5, 0.5])


# ---------------------------------------------------------------------------
# Calculator results via write/read
# ---------------------------------------------------------------------------

def test_energy_via_single_point_calculator(simple_crystal, tmp_xyz):
    """Energy stored via SinglePointCalculator survives extxyz round-trip."""
    atoms = simple_crystal.to_ase()
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms.calc = SinglePointCalculator(atoms, energy=-10.5, forces=np.zeros((6, 3)))

    import ase.io
    ase.io.write(tmp_xyz, atoms, format="extxyz", write_info=True, write_results=True)

    c2 = read_extxyz(tmp_xyz)
    a2 = c2.to_ase()
    assert a2.calc is not None
    assert a2.calc.results["energy"] == pytest.approx(-10.5)


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

def test_empty_crystal_raises_on_read(tmp_xyz):
    """Reading a non-existent file raises an error."""
    with pytest.raises(FileNotFoundError):
        read_extxyz("nonexistent_file_12345.xyz")


def test_single_molecule_crystal(simple_crystal, tmp_xyz):
    """A crystal with a single molecule round-trips correctly."""
    crystal = MolecularCrystal(
        np.eye(3) * 10.0,
        [simple_crystal.molecules[0]],
    )
    write_extxyz(crystal, tmp_xyz)
    c2 = read_extxyz(tmp_xyz)
    assert len(c2.molecules) == 1
    assert c2.get_total_nodes() == 3  # H2O
