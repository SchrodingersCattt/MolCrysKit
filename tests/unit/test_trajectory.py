"""Tests for the CrystalTrajectory container."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.trajectory import (
    CrystalTrajectory,
    CrystalTrajectoryReader,
    CrystalTrajectoryWriter,
)


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
    """Managed temporary .xyz file."""
    fd, path = tempfile.mkstemp(suffix=".xyz")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Writer → Reader round-trip
# ---------------------------------------------------------------------------

def test_write_read_single_frame(simple_crystal, tmp_xyz):
    """Write one frame, read it back."""
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        traj.write(simple_crystal)

    reader = CrystalTrajectory(tmp_xyz, "r")
    assert len(reader) == 1
    frame = reader[0]
    assert isinstance(frame, MolecularCrystal)
    assert frame.get_total_nodes() == simple_crystal.get_total_nodes()


def test_write_read_multi_frame(simple_crystal, tmp_xyz):
    """Write 3 frames, read them all."""
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        for _ in range(3):
            traj.write(simple_crystal)

    reader = CrystalTrajectory(tmp_xyz, "r")
    assert len(reader) == 3
    for frame in reader:
        assert isinstance(frame, MolecularCrystal)
        assert frame.get_total_nodes() == simple_crystal.get_total_nodes()


def test_len_reader(simple_crystal, tmp_xyz):
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        for _ in range(5):
            traj.write(simple_crystal)

    reader = CrystalTrajectory(tmp_xyz, "r")
    assert len(reader) == 5


def test_getitem_int(simple_crystal, tmp_xyz):
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        traj.write(simple_crystal)
        traj.write(simple_crystal)

    reader = CrystalTrajectory(tmp_xyz, "r")
    assert isinstance(reader[0], MolecularCrystal)
    assert isinstance(reader[1], MolecularCrystal)


def test_getitem_slice(simple_crystal, tmp_xyz):
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        for _ in range(4):
            traj.write(simple_crystal)

    reader = CrystalTrajectory(tmp_xyz, "r")
    subset = reader[0:2]
    assert isinstance(subset, list)
    assert len(subset) == 2


def test_iteration(simple_crystal, tmp_xyz):
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        for _ in range(3):
            traj.write(simple_crystal)

    reader = CrystalTrajectory(tmp_xyz, "r")
    count = 0
    for frame in reader:
        assert isinstance(frame, MolecularCrystal)
        count += 1
    assert count == 3


# ---------------------------------------------------------------------------
# Calculator results
# ---------------------------------------------------------------------------

def test_energy_round_trip(simple_crystal, tmp_xyz):
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        traj.write(simple_crystal, energy=-5.0)
        traj.write(simple_crystal, energy=-5.5)

    reader = CrystalTrajectory(tmp_xyz, "r")
    e0 = reader[0].to_ase().calc.results["energy"]
    e1 = reader[1].to_ase().calc.results["energy"]
    assert e0 == pytest.approx(-5.0)
    assert e1 == pytest.approx(-5.5)


def test_forces_round_trip(simple_crystal, tmp_xyz):
    forces = np.random.randn(simple_crystal.get_total_nodes(), 3)
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        traj.write(simple_crystal, forces=forces)

    reader = CrystalTrajectory(tmp_xyz, "r")
    f_read = reader[0].to_ase().calc.results["forces"]
    assert np.allclose(f_read, forces)


# ---------------------------------------------------------------------------
# Append mode
# ---------------------------------------------------------------------------

def test_append_mode(simple_crystal, tmp_xyz):
    """Write 2 frames in 'w' mode, then append 1 more with 'a' mode."""
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        traj.write(simple_crystal)
        traj.write(simple_crystal)

    with CrystalTrajectory(tmp_xyz, "a") as traj:
        traj.write(simple_crystal)

    reader = CrystalTrajectory(tmp_xyz, "r")
    assert len(reader) == 3


# ---------------------------------------------------------------------------
# Writer __len__
# ---------------------------------------------------------------------------

def test_writer_len(simple_crystal, tmp_xyz):
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        assert len(traj) == 0
        traj.write(simple_crystal)
        assert len(traj) == 1
        traj.write(simple_crystal)
        assert len(traj) == 2


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_returns_correct_types(tmp_xyz):
    r = CrystalTrajectory(tmp_xyz, "r")
    assert isinstance(r, CrystalTrajectoryReader)

    w = CrystalTrajectory(tmp_xyz, "w")
    assert isinstance(w, CrystalTrajectoryWriter)

    a = CrystalTrajectory(tmp_xyz, "a")
    assert isinstance(a, CrystalTrajectoryWriter)


def test_factory_invalid_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        CrystalTrajectory("test.xyz", "x")


# ---------------------------------------------------------------------------
# Empty trajectory
# ---------------------------------------------------------------------------

def test_empty_reader(tmp_xyz):
    """Reader on a file with no frames returns len=0."""
    # Create an empty file so ASE can open it
    with open(tmp_xyz, "w") as f:
        pass

    reader = CrystalTrajectory(tmp_xyz, "r")
    assert len(reader) == 0
    assert list(reader) == []


# ---------------------------------------------------------------------------
# Context manager: reader
# ---------------------------------------------------------------------------

def test_reader_context_manager(simple_crystal, tmp_xyz):
    with CrystalTrajectory(tmp_xyz, "w") as traj:
        traj.write(simple_crystal)

    with CrystalTrajectory(tmp_xyz, "r") as reader:
        assert len(reader) == 1
        assert reader[0].get_total_nodes() == simple_crystal.get_total_nodes()
