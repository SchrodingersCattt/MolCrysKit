"""
Shared pytest configuration and fixtures for MolCrysKit tests.
"""

import os

import pytest
import numpy as np
from ase import Atoms

from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.crystal import MolecularCrystal


# ----- Paths -----
@pytest.fixture(scope="session")
def tests_dir():
    """Root of tests directory."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def data_dir(tests_dir):
    """Path to tests/data."""
    return os.path.join(tests_dir, "data")


@pytest.fixture(scope="session")
def test_cif_path(data_dir):
    """Path to test CIF with full coordinates."""
    return os.path.join(data_dir, "test_full_coords.cif")


# ----- Lattices -----
@pytest.fixture
def cubic_lattice_10():
    """10 Å cubic lattice."""
    return np.eye(3) * 10.0


@pytest.fixture
def cubic_lattice_5():
    """5 Å cubic lattice."""
    return np.eye(3) * 5.0


# ----- Molecules (ASE Atoms) -----
@pytest.fixture
def water_atoms():
    """Single water molecule as ASE Atoms."""
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


@pytest.fixture
def ch2_atoms():
    """CH2 fragment (C with 2 H)."""
    return Atoms(
        symbols=["C", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )


@pytest.fixture
def co_molecule():
    """CO molecule."""
    return Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])


@pytest.fixture
def n2_molecule():
    """N2 molecule."""
    return Atoms("NN", positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])


# ----- Crystals -----
@pytest.fixture
def simple_crystal(cubic_lattice_10, co_molecule, n2_molecule):
    """Crystal with 2 CO and 4 N2 (for stoichiometry/defects tests)."""
    molecules = [
        CrystalMolecule(co_molecule),
        CrystalMolecule(Atoms("CO", positions=[[2.0, 0.0, 0.0], [3.2, 0.0, 0.0]])),
        CrystalMolecule(Atoms("NN", positions=[[0.0, 2.0, 0.0], [1.1, 2.0, 0.0]])),
        CrystalMolecule(Atoms("NN", positions=[[2.0, 2.0, 0.0], [3.1, 2.0, 0.0]])),
        CrystalMolecule(Atoms("NN", positions=[[0.0, 4.0, 0.0], [1.1, 4.0, 0.0]])),
        CrystalMolecule(Atoms("NN", positions=[[2.0, 4.0, 0.0], [3.1, 4.0, 0.0]])),
    ]
    return MolecularCrystal(cubic_lattice_10, molecules)


@pytest.fixture
def empty_crystal(cubic_lattice_10):
    """Crystal with no molecules."""
    return MolecularCrystal(cubic_lattice_10, [])


@pytest.fixture
def crystal_single_water(cubic_lattice_10, water_atoms):
    """Crystal with one water molecule."""
    return MolecularCrystal(cubic_lattice_10, [CrystalMolecule(water_atoms)])
