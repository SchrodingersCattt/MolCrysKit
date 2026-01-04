"""
Unit tests for stoichiometry analysis.

This module provides comprehensive tests for the StoichiometryAnalyzer class.
"""

import pytest
import numpy as np
from ase import Atoms
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer


@pytest.fixture
def simple_crystal():
    """Create a simple crystal fixture with 2 types of molecules."""
    # Create a simple lattice
    lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # Create 2 CO molecules (type A)
    co_mol1 = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    co_mol2 = Atoms("CO", positions=[[2.0, 0.0, 0.0], [3.2, 0.0, 0.0]])

    # Create 4 N2 molecules (type B)
    n2_mol1 = Atoms("NN", positions=[[0.0, 2.0, 0.0], [1.1, 2.0, 0.0]])
    n2_mol2 = Atoms("NN", positions=[[2.0, 2.0, 0.0], [3.1, 2.0, 0.0]])
    n2_mol3 = Atoms("NN", positions=[[0.0, 4.0, 0.0], [1.1, 4.0, 0.0]])
    n2_mol4 = Atoms("NN", positions=[[2.0, 4.0, 0.0], [3.1, 4.0, 0.0]])

    # Create CrystalMolecule objects
    molecules = [
        CrystalMolecule(co_mol1),
        CrystalMolecule(co_mol2),
        CrystalMolecule(n2_mol1),
        CrystalMolecule(n2_mol2),
        CrystalMolecule(n2_mol3),
        CrystalMolecule(n2_mol4),
    ]

    return MolecularCrystal(lattice, molecules)


@pytest.fixture
def isomer_crystal():
    """Create a crystal with two isomers with different connectivity."""
    lattice = np.array([[20.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # Create butane: C-C-C-C (linear chain: 0-1-2-3)
    butane = Atoms(
        "CCCC",
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
    )

    # Create isobutane with different connectivity: a central C bonded to 3 others (branched: has a central C connected to 3 others)
    # Molecular structure: central C (index 2) connected to C0, C1, and C3, making it branched
    isobutane = Atoms(
        "CCCC",
        positions=[[0.0, 2.0, 0.0], [3.0, 2.0, 0.0], [1.5, 2.0, 0.0], [1.5, 3.5, 0.0]],
    )

    molecules = [
        CrystalMolecule(
            butane
        ),  # Should be identified as one species due to linear connectivity
        CrystalMolecule(
            isobutane
        ),  # Should be identified as a different species due to branched connectivity
    ]

    return MolecularCrystal(lattice, molecules)


def test_stoichiometry_analyzer_identification(simple_crystal):
    """Test that StoichiometryAnalyzer correctly identifies species and counts them."""
    analyzer = StoichiometryAnalyzer(simple_crystal)

    # There should be 2 species: CO and N2
    assert len(analyzer.species_map) == 2

    # Check that we have the right counts
    co_count = len(analyzer.species_map["CO_1"])
    n2_count = len(analyzer.species_map["N2_1"])

    assert co_count == 2  # 2 CO molecules
    assert n2_count == 4  # 4 N2 molecules


def test_stoichiometry_analyzer_simplest_unit(simple_crystal):
    """Test that get_simplest_unit() returns the correct simplest formula unit."""
    analyzer = StoichiometryAnalyzer(simple_crystal)

    simplest_unit = analyzer.get_simplest_unit()

    # Since we have 2 CO and 4 N2, the GCD is 2, so simplest unit should be 1 CO and 2 N2
    expected = {"CO_1": 1, "N2_1": 2}

    assert simplest_unit == expected


def test_isomer_distinction(isomer_crystal):
    """Test that analyzer assigns different IDs to isomers with different connectivity."""
    analyzer = StoichiometryAnalyzer(isomer_crystal)

    # There should be 2 different species due to different connectivity
    assert len(analyzer.species_map) == 2

    # Get the species IDs
    species_ids = list(analyzer.species_map.keys())

    # Both should have the same formula but different IDs
    formulas = [sid.split("_")[0] for sid in species_ids]
    assert len(set(formulas)) == 1  # Same base formula (C4)
    assert len(set(species_ids)) == 2  # But different IDs due to different topology
