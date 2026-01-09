"""
Unit tests for defect generation.

This module provides comprehensive tests for the VacancyGenerator class.
"""

import pytest
import numpy as np
from ase import Atoms
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.operations.defects import VacancyGenerator


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


def test_vacancy_generation_target_spec(simple_crystal):
    """Test vacancy generation with a specific target specification."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Target to remove 1 CO and 2 N2 molecules
    target_spec = {"CO_1": 1, "N2_1": 2}

    new_crystal = vacancy_gen.generate_vacancy(target_spec=target_spec)

    # Original has 6 molecules, removing 3, so should have 3 left
    assert len(new_crystal.molecules) == 3

    # Total atoms should be reduced accordingly
    original_atoms = sum(len(mol) for mol in simple_crystal.molecules)
    new_atoms = sum(len(mol) for mol in new_crystal.molecules)

    # We removed 1 CO (2 atoms) and 2 N2 (4 atoms) = 6 atoms total
    assert original_atoms - new_atoms == 6


def test_vacancy_generation_with_seed(simple_crystal):
    """Test that specifying a seed_index removes that specific molecule."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Target to remove 1 CO molecule, starting from molecule index 0
    target_spec = {"CO_1": 1}
    seed_index = 0  # Should remove the first CO molecule

    removal_indices = vacancy_gen.find_removable_cluster_indices(
        target_spec, seed_index
    )

    # Should contain exactly the seed index
    assert seed_index in removal_indices
    assert len(removal_indices) == 1


def test_find_removable_cluster_indices(simple_crystal):
    """Test the helper method to find removable cluster indices."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Target to remove 1 CO and 1 N2 molecule
    target_spec = {"CO_1": 1, "N2_1": 1}

    removal_indices = vacancy_gen.find_removable_cluster_indices(target_spec)

    # Should return 2 indices
    assert len(removal_indices) == 2

    # Each index should be valid
    for idx in removal_indices:
        assert 0 <= idx < len(simple_crystal.molecules)


def test_vacancy_generation_no_target_spec(simple_crystal):
    """Test vacancy generation without specifying target_spec (should use simplest unit)."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Don't specify target_spec, should use simplest unit (1 CO, 2 N2)
    new_crystal = vacancy_gen.generate_vacancy()

    # Original has 6 molecules, simplest unit is 1 CO + 2 N2 = 3 molecules to remove
    # So should have 3 left
    assert len(new_crystal.molecules) == len(simple_crystal.molecules) - 3


def test_invalid_target_spec(simple_crystal):
    """Test that an invalid target specification raises an error."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Try to remove more molecules than available
    invalid_target_spec = {"CO_1": 10}  # Only 2 CO molecules available

    with pytest.raises(ValueError):
        vacancy_gen.generate_vacancy(target_spec=invalid_target_spec)


def test_invalid_seed_index(simple_crystal):
    """Test that an invalid seed index raises an error."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Try to use a seed index that doesn't belong to requested species
    target_spec = {"N2_1": 1}  # Only N2 molecules
    invalid_seed_index = 0  # This is a CO molecule

    with pytest.raises(ValueError):
        vacancy_gen.generate_vacancy(
            target_spec=target_spec, seed_index=invalid_seed_index
        )


def test_return_removed_cluster(simple_crystal):
    """Test the return_removed_cluster functionality."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Target to remove 1 CO and 2 N2 molecules
    target_spec = {"CO_1": 1, "N2_1": 2}

    new_crystal, removed_cluster = vacancy_gen.generate_vacancy(
        target_spec=target_spec, return_removed_cluster=True
    )

    # Check that both returned objects are MolecularCrystal instances
    assert isinstance(new_crystal, MolecularCrystal)
    assert isinstance(removed_cluster, MolecularCrystal)

    # Check that the number of molecules matches expectations
    assert len(new_crystal.molecules) == 3  # 6 original - 3 removed
    assert len(removed_cluster.molecules) == 3  # 1 CO + 2 N2 = 3 removed

    # Complementarity check: total should equal original
    assert len(new_crystal.molecules) + len(removed_cluster.molecules) == len(
        simple_crystal.molecules
    )

    # Check that removed cluster has the correct species
    removed_species = [mol.get_chemical_formula() for mol in removed_cluster.molecules]
    assert removed_species.count("CO") == 1  # 1 CO molecule removed
    assert removed_species.count("N2") == 2  # 2 N2 molecules removed

    # Check that the lattices and PBC settings are preserved in the removed cluster
    np.testing.assert_array_equal(removed_cluster.lattice, simple_crystal.lattice)
    assert removed_cluster.pbc == simple_crystal.pbc


def test_return_removed_cluster_compatibility_check(simple_crystal):
    """Test that the total number of atoms in original crystal equals the sum of new and removed crystals."""
    vacancy_gen = VacancyGenerator(simple_crystal)

    # Target to remove 1 CO and 1 N2 molecule
    target_spec = {"CO_1": 1, "N2_1": 1}

    new_crystal, removed_cluster = vacancy_gen.generate_vacancy(
        target_spec=target_spec, return_removed_cluster=True
    )

    original_atoms = sum(len(mol) for mol in simple_crystal.molecules)
    new_crystal_atoms = sum(len(mol) for mol in new_crystal.molecules)
    removed_cluster_atoms = sum(len(mol) for mol in removed_cluster.molecules)

    # Total atoms should be preserved between original and the split
    assert original_atoms == new_crystal_atoms + removed_cluster_atoms

    # Removed cluster should have 1 CO (2 atoms) + 1 N2 (2 atoms) = 4 atoms
    assert removed_cluster_atoms == 4

    # New crystal should have original - 4 atoms
    assert new_crystal_atoms == original_atoms - 4
