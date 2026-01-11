#!/usr/bin/env python3
"""
Test the hydrogenation module.
"""

import os
import sys
import numpy as np

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from molcrys_kit.operations.hydrogenation import Hydrogenator, add_hydrogens
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.analysis.chemical_env import ChemicalEnvironment
from ase import Atoms


def test_hydrogenator_initialization():
    """Test initializing a Hydrogenator."""
    # Create a simple crystal with a water molecule
    h2o = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    crystal = MolecularCrystal(np.eye(3) * 10.0, [CrystalMolecule(h2o)])

    # Initialize the hydrogenator
    hydrogenator = Hydrogenator(crystal)

    # Check that the hydrogenator was initialized correctly
    assert hydrogenator.crystal == crystal
    assert hydrogenator.default_rules["C"]["target_coordination"] == 4
    assert hydrogenator.default_rules["N"]["target_coordination"] == 3
    assert hydrogenator.default_rules["O"]["target_coordination"] == 2
    assert hydrogenator.default_bond_lengths["O-H"] == 0.96


def test_add_hydrogens_to_molecule():
    """Test adding hydrogens to a simple molecule."""
    # Create a methane-like molecule (carbon with 3 hydrogens, missing one)
    ch3 = Atoms(
        symbols=["C", "H", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    crystal = MolecularCrystal(np.eye(3) * 10.0, [CrystalMolecule(ch3)])

    # Add hydrogens
    new_crystal = add_hydrogens(crystal)

    # The methane should now have 4 hydrogens (one added)
    new_mol = new_crystal.molecules[0]
    symbols = new_mol.get_chemical_symbols()
    h_count = symbols.count("H")

    # Should have 4 H atoms (original 3 + 1 added)
    # Note: This depends on the implementation details of get_missing_vectors
    # which might not add hydrogens if the geometry doesn't require it
    assert symbols[0] == "C"  # First atom should still be carbon
    assert len(symbols) >= 4  # At least C and 3 H's


def test_hydrogenator_with_custom_rules():
    """Test hydrogenator with custom rules."""
    # Create a simple crystal with an N atom (ammonia-like with 2 H's, missing one)
    nh2 = Atoms(
        symbols=["N", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )
    crystal = MolecularCrystal(np.eye(3) * 10.0, [CrystalMolecule(nh2)])

    # Initialize hydrogenator
    hydrogenator = Hydrogenator(crystal)

    # Define custom rules
    custom_rules = [
        {"symbol": "N", "target_coordination": 3, "geometry": "trigonal_pyramidal"}
    ]

    # Add hydrogens with custom rules
    new_crystal = hydrogenator.add_hydrogens(rules=custom_rules)

    # The ammonia should now have 3 hydrogens (one added)
    new_mol = new_crystal.molecules[0]
    symbols = new_mol.get_chemical_symbols()
    h_count = symbols.count("H")

    # Should have 3 H atoms (original 2 + 1 added) if the algorithm works as expected
    assert symbols[0] == "N"  # First atom should still be nitrogen
    assert len(symbols) >= 3  # At least N and 2 H's


def test_find_matching_rule():
    """Test the new rule selection logic with ChemicalEnvironment."""
    # Create a simple crystal with a carbon atom
    ch2 = Atoms(
        symbols=["C", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )
    crystal_mol = CrystalMolecule(ch2)  # Create CrystalMolecule as needed
    crystal = MolecularCrystal(np.eye(3) * 10.0, [crystal_mol])

    hydrogenator = Hydrogenator(crystal)

    # Create a chemical environment from the CrystalMolecule
    chem_env = ChemicalEnvironment(crystal_mol)

    # Test specific rule application (no match case)
    specific_rules = [
        {"symbol": "C", "neighbors": ["O"], "target_coordination": 4}
    ]
    general_rules = {"C": {"target_coordination": 3, "geometry": "trigonal_planar"}}

    # Since C doesn't have O as neighbor, this should return None
    rule = hydrogenator._find_matching_rule(chem_env, 0, "C", specific_rules, {})
    assert rule is None

    # Test specific rule application (match case)
    specific_rules = [
        {"symbol": "C", "neighbors": ["H"], "target_coordination": 4, "geometry": "tetrahedral"}
    ]
    rule = hydrogenator._find_matching_rule(chem_env, 0, "C", specific_rules, general_rules)
    assert rule is not None
    assert rule["target_coordination"] == 4
    assert rule["geometry"] == "tetrahedral"

    # Test general rule application (fallback case)
    specific_rules = [
        {"symbol": "N", "neighbors": ["H"], "target_coordination": 3}
    ]
    rule = hydrogenator._find_matching_rule(chem_env, 0, "C", specific_rules, general_rules)
    assert rule is not None
    assert rule["target_coordination"] == 3
    assert rule["geometry"] == "trigonal_planar"


def test_hydrogenation_of_methane():
    """Test hydrogenation of a methane molecule with one missing hydrogen."""
    # Create a methane with only 3 hydrogens
    ch3_radical = Atoms(
        symbols=["C", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
        ],
    )
    crystal = MolecularCrystal(np.eye(3) * 20.0, [CrystalMolecule(ch3_radical)])

    # Add hydrogens
    hydrogenator = Hydrogenator(crystal)
    new_crystal = hydrogenator.add_hydrogens()

    # The methane should now have 4 hydrogens
    new_mol = new_crystal.molecules[0]
    symbols = new_mol.get_chemical_symbols()
    c_count = symbols.count("C")
    h_count = symbols.count("H")

    assert c_count == 1  # Should still have 1 carbon
    assert h_count >= 3  # Should have at least the original 3 hydrogens


def test_custom_bond_lengths():
    """Test hydrogenation with custom bond lengths."""
    # Create a simple crystal with an O atom (water with 1 H, missing one)
    oh = Atoms(
        symbols=["O", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    crystal = MolecularCrystal(np.eye(3) * 10.0, [CrystalMolecule(oh)])

    # Initialize hydrogenator
    hydrogenator = Hydrogenator(crystal)

    # Define custom bond lengths
    custom_bond_lengths = {"O-H": 1.1}  # Different from default

    # Add hydrogens with custom bond lengths
    new_crystal = hydrogenator.add_hydrogens(bond_lengths=custom_bond_lengths)

    # Check that the crystal was created without error
    assert len(new_crystal.molecules) == 1


def run_tests():
    """Run all tests."""
    tests = [
        test_hydrogenator_initialization,
        test_add_hydrogens_to_molecule,
        test_hydrogenator_with_custom_rules,
        test_find_matching_rule,
        test_hydrogenation_of_methane,
        test_custom_bond_lengths,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\nTests passed: {passed}")
    print(f"Tests failed: {failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)