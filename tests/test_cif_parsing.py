#!/usr/bin/env python3
"""
Test CIF parsing functionality.
"""

import os
import sys
import warnings

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from molcrys_kit.io.cif import read_mol_crystal, parse_cif_advanced  # noqa: E402


def test_parse_cif():
    """Test basic CIF parsing."""
    test_cif_path = os.path.join(
        os.path.dirname(__file__), "data", "test_full_coords.cif"
    )

    try:
        # Test new parsing function
        print("Testing new CIF parsing function...")
        crystal = read_mol_crystal(test_cif_path)
        print(f"Success! Parsed crystal with {len(crystal.molecules)} molecule(s)")

        # Verify that molecules are CrystalMolecule instances
        from molcrys_kit.structures.molecule import CrystalMolecule

        assert all(
            isinstance(mol, CrystalMolecule) for mol in crystal.molecules
        ), "Not all molecules are CrystalMolecule instances"

        # Test deprecated function (should show warning)
        print("Testing deprecated CIF parsing function...")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            crystal_adv = parse_cif_advanced(test_cif_path)
            if w and issubclass(w[0].category, DeprecationWarning):
                print(f"Deprecation warning correctly issued: {w[0].message}")
            else:
                print("ERROR: Deprecation warning was not issued!")

        for molecule in crystal_adv.molecules:
            # Access the underlying ASE Atoms object
            symbols = molecule.get_chemical_symbols()
            unique_symbols = list(set(symbols))
            print(
                f"Molecule with {len(symbols)} atoms has elements: {', '.join(unique_symbols)}"
            )
        print(f"Success! Parsed crystal with {len(crystal_adv.molecules)} molecule(s)")

        # Verify that molecules are CrystalMolecule instances in deprecated function too
        assert all(
            isinstance(mol, CrystalMolecule) for mol in crystal_adv.molecules
        ), "Not all molecules are CrystalMolecule instances in deprecated function"

        return True
    except Exception as e:
        print(f"Error parsing CIF file: {e}")
        return False


def test_identify_molecules_function():
    """Test that identify_molecules works with CrystalMolecule class."""
    test_cif_path = os.path.join(
        os.path.dirname(__file__), "data", "test_full_coords.cif"
    )

    try:
        # Parse the CIF file
        crystal = read_mol_crystal(test_cif_path)

        # Check that we have molecules
        assert len(crystal.molecules) > 0, "No molecules found in CIF file"

        # Check that each molecule has the expected properties
        for molecule in crystal.molecules:
            # Check that it's a CrystalMolecule instance
            from molcrys_kit.structures.molecule import CrystalMolecule

            assert isinstance(
                molecule, CrystalMolecule
            ), f"Molecule is not a CrystalMolecule instance: {type(molecule)}"

            # Check that it has a graph
            assert hasattr(
                molecule, "graph"
            ), "Molecule does not have a graph attribute"

            # Check that it has methods from ASE Atoms
            assert hasattr(
                molecule, "get_chemical_symbols"
            ), "Molecule does not have get_chemical_symbols method"
            assert hasattr(
                molecule, "get_positions"
            ), "Molecule does not have get_positions method"
            assert hasattr(
                molecule, "get_chemical_formula"
            ), "Molecule does not have get_chemical_formula method"

        print(
            f"Successfully validated {len(crystal.molecules)} CrystalMolecule instances"
        )
        return True
    except Exception as e:
        print(f"Error in identify_molecules test: {e}")
        return False


if __name__ == "__main__":
    success1 = test_parse_cif()
    success2 = test_identify_molecules_function()

    if success1 and success2:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)
