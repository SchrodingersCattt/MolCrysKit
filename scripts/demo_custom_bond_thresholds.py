#!/usr/bin/env python3
"""
Demonstrate custom bond thresholds in molecular crystal parsing.

This script shows how to customize the bonding thresholds used when identifying
molecules in a crystal structure.
"""

import sys
import os

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from molcrys_kit.io import read_mol_crystal
    from ase.visualize import view
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure MolCrysKit and ASE are installed.")
    sys.exit(1)

def demonstrate_default_parsing():
    """Demonstrate the default molecule parsing behavior."""
    print("MolCrysKit: Default Molecule Parsing")
    print("=" * 50)
    
    # Parse a CIF file with default settings
    crystal = read_mol_crystal("tests/data/test_full_coords.cif")
    
    # Get default atomic radii
    radii = crystal.get_default_atomic_radii()
    
    print("Default atomic radii (first 10 elements):")
    for i, (element, radius) in enumerate(radii.items()):
        if i >= 10:
            break
        print(f"  {element}: {radius} Å")
    
    print(f"\nTotal number of elements with defined radii: {len(radii)}")
    
    # Show molecule identification result
    print(f"\nDefault molecule identification:")
    print(f"  Found {len(crystal.molecules)} molecule(s)")
    for i, molecule in enumerate(crystal.molecules):
        formula = molecule.get_chemical_formula()
        print(f"  - Molecule {i+1}: {formula}")
    
    return crystal

def demonstrate_custom_bond_thresholds():
    """Demonstrate how to use custom bond thresholds."""
    print("\nMolCrysKit: Custom Bond Thresholds Demo")
    print("=" * 50)
    
    # Example 1: Default behavior
    print("\n1. Default bond thresholds:")
    try:
        crystal_default = read_mol_crystal("tests/data/test_full_coords.cif")
        print(f"   Identified {len(crystal_default.molecules)} molecule(s)")
        for i, molecule in enumerate(crystal_default.molecules):
            print(f"   - Molecule {i+1}: {molecule.get_chemical_formula()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 2: Looser C-C bond threshold
    print("\n2. Looser C-C bond threshold (2.0 Å):")
    try:
        loose_thresholds = {('C', 'C'): 2.0}
        crystal_loose = read_mol_crystal("tests/data/test_full_coords.cif", 
                                         bond_thresholds=loose_thresholds)
        print(f"   Identified {len(crystal_loose.molecules)} molecule(s)")
        for i, molecule in enumerate(crystal_loose.molecules):
            print(f"   - Molecule {i+1}: {molecule.get_chemical_formula()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 3: Stricter C-C bond threshold
    print("\n3. Stricter C-C bond threshold (1.0 Å):")
    try:
        strict_thresholds = {('C', 'C'): 1.0}
        crystal_strict = read_mol_crystal("tests/data/test_full_coords.cif", 
                                           bond_thresholds=strict_thresholds)
        print(f"   Identified {len(crystal_strict.molecules)} molecule(s)")
        for i, molecule in enumerate(crystal_strict.molecules):
            print(f"   - Molecule {i+1}: {molecule.get_chemical_formula()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 4: Multiple custom thresholds
    print("\n4. Multiple custom thresholds:")
    try:
        multi_thresholds = {
            ('C', 'C'): 1.8,
            ('C', 'H'): 1.4,
            ('O', 'H'): 1.2
        }
        crystal_multi = read_mol_crystal("tests/data/test_full_coords.cif", 
                                         bond_thresholds=multi_thresholds)
        print(f"   Identified {len(crystal_multi.molecules)} molecule(s)")
        for i, molecule in enumerate(crystal_multi.molecules):
            print(f"   - Molecule {i+1}: {molecule.get_chemical_formula()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Optional: Visualize one of the structures
    print("\nNote: To visualize a structure, uncomment the view() calls in the code.")
    # view(crystal_default.to_ase_atoms())


def main():
    """Main demonstration function."""
    try:
        # Run the demonstrations
        base_crystal = demonstrate_default_parsing()
        demonstrate_custom_bond_thresholds()
        
        print("\n✅ Demonstration completed successfully!")
        print("\nSummary:")
        print("- Use crystal.get_default_atomic_radii() to retrieve atomic radii")
        print("- Use bond_thresholds parameter in read_mol_crystal() for custom thresholds")
        print("- Thresholds are specified as a dictionary with atom pair tuples as keys")
        print("  and distance values in Angstroms as values")
        print("- This allows fine-grained control over molecular fragmentation")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()