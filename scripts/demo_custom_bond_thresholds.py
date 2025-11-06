#!/usr/bin/env python3
"""
Demonstration script for MolecularCrystal custom bond thresholds and atomic radii features.

This script shows how to:
1. Retrieve default atomic radii from a MolecularCrystal instance
2. Use custom bond thresholds for specific atom pairs
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from molcrys_kit.io import parse_cif_advanced

def demo_atomic_radii():
    """Demonstrate how to retrieve default atomic radii."""
    print("=== Atomic Radii Demo ===")
    
    # Parse a CIF file
    crystal = parse_cif_advanced("tests/data/test_full_coords.cif")
    
    # Get default atomic radii
    radii = crystal.get_default_atomic_radii()
    
    print("First 10 elements and their atomic radii (Å):")
    for i, (element, radius) in enumerate(radii.items()):
        if i >= 10:
            break
        print(f"  {element}: {radius}")
    
    print(f"\nTotal elements with defined radii: {len(radii)}")
    print()


def demo_custom_bond_thresholds():
    """Demonstrate how to use custom bond thresholds for specific atom pairs."""
    print("=== Custom Bond Thresholds Demo ===")
    
    # Parse with default thresholds
    crystal_default = parse_cif_advanced("tests/data/test_full_coords.cif")
    print(f"Default molecular identification: {len(crystal_default.molecules)} molecules")
    
    # Example 1: Modify C-C bond threshold
    # Looser threshold (higher value) might connect more atoms into larger molecules
    loose_thresholds = {('C', 'C'): 2.0}  # 2.0 Å for C-C bonds
    crystal_loose = parse_cif_advanced("tests/data/test_full_coords.cif", 
                                       bond_thresholds=loose_thresholds)
    print(f"With loose C-C threshold (2.0 Å): {len(crystal_loose.molecules)} molecules")
    
    # Example 2: Stricter C-C bond threshold
    # Stricter threshold (lower value) might split molecules into smaller fragments
    strict_thresholds = {('C', 'C'): 1.0}  # 1.0 Å for C-C bonds
    crystal_strict = parse_cif_advanced("tests/data/test_full_coords.cif", 
                                        bond_thresholds=strict_thresholds)
    print(f"With strict C-C threshold (1.0 Å): {len(crystal_strict.molecules)} molecules")
    
    # Example 3: Custom thresholds for multiple atom pairs
    multi_thresholds = {
        ('C', 'C'): 1.8,   # Slightly looser for C-C
        ('C', 'H'): 1.4,   # Standard for C-H
        ('O', 'H'): 1.2    # Standard for O-H
    }
    crystal_multi = parse_cif_advanced("tests/data/test_full_coords.cif", 
                                       bond_thresholds=multi_thresholds)
    print(f"With multiple custom thresholds: {len(crystal_multi.molecules)} molecules")
    
    print()


def main():
    """Main demonstration function."""
    print("MolCrysKit Custom Features Demonstration")
    print("========================================")
    print()
    
    try:
        demo_atomic_radii()
        demo_custom_bond_thresholds()
        
        print("✅ Demonstration completed successfully!")
        print()
        print("Summary:")
        print("- Use crystal.get_default_atomic_radii() to retrieve atomic radii")
        print("- Use bond_thresholds parameter in parse_cif_advanced() for custom thresholds")
        print("- Thresholds are specified as dict with atom pair tuples as keys")
        print("  and distance values in Angstroms as values")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()