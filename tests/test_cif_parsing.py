#!/usr/bin/env python3
"""
Test CIF parsing functionality.
"""

import os
import sys
import warnings

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from molcrys_kit.io.cif import read_mol_crystal, parse_cif_advanced

def test_parse_cif():
    """Test basic CIF parsing."""
    test_cif_path = os.path.join(os.path.dirname(__file__), 'data', 'test_full_coords.cif')
    
    try:
        # Test new parsing function
        print("Testing new CIF parsing function...")
        crystal = read_mol_crystal(test_cif_path)
        print(f"Success! Parsed crystal with {len(crystal.molecules)} molecule(s)")
        
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
            print(f"Molecule with {len(symbols)} atoms has elements: {', '.join(unique_symbols)}")
        print(f"Success! Parsed crystal with {len(crystal_adv.molecules)} molecule(s)")
        
        return True
    except Exception as e:
        print(f"Error parsing CIF file: {e}")
        return False

if __name__ == "__main__":
    success = test_parse_cif()
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)