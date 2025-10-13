"""
Test script to verify CIF parsing with full coordinates works correctly.
"""

import sys
import os

# Add the project root to the path so we can import molcrys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from molcrys.io.cif import parse_cif, parse_cif_advanced

def test_parse_cif_with_full_coordinates():
    """Test parsing CIF files with full coordinates."""
    test_cif_path = os.path.join(os.path.dirname(__file__), 'data', 'test_full_coords.cif')
    
    try:
        # Test basic parsing
        print("Testing basic CIF parsing...")
        crystal = parse_cif(test_cif_path)
        print(f"Success! Parsed crystal with {len(crystal.molecules)} molecule(s)")
        
        # Test advanced parsing
        print("Testing advanced CIF parsing...")
        crystal_adv = parse_cif_advanced(test_cif_path)
        for molecule in crystal_adv.molecules:
            print(f"Molecule {molecule.atoms.get_chemical_formula()} has {len(molecule.atoms)} atom(s)")
        print(f"Success! Parsed crystal with {len(crystal_adv.molecules)} molecule(s)")
        
        return True
    except Exception as e:
        print(f"Error parsing CIF file: {e}")
        return False

if __name__ == "__main__":
    success = test_parse_cif_with_full_coordinates()
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)