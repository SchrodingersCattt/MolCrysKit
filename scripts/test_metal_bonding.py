#!/usr/bin/env python3
"""
Test metal bonding functionality.
"""

import sys
import os

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    try:
        # Try to import required modules
        from molcrys_kit.structures.atom import Atom
        from molcrys_kit.structures.molecule import Molecule
        from molcrys_kit.constants import is_metal_element

        def test_metal_detection():
            """Test metal element detection."""
            print("Testing metal element detection:")
            print(f"Is Cu a metal? {is_metal_element('Cu')}")  # Should be True
            print(f"Is C a metal? {is_metal_element('C')}")    # Should be False
            print(f"Is Fe a metal? {is_metal_element('Fe')}")  # Should be True
            print(f"Is O a metal? {is_metal_element('O')}")    # Should be False
            print()

        def test_bond_detection():
            """Test bond detection with different threshold factors."""
            # Create a simple lattice (cubic box)
            lattice = np.array([[10.0, 0.0, 0.0],
                                [0.0, 10.0, 0.0],
                                [0.0, 0.0, 10.0]])
            
            print("Testing bond detection with different element types:")
            
            # Test case 1: Metal-Metal (Cu-Cu)
            print("1. Cu-Cu molecule (Metal-Metal):")
            atoms_cu = [
                Atom('Cu', np.array([0.0, 0.0, 0.0])),
                Atom('Cu', np.array([0.0, 0.0, 1.0]))  # 1 Angstrom apart
            ]
            molecule_cu = Molecule(atoms_cu, lattice=lattice)
            
            # Test with different threshold factors
            bonds_cu_09 = molecule_cu.get_bonds(threshold_factor=0.9)
            print(f"   Bonds with factor 0.9: {len(bonds_cu_09)} bonds found")
            
            bonds_cu_12 = molecule_cu.get_bonds(threshold_factor=1.2)
            print(f"   Bonds with factor 1.2: {len(bonds_cu_12)} bonds found")
            
            bonds_cu_15 = molecule_cu.get_bonds(threshold_factor=1.5)
            print(f"   Bonds with factor 1.5: {len(bonds_cu_15)} bonds found")
            print()
            
            # Test case 2: Non-Metal-Non-Metal (C-O)
            print("2. CO molecule (Non-Metal-Non-Metal):")
            atoms_co = [
                Atom('C', np.array([0.0, 0.0, 0.0])),
                Atom('O', np.array([0.0, 0.0, 1.2]))  # 1.2 Angstrom apart
            ]
            molecule_co = Molecule(atoms_co, lattice=lattice)
            
            # Test with different threshold factors
            bonds_co_09 = molecule_co.get_bonds(threshold_factor=0.9)
            print(f"   Bonds with factor 0.9: {len(bonds_co_09)} bonds found")
            
            bonds_co_12 = molecule_co.get_bonds(threshold_factor=1.2)
            print(f"   Bonds with factor 1.2: {len(bonds_co_12)} bonds found")
            
            bonds_co_15 = molecule_co.get_bonds(threshold_factor=1.5)
            print(f"   Bonds with factor 1.5: {len(bonds_co_15)} bonds found")
            print()
            
            # Test case 3: Metal-Non-Metal (Na-Cl)
            print("3. NaCl molecule (Metal-Non-Metal):")
            atoms_nacl = [
                Atom('Na', np.array([0.0, 0.0, 0.0])),
                Atom('Cl', np.array([0.0, 0.0, 1.1]))  # 1.1 Angstrom apart
            ]
            molecule_nacl = Molecule(atoms_nacl, lattice=lattice)
            
            # Test with different threshold factors
            bonds_nacl_09 = molecule_nacl.get_bonds(threshold_factor=0.9)
            print(f"   Bonds with factor 0.9: {len(bonds_nacl_09)} bonds found")
            
            bonds_nacl_12 = molecule_nacl.get_bonds(threshold_factor=1.2)
            print(f"   Bonds with factor 1.2: {len(bonds_nacl_12)} bonds found")
            
            bonds_nacl_15 = molecule_nacl.get_bonds(threshold_factor=1.5)
            print(f"   Bonds with factor 1.5: {len(bonds_nacl_15)} bonds found")
            print()

        if __name__ == "__main__":
            test_metal_detection()
            test_bond_detection()
            print("Test completed successfully!")

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you have installed the molcrys-kit package:")
        print("pip install -e .")
        return 1


if __name__ == "__main__":
    main()
