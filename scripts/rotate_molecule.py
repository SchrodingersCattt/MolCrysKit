#!/usr/bin/env python3
"""
Example: Molecular rotation operations.
"""

import numpy as np

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE not available. Install with 'pip install ase' for full functionality.")

from molcrys.structures import MolecularCrystal
from molcrys.operations import rotate_molecule


def create_benzene_molecule():
    """Create a benzene molecule for demonstration."""
    if not ASE_AVAILABLE:
        print("ASE not available, cannot create benzene molecule")
        return None
    
    # Create a benzene molecule (simplified planar structure)
    positions = np.array([
        [ 0.0,  1.0, 0.0],
        [ 0.866, 0.5, 0.0],
        [ 0.866, -0.5, 0.0],
        [ 0.0, -1.0, 0.0],
        [-0.866, -0.5, 0.0],
        [-0.866, 0.5, 0.0]
    ])
    
    benzene = Atoms('C6', positions=positions)
    return benzene


def main():
    """Main function to demonstrate molecular rotation."""
    print("MolCrysKit Example: Molecular Rotation")
    print("=" * 40)
    
    if not ASE_AVAILABLE:
        print("This example requires ASE. Please install it with 'pip install ase'")
        return
    
    # Create benzene molecule
    benzene = create_benzene_molecule()
    
    if benzene is None:
        return
    
    print(f"Initial benzene center of mass: {benzene.get_center_of_mass()}")
    
    # Rotate around z-axis by 60 degrees (π/3 radians)
    axis = np.array([0, 0, 1])  # z-axis
    angle = np.pi / 3  # 60 degrees
    
    print(f"\nRotating benzene around z-axis by {angle*180/np.pi:.1f} degrees...")
    rotate_molecule(benzene, axis, angle)
    
    print(f"New benzene center of mass: {benzene.get_center_of_mass()}")
    
    # Rotate around x-axis by 90 degrees (π/2 radians)
    axis = np.array([1, 0, 0])  # x-axis
    angle = np.pi / 2  # 90 degrees
    
    print(f"\nRotating benzene around x-axis by {angle*180/np.pi:.1f} degrees...")
    rotate_molecule(benzene, axis, angle)
    
    print(f"New benzene center of mass: {benzene.get_center_of_mass()}")


if __name__ == "__main__":
    main()