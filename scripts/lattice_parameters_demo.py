#!/usr/bin/env python3
"""
Demo script to showcase lattice parameters functionality.
"""

import sys
import os
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE not available. Install with 'pip install ase' for full functionality.")

try:
    # Import molcrys modules
    from molcrys_kit.structures import MolecularCrystal
except ImportError as e:
    print(f"Error importing MolCrysKit: {e}")
    print("Please ensure the package is installed with 'pip install -e .'")
    sys.exit(1)


def demo_lattice_parameters():
    """Demonstrate accessing lattice parameters."""
    print("Lattice Parameters Demo")
    print("=" * 30)
    
    # Create a simple cubic lattice
    print("\n1. Simple cubic lattice:")
    lattice = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ])
    
    # Create some dummy molecules with cell information
    water = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.756950, 0.585809, 0.000000],
            [-0.756950, 0.585809, 0.000000]
        ],
        cell=lattice,
        pbc=(True, True, True)
    )
    
    crystal = MolecularCrystal(lattice, [water])
    
    print("Lattice vectors:")
    lattice_matrix = crystal.get_lattice_vectors()
    for i, vec in enumerate(lattice_matrix):
        print(f"  a{i+1}: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]")
    
    print("\nLattice parameters:")
    a, b, c, alpha, beta, gamma = crystal.get_lattice_parameters()
    print(f"  a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å")
    print(f"  α = {alpha:.2f}°, β = {beta:.2f}°, γ = {gamma:.2f}°")
    
    # Create a monoclinic lattice
    print("\n2. Monoclinic lattice:")
    lattice_mono = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 0.0, 6.0]  # Non-orthogonal
    ])
    
    water_mono = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.756950, 0.585809, 0.000000],
            [-0.756950, 0.585809, 0.000000]
        ],
        cell=lattice_mono,
        pbc=(True, True, True)
    )
    
    crystal_mono = MolecularCrystal(lattice_mono, [water_mono])
    
    print("Lattice vectors:")
    lattice_matrix_mono = crystal_mono.get_lattice_vectors()
    for i, vec in enumerate(lattice_matrix_mono):
        print(f"  a{i+1}: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]")
    
    print("\nLattice parameters:")
    a, b, c, alpha, beta, gamma = crystal_mono.get_lattice_parameters()
    print(f"  a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å")
    print(f"  α = {alpha:.2f}°, β = {beta:.2f}°, γ = {gamma:.2f}°")
    
    # Create a triclinic lattice
    print("\n3. Triclinic lattice:")
    lattice_tric = np.array([
        [4.0, 0.2, 0.1],
        [0.3, 5.0, 0.1],
        [0.1, 0.2, 6.0]
    ])
    
    water_tric = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.756950, 0.585809, 0.000000],
            [-0.756950, 0.585809, 0.000000]
        ],
        cell=lattice_tric,
        pbc=(True, True, True)
    )
    
    crystal_tric = MolecularCrystal(lattice_tric, [water_tric])
    
    print("Lattice vectors:")
    lattice_matrix_tric = crystal_tric.get_lattice_vectors()
    for i, vec in enumerate(lattice_matrix_tric):
        print(f"  a{i+1}: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]")
    
    print("\nLattice parameters:")
    a, b, c, alpha, beta, gamma = crystal_tric.get_lattice_parameters()
    print(f"  a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å")
    print(f"  α = {alpha:.2f}°, β = {beta:.2f}°, γ = {gamma:.2f}°")


def main():
    """Main function."""
    if not ASE_AVAILABLE:
        print("Error: ASE is required for this demo. Please install it with 'pip install ase'")
        sys.exit(1)
    
    demo_lattice_parameters()


if __name__ == "__main__":
    main()