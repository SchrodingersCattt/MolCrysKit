# MolCrysKit: Molecular Crystal Toolkit

A Python toolkit for handling molecular crystals - crystals made of discrete molecular units connected by weak intermolecular interactions (e.g., hydrogen bonds, van der Waals).

## Overview

MolCrysKit provides a clean, modular, and testable package focused on molecular crystals with capabilities for:

- Automatic molecule extraction from crystal structures
- Rigid-body manipulation of molecular units
- Disorder resolution and enumeration
- Analysis of intermolecular interactions
- Structure transformations and perturbations

## Installation

```bash
# Clone the repository
git clone https://github.com/SchrodingersCattt/MolCrysKit.git
cd MolCrysKit

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .
```

## Key Features

### Core Data Structures
- `Atom`: Representation of atomic species and coordinates
- `Molecule`: Rigid body of atoms with transformation capabilities
- `MolecularCrystal`: Main container for molecular crystal structures

### Constants and Properties
- Atomic masses for over 100 elements (in atomic mass units)
- Atomic radii for over 90 elements (in Angstroms)
- Functions to access and check availability of atomic properties

### File I/O
- CIF parsing (using `pymatgen` or `ASE`)
- Export to visualization formats (XYZ, VESTA, etc.)

### Analysis Capabilities
- Molecular unit identification
- Intermolecular interaction detection (H-bonds, van der Waals)
- Disorder analysis and enumeration
- Structure ranking based on physical plausibility

### Structure Operations
- Rigid-body rotations and translations
- Structure perturbations
- Supercell and surface construction

## Usage Examples

### Basic Usage

```python
from molcrys.structures import Atom, Molecule, MolecularCrystal
from molcrys.constants import get_atomic_mass, get_atomic_radius

# Access atomic properties
print(f"Mass of Carbon: {get_atomic_mass('C')} amu")
print(f"Radius of Oxygen: {get_atomic_radius('O')} Angstroms")

# Create atoms with real properties
atoms = [
    Atom("O", [0.0, 0.0, 0.0]),
    Atom("H", [0.757, 0.586, 0.0]),
    Atom("H", [-0.757, 0.586, 0.0])
]

# Create a water molecule (mass-aware center of mass calculation)
water = Molecule(atoms)
center_of_mass = water.compute_center_of_mass()
print(f"Water center of mass: {center_of_mass}")

# Parse a CIF file
from molcrys.io import parse_cif
crystal = parse_cif("structure.cif")

# Identify molecular units
from molcrys.analysis import identify_molecules
molecules = identify_molecules(crystal)

# Print crystal summary
print(crystal.summary())
```

### Working with Molecular Crystals

```python
# Parse a CIF file with molecular identification
from molcrys.io import parse_cif_advanced

# This function properly identifies molecular units in the crystal
crystal = parse_cif_advanced("molecular_crystal.cif")

# Access the molecules in the crystal
print(f"Number of molecules in crystal: {len(crystal.molecules)}")

# Iterate through each molecule and get properties
for i, molecule in enumerate(crystal.molecules):
    print(f"\nMolecule {i+1}:")
    print(f"  Formula: {molecule.atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(molecule.atoms)}")
    print(f"  Centroid: {molecule.get_centroid()}")
    print(f"  Center of mass: {molecule.get_center_of_mass()}")
    
    # Get ellipsoid radii (semi-axes of the best-fit ellipsoid)
    radii = molecule.get_ellipsoid_radii()
    print(f"  Ellipsoid radii: a={radii[0]:.3f}, b={radii[1]:.3f}, c={radii[2]:.3f}")
    
    # Get principal axes (direction vectors of the ellipsoid)
    ax1, ax2, ax3 = molecule.get_principal_axes()
    print(f"  Principal axes:")
    print(f"    [{ax1[0]:.3f}, {ax1[1]:.3f}, {ax1[2]:.3f}]")
    print(f"    [{ax2[0]:.3f}, {ax2[1]:.3f}, {ax2[2]:.3f}]")
    print(f"    [{ax3[0]:.3f}, {ax3[1]:.3f}, {ax3[2]:.3f}]")
```

### Creating Molecular Crystals Programmatically

```python
import numpy as np
from ase import Atoms
from molcrys.structures import MolecularCrystal

# Define lattice vectors
lattice = np.array([
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0]
])

# Create individual molecules as ASE Atoms objects
water1 = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[
        [1.0, 1.0, 1.0],
        [1.757, 1.586, 1.0],
        [0.243, 1.586, 1.0]
    ]
)

water2 = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[
        [5.0, 5.0, 5.0],
        [5.757, 5.586, 5.0],
        [4.243, 5.586, 5.0]
    ]
)

# Create the molecular crystal
crystal = MolecularCrystal(lattice, [water1, water2])

print(f"Created crystal with {len(crystal.molecules)} molecules")
```

### Advanced Analysis

```python
# Analyze molecular properties in detail
for i, molecule in enumerate(crystal.molecules):
    print(f"\nDetailed analysis of molecule {i+1}:")
    
    # Get atomic positions and symbols
    positions = molecule.positions
    symbols = molecule.symbols
    
    print("  Atomic coordinates:")
    for j, (symbol, pos) in enumerate(zip(symbols, positions)):
        print(f"    {symbol}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Calculate geometric properties
    centroid = molecule.get_centroid()
    com = molecule.get_center_of_mass()
    
    print(f"  Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"  Center of mass: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
    
    # Calculate molecular extent
    radii = molecule.get_ellipsoid_radii()
    print(f"  Molecular size (ellipsoid radii): {radii[0]:.3f} × {radii[1]:.3f} × {radii[2]:.3f} Å")
```

## Documentation

See the [docs](docs/) directory for detailed documentation.

## Dependencies

- `numpy`: Numerical operations
- `scipy`: Scientific computing
- `pymatgen` or `ase`: CIF I/O operations
- `pytest`: Testing (optional)

## License

MIT License. See [LICENSE](LICENSE) for details.