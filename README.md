# MolCrysKit: Molecular Crystal Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](#)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)

## Overview

MolCrysKit is a Python toolkit designed for handling molecular crystals, providing utilities for parsing crystallographic data, identifying molecules within crystals, and performing various analyses on molecular crystals.

## Features

- Parse crystallographic data from CIF files
- Identify individual molecules within a crystal structure
- Analyze molecular properties such as centroids, center of mass, and principal axes
- Handle atomic properties (mass, radius, etc.)
- Access crystal lattice parameters (vectors and cell parameters)
- Retrieve default atomic radii from crystal structures
- Customize bond thresholds for specific atom pairs

## Installation

To install MolCrysKit, you can use pip:

```bash
pip install .
```

Or for development purposes, install in editable mode:

```bash
pip install -e .
```

## Quick Start

Here's a simple example of how to use MolCrysKit:

```python
# Import the necessary classes and functions
from molcrys_kit.structures import Atom, Molecule, MolecularCrystal
from molcrys_kit.constants import get_atomic_mass, get_atomic_radius

# Create atoms
atom1 = Atom(element='C', position=[0.0, 0.0, 0.0])
atom2 = Atom(element='H', position=[0.0, 0.0, 1.0])

# Create a molecule
molecule = Molecule(atoms=[atom1, atom2])

# Access atomic properties
carbon_mass = get_atomic_mass('C')
hydrogen_radius = get_atomic_radius('H')

print(f"Carbon mass: {carbon_mass}")
print(f"Hydrogen radius: {hydrogen_radius}")
```

### Parsing CIF Files

You can parse CIF files to create molecular crystals:

```python
from molcrys_kit.io import parse_cif

# Parse a CIF file
crystal = parse_cif('path/to/your/file.cif')

# Access crystal properties
print(f"Crystal name: {crystal.name}")
print(f"Number of atoms: {len(crystal.atoms)}")
```

### Identifying Molecules in Crystals

Identify individual molecules within a crystal structure:

```python
from molcrys_kit.analysis import identify_molecules

# Assuming you have a loaded crystal
crystal = parse_cif('path/to/your/file.cif')

# Identify molecules
molecules = identify_molecules(crystal)

print(f"Number of molecules identified: {len(molecules)}")
```

### Advanced CIF Parsing

For more advanced CIF parsing that automatically identifies molecules:

```python
from molcrys_kit.io import parse_cif_advanced

# Parse a CIF file with automatic molecule identification
crystal = parse_cif_advanced('path/to/your/file.cif')

print(f"Number of molecules: {len(crystal.molecules)}")
for i, molecule in enumerate(crystal.molecules):
    print(f"Molecule {i+1}: {molecule.get_chemical_formula()}")
```

### Getting Default Atomic Radii

Retrieve the default atomic radii used for bond detection:

```python
from molcrys_kit.io import parse_cif_advanced

# Parse a CIF file
crystal = parse_cif_advanced('path/to/your/file.cif')

# Get default atomic radii
radii = crystal.get_default_atomic_radii()
for element, radius in list(radii.items())[:5]:  # Show first 5 elements
    print(f"{element}: {radius} Å")
```

### Customizing Bond Thresholds

Customize bond detection thresholds for specific atom pairs:

```python
from molcrys_kit.io import parse_cif_advanced

# Define custom bond thresholds for specific atom pairs
custom_thresholds = {
    ('C', 'C'): 1.8,  # Custom threshold for C-C bonds
    ('C', 'H'): 1.4,  # Custom threshold for C-H bonds
    ('O', 'H'): 1.2   # Custom threshold for O-H bonds
}

# Parse CIF with custom bond thresholds
crystal = parse_cif_advanced('path/to/your/file.cif', bond_thresholds=custom_thresholds)

print(f"Number of molecules with custom thresholds: {len(crystal.molecules)}")
```

### Getting Crystal Cell Parameters

After instantiating a crystal, you can retrieve its lattice parameters:

```python
from molcrys_kit.io import parse_cif_advanced

# Parse a CIF file
crystal = parse_cif_advanced('path/to/your/file.cif')

# Get the lattice matrix (vectors)
lattice_matrix = crystal.get_lattice_vectors()
print("Lattice vectors:")
print(lattice_matrix)

# Get the lattice parameters (a, b, c, α, β, γ)
lattice_params = crystal.get_lattice_parameters()
a, b, c, alpha, beta, gamma = lattice_params
print(f"Lattice parameters:")
print(f"  a={a:.4f} Å, b={b:.4f} Å, c={c:.4f} Å")
print(f"  α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°")
```

### Working with Crystal Structures

Access detailed information about crystal structures:

```python
from molcrys_kit.structures import MolecularCrystal

# Load a crystal
crystal = MolecularCrystal.from_cif('path/to/your/file.cif')

# Access molecules
for i, molecule in enumerate(crystal.molecules):
    centroid = molecule.get_centroid()
    com = molecule.get_center_of_mass()
    formula = molecule.get_chemical_formula()
    
    print(f"Molecule {i+1} ({formula}):")
    print(f"  Centroid: {centroid}")
    print(f"  Center of Mass: {com}")
```

### Getting Fractional Coordinates of Molecular Centroids

You can directly obtain fractional coordinates of molecular centroids:

```python
from molcrys_kit.io import parse_cif_advanced

# Parse a CIF file with automatic molecule identification
crystal = parse_cif_advanced('path/to/your/file.cif')

# Access molecules and their fractional coordinates
for i, molecule in enumerate(crystal.molecules):
    # Get centroid in Cartesian coordinates
    centroid_cart = molecule.get_centroid()
    
    # Get centroid directly in fractional coordinates
    centroid_frac = molecule.get_centroid_frac()
    
    print(f"Molecule {i+1}:")
    print(f"  Cartesian centroid: [{centroid_cart[0]:.4f}, {centroid_cart[1]:.4f}, {centroid_cart[2]:.4f}]")
    print(f"  Fractional centroid: [{centroid_frac[0]:.4f}, {centroid_frac[1]:.4f}, {centroid_frac[2]:.4f}]")
```

## Modules

- [structures](./molcrys_kit/structures): Core data structures for atoms, molecules, and crystals
- [io](./molcrys_kit/io): Input/output operations, primarily for CIF files
- [analysis](./molcrys_kit/analysis): Analysis tools for molecular crystals
- [constants](./molcrys_kit/constants): Physical constants and atomic properties
- [operations](./molcrys_kit/operations): Operations on molecular crystals
- [utils](./molcrys_kit/utils): Utility functions

## Scripts

The project includes several example scripts in the `scripts/` directory that demonstrate various functionalities:

- `scripts/lattice_parameters_demo.py` - Demonstrates how to get lattice parameters from a crystal
- `scripts/atomic_properties_demo.py` - Shows how to access atomic properties
- `scripts/enhanced_molecule_example.py` - Example usage of the EnhancedMolecule class
- `scripts/molecular_centroids.py` - Calculate molecular centroids in fractional coordinates
- `scripts/demo_custom_bond_thresholds.py` - Demonstrate custom bond thresholds and atomic radii features

To run these scripts, first install the package in development mode:

```bash
pip install -e .
```

Then run any script directly:

```bash
python scripts/lattice_parameters_demo.py
```

## Contributing

Contributions to MolCrysKit are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.