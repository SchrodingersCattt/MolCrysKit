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

## Modules

- [structures](./molcrys_kit/structures): Core data structures for atoms, molecules, and crystals
- [io](./molcrys_kit/io): Input/output operations, primarily for CIF files
- [analysis](./molcrys_kit/analysis): Analysis tools for molecular crystals
- [constants](./molcrys_kit/constants): Physical constants and atomic properties
- [operations](./molcrys_kit/operations): Operations on molecular crystals
- [utils](./molcrys_kit/utils): Utility functions

## Contributing

Contributions to MolCrysKit are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.