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
git clone https://github.com/your-username/MolCrysKit.git
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

## Documentation

See the [docs](docs/) directory for detailed documentation.

## Dependencies

- `numpy`: Numerical operations
- `scipy`: Scientific computing
- `pymatgen` or `ase`: CIF I/O operations
- `pytest`: Testing (optional)

## License

MIT License. See [LICENSE](LICENSE) for details.