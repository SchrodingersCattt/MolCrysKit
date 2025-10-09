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
from molcrys import MolecularCrystal
from molcrys.io import parse_cif
from molcrys.analysis import identify_molecules

# Parse a CIF file
crystal = parse_cif("structure.cif")

# Identify molecular units
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