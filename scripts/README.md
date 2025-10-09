# MolCrysKit Scripts

This directory contains example scripts that demonstrate how to use the MolCrysKit library for various molecular crystal analysis tasks.

## Available Scripts

### 1. Complete Molecular Center Analyzer
**File**: [complete_molecule_center_analyzer.py](file:///aisi-nas/guomingyu/personal/MolCrysKit/scripts/complete_molecule_center_analyzer.py)

This script demonstrates how to:
- Create molecular crystals with proper molecular units
- Analyze molecular centers of mass
- Perform basic molecular operations

**Usage**:
```bash
python complete_molecule_center_analyzer.py
```

### 2. Improved Molecular Center Analyzer
**File**: [improved_molecule_center_analyzer.py](file:///aisi-nas/guomingyu/personal/MolCrysKit/scripts/improved_molecule_center_analyzer.py)

This script shows how to:
- Parse CIF files (requires pymatgen)
- Identify molecular units in a crystal structure
- Analyze molecular centers of mass

**Usage**:
```bash
python improved_molecule_center_analyzer.py --sample  # Run on sample data
python improved_molecule_center_analyzer.py structure.cif  # Run on actual CIF file
```

### 3. Basic Molecular Center Analyzer
**File**: [molecule_center_analyzer.py](file:///aisi-nas/guomingyu/personal/MolCrysKit/scripts/molecule_center_analyzer.py)

A basic script showing the core functionality:
- Basic CIF parsing
- Simple molecular center analysis

**Usage**:
```bash
python molecule_center_analyzer.py --sample  # Run on sample data
python molecule_center_analyzer.py structure.cif  # Run on actual CIF file
```

## How to Write Your Own Analysis Scripts

To write a script that loads a crystal structure and analyzes molecular centers:

1. Import the necessary modules:
```python
from molcrys.structures import Atom, Molecule, MolecularCrystal
# Or for file I/O:
from molcrys.io import parse_cif
from molcrys.analysis import assign_atoms_to_molecules
```

2. Load or create a crystal structure:
```python
# From CIF file:
crystal = parse_cif("path/to/structure.cif")

# Or create manually:
crystal = MolecularCrystal(lattice, [molecule1, molecule2, ...])
```

3. Analyze molecular centers:
```python
for i, molecule in enumerate(crystal.molecules):
    center = molecule.compute_center_of_mass()
    print(f"Molecule {i}: {center}")
```

## Requirements

Make sure you have installed MolCrysKit in development mode:
```bash
pip install -e .
```

Some scripts may require additional dependencies like `pymatgen` for CIF file parsing.