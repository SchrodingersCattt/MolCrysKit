# MolCrysKit Scripts

This directory contains example scripts demonstrating the usage of MolCrysKit.

## Available Scripts

1. [atomic_properties_demo.py](file:///aisi-nas/guomingyu/personal/MolCrysKit/scripts/atomic_properties_demo.py) - Demonstrates how to access atomic properties
2. [extract_molecules.py](file:///aisi-nas/guomingyu/personal/MolCrysKit/scripts/extract_molecules.py) - Shows how to extract molecules from a crystal structure
3. [molecule_center_analyzer.py](file:///aisi-nas/guomingyu/personal/MolCrysKit/scripts/molecule_center_analyzer.py) - Analyzes molecular centers in a crystal

## Usage

To run any of these scripts, make sure you have installed MolCrysKit:

```bash
pip install -e .
```

Then run a script with Python:

```bash
python scripts/script_name.py
```

Example:

```python
from molcrys_kit.structures import Atom, Molecule, MolecularCrystal
from molcrys_kit.io import parse_cif
from molcrys_kit.analysis import assign_atoms_to_molecules
```

Please note that some scripts may require CIF files to demonstrate their functionality.