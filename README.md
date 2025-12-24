# MolCrysKit: Molecular Crystal Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](#)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)

## Overview

MolCrysKit is a Python toolkit designed for handling molecular crystals, providing utilities for parsing crystallographic data, identifying molecules within crystals, and performing various analyses on molecular crystals.

## Features

- Parse crystallographic data from CIF files
- Identify individual molecules within a crystal structure using graph-based algorithms
- Analyze molecular properties such as centroids, center of mass, and principal axes
- Handle atomic properties (mass, radius, etc.)
- Access crystal lattice parameters (vectors and cell parameters)
- Retrieve default atomic radii from crystal structures
- Customize bond thresholds for specific atom pairs
- Graph-based molecular representation using NetworkX
- Convert molecular crystals to ASE Atoms objects with the to_ase method

## Dependencies

- Python 3.7+
- NumPy
- SciPy
- ASE (Atomic Simulation Environment)
- NetworkX
- Pymatgen (optional, for CIF parsing)

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
from ase import Atoms
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.io.cif import read_mol_crystal

# Create a CrystalMolecule from ASE Atoms
atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[
        [0.000000, 0.000000, 0.000000],
        [0.756950, 0.585809, 0.000000],
        [-0.756950, 0.585809, 0.000000]
    ]
)

# Create a CrystalMolecule (inherits from ASE Atoms)
molecule = CrystalMolecule(atoms)

# Access molecular properties
print(f"Chemical formula: {molecule.get_chemical_formula()}")
print(f"Number of atoms: {len(molecule)}")
print(f"Center of mass: {molecule.get_center_of_mass()}")

# Work with molecular graphs
graph = molecule.graph
print(f"Graph nodes: {graph.number_of_nodes()}")
print(f"Graph edges: {graph.number_of_edges()}")

# Display graph structure
print("Molecular connectivity:")
for node, data in graph.nodes(data=True):
    position = molecule.get_positions()[node]
    print(f"  Atom {node} ({data['symbol']}): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")

for u, v, data in graph.edges(data=True):
    print(f"  Bond {u}-{v}: {data['distance']:.3f} Å")

# Load a molecular crystal from a CIF file
crystal = read_mol_crystal('bulk.cif')
print(f"Loaded crystal with {len(crystal)} atoms and {len(crystal.molecules)} molecules")
print(f"Cell parameters: {crystal.get_cell_lengths_and_angles()}")
print(f"Space group: {crystal.get_space_group_symbol()}")

# The crystal object contains all molecules identified in the CIF file
# You can access individual molecules or perform analyses on the entire crystal structure

# Convert molecular crystal to ASE Atoms object
ase_atoms = crystal.to_ase()
print(f"Converted to ASE Atoms: {len(ase_atoms)} atoms")
```

## Project Structure

- `molcrys_kit/` - Main source code directory
  - `analysis/` - Analysis tools for molecular crystals
  - `constants/` - Physical constants and atomic properties
  - `io/` - Input/output functionality (CIF parsing, etc.)
  - `operations/` - Operations on molecular structures (rotation, perturbation, etc.)
  - `structures/` - Core structural classes (Atom, CrystalMolecule, Crystal)
  - `utils/` - Utility functions and helper classes
- `scripts/` - Example scripts demonstrating various functionalities
- `tests/` - Unit tests and integration tests
  - `unit/` - Unit tests for individual components
  - `data/` - Test data files (CIF files, etc.)

## Examples

Several example scripts are provided in the `scripts/` directory:

- `scripts/demo_custom_bond_thresholds.py` - Demonstration of customizing bond thresholds
- `scripts/disorder_enumeration.py` - Disorder enumeration example
- `scripts/lattice_parameters_demo.py` - Lattice parameter calculation demonstration
- `scripts/molecular_centroids.py` - Calculation of molecular centroids
- `scripts/molecule_center_analyzer.py` - Molecular center analysis
- `scripts/molecule_center_analyzer_with_ase.py` - ASE-integrated molecular center analysis
- `scripts/rotate_molecule.py` - Example of rotating molecules
- `scripts/test_atomic_properties.py` - Testing atomic property functions
- `scripts/test_metal_bonding.py` - Testing metal bonding analysis

## Tests

Unit tests can be run with:

```bash
python tests/unit/test_structures.py
python tests/test_cif_parsing.py
```

Or with pytest (if installed):

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Surface Generation (Slab Creation)

MolCrysKit provides functionality for generating surface slabs from molecular crystals while preserving molecular topology. The surface generation ensures that intramolecular bonds are not broken during the cutting process, treating molecules as rigid units.

### Key Features:
- Miller index specification for surface orientation (h, k, l)
- Preservation of molecular topology during cutting
- Adjustable number of layers and vacuum spacing
- Automatic molecular centroid-based layer assignment

### Best Practices:
- Choose Miller indices that align with crystal symmetry for optimal results
- Use sufficient vacuum spacing (typically 10-20 Å) to avoid inter-slab interactions
- Consider the number of layers needed for your specific application
- Validate that molecules remain intact after slab generation

### Basic Usage:
```python
from ase.io import write
from molcrys_kit.operations import generate_topological_slab

# Load a molecular crystal from a CIF file
crystal = read_mol_crystal('bulk.cif')

# Generate a surface slab with specific Miller indices
slab = generate_topological_slab(
    crystal=crystal,
    miller_indices=(1, 1, 0),  # Miller indices of the surface
    layers=3,                  # Number of layers in the slab
    vacuum=10.0                # Vacuum thickness in Angstroms
)

print(f"Generated slab with {len(slab.molecules)} molecules")

# Convert the slab to ASE Atoms object using the to_ase method
slab_atoms_obj = slab.to_ase()

# Save the generated slab to a CIF file
write_mol_crystal(slab_atoms_obj, 'slab.cif')
```
