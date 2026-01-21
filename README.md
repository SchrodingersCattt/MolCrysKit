# MolCrysKit: Molecular Crystal Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](#)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)

## Overview

MolCrysKit is a Python toolkit designed for handling molecular crystals, providing utilities for parsing crystallographic data, identifying molecules within crystals, and performing various analyses on molecular crystals using graph theory and the Atomic Simulation Environment (ASE).

## Key Features

- Robust Molecule Identification: Identify individual molecules within a crystal structure using graph-based algorithms
- Disorder Handling: Process disordered structures with graph algorithms
- Topological Surface Generation: Create surface slabs while preserving molecular topology
- Hydrogen Addition: Automatically add hydrogen atoms based on geometric rules

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
```

## Documentation

For detailed architecture, tutorials, and API reference, please see the [`docs/`](docs/) directory.

## Project Structure

See the [`molcrys_kit/`](molcrys_kit/) directory for source code and the [`scripts/`](scripts/) directory for examples.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
