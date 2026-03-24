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
from ase import Atoms
from molcrys_kit.structures.crystal import MolecularCrystal

# 1. Create a toy system (e.g., 2 Water molecules in a unit cell)
# In practice, you would typically load this from a file: atoms = read('cif_file.cif')
atoms = Atoms(
    symbols=['O', 'H', 'H', 'O', 'H', 'H'],
    positions=[
        [1.0, 1.0, 1.0], [1.8, 1.0, 1.0], [0.7, 1.6, 1.0],  # Molecule 1
        [5.0, 5.0, 5.0], [5.8, 5.0, 5.0], [4.7, 5.6, 5.0]   # Molecule 2
    ],
    cell=[10.0, 10.0, 10.0],
    pbc=True
)

# 2. Initialize MolecularCrystal (Automatically identifies molecules via graph logic)
crystal = MolecularCrystal.from_ase(atoms)

# 3. Access Crystal & Molecular Properties
print(f"Lattice Parameters: {crystal.get_lattice_parameters()}")
print(f"Identified Molecules: {len(crystal.molecules)}") 

mol = crystal.molecules[0]
print(f"Molecule 1 Formula: {mol.get_chemical_formula()}")
print(f"Molecule 1 Center of Mass: {mol.get_center_of_mass()}")
```


## Running with Docker (no local installation required)

A `Dockerfile` is included so that MolCrysKit and its bundled CIF examples can
be run entirely inside a container — no local Python environment needed.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS / Windows)
  or the Docker Engine (Linux).

### Quick start

```bash
# 1. Clone the repository and enter the package directory
git clone https://github.com/SchrodingersCattt/MolCrysKit.git
cd MolCrysKit

# 2. Build the image (≈ 5–10 min on first run; subsequent builds use the cache)
docker build -t molcryskit:latest .

# 3. Run the smoke test to confirm everything works
docker run --rm molcryskit:latest python /opt/molcryskit/docker_smoke_test.py

# 4. Start the Jupyter notebook server
docker run -it --rm -p 8888:8888 molcryskit:latest
# Then open http://localhost:8888 in your browser.
# Example CIF files are available at /workspace/examples/ inside the container.
```

### One-step build + test helper

```bash
# From the MolCrysKit/ directory:
bash docker-test.sh
```

This script builds the image and runs the smoke test automatically, reporting
`ALL CHECKS PASSED` on success.

### Mounting your own data

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v /path/to/your/cif/files:/workspace/my_data \
    molcryskit:latest
```

Your files will be accessible at `/workspace/my_data/` inside the container.

## Documentation

For detailed architecture, tutorials, and API reference, please see the [`docs/`](docs/) directory.

## Project Structure

See the [`molcrys_kit/`](molcrys_kit/) directory for source code and the [`scripts/`](scripts/) directory for utility scripts (e.g. disorder diagnostics, molecule identification, CIF processing).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
