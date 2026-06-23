# MolCrysKit: Molecular Crystal Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![PyPI](https://img.shields.io/pypi/v/molcrys-kit.svg)](https://pypi.org/project/molcrys-kit/)
[![unit-tests](https://github.com/SchrodingersCattt/MolCrysKit/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/SchrodingersCattt/MolCrysKit/actions/workflows/unit-tests.yml)

## Overview

MolCrysKit is a Python toolkit designed for handling molecular crystals, providing utilities for parsing crystallographic data, identifying molecules within crystals, and performing various analyses on molecular crystals using graph theory and the Atomic Simulation Environment (ASE).

## Key Features

- Robust Molecule Identification: Identify individual molecules within a crystal structure using graph-based algorithms
- Disorder Handling: Process disordered structures with graph algorithms
- Topological Surface Generation: Create surface slabs while preserving molecular topology
- Hydrogen Completion: Add hydrogen atoms with heuristic geometric placement rules

## Installation

### From PyPI (recommended)

```bash
pip install molcrys-kit
```

### From source (development)

```bash
git clone https://github.com/SchrodingersCattt/MolCrysKit.git
cd MolCrysKit
pip install -e ".[dev]"
```

All dependencies are declared in `pyproject.toml` (there is no separate
`requirements.txt`). `requires-python = ">=3.10"`. The available extras are:

| Extra | Adds |
|---|---|
| `[test]` | `pytest`, `pytest-cov` |
| `[vis]` | `nglview`, `py3Dmol` for 3-D visualisation in notebooks |
| `[dev]` | `[test]` + `[vis]` + `build`, `ruff>=0.15`, `pre-commit`, `nbstripout`, `twine` |

So a contributor environment is `pip install -e ".[dev]"` and a CI / minimal
test environment is `pip install -e ".[test]"`.

## Quick Start

Here's a simple example of how to use MolCrysKit:

```python
import molcrys_kit as mck
from ase import Atoms

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
crystal = mck.MolecularCrystal.from_ase(atoms)

# 3. Access Crystal & Molecular Properties
print(f"Lattice Parameters: {crystal.get_lattice_parameters()}")
print(f"Identified Molecules: {len(crystal.molecules)}") 

mol = crystal.molecules[0]
print(f"Molecule 1 Formula: {mol.get_chemical_formula()}")
print(f"Molecule 1 Center of Mass: {mol.get_center_of_mass()}")
```

## Citation

If you use MolCrysKit in academic work, please cite:

> Guo, M.-Y.; Zhang, W.-X. *MolCrysKit: A Topology-Aware Toolkit for Bridging Experimental Molecular-Crystal Structures and Simulation-Ready Modeling*. **J. Chem. Inf. Model.** **2026**, *66* (9), 4999-5007. https://doi.org/10.1021/acs.jcim.6c00168

For exact reproduction of the published JCIM results, use the archived
`v0.1.0` release together with the versioned container image and the material
under `paper/`. The `main` branch may continue to evolve after publication.


## Running with Docker

Two Dockerfiles are provided: `Dockerfile` (python:3.10-slim) for local use and
`Dockerfile.bohrium` for the Bohrium cloud platform. Both install MolCrysKit
from the GitHub archive.

```bash
git clone https://github.com/SchrodingersCattt/MolCrysKit.git
cd MolCrysKit
docker build -t molcryskit:latest .
docker run --rm molcryskit:latest python /opt/molcryskit/scripts/docker_smoke_test.py
docker run -it --rm -p 8888:8888 molcryskit:latest  # Jupyter
```

For Bohrium deployment, GHCR image publication, and mounting custom data, see
the [Docker Guide](docs/docker.md).

## Documentation

### Disorder Handling

MolCrysKit resolves crystallographic disorder through two complementary paths:
the **explicit path** processes CIF `_atom_site_disorder_assembly` / `_disorder_group`
tags (e.g. SHELXL `PART` groups), while the **implicit SP path** handles
partial-occupancy atoms on special positions without disorder tags (SHELX
riding-H refinements). A motif-merge post-pass reconstructs isolated XH_n
centres (NH4+, H2O), and three replica-generation modes (`optimal`, `random`,
`enumerate`) support downstream ensemble workflows. Valence-completeness
diagnostics flag incomplete H-shells automatically.

For the full three-phase pipeline, edge-type priority table, solver modes, and
symmetry-copy decoupling details, see [Architecture](docs/architecture.md).

### Documentation Index

| Document | Covers |
|---|---|
| [Architecture](docs/architecture.md) | Core philosophy, disorder solver pipeline, edge types, solver modes |
| [Tutorials](docs/tutorials.md) | Hydrogen completion, surface slabs, BFDH facets, cluster carving, molecule manipulation |
| [API Reference](docs/api_reference.md) | Key classes and functions by module |
| [Docker Guide](docs/docker.md) | Docker quick start, Bohrium cloud, GHCR archival, mounting data |

## Project Structure

See the [`molcrys_kit/`](molcrys_kit/) directory for source code and the [`scripts/`](scripts/) directory for utility scripts (e.g. disorder diagnostics, molecule identification, CIF processing).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
