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

Two Dockerfiles are provided for different environments:

| File | Base image | Use case |
|------|-----------|----------|
| `Dockerfile` | `python:3.10-slim` | Local use, reviewers, CI |
| `Dockerfile.bohrium` | `registry.dp.tech/dptech/ubuntu:ubuntu24.04-py3.12` | Bohrium cloud platform |

Both install MolCrysKit directly from the GitHub archive. A local Docker build context is still used to start the build, but the package, notebook assets, and helper scripts are fetched from the selected GitHub ref inside the image build.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS / Windows)
  or the Docker Engine (Linux).

### Quick start (general use)

```bash
# 1. Clone the repository and enter the package directory
git clone https://github.com/SchrodingersCattt/MolCrysKit.git
cd MolCrysKit

# 2. Build the image (≈ 5–10 min on first run; subsequent builds use the cache)
docker build -t molcryskit:latest .

# 3. Run the smoke test to confirm everything works
docker run --rm molcryskit:latest python /opt/molcryskit/scripts/docker_smoke_test.py

# 4. Start the Jupyter notebook server
docker run -it --rm -p 8888:8888 molcryskit:latest
# Then open http://localhost:8888 in your browser.
# Example CIF files are available at /workspace/notebook/example/ inside the container.
```

### One-step build + test helper

```bash
# From the MolCrysKit/ directory:
bash scripts/docker-test.sh
```

This script builds the image and runs the smoke test automatically, reporting
`ALL CHECKS PASSED` on success.

### Bohrium cloud platform

```bash
# Build with the Bohrium-specific Dockerfile
docker build -f Dockerfile.bohrium -t molcryskit-bohrium:latest .

# Pin to an immutable Git tag instead of the moving main branch
# (recommended for archival/reviewer reproducibility)
docker build -f Dockerfile.bohrium \
    --build-arg MOLCRYSKIT_REF=refs/tags/v0.2.0 \
    -t molcryskit-bohrium:v0.2.0 .
```

The Bohrium image uses `pip install` from the GitHub archive zip (no `git clone`
required) and does not include Jupyter — Bohrium provides its own notebook
environment.

### Permanent image publication with GHCR

The Bohrium registry is convenient for cloud execution, but it should not be the
only archival location because image retention is controlled by the platform and
project namespace. For a stable public anchor, publish immutable release images
to GitHub Container Registry (GHCR).

The workflow [`publish-ghcr.yml`](.github/workflows/publish-ghcr.yml) pushes
[`Dockerfile`](Dockerfile) images to `ghcr.io/<owner>/molcryskit`:

- pushing a Git tag such as `v0.2.0` publishes `ghcr.io/<owner>/molcryskit:v0.2.0`
- stable release tags also receive `latest`
- manual dispatch can publish a development snapshot from a chosen Git ref

Recommended archival pattern:

```bash
# 1. Create and push an immutable release tag
git tag v0.2.0
git push origin v0.2.0

# 2. GitHub Actions publishes the image automatically to GHCR
#    ghcr.io/<owner>/molcryskit:v0.2.0
```

For Bohrium, keep using [`Dockerfile.bohrium`](Dockerfile.bohrium) as the
platform-specific runtime image, but cite the GitHub repository and GHCR image
as the permanent public anchor.

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

### Disorder Handling: Explicit vs Implicit Path Compatibility

MolCrysKit resolves crystallographic disorder through two complementary paths
that are designed to be fully compatible:

**Explicit path** (`_add_explicit_conflicts` + `_add_conformer_conflicts`):
Processes atoms tagged with `_atom_site_disorder_assembly` / `_atom_site_disorder_group`
in the CIF (e.g. SHELXL `PART` groups).  Mutual-exclusion edges are added for
atoms in different groups of the same assembly.  Hydrogen atoms bonded to an
explicit-disorder centre inherit the centre's group tag and are resolved together
with it.

**Implicit SP path** (`_add_implicit_sp_conflicts` + `_resolve_valence_conflicts`):
Processes partial-occupancy atoms on crystallographic special positions that carry
*no* disorder tags (common in SHELX riding-H refinements).  The algorithm
clusters copies of each asymmetric-unit site by proximity and adds mutual-exclusion
edges within each cluster.  For heavy atoms (N, P, S) with sufficient copies
around them, a tetrahedral/trigonal decomposition (`_sp_tetrahedral_single`) finds
geometrically valid orientation combinations and adds cross-cluster compatibility
constraints.

**Motif merge post-pass** (`_merge_chemical_motifs`):
After the Maximum-Weight Independent Set is solved, isolated XH_n centres
(e.g. NH4+, H2O) are reconstructed by a greedy distance- and angle-sorted H
selection.  Soft conflict edges (`valence_geometry`, `implicit_sp`, `geometric`)
are ignored here so that the strongly disordered SP-position motifs are not
excluded wholesale.  A key guard enforces **one H per crystallographic site
(asym_id)** when at least `max_H` distinct asym_ids are present: this prevents
the greedy from picking multiple copies of the same SHELX H position (which point
in nearly identical directions) before exhausting the other sites.

**Known-compatible CIF styles for NH4+:**

| Style | Example | Tags | Resolution path |
|---|---|---|---|
| Explicit PART groups (`dg=-1`) | DAI-4 N4 | `disorder_assembly=A/B/C/D`, `disorder_group=-1` | Explicit path |
| Implicit SHELX riding-H (`dg=0`) | DAI-4 N1 | No tags, `occ=1/sso`, `sso>1` | Implicit SP + motif merge |
| High-multiplicity SP (24 orientations) | PAP-4 | `occ=1/24`, no tags | Implicit SP + motif merge |

Both styles produce correct NH4+ (4 H per nitrogen) after resolution.

**Valence-completeness diagnostics:**
`DisorderSolver.solve()` automatically calls
`molcrys_kit.analysis.disorder.diagnostics.check_valence_completeness` on
each resolved structure.  If an isolated N or O centre has an H count outside
the expected range (N: 3–4; O: 0–2), a `WARNING` is emitted via the standard
logging system.  This catch-all does not modify the structure; it only makes
potential resolution artefacts visible.

## Project Structure

See the [`molcrys_kit/`](molcrys_kit/) directory for source code and the [`scripts/`](scripts/) directory for utility scripts (e.g. disorder diagnostics, molecule identification, CIF processing).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
