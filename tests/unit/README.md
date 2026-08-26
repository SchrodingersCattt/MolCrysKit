# Unit tests (tests/unit)

Unit tests mirror the top-level `molcrys_kit` package areas. They use pytest and shared fixtures from `tests/conftest.py`.

## File layout

| Directory | Coverage |
|-----------|----------|
| `analysis/` | Analysis, disorder resolution, interactions, shapes, and volume |
| `cli/` | Command-line interface behavior |
| `constants/` | Constants and configuration values |
| `contracts/` | Documentation, skill, and renderer contracts that span packages |
| `io/` | CIF, EXTXYZ, POSCAR, and output handling |
| `operations/` | Structure transformations, carving, paths, and surfaces |
| `structures/` | Core structure types, polyhedra, symmetry, and trajectories |
| `utils/` | Geometry and graph-adjacent utilities |

Keep a test in the directory of the production area it primarily exercises. Use `contracts/` only when no single production area owns the behavior.

## Running tests

From the project root:

```bash
# Unit tests only
pytest tests/unit -v

# One production area
pytest tests/unit/io -v

# Cross-package contracts
pytest tests/unit/contracts -v

# With coverage
pytest tests/unit --cov=molcrys_kit --cov-report=term-missing
```

Shared fixtures in `tests/conftest.py` include `test_cif_path`, `simple_crystal`, `water_atoms`, `cubic_lattice_10`, and others. Do not redefine shared fixtures or modify `sys.path` in unit tests.
