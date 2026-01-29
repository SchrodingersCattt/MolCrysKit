# Unit tests (tests/unit)

Unit tests live in this directory, grouped by module. They use pytest and shared fixtures from `tests/conftest.py`.

## File layout

| File | Modules covered |
|------|-----------------|
| `test_geometry.py` | `molcrys_kit.utils.geometry` (coordinate transforms, distances, angles, rotation, etc.) |
| `test_structures.py` | `molcrys_kit.structures` (MolAtom, CrystalMolecule, MolecularCrystal) |
| `test_constants.py` | `molcrys_kit.constants` and `constants.config` |
| `test_io_cif.py` | `molcrys_kit.io.cif` (read_mol_crystal, parse_cif_advanced, identify_molecules) |
| `test_analysis.py` | `analysis.species`, `analysis.interactions`, `analysis.stoichiometry` |
| `test_operations.py` | `operations.defects`, `operations.hydrogen_completion`, `operations.rotation` |
| `test_disorder.py` | `analysis.disorder` (DisorderInfo, DisorderGraphBuilder) |

## Running tests

From the project root:

```bash
# Unit tests only
pytest tests/unit -v

# With coverage
pytest tests/unit --cov=molcrys_kit --cov-report=term-missing
```

Shared fixtures in `tests/conftest.py` include `test_cif_path`, `simple_crystal`, `water_atoms`, `cubic_lattice_10`, and others. Do not modify `sys.path` in unit tests.
