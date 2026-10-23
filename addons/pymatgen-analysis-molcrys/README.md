# pymatgen-analysis-molcrys

Pymatgen namespace addon that exposes [MolCrysKit](https://github.com/SchrodingersCattt/MolCrysKit)
molecular-crystal workflows under `pymatgen.analysis.molcrys`.

After installation:

```python
from pymatgen.core import Structure
from pymatgen.analysis.molcrys import TopologicalSlabGenerator, generate_topological_slab

structure = Structure.from_file("molecular_crystal.cif")
slab = generate_topological_slab(structure, (0, 0, 1), layers=3, vacuum=15.0)
# slab is a pymatgen Structure; molecules are not cut through
```

## Install

From this monorepo (development):

```bash
pip install -e .                 # molcrys-kit at repo root
pip install -e addons/pymatgen-analysis-molcrys
```

From PyPI (when published):

```bash
pip install pymatgen-analysis-molcrys
```

## Design

- This package is a thin Structure-facing facade.
- Algorithms live in `molcrys-kit`.
- Conversion uses MolCrysKit's occupancy-preserving pymatgen bridge
  (`MolecularCrystal.to_pymatgen` / `from_pymatgen`), not `AseAtomsAdaptor`
  (which rejects partially occupied structures).

## Versioning

This addon has its own semver (`0.1.0` initially) and depends on
`molcrys-kit>=0.8.0b2`. Bump the lower bound when the facade needs a newer
engine API.
