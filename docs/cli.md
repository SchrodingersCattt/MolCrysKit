# MolCrysKit CLI Reference

> **Status: Planned.** The CLI is not yet implemented.
> All functionality is currently accessed via the Python API.
> See [API & Recipes](api.md) for the quick reference.

## Planned Architecture

The CLI will follow a two-group subcommand pattern:

```
mck operate <command> [options]    # structural operations
mck analyze <command> [options]    # analysis & diagnostics
```

### `mck operate` — Structural Operations (planned)

| Command | Description | Python API |
|---|---|---|
| `mck operate read <CIF>` | Parse CIF, print molecule summary | `read_mol_crystal()` |
| `mck operate slab <CIF> -m 1,0,0` | Generate topological surface slab | `generate_topological_slab()` |
| `mck operate hydrogens <CIF>` | Add hydrogen atoms | `add_hydrogens()` |
| `mck operate carve <CIF>` | Carve QM cluster | `ClusterCarver.carve()` |
| `mck operate disorder <CIF>` | Resolve crystallographic disorder | `generate_ordered_replicas_from_disordered_sites()` |
| `mck operate desolvate <CIF>` | Remove solvent molecules | `Desolvator.remove_solvents()` |
| `mck operate vacancy <CIF>` | Create vacancy defect | `generate_vacancy()` |

### `mck analyze` — Analysis & Diagnostics (planned)

| Command | Description | Python API |
|---|---|---|
| `mck analyze bfdh <CIF>` | Enumerate BFDH facet candidates | `enumerate_bfdh_facets()` |
| `mck analyze interactions <CIF>` | Find intermolecular interactions | `find_hydrogen_bonds()` etc. |
| `mck analyze polyhedra <CIF>` | Coordination polyhedron analysis | `find_polyhedra()` |
| `mck analyze invariants --parent <CIF> <XYZ>...` | Check cluster carve invariants | existing diagnostic tool |

## Existing Diagnostic Tool

```bash
python -m molcrys_kit.analysis.cluster_invariants \
    --parent-cif <CIF> <XYZ>...
```

Checks C1–C10 carve invariants on carved QM clusters against the parent crystal.
Not yet registered as a `console_scripts` entry point.
