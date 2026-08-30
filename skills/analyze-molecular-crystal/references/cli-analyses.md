# CLI Analyses

Use only the command needed for the question.

## Structure summary and readiness

```bash
mck analyze summary input.cif --json
mck analyze sanity-check input.cif --json
```

Summary reports composition, cell, symmetry, Wyckoff sites, and disorder. Its
formula is reduced; atom/species counts are crystallographic sites, with partial
occupancy reported separately. The sanity suite checks clashes, isolated atoms,
hydrogen presence, formula consistency, and bond distances.

## Components

```bash
mck io molecules input.cif --json
```

Report zero-based molecule index, topology-aware species ID, formula, count, and
centroid. State whether the inventory represents unresolved disorder or an
ordered realization. Change bond scale only after inspecting a concrete missing
or implausible bond.

## Intermolecular interactions

```bash
mck analyze interactions input.cif --json
```

Verify H/protonation and molecule partitioning first. Preserve atom, molecule,
and periodic-image identities. The command aggregates hydrogen bonds, halogen
bonds, and parallel/T-shaped pi stacking; geometry scores are not energies.

## BFDH facets

```bash
mck analyze bfdh input.cif --max-index 2 --top-n 10 --json
```

Report Miller index, interplanar spacing, rank, max index, and symmetry handling.
BFDH is a morphology shortlist, not surface energy or growth kinetics.

## Coordination and shape

```bash
mck analyze polyhedra input.cif \
  --central Zn --ligand O --level atom --json
mck analyze polyhedra input.cif \
  --central Zn --ligand H2O --level molecule --cutoff 6.0 --json
```

Record center/ligand identity, level, neighbors, distances, coordination number,
cutoff, and shape evidence. `atom` uses coordinating atoms; `molecule` uses
molecular units or moieties. An explicit cutoff is a chemical model. Use detailed
CShM only when the question requires distinguishing close reference shapes.
