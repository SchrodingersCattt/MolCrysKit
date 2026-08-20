# Diagnose Input Before Operating

Read this before modifying a structure. A parseable file is not necessarily a chemically trustworthy model.

## Required first pass

```bash
mck io info input.cif
mck io molecules input.cif --json
mck analyze sanity-check input.cif --json -o input.sanity.json
```

Record cell parameters, atom and molecule counts, formulas/species IDs, disorder, and failed checks.

## Decide whether disorder must be resolved

Alternate sites, partial occupancies, PART groups, and disorder assemblies do not define one unique full-occupancy structure.

- `optimal`: one occupancy-favoured conflict-free model.
- `random`: a reproducible ensemble; set `--seed`.
- `enumerate`: alternatives when the state space is small.
- default: symmetry-expanded copies choose independently.
- `--coupled`: lock symmetry-related choices only when physically intended.

Validate every generated replica and do not silently select the first member of an ensemble.

## Decide whether hydrogen completion is justified

Compare molecule formulas, `_chemical_formula_moiety`, sanity results, expected valence, and intended protonation. Formula metadata can constrain H counts, but H placement remains heuristic. Determine charge and protonation before adding H.

## Identify removable and editable components

Use topology-aware species IDs to distinguish target molecules, counterions, coformers, solvent, and formula-identical isomers. Use zero-based molecule and atom indices only after saving the inventory that defines them.

## Check bond perception

Start with `--bond-scale 1.0`.

- decrease it only when implausible long bonds merge components;
- increase it only when plausible bonds are missing;
- rerun inventory and sanity checks after every change;
- record the effective value.

Do not tune the scale solely to force an expected molecule count.

## Framework-like structures

Molecule-level operations assume a meaningful graph partition. Infinite covalent or coordination networks may not map to conventional molecules. Inspect the inventory before applying molecule translation, vacancy, desolvation, slab, or cluster workflows.
