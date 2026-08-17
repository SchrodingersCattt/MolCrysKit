# Diagnose a CIF Before Modeling

Read this before modifying or interpreting a structure. A parseable CIF is not necessarily a chemically trustworthy model.

## Required first pass

```bash
mck io info input.cif
mck io molecules input.cif --json
mck analyze sanity-check input.cif --json -o input.sanity.json
```

Record cell parameters, atom and molecule counts, molecular formulas/species IDs, detected disorder, and failed sanity checks.

## Disorder decision

If occupancies, disorder groups/assemblies, alternate positions, or explicit disorder are reported, do not treat the unresolved structure as a unique physical configuration.

- `optimal`: generate one occupancy-favoured, conflict-free model.
- `random`: sample an ensemble; always provide and record `--seed`.
- `enumerate`: generate alternatives when the state space is small enough to inspect.
- default decoupled behavior: symmetry-expanded copies can choose disorder alternatives independently.
- `--coupled`: lock symmetry-related copies together only when that physical assumption is intended.

Diagnose every generated replica separately; a successful solver exit is not chemical validation.

## Missing hydrogen decision

Compare:

1. formulas reported by `mck io molecules`,
2. CIF `_chemical_formula_moiety` when present,
3. `hydrogen_presence` and `formula_consistency` sanity checks,
4. expected valence and protonation state.

Do not add hydrogen merely because an X-ray structure contains few H atoms. Determine the intended protonation state first. Formula-moiety metadata can correct per-fragment H counts, but placement remains heuristic and requires inspection.

## Molecule and component diagnosis

Use molecule inventory to distinguish:

- target molecules or framework fragments,
- counterions,
- coformers,
- crystallization solvent,
- repeated topology-equivalent species,
- suspicious fragments caused by incorrect bond perception.

Species IDs are topology-aware. Use them, not formula alone, when isomers or multiple connected topologies share a composition.

## Bond perception

`--bond-scale` changes distance thresholds used to construct molecular connectivity:

- start at `1.0`;
- decrease slightly only when spurious long bonds merge components;
- increase slightly only when plausible bonds are missing and molecules fragment;
- rerun molecule inventory and sanity checks after every change;
- record the selected value.

Do not tune `--bond-scale` until the output merely matches an expected molecule count. Confirm the resulting bonds are chemically plausible.

## Broad material classes

MolCrysKit is useful beyond neutral organic crystals, including multicomponent crystals, polyatomic ion salts, molecular–inorganic hybrids, and some MOF-like structures. Its graph partitioning may not map an infinite covalent or coordination framework to conventional "molecules." Inspect the inventory before applying molecule-level operations.
