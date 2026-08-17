# Diagnose Input Before Analysis

Read this before interpreting a structure. Input quality determines which conclusions remain defensible.

## Required first pass

```bash
mck io info input.cif
mck io molecules input.cif --json
mck analyze sanity-check input.cif --json -o input.sanity.json
```

Record cell parameters, atom and molecule counts, species IDs, disorder, hydrogen status, and failed checks.

## Match the structural model to the question

State whether analysis uses:

- unresolved crystallographic disorder;
- one ordered realization;
- a reproducible ordered ensemble.

Compare replicas when disorder changes donors, acceptors, rings, coordination, contacts, surface ranking, or molecular identity. Do not silently analyze one arbitrary replica.

## Hydrogen and protonation

Missing or misplaced H atoms invalidate parts of interaction, charge, and chemical-environment analysis. Establish the intended protonation state before interpreting H bonds, CH-pi contacts, molecular charge, or anion protonation groups.

## Molecule partition and bond perception

Species identity and stoichiometry depend on graph construction. Start at `--bond-scale 1.0`; change it only after inspecting implied bonds. Rerun inventory and sanity checks after adjustment and record the effective scale.

Formula alone does not distinguish isomers. Use topology-aware species IDs when multiple graphs share a composition.

## Failed checks and claim limits

- clashes affect interaction and accessible-surface conclusions;
- isolated atoms or fragmented molecules affect inventory and stoichiometry;
- absent H affects donor/acceptor and charge interpretation;
- topology or formula inconsistency affects nearly every molecule-level result;
- abnormal bond distances affect chemical-environment and ring heuristics.

A failed check may be documented and tolerated for a specific purpose, but it must narrow the reported conclusion.

## Framework-like structures

Infinite covalent or coordination networks may not partition into conventional molecules. Inspect inventory and graph semantics before using molecule-level stoichiometry, formal charge, or weak-interaction analysis.
