---
name: analyze-molecular-crystal
description: "Use before reasoning about molecular, ionic, or framework crystals: readiness, components, solvent/coordination, interactions, facets, geometry, stoichiometry, charge, protonation, H-bonds, pores, and accessible volume."
---

# Analyze Molecular Crystals

Use structural analysis to answer a stated chemical or modeling question. Report assumptions and parameters rather than treating parser output as ground truth.

## Core workflow

1. Read [installation and verification](./references/install.md).
2. Always read [CLI and Python API](./references/cli-and-api.md). Read the repository `docs/cli.md`, probe the installed command, and prefer the CLI when it covers the requested analysis.
3. Always read [input diagnosis](./references/diagnose-input.md), then inventory the components and run the complete single-structure sanity suite.
4. Decide whether the question concerns the observed disordered model, one ordered realization, or an ensemble. Do not switch between them silently.
5. Select the relevant analysis below and record every threshold, cutoff, index range, radii set, and charge/protonation assumption.
6. Read [interpretation and reporting](./references/interpret-results.md) before interpreting or delivering results.

Read [the Python analysis API](./references/analysis-api.md) only for API-only analyses or detailed result objects. If the input is a CSD refcode, retrieve and export it first with the separately installable `query-csd-structures` skill. Probe each live subcommand with `--help` before relying on its options.

## Start with structure quality

```bash
mck analyze sanity-check input.cif --json -o input.sanity.json
```

The default single-structure suite runs six checks: hard clashes, intermolecular clashes, isolated atoms, hydrogen presence, formula consistency, and bond distances. Topology preservation is a separate before/after comparison that requires two structures. Use `--checks` or threshold overrides only for a documented domain reason, not to manufacture a pass.

A failed sanity check does not always make analysis impossible, but it changes what can be claimed. For example, interaction analysis is incomplete when H atoms are missing, and molecular inventory is unreliable when bond perception merges components.

## Identify molecules and topology-aware species

```bash
mck io info input.cif
mck io molecules input.cif --json
```

Report formula, count, zero-based molecule index, species ID, and centroid. Species IDs distinguish topology groups within one formula and are preferable to formula-only labels for isomers, defects, solvent removal, or vacancy generation.

If disorder is present, state whether the inventory describes unresolved sites or an ordered replica. Adjust `--bond-scale` only after inspecting the implied bonds.

## Analyze weak intermolecular interactions

```bash
mck analyze interactions input.cif --json
```

The CLI profile aggregates hydrogen bonds, halogen bonds, and parallel/T-shaped pi stacking with continuous scores. CH-pi is subsumed by T-shaped pi stacking in the profile, while standalone CH-pi and H-H contact records require the public Python detectors shown in [CLI and Python API](./references/cli-and-api.md). Interpret all geometry-based detections under the active criteria, not as interaction energies.

Before interpretation:

- verify H presence and protonation;
- verify molecule partitioning;
- distinguish intra- and intermolecular contacts;
- retain atom and molecule identities plus periodic-image information;
- report criteria or scoring parameters if the Python API is used to override defaults.

Compare ordered replicas when disorder changes donors, acceptors, rings, or close contacts.

## Rank BFDH facets

```bash
mck analyze bfdh input.cif --max-index 2 --top-n 10 --json
```

Report Miller index, interplanar spacing, relative morphological importance, `max_index`, symmetry/equivalence handling, and rank. BFDH is an empirical morphology shortlist based mainly on `d_hkl`; it is not a surface energy, growth kinetics calculation, or proof of experimental habit.

To construct a candidate surface, pass a selected Miller index to `mck operate slab` and compare terminations explicitly.

## Analyze coordination polyhedra and shape

```bash
mck analyze polyhedra input.cif \
  --central Zn --ligand O --level atom --json
mck analyze polyhedra input.cif \
  --central Zn --ligand H2O --level molecule --cutoff 6.0 --json
```

- `atom` level uses coordinating atoms.
- `molecule` level uses molecular units or moieties around a center.
- An explicit cutoff imposes a chemical model; report it.

Record center identity, ligand identity, selected neighbors, distances, coordination number, hull/planarity diagnostics, and the shape comparison. Use the Python API for detailed CShM classification:
see [the Python analysis API](./references/analysis-api.md).

Do not call a shape "octahedral" from coordination number alone. Report CShM/reference-shape evidence and ambiguity between close candidates.

## Analyze topology-aware stoichiometry

This is Python-API-only; use the [stoichiometry example](./references/analysis-api.md#topology-aware-stoichiometry).

The analyzer groups molecules first by formula and then graph isomorphism, so constitutional isomers can remain distinct. Report both the unit-cell population and GCD-reduced simplest ratio. Solvent identification is a heuristic lookup and must be confirmed chemically.

## Calculate van der Waals volume and accessible boundary

This is Python-API-only; use the [volume and boundary example](./references/analysis-api.md#volume-and-accessible-boundary).

The simple volume is a union/sum model of atomic spheres, not the crystallographic cell volume. Accessible-boundary calculation is documented for non-periodic structures; carve or extract a finite cluster before using it on a periodic solid. Report radii type, overlap correction, voxel size, probe radius, and sphere-point density. Converge voxel and sampling parameters for quantitative comparisons.

## Assign molecular formal charges

This is Python-API-only; use the [formal-charge example](./references/analysis-api.md#molecular-formal-charge).

Assignment priority is user map, pymatgen bond-valence auto-guess, then zero-valued fallback with source `none`. Always report `source`. Treat `auto_guess` as a hypothesis to verify, especially for radicals, mixed valence, organometallics, proton-transfer salts, and unusual coordination environments. Check cell electroneutrality independently.

## Characterize local chemical environments

This is Python-API-only; use the [chemical-environment example](./references/analysis-api.md#local-chemical-environment).

Use this for coordination counts, bond angles, ring membership/aromatic-ring heuristics, local planarity, and anion protonation-group diagnostics. Atom indices are molecule-local and zero-based. Geometry-derived aromaticity and anion-group assignment are heuristics; report the method and verify unusual motifs manually.

## Non-negotiable rules

- Run the six single-structure sanity checks before detailed analysis; compare topology separately when a reference and generated structure are available.
- State whether analysis used unresolved disorder, one ordered replica, or an ensemble.
- Do not interpret geometry-only interaction scores as energies.
- Do not interpret BFDH ranking as surface thermodynamics.
- Do not classify polyhedra from coordination number alone.
- Do not report formal charges without their source or accessible surfaces without probe/sampling parameters.
- Preserve machine-readable reports and the exact parameters needed to reproduce them.
