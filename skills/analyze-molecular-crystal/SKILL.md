---
name: analyze-molecular-crystal
description: 'Analyze molecular crystals with MolCrysKit. Use when identifying molecules and topology-aware species, checking seven structural sanity criteria, finding hydrogen bonds, pi stacking, halogen bonds, CH-pi interactions or H-H contacts, ranking BFDH facets, enumerating coordination polyhedra and CShM shapes, characterizing coordination, rings, planarity or anion groups, calculating stoichiometry, van der Waals volume or solvent-accessible boundaries, or assigning formal charges. Covers organic crystals, multicomponent crystals, hybrids, MOF-like structures and polyatomic-ion salts.'
---

# Analyze Molecular Crystals

Use structural analysis to answer a stated chemical or modeling question. Report assumptions and parameters rather than treating parser output as ground truth.

## Core workflow

1. Read [installation and verification](../references/install.md) and probe the live `mck` command.
2. Always read [CIF diagnosis](../references/cif-diagnosis.md), then inventory the components and run the complete sanity suite.
3. Decide whether the question concerns the observed disordered model, one ordered realization, or an ensemble. Do not switch between them silently.
4. Select the relevant analysis below and record every threshold, cutoff, index range, radii set, and charge/protonation assumption.
5. Read [verification and reporting](../references/verification.md) before interpreting or delivering results.

Use [the CLI reference](../references/cli-reference.md) for command shapes, [input and output](../references/input-output.md) for reports and datasets, [batch processing](../references/batch-processing.md) for many structures, and [CSD integration](../references/csd-integration.md) for refcode-based work.

## Start with structure quality

```bash
mck analyze sanity-check input.cif --json -o input.sanity.json
```

The complete suite checks hard clashes, intermolecular clashes, isolated atoms, hydrogen presence, topology preservation, formula consistency, and bond distances. Run all seven first. Use `--checks` or threshold overrides only for a documented domain reason, not to manufacture a pass.

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

MolCrysKit profiles hydrogen bonds, pi stacking, halogen bonds, CH-pi interactions, and H-H contacts with continuous scores. Interpret these as geometry-based detections under the active criteria, not as interaction energies.

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

To construct a candidate surface, pass a selected Miller index to the `operate-molecular-crystal` slab workflow and then compare terminations explicitly.

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

```python
from molcrys_kit.analysis import classify_shell, cshm, find_polyhedra
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
polyhedra = find_polyhedra(crystal, central="Zn", ligand="O", level="atom")
# Pass the centered ligand-coordinate array from a selected record to
# classify_shell or compare it with a chosen ideal using cshm.
```

Do not call a shape "octahedral" from coordination number alone. Report CShM/reference-shape evidence and ambiguity between close candidates.

## Analyze topology-aware stoichiometry

This is Python-API-only:

```python
from molcrys_kit.analysis import StoichiometryAnalyzer
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
analyzer = StoichiometryAnalyzer(crystal)
print(analyzer.species_map)
print(analyzer.get_simplest_unit())
analyzer.print_species_summary()
```

The analyzer groups molecules first by formula and then graph isomorphism, so constitutional isomers can remain distinct. Report both the unit-cell population and GCD-reduced simplest ratio. Solvent identification is a heuristic lookup and must be confirmed chemically.

## Calculate van der Waals volume and accessible boundary

This is Python-API-only:

```python
from molcrys_kit.analysis import calculate_accessible_boundary, calculate_total_volume
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
atoms = crystal.to_ase()
volume = calculate_total_volume(
    atoms,
    radii_type="vdw",
    overlap_correction=True,
    voxel_size=0.2,
)
boundary = calculate_accessible_boundary(
    atoms,
    probe_radius=1.4,
    radii_type="vdw",
    n_sphere_points=200,
)
print(volume, boundary.shape)
```

The simple volume is a union/sum model of atomic spheres, not the crystallographic cell volume. Accessible-boundary calculation is documented for non-periodic structures; carve or extract a finite cluster before using it on a periodic solid. Report radii type, overlap correction, voxel size, probe radius, and sphere-point density. Converge voxel and sampling parameters for quantitative comparisons.

## Assign molecular formal charges

This is Python-API-only:

```python
from molcrys_kit.analysis import assign_mol_formal_charges
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
charges = assign_mol_formal_charges(
    crystal,
    mol_charge_map={"NH4": 1, "Cl": -1},
)
for signature, result in charges.items():
    print(signature, result.formula, result.formal_charge, result.source)
```

Assignment priority is user map, pymatgen bond-valence auto-guess, then zero-valued fallback with source `none`. Always report `source`. Treat `auto_guess` as a hypothesis to verify, especially for radicals, mixed valence, organometallics, proton-transfer salts, and unusual coordination environments. Check cell electroneutrality independently.

## Characterize local chemical environments

This is Python-API-only:

```python
from molcrys_kit.analysis import ChemicalEnvironment
from molcrys_kit.io import read_mol_crystal

crystal = read_mol_crystal("input.cif")
molecule = crystal.molecules[0]
environment = ChemicalEnvironment(molecule)

print(environment.rings())
print(environment.get_local_geometry_stats(0))
print(environment.detect_ring_info(0))
print(environment.compute_anion_protonation_groups())
```

Use this for coordination counts, bond angles, ring membership/aromatic-ring heuristics, local planarity, and anion protonation-group diagnostics. Atom indices are molecule-local and zero-based. Geometry-derived aromaticity and anion-group assignment are heuristics; report the method and verify unusual motifs manually.

## Non-negotiable rules

- Run all seven sanity checks before interpreting detailed analysis.
- State whether analysis used unresolved disorder, one ordered replica, or an ensemble.
- Do not interpret geometry-only interaction scores as energies.
- Do not interpret BFDH ranking as surface thermodynamics.
- Do not classify polyhedra from coordination number alone.
- Do not report formal charges without their source or accessible surfaces without probe/sampling parameters.
- Preserve machine-readable reports and the exact parameters needed to reproduce them.
