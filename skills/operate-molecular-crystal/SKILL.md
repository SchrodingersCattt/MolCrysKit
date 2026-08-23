---
name: operate-molecular-crystal
description: 'Generate or modify molecular crystal models with MolCrysKit. Use when resolving CIF disorder into ordered replicas, completing hydrogen atoms, generating topology-preserving surface slabs, carving finite nanoparticles or shaped voids, carving H-capped QM clusters, creating supercells, vacancies or desolvated structures, reorienting crystals, interpolating structures, translating, rotating or replacing molecules, or converting CIF, POSCAR, XYZ and ExtXYZ files. Covers organic crystals, multicomponent crystals, hybrids, MOF-like structures and polyatomic-ion salts, with operation-specific network limitations.'
---

# Operate on Molecular Crystals

Build computational models without losing track of molecular topology, chemical assumptions, or provenance.

## Core workflow

1. Read [installation and verification](./references/install.md).
2. Always read [CLI and Python API](./references/cli-and-api.md). Read the repository `docs/cli.md`, probe the installed command, and prefer the CLI when it covers the requested operation.
3. Always read [input diagnosis](./references/diagnose-input.md). Diagnose the input with `mck io info`, molecule inventory, and a sanity check before changing it.
4. Resolve ambiguous disorder, protonation, components, and bond perception before selecting an operation.
5. Select exactly one primary scenario below. Compose operations only when their order is chemically justified.
6. Read [formats and provenance](./references/formats-and-provenance.md) before choosing a downstream format.
7. Always read [operation verification](./references/verify-operations.md), validate every output, and retain parameters and provenance.

If the input is a CSD refcode rather than a local structure file, retrieve and export it first with the separately installable `query-csd-structures` skill. Probe each live subcommand with `--help` before relying on its options.

## Choose the operation from the modeling question

### Resolve disorder into ordered models

Use when a CIF contains alternate sites, partial occupancies, PART groups, or disorder assemblies and the target calculation needs full occupancy.

```bash
mck operate disorder input.cif -o ordered.cif --method optimal
mck operate disorder input.cif -o ensemble.cif --method random --count 20 --seed 42
mck operate disorder input.cif -o states.cif --method enumerate --count 20
```

- `optimal` produces one occupancy-favoured conflict-free model; it does not prove that the model is the experimentally dominant state.
- `random` is for reproducible ensembles; always set and report `--seed`.
- `enumerate` is for a bounded alternative set; inspect combinatorial growth.
- Symmetry-expanded copies are independent by default. Use `--coupled` only when the intended model requires symmetry copies to choose the same alternative.

Do not silently select the first replica from an ensemble.

### Complete missing hydrogen atoms

Use after deciding the intended protonation state and resolving disorder.

```bash
mck operate add-h ordered.cif -o hydrogenated.cif
mck operate add-h ordered.cif -o hydrogenated.cif \
  --target-elements N --target-elements O --optimize-torsion
mck operate add-h ordered.cif -o hydrogenated.cif \
  --rule 'N:target_coordination=3,geometry=trigonal_pyramidal'
```

`--target-elements` is a whitelist. Formula-moiety metadata corrects expected H counts when parseable; it does not determine an experimentally verified protonation state. Use `--no-formula-moiety` only when that CIF field is known to be unsuitable. Validate X–H distances, total H counts, clashes, and the interaction network.

### Generate a topology-preserving surface slab

Use for periodic surface calculations. First rank candidate facets with BFDH:

```bash
mck analyze bfdh bulk.cif --max-index 2 --top-n 10
mck operate slab bulk.cif -o slab_001.cif \
  --miller 0 0 1 --layers 5 --vacuum 15 --terminations tasker_preferred
```

BFDH ranks morphology candidates from interplanar spacing; it is not a surface-energy calculation. Choose `--layers` when a discrete plane count matters, or `--min-thickness` when physical thickness matters. Inspect all plausible terminations when termination chemistry can affect the result. Verify intact molecules, slab thickness, vacuum, charge, dipole, and both exposed surfaces.

### Carve an H-capped QM cluster

Choose seeds from molecule or atom inventory before carving.

```bash
mck operate cluster bulk.cif -o cluster \
  --mode bond_shells --seed-index 42 --max-atoms 500 --freeze-shell 1
mck operate cluster bulk.cif -o cluster \
  --mode rcut --seed-index 42 --rcut 12 --freeze-shell 1
```

- `bond_shells` follows topology and is preferred when chemically connected shells define the model.
- `rcut` is preferred when a geometric embedding radius defines the model.
- Use either `--seed-element` or one or more `--seed-index` values, not both.
- Record explicit C–C truncations passed as `--cut-cc-bonds '12,13;27,28'`.
- Review cap positions and never assume metal-boundary treatment is chemically valid without inspection.

Deliver every generated XYZ together with its JSON sidecar and convention note.

### Build a supercell

```bash
mck operate supercell bulk.cif -o bulk_2x2x2.cif --scale 2 2 2
```

Choose replication factors from the physical question: interaction cutoff, defect separation, phonon wavelength, or finite-size convergence. Verify cell vectors and that atom and molecule counts increase by the scale product.

Use `--resolve-disorder` for a CIF that must become one full-occupancy model.
Without it, large-model builders warn and preserve the unresolved input state in
metadata rather than silently choosing a disorder alternative.

### Carve a finite nanocluster or a shaped void

```bash
mck operate nanocluster ordered.cif -o particle.extxyz \
  --shape ellipsoid --semi-axes 20 35 60 --vacuum 10 \
  --json-sidecar particle.json
mck operate void ordered_supercell.extxyz -o void.extxyz \
  --shape cylinder --radius 10 --height 40 --axis-vector 1 1 0.5 \
  --target-units 48 --json-sidecar void.json
```

Use `nanocluster` when the selected complete molecules/ions become a finite,
non-periodic particle. Use `void` when the selected complete units are removed
and the periodic host remains. For comparable defect morphologies, provide one
`--species` ratio and the same `--target-units`; provide a complete
`--species-charge` map when neutrality must be verified. `centroid`,
`any_atom`, and `all_atoms` affect geometric hits but never permit partial
molecule deletion.

These operations do not cut or cap 3-D MOF/framework bonds. Reject or manually
review any input whose finite molecule/ion partition cannot be established.

### Create a vacancy or remove solvent

List species IDs first:

```bash
mck io molecules bulk.cif --json
mck operate vacancy bulk.cif -o vacancy.cif \
  --species C2H6O_1 1 --seed-index 0 --random-seed 42
mck operate desolvate bulk.cif -o dry.cif --targets H2O_1 --targets C2H6O_1
```

Use topology-aware species IDs rather than formula alone. State whether removal represents a neutral molecule, ion, correlated cluster, or occupancy model. After removal, reassess composition, net charge, local coordination, periodic image separation, and whether relaxation is required.

### Reorient a crystal

Use when a crystallographic direction must align with a Cartesian simulation axis, for example in shock or transport simulations.

```bash
mck operate reorient bulk.cif -o aligned.cif \
  --direction 1 1 0 --target-axis z
```

The direction is a Miller direction, not a plane normal request. Keep the default in-plane reduction unless a downstream convention requires the unreduced basis. Numerically verify the final alignment and cell handedness.

### Interpolate between endpoint structures

Use for initialization of a pathway; do not present interpolation as a converged NEB or transition state.

```bash
mck operate interpolate start.cif end.cif -o path.extxyz \
  --method se3_screw --n-images 11 --include-endpoints
```

- `se3_screw`: default rigid-body translation/rotation interpolation.
- `com_so3`: center-of-mass and rotation interpolation when that decomposition is intended.
- `slerp`: quaternion orientation interpolation for matching molecular poses.

Require compatible chemistry, atom counts, molecule matching, and atom ordering. Inspect all images for periodic jumps, flips, overlaps, and implausible cell deformation.

### Translate, rotate, or replace one molecule

This is Python-API-only. Use the [molecule-editing example](./references/cli-and-api.md#api-only-molecule-editing). Indices are zero-based, translation defaults to Cartesian Å, and rotation angles are degrees. For fractional translation, pass `fractional=True`.

### Convert structure formats

```bash
mck io convert input.cif -o output.extxyz
mck io convert input.cif -o POSCAR
mck io extract-molecule input.cif -o molecule.xyz --index 0 --json-sidecar molecule.json
```

Conversion is representation change, not chemical cleanup. Use ExtXYZ for MolCrysKit/ASE round-trips and multi-frame paths; use XYZ only for non-periodic molecules or clusters.

## Non-negotiable rules

- Diagnose before operating; never infer trustworthy chemistry from a successful parse.
- Resolve disorder before calculations that require one full-occupancy model, and preserve ensemble identity when more than one model is generated.
- Do not add H without an explicit protonation assumption.
- Use BFDH to shortlist slab orientations, not to claim surface stability.
- Never discard cluster sidecars, random seeds, charge assumptions, or operation parameters.
- Run the six single-structure sanity checks after every structural operation, compare topology separately when a reference exists, and inspect geometry visually.
