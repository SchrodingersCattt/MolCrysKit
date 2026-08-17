---
name: operate-molecular-crystal
description: 'Generate or modify molecular crystal models with MolCrysKit. Use when resolving CIF disorder into ordered replicas, completing hydrogen atoms, generating topology-preserving surface slabs, carving H-capped QM clusters, creating supercells, vacancies or desolvated structures, reorienting crystals, interpolating structures, translating, rotating or replacing molecules, or converting CIF, POSCAR, XYZ and ExtXYZ files. Covers organic crystals, multicomponent crystals, hybrids, MOF-like structures and polyatomic-ion salts.'
---

# Operate on Molecular Crystals

Build computational models without losing track of molecular topology, chemical assumptions, or provenance.

## Core workflow

1. Read [installation and verification](../references/install.md) and probe the live `mck` command.
2. Always read [CIF diagnosis](../references/cif-diagnosis.md). Diagnose the input with `mck io info`, molecule inventory, and a sanity check before changing it.
3. Resolve ambiguous disorder, protonation, components, and bond perception before selecting an operation.
4. Select exactly one primary scenario below. Compose operations only when their order is chemically justified.
5. Read [input and output](../references/input-output.md) before choosing a downstream format.
6. Always read [verification](../references/verification.md), validate every output, and retain parameters and provenance.

Use [the CLI reference](../references/cli-reference.md) for exact command shapes. Read [batch processing](../references/batch-processing.md) for datasets and [CSD integration](../references/csd-integration.md) when starting from a CSD refcode.

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

Use for periodic surface calculations. First rank candidate facets with the `analyze-molecular-crystal` skill:

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

This is Python-API-only:

```python
import numpy as np

from molcrys_kit.io import read_mol_crystal, write_cif
from molcrys_kit.operations import replace_molecule, rotate_molecule, translate_molecule

crystal = read_mol_crystal("bulk.cif")
crystal = translate_molecule(crystal, 0, np.array([0.2, 0.0, 0.0]))
crystal = rotate_molecule(crystal, 0, np.array([0.0, 0.0, 1.0]), 15.0, center="com")
crystal = replace_molecule(crystal, 1, "replacement.xyz", clash_threshold=1.2)
write_cif(crystal, "edited.cif")
```

Indices are zero-based, translation defaults to Cartesian Å, and rotation angles are degrees. For fractional translation, pass `fractional=True`. Replacement aligns centers and attempts random rotations to resolve clashes; set and document any reproducibility control available in the active API.

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
- Run all seven sanity checks after every structural operation and inspect geometry visually.
