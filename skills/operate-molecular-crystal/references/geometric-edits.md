# Geometric Edits

Use one matching command, then return to the verification step in `SKILL.md`.

## Surface slab

```bash
mck analyze bfdh bulk.cif --max-index 2 --top-n 10
mck operate slab bulk.cif -o slab_001.cif \
  --miller 0 0 1 --layers 5 --vacuum 15 --terminations tasker_preferred
```

BFDH shortlists facets from interplanar spacing; it is not a surface-energy
calculation. Choose layer count or physical thickness from the task and inspect
plausible terminations when surface chemistry matters.

## Capped QM cluster

```bash
mck operate cluster bulk.cif -o cluster \
  --mode bond_shells --seed-index 42 --max-atoms 500 --freeze-shell 1
mck operate cluster bulk.cif -o cluster \
  --mode rcut --seed-index 42 --rcut 12 --freeze-shell 1
```

`bond_shells` follows topology; `rcut` follows a geometric embedding radius.
Use either seed element or seed indices, not both. Keep every generated XYZ with
its JSON sidecar and inspect each cap.

## Supercell

```bash
mck operate supercell bulk.cif -o bulk_2x2x2.cif --scale 2 2 2
```

Choose replication from the physical cutoff or convergence question. Confirm
vectors and atom/component counts scale by the product. Use
`--resolve-disorder` only when one ordered realization is intended.

## Finite particle or periodic void

```bash
mck operate nanocluster ordered.cif -o particle.extxyz \
  --shape ellipsoid --semi-axes 20 35 60 --vacuum 10 \
  --json-sidecar particle.json
mck operate void ordered_supercell.extxyz -o void.extxyz \
  --shape cylinder --radius 10 --height 40 --axis-vector 1 1 0.5 \
  --target-units 48 --json-sidecar void.json
```

`nanocluster` keeps selected complete units as a nonperiodic particle; `void`
removes selected complete units from a periodic host. These operations do not
cap a three-dimensional framework network.

## Reorientation

```bash
mck operate reorient bulk.cif -o aligned.cif \
  --direction 1 1 0 --target-axis z
```

`--direction` is a Miller direction. Verify the final alignment, handedness,
cell, and PBC numerically.

## Interpolation

```bash
mck operate interpolate start.cif end.cif -o path.extxyz \
  --method se3_screw --n-images 11 --include-endpoints
```

Require matching chemistry, atom count, molecule mapping, and atom order. Inspect
every image for periodic jumps, overlaps, flips, and implausible cell changes.
This initializes a path; it is not a converged NEB or transition state.

---
→ Return to [verification](./verification.md) after completing the edit.
