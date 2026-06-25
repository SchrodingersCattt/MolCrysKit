# MolCrysKit CLI

MolCrysKit installs the `mck` command.

```text
mck [--verbose | --quiet] [--version]
├── io       # read, summarize, convert
├── operate  # generate modified structures
└── analyze  # print analysis reports
```

## I/O

```bash
mck io info structure.cif
mck io convert structure.cif -o structure.poscar
mck io convert structure.cif -o structure.extxyz
```

Output format is selected from the output extension: `.cif`, `.poscar`/`.vasp`, `.xyz`, or `.extxyz`.

## Operations

```bash
mck operate disorder disordered.cif -o ordered.cif --method optimal
mck operate add-h bulk.cif -o bulk_h.cif --target-elements C --target-elements N --target-elements O
mck operate slab bulk.cif -o slab.cif --miller 1 0 0 --layers 3 --vacuum 15
mck operate slab bulk.cif -o slab.cif --miller 1 0 0 --terminations all
mck operate supercell bulk.cif -o supercell.cif --scale 2 2 1
mck operate vacancy bulk.cif -o vacancy.cif --species Water 1
mck operate desolvate bulk.cif -o dry.cif --targets Water
mck operate interpolate start.cif end.cif -o path.extxyz --n-images 11
```

### Cluster carving

Canonical command:

```bash
mck operate cluster bulk.cif -o cluster --seed-index 17 --mode bond_shells --freeze-shell 1
mck operate cluster bulk.cif -o cluster --seed-element Zn --seed-merge-radius 3.8
mck operate cluster bulk.cif -o cluster_rcut --seed-index 17 --mode rcut --rcut 6.5
```

Each cluster group writes `<stem>__group<k>.xyz` plus `<stem>__group<k>.xyz.cluster.json`.

The old script entry point remains available as a compatibility wrapper:

```bash
python scripts/carve_cluster.py --cif bulk.cif --seed-index 17 --mode bond_shells --out cluster
```

## Analysis

```bash
mck analyze bfdh bulk.cif --top-n 5
mck analyze bfdh bulk.cif --top-n 5 --json
mck analyze interactions bulk.cif
mck analyze polyhedra bulk.cif --central Pb --ligand I --level atom
```

Analysis commands print tables by default. Use `--json` for machine-readable output.
