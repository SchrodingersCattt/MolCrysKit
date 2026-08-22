# CLI Reference

MolCrysKit's command-line interface (`mck`) provides tools for reading, analyzing, and generating molecular crystal structures.

## Installation & Usage

Installing MolCrysKit via `pip install molcrys-kit` automatically installs the `mck` command. All commands are self-documenting; use `--help` at any level to see details:

```bash
mck --help
mck io --help
mck operate cluster --help
```

## Global Options

- `--verbose` — Enable debug logging
- `--quiet` — Only show warnings and errors
- `--version` — Print version and exit
- `-h`, `--help` — Show help message

## Command Groups

### `mck io` — Read, summarize, and convert structures

| Command | Description | Key Options |
|---------|-------------|-------------|
| `mck io info INPUT` | Print molecular-crystal summary | `--resolve-disorder`, `--bond-scale FLOAT` |
| `mck io molecules INPUT` | List molecule inventory | `--json`, `--resolve-disorder`, `--bond-scale FLOAT` |
| `mck io extract-molecule INPUT` | Extract molecule(s) to separate file(s) | `-o/--output OUTPUT` (required), `--index INT`, `--formula STR`, `--species-id STR`, `--largest`, `--all`, `--center-vacuum FLOAT`, `--pbc BOOL`, `--json-sidecar PATH`, `--resolve-disorder`, `--bond-scale FLOAT` |
| `mck io convert INPUT` | Convert crystal structure format | `-o/--output OUTPUT` (required), `--resolve-disorder`, `--bond-scale FLOAT` |

### `mck operate` — Generate modified structures

| Command | Description | Key Options |
|---------|-------------|-------------|
| `mck operate disorder INPUT` | Resolve CIF disorder into ordered replicas | `-o/--output OUTPUT`, `--method {optimal,random,enumerate}`, `--count INT`, `--seed INT`, `--coupled` |
| `mck operate add-h INPUT` | Add missing hydrogen atoms | `-o/--output OUTPUT`, `--bond-scale FLOAT`, `--target-elements STR` (repeatable), `--rule STR` (repeatable), `--optimize-torsion`, `--no-formula-moiety` |
| `mck operate slab INPUT` | Generate surface slab models | `-o/--output OUTPUT`, `--miller H K L`, `--layers INT`, `--min-thickness FLOAT`, `--vacuum FLOAT`, `--terminations {single,tasker_preferred,all,INDEX}` |
| `mck operate cluster INPUT` | Carve molecular clusters | `-o/--output OUTPUT`, `--mode {bond_shells,rcut}`, `--seed-element STR`, `--seed-index INT` (repeatable), `--max-atoms INT`, `--cut-cc-bonds I,J;K,L`, `--rcut FLOAT`, `--freeze-shell {0,1,2}`, `--cap-distance FLOAT`, `--cap-bond-length ELEM=DIST` (repeatable), `--seed-merge-radius FLOAT`, `--convention-reference STR`, `--no-stop-at-non-seed-metals` |
| `mck operate nanocluster INPUT` | Carve a finite nanocluster without cutting molecules | `-o/--output OUTPUT`, `--shape {sphere,box,ellipsoid,cylinder}`, `--size X Y Z`, `--radius FLOAT`, `--semi-axes A B C`, `--height FLOAT`, `--axis {x,y,z}`, `--axis-vector X Y Z`, `--topology-unit {molecule,unit_cell}`, `--target-units INT`, `--center X Y Z`, `--center-frac U V W`, `--center-kind {centroid,com}`, `--vacuum FLOAT`, `--batch-size INT`, `--resolve-disorder`, `--json-sidecar PATH` |
| `mck operate void INPUT` | Remove complete molecular/ionic units to form a shaped void | `-o/--output OUTPUT`, `--shape {sphere,box,ellipsoid,cylinder,through-cylinder}`, `--size X Y Z`, `--radius FLOAT`, `--semi-axes A B C`, `--height FLOAT`, `--axis {x,y,z}`, `--axis-vector X Y Z`, `--direction-hkl H K L`, `--center X Y Z`, `--center-frac U V W`, `--hit-mode {centroid,any_atom,all_atoms}`, `--boundary-policy {inside,cover}`, `--target-units INT`, `--species SPECIES_ID COUNT`, `--species-charge SPECIES_ID CHARGE`, `--periodic-images/--no-periodic-images`, `--batch-size INT`, `--resolve-disorder`, `--json-sidecar PATH` |
| `mck operate supercell INPUT` | Create supercells | `-o/--output OUTPUT`, `--scale A B C`, `--resolve-disorder` |
| `mck operate vacancy INPUT` | Generate vacancy defects | `-o/--output OUTPUT`, `--species SPECIES_ID COUNT` (repeatable), `--seed-index INT`, `--method STR`, `--random-seed INT` |
| `mck operate desolvate INPUT` | Remove solvent molecules | `-o/--output OUTPUT`, `--targets STR` (repeatable, required) |
| `mck operate interpolate START END` | Interpolate between structures | `-o/--output OUTPUT`, `--method {se3_screw,com_so3,slerp}`, `--n-images INT`, `--include-endpoints/--exclude-endpoints` |
| `mck operate reorient INPUT` | Reorient crystal for axis-aligned simulations | `-o/--output OUTPUT`, `--direction H K L`, `--target-axis {x,y,z}` (default: z), `--no-reduce` |

### `mck analyze` — Analyze crystals and print reports

| Command | Description | Key Options |
|---------|-------------|-------------|
| `mck analyze bfdh INPUT` | Rank low-index facets by BFDH morphology | `--max-index INT`, `--top-n INT`, `--json` |
| `mck analyze interactions INPUT` | Summarize weak interactions | `--json` |
| `mck analyze polyhedra INPUT` | Enumerate coordination polyhedra | `--central STR` (required), `--ligand STR` (required), `--level {atom,molecule}`, `--cutoff FLOAT`, `--json` |
| `mck analyze sanity-check INPUT` | Run structural sanity checks | `--checks STR`, `--hard-clash-scale FLOAT`, `--hard-clash-tolerance FLOAT`, `--intermolecular-clash-scale FLOAT`, `--intermolecular-clash-tolerance FLOAT`, `--ignore-hh/--no-ignore-hh`, `--max-clashes INT`, `--bond-min-factor FLOAT`, `--bond-max-factor FLOAT`, `--isolated-elements STR`, `-o/--output OUTPUT`, `--json` |

## Common Patterns

### Disorder Resolution

```bash
# View disorder information
mck io info structure.cif

# Generate optimal ordered replica
mck operate disorder structure.cif -o ordered.cif

# Generate 10 random configurations
mck operate disorder structure.cif -o replicas.cif --method random --count 10
```

### Molecule Extraction

```bash
# List all molecules with species IDs
mck io molecules structure.cif --json

# Extract largest molecule
mck io extract-molecule structure.cif -o molecule.xyz --largest

# Extract by formula
mck io extract-molecule structure.cif -o caffeine.xyz --formula C8H10N4O2

# Extract all molecules as separate files
mck io extract-molecule structure.cif -o mol.xyz --all
```

### Cluster Carving

```bash
# 3-shell cluster around carbon atoms
mck operate cluster structure.cif -o cluster.xyz --seed-element C --mode bond_shells

# Radius cutoff with hydrogen caps
mck operate cluster structure.cif -o cluster.xyz --seed-index 42 --mode rcut --rcut 12.0 --cap-distance 1.1
```

### Topology-Preserving Nanoclusters

`nanocluster` selects complete molecules or complete translated source-cell
packets. It never cuts atoms or adds caps. With no `--target-units`, the preset
shape is applied exactly. With `--target-units N`, the nearest `N` units by the
shape field are selected from the preset's bounding box.

```bash
# Fixed 60 Å sphere, selecting complete molecules by geometric centroid
mck operate nanocluster structure.cif -o sphere.extxyz \
  --shape sphere --radius 30 --vacuum 10

# Exactly 537 complete source-cell packets in a needle-shaped search box
mck operate nanocluster adn.cif -o needle.extxyz \
  --shape box --size 30 30 600 --topology-unit unit_cell --target-units 537

# Finite cylinder along x, selecting molecules by center of mass
mck operate nanocluster structure.cif -o cylinder.extxyz \
  --shape cylinder --radius 25 --height 100 --axis x --center-kind com
```

For a 12-atom source cell, the fixed-count example always contains exactly
`537 × 12 = 6444` atoms, independent of whether the search box is needle-like,
plate-like, or nearly isotropic. Custom implicit functions are available through
the Python API only.

### Topology-Preserving Voids

`void` removes whole topology-derived molecules or polyatomic ions and retains
the source lattice and PBC. It does not cut atoms, rebuild bonds, or add caps.
The default `target_spec` is the simplest source species ratio. With
`--target-units N`, each species contributes exactly `N × ratio` molecules,
making different shapes directly comparable by missing composition, mass, and
atom count.

```bash
# Fixed-count spherical void with an explicit neutral ionic formula unit
mck operate void ordered.extxyz -o sphere_void.extxyz \
  --shape sphere --radius 18 --center-frac 0.5 0.5 0.5 \
  --target-units 48 \
  --species C6H14N2_1 1 --species ClO4_1 3 --species H4N_1 1 \
  --species-charge C6H14N2_1 2 --species-charge ClO4_1 -1 \
  --species-charge H4N_1 1 --json-sidecar sphere_void.json

# Periodically continuous cylinder along the reduced [1 1 0] lattice direction
mck operate void ordered.extxyz -o through_void.extxyz \
  --shape through-cylinder --radius 10 --direction-hkl 1 1 0
```

`centroid`, `any_atom`, and `all_atoms` change geometric hit testing only;
every selected object is still removed whole. With no fixed count,
`--boundary-policy inside` removes the largest complete stoichiometric packet
contained by the raw hits, while `cover` supplements each deficient species
with its nearest outside candidates. If charge validation is requested, the
charge map must cover every target species and both the formula unit and final
removed cluster must be neutral.

For large production models, use `--resolve-disorder` on CIF input or resolve
an ordered replica first. Without it, unresolved partial occupancies issue a
warning and are recorded in metadata/sidecars; they are never silently chosen.
Three-dimensional MOFs and other infinite bonded networks are not supported by
`nanocluster` or `void`: these commands do not sever or cap framework bonds.

### Surface Slabs

```bash
# (001) slab with 5 layers and 15 Å vacuum
mck operate slab structure.cif -o slab_001.cif --miller 0 0 1 --layers 5 --vacuum 15.0
```

### Crystal Reorientation

```bash
# Reorient crystal so [110] direction is along Z (for MSST shock simulations)
mck operate reorient structure.cif -o reoriented.cif --direction 1 1 0

# Align [111] along X axis
mck operate reorient structure.cif -o reoriented.cif --direction 1 1 1 --target-axis x
```

## See Also

- [API Documentation](api.md) — Python library reference
- [Tutorials](tutorials.md) — Step-by-step guides
- [Architecture](architecture.md) — Design rationale
