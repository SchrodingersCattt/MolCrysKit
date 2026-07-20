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
| `info INPUT` | Print molecular-crystal summary | `--resolve-disorder`, `--bond-scale FLOAT` |
| `molecules INPUT` | List molecule inventory | `--json`, `--resolve-disorder`, `--bond-scale FLOAT` |
| `extract-molecule INPUT` | Extract molecule(s) to separate file(s) | `-o OUTPUT` (required), `--index INT`, `--formula STR`, `--species-id STR`, `--largest`, `--all`, `--center-vacuum FLOAT`, `--pbc BOOL`, `--json-sidecar PATH` |
| `convert INPUT` | Convert crystal structure format | `-o OUTPUT` (required), `--resolve-disorder`, `--bond-scale FLOAT` |

### `mck operate` — Generate modified structures

| Command | Description | Key Options |
|---------|-------------|-------------|
| `disorder INPUT` | Resolve CIF disorder into ordered replicas | `-o OUTPUT`, `--method {optimal,random,enumerate}`, `--count INT`, `--seed INT`, `--coupled` |
| `add-h INPUT` | Add missing hydrogen atoms | `-o OUTPUT`, `--target-elements STR` (repeatable), `--optimize-torsion`, `--no-formula-moiety` |
| `slab INPUT` | Generate surface slab models | `-o OUTPUT`, `--miller H K L`, `--layers INT`, `--min-thickness FLOAT`, `--vacuum FLOAT`, `--terminations {single,tasker_preferred,all}` |
| `cluster INPUT` | Carve molecular clusters | `-o OUTPUT`, `--mode {bond_shells,rcut}`, `--seed-element STR`, `--seed-index INT` (repeatable), `--max-atoms INT`, `--rcut FLOAT`, `--freeze-shell {0,1,2}`, `--cap-distance FLOAT`, `--cap-bond-length ELEM=DIST` (repeatable) |
| `supercell INPUT` | Create supercells | `-o OUTPUT`, `--scale A B C` |
| `vacancy INPUT` | Generate vacancy defects | `-o OUTPUT`, `--species SPECIES_ID COUNT` (repeatable), `--seed-index INT`, `--method STR`, `--random-seed INT` |
| `desolvate INPUT` | Remove solvent molecules | `-o OUTPUT`, `--targets STR` (repeatable, required) |
| `interpolate START END` | Interpolate between structures | `-o OUTPUT`, `--method {se3_screw,com_so3,slerp}`, `--n-images INT`, `--include-endpoints/--exclude-endpoints` |
| `reorient INPUT` | Reorient crystal for axis-aligned simulations | `-o OUTPUT`, `--direction H K L`, `--target-axis {x,y,z}` (default: z), `--no-reduce` |

### `mck analyze` — Analyze crystals and print reports

| Command | Description | Key Options |
|---------|-------------|-------------|
| `bfdh INPUT` | Rank low-index facets by BFDH morphology | `--max-index INT`, `--top-n INT`, `--json` |
| `interactions INPUT` | Summarize weak interactions | `--json` |
| `polyhedra INPUT` | Enumerate coordination polyhedra | `--central STR` (required), `--ligand STR` (required), `--level {atom,molecule}`, `--cutoff FLOAT`, `--json` |

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
