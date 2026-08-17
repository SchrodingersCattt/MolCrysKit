# CLI Reference

Read this for command selection. Probe the installed command with `--help` before relying on an option because the installed release is authoritative.

## Command tree

```text
mck
├── io
│   ├── info INPUT
│   ├── molecules INPUT
│   ├── extract-molecule INPUT
│   └── convert INPUT
├── operate
│   ├── disorder INPUT
│   ├── add-h INPUT
│   ├── slab INPUT
│   ├── cluster INPUT
│   ├── supercell INPUT
│   ├── vacancy INPUT
│   ├── desolvate INPUT
│   ├── interpolate START END
│   └── reorient INPUT
└── analyze
    ├── bfdh INPUT
    ├── interactions INPUT
    ├── polyhedra INPUT
    └── sanity-check INPUT
```

Global options include `--verbose`, `--quiet`, `--version`, and `--help`.

## Diagnose and convert

| Command | Main options |
|---|---|
| `mck io info INPUT` | `--resolve-disorder`, `--bond-scale FLOAT` |
| `mck io molecules INPUT` | `--json`, `--resolve-disorder`, `--bond-scale FLOAT` |
| `mck io extract-molecule INPUT` | `-o OUTPUT`; one of `--index`, `--formula`, `--species-id`, `--largest`, `--all`; optional `--center-vacuum`, `--pbc`, `--json-sidecar` |
| `mck io convert INPUT` | `-o OUTPUT`, `--resolve-disorder`, `--bond-scale FLOAT` |

Selectors for `extract-molecule` are mutually exclusive. Molecule indices are zero-based.

## Operate

| Command | Main options and constraints |
|---|---|
| `disorder` | `-o`; `--method optimal|random|enumerate`; `--count`; `--seed`; `--coupled` |
| `add-h` | `-o`; repeatable `--target-elements`; repeatable `--rule SYMBOL:key=value,...`; `--optimize-torsion`; `--no-formula-moiety`; `--bond-scale` |
| `slab` | `-o`; required `--miller H K L`; one of `--layers` or `--min-thickness`; `--vacuum`; `--terminations single|tasker_preferred|all|INDEX` |
| `cluster` | `-o`; `--mode bond_shells|rcut`; exactly one seed mechanism, `--seed-element` or repeatable `--seed-index`; `--max-atoms`; `--cut-cc-bonds`; `--rcut`; `--freeze-shell 0|1|2`; cap and provenance options |
| `supercell` | `-o`; required `--scale A B C` with positive integers |
| `vacancy` | `-o`; repeatable `--species SPECIES_ID COUNT`; `--seed-index`; `--method`; `--random-seed` |
| `desolvate` | `-o`; repeatable required `--targets SPECIES_ID` |
| `interpolate` | `START END -o`; `--method se3_screw|com_so3|slerp`; `--n-images`; endpoint toggle |
| `reorient` | `-o`; required `--direction H K L`; `--target-axis x|y|z`; `--no-reduce` |

Examples:

```bash
mck operate disorder input.cif -o ordered.cif --method optimal
mck operate add-h ordered.cif -o hydrogenated.cif --target-elements N --target-elements O
mck operate slab bulk.cif -o slab.cif --miller 0 0 1 --layers 5 --vacuum 15
mck operate cluster bulk.cif -o qm_cluster.xyz --seed-index 42 --mode rcut --rcut 12
mck operate supercell bulk.cif -o supercell.cif --scale 2 2 2
mck operate reorient bulk.cif -o aligned.cif --direction 1 1 0 --target-axis z
```

`--cut-cc-bonds` uses parent atom-index pairs such as `12,13;27,28`.

## Analyze

| Command | Main options |
|---|---|
| `bfdh` | `--max-index`, `--top-n`, `--json` |
| `interactions` | `--json` |
| `polyhedra` | required `--central` and `--ligand`; `--level atom|molecule`; `--cutoff`; `--json` |
| `sanity-check` | `--checks`; clash thresholds; `--ignore-hh/--no-ignore-hh`; bond factors; `--isolated-elements`; `-o`; `--json` |

```bash
mck analyze bfdh bulk.cif --max-index 2 --top-n 10
mck analyze interactions bulk.cif --json
mck analyze polyhedra bulk.cif --central Zn --ligand O --level atom --json
mck analyze sanity-check output.cif --json -o output.sanity.json
```
