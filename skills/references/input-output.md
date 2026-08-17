# Input, Output, and Format Choice

Read this when choosing an interchange format or handing a structure to another code.

## Supported CLI structure formats

The CLI reads:

- CIF (`.cif`)
- VASP/POSCAR-style files (`.vasp`, `.poscar`, `.contcar`)
- ASE Extended XYZ (`.extxyz`)

Writers select the format from the output extension. Probe `mck io convert --help` in the installed release.

## Format decision

| Need | Preferred format | Reason |
|---|---|---|
| Preserve crystallographic cell and CIF metadata | CIF | Human-readable crystallographic exchange |
| VASP periodic calculation | POSCAR/VASP | Native lattice and Cartesian/fractional structure input |
| MolCrysKit/ASE round-trip | ExtXYZ | Preserves cell, PBC, `molecule_index`, metadata, and multiple frames |
| Isolated molecule or capped QM cluster | XYZ plus JSON sidecar when produced | Widely accepted non-periodic geometry; sidecar preserves provenance/freeze data |
| Dataset or interpolation path | Multi-frame ExtXYZ | One frame per structure with per-frame metadata |

Plain XYZ does not preserve the periodic cell, PBC, molecule membership, disorder metadata, or frame provenance. Do not use it as the only copy of a periodic crystal.

## Conversion

```bash
mck io convert input.cif -o output.extxyz
mck io convert input.extxyz -o POSCAR
mck io extract-molecule input.cif -o molecule.xyz --index 0 --json-sidecar molecule.json
```

Resolve disorder before conversion when a downstream code requires full occupancy:

```bash
mck io convert input.cif -o ordered.extxyz --resolve-disorder
```

Conversion changes representation, not chemistry. It does not validate hydrogen counts, charges, connectivity, surface termination, or clashes.

## Multi-frame ExtXYZ

Use the Python API when all frames are required:

```python
from molcrys_kit.io import read_extxyz, write_extxyz

frames = read_extxyz("dataset.extxyz", index=":")
write_extxyz(frames, "copy.extxyz")
```

The default single-frame read follows ASE convention and may return only the last frame. Pass `index=":"` for a dataset or trajectory bundle.

## Provenance to retain

Record at minimum:

- source file or CSD refcode;
- MolCrysKit version;
- disorder method, count, seed, and coupling assumption;
- bond scale;
- operation parameters;
- frame index for multi-frame output;
- sanity-check report path;
- any user-supplied chemical assumptions, especially charge and protonation.
