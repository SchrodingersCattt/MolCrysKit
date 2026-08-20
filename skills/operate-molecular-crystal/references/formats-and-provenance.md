# Formats and Provenance for Operations

Read this before selecting an output or handing a generated model to another code.

## Format choice

| Need | Preferred format |
|---|---|
| Crystallographic exchange and periodic metadata | CIF |
| VASP periodic input | POSCAR/VASP |
| MolCrysKit/ASE round-trip or multi-frame path | ExtXYZ |
| Isolated molecule or QM cluster | XYZ plus generated JSON sidecar |

The CLI reads CIF, VASP/POSCAR-style files, and ExtXYZ. Writers use the output extension. Probe `mck io convert --help` in the installed release.

Plain XYZ loses the periodic cell, PBC, molecule membership, disorder metadata, and frame provenance. Never use it as the only retained copy of a periodic crystal.

## Conversion

```bash
mck io convert input.cif -o output.extxyz
mck io convert input.extxyz -o POSCAR
mck io extract-molecule input.cif -o molecule.xyz \
  --index 0 --json-sidecar molecule.json
```

Conversion changes representation, not chemistry. It does not resolve protonation, repair connectivity, choose a surface termination, or remove clashes.

## Multi-frame output

Use ExtXYZ for disorder ensembles and interpolation paths. With the Python API, request all frames explicitly:

```python
from molcrys_kit.io import read_extxyz

frames = read_extxyz("path.extxyz", index=":")
```

A default ASE-style read may return only the last frame.

## Minimum provenance

Retain:

- source path, checksum, or CSD refcode;
- MolCrysKit version;
- bond scale;
- disorder method, count, seed, and coupling assumption;
- complete operation parameters;
- charge and protonation assumptions;
- frame/replica index;
- generated cluster sidecars;
- sanity-report path;
- requested versus effective result when fallback occurred.
