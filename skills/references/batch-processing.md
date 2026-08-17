# Batch Processing and Python Workflows

Read this when processing many structures, retaining per-frame provenance, or using capabilities without a CLI command.

## General pattern

1. create a manifest with stable input IDs and intended parameters;
2. process one structure per record;
3. never overwrite source CIFs;
4. catch and record failures per record rather than dropping them;
5. validate every output;
6. store parameters and MolCrysKit version with the result.

## Python batch skeleton

```python
from pathlib import Path

from molcrys_kit.io import read_mol_crystal, write_cif
from molcrys_kit.operations import create_supercell
from molcrys_kit.analysis import sanity_check

input_dir = Path("inputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

for source in sorted(input_dir.glob("*.cif")):
    crystal = read_mol_crystal(source)
    result = create_supercell(crystal, (2, 2, 2))
    report = sanity_check(result)
    if not report.passed:
        raise RuntimeError(f"Sanity check failed for {source.name}: {report}")
    write_cif(result, output_dir / f"{source.stem}__2x2x2.cif")
```

Inspect the installed API before assuming the exact serialization method of report objects.

## Dataset bundles

```python
from molcrys_kit.io import read_mol_crystal, read_extxyz, write_extxyz

crystals = [read_mol_crystal(path) for path in ["A.cif", "B.cif"]]
metadata = [
    {"source_id": "A", "frame_index": 0},
    {"source_id": "B", "frame_index": 1},
]
write_extxyz(crystals, "dataset.extxyz", info=metadata)

frames = read_extxyz("dataset.extxyz", index=":")
```

Use `index=":"` for all frames. Confirm that atom order, cell, PBC, and `molecule_index` survive the round trip.

## CLI automation

For a small homogeneous batch, call `mck` once per input and preserve stderr/stdout in logs. Quote paths and use unique output names. Probe live help first; do not build automation around undocumented options.

Prefer the Python API when:

- the workflow passes in-memory crystals between several operations;
- detailed result objects are needed;
- molecule manipulation or API-only analyses are required;
- per-frame metadata must be retained;
- failures need structured handling.

## Reproducibility

For stochastic disorder, vacancy, replacement, or perturbation procedures, set and record random seeds. A seed alone is insufficient: also record package versions, input checksum, parameters, and output ordering.
