# I/O and API-only Edits

## Convert or extract

```bash
mck io convert input.cif -o output.extxyz
mck io convert input.extxyz -o POSCAR
mck io extract-molecule input.cif -o molecule.xyz \
  --index 0 --json-sidecar molecule.json
```

Use CIF for crystallographic exchange, POSCAR/VASP for VASP, ExtXYZ for
round-trips and multi-frame paths, and XYZ only for an isolated molecule or
cluster. Plain XYZ loses periodic metadata. Conversion changes representation,
not protonation, connectivity, disorder, or clashes.

## Translate, rotate, or replace one molecule

These edits are public-API-only:

```python
import numpy as np

from molcrys_kit.io import read_mol_crystal, write_cif
from molcrys_kit.operations import replace_molecule, rotate_molecule, translate_molecule

crystal = read_mol_crystal("bulk.cif")
crystal = translate_molecule(crystal, 0, np.array([0.2, 0.0, 0.0]))
crystal = rotate_molecule(
    crystal,
    0,
    np.array([0.0, 0.0, 1.0]),
    15.0,
    center="com",
)
crystal = replace_molecule(crystal, 1, "replacement.xyz", clash_threshold=1.2)
write_cif(crystal, "edited.cif")
```

Indices are zero-based, translations default to Cartesian angstrom, and rotation
angles are degrees. Replacement may try random orientations without exposing a
seed; record that limitation.

For multi-frame ExtXYZ, read every frame explicitly. A default ASE-style read may
return only the last frame. Preserve source hashes, frame/replica IDs, bond scale,
operation parameters, charge/protonation assumptions, and generated sidecars.
