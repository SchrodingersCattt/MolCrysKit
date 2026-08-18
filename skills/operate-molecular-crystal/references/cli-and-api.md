# CLI and Python API

Read this before choosing how to execute an operation.

## Default: CLI

Read the repository `docs/cli.md`, then probe the installed command:

```bash
mck operate --help
mck operate slab --help
```

Use the CLI when it exposes the complete operation. It gives stable file-based inputs, explicit options, and reproducible command records.

## Use the API when needed

Use the public Python API when no CLI exists, detailed result objects are required, or several operations must remain in memory. Read `docs/api.md` and the source docstring for the installed version.

```python
from molcrys_kit.io import read_mol_crystal, write_cif
from molcrys_kit.operations import create_supercell

crystal = read_mol_crystal("bulk.cif")
result = create_supercell(crystal, (2, 2, 2))
write_cif(result, "bulk_2x2x2.cif")
```

CLI equivalent:

```bash
mck operate supercell bulk.cif -o bulk_2x2x2.cif --scale 2 2 2
```

## API-only molecule editing

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

Indices are zero-based, translation defaults to Cartesian Å, and rotation angles are degrees. Replacement may try random orientations; the convenience API does not expose a seed, so record that limitation.
