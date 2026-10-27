# Independently reproduced findings

Target package: `9bf0ffb0fb321d1084dbc444250b5f186af3c873`.
Read [triage.md](triage.md) for the repair map, exact code locations and
per-problem refcodes. RT08 corrects the interpretation of unavailable CCDC
chemistry in the original v2 totals; raw rows are retained unchanged.
The following examples generate synthetic structures and require no CSD data
or CCDC licence. Run from a checkout of the pinned package with its normal
dependencies. They supplement the frozen CSD corpus rather than increasing its
1004-case denominator. No production code was changed.

## ASU readback merges distinct complete molecules

Observed: two constructed CH4 molecules, ten atoms after independent pymatgen
expansion, but one MolCrysKit molecule with formula C2H8 after true-ASU readback.
Atomic inventory alone therefore passes while molecular identity fails.
The same molecular-inventory failure occurs for the targeted HXMTAM10 case.

```python
from pathlib import Path
from itertools import product
import numpy as np
from ase import Atoms
from ase.spacegroup import Spacegroup
from pymatgen.io.cif import CifParser
from molcrys_kit.structures.symmetry import CrystalSymmetry, FractionalAffineOperation
from molcrys_kit.operations.forward import (
    MolecularTemplate, MolecularPose, prepare_construction_context,
    construct_molecular_crystal,
)
from molcrys_kit.io.output import write_cif_asu
from molcrys_kit.io import read_mol_crystal

ops = tuple(FractionalAffineOperation(r, t, index=i)
            for i, (r, t) in enumerate(Spacegroup(217).get_symop()))
ctx = prepare_construction_context(
    np.eye(3) * 10,
    CrystalSymmetry(ops, space_group_number=217, space_group_symbol="I -4 3 m"),
)
tetra = np.array([v for v in product((-1, 1), repeat=3) if np.prod(v) == 1])
atoms = Atoms("CH4", positions=np.vstack((np.zeros(3), tetra * .63)))
template = MolecularTemplate(atoms, connectivity=tuple((0, j) for j in range(1, 5)))
crystal = construct_molecular_crystal(ctx, MolecularPose(template, np.zeros(3)))
path = Path("synthetic_ch4_asu.cif")
write_cif_asu(crystal, str(path))
independent = CifParser(str(path)).parse_structures(primitive=False)[0]
reread = read_mol_crystal(str(path))
print(len(crystal.molecules), len(independent),
      [m.get_chemical_formula() for m in reread.molecules])
# Observed: 2 10 ['C2H8']
# Required molecular inventory: ['CH4', 'CH4']
```

This is a molecular-grouping/readback result. It does not assert that the
independent parser lost atomic positions or that the requested space group
was absent from the CIF.

## Enantiopure policy is not applied to general-position improper mates

Observed: the explicit `enantiopure` request below accepts two inversion-related
molecules and copies `R` onto both. The site stabilizer is trivial at this
general position; a site-only compatibility check does not enforce the policy
over the full space group. This confirms the corresponding existing #198 item.

```python
import numpy as np
from itertools import product
from ase import Atoms
from molcrys_kit.structures.symmetry import CrystalSymmetry, FractionalAffineOperation
from molcrys_kit.operations.forward import (
    MolecularTemplate, MolecularPose, prepare_construction_context,
    construct_molecular_crystal,
)

ops = (FractionalAffineOperation(np.eye(3), np.zeros(3)),
       FractionalAffineOperation(-np.eye(3), np.zeros(3)))
ctx = prepare_construction_context(
    np.eye(3) * 20,
    CrystalSymmetry(ops, space_group_number=2, space_group_symbol="P -1"),
    chemistry_policy="enantiopure",
)
tetra = np.array([v for v in product((-1, 1), repeat=3) if np.prod(v) == 1])
atoms = Atoms(["C", "H", "F", "Cl", "Br"],
              positions=np.vstack((np.zeros(3), tetra * .7)))
template = MolecularTemplate(
    atoms, connectivity=tuple((0, j) for j in range(1, 5)),
    stereochemistry=("R", "", "", "", ""),
    chemical_identity="explicit-enantiopure-test",
)
crystal = construct_molecular_crystal(
    ctx, MolecularPose(template, np.array([.21, .29, .37])),
)
print(len(crystal.molecules),
      [str(m.arrays["stereochemistry"][0]) for m in crystal.molecules])
# Observed: 2 ['R', 'R']; the incompatible explicit policy should be rejected.
```

## Capability gates that must remain separate from geometric success

- `MolecularTemplate.connectivity` supplies untyped atom pairs. Known source
  bond orders cannot be carried through this interface as typed bonds; the
  benchmark records this separately from coordinate reconstruction.
- The pinned disorder companion omits full alternative molecular geometry
  and connectivity, and no lossless inverse reader is provided. A readable
  JSON dictionary is not a reconstructed statistical model.
- The statistical writer produces a materialized P1 average; requesting a
  true-ASU statistical representation is unavailable on this target.
- CCDC group weights and atom occupancies can differ, including for H atoms.
  The benchmark preserves both source fields. Differences involving a
  selected conformer, source-model ambiguity or the export adapter are not
  automatically attributed to a constructor defect.

## Explicit deuterium is lost by the CIF round trip

For the frozen `AMEGEF03` entry, the corrected adapter converts CCDC's explicit
`D` symbols to hydrogen with isotope 2 and its isotope mass. In-memory atomic
geometry and both P1/ASU geometric round trips pass. Both independent CCDC
chemical readbacks report the known isotope annotation as absent. The narrow
geometric score therefore passes while complete source-record preservation
does not. This is recorded as `unsupported` chemical identity rather than an
ASE `D`-symbol adapter error.

The target's source-CIF MolCrysKit reader also raises `KeyError` for this input
during the optional source sanity diagnostic; that is retained separately in
`diagnostics.source_sanity_error`. It does not suppress completed geometry,
symmetry or isotope-loss evidence.

The generated `report.md`, `summary.json` and `results.jsonl` contain the complete
corpus evidence and stage denominators. Overall pass requires all applicable
gates. A narrower geometric score is labeled explicitly.

## Positive geometric controls

The following entries satisfy the complete ordered geometric gate set, including molecular grouping in both P1 and ASU readback. Chemical annotation outcomes remain separate.

| Refcode | Source stratum | Cell molecules | Atoms | P1 max error / Å | ASU max error / Å |
|---|---|---:|---:|---:|---:|
| AGANUU | single_component | 2 | 160 | 1.28e-14 | 1.28e-14 |
| ABEDUH | multi_component | 20 | 280 | 7.16e-15 | 7.16e-15 |
| ANOSUS | multi_ion | 8 | 68 | 3.69e-15 | 3.79e-15 |
| ACIPIN | special_position | 1 | 114 | 4.44e-15 | 4.44e-15 |
