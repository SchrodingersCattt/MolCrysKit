# Mercury MRYX structural import

`molcrys_kit.io.read_mryx_atoms(path, index=0)` reads gzip-compressed Mercury
CCDC_XML (or its plain XML representation) and returns **stored 3-D sites** as
ASE `Atoms`. `read_mryx` returns the normal `MolecularCrystal`, so existing CIF,
POSCAR, XYZ and ExtXYZ writers can export the stored-site geometry. This is a
read-only, conservative subset reader, not a Mercury session converter.

Both functions select the first structure by default, an integer index selects
another structure (negative indices count from the end), and `index=":"` returns
a list in document order. Every structure is validated. A single index then
keeps ASE atoms and raw provenance for that structure only. `index=":"` is
eager: every structure is materialized before the list is returned, and the
result is not a streaming iterator. `read_mryx` uses the same selection.
Missing or unsupported structural layouts, nonstandard atom labels, invalid
coordinate matrices/vectors, bad bond endpoints and unresolved disorder
references fail with `ValueError`. Missing files retain `FileNotFoundError`.

```python
from molcrys_kit.io import read_mryx_atoms, read_mryx, write_extxyz

frames = read_mryx_atoms("document.mryx", index=":")
crystal = read_mryx("document.mryx", index=-1)
write_extxyz(crystal, "stored-sites.extxyz")
```

## Geometry and semantics

- Cell lengths are Å; XML cell angles are **radians**, converted to degrees for
  conventional-cell construction. Explicit fractional-to-orthogonal matrices
  preserve the original Cartesian orientation. Orthogonal coordinates require
  that matrix; fractional-only input can use a conventional cell. When both
  coordinate representations exist they must agree within 0.0001 Å.
- **No automatic symmetry expansion.** The imported periodic lattice contains
  exactly the stored sites, possibly only an asymmetric unit. It is not a claim
  of full-cell composition or a complete crystal packing. Space-group fields and
  symmetry operators are preserved in raw cell metadata, not applied.
- Elements are inferred from conventional case-sensitive element-prefixed atom
  labels (for example `C12` or `Cl3`). Internal `element/code` values are **not
  atomic numbers**. Conflicting label/code associations fail; custom labels that
  cannot be resolved conservatively are unsupported. Labels use the existing
  per-atom `label` array.
- Occupancies are **unspecified**. `Site.uncertain_value` is retained raw, never
  asserted to mean occupancy. Downstream unit-occupancy defaults are placeholders,
  not measurements; do not use this import to infer occupancy-weighted formulas.
- Stored bonds/types, disorder assemblies, site fields and ADPs are preserved as
  raw JSON-compatible provenance in `atoms.info["mryx"]` (or
  `crystal.metadata["mryx"]`). They do not establish bond orders or resolved
  disorder. `read_mryx` identifies molecules/bonds through the existing geometric
  pipeline instead. Atom order may change during molecule grouping; raw bond
  indices always refer to the original stored-site order in the raw metadata.
- Each read emits `UserWarning` describing these limitations. The same diagnostics
  are available in the `mryx` metadata, together with structure index and label.
- Scene styling, cameras, selections and other Mercury UI state are ignored.
  There is no MRYX writer. Standard-format export is not a session round trip;
  in particular, P1 geometry output does not reconstruct missing symmetry mates.
  ExtXYZ retains raw metadata; other writers may retain only geometry.

## Safety and dependencies

`read_mryx` and `read_mryx_atoms` import ASE and `defusedxml`. Both come from
the `desktop` extra (`pip install "molcrys-kit[desktop]"`). The core install
is NumPy, NetworkX, and Click. Importing `molcrys_kit.io` does not import ASE,
`defusedxml`, or the Mercury module; the first call to either reader does.
No CCDC installation is needed. No file references are opened, no expressions
are evaluated and no archive paths are extracted. Entities and external XML
references are forbidden; the plain `<!DOCTYPE CCDC_XML>` declaration is
accepted.

Input is capped at 16 MiB and expanded XML at 64 MiB, with depth 128 and
500,000 nodes. Atom labels are limited to 128 characters so one long label
cannot set the width of the NumPy label array. A document may contain at most
4,096 structures, and each structure at most 100,000 stored sites. Malformed
or truncated gzip/XML input raises `ValueError`. Graphics items are visited
one at a time. Finished markup that is not the graphics item being validated,
including ignored scene or camera markup, is discarded during the walk. A
single index copies only the selected structure into ASE atoms and raw
provenance; each other item is validated and discarded before the next item
is kept. `index=":"` still builds the full list before returning. The bounded
document bytes remain available for that one call.

These checks use independently constructed documents. They do not certify
every Mercury version, and this package does not ship Mercury exports.
Real-format compatibility is therefore unverified. A local comparison that
would close that gap is one Mercury-generated file, or a set of them, checked
against a reliable export of the same structures: a single stored structure,
several stored structures, one disordered structure, and one non-orthogonal
cell. Compare atom count, cell lengths and angles, Cartesian or fractional
coordinates, atom labels, and the raw metadata that this reader keeps. Record
the Mercury version. Do not treat the synthetic fixtures as that comparison.
