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

Synthetic unit tests use independently constructed XML. They prove parser
contracts only; they are not Mercury exports.

**Real-format status:** tier **(A)** has been run on pinned public Mercury
exports (see below): the reader ingests authentic `.mryx` bytes and the stored
geometry survives cross-format export. That is **not** the same as tier **(B)**
(CIF → Mercury save → this reader) and does **not** certify every Mercury
version, multi-structure bundles, disorder exports, or session fidelity.
Compare against a reliable independent structure reference only when one exists
for the **same** stored sites; do not treat an unrelated CIF from the same
dataset as a paired reference.

## Public-format validation (tier A)

`tests/data/mryx/public_manifest.json` pins traceable Mercury exports that can
be downloaded under stated licenses (currently one CC-BY-4.0 Zenodo deposit from
`10.1021/acs.joc.4c02538`). Run:

```bash
python scripts/validate_mryx_public_roundtrip.py
pytest tests/integration/test_mryx_public_roundtrip.py
```

Offline manifest contracts (no network): `pytest tests/unit/io/test_mryx_public_manifest.py`

Tier **(A)** checks: real `.mryx` → `read_mryx` / `read_mryx_atoms` (stored
sites, no symmetry expansion) → `write_extxyz` → ASE ExtXYZ read, comparing
element identities, cell parameters, and Cartesian coordinates within
`MRYX_COORDINATE_TOLERANCE_A`. This validates cross-format geometry transfer,
not a second independent interpretation of the Mercury XML tree.

Tier **(B)** (public CIF → Mercury save → this reader) is **not** run in CI:
Mercury needs a desktop install and CSD-Community activation. Coordinators with
Mercury: install [Free Mercury](https://www.ccdc.cam.ac.uk/support-and-resources/downloads/),
activate **CSD-Community**, open a public CIF, save `.mryx`, and compare cell,
site count, labels, and structure index with `read_mryx_atoms` or `mck io convert`.

**Current manifest coverage:** single-structure triclinic periodic crystal; not
multi-structure, disordered, or version-tagged. ACS/RSC supplementary `.mryx`
bundles named in the literature are listed under `investigated_but_blocked` in
the manifest (publisher bot protection from this environment).
