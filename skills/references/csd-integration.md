# Integrate with the Cambridge Structural Database

Read this when the input starts from a CSD refcode rather than an existing CIF.

## Requirements

- a licensed CSD installation;
- the CCDC Python API in its vendor-managed environment;
- MolCrysKit installed into that same environment, or a deliberate export/import boundary between environments.

The CCDC Python API is usually unavailable from an ordinary Python environment. Verify before use:

```python
import sys
import ccdc
print(sys.executable)
print(ccdc.__file__)
```

Follow the local CSD installation instructions for environment activation. Do not copy machine-specific activation paths into a portable workflow.

## Critical disorder warning

CCDC built-in CIF writers can omit `_atom_site_occupancy`, `_atom_site_disorder_group`, and `_atom_site_disorder_assembly`. That loss makes later disorder resolution unreliable.

For disorder-sensitive work, do not assume that a CIF exported successfully is complete. Explicitly verify those CIF tags and compare the exported site count with `crystal.disordered_molecule`.

MolCrysKit includes a worked exporter and end-to-end example in:

```text
scripts/demo_csd_disorder_workflow.py
```

Its `export_full_cif_from_csd` function extracts occupancy and disorder metadata and writes a complete CIF instead of relying on the built-in writer.

## Minimal retrieval pattern

```python
from ccdc.io import EntryReader

refcode = "ABACIR"
with EntryReader("CSD") as reader:
    entry = reader.entry(refcode)
    crystal = reader.crystal(refcode)

molecule = crystal.disordered_molecule or crystal.molecule
print(entry.identifier, len(molecule.atoms))
```

Use the repository exporter for the actual CIF when disorder provenance matters.

## Handoff to MolCrysKit

After export:

```bash
mck io info ABACIR.cif
mck io molecules ABACIR.cif --json
mck analyze sanity-check ABACIR.cif --json -o ABACIR.sanity.json
```

If disorder is present:

```bash
mck operate disorder ABACIR.cif -o ABACIR_ordered.cif --method optimal
```

For an ensemble, use `random` with an explicit seed or bounded `enumerate`, then validate every replica.

## Provenance

Retain:

- CSD refcode and database version;
- retrieval date;
- whether `entry.crystal`, `crystal.molecule`, or `crystal.disordered_molecule` was used;
- exporter implementation/version;
- preserved CIF disorder tags;
- MolCrysKit version and downstream parameters;
- CSD licensing restrictions on redistributed coordinates.
