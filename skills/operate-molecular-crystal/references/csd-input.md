# Obtain Operation Inputs from the CSD

Read this when a modeling workflow starts from a CSD refcode.

## Requirements

Use a licensed CSD installation and its vendor-managed CCDC Python environment. Install MolCrysKit into that environment or establish an explicit CIF handoff. Verify `ccdc` and MolCrysKit are loaded by the intended interpreter.

## Preserve disorder metadata

CCDC built-in CIF writers can omit `_atom_site_occupancy`, `_atom_site_disorder_group`, and `_atom_site_disorder_assembly`. Losing these fields prevents reliable ordered-model generation.

Use the repository example:

```text
scripts/demo_csd_disorder_workflow.py
```

Its `export_full_cif_from_csd` implementation extracts occupancy and disorder metadata instead of relying on the built-in writer.

Minimal retrieval is:

```python
from ccdc.io import EntryReader

with EntryReader("CSD") as reader:
    entry = reader.entry("ABACIR")
    crystal = reader.crystal("ABACIR")

molecule = crystal.disordered_molecule or crystal.molecule
print(entry.identifier, len(molecule.atoms))
```

Use the full exporter for the actual disorder-sensitive CIF.

## Handoff

```bash
mck io info ABACIR.cif
mck io molecules ABACIR.cif --json
mck analyze sanity-check ABACIR.cif --json -o ABACIR.sanity.json
mck operate disorder ABACIR.cif -o ABACIR_ordered.cif --method optimal
```

Record CSD refcode/version, retrieval date, exporter version, preserved tags, MolCrysKit version, and licensing restrictions on coordinate redistribution.
