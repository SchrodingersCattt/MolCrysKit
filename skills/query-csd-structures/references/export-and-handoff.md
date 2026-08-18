# Export and Handoff

Read this before writing CIF files or passing CSD structures to MolCrysKit.

## Bundled exporter

Run from the skill directory:

```bash
python scripts/demo_csd_disorder_workflow.py \
  --refcodes ABACIR ABABUB \
  --output-dir csd_export
```

Or provide one identifier per line:

```bash
python scripts/demo_csd_disorder_workflow.py \
  --refcodes-file refcodes.txt \
  --output-dir csd_export
```

The script writes:

```text
csd_export/
├── cifs/<REFCODE>.cif
└── retrieval_manifest.json
```

It exports only. It does not resolve disorder, add hydrogen, or analyze structures.

## Why not use a built-in CIF writer blindly

Some CCDC writer paths omit `_atom_site_occupancy`, `_atom_site_disorder_group`, or `_atom_site_disorder_assembly`. The bundled exporter reads `crystal.disordered_molecule`, extracts those fields explicitly, and records partial-occupancy statistics.

After export, inspect the CIF and manifest. For a disordered entry, require the expected occupancy/disorder tags and compare atom counts with the CCDC object.

## MolCrysKit handoff

```bash
mck io info csd_export/cifs/ABACIR.cif
mck io molecules csd_export/cifs/ABACIR.cif --json
mck analyze sanity-check csd_export/cifs/ABACIR.cif --json \
  -o csd_export/ABACIR.sanity.json
```

Resolve disorder only through the maintained MolCrysKit CLI/API:

```bash
mck operate disorder csd_export/cifs/ABACIR.cif \
  -o csd_export/ABACIR_ordered.cif --method optimal
```

This keeps retrieval and structural modeling as separate, auditable stages.
