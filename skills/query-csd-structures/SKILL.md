---
name: query-csd-structures
description: 'Query and export crystal structures from the Cambridge Structural Database with the CCDC Python API. Use when retrieving CSD refcodes, building reproducible CSD searches and filters, exporting CIF files while preserving occupancy and disorder metadata, creating retrieval manifests, or handing CSD structures to MolCrysKit. Requires a licensed CSD installation.'
---

# Query CSD Structures

Retrieve licensed CSD data reproducibly and export analysis-ready CIF files without silently losing disorder metadata.

## Core workflow

1. Read [CSD environment](./references/csd-environment.md) and verify the licensed CCDC Python interpreter.
2. Read [query and filter](./references/query-and-filter.md). Record the CSD version, exact query, filters, refcodes, and retrieval date.
3. For exact refcodes, use the bundled [export script](./scripts/demo_csd_disorder_workflow.py) as described in [export and handoff](./references/export-and-handoff.md).
4. Verify exported atom counts and CIF occupancy/disorder tags before using the structures.
5. Read [provenance and licensing](./references/provenance-and-licensing.md) before sharing files or reporting a dataset.

## Exact-refcode export

```bash
python scripts/demo_csd_disorder_workflow.py \
  --refcodes ABACIR ABABUB \
  --output-dir csd_export
```

When running from another directory, use the path to this skill's bundled script. The script lazily imports `ccdc`, exports one metadata-preserving CIF per refcode, and writes `retrieval_manifest.json` with successes and failures.

## Search before export

Use the CCDC Python API for text, substructure, similarity, or numeric searches. Keep search logic separate from export: save the resulting identifiers, then pass the stable refcode list to the bundled exporter.

Do not embed credentials or licensed coordinates in the skill. Do not substitute an unlicensed web scrape for the CSD API.

## Handoff to MolCrysKit

After export, diagnose before operating or analyzing:

```bash
mck io info csd_export/cifs/ABACIR.cif
mck io molecules csd_export/cifs/ABACIR.cif --json
mck analyze sanity-check csd_export/cifs/ABACIR.cif --json \
  -o csd_export/ABACIR.sanity.json
```

If disorder must be resolved, use the production MolCrysKit CLI rather than implementing a second solver in the retrieval script:

```bash
mck operate disorder csd_export/cifs/ABACIR.cif \
  -o csd_export/ABACIR_ordered.cif --method optimal
```

## Non-negotiable rules

- Use the licensed CCDC Python API and record its CSD version.
- Preserve `_atom_site_occupancy`, disorder group, and disorder assembly fields when present.
- Treat retrieval/export separately from disorder resolution and scientific analysis.
- Keep a manifest of every requested refcode, success, failure, and output path.
- Respect CSD redistribution and publication licensing requirements.
