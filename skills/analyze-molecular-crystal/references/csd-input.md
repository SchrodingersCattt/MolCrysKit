# Analyze Structures Retrieved from the CSD

Read this when analysis begins from a CSD refcode or a CSD-derived dataset.

## Environment and retrieval

Use a licensed CSD installation and its CCDC Python environment. Verify the active interpreter and database version. A minimal lookup is:

```python
from ccdc.io import EntryReader

with EntryReader("CSD") as reader:
    entry = reader.entry("ABACIR")
    crystal = reader.crystal("ABACIR")
```

Record refcode, CSD version, retrieval date, query criteria, and any entry filtering. Respect licensing restrictions on coordinate redistribution.

## Preserve analysis-relevant metadata

Built-in CCDC CIF writers can omit occupancy and disorder group/assembly fields. For disorder-sensitive analysis, use the repository example:

```text
scripts/demo_csd_disorder_workflow.py
```

Its `export_full_cif_from_csd` implementation preserves disorder metadata. Verify exported CIF tags and compare site counts against `crystal.disordered_molecule`.

## Handoff and diagnosis

```bash
mck io info ABACIR.cif
mck io molecules ABACIR.cif --json
mck analyze sanity-check ABACIR.cif --json -o ABACIR.sanity.json
```

State whether later analysis uses the deposited disordered model, one resolved model, or an ensemble. Database-derived H placement, charge, and disorder should not be assumed correct without diagnosis.

## Dataset reproducibility

For aggregate studies, retain the exact refcode list, search query, CSD version, exclusion rules, failed-entry log, exporter version, MolCrysKit version, and per-entry parameters. Do not treat a changing live database query as a fixed dataset.
