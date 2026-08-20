# Provenance and Licensing

Read this before delivering CSD-derived structures or statistics.

## Retrieval manifest

Retain at minimum:

- CSD version;
- retrieval timestamp;
- CCDC Python version/interpreter;
- exact query and filters, or exact requested refcodes;
- stable ordering and random seed if sampled;
- exporter version or source commit;
- per-refcode status, output path, atom count, partial-occupancy count, and error;
- downstream MolCrysKit version and parameters.

Do not drop failures from the denominator of a dataset report.

## Licensing

CSD access and coordinate redistribution are governed by the applicable CCDC licence. Before committing or sharing exported CIF files:

- verify whether coordinates may be redistributed;
- prefer sharing refcodes, queries, scripts, and manifests when coordinates are restricted;
- do not include licensed structures in public tests;
- sanitize logs and error reports that contain restricted data.

## Citation and reporting

Cite the CSD and CCDC software according to the installed release guidance. Report the database version and retrieval date because search results can change across releases.
