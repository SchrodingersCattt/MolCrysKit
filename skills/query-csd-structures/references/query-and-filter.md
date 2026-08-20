# Query and Filter the CSD

Read this when the user needs more than exact-refcode retrieval.

## Query contract

Before executing a search, define:

- CSD version and retrieval date;
- query type and all parameters;
- inclusion/exclusion filters;
- treatment of disorder, errors, polymers, organometallics, and 3D coordinates;
- duplicate/family handling;
- maximum hits or sampling method;
- random seed for sampling.

## Exact identifiers

```python
from ccdc.io import EntryReader

refcodes = ["ABACIR", "ABABUB"]
with EntryReader("CSD") as reader:
    entries = [reader.entry(refcode) for refcode in refcodes]
```

Save the requested identifiers even if some retrievals fail.

## Search APIs

Use the search classes supplied by the installed CCDC release, for example `ccdc.search.TextNumericSearch`, `SubstructureSearch`, or `SimilaritySearch`. Probe the installed API documentation because signatures can vary by CSD release.

Keep the search stage responsible only for producing identifiers and search metadata. Export those identifiers with the bundled script. This avoids mixing query logic, CIF serialization, and MolCrysKit operations.

## Deterministic filtering

Apply filters in a documented order and report counts after each stage. Sort stable refcode lists before export. For random samples, sort the candidate population first, set a seed, and retain the complete candidate list or its checksum.

Do not infer chemical quality from an entry being present in the CSD. Downstream CIF diagnosis remains necessary.
