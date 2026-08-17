# Verify Generated Structures

Read this before accepting any operation output.

## Required machine checks

```bash
mck analyze sanity-check output.cif --json -o output.sanity.json
mck io info output.cif
mck io molecules output.cif --json
```

The complete suite checks hard clashes, intermolecular clashes, isolated atoms, hydrogen presence, topology preservation, formula consistency, and bond distances. Do not disable a failure merely to produce a passing report.

## Operation-specific invariants

| Operation | Verify |
|---|---|
| Disorder | expected replicas; required full occupancy; no incompatible alternatives; method, seed, coupling recorded |
| Add H | intended H count/protonation; X–H distances; no new clashes; interaction network inspected |
| Slab | Miller orientation; intact molecules; thickness/vacuum; termination; both surfaces; charge and dipole suitability |
| QM cluster | seed retained; intended shells/cutoff; cap placement; no unintended metal caps; sidecar delivered |
| Supercell | vectors and counts scale by `A×B×C` |
| Vacancy/desolvation | only selected species removed; resulting composition and charge understood |
| Reorientation | requested direction aligns with target axis; handedness and PBC preserved |
| Interpolation | endpoints, atom order/count, molecule mapping, and all intermediate geometries are valid |
| Molecule manipulation | only selected molecule changed; alignment and clash threshold satisfied |
| Conversion | cell, PBC, coordinates, species, and membership survive where supported |

## Human inspection

Visualize input and output whenever geometry changes. Inspect periodic boundaries, whole molecules, labels, caps, terminations, vacancies, and close contacts. A zero exit code does not establish chemical correctness.

## Delivery record

Report exact input/output paths, software version, parameters, assumptions, warnings, and verification result. If an operation produced multiple candidates, preserve their identities rather than presenting one as uniquely correct.
