# CSD round-trip benchmark: interpretation

The frozen corpus has **1004/1004 executed entries**: 1000 stratified entries plus four targeted references.
The package tree is `9bf0ffb0fb321d1084dbc444250b5f186af3c873`; benchmark-only commits change no production code.

**Follow-up:** read [triage.md](triage.md) before interpreting chemical mismatch counts. CCDC unavailable-assignment sentinels were compared as known labels in v2. The supplementary audit separates confirmed defects, missing evidence and benchmark limitations. Frozen raw rows and geometric results are unchanged.

## Provenance and primary results

- CSD 6.01, active February/May/September 2026 updates; 1,469,017 entries in the source pool.
- Licensed CCDC API 3.7.2 in its own Python environment; environment versions and active pool are recorded in `environment.json`.
- Seed 42; each of the five frozen strata contains 200 entries. Family suffixes remain distinct exact refcodes.
- Historical v2 strict-evaluator output: **0/1004** (read RT08/RT09 before interpreting it as chemical equivalence).
- Ordered in-memory geometric reconstruction: **712/803**.
- All required ordered geometric round-trip gates satisfied: **566/803**.
- Disorder raw marginal-site inventory match in the documented selected-state seed path: **22/201**.

The strict score includes known chemical annotations and lossless whole-molecule disorder restoration. The geometric score
checks the original cell, origin, molecular inventory, graph connectivity and coordinates with coherent molecular lattice images,
plus both P1 and ASU readback. It does not claim typed chemical-annotation preservation.

| Stratum | Entries | Strict pass | Ordered memory geometry | All ordered geometry gates | Disorder memory average |
|---|---:|---:|---:|---:|---:|
| disordered | 200 | 0 | 0 | 0 | 22 |
| special_position | 200 | 0 | 164 | 91 | 0 |
| multi_ion | 200 | 0 | 178 | 160 | 0 |
| multi_component | 200 | 0 | 186 | 153 | 0 |
| single_component | 200 | 0 | 181 | 160 | 0 |
| targeted | 4 | 0 | 3 | 2 | 0 |
| all | 1004 | 0 | 712 | 566 | 22 |

## Interpretation of non-passes

1. Known source bond orders cannot be supplied as typed connectivity through this template interface. The missing
   annotation gate remains separate from successful geometry; CIF output may also assert changed chemical annotations.
2. The synthetic I-43m methane example independently demonstrates ASU molecular grouping loss: two CH4 molecules
   become one C2H8 object in MolCrysKit, while pymatgen retains the correct ten-atom inventory. See `findings.md`.
3. The explicit enantiopure general-position P-1 example is accepted and copies R labels onto inversion mates; this
   confirms the corresponding #198 concern without using any CSD coordinates.
4. Statistical-average P1 data are insufficient to restore whole-molecule alternatives. The pinned target has no lossless
   companion reader and no verified true-ASU statistical path. Full disorder round trips therefore remain unestablished.
5. Disorder marginal mismatches in the selected-state seed experiment can involve source occupancy/group-weight
   differences, split/merged site representations and non-symmetry-related alternatives. These are not all attributed to constructor defects. Source
   correlations absent from CIF remain unknown; no physical joint distribution is invented.
6. Parser errors and 300-second evaluation deadlines remain in the denominator. Stage counts include not-reached cases
   explicitly in `analysis.json`; an absent stage is not a demonstrated structural mismatch.

Late evaluation timeouts with all geometric gates already recorded: **3**. These remain overall timeouts;
the geometric synthesis counts their completed checks. The worker-completion boolean in the primary CSV can be more conservative.

## Coverage, completeness and integrity

Source completeness flags: `{"complete": 851, "missing_coordinates": 153}`.
Completeness means coordinate presence for API-listed atoms, not an independently established protonation/valence model.
Water, solvent-formula and neutral-multispecies tags are coverage candidates, not verified crystallization roles.
All 1004 private source hashes and CIF byte hashes were verified; all 803 ordered
packing references were checked for repeated whole-molecule lattice images and none were found.
The public JSON projection was checked for every case. Coordinates, CIFs, templates, mappings, raw logs and companions remain ignored/local.

## Validation

- Full tests: 2441 passed, 6 skipped.
- Focused forward/CIF/benchmark tests: 61 passed; 21 new synthetic benchmark tests.
- Changed-file Ruff, format check and compile checks passed. No licensed retrieval runs in ordinary CI.
- Windows fixture checksum validation used exact pinned Git-blob bytes to avoid checkout CRLF conversion.

The fixed-quota totals describe this corpus. They are not a CSD-population success-rate estimate or evidence of stable packing energies.
