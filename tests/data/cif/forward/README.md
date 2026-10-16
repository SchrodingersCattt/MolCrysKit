# Forward-construction reference fixtures

These four minimal CIFs exercise the Issue #172 forward-construction contract:

| Fixture | Setting | Role | Notes |
| --- | --- | --- | --- |
| `IBPRAC.cif` | `P 21/c` (No. 14) | General-position molecular expansion | Four symmetry-related molecular instances; complete H-containing reference sites. |
| `BENZEN02.cif` | `P b c a` (No. 61) | Ordered inversion-site special position | The local CCDC API returned six carbon coordinates but no H coordinates. The manifest records this limitation; tests must provide an explicit H-placement policy. |
| `HXMTAM10.cif` | `I -4 3 m` (No. 217) | Centered, high-symmetry ordered molecule | The 48 conventional-cell operations retain the body-centering coset. |
| `IQUHUY.cif` | `C 2/m` (No. 12) | Symmetry-related whole-molecule disorder | Four partial sites and both alternatives are retained. CCDC group IDs `0` and `1` are mapped to positive fixture groups `1` and `2` because MolCrysKit reserves group `0` for ordered atoms. |

The accompanying `manifest.json` is the provenance record. It keeps the source
refcode, CSD/API release information, operation/site counts, checksums, and the
raw-to-fixture disorder-group mapping, while omitting local paths, author/name
fields, refinement statistics, and audit records from the CIF files.

The files were generated from a locally licensed CCDC Python API export using
`skills/query-csd-structures/scripts/demo_csd_disorder_workflow.py`, then
rebuilt by `scripts/sanitize_forward_fixtures.py`. They are pinned regression
inputs, not live-download test data. The source CCDC terms and any applicable
redistribution restrictions remain in force.

The repository-level acceptance policy is recorded in `manifest.json` under
`source.redistribution_policy`: these are derived fixtures only, the applicable
CCDC source licence remains authoritative, and CI must use the pinned files
without contacting CCDC or downloading source records.
