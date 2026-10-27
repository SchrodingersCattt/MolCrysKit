# Actionable round-trip triage

This is the developer entry point for #198 / PR #199. Product source links are pinned to 9bf0ffb; benchmark links to the frozen v2 producer. No production code or frozen results were changed.

## Files

- `../../refcodes.txt`: all 1004 exact IDs, one per line.
- `../../refcodes.csv`: the same IDs with frozen stratum/targeted flags.
- `failure_catalogue.csv`: every non-pass stage, repair ID, confidence, expected/observed record, code entry point and next action.
- `problem_refcodes/*.confirmed.txt` and `*.related.txt`: confirmed associations versus candidates; synthetic-only defects need no CSD refcode.
- `reproduce_confirmed.py` / `synthetic_diagnostics.json`: executable licence-free probes and measured outcomes.
- `identity_audit.json`: per-field local-cache audit; contains counts/transitions only.

## Interpretation correction

The 920 original `mismatch` outcomes are historical pipeline outputs, not 920 confirmed product defects. CCDC returns `Error` when chirality cannot be assigned and `Unknown` for unassigned bond types. The v2 adapter compared these as if they were known labels. In a synthetic C2 CIF, all tested bond-tag/token variants import as Unknown/Error in CCDC 3.7.2; merely renaming a tag is not a demonstrated repair.

Of 108 raw in-memory identity flags, the audit confirms **93 CSD cases / 450 known R/S conflicts, all on improper operations**. NUYVOU instead has 4 input/API-availability differences: a direct original-template match exists, so this is not evidence of a failed symmetry-mate lookup. The remaining 14 cases show no valid R/S contradiction after separating vendor unavailable/mixed statuses. Geometry results and the frozen raw rows are unchanged; chemical-cause claims must use this triage.

## Repair map

| ID | Owner | Priority | Problem | Confirmed refs | Related refs |
|---|---|---|---|---:|---:|
| RT01 | product | P1 | ASU bond expansion creates cross-molecule edges | 1 | 156 |
| RT02 | product | P1 | Whole-group chirality policy and improper-mate labels | 93 | 93 |
| RT03 | product | P1 | Typed bonds are stripped and CIF output hardcodes single | 712 | 712 |
| RT04 | product | P1 | Isotope and explicit charge metadata disappear on writing | 1 | 1 |
| RT05 | product | P2 | D/T aliases are passed to ASE as element names | 1 | 1 |
| RT06 | product | P1 | Statistical companion is not a lossless molecular model | 190 | 190 |
| RT07 | product | P1 | Statistical true-ASU dispatch enters ordered writer | 190 | 190 |
| RT08 | benchmark | P1 | Benchmark treats unavailable CCDC chemistry as known labels | 708 | 708 |
| RT09 | benchmark | P1 | Disorder seed/oracle is a limited benchmark probe | 0 | 175 |
| RT10 | product | P1 | Zero-probability alternatives can be selected as occupied | 0 | 0 |
| RT11 | product | P1 | Stabilizer matching uses independent atom image shifts | 0 | 0 |
| RT12 | product | P1 | Declared group is not checked against the complete operation set | 0 | 0 |

A related refcode is a reproducible queue item, not proof that the same root cause explains every case. RT08 and RT09 are benchmark issues; do not change production chemistry to make those probes pass.

## Reproduce

From the repository root, without CCDC:

```powershell
python -c "import runpy; runpy.run_path('benchmarks/csd_roundtrip/reports/csd601-sep2026-baseline-v2/reproduce_confirmed.py', run_name='__main__')"
```

Run the same script with `--ccdc-reader` in the licensed vendor interpreter for the optional synthetic reader probe. During repair, these probes can run on changed production code. The frozen 1004-case evaluator intentionally refuses a different target package; create a new versioned experiment for future full-corpus comparisons.

## RT01 — ASU bond expansion creates cross-molecule edges

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/io/cif.py:2293 — _published_bond_pairs fallback](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L2293)
- [molcrys_kit/io/cif.py:2303 — independent ASU endpoint candidate loops](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L2303)
- [molcrys_kit/io/cif.py:2475 — published bonds enter component graph](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L2475)

**Expected:** Two CH4 components and 8 intramolecular C-H edges

**Observed / mechanism:** Special-site dedup removes identity-image representatives. Fallback at 2296/2299 selects ALL mates; the independent endpoint Cartesian product at 2303/2304 creates cross-molecule edges. The symmetry-coded branch bypasses distance checking. CH4 has 16 edges (8 valid plus 8 spurious 7.569062-A edges). HXMTAM10 has 384 explicit pairs, 336 longer than every published bond; a diagnostic removal restores [22,22] components.

**Repair direction:** Retain original ASU source coordinates/op-to-retained-atom mapping; transform both endpoints with the same group operation; preserve directed lattice images. Do not select every mate when identity images were deduplicated.

**Regression acceptance:**

- Synthetic I-43m CH4: 2 molecules, 10 atoms, exactly 8 C-H edges; no cross-centre edges.
- HXMTAM10: 2 components of 22 atoms; keep legitimate special-position and boundary-crossing bonds.
- Do not use distance filtering as the general fix; published endpoint images remain authoritative.

**Refcodes:** HXMTAM10 Full lists: `problem_refcodes/RT01.confirmed.txt` and `.related.txt`.

## RT02 — Whole-group chirality policy and improper-mate labels

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/operations/forward.py:614 — analyze_molecular_pose only iterates the site subgroup](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L614)
- [molcrys_kit/operations/forward.py:799 — constructor calls site-only compatibility](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L799)
- [molcrys_kit/operations/forward.py:704 — _copy_generated_atoms copies stereo arrays unchanged](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L704)

**Expected:** Reject incompatible explicit enantiopure requests; allowed racemic mates have valid transformed stereo annotations

**Observed / mechanism:** General-position P-1 accepts enantiopure input and copies R to both mates; audited known R/S conflicts occur on improper operations

**Repair direction:** Validate the explicit chemistry policy against the full operation set before expansion; reassign stereo from complete typed graph plus transformed geometry for allowed improper mates, or preserve unknown status. Do not mechanically flip every stereo descriptor.

**Regression acceptance:**

- Explicit enantiopure general-position P -1 request fails closed; a deliberately allowed racemate succeeds.
- Allowed improper mates have correct known stereo under their typed graph; do not blanket-swap all descriptors.
- Rerun the 93 confirmed CSD R/S cases and separate unknown/assignment-error states.

**Refcodes:** AFOSUK, APUVUE, ARUMOP, AWESOO, AXEHEQ, AXOXEQ Full lists: `problem_refcodes/RT02.confirmed.txt` and `.related.txt`.

## RT03 — Typed bonds are stripped and CIF output hardcodes single

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/operations/forward.py:204 — MolecularTemplate untyped connectivity](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L204)
- [molcrys_kit/operations/forward.py:239 — graph-to-connectivity drops edge attributes](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L239)
- [molcrys_kit/operations/forward.py:766 — generated edges omit bond order/kind](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L766)
- [molcrys_kit/io/output.py:658 — P1 literal single](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L658)
- [molcrys_kit/io/output.py:1284 — ASU literal single](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L1284)
- [molcrys_kit/io/cif.py:2200 — published typed records reduced to integer pairs](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L2200)
- [molcrys_kit/io/cif.py:588 — materialized graph omits parsed bond types](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L588)

**Expected:** Known double/triple/aromatic/coordination types and unknown states survive construction and both CIF paths

**Observed / mechanism:** An explicitly annotated double edge is serialized as single; template input accepts only two-index edges

**Repair direction:** Add optional typed periodic bond records, preserving legacy pair inputs; propagate order/kind/unknown masks through template, generated graph, CIF bond loop and reader. Validate encoded tags separately from CCDC's unassigned chemistry.

**Regression acceptance:**

- C=C, C=O, aromatic and coordination examples retain order/kind through template, graph, P1, ASU and reader.
- Unknown bond type remains unknown; legacy pair inputs remain accepted with documented semantics.
- Verify raw encoded typed records before judging a native CCDC reader that may leave imported bonds unassigned.

**Refcodes:** ABEDUH, ACAKIC, ACIPIN, ADYLAD01, AFOSUK, AGANUU Full lists: `problem_refcodes/RT03.confirmed.txt` and `.related.txt`.

## RT04 — Isotope and explicit charge metadata disappear on writing

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/io/output.py:615 — P1 writes element-only type symbol](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L615)
- [molcrys_kit/io/output.py:837 — _forward_atom_rows omits isotope/charge fields](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L837)
- [molcrys_kit/io/output.py:1157 — ASU writes element-only type symbol](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L1157)
- [molcrys_kit/operations/forward.py:1145 — average-site records omit chemical annotations](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L1145)

**Expected:** D2 retains isotope 2; Na+/Cl- retain known +1/-1, with unknown distinguished from explicit zero

**Observed / mechanism:** Synthetic D2 reads isotopes [0,0]; synthetic ion charges read [None,None]; AMEGEF03 loses isotope metadata

**Repair direction:** Carry isotope, formal charge, known masks and permitted site identity into writer/ASU-reduction rows and averaged-site records; encode in supported CIF/companion fields and validate readback without charge guessing.

**Regression acceptance:**

- Synthetic D2 retains isotope 2 and known mass; Na+/Cl- retain explicit +1/-1.
- Unknown charge is distinct from explicit neutral zero; do not treat CCDC default zero as a known encoded value.
- ASU/average-site reduction never merges different known isotopes or incompatible site identities.

**Refcodes:** AMEGEF03 Full lists: `problem_refcodes/RT04.confirmed.txt` and `.related.txt`.

## RT05 — D/T aliases are passed to ASE as element names

**Owner / priority:** product / P2

**Modification entry points:**

- [molcrys_kit/io/cif.py:1299 — _parse_atom_type_symbol](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L1299)
- [molcrys_kit/io/cif.py:1741 — symbols/isotopes extracted](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L1741)
- [molcrys_kit/io/cif.py:2410 — ASE Atoms receives D](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/cif.py#L2410)

**Expected:** D maps to H/isotope 2; T to H/isotope 3; raw token remains auditable

**Observed / mechanism:** Synthetic one-D CIF and AMEGEF03 source sanity read raise KeyError

**Repair direction:** Normalize explicit isotope aliases before ASE construction; validate conflicts with leading mass numbers and preserve raw token, charge and known status.

**Regression acceptance:**

- D and T parse as H/isotope 2 and 3; leading-mass syntax and raw tokens remain supported.
- AMEGEF03 source read succeeds without losing isotope metadata; invalid conflicting annotations fail explicitly.

**Refcodes:** AMEGEF03 Full lists: `problem_refcodes/RT05.confirmed.txt` and `.related.txt`.

## RT06 — Statistical companion is not a lossless molecular model

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/operations/forward.py:913 — StatisticalDisorderModel.to_dict](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L913)
- [molcrys_kit/io/output.py:1407 — sidecar dump](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L1407)

**Expected:** A shipped reader reconstructs complete alternatives, graph/identity, context, weights, constraints and valid selections

**Observed / mechanism:** Alternative records contain IDs/weight/op/centre but no complete molecular record; no public inverse model reader is supplied

**Repair direction:** Define a versioned lossless schema and reader, including unwrapped atoms, typed graph, atom/site mappings, cell/setting/operations/tolerances, occupations and constraint status. Bind companion to CIF; fail explicitly on insufficient legacy data.

**Regression acceptance:**

- Statistical model write/read/write preserves complete alternative atoms, typed connectivity, mappings, weights and context.
- A selected restored microstate contains only its chosen complete alternatives.
- Old incomplete companions report loss explicitly rather than inventing graph/correlation data.

**Refcodes:** ABEDUL, AJIVAS, ANIMAN, APEFEH, APEFIN, AQIZAC Full lists: `problem_refcodes/RT06.confirmed.txt` and `.related.txt`.

## RT07 — Statistical true-ASU dispatch enters ordered writer

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/io/output.py:304 — ASU dispatch unconditionally calls write_cif_asu](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L304)
- [molcrys_kit/io/output.py:767 — ordered writer expects crystal.molecules](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L767)
- [molcrys_kit/io/output.py:1353 — statistical writer always declares P1](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/io/output.py#L1353)

**Expected:** True-source-group ASU marginal output independently expands to the same weighted inventory, with companion retaining alternative/correlation semantics

**Observed / mechanism:** write_forward_cif(statistical_model,representation='asu') raises AttributeError; supported statistical output is P1

**Repair direction:** Dispatch on representation AND object type; implement chemical-identity-aware reduction of average sites, correct multiplicities/occupancies and companion group action. Do not attach the full group to already expanded P1 rows.

**Regression acceptance:**

- Inversion-site water: true-ASU output expands to the same O=1, H=2 weighted composition and correct marginal sites.
- Centred-cell multiplicity and shared-site identity are verified independently; P1 averages are never expanded a second time.
- CIF represents marginals; companion preserves alternative/group action and correlations where CIF tags cannot.

**Refcodes:** ABEDUL, AJIVAS, ANIMAN, APEFEH, APEFIN, AQIZAC Full lists: `problem_refcodes/RT07.confirmed.txt` and `.related.txt`.

## RT08 — Benchmark treats unavailable CCDC chemistry as known labels

**Owner / priority:** benchmark / P1

**Modification entry points:**

- [benchmarks/csd_roundtrip/ccdc_adapter.py:90 — snapshot converts chirality sentinel into stereo value](https://github.com/SchrodingersCattt/MolCrysKit/blob/12b06e79ef4b61777f9c4575d54a3a103e2726aa/benchmarks/csd_roundtrip/ccdc_adapter.py#L90)
- [benchmarks/csd_roundtrip/worker.py:75 — chemistry_check availability and identity logic](https://github.com/SchrodingersCattt/MolCrysKit/blob/12b06e79ef4b61777f9c4575d54a3a103e2726aa/benchmarks/csd_roundtrip/worker.py#L75)
- [benchmarks/csd_roundtrip/worker.py:148 — annotation transfer falls back to CIF-derived chemistry](https://github.com/SchrodingersCattt/MolCrysKit/blob/12b06e79ef4b61777f9c4575d54a3a103e2726aa/benchmarks/csd_roundtrip/worker.py#L148)

**Expected:** Error/Unknown/Mixed statuses are separate from known R/S; unavailable bond types are not contradictory known orders; source annotations are bound without inference

**Observed / mechanism:** Identical geometry with stereo None versus Error is scored mismatch. CCDC synthetic CIF import returns bond Unknown/chirality Error for both tested tag aliases. Of 108 raw memory identity flags, 93 have known R/S conflicts, one has missing known stereo and 14 have no valid-label contradiction after sentinel normalization.

**Repair direction:** Preserve raw vendor status plus availability masks; compare valid assignments only and record missing/indeterminate separately. Check encoded bond data directly; do not repair validation by guessing bond orders or copying truth into outputs. Bind input annotations against symmetry-complete source instances when the CIF reader chooses another mate.

**Regression acceptance:**

- Vendor Error/Unknown are unavailable statuses, not valid CIP configurations; Mixed is a qualified state.
- Known R versus unassigned imported chemistry is missing evidence, not an observed R-to-S conversion.
- Input/oracle availability is aligned (NUYVOU); direct template matches must be inspected before attributing a flag to failed transfer. A different CIF-selected mate needs an explicit source-instance map.
- An output validator reads only what was encoded; it must not copy source truth or perceive replacement bond types to hide losses.

**Refcodes:** ABEDUH, ACAKIC, ACIPIN, ADYLAD01, AFOSUK, AGANUU Full lists: `problem_refcodes/RT08.confirmed.txt` and `.related.txt`.

## RT09 — Disorder seed/oracle is a limited benchmark probe

**Owner / priority:** benchmark / P1

**Modification entry points:**

- [benchmarks/csd_roundtrip/worker.py:129 — one selected source template set](https://github.com/SchrodingersCattt/MolCrysKit/blob/12b06e79ef4b61777f9c4575d54a3a103e2726aa/benchmarks/csd_roundtrip/worker.py#L129)
- [benchmarks/csd_roundtrip/worker.py:207 — q inferred from minimum atomic occupancy](https://github.com/SchrodingersCattt/MolCrysKit/blob/12b06e79ef4b61777f9c4575d54a3a103e2726aa/benchmarks/csd_roundtrip/worker.py#L207)
- [benchmarks/csd_roundtrip/worker.py:379 — strict average-site count comparison](https://github.com/SchrodingersCattt/MolCrysKit/blob/12b06e79ef4b61777f9c4575d54a3a103e2726aa/benchmarks/csd_roundtrip/worker.py#L379)

**Expected:** All supplied source molecular alternatives/weights are explicit; unknown input relations remain indeterminate; valid split/merged shared sites are compared by justified identity

**Observed / mechanism:** Only the selected template seed is supplied; q is inferred; source-group site counts are compared without a complete shared-site equivalence oracle

**Repair direction:** Build the complete source input model before grading the constructor, or label this probe incomplete. Supply non-symmetry-related conformers explicitly, validate q and compare occupancy measures with permitted site identities. Never force the constructor to invent missing experimental alternatives.

**Regression acceptance:**

- Two non-symmetry-related states at q=1 with weights .25/.75 are both supplied, not converted into a .25 vacancy model.
- Shared site identity permits justified split/merge equivalence while distinct coincident entities remain distinct.
- Insufficient source coupling/input data produce an indeterminate input-model status.

**Refcodes:** Synthetic-only or no individually confirmed CSD association. Full lists: `problem_refcodes/RT09.confirmed.txt` and `.related.txt`.

## RT10 — Zero-probability alternatives can be selected as occupied

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/operations/forward.py:1195 — select_ordered_realization defaults to first state](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L1195)
- [molcrys_kit/operations/forward.py:1203 — selected molecule appended without weight/occupation check](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L1203)

**Expected:** q=0 yields a vacant centre or a documented typed refusal; zero-weight occupied alternatives cannot be selected

**Observed / mechanism:** Synthetic q=0 water has alternative weight 0 but selection materializes 3 atoms

**Repair direction:** Respect centre occupation and zero weights, represent vacancy explicitly, reject impossible selections, and validate declared coupling before reporting a physical realization. Recompute actual selected-state symmetry separately from source average symmetry.

**Regression acceptance:**

- q=0 cannot materialize an occupied molecule; explicitly requesting a zero-weight state is rejected.
- q<1 vacancy selection and known coupling are checked; local marginals alone do not establish global feasibility.

**Refcodes:** Synthetic-only or no individually confirmed CSD association. Full lists: `problem_refcodes/RT10.confirmed.txt` and `.related.txt`.

## RT11 — Stabilizer matching uses independent atom image shifts

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/operations/forward.py:466 — common anchor shift computed](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L466)
- [molcrys_kit/operations/forward.py:478 — pairwise atom differences wrapped independently](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L478)

**Expected:** An operation stabilizes the complete unwrapped molecule under ONE common lattice shift and a valid atom permutation

**Observed / mechanism:** C-O at x=0,1.2 in a 2.4-A cell is declared inversion-compatible although C fixes shift 0 and O would require another image; the independent whole-molecule comparator rejects it

**Repair direction:** After the common shift, match complete unwrapped Cartesian geometry without per-atom MIC. Keep chemically valid permutations and graph/image invariants; unwrap input templates coherently rather than rescuing a bad pose by atom-wise wrapping.

**Regression acceptance:**

- The synthetic C-O inversion example has stabilizer order 1 and is not ordered-compatible.
- Valid long boundary-crossing molecules still match after one coherent whole-molecule translation.
- Retain the independent-atom-wrap negative regression in the benchmark matcher.

**Refcodes:** Synthetic-only or no individually confirmed CSD association. Full lists: `problem_refcodes/RT11.confirmed.txt` and `.related.txt`.

## RT12 — Declared group is not checked against the complete operation set

**Owner / priority:** product / P1

**Modification entry points:**

- [molcrys_kit/operations/forward.py:116 — _validate_group validates only the supplied set](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L116)
- [molcrys_kit/operations/forward.py:328 — context copies declared group metadata after closure check](https://github.com/SchrodingersCattt/MolCrysKit/blob/9bf0ffb0fb321d1084dbc444250b5f186af3c873/molcrys_kit/operations/forward.py#L328)

**Expected:** P -1 / IT 2 with only identity is rejected as contradictory

**Observed / mechanism:** prepare_construction_context accepts declared group 2 with 1 operation

**Repair direction:** Validate the complete operation set against number/symbol/Hall/setting/origin/basis in the same authority used by the strict CIF reader. Reject inconsistent declarations, while allowing explicitly custom groups without invented crystallographic labels.

**Regression acceptance:**

- Declared P -1 with identity-only operations raises InvalidSymmetryError.
- Closed same-order but wrong groups/settings/origins also fail; legitimate basis transforms and custom unlabelled groups remain valid.

**Refcodes:** Synthetic-only or no individually confirmed CSD association. Full lists: `problem_refcodes/RT12.confirmed.txt` and `.related.txt`.

## Unresolved queues

U_STAGE_FAILURE / U_TIMEOUT / U_DISORDER_INPUT keep their raw stage and code. They have not been promoted to confirmed product bugs. Use the licensed local inputs and run-specific diagnostics to minimize each case; keep successful controls such as ABEDUH and ANOSUS green.

No CIF, coordinate, atom map or raw exception is included here. All named source entries remain retrievable through licensed CSD access.
