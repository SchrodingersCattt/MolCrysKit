> **Follow-up interpretation:** start with [triage.md](triage.md). These are frozen v2 pipeline counts, not counts of confirmed product defects. RT08 separates unavailable CCDC assignments from valid chemical-label conflicts; RT09 documents the limited disorder-input/oracle path. Raw result rows are unchanged.

# CSD molecular-crystal round-trip benchmark

Run: `csd601-sep2026-baseline-v2`. Target: `9bf0ffb0fb321d1084dbc444250b5f186af3c873`.
Manifest SHA-256: `829afa3517cfc1cd491221a3dd678646344cb8261592b1c767bcb749509b94a6`.

Results received: **1004/1004**. Missing results remain indeterminate in the denominator.

## Match contract

No rigid fitting, cell scaling or origin fitting. Molecules permit only coherent integer lattice translations,
valid atom permutations and whole-molecule permutations. Position max 1e-4 Å; cell lengths 1e-6 Å;
angles 1e-5 degrees; operation translations 1e-7 fractional; occupancy 1e-5.

Overall pass requires all applicable stages, including known chemical identity, both CIF representations,
and reconstructable whole-molecule disorder where applicable. Source-missing coordinates are reported separately;
no atoms are added. Marginal average coordinates do not establish a lossless disorder round trip.

## Fixed-test-set results

| Stratum | N | Pass | Mismatch | Unsupported | Error | Timeout | Indeterminate | Record match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| disordered | 200 | 0 | 173 | 16 | 3 | 8 | 0 | 0 |
| special_position | 200 | 0 | 168 | 1 | 26 | 5 | 0 | 0 |
| multi_ion | 200 | 0 | 189 | 0 | 10 | 1 | 0 | 0 |
| multi_component | 200 | 0 | 195 | 0 | 2 | 3 | 0 | 0 |
| single_component | 200 | 0 | 191 | 0 | 3 | 6 | 0 | 0 |
| targeted | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| all | 1004 | 0 | 920 | 17 | 44 | 23 | 0 | 0 |

## Geometric recovery

| Stratum | Ordered memory geometry | Disorder memory average | Complete ordered geometry round trip |
|---|---:|---:|---:|
| disordered | 0 | 22 | 0 |
| special_position | 164 | 0 | 89 |
| multi_ion | 178 | 0 | 160 |
| multi_component | 186 | 0 | 152 |
| single_component | 181 | 0 | 160 |
| targeted | 3 | 0 | 2 |
| all | 712 | 22 | 563 |

## Stage results

| Stage | Status counts |
|---|---|
| asu_average | {"unsupported": 190} |
| asu_ccdc_chemistry | {"mismatch": 752, "unsupported": 1} |
| asu_ccdc_geometry | {"mismatch": 50, "pass": 703} |
| asu_cell | {"pass": 753} |
| asu_mck_readback | {"mismatch": 156, "pass": 596} |
| asu_operations | {"pass": 753} |
| asu_pymatgen | {"mismatch": 8, "pass": 745} |
| asu_readback | {"error": 39} |
| asu_written | {"pass": 758} |
| average_chemistry | {"unsupported": 190} |
| average_source_symmetry | {"pass": 190} |
| context | {"pass": 1003} |
| correlation_preservation | {"pass": 190} |
| disorder_model_readback | {"unsupported": 190} |
| execution | {"error": 5, "timeout": 23} |
| export | {"mismatch": 118, "pass": 886} |
| memory_average | {"mismatch": 171, "pass": 22} |
| memory_cell | {"pass": 801} |
| memory_chemistry | {"mismatch": 197, "unsupported": 604} |
| memory_geometry | {"mismatch": 89, "pass": 712} |
| memory_symmetry | {"pass": 801} |
| microstate_exclusivity | {"indeterminate": 3, "pass": 187} |
| p1_average_ccdc | {"mismatch": 173, "pass": 17} |
| p1_average_pymatgen | {"mismatch": 173, "pass": 20} |
| p1_ccdc_chemistry | {"mismatch": 796, "unsupported": 1} |
| p1_ccdc_geometry | {"mismatch": 89, "pass": 708} |
| p1_cell | {"pass": 797} |
| p1_mck_readback | {"mismatch": 129, "pass": 629} |
| p1_pymatgen | {"mismatch": 8, "pass": 750} |
| p1_readback | {"error": 39} |
| p1_source_operations | {"pass": 797} |
| p1_written | {"pass": 801} |

## Failure categories

| Stage and code | N | Example refcodes |
|---|---:|---|
| p1_ccdc_chemistry:chemical_identity_or_bond_order_mismatch | 707 | BENZEN02, HXMTAM10, IBPRAC, ABEDUH, ACAKIC |
| asu_ccdc_chemistry:chemical_identity_or_bond_order_mismatch | 702 | BENZEN02, HXMTAM10, IBPRAC, ABEDUH, ACAKIC |
| memory_chemistry:known_chemical_annotations_not_preserved | 604 | BENZEN02, HXMTAM10, ABEDUH, ACAKIC, ACIPIN |
| asu_average:true_asu_average_unavailable | 190 | IQUHUY, ABEDUL, AJIVAS, ANIMAN, APEFEH |
| average_chemistry:average_sites_omit_known_chemical_annotations | 190 | IQUHUY, ABEDUL, AJIVAS, ANIMAN, APEFEH |
| disorder_model_readback:no_lossless_disorder_sidecar_reader | 190 | IQUHUY, ABEDUL, AJIVAS, ANIMAN, APEFEH |
| p1_average_ccdc:average_site_count | 173 | IQUHUY, AJIVAS, ANIMAN, APEFEH, APEFIN |
| memory_average:average_site_count | 170 | IQUHUY, AJIVAS, ANIMAN, APEFEH, APEFIN |
| p1_average_pymatgen:average_site_count | 170 | IQUHUY, AJIVAS, ANIMAN, APEFEH, APEFIN |
| p1_mck_readback:whole_molecule_geometry_or_graph | 119 | ACAKIC, APEYIF, AQEXAX, BEKNAI, BEYCIV |
| memory_chemistry:atomic_chemical_identity_not_preserved | 108 | IBPRAC, AFOSUK, APUVUE, ARUMOP, AWESOO |
| asu_mck_readback:whole_molecule_geometry_or_graph | 84 | APEYIF, AQEXAX, BEYCIV, BIVLAU, BOXNUY |
| asu_mck_readback:molecule_count | 72 | HXMTAM10, AJUSEH01, AMOJET, BETDCU10, CIQSOK |
| p1_ccdc_chemistry:whole_molecule_geometry_or_graph | 56 | BANVOB10, BEZMEA, BOXNUY, CEYGOC, COBPES |
| p1_ccdc_geometry:whole_molecule_geometry_or_graph | 56 | BANVOB10, BEZMEA, BOXNUY, CEYGOC, COBPES |
| export:molecule_count | 51 | AGUNUO, ALAXAQ, ARECUW, BANVOB10, BETDCU10 |
| memory_chemistry:molecule_count | 46 | AGUNUO, ALAXAQ, ARECUW, BANVOB10, BETDCU10 |
| memory_geometry:molecule_count | 46 | AGUNUO, ALAXAQ, ARECUW, BANVOB10, BETDCU10 |
| export:whole_molecule_geometry_or_graph | 45 | BEZMEA, BOXNUY, CEYGOC, DATYOO, ETOPUZ |
| memory_chemistry:whole_molecule_geometry_or_graph | 43 | BEZMEA, BOXNUY, CEYGOC, ETOPUZ, FULBET |
| memory_geometry:whole_molecule_geometry_or_graph | 43 | BEZMEA, BOXNUY, CEYGOC, ETOPUZ, FULBET |
| asu_ccdc_chemistry:whole_molecule_geometry_or_graph | 39 | BEZMEA, BOXNUY, CEYGOC, ETOPUZ, FULBET |
| asu_ccdc_geometry:whole_molecule_geometry_or_graph | 39 | BEZMEA, BOXNUY, CEYGOC, ETOPUZ, FULBET |
| asu_readback:serialization_or_independent_readback | 39 | AGUNUO, ALAXAQ, ARECUW, BANVOB10, COBPES |
| p1_readback:serialization_or_independent_readback | 39 | AGUNUO, ALAXAQ, ARECUW, BANVOB10, COBPES |
| p1_ccdc_chemistry:molecule_count | 33 | AGUNUO, ALAXAQ, ARECUW, BETDCU10, CIQSOK |
| p1_ccdc_geometry:molecule_count | 33 | AGUNUO, ALAXAQ, ARECUW, BETDCU10, CIQSOK |
| execution:case_300s_limit | 23 | BIDPAJ, BIGBEZ, DEXNUS, DIGHEK, ETOPUZ |
| export:average_site_count | 12 | BOZBOL, ECAWAL, FUNDEZ, HIPLID, KAYQEJ |
| asu_ccdc_chemistry:molecule_count | 11 | BETDCU10, CIQSOK, DAPRCV, FAJNUC, FEGCEA |
| asu_ccdc_geometry:molecule_count | 11 | BETDCU10, CIQSOK, DAPRCV, FAJNUC, FEGCEA |
| export:average_site_geometry_occupancy | 10 | AREZAZ, AXUQIW, CERQOJ, CEXYIR, EFEZOF |
| p1_mck_readback:molecule_count | 10 | BETDCU10, CIQSOK, DAPRCV, FAJNUC, GOCDOR |
| asu_pymatgen:independent_expanded_composition | 8 | GOCDOR, ILUCIB, POVJIV, QOVCEL04, TECNAQ10 |
| p1_pymatgen:independent_expanded_composition | 8 | GOCDOR, ILUCIB, POVJIV, QOVCEL04, TECNAQ10 |
| execution:exception | 5 | DATYOO, KUKPEO, OFIGOD, TENLUT, XOHZIF |
| microstate_exclusivity:microstate_enumeration_truncated | 3 | ILOWUC10, TIDRIJ, YIJMAH |
| p1_average_pymatgen:average_site_geometry_occupancy | 3 | AXUQIW, ECAKAZ, IHIWEE |
| asu_ccdc_chemistry:known_chemical_annotations_not_preserved | 1 | AMEGEF03 |
| memory_average:average_site_geometry_occupancy | 1 | IHIWEE |
| p1_ccdc_chemistry:known_chemical_annotations_not_preserved | 1 | AMEGEF03 |

## Interpretation and reproducibility

- The five strata have fixed quotas; totals describe this test set, not a population-weighted CSD success rate.
- Export/CIF extraction defects are separate from constructor failures. An earlier failure can prevent later stages;
  stage denominators and absent stages must be read with the overall status.
- CCDC supplies original chemical annotations. The pinned constructor accepts untyped connectivity; known bond-order
  information that cannot be represented is an unsupported chemistry gate, even if coordinates match.
- CSD coordinates, CIFs, templates, mappings and raw logs remain under ignored `_local/`.
- Use the frozen manifest and recorded environment; `run --refcodes REFCODE` reproduces a case.
- Source correlations absent from CIF stay unknown. Bounded enumeration is not experimental joint-distribution evidence.
