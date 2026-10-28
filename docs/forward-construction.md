# Forward construction contracts

The strict constructor expands explicit rigid molecular poses. It does not
search, relax, infer missing conformers or determine experimental correlations.
The contract is defined by [#172](https://github.com/SchrodingersCattt/MolCrysKit/issues/172)
and [#198](https://github.com/SchrodingersCattt/MolCrysKit/issues/198).

## Geometry, chemistry and accounting

Coordinates use angstroms and row lattice vectors. Stabilizers must preserve a
complete unwrapped molecule under one common lattice translation, atom identity
(including charge availability), and typed connectivity. Fractional near-special
residuals are dimensionless; their Cartesian equivalents are in angstroms.
Neither diagnostic snaps a pose. Declared groups and settings must agree with
the complete supplied operation set. Unlabelled closed custom groups are valid.

`chemistry_policy="enantiopure"` applies to ordered and statistical construction,
including general positions. Allowed improper mates reassign known stereo using
typed connectivity and geometry; unknown or unassignable stereo stays unknown.
An annotation is not an assignment oracle. Legacy two-index bonds have unknown
order and type. Explicit known masks distinguish unavailable annotations from
known values; charge zero is never guessed from element counts.

Both constructors accept `expected_formula` as full-cell element counts,
`formula_unit` together with a positive integer `Z`, and `expected_charge` as the
full-cell formal charge. Statistical checks use weighted complete alternatives.
A charge check with any unknown contributing formal charge fails explicitly as
indeterminate. The `composition_accounting` report separates known partial
charge sums from a known total. Formula-unit counts do not imply molecule counts
for a molecular salt.

`intermolecular_contacts` uses a periodic neighbor search in the actual cell,
including skew cells and nonzero images of the same molecule. Records include
angstrom distances, covalent-radius limits and directed integer image shifts.
`reject_contacts=True` opts into rejection for ordered construction and selection.
Coordinates are never repaired. Selected coordinates are checked as complete
molecules against each source operation; isotope/charge differences can lower
the realized symmetry. A separate spglib coordinate analysis reports detected
space-group number, operation count and accidental geometric supergroups. It
uses element/isotope/charge types; it is a geometric diagnostic, not a chemical
space-group assignment, and an unsuccessful detection stays indeterminate.

## Disorder constraints and selection

`molcrys.disorder_constraints.v1` uses explicit `joints`, each with `weight` and
an `assignment` mapping every occupied/vacant center-orbit slot to an alternative
ID. A distribution declared `feasible` must be normalized, reproduce all
marginals and be invariant under the source group acting on complete molecular
states. Molecular operation representatives are cosets: different labels can
represent the same orientation. Duplicate assignments contribute summed weight.

`forbidden` contains assignment mappings and can specify subsets of orbits.
Every named alternative must belong to its named orbit. Any positive joint
extending a forbidden subset is rejected. Selection checks those subsets and,
for a feasible joint distribution, requires a positive-weight joint state.
Missing joint data remains unknown; truncated data does not establish feasibility.
Opaque legacy correlation metadata is retained only as unvalidated local data.

Vacancies are explicit alternatives. Each partially occupied center requires an
explicit occupied/vacant choice. Every selected state must have positive weight.
Unknown orbit names and noninteger alternative indices fail validation. A selected
microstate can have different composition and lower symmetry than the average;
its selected alternative IDs and source model remain available as provenance.

## Lossless companion v2 and migration

The current schema is `molcrys.statistical_disorder.v2`. It stores unwrapped
coordinates; typed, directed periodic bond records; isotopes; explicit charge
masks; masses; supported ASE arrays with dtype and shape; complete atom/site and
operation mappings; cell/setting/tolerances; occupations; vacancies; constraints;
and marginal sites. JSON arrays support boolean, integer, float and string dtypes.
Unsupported object/complex arrays or nonfinite/non-JSON metadata fail explicitly
instead of being stringified. The simple molecular graph has one edge per atom
pair; complete periodic records remain available in `forward_bonds` and in the
companion bond list.

Readers validate array dimensions, atom annotations, bond records, unique state
IDs, per-orbit normalization and occupation totals, then recompute marginal sites
from complete alternatives and revalidate constraints. A serialized feasibility
report is never trusted as evidence. `write/read/write` preserves the payload.

Both statistical CIF writers bind the companion to the exact emitted UTF-8 CIF
with SHA-256. Pass `cif_text` to `read_statistical_disorder_companion` to check the
binding. In-memory `model.to_dict()` remains independently reconstructable without
a CIF; omitting `cif_text` does not check a file association. A `sidecar_path` can
also be supplied when the CIF is returned in memory. Hashes bind files, not their
scientific truth.

Id-only companions and the earlier v1 schema are rejected with
`DisorderCompanionLossError`: their missing masses, arrays and mappings cannot be
recovered losslessly. Rebuild from the complete source poses and export v2; a
schema-string edit is not a migration. No licensed database is needed for v2.

## CIF and other exports

`write_cif` keeps its expanded-P1 default. Standard bond types plus MolCrysKit
order/kind/known-mask columns preserve explicit chemistry in P1 and ordered ASU
round trips. Published symmetry-coded endpoint images are authoritative even
when longer than the nearest periodic image. This is not a bond-distance filter.
External readers may leave imported bond chemistry unassigned; inspect the raw
encoded columns and a decoder that supports them before comparing assignments.

`write_forward_cif(model, representation="asu")` reduces full-cell marginals and
emits the source operation set exactly once. Chemically different coincident
sites remain separate rows. Complete-state symmetry matching handles transformed
stereo without a blanket descriptor swap. P1 marginals give each complete local
alternative its own disorder group and use group 0 for shared sites. In an ASU,
where symmetry exchanges states or site assemblies, fixed CIF group labels
cannot describe the coupling: those tags are explicitly unknown (`.`). The
companion supplies complete exclusivity and correlations. Marginal atom rows
alone are never an ordered molecular microstate.

XYZ and POSCAR warn when dropping partial occupancy. ASE and ExtXYZ preserve
supported per-atom arrays but cannot reconstruct whole-molecule alternatives,
correlations or full provenance; use the companion for that purpose.

## Acceptance evidence

The following are deterministic, licence-free tests, except for the already
pinned de-identified fixture files. No live CSD retrieval or benchmark runtime
dependency is introduced.

| Contract | Test evidence |
| --- | --- |
| Complete group/setting/origin/basis and deterministic operation ordering | `test_declared_group_must_match_the_complete_operation_set`, `test_acceptance_geometry_symmetry_and_export_contracts` |
| Whole-group chemistry in both construction paths; valid improper stereo | `test_explicit_enantiopure_general_position_fails_and_racemate_reassigns_stereo`, `test_statistical_path_enforces_enantiopure_policy` |
| Coherent molecular images, typed stabilizers and charge availability | `test_stabilizer_matching_does_not_wrap_atoms_independently`, `test_typed_graph_and_charge_availability_constrain_stabilizers` |
| Special-site methane: 2 CH4, 10 atoms, exactly 8 bonds | `test_special_methane_asu_keeps_two_molecules_and_eight_bonds`, `test_forward_methane_i43m_topology`; synthetic CIF in `tests/data/cif/` |
| Typed chemistry including double/triple/aromatic/coordination/unknown | `test_complete_typed_bond_records_survive_both_cif_readers`, `test_unknown_bond_masks_and_invalid_image_shifts_are_not_promoted` |
| Authoritative long and boundary-crossing bonds | `test_long_published_bond_uses_authoritative_image_instead_of_nearest_image`, existing boundary/ring forward regressions |
| Isotope aliases, masses, explicit charges and chemically distinct averages | `test_deuterium_and_tritium_aliases_and_conflicts`, `test_isotope_and_explicit_charge_round_trip_and_do_not_merge`, `test_companion_is_bound_to_cif_and_preserves_known_mass`, `test_statistical_asu_keeps_coincident_isotope_alternatives` |
| Joint feasibility, coupling, cosets, partial forbidden assignments and vacancies | `test_joint_constraints_distinguish_coupling_and_reject_contradictions`, `test_feasible_joint_uses_molecular_cosets_and_preserves_pose_identity`, `test_partial_forbidden_constraints_and_duplicate_joint_weights`, `test_zero_weight_and_partial_occupation_selection` |
| Reconstructable companion and rejection of corrupt/stale data | `test_companion_preserves_masses_arrays_and_revalidates_constraints`, `test_companion_rejects_inconsistent_marginals`, CIF binding tests |
| Nonorthogonal P1, order-4 site with order-2 molecular stabilizer, high-symmetry general site and near-special boundaries | `test_acceptance_geometry_symmetry_and_export_contracts` plus targeted boundary/input tests |
| Multi-site molecular salt, formula/Z, partial occupation and explicit charge balance | `test_formula_Z_and_explicit_charge_constraints_on_multispecies_salt` |
| Periodic self-image/skew-cell contacts and independently detected symmetry | `test_periodic_contacts_include_self_images_and_skew_cell`, `test_selected_realization_checks_contacts_and_detects_accidental_symmetry` |
| IBPRAC, BENZEN02, HXMTAM10 and IQUHUY; independent CIF expansion | `tests/unit/io/test_forward_fixtures.py` |
| Separate complete BENZEN02 C6H6 acceptance | `test_benzen02_complete_molecule_with_explicit_modeled_hydrogen_policy`; measured C retained, radial 1.09 A modeled H explicitly documented in fixture manifest; frozen 4 molecules/48 atoms |
| IQUHUY 50:50 local marginals, lossless restoration, explicit joint ensemble and 30-atom selections | `test_iquhuy_declared_independent_joint_restoration_and_true_asu`; declared independence is an example forward ensemble, not an experimental correlation claim |

The new targeted tests are also run against pre-fix main to record failures;
existing #173/#178 tests remain baseline controls. The historical 93 CSD stereo
cases and other large licensed-corpus associations remain optional local
validation. They are not relabelled as independently reproduced here. RT08/RT09
are benchmark limitations, outside the production constructor contract. Frozen
formula/multiplicity oracles and independent pymatgen expansion are used in
addition to constructor diagnostics.
