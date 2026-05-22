# MolCrysKit – Agent / AI-Coding Guidelines

This file describes conventions and constraints that AI coding agents should
follow when working in this repository.

---

## Repository Layout

```
molcrys_kit/
  analysis/disorder/   Core disorder-resolution pipeline
    graph.py           DisorderGraphBuilder  – builds the exclusion graph
    solver.py          DisorderSolver        – MWIS solve + motif merge
    diagnostics.py     Post-resolution valence-completeness checker
    process.py         High-level entry point
  io/cif.py            CIF parser / scan_cif_disorder
  structures/          MolecularCrystal, Molecule types
  constants/           Element radii, config thresholds
examples/              CIF files used by regression tests
scripts/               One-off diagnostics / repro scripts
tests/unit/            Pytest regression suite
```

---

## Disorder Pipeline: Key Design Decisions

### Two-path architecture

The disorder solver has two **complementary** paths that both operate on the
same exclusion graph and must not be collapsed into one:

1. **Explicit path** – uses `_atom_site_disorder_assembly` / `_disorder_group`
   tags from the CIF.  Handled by `_add_explicit_conflicts` and
   `_add_conformer_conflicts`.

2. **Implicit SP path** – for partial-occupancy atoms on crystallographic
   special positions *without* disorder tags (e.g. SHELX riding-H).
   Handled by `_add_implicit_sp_conflicts` (pairwise within-cluster edges)
   and `_resolve_valence_conflicts` → `_sp_tetrahedral_single` →
   `_sp_apply_group_constraints` (cross-cluster tetrahedral compatibility).

Both paths add edges of type `"valence_geometry"` or `"implicit_sp"` to the
same `networkx.Graph`.  The edge-priority table in `edge_priority.py` ensures
higher-confidence types always win via `add_or_promote_edge`.

### Motif merge post-pass

During group identification, `_merge_chemical_motifs` reconstructs isolated
XH_n motifs (NH4+, H2O) from the remaining singletons.  For nitrogen centres,
soft conflicts (`valence_geometry`, `implicit_sp`, `geometric`) are ignored
(`_MOTIF_REJECT_SOFT["N"] = False`) because the SP disorder typically adds soft
edges between *every* H pair.

**One-per-asym_id guard** (added in feat/sp-nh4-implicit-hardening):
When the number of distinct `asym_id` values among candidate H atoms is
≥ `max_H` (4 for N), the greedy selection in `_select_motif_hydrogens`
enforces at most one pick per `asym_id`.  This prevents selecting multiple
copies of the same SHELX disorder position (which are nearly co-linear and
would monopolise the 4 available slots before other H sites can be chosen).
The guard is disabled when fewer distinct asym_ids exist (e.g. DAP-4 NH4+
with only 2 H labels but needing 4 H), preserving the original angle-only
heuristic for those cases.

### Three-mode solver contract

`DisorderSolver.solve()` supports three modes with deliberately different
semantics:

- `optimal`: deterministic greedy MWIS over rigid groups.  The group score is
  `occupancy_sum * group_size / (external_degree + 1)`, preserving the
  ClO3-vs-lone-O heuristic.
- `random`: replica 0 is the deterministic MWIS reference.  Later replicas are
  occupancy-weighted samples over PART/SP alternatives and may legitimately
  vary in guest/alternative populations, but must remain chemically valid.
- `enumerate`: deterministic top-N enumeration.  Every returned replica must be
  chemistry-equivalent to the MWIS reference (same element totals and motif
  counts), even if the raw alternative pool contains lower-quality variants.

The high-level API and graph/solver constructors accept `coupled`:

- `coupled=False` (default): symmetry-expanded copies of one PART/assembly or
  implicit-SP motif are independent decision components.  This is the intended
  mode for mixed replicas such as `ABAB`.
- `coupled=True`: legacy compatibility mode where symmetry copies sharing a
  disorder assembly are locked to the same PART choice, and motif merge keeps
  the single greedy best X(H)n orientation.

Every crystal returned by `generate_ordered_replicas_from_disordered_sites`
carries `crystal.disorder_provenance` with the final cleaned `kept_indices`,
`dropped_indices`, `method`, `coupled`, and per-kept-site `sym_op_indices`
when available.  `kept_indices` always index the source `DisorderInfo` arrays;
`molecule.info["atom_indices"]` remains the local output-atom index used by
`MolecularCrystal.to_ase()` to restore order.  For PART-switching NEB workflows,
compare two replicas by intersecting / diffing their provenance `kept_indices`.
Do not propagate disorder provenance through `get_supercell()` or `from_ase()`;
those structures no longer have a one-to-one source disorder-site contract.

Random/enumerate alternatives intentionally keep a linear occupancy weight
(`sum(group occupancy)`) so the sampler remains occupancy-weighted rather than
using the optimal-mode `* group_size / degree` heuristic.  The solver instead
stabilises outputs by:

1. Scanning clique alternatives with a bounded top-K heap rather than keeping
   the first `max_alts * 64` cliques yielded by NetworkX.  This keeps memory
   bounded while preventing high-weight alternatives from being missed simply
   because they appear late in DFS order.
2. Injecting the MWIS reference into each component's alternative pool and
   returning it as replica 0 for both `random` and `enumerate`.
3. Running the post-pass chain in this order:
   `_repair_motifs_in_set` → `_apply_sp_completion` →
   `_remove_too_close_sp_hydrogens` → `_remove_orphan_hydrogens` →
   `_relocate_overcoord_sp_hydrogens`.
4. Replacing non-reference `random` samples with the MWIS reference when they
   break complete NH4 motif counts or contain sub-0.65 Å contacts.  Replacing
   non-reference `enumerate` samples when their element totals drift from the
   reference.

When adding disorder regression cases, make the intended `coupled` value
explicit in the test.  Keep legacy regression expectations on `coupled=True`
when they are asserting historical behaviour, and add targeted `coupled=False`
tests for symmetry-copy decoupling or mixed-replica coverage.

### Do not add a single "unified" path

Merging the explicit and implicit paths would break the large regression
suite (24+ CIFs).  When you need to fix a new structure type, extend one of
the two paths or add a new post-pass; do not rewrite the graph-building logic.

---

## Regression Tests

`tests/unit/test_disorder_regression.py` is the single source of truth for
"how the solver is supposed to behave on real-world CIFs".

Rules:
- Every new structural motif that was previously broken must get its own
  `CifCase` in `CASES` **and** a targeted `test_<material>_topology()`
  function that asserts per-molecule formula counts.
- `xfail_reason` is allowed for known-broken cases but must be removed when
  fixed.
- The corresponding CIF must be copied to `tests/data/cif/`.
- Full suite (including PAP-4, timeout=180 s) must remain green.

Current targeted assertions beyond atom count:
- `test_dai4_topology`: `H4N1 == 8` — both implicit (N1) and explicit (N4)
  NH4+ must resolve to full tetrahedra.
- `test_dap4_topology`: `H4N1 == 8` — multi-orientation SP NH4+.
- `test_dap7_topology`: `H6N2 == 1` — hydrazinium cation.
- `test_paphm4_topology`: `H4N1 == 4` — NH4+ in PAP salt.
- `test_natcomm1_topology`: bridged Cd2(SCN)6 cluster formula.

---

## Edge Types and Priorities

From `edge_priority.py` (highest to lowest priority):

| Type | Priority | Meaning |
|---|---|---|
| `logical_alternative` | 100 | CIF PART/assembly explicit alternative |
| `symmetry_clash` | 90 | Ghost overlap from symmetry expansion |
| `explicit` | 80 | Assembly-based exclusion |
| `valence` | 70 | Over-coordination chemistry |
| `valence_geometry` | 60 | Geometry-unreasonable bond angle/distance |
| `geometric` | 30 | Pure distance clash |
| `implicit_sp` | 20 | Weak SP proximity overlap |

Hard conflicts in `_MOTIF_HARD_CONFLICTS`:
`{"logical_alternative", "symmetry_clash", "explicit", "valence"}`

Soft conflicts ignored for N in `_select_motif_hydrogens`:
`{"geometric", "valence_geometry", "implicit_sp"}`

---

## Diagnostics

`molcrys_kit.analysis.disorder.diagnostics.check_valence_completeness`
is called automatically at the end of `DisorderSolver.solve()`.  It emits
`logger.warning` for isolated N/O centres whose bonded-H count falls outside
the expected range.  It never modifies the structure.

Run standalone:
```python
from molcrys_kit.analysis.disorder.diagnostics import check_valence_completeness
issues = check_valence_completeness(crystal, info)
for issue in issues:
    print(issue)
```

---

## Coordination Polyhedra Analysis

`molcrys_kit.analysis.packing_shell.find_polyhedra` is the single entry point
for first-shell A--B coordination analysis.  It must dispatch on the
``level`` argument:

* ``level="atom"`` (default, backward compatible) — match by chemical symbol
  on a flat `ase.Atoms`.  Use for purely atomic ionic crystals
  (Pb--I, Cs--Cl, M--X in inorganic perovskites).
* ``level="molecule"`` — match by single-fragment moiety string
  (`"N H4"`, `"Cl O4"`, `"C2 H10 N2"`, etc.) against each molecule's
  heavy-atom signature on a ``MolecularCrystal``; the search runs on
  molecule centroids with a configurable ``center_kind``
  (``"centroid"`` / ``"com"`` / ``"heavy_centroid"``).  Use for hybrid
  molecular crystals (ABX3 / ABX4 / A2BX5 hybrid perovskites).

Do **not** introduce a separate `find_molecular_polyhedra` (or any sibling
function) for the molecule case: that would create an asymmetric API
surface where users have to choose between two near-identical signatures.
All future entry points for new "what counts as a unit" semantics should
extend the same ``level=...`` enum.

The two paths share `detect_coordination_number` for the gap+enclosure CN
selection; they only differ in (a) how the central / ligand identity is
established and (b) how PBC images are enumerated (ASE neighbour list for
atom level, lattice-translation grid for molecule level).  Keep both
paths' return schemas distinct (`center_index` / `shell_indices` for atom
level; `center_molecule_index` / `shell_molecule_indices` /
`center_formula` / `shell_formula` for molecule level) so callers can
tell what kind of result they are handling.

### Radial cutoffs at atom vs. molecule level

The three radius kwargs (`cutoff`, `search_cutoff`, `hard_cutoff`) carry
*different meanings on the two levels by design*.  Atom-level users
historically wrote `find_polyhedra(atoms, "Pb", "I", cutoff=3)` to mean
"the Pb--I shell within 3 Å"; molecule-level callers using a perovskite
A--X12 query under the same semantics would unwittingly inflate their CN
to "every X within the 8 Å ball", which is rarely what they want.  The
split is:

* `level="atom"`: `cutoff` is the **hard radial cap** (forwarded to
  `detect_coordination_number` as `cutoff=`).  `search_cutoff` is the
  candidate radius for the ASE neighbour list.  `hard_cutoff` is not
  accepted here — passing it raises `ValueError` and the error message
  points users at `cutoff`.
* `level="molecule"`: `cutoff` is the **candidate search radius** that
  feeds `detect_coordination_number`'s gap+enclosure heuristic.
  `search_cutoff` is a non-deprecated synonym of `cutoff` (passing
  both raises `ValueError` because the resolution would be ambiguous).
  `hard_cutoff`, when set, is forwarded to `detect_coordination_number`
  as `cutoff=` and restores the historical "fill the ball" behaviour
  (`mode="cutoff"`).  If `hard_cutoff` is larger than the current search
  radius, bump the search radius to `hard_cutoff` automatically so the
  requested hard sphere cannot be silently truncated during candidate
  collection.

The record fields echo the kwargs faithfully:
`record["search_cutoff"]` always holds the search radius actually used;
`record["hard_cutoff"]` echoes the `hard_cutoff` kwarg (and is `None`
in pure gap+enclosure mode); `record["cutoff"]` echoes what
`detect_coordination_number` received — i.e. the hard cap value, or
`None` when no hard cap was applied.  Downstream code that wants to
inspect "was a hard cap applied?" should prefer `record["hard_cutoff"]`
over `record["cutoff"]` on the molecule level.

---

## Cluster Carving (`operations/cluster.py`)

`ClusterCarver` / `carve_cluster()` carve a finite, hydrogen-capped
`CrystalCluster` (a non-periodic `CrystalMolecule` subclass) out of a
periodic `MolecularCrystal` for downstream Gaussian / ORCA / Psi4 work.
Unlike the other entries in `operations/`, the return type is **not**
`MolecularCrystal` -- a cluster genuinely is not a crystal (it has
`pbc=False` and no lattice), so we expose the cluster directly.  Two
modes coexist: `bond_shells` is chemistry-aware (BFS that only cuts
non-ring single C-C bonds; `n_shells` counts cut-boundary layers
crossed, not raw hops) and is the production path; `rcut` is a
diagnostic radial cut that warns on non-C-C cuts.  Cap H atoms are
placed along the original (PBC-aware) bond vector at the element-keyed
X-H length pulled from `molcrys_kit.constants.config.BOND_LENGTHS` --
the same table `operations.add_hydrogens` consumes -- so an N-H cap is
1.01 Å and an O-H cap is 0.96 Å rather than a one-size-fits-all
1.09 Å.  Pass a positive `cap_distance` to force a uniform override
(retained for backwards compatibility and for callers that want the
old uniform-cap behaviour); image offsets are accumulated via
`neighbor_list("ijdD", ...)` bookkeeping.  Freeze sets
(`freeze_shell=0|1|2`) and a free-text `convention_reference` travel
in the JSON sidecar emitted by `write_xyz_with_freeze`.  The carver
does not infer charge / spin and does not write QM inputs -- those
belong to the downstream toolchain that consumes the sidecar.
Multi-atom nodes (paddle-wheels, M3 trimers, M6 nodes, ...) can be
collapsed into a single seed group via `seed_merge_radius`; the
default is `0.0` (no auto-grouping) so the parameter is meaningful
only when the caller opts in.

Internal note: the carver builds its own image-offset bond graph
because `CrystalMolecule._build_graph()` does not store offsets.  The
stored edge vector points from `min(i, j)` to `max(i, j)`; BFS code
must sign-flip when traversing in the reverse direction.  Ring
detection runs on a BFS-local subgraph (envelope: ~`6*(n_shells+1)+2`
hops) **and rejects rings larger than 8 atoms**; without that cap, the
topological macrocycles in a periodic framework graph would mark every
linker C-C as "ring-bonded" and the carver would never cut anything.
A second guardrail (`stop_at_non_seed_metals=True`, on by default)
cuts at any bond reaching a metal that is not in the current seed
group; this is necessary whenever two metal nodes are bridged through
non-C-C paths (M-X-X-M), since a pure C-C-only cut rule would let
BFS walk past every other node via those bridges and silently return
the whole framework.
