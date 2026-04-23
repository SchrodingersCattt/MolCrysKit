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

After MWIS, `_merge_chemical_motifs` reconstructs isolated XH_n motifs
(NH4+, H2O) from the remaining singletons.  For nitrogen centres, soft
conflicts (`valence_geometry`, `implicit_sp`, `geometric`) are ignored
(`_MOTIF_REJECT_SOFT["N"] = False`) because the SP disorder typically adds
soft edges between *every* H pair.

**One-per-asym_id guard** (added in feat/sp-nh4-implicit-hardening):
When the number of distinct `asym_id` values among candidate H atoms is
≥ `max_H` (4 for N), the greedy selection in `_select_motif_hydrogens`
enforces at most one pick per `asym_id`.  This prevents selecting multiple
copies of the same SHELX disorder position (which are nearly co-linear and
would monopolise the 4 available slots before other H sites can be chosen).
The guard is disabled when fewer distinct asym_ids exist (e.g. DAP-4 NH4+
with only 2 H labels but needing 4 H), preserving the original angle-only
heuristic for those cases.

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
- The corresponding CIF must be copied to `examples/`.
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
