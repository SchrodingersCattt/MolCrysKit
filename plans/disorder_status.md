# Disorder Module — Current Status & Refactor Plan

This document supersedes the previous five SP/disorder plan files (archived
at `/tmp/molcrys_archive/plans/`).  Read this first.

## 1. Single source of truth

The behavioural contract for disorder resolution lives in
`tests/unit/test_disorder_regression.py`.  It is parametrised over every
hand-validated CIF in `examples/` and asserts:

* `n_atoms == expected_atoms`
* number of over-coordinated atoms == `expected_overcoord` (typically 0)

`scripts/regression_quick.py` and `scripts/regression_all.py` are
print-only and remain useful for ad-hoc inspection, but they are not
guardrails — pytest is.

## 2. State on `main` (locked baseline)

Twelve CIFs are green and cannot regress without breaking the suite:

```
NatComm-1, PAP-HM4, DAP-4, DAC-4,
anhydrousCaffeine, anhydrousCaffeine2,
ZIF-4, TILPEN, 1-HTP, MAF-4, DAN-2, PAP-H4
```

Recently fixed (formerly `xfail`):

| CIF      | symptom (resolved)                     | fix landed in              |
|----------|----------------------------------------|----------------------------|
| 368K     | "4 over-coord" was just isolated Br⁻   | regression suite update (`ISOLATED_OK`) |
| DAI-X1   | resolved to 100, expected 112          | `graph._add_conformer_conflicts` / `_add_explicit_conflicts` now branch on `sym_op_index` for negative PARTs |
| ZIF-8    | 316 atoms but 40 "over-coord"          | the 40 are solvent-water O ghosts (CIF labels `O*S`), not bugs; declared as `expected_defects=40` |
| PAP-M5   | 296 atoms but 8 "over-coord"           | the 8 are isolated Ag⁺ counter-ions; added Ag (and other free-cation metals) to `ISOLATED_OK` |
| DAP-O4   | timed out (>60 s)                      | parser dedup loop in `cif.scan_cif_disorder` was O(N²); now O(N) per-element vectorised → DAP-O4 finishes in ~66 s, atom count 344 matches formula |
| PAP-4    | timed out (>60 s)                      | same parser fix → finishes in ~38 s; **but** uncovered a separate atom-count issue (see worklist) |

Still `xfail`'d — open worklist:

| CIF      | symptom                                | suspected area                                          |
|----------|----------------------------------------|---------------------------------------------------------|
| PAP-4    | resolves to 280 atoms, expected 304    | NH4⁺ in the highly-disordered ammonium fragment (24 orientational alternatives, occ ≈ 1/24) is reconstructed as bare NH; the H atoms from any one orientation get knocked out by `symmetry_clash` against the H atoms of every other orientation.  Needs orientation-aware motif merging in `solver._merge_chemical_motifs`. |

When a fix lands, drop the `xfail_reason` in the test case and watch the
suite turn green.

## 3. Why the modules are messy

### 3.1 `molcrys_kit/analysis/disorder/graph.py` (1454 lines on `main`)

* Six `_add_*_conflicts` passes share a hard-coded priority list inside
  `_add_geometric_conflicts` (the `high_priority_conflicts` literal).
  The semantics of "which type of conflict overrides which" is not
  documented anywhere except in this list.
* Special-position handling spans many `_sp_*` helpers added across at
  least three iterative fix rounds; some are now dead, some duplicate
  each other (see archived plans).
* `_decompose_cliques` mixes geometric search and conflict emission.

### 3.2 `molcrys_kit/analysis/disorder/solver.py` (795 lines)

* `_identify_atom_groups` does at least four merging steps in one
  function (`Step 1`, `Step 1.5` SP-valid composites, `Step 2`
  chemical-motif merging, then post-hoc `_remove_orphan_hydrogens`).
* `_remove_orphan_hydrogens` is a "patch after the fact" step that
  silently changes the random-mode atom count for MAF-4 etc.  The
  random-mode design doc proposed killing this patch; that work is not
  yet done.

### 3.3 Local cruft (not on `main`)

* The local `e091001` commit on `fix/disorder` ("constraints geometry in
  disorder solving") regressed PAP-HM4 (176→204), DAC-4 (39→115),
  TILPEN (84→102), DAN-2 (35→59) while attempting to fix PAP-4.  It is
  archived as
  `/tmp/molcrys_archive/0001-constraints-geometry-in-disorder-solving.patch`
  for cherry-picking ideas.
* An additional uncommitted "Bug D" attempt for DAI-X1 is at
  `/tmp/molcrys_archive/stash_disorder_dai_x1_attempt.patch`.
* 18 `scripts/debug_daix1*.py` / `debug_pap4*.py` are archived at
  `/tmp/molcrys_archive/debug_scripts/`.

## 4. Refactor plan

Sequenced so each step keeps the regression suite green.

### Step 1 — fix the easy `xfail`s while the structure is intact

Tackle in this order; each unblocks the next:

1. ~~**368K** — over-coord without count change → bond-pruning bug only.~~
   Done; the "over-coord" atoms were isolated halide counter-ions, not a
   bug.  Suite was teaching the wrong thing.
2. ~~**DAI-X1** — count is 100/112; investigate logical-alternative edges
   between dg=-1 / dg=-2 sym-mates.~~  Done; the assembly label and
   centroid distance are not enough to discriminate alternatives among
   negative PARTs (the assembly is shared across all symmetry copies).
   Both `_add_conformer_conflicts` and `_add_explicit_conflicts` now
   require negative-PART pairs to share `sym_op_index` before flagging a
   `logical_alternative` / `explicit` conflict.
3. ~~**ZIF-8 / PAP-M5** — both are over-coord on Zn/N partial-occ; same
   class of bug.~~  Done; both were false alarms — solvent-water orphans
   (ZIF-8) and isolated Ag⁺ counter-ions (PAP-M5).  Test suite updated.
4. ~~**PAP-4 / DAP-O4** — combinatorial blow-up; needs cap-and-prune in
   `_decompose_cliques`.~~  Done; the bottleneck was actually in the CIF
   parser, not the solver: `scan_cif_disorder` was running a Python
   double loop with one PBC distance call per (new image, existing image)
   pair.  Replaced with a per-element vectorised `numpy` minimum-image
   distance (precomputed 27-shift table).  PAP-4 went 223 s → 38 s,
   DAP-O4 went 60 s timeout → 66 s pass.

### Step 1b — leftover atom-count issue in PAP-4

PAP-4 now finishes quickly but resolves only 280 of the expected 304
atoms.  The 24 missing atoms are all H from highly-orientationally-
disordered NH4⁺ (formula moiety `0.04(H95.84 N24)`, occ ≈ 1/24 per H).
The solver picks 8 N atoms but only 1 H per N because the H atoms from
any one NH4⁺ orientation are in `symmetry_clash` with the H atoms of
every other orientation at the same N.

The fix is at the motif level, not the conflict-graph level: the
solver should treat (N, 4 × H) at each NH4⁺ orientation as a single
indivisible 5-atom rigid body before MWIS, so that selecting the N
implies committing to one orientation and brings 4 H along with it.
This is in scope for the `solver.py` refactor (Step 3 below).

### Step 2 — split `graph.py` into a passes module

Target layout:

```
molcrys_kit/analysis/disorder/graph/
    __init__.py         # re-exports DisorderGraphBuilder
    builder.py          # high-level orchestration + node setup
    metrics.py          # _precompute_metrics, distance utilities
    passes/
        __init__.py
        conformers.py
        explicit.py
        geometric.py
        implicit_sp.py
        valence.py
    priority.py         # the conflict-type ordering, was a list literal
```

Re-export the public name from the package `__init__.py` so existing
imports (`from molcrys_kit.analysis.disorder.graph import
DisorderGraphBuilder`) keep working.

### Step 3 — split `solver.py` along the same lines

```
molcrys_kit/analysis/disorder/solver/
    __init__.py
    grouping.py         # was _identify_atom_groups + Step 1.5
    motif_merge.py      # was _merge_chemical_motifs
    mwis.py             # _max_weight_independent_set_by_groups
    reconstruct.py      # crystal assembly
```

Drop `_remove_orphan_hydrogens`; the design proposal in the archived
`valid_solution_space_sampling.md` shows how to make every raw MWIS
solution chemically valid up front.

### Step 4 — collapse `__init__.py` re-export surface

`molcrys_kit/analysis/disorder/info.py` is currently a one-liner
re-export; either inline the dataclass here or keep it as the canonical
home (the dataclass currently lives in `molcrys_kit/io/cif.py`, which is
backwards).

## 5. House rules going forward

* No new `scripts/debug_*.py`.  If a debug session reveals a
  generally-useful inspection routine, fold it into
  `scripts/visualize_disorder_graphs.py` or a `tests/dev/` helper.
* Every behavioural change to the disorder solver/graph must update or
  add a test in `tests/unit/test_disorder_regression.py`.  No more
  silent regressions like `e091001`.
* Plans live in this single file; archive completed/superseded ones
  outside the repo (or under `plans/archive/` if they have unique
  insights worth preserving).
