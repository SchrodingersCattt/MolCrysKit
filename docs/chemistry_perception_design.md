# Bond-order perception design

This document defines the chemistry ownership and uncertainty contract used by
the bond-order engine. `molcrys_kit.chemistry` is the only authority for
formal bond order, covalent/coordination/metallic kind, formal charge,
aromatic flags, and resonance alternatives. Structure modules own connectivity,
coordinates, periodic image shifts, disorder and source provenance. Geometry is
evidence used to rank compatible states; a distance alone never asserts a
formal order.

The pure entry point is:

```python
from molcrys_kit.chemistry import (
    BondOrderOptions, FiniteChemicalEntity, infer_bond_orders,
)

result = infer_bond_orders(
    entity,
    total_charge=None,
    options=BondOrderOptions(strict=False, max_search_states=100_000),
)
```

Inputs are immutable entities with stable atom IDs. A source order or aromatic
flag is fixed evidence and is never overwritten; contradictory source evidence
is retained as a diagnostic. A user-confirmed value has higher precedence than
an explicit file value, which has higher precedence than inference. The legacy
`infer_chemistry` function is an attaching adapter and preserves its historical
signature, while delegating to the same solver.

The solver first removes fixed edges from the residual problem, computes atom
valence domains, and propagates lower/upper residual bounds. Independent
components are solved separately when their charge constraints permit it.
Branching uses the smallest remaining domain and an incremental geometry score.
`max_search_states` bounds work and is reported as `TRUNCATED` when reached;
`max_candidates` only controls retained alternatives and cannot change the
best assignment. No fabricated single-bond result is returned after truncation.
Unknown elements, unsupported metal oxidation states, unresolved charge, and
chemically competitive assignments are represented as provisional or
indeterminate evidence.

An explicit ``total_charge`` is enforced for every candidate even when one or
more atoms have no inferred formal charge.  A missing atomic charge does not
skip the check and is not treated as zero.  The candidate is kept only when
every atom has a charge and the sum equals the requested total.  If no such
candidate remains, the result is ``INDETERMINATE``, strict mode raises, and
``diagnostics["total_charge_unsatisfied"]`` is true.  The constraint is not
applied when both the argument and the entity net charge are omitted.

Periodic vectors use the row-vector convention
`r_j + image_shift @ cell_A - r_i`, including triclinic cells. A periodic
self-edge contributes twice to the valence of its sole atom. Coordinates and
topology are never mutated by pure perception. Strict legacy calls restore all
existing chemistry and molecule attachments before raising on unresolved
results.

Aromaticity is its own graph authority, separate from numeric `1.5` Kekulé
averages. A chordless 5-, 6-, or 7-membered C/N/O/S ring is aromatic when its
pi-electron count is 4n+2: carbon and pyridine nitrogen contribute one, a
positively charged carbon contributes an empty p orbital, pyrrole nitrogen
and furan/pyrrole/thiophene heteroatoms contribute a lone pair, and a
saturated carbon (four sigma neighbors) blocks the ring. Fused rings share
the bonds that lie on any accepted cycle; a biphenyl bridge does not.
User-confirmed non-aromatic evidence locks the flag off. Ring planarity and
lengths only rank Kekulé forms.

When hydrogen metadata are absent, the ring-level geometry check also guards
against mistaking an alternating ring for an aromatic one. For each element
pair represented by at least four ring edges, it divides each edge length by
that pair's configured single-bond reference, sorts the normalized lengths,
and considers the largest internal gap. The split is eligible only when at
least two edges fall on each side; the pair contributes alternation evidence
when that gap is at least `AROMATIC_RING_BOND_ALTERNATION_MAX`. A ring is
treated as alternating when any represented pair passes this test. Unknown
element-pair references are skipped, and geometry abstains when fewer than
half of the ring edges have a known reference. This largest-gap, two-sided
rule preserves 2+4 patterns such as 1,4-cyclohexadiene that a median split
could hide.

Source precedence is user-confirmed, then explicit file evidence (CIF bond
type, line notation, IUPAC), then inference. `S`/`D`/`T` lock orders 1/2/3.
`A` and `delocalised` set the aromatic flag and leave the numeric order to
the solver. Contradictory evidence at the winning rank is flagged and does
not overwrite either value.

Variable bonds are decomposed into connected components. Forests are solved
by tree dynamic programming with the same valence rules as the bounded
search. Cyclic components stay on that search. A requested total charge is
checked after the pieces are merged. Independent optima that miss it are
recombined from each component's charge frontier. The whole graph is not
searched again. Geometry scores are squared bond-length residuals and
never create an order by themselves.

### CSD calibration of ring geometry

The ring resolver has a separate calibration hand-off for laboratories that
have access to the licensed Cambridge Structural Database (CSD) Python API.
`molcrys_kit.chemistry.aromaticity_calibration` consumes a JSON or JSONL export
so the production package and its tests do not need the CCDC dependency. Each
source-confirmed single-bond edge is exported as, for example,
`{"bond_class": "single", "pair": ["C", "C"], "length_A": 1.536}`.
Each ring summary carries `ring_class` (`aromatic`, `saturated`, or
`conjugated_non_aromatic`) and the three resolver metrics: `planarity`,
`bond_metric`, and `alternation_gap`. It also carries a `references_A` object
with the single-bond reference length for every supported element pair. The
calibrator checks that this object matches the medians of the source-confirmed
single-bond rows before accepting any ring metric; this keeps promoted runtime
references and exported metrics on the same scale. A legacy `non_aromatic`
class is accepted as an alias for `conjugated_non_aromatic`, but snapshots use
the explicit classes. Reference medians are compared with a `1e-4` Å absolute
tolerance, matching the precision of reported CSD bond lengths.

The calibrator reports the number of observations for every C/N/O/S element
pair, quantile summaries, and explicit uncertainty bands between the relevant
tails. Planarity separation is calibrated from `aromatic` versus `saturated`;
`conjugated_non_aromatic` planarity remains a reported distribution only and
does not set the runtime planarity band. Bond-metric separation is calibrated
from `aromatic` versus `saturated`, while alternation separation is calibrated
from `aromatic` versus `conjugated_non_aromatic`; saturated rings therefore do
not make an alternation sample look ambiguous. It refuses to emit a snapshot
when any requested pair or valid metric lacks the minimum coverage, when
unknown rows are encountered without being counted, or when any tails overlap.
This keeps a small or biased CSD sample from silently changing chemistry
perception. A licensed CSD run can be reproduced with:

```bash
python scripts/calibrate_aromaticity.py rings.jsonl aromaticity_geometry.json \
  --source csd-2024-sample.json
```

The checked-in PR #179 thresholds remain the conservative runtime fallback;
the generated snapshot is a review artifact and should only replace those
defaults after its coverage and validation fixtures have been reviewed.
Call `runtime_overrides(snapshot)` to obtain the exact
`CHEMISTRY_PERCEPTION_CONFIG` keys for that reviewed promotion; no import-time
environment variable can silently change a running application's thresholds.

Functional acceptance for issue #174 is this engine: aromaticity, source
precedence, component decomposition, tree dynamic programming, geometry
ranking, and the section 4.1 micro-family tests. The benchmark package freezes
the independent micro manifest and records unknowns, timeouts, and missing
comparisons. Open Babel 3.2.1 paired runs, the section 5 superiority gates,
latency, throughput, RSS, QM9, and the curated external and CIF corpora are
issue #176. They stay visible on the scorecard and do not change
`score --strict-gates`. Functional strict mode also requires complete selected
MolCrysKit coverage and at least one known-truth order with no incorrect known
orders; abstention or omitted rows cannot pass it.

```mermaid
flowchart LR
  A[stable graph + source evidence] --> B[atom/edge domains]
  B --> C[propagation]
  C --> D[component decomposition]
  D --> E[bounded optimization]
  E --> F[aromaticity/resonance canonicalization]
  F --> G[immutable result + diagnostics]
```
