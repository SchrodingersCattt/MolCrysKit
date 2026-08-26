# Bond inference

Use this API when a trajectory renderer or numerical pipeline needs bond pairs
without a NetworkX graph or per-bond Python records. Arrays are the default
representation for positions, atomic numbers, pairs, vectors, and distances.

```python
from molcrys_kit.structures import VerletBondTracker

tracker = VerletBondTracker(skin=0.5)
bonds = tracker.update(
    positions,
    atomic_numbers,
    cell=cell,
    pbc=pbc,
)
# bonds.pairs: int32 (M, 2)
# bonds.vectors: float32 minimum-image vectors
# bonds.distances: float32 (M,)
```

The tracker retains the candidate-pair array and reference positions between
frames. Call `tracker.clear()` to release that memory when reuse is no longer
needed.

The API uses MolCrysKit atomic radii and metal/non-metal threshold factors.
`VerletBondTracker` evaluates distances every frame and rebuilds its candidate
list when any displacement exceeds half the skin, or when the cell or PBC
changes. Candidate distances are evaluated on every frame, so both bond
formation and bond breaking are detected without duplicating chemistry rules
in a caller.

`infer_bond_pairs` is the stateless one-frame entry point.
`build_bond_candidates` plus `evaluate_bond_candidates` exposes the two
stages for callers that manage their own trajectory state. Axis-aligned
orthogonal cells use a periodic cKDTree fast path; rotated or triclinic cells
use the ASE neighbour-list path.
