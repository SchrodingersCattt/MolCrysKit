# Array bond inference

Use the array API when a trajectory renderer or numerical pipeline needs bond
pairs without a NetworkX graph or per-bond Python records.

```python
from molcrys_kit.analysis import VerletBondTracker

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

The API uses MolCrysKit atomic radii and metal/non-metal threshold factors.
`VerletBondTracker` evaluates distances every frame and rebuilds its candidate
list when any displacement exceeds half the skin, or when the cell or PBC
changes. It therefore detects both bond formation and bond breaking without
duplicating chemistry rules in a caller.

`infer_bond_pairs` is the stateless one-frame entry point.
`build_bond_candidates` plus `evaluate_bond_candidates` exposes the two
stages for callers that manage their own trajectory state. Orthogonal cells use
a periodic cKDTree fast path; triclinic cells use the ASE neighbour-list path.

