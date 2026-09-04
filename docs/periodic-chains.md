# Geometry-native periodic chains

The periodic-chain API is independent of `MolecularCrystal` and the chemistry
perception layer. A caller supplies a row-vector cell `r = f @ H`, PBC flags,
templates, ports, connection rules, chain count, and any closure constraints.
The builder preserves fragment geometry and returns a `PeriodicBundle`.

`PeriodicGraph.winding_cycles()` computes winding from integer edge image shifts;
it is never copied from a requested answer. Ordinary translation closure supports
zero, non-zero, and mixed-direction winding. `ScrewSpec` checks finite order and
cell compatibility and fails instead of expanding the cell implicitly.

The builder uses explicit port rules followed by distance checks. It does not
infer bonds, valence, charge, oxidation state, reactivity, density, or a force
field, and it never calls MD to repair geometry. The collision index is periodic
and cell-list based; its image range is derived from the cell metric rather than
assuming 27 images.

The CLI accepts an explicit JSON request:

```text
mck build chain request.json --output bundle
mck validate-periodic-bundle bundle/structure.extxyz --json
```

The authoritative output is `structure.extxyz` plus `structure.json`. ExtXYZ
stores symbols, cell/PBC, and atom/chain/fragment/repeat arrays. The sidecar
stores hashes, transforms, ports, the port-level graph, image shifts,
closure/winding, tolerances, and validation/provenance. It does not store an
atom edge table, model/type map, force-field data, or trajectories.
