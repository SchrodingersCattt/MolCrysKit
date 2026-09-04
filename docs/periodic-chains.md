# Geometry-native periodic chains

The periodic-chain API is independent of `MolecularCrystal` and the chemistry
perception layer. A caller supplies a row-vector cell `r = f @ H`, PBC flags,
templates, ports, connection rules, chain count, and any closure constraints.
The builder preserves fragment geometry and returns a `PeriodicBundle`.

`PeriodicGraph.winding_cycles()` computes winding from integer edge image shifts;
it is never copied from a requested answer. Ordinary translation closure supports
zero, non-zero, and mixed-direction winding. `ScrewSpec` checks finite order and
cell compatibility and fails instead of expanding the cell implicitly.

The builder uses explicit port rules followed by distance checks. A closed
one-instance chain must expose distinct endpoint ports or a non-zero image
shift; a same-port zero-winding self-loop is rejected. It does not
infer bonds, valence, charge, oxidation state, reactivity, density, or a force
field, and it never calls MD to repair geometry. The collision index is periodic
and cell-list based; its image range is derived from the cell metric rather than
assuming 27 images.

The CLI accepts an explicit JSON request:

```text
mck build chain request.json --output bundle
mck build chain request.json --output structure.extxyz --format extxyz
mck validate-periodic-bundle bundle/structure.cif --json
```

The authoritative output is one structure file plus `structure.json`; the
default structure format is CIF. The bundle also supports POSCAR, XYZ, and
ExtXYZ via `--format` or a known output suffix. CIF/POSCAR/XYZ do not carry all
geometry-native arrays, so the sidecar remains authoritative for cell/PBC, atom,
chain, fragment, repeat, graph, hash, transform, closure/winding, tolerance, and
provenance metadata; bundle reads restore these fields before validation. It does
not store an atom edge table, model/type map, force-field data, or trajectories.
