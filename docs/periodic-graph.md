# Periodic quotient graphs

`molcrys_kit.periodic` is the shared mathematical graph for periodic
connectivity. It is an immutable quotient multigraph plus adapters. Fragment
chains, chemical entities, and coordination-shell networks stay domain
objects. This page is the Stage 1 contract: exact image shifts, rank, and
translation bases. It does not name nets, prove graph isomorphism, detect
interpenetration or catenation, or construct a dual tiling.

```python
import molcrys_kit as mck

core_graph = mck.to_periodic_quotient_graph(existing_graph_or_entity)
rank = mck.periodic_rank(core_graph)
```

`adapt_periodic_graph` returns the same graph together with an optional
embedding and the domain provenance. `analyze_periodic_graph` returns
quotient components, cover families, the quotient cycle rank, and the
periodic rank.

## Inventory

The four existing representations are not interchangeable. Shared fields are
the node id, the edge id, the exact integer image shift, and the periodic
boundary flags. Everything else stays in provenance.

| Source | Shared fields copied into the quotient | Fields that stay outside the quotient |
|---|---|---|
| `PeriodicGraph` / `PeriodicEdge` in `molcrys_kit.structures.periodic_geometry` | node ids, directed edge rewritten to the stored orientation, `right_image_shift` | closure mode, rule id, closure flag, left and right ports, original node order |
| `PeriodicChemicalEntity` / `FiniteChemicalEntity` / `ChemicalBond` | atom ids, `atom2_image_shift` | element, bond kind, order, aromaticity, stereochemistry, evidence, declared rank and generators |
| `PolyhedronNetwork` / `SharedContact` / `PolyhedronComponent` | centre ids, `right_image_shift`, `pbc` | shells, shared vertices, contact relation, ambiguity, disorder policy, boundary diff |
| Caller-supplied nodes and edges | ids, endpoints, image shift, `pbc` | JSON-scalar node and edge attributes |

`analyze_periodic_topology` and `integer_translation_basis` remain the exact
rank and Hermite implementations. The new algorithms call them. They do not
reimplement the Hermite reduction. `PeriodicGraph.winding_cycles` is a
separate optimized path and is not called.

## Graph, embedding, and provenance

Three objects stay separate:

1. `PeriodicQuotientGraph` stores nodes, edges, and PBC flags. Equality is
   topology, ids, canonical shifts, and PBC. Domain metadata is absent.
2. `PeriodicEmbedding` stores Cartesian positions and, when a cell is known,
   fractional coordinates. A skew cell changes lengths and angles here. It
   does not change integer image shifts. Rank and components never read the
   embedding.
3. Provenance on `PeriodicGraphAdaptation` holds the domain records.
   `to_dict` is JSON-ready. Fragment provenance is a snapshot taken at
   adaptation time, so a later edit of the mutable `PeriodicGraph` does not
   change the adaptation. `fragment_graph_from_provenance` rebuilds that
   snapshot, including node order, ports, rules, and closure.

`PeriodicQuotientView` is a read-only `Protocol` with `nodes`, `edges`, and
`pbc`. Domain classes do not inherit it. Algorithms accept any object that
provides those attributes and normalize it to `PeriodicQuotientGraph`.

Row vectors match the rest of the library. An image shift `s` displaces a
point by `s @ L` when the rows of `L` are the cell vectors.

## Nodes, edges, and ids

A node is one translational orbit. `node_id` is a non-empty string and is
unique in the graph. An edge joins `source` to the image of `target`
displaced by `image_shift`. Endpoints must name nodes in the graph. Shifts
are exact integers of length 3. Booleans and floats are rejected. Parallel
edges are distinct ids with their own shifts. A self-image edge has
`source == target`. A nonzero self-image shift is a lattice winding.

`PeriodicQuotientGraph` normalizes orientation before storing an edge:

- Distinct endpoints are ordered by node id. The shift is rewritten so it
  still points at the stored target.
- A self-image shift is flipped so its first nonzero component is positive.
  `(2, 0, 0)` stays `(2, 0, 0)`. `(-2, 0, 0)` becomes `(2, 0, 0)` and still
  generates `2Z`, not `Z`. The sign choice is orientation only. It is not a
  greatest-common-divisor reduction.

`reversed_edge` swaps the endpoints, negates the shift, and keeps the edge
id. Building the graph from either orientation compares equal. Node and edge
storage order is sorted, so supplying the same ids in another order also
compares equal.

Missing caller edge ids are generated from the undirected endpoints, the
oriented shift, and an occurrence among identical input rows:

```text
edge:{left}->{right}@{h}_{k}_{l}#{occurrence}
```

Explicit ids are preserved. Reversing one edge or reordering distinct edges
does not change a generated id.

Adapter ids keep the original orientation in the id string, then the graph
stores the normalized edge under that same id:

- Fragment: `fragment:{left}->{right}@{h}_{k}_{l}/rule={rule}/closure={0|1}/ports={left},{right}#{occurrence}`
- Chemical: `bond:{atom1}->{atom2}@{h}_{k}_{l}/kind={kind}/order={order}/aromatic={0|1}/stereo={stereo}#{occurrence}`
- Polyhedron node: `centre:{molecule}:{realization}`. Contact:
  `contact:{left_mol}:{left_real}->{right_mol}:{right_real}@{h}_{k}_{l}#{occurrence}`

Supercell replication appends U+001F (information separator one) and the
image `i,j,k`. Node ids and edge ids that already contain U+001F are
rejected. String sort of centre ids is lexicographic, not numeric. The
translation basis does not depend on that sort.

## PBC flags

PBC is three real booleans on the quotient, default `(True, True, True)`.
Integers `0` and `1` are rejected. A nonzero shift component on a `False`
axis is rejected. There is no silent mask.

Fragment graphs and chemical entities do not store PBC. Their adapters
default to full periodicity, the same default as
`analyze_periodic_topology`, and accept `pbc=`. The polyhedron adapter uses
`PolyhedronNetwork.pbc`. Rank is unchanged when every component on a
non-periodic axis is already zero. Cover multiplicity does depend on how many
axes are periodic.

## Gauge and basis

A node-image gauge `g(v)` replaces the representative of node `v` by the
image `g(v)`. Before orientation is normalized again, an edge shift becomes

```text
s' = s + g(source) - g(target)
```

A self-image shift is invariant. A gauge that is the same integer on every
node is an origin change and does not alter shifts. A relative gauge can
change a bridge shift. It does not change the cycle-translation lattice.
Gauge components on a non-periodic axis are rejected. `apply_node_gauge`
also moves an embedding's fractional coordinates by `g(v)` when a lattice is
present.

`shift_embedding_origin` moves only the embedding. Fractional coordinates
decrease by the origin. The quotient is unchanged.
`replace_lattice` keeps fractional coordinates and installs a new row-vector
cell, which is the skew-cell operation. Image shifts stay on the graph.

Basis changes use the same convention as `LatticeBasisChange`:

```text
L_new = B @ L_old
s_new = s_old @ inv(B)
```

`B` is an integer 3×3 matrix. `apply_unimodular_basis` requires
`det B = ±1`. Node ids are unchanged. Cartesian embedding coordinates are
unchanged. Fractional coordinates and the stored lattice change with `B`.
Map a transformed shift back with `s_new @ B`.

`equivalent_supercell` accepts any invertible integer `B`, including
`|det| > 1`. Each parent node and edge becomes `|det B|` images.
Representatives are the integer vectors `t` with
`floor(t @ inv(B)) = 0`, so `t = representative + n @ B`. An edge that
leaves source image `rep` along the parent shift arrives at `rep + s`, which
splits into a target representative plus a supercell image shift.
`SupercellReplication.parent_shift` maps that supercell shift back with
`s @ B`.

What an equivalent supercell preserves:

- periodic rank of each full-dimensional component, and therefore the graph
  periodic rank when every component is considered;
- for every component whose rank equals the number of periodic axes, the
  sum of cover multiplicities.

Integer entries of the basis are written in the supercell. They are not the
parent integers. A lone parent self-image `(2, 0, 0)` under
`diag(2, 1, 1)` becomes two supercell components, each with supercell basis
`(1, 0, 0)`. `parent_shift((1, 0, 0))` is `(2, 0, 0)`. A primitive
`(1, 0, 0)` self-image under the same supercell stays one component. The
mapped Hermite vector alone does not tell those cases apart. The component
count does. The parent graph is not rewritten to a primitive generator.

## Algorithms

Finite-quotient components ignore image shifts. Self-loops do not connect
distinct nodes. `quotient_cycle_rank` is `edges - nodes + components`. A
zero-shift cycle and a lattice winding both increase it.

Infinite-cover connectivity uses closed-walk translations. A deterministic
depth-first spanning tree supplies one residual per cotree edge, including
zeros. Self-loops are never tree edges. The number of residuals equals the
component cycle rank.

Each component then has a `TranslationSubgroup`:

- `basis` is `integer_translation_basis` of those residuals. A lone
  `(2, 0, 0)` stays `(2, 0, 0)`. `(2, 0, 0)` together with `(3, 0, 0)`
  becomes `(1, 0, 0)`.
- `independent_generators` is the subset selected by
  `analyze_periodic_topology` on that component only. For the same pair the
  independent generator is `(2, 0, 0)`. The two descriptions have the same
  rank. They are not the same vectors.
- `rank` is that common dimension, an integer from 0 to 3. The algorithm
  raises if the Hermite rank and the topology rank disagree.

`periodic_rank` is the maximum component rank. It is not the rank of the
pooled translations. A chain along `a` and a separate chain along `b` have
periodic rank 1. `analyze_periodic_topology`, called on a whole disconnected
bond list, still pools those translations. Pass one component, or use
`periodic_rank`, when the per-component answer is required.

`multiplicity` is the number of infinite-cover components of one quotient
component. It is `None` when the rank is lower than the number of periodic
axes, because there are infinitely many copies. When the rank equals that
number, multiplicity is the absolute value of the determinant of the basis
restricted to the periodic coordinates. With no periodic axes, a rank-0
component has multiplicity 1. An empty graph has periodic rank 0, quotient
cycle rank 0, and cover multiplicity 0. `cover_multiplicity` sums the finite
component multiplicities and returns `None` if any component is infinite.

## Adapters

| Function | Input | Provenance kind |
|---|---|---|
| `adapt_fragment_graph` | `PeriodicGraph` | `fragment_graph` |
| `adapt_chemical_entity` | `FiniteChemicalEntity` or `PeriodicChemicalEntity` | `chemical_entity` |
| `adapt_polyhedron_network` | `PolyhedronNetwork` | `polyhedron_network` |
| `adapt_caller_graph` | node ids and edge rows | `caller` |
| `adapt_periodic_graph` | any of the above, a `PeriodicQuotientGraph`, or an existing adaptation | as above, or `quotient_graph` |
| `to_periodic_quotient_graph` | the same inputs as `adapt_periodic_graph` | discarded; use `adapt_periodic_graph` to keep it |

`to_periodic_quotient_graph` on an adaptation returns that adaptation's
graph. On a `PeriodicQuotientGraph` the embedding is not carried through
this function. Polymer and multicomponent records are not bond graphs.
`adapt_chemical_entity` rejects them. A partial chemical embedding, one that
omits an atom of the entity, is rejected rather than dropping coordinates.
`coordinate_system` must be `cartesian` or `fractional`.

Caller edges are `PeriodicQuotientEdge`, `(source, target, shift)`, or
`(edge_id, source, target, shift)`. `edge_attributes` aligns with the input
rows and may contain `None`. Attribute values are strings, finite numbers,
booleans, or null. Booleans are stored before integers, so `True` is not
recorded as `1`.

Declared `periodic_rank` and `translation_generators` on a
`PeriodicChemicalEntity` are copied and are not recomputed. The mathematical
rank is `periodic_rank` of the adapted graph. Those two numbers can differ
when the declared generators were not derived from the bonds.

## Migration

Existing constructors, `PeriodicGraph.winding_cycles`, chemistry models,
`polyhedron_network_from_shells`, renderer site and bond records, CLI
commands, and periodic-bundle JSON are unchanged. No type was renamed and no
call site is required to import `molcrys_kit.periodic`.

The chain sidecar graph object still has keys `closure`, `cycle_rank`,
`edges`, `nodes`, and `winding_cycles`. Each edge still has `closure`,
`left_node`, `left_port`, `right_image_shift`, `right_node`, `right_port`,
and `rule_id`. `mck build chain` and `mck validate-periodic-bundle` are the
same commands.

New root names, also available from `molcrys_kit.periodic`, are
`PeriodicQuotientGraph`, `adapt_periodic_graph`, `analyze_periodic_graph`,
`periodic_rank`, and `to_periodic_quotient_graph`. The subpackage
`molcrys_kit.periodic` exports the full surface listed in the
`mck.periodic` section of [API & Capabilities](api.md).

Large fragment-chain builds keep the disjoint-cycle fast path inside
`winding_cycles`. Adapting a graph afterwards is a separate call. Do not
replace a winding query in the builder with `analyze_periodic_graph`.

## Out of scope for this contract

Crystal-specific node and edge policies, subnet decomposition, net-name
recognition, periodic-graph isomorphism, interpenetration or catenation
witnesses, duality, and an `mck` net-topology command are a later stage.
They must use this graph rather than a second quotient implementation.
A 3-dimensional rank is not a topology identification.
