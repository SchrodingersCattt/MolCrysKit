# Periodic net topology

This page is the contract for multi-net analysis. The connectivity it analyzes
is a [periodic quotient graph](periodic-graph.md). This package does not keep
a second graph type.

```python
import molcrys_kit as mck

site_graph = mck.build_periodic_site_graph(
    crystal,
    node_policy=mck.nets.NodePolicy("atom_sites"),
    edge_policy=mck.nets.EdgePolicy("supplied"),
)
candidates = mck.decompose_periodic_nets(site_graph)
report = mck.analyze_periodic_net_system(candidates)
payload = report.to_dict()
```

`payload["schema"]` is `mck.net_topology/v1`. The same object is printed by
`mck analyze net-topology --json`.

## What a status means

Topology, embedding, and duality are separate fields.

| Question | Proved | Not proved |
|---|---|---|
| Same abstract net | `isomorphic` with `explicit_isomorphism`, or `non_isomorphic` with `invariant_mismatch` or `exhaustive_search` | `undetermined` |
| Same embedding relation | `linked` with a linking-number witness, or `coexistence` with a separation or a finished search | `unverified`, `not_detected_within_bounds`, `not_assessed` |
| Dual nets | `dual` or `not_dual` from a validated tiling | `not_assessed` when no tiling was supplied, `undetermined` when the tiling or the comparison does not decide |

`invalid_or_indeterminate` means the inputs do not describe one realizable
pair: overlapping labels, contradictory edges, disorder that was not reduced
to one occupation, inexact coordinates, or a lattice mismatch. The report
does not fill a topology key in that case.

A coordination sequence, a degree sequence, or a matching fingerprint is never
an isomorphism proof. Those values can still prove two nets different. The
reason is then `coordination_sequence` or another invariant mismatch.

## Site graph

`build_periodic_site_graph` copies one crystal into the quotient graph.
Policies are predicates and explicit selections. They do not match an element,
a formula, a Wyckoff letter, or a fixed number of nets.

Node policies:

- `atom_sites`: one node per realized site.
- `molecular_units`: one node per molecule index. Equal formulas stay distinct.
- `coordination_centres`: the listed site ids, or every site with at least
  `minimum_incident` bonds of `bond_role`.
- `explicit_nodes`: the caller's groups of site ids. A repeated or unknown
  site is an error. Sites left out of every group are reported in
  `unassigned_site_ids` and are not nodes.

Edge policies:

- `supplied`: intramolecular bond records. A bond whose Cartesian vector is an
  integer lattice image of the two site positions uses that image. A nonzero
  site-image field that only unwraps a finite molecule is recorded and is not
  turned into a periodic edge. A vector that is not an integer image is
  rejected.
- `coordination_shell`: an edge between centres that share a ligand of the
  selected bond role. The bridge shift is the difference of the centre-to-ligand
  images.
- `contacts`: pairs of node positions within `cutoff_A`. `None` uses
  `DEFAULT_NEIGHBOR_CUTOFF` and the report stores the value that was used.
- `hydrogen_bonds`: hydrogen bonds between realized sites. The distance and
  angle criteria are stored on the edge record.
- `user_edges`: caller rows `(source_node, target_node, image_shift)`. An
  unknown node or a shift on a nonperiodic axis is rejected.

Pairs inside the cutoff are defining contacts only when the edge policy is
`contacts`. Otherwise they are `observed` and are not quotient edges. Pairs in
`(cutoff, cutoff + margin]` are `borderline` and are not edges. The margin is
`CONTACT_BORDERLINE_MARGIN_A`.

Each node record lists the site ids, molecule indices, asymmetric-unit
indices, and symbols that were contracted into it. Symbols are audit data.
They are not a merge key. `sites_not_in_nodes` lists realized sites that are
not in any node, so the map can be checked in both directions.

### Disorder

Disorder handling in this package does not call the replica solver.

- `as_refined` keeps every site. Two or more competing groups in one assembly
  make the crystal `indeterminate` (`competitors_retained`). Partial occupancy
  without a competitor is `conditional`.
- `major` keeps the group with the largest occupancy sum in each exclusive
  assembly. A tie is `indeterminate`. Two assemblies are not combined unless
  `independent_assemblies` is true; otherwise the status is
  `joint_realization_not_unique` and every site is kept for audit.
- `explicit` keeps the selected `(assembly, group)` rows plus group 0. Two
  selections for one assembly, or a missing selection, are `indeterminate`.

`indeterminate` disorder suppresses topology keys and pairwise linking.
`conditional` disorder keeps the classification and prefixes the reason with
`conditional:`. The classification is conditional on that occupation. It is
not rewritten into a different topology.

## Decomposition

`decompose_periodic_nets` has two modes.

Explicit mode checks the caller's `(node_id, net_id)` rows and optional
`(edge_id, net_id)` rows. Overlap, an unknown node, an uncovered node, or an
edge whose ends are not both in the named net makes the proposal `invalid`.
Invalid proposals are returned. They are not repaired into a topology.

Discovery mode returns:

1. `components`: one candidate per finite-quotient component. This is the
   primary proposal when it is admissible.
2. `entire`: all nodes in one candidate, when that partition is different.
3. `orbits`: groups that share asymmetric-unit indices, plus singleton groups
   for nodes that have no index. This proposal is omitted when no node has an
   index. Missing indices are not filled with Wyckoff letters. If the number
   of groups exceeds `discovery_partition_limit`, the proposal is omitted and
   `truncated` is true.

Duplicate partitions keep the earlier id. `primary_is_not_unique` is true when
two admissible proposals differ. The report then includes
`primary_partition_is_not_unique`. Discovery does not invent a single
partition to hide that.

Each candidate lists its nodes, induced edges, and cross edges. Cross edges
are contacts between candidates. They are not net edges and they are not a
link witness. Translated copies of one component stay in that candidate.
`multiplicity` is the number of those copies when the periodic rank equals
the number of periodic axes; it is `None` when there are infinitely many.
`translated_copies_are_one_topology` is true for every candidate this builder
emits.

`periodic_rank` is the maximum component rank. It is not the sum of the
component ranks, and it is not `quotient_cycle_rank`. A finite square has
periodic rank 0 and cycle rank 1. Two chains, one along each of two axes,
have periodic rank 1, two component ranks of 1, and cycle rank 2.

## Isomorphism

`compare_periodic_nets` decides whether two quotient graphs are the same
abstract net. The proof, when it finishes, is a witness that the canonical
keys agree. `verify_topology_witness` recomputes that decision.

For each connected component the steps are:

1. Exact barycentric coordinates on the pinned quotient Laplacian, unless
   every image shift is zero. A collision, a nonunique solution, or an
   inconsistent system stops the component as `undetermined`.
2. Translational symmetry of those coordinates. Extra translations fold the
   component. The folded graph is an abstract cell with periodic boundaries
   on every axis. Unused periodicity flags of the original crystal are not
   part of the topology key. Folding a supercell of a one-node net returns
   that one-node net.
3. The lattice generated by the image shifts is rewritten to a Hermite basis
   completed with the standard basis vectors that increase the rational rank.
   A lone shift `(2, 0, 0)` becomes `(1, 0, 0)` in that abstract cell.
   Multiplicity against the host lattice is a separate field and is not
   changed by this rewrite.
4. The tree-gauged voltage matrix is reduced to row-aligned Hermite form.
   The search enumerates labelings inside degree classes and permutations of
   parallel edges, zeroes one spanning-tree edge in each bundle, and keeps
   the lexicographically least form. Degree classes use the directed degree
   and the number of self-loops. They do not use the self-loop shifts, because
   a basis change rewrites those shifts.

The key records the original periodic rank. Quotient cycle rank is not in the
key and is not used as a substitute.

The search is complete when it finishes. A differing key is then
`non_isomorphic` with proof `exhaustive_search`. A matching key is
`isomorphic` with proof `explicit_isomorphism`. If the node count exceeds
`canonical_node_limit`, or the number of labelings and bundle orders exceeds
`backtrack_limit`, the status is `undetermined` and no partial key is kept.
`basis_entry_limit` and `lll_step_limit` are copied into the reported settings.
The proof does not consult them and does not search a bounded set of
`GL(3, Z)` matrices.

Rank and a completed coordination census are negative tests. They can return
`non_isomorphic` with proof `invariant_mismatch` even when the canonical
budget is zero. They cannot return `isomorphic`.

The same key is stable under relabeling, edge order, integer gauge, a
unimodular change of basis, and an equivalent supercell. A skew embedding
does not change the quotient. Equal keys with different embeddings are
allowed. Topology and catenation are independent.

## Linking

`assess_linking` compares two embedded graphs in one lattice. Coordinates
must be exact rationals within `fraction_residual_A` after
`limit_denominator`. A failure is `inexact_geometry`, not a link.

With `same_vertex_namespace=True`, shared node ids or the same undirected
edge are `invalid_or_indeterminate`. The report uses that mode because the
candidates come from one crystal. Independently built fixtures may reuse
labels; pass `same_vertex_namespace=False` for that case.

A verified link is an integer linking number of two disjoint cover cycles
after an orientation-preserving shear. The witness records the cycles, the
shear, and the number. Projection crossings, close distances, bounding boxes,
and a repeated abstract net are not witnesses.

The decision order is:

1. Shared vertices or edges.
2. A separating interval on an axis that neither curve winds around, or
   disjoint coplanar curves: `coexistence`.
3. A linking number with absolute value at least 1: `linked`.
4. A finished search whose linking numbers are all zero, or two graphs with
   no cover cycle and no reason to believe a longer cycle was missed:
   `coexistence`.
5. Otherwise `unverified` when the bounded regions meet, or
   `not_detected_within_bounds` when they do not.

Any nonzero image shift, or a quotient larger than `ring_length_limit`, means
a longer cover cycle may exist. Stopping there is not proof that the nets are
unlinked. The Hopf pair used in the tests is one ring and a second ring with
a single edge through its interior. Translating that second ring aside is
`coexistence`. Capping the ring length below 4 leaves the same pair
`unverified` or `not_detected_within_bounds`.

Brunnian links and links of more than two components are not given a class.
A missing embedding is `not_assessed`.

## Duality

`assess_duality` builds a dual only from a `PeriodicTiling`. Each face is an
ordered cover cycle of the primal, and each adjacency names the face crossed
from one tile to another. The dual has one node per tile and one edge per
adjacency. That dual is compared with the other net by `compare_periodic_nets`.

No tiling is supplied by discovery. The status is then `not_assessed`. An
invalid face is `undetermined`. A validated dual that is proved isomorphic to
the other net is `dual`. A proved mismatch is `not_dual`. An unfinished
isomorphism search is `undetermined`. Coordination numbers and coordination
sequences are not used.

## Command

```bash
mck analyze net-topology structure.cif --json
mck analyze net-topology --graph net.json --policy policy.json --json
```

Pass a crystal file or `--graph`, not both. `--nodes`, `--edges`, `--cutoff`,
`--disorder`, and `--independent-assemblies` override a crystal policy and are
rejected for a graph file. `--ring-length` and `--coordination-depth` override
the packaged limits. The policy file is JSON. A `.yml` or `.yaml` file is read
only when PyYAML is already installed. PyYAML is not a dependency.

The graph file has `nodes`, `edges`, optional `pbc`, and an optional
`embedding` with `positions_A` and `lattice_A`. An edge is
`[source, target, shift]` or `[id, source, target, shift]`.

Text output is one line per primary candidate: periodic rank, quotient cycle
rank, topology status, and proof. It does not print a prototype name.

## Limits that stay in the report

| Setting | Role |
|---|---|
| `coordination_depth` | Cover breadth-first depth. A truncated census is not compared as if it were complete. |
| `cover_state_limit` | State cap for that census. |
| `primitive_node_limit` | Largest quotient sent through barycentric folding. |
| `canonical_node_limit` | Largest quotient whose labeling is attempted. |
| `backtrack_limit` | Cap on labelings times parallel-edge orders. |
| `basis_entry_limit`, `lll_step_limit` | Stored only. Not used by the proof. |
| `ring_length_limit`, `ring_count_limit`, `ring_state_limit` | Cover-cycle search. |
| `link_pair_limit`, `link_image_limit` | Pair and image caps for linking. |
| `discovery_partition_limit` | Largest orbit partition that is returned. |
| `contact_borderline_margin_A` | Width of the borderline contact shell. |
| `fraction_residual_A`, `fraction_denominator_limit` | Exact-coordinate gate for linking. |

Packaged values live in `NET_TOPOLOGY_CONFIG`.

## Out of scope

This analysis does not assign database or prototype names, choose a unique
net when the evidence is ambiguous, discover a tiling, or draw the embedding.
Energies, force fields, and synthesis are out of scope. A three-dimensional
rank is not an identification. A search that stops at a cap is not a negative
proof.
