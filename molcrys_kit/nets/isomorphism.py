"""Proved equivalence of periodic nets.

Two quotient graphs describe the same abstract net when their connected covers
are isomorphic by a node bijection, an integer gauge, and a unimodular change
of the translation lattice, after supercell copies have been folded. The proof
is the Hermite form of one tree-gauged voltage matrix. Coordination sequences
and degrees are computed and may prove two nets different. They never prove
two nets the same.

The canonical search is finite. If the quotient is larger than the node cap,
or the labeling budget is exhausted, the status is ``undetermined``.

A disconnected quotient joins one key per component. Those keys are sorted
before they are joined, so the system key does not follow source node ids.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction

from molcrys_kit.chemistry.topology import integer_translation_basis
from molcrys_kit.periodic._lattice import ImageShift, add, neg, sub
from molcrys_kit.periodic.algorithms import analyze_periodic_graph
from molcrys_kit.periodic.graph import (
    PeriodicQuotientEdge,
    PeriodicQuotientGraph,
    PeriodicQuotientNode,
)

from ._exact import (
    F3,
    add_f,
    as_integer_shift,
    fraction_vector,
    hermite_form_shifts,
    integer_matrix_inverse,
    matvec_f,
    mod1_vector,
    solve_pinned_system,
    sub_f,
)
from .invariants import coordination_census, degree_sequence, directed_adjacency
from .limits import NetAnalysisLimits


@dataclass(frozen=True)
class ComponentTopology:
    """Topology of one finite-quotient component."""

    status: str
    reason: str
    canonical_key: str | None
    periodic_rank: int
    quotient_cycle_rank: int
    multiplicity: int | None
    node_ids: tuple[str, ...]
    primitive_nodes: int | None
    primitive_edges: int | None
    primitive_degrees: tuple[int, ...] | None
    coordination_census: tuple[tuple[tuple[int, ...], int], ...] | None
    witness: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "canonical_key": self.canonical_key,
            "periodic_rank": self.periodic_rank,
            "quotient_cycle_rank": self.quotient_cycle_rank,
            "multiplicity": self.multiplicity,
            "node_ids": list(self.node_ids),
            "primitive_nodes": self.primitive_nodes,
            "primitive_edges": self.primitive_edges,
            "primitive_degrees": (
                None if self.primitive_degrees is None else list(self.primitive_degrees)
            ),
            "coordination_census": _census_to_json(self.coordination_census),
            "witness": self.witness,
        }


@dataclass(frozen=True)
class TopologyCharacterization:
    """Canonical form of every quotient component, or an explicit failure."""

    status: str
    reason: str
    canonical_key: str | None
    periodic_rank: int
    quotient_cycle_rank: int
    components: tuple[ComponentTopology, ...]
    coordination_census: tuple[tuple[tuple[int, ...], int], ...] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "canonical_key": self.canonical_key,
            "periodic_rank": self.periodic_rank,
            "quotient_cycle_rank": self.quotient_cycle_rank,
            "components": [component.to_dict() for component in self.components],
            "coordination_census": _census_to_json(self.coordination_census),
        }


@dataclass(frozen=True)
class NetEquivalence:
    """Isomorphism decision for two periodic graphs.

    ``proof`` is ``explicit_isomorphism`` only when both canonical forms were
    finished and agree. ``invariant_mismatch`` is a proof that the nets differ.
    ``exhaustive_search`` is a finished canonical search whose keys differ.
    ``none`` means the search did not decide.
    """

    status: str
    proof: str
    reason: str
    witness: dict[str, object] | None
    left: TopologyCharacterization
    right: TopologyCharacterization

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "proof": self.proof,
            "reason": self.reason,
            "witness": self.witness,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


def characterize_periodic_net(
    graph: PeriodicQuotientGraph,
    limits: NetAnalysisLimits | None = None,
) -> TopologyCharacterization:
    """Return a canonical key when the bounded exact search finishes."""
    limits = limits or NetAnalysisLimits.from_config()
    analysis = analyze_periodic_graph(graph)
    components: list[ComponentTopology] = []
    if not graph.nodes:
        empty = ComponentTopology(
            status="identified",
            reason="empty_graph",
            canonical_key="n0|r0|",
            periodic_rank=0,
            quotient_cycle_rank=0,
            multiplicity=0,
            node_ids=(),
            primitive_nodes=0,
            primitive_edges=0,
            primitive_degrees=(),
            coordination_census=(),
            witness={"kind": "empty"},
        )
        return TopologyCharacterization(
            "identified",
            "empty_graph",
            empty.canonical_key,
            0,
            0,
            (empty,),
            (),
        )
    for component, family in zip(
        analysis.quotient_components, analysis.cover_families, strict=True
    ):
        subgraph = _subgraph(graph, component.node_ids, component.edge_ids)
        components.append(
            _characterize_component(
                subgraph,
                periodic_rank=family.subgroup.rank,
                cycle_rank=component.cycle_rank,
                multiplicity=family.subgroup.multiplicity,
                limits=limits,
            )
        )
    return _assemble(
        components, analysis.periodic_rank, analysis.quotient_cycle_rank, graph, limits
    )


def compare_periodic_nets(
    left: PeriodicQuotientGraph,
    right: PeriodicQuotientGraph,
    limits: NetAnalysisLimits | None = None,
) -> NetEquivalence:
    """Decide whether two periodic graphs are the same abstract net."""
    limits = limits or NetAnalysisLimits.from_config()
    left_record = characterize_periodic_net(left, limits)
    right_record = characterize_periodic_net(right, limits)
    mismatch = _invariant_mismatch(left_record, right_record)
    if mismatch is not None:
        return NetEquivalence(
            "non_isomorphic",
            "invariant_mismatch",
            mismatch,
            None,
            left_record,
            right_record,
        )
    if (
        left_record.status == "identified"
        and right_record.status == "identified"
        and left_record.canonical_key == right_record.canonical_key
    ):
        return NetEquivalence(
            "isomorphic",
            "explicit_isomorphism",
            "canonical_voltage_graphs_agree",
            {
                "kind": "canonical_form",
                "canonical_key": left_record.canonical_key,
                "left": [component.witness for component in left_record.components],
                "right": [component.witness for component in right_record.components],
            },
            left_record,
            right_record,
        )
    if left_record.status == "identified" and right_record.status == "identified":
        return NetEquivalence(
            "non_isomorphic",
            "exhaustive_search",
            "canonical_voltage_graphs_differ",
            {
                "kind": "canonical_form",
                "left_key": left_record.canonical_key,
                "right_key": right_record.canonical_key,
            },
            left_record,
            right_record,
        )
    reason = (
        left_record.reason
        if left_record.status != "identified"
        else right_record.reason
    )
    return NetEquivalence(
        "undetermined",
        "none",
        reason,
        None,
        left_record,
        right_record,
    )


def verify_topology_witness(
    left: PeriodicQuotientGraph,
    right: PeriodicQuotientGraph,
    equivalence: NetEquivalence,
    limits: NetAnalysisLimits | None = None,
) -> bool:
    """Recompute the decision and require the same status, proof, and key."""
    recomputed = compare_periodic_nets(left, right, limits)
    if recomputed.status != equivalence.status or recomputed.proof != equivalence.proof:
        return False
    return recomputed.left.canonical_key == equivalence.left.canonical_key and (
        recomputed.right.canonical_key == equivalence.right.canonical_key
    )


def _assemble(
    components: list[ComponentTopology],
    periodic_rank: int,
    cycle_rank: int,
    graph: PeriodicQuotientGraph,
    limits: NetAnalysisLimits,
) -> TopologyCharacterization:
    census = coordination_census(graph, limits)
    if any(component.status != "identified" for component in components):
        failed = next(
            component for component in components if component.status != "identified"
        )
        return TopologyCharacterization(
            "undetermined",
            failed.reason,
            None,
            periodic_rank,
            cycle_rank,
            tuple(components),
            census,
        )
    # Sort by each component key. Source-node order changes under a relabel
    # when the components are different, and that order is not an invariant.
    ordered = tuple(
        sorted(components, key=lambda component: component.canonical_key or "")
    )
    key = "||".join(component.canonical_key or "" for component in ordered)
    return TopologyCharacterization(
        "identified",
        "canonical_form",
        key,
        periodic_rank,
        cycle_rank,
        ordered,
        census,
    )


def _invariant_mismatch(
    left: TopologyCharacterization,
    right: TopologyCharacterization,
) -> str | None:
    if left.periodic_rank != right.periodic_rank:
        return "periodic_rank"
    left_ranks = tuple(sorted(component.periodic_rank for component in left.components))
    right_ranks = tuple(
        sorted(component.periodic_rank for component in right.components)
    )
    if left_ranks != right_ranks:
        return "component_periodic_ranks"
    if (
        left.coordination_census is not None
        and right.coordination_census is not None
        and left.coordination_census != right.coordination_census
    ):
        return "coordination_sequence"
    if all(
        component.primitive_nodes is not None for component in left.components
    ) and all(component.primitive_nodes is not None for component in right.components):
        if _primitive_signature(left) != _primitive_signature(right):
            return "primitive_quotient"
    return None


def _primitive_signature(
    record: TopologyCharacterization,
) -> tuple[tuple[int, int, tuple[int, ...], int], ...]:
    rows = []
    for component in record.components:
        rows.append(
            (
                int(component.primitive_nodes or 0),
                int(component.primitive_edges or 0),
                component.primitive_degrees or (),
                component.periodic_rank,
            )
        )
    return tuple(sorted(rows))


def _characterize_component(
    graph: PeriodicQuotientGraph,
    *,
    periodic_rank: int,
    cycle_rank: int,
    multiplicity: int | None,
    limits: NetAnalysisLimits,
) -> ComponentTopology:
    base = dict(
        periodic_rank=periodic_rank,
        quotient_cycle_rank=cycle_rank,
        multiplicity=multiplicity,
        node_ids=graph.node_ids,
        coordination_census=coordination_census(graph, limits),
    )
    if len(graph.nodes) > limits.primitive_node_limit:
        return ComponentTopology(
            status="undetermined",
            reason="primitive_node_limit",
            canonical_key=None,
            primitive_nodes=None,
            primitive_edges=None,
            primitive_degrees=None,
            witness=None,
            **base,
        )
    folded = _fold_translations(graph)
    if isinstance(folded, str):
        return ComponentTopology(
            status="undetermined",
            reason=folded,
            canonical_key=None,
            primitive_nodes=None,
            primitive_edges=None,
            primitive_degrees=None,
            witness=None,
            **base,
        )
    reduced = _rewrite_shift_lattice(folded.graph)
    if isinstance(reduced, str):
        return ComponentTopology(
            status="undetermined",
            reason=reduced,
            canonical_key=None,
            primitive_nodes=len(folded.graph.nodes),
            primitive_edges=len(folded.graph.edges),
            primitive_degrees=degree_sequence(folded.graph),
            witness=None,
            **base,
        )
    lattice_graph, hnf_basis = reduced
    canonical = _lexicographic_key(lattice_graph, periodic_rank, limits)
    if isinstance(canonical, str):
        return ComponentTopology(
            status="undetermined",
            reason=canonical,
            canonical_key=None,
            primitive_nodes=len(lattice_graph.nodes),
            primitive_edges=len(lattice_graph.edges),
            primitive_degrees=degree_sequence(lattice_graph),
            witness=None,
            **base,
        )
    key, detail = canonical
    witness = {
        "kind": "canonical_form",
        "canonical_key": key,
        "fold_orbits": folded.orbits,
        "hnf_basis": [list(vector) for vector in hnf_basis],
        "minimizer": detail,
    }
    return ComponentTopology(
        status="identified",
        reason="canonical_form",
        canonical_key=key,
        primitive_nodes=len(folded.graph.nodes),
        primitive_edges=len(folded.graph.edges),
        primitive_degrees=degree_sequence(folded.graph),
        witness=witness,
        **base,
    )


@dataclass(frozen=True)
class _Folded:
    graph: PeriodicQuotientGraph
    orbits: tuple[dict[str, object], ...]


def _fold_translations(graph: PeriodicQuotientGraph) -> _Folded | str:
    """Fold translational copies inside one connected quotient, if they exist."""
    if all(edge.image_shift == (0, 0, 0) for edge in graph.edges):
        return _singleton_fold(graph)
    positions = _barycentric_positions(graph)
    if isinstance(positions, str):
        return positions
    moduli = [mod1_vector(position) for position in positions]
    if len(set(moduli)) != len(moduli):
        return "barycentric_collision"
    translations = _translation_group(graph, positions, moduli)
    if isinstance(translations, str):
        return translations
    if len(translations) <= 1:
        return _singleton_fold(graph)
    return _quotient_by_translations(graph, positions, translations)


def _singleton_fold(graph: PeriodicQuotientGraph) -> _Folded:
    abstract = PeriodicQuotientGraph(graph.nodes, graph.edges, pbc=(True, True, True))
    orbits = tuple(
        {"primitive_id": node_id, "source_ids": [node_id]}
        for node_id in abstract.node_ids
    )
    return _Folded(abstract, orbits)


def _barycentric_positions(graph: PeriodicQuotientGraph) -> list[F3] | str:
    order = graph.node_ids
    size = len(order)
    index = {node_id: position for position, node_id in enumerate(order)}
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    rhs = [[0, 0, 0] for _ in range(size)]
    for node_id, stubs in directed_adjacency(graph).items():
        row = index[node_id]
        for neighbor, shift in stubs:
            matrix[row][row] += 1
            matrix[row][index[neighbor]] -= 1
            for axis in range(3):
                rhs[row][axis] += shift[axis]
    solved = solve_pinned_system(matrix, rhs, 0)
    if isinstance(solved, str):
        return f"barycentric_{solved}"
    return [tuple(row) for row in solved]  # type: ignore[misc]


def _translation_group(
    graph: PeriodicQuotientGraph,
    raw: list[F3],
    moduli: list[F3],
) -> tuple[tuple[F3, dict[str, str]], ...] | str:
    """Return torus translations that preserve geometric edge displacements."""
    adjacency = directed_adjacency(graph)
    displacement = _displacement_table(graph, raw, adjacency)
    found: dict[F3, dict[str, str]] = {}
    order = graph.node_ids
    for source in range(len(order)):
        for target in range(len(order)):
            translation = mod1_vector(sub_f(moduli[target], moduli[source]))
            if translation in found:
                continue
            mapping: dict[str, str] = {}
            valid = True
            for index, position in enumerate(moduli):
                destination = mod1_vector(add_f(position, translation))
                hits = [hit for hit, point in enumerate(moduli) if point == destination]
                if len(hits) != 1:
                    valid = False
                    break
                mapping[order[index]] = order[hits[0]]
            if not valid or len(set(mapping.values())) != len(order):
                continue
            if not _displacements_match(displacement, mapping):
                continue
            if not _translation_maps_edges(adjacency, raw, mapping):
                continue
            found[translation] = mapping
    if (Fraction(0), Fraction(0), Fraction(0)) not in found:
        return "missing_identity_translation"
    ordered = tuple(sorted(found, key=lambda item: (item[0], item[1], item[2])))
    return tuple((translation, found[translation]) for translation in ordered)


def _displacement_table(
    graph: PeriodicQuotientGraph,
    raw: list[F3],
    adjacency: dict[str, list[tuple[str, ImageShift]]],
) -> dict[str, tuple[F3, ...]]:
    index = {node_id: position for position, node_id in enumerate(graph.node_ids)}
    table: dict[str, tuple[F3, ...]] = {}
    for node_id in graph.node_ids:
        vectors = []
        for neighbor, shift in adjacency[node_id]:
            vectors.append(
                sub_f(
                    add_f(raw[index[neighbor]], fraction_vector(shift)),
                    raw[index[node_id]],
                )
            )
        table[node_id] = tuple(sorted(vectors))
    return table


def _displacements_match(
    table: dict[str, tuple[F3, ...]],
    mapping: dict[str, str],
) -> bool:
    for node_id, image in mapping.items():
        if table[node_id] != table[image]:
            return False
    return True


def _translation_maps_edges(
    adjacency: dict[str, list[tuple[str, ImageShift]]],
    raw: list[F3],
    mapping: dict[str, str],
) -> bool:
    index = {node_id: position for position, node_id in enumerate(adjacency)}
    for source, stubs in adjacency.items():
        image_source = mapping[source]
        image_stubs = adjacency[image_source]
        for neighbor, shift in stubs:
            displacement = sub_f(
                add_f(raw[index[neighbor]], fraction_vector(shift)),
                raw[index[source]],
            )
            predicted = sub_f(
                add_f(raw[index[image_source]], displacement),
                raw[index[mapping[neighbor]]],
            )
            try:
                predicted_shift = as_integer_shift(predicted, "translation image")
            except ValueError:
                return False
            # Directed stubs already contain both orientations. A translation
            # must reproduce the predicted image shift, not its reverse.
            if (mapping[neighbor], predicted_shift) not in image_stubs:
                return False
    return True


def _quotient_by_translations(
    graph: PeriodicQuotientGraph,
    raw: list[F3],
    translations: tuple[tuple[F3, dict[str, str]], ...],
) -> _Folded | str:
    index = {node_id: position for position, node_id in enumerate(graph.node_ids)}
    maps = [mapping for _translation, mapping in translations]
    parent = {node_id: node_id for node_id in graph.node_ids}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for mapping in maps:
        for source, image in mapping.items():
            union(source, image)
    members: dict[str, list[str]] = defaultdict(list)
    for node_id in graph.node_ids:
        members[find(node_id)].append(node_id)
    cell = _symmetry_cell([translation for translation, _mapping in translations])
    if isinstance(cell, str):
        return cell
    inverse = cell
    renewed = [matvec_f(position, inverse) for position in raw]

    def representative(node_id: str) -> str:
        orbit = members[find(node_id)]

        def sort_key(candidate: str) -> tuple[F3, str]:
            point = mod1_vector(renewed[index[candidate]])
            return (point, candidate)

        return min(orbit, key=sort_key)

    rep_of = {node_id: representative(node_id) for node_id in graph.node_ids}
    distinct = sorted(
        set(rep_of.values()),
        key=lambda node_id: (mod1_vector(renewed[index[node_id]]), node_id),
    )
    primitive_ids = {
        node_id: f"v{position}" for position, node_id in enumerate(distinct)
    }
    orbits = tuple(
        {
            "primitive_id": primitive_ids[node_id],
            "source_ids": sorted(members[find(node_id)]),
        }
        for node_id in distinct
    )

    def coefficient(node_id: str) -> ImageShift:
        delta = sub_f(renewed[index[node_id]], renewed[index[rep_of[node_id]]])
        return as_integer_shift(delta, f"orbit offset of {node_id}")

    try:
        offsets = {node_id: coefficient(node_id) for node_id in graph.node_ids}
    except ValueError:
        return "fold_offset_not_integral"

    seen_edges: set[int] = set()
    folded: list[PeriodicQuotientEdge] = []
    adjacency = directed_adjacency(graph)
    for start, edge in enumerate(graph.edges):
        if start in seen_edges:
            continue
        orbit_indexes = _edge_orbit(graph, start, maps, raw, adjacency)
        if isinstance(orbit_indexes, str):
            return orbit_indexes
        seen_edges.update(orbit_indexes)
        representative_edge = graph.edges[min(orbit_indexes)]
        try:
            folded_shift = _folded_shift(representative_edge, offsets, inverse)
        except ValueError:
            return "fold_shift_not_integral"
        source = primitive_ids[rep_of[representative_edge.source]]
        target = primitive_ids[rep_of[representative_edge.target]]
        folded.append(
            PeriodicQuotientEdge(
                f"f{len(folded)}",
                source,
                target,
                folded_shift,
            )
        )
    nodes = tuple(PeriodicQuotientNode(primitive_ids[node_id]) for node_id in distinct)
    try:
        folded_graph = PeriodicQuotientGraph(
            nodes, tuple(folded), pbc=(True, True, True)
        )
    except ValueError:
        return "fold_graph_invalid"
    return _Folded(folded_graph, orbits)


def _edge_orbit(
    graph: PeriodicQuotientGraph,
    start: int,
    maps: list[dict[str, str]],
    raw: list[F3],
    adjacency: dict[str, list[tuple[str, ImageShift]]],
) -> tuple[int, ...] | str:
    index = {node_id: position for position, node_id in enumerate(graph.node_ids)}
    queue = [start]
    seen = {start}
    while queue:
        current = queue.pop()
        edge = graph.edges[current]
        for mapping in maps:
            image = _image_edge_index(graph, edge, mapping, raw, index, adjacency)
            if image is None:
                return "fold_edge_orbit_incomplete"
            if image not in seen:
                seen.add(image)
                queue.append(image)
    return tuple(sorted(seen))


def _image_edge_index(
    graph: PeriodicQuotientGraph,
    edge: PeriodicQuotientEdge,
    mapping: dict[str, str],
    raw: list[F3],
    index: dict[str, int],
    adjacency: dict[str, list[tuple[str, ImageShift]]],
) -> int | None:
    displacement = sub_f(
        add_f(raw[index[edge.target]], fraction_vector(edge.image_shift)),
        raw[index[edge.source]],
    )
    source = mapping[edge.source]
    target = mapping[edge.target]
    predicted = sub_f(add_f(raw[index[source]], displacement), raw[index[target]])
    try:
        predicted_shift = as_integer_shift(predicted, "edge image")
    except ValueError:
        return None
    if (target, predicted_shift) not in adjacency[source]:
        return None
    for position, other in enumerate(graph.edges):
        if _same_undirected(other, source, target, predicted_shift):
            return position
    return None


def _same_undirected(
    edge: PeriodicQuotientEdge,
    source: str,
    target: str,
    shift: ImageShift,
) -> bool:
    if edge.source == source and edge.target == target and edge.image_shift == shift:
        return True
    return (
        edge.source == target
        and edge.target == source
        and edge.image_shift == neg(shift)
    )


def _folded_shift(
    edge: PeriodicQuotientEdge,
    offsets: dict[str, ImageShift],
    inverse: tuple[F3, F3, F3],
) -> ImageShift:
    linear = as_integer_shift(matvec_f(edge.image_shift, inverse), "folded edge")
    return add(sub(add(offsets[edge.target], linear), offsets[edge.source]), (0, 0, 0))


def _symmetry_cell(translations: list[F3]) -> tuple[F3, F3, F3] | str:
    """Return the inverse of a basis whose lattice is the translation group plus Z^3."""
    denominator = 1
    for translation in translations:
        for component in translation:
            denominator = math.lcm(denominator, component.denominator)
    generators: list[ImageShift] = [
        (denominator, 0, 0),
        (0, denominator, 0),
        (0, 0, denominator),
    ]
    for translation in translations:
        generators.append(
            (
                int(translation[0] * denominator),
                int(translation[1] * denominator),
                int(translation[2] * denominator),
            )
        )
    basis = integer_translation_basis(generators)
    if len(basis) != 3:
        return "symmetry_lattice_rank"
    scaled = tuple(
        tuple(Fraction(component, denominator) for component in vector)
        for vector in basis
    )
    integer_basis = (
        (
            int(scaled[0][0] * denominator),
            int(scaled[0][1] * denominator),
            int(scaled[0][2] * denominator),
        ),
        (
            int(scaled[1][0] * denominator),
            int(scaled[1][1] * denominator),
            int(scaled[1][2] * denominator),
        ),
        (
            int(scaled[2][0] * denominator),
            int(scaled[2][1] * denominator),
            int(scaled[2][2] * denominator),
        ),
    )
    try:
        inverse_basis = integer_matrix_inverse(integer_basis)
    except ValueError:
        return "symmetry_lattice_singular"
    # ``integer_basis`` is ``denominator * A``, so ``inv(A) = denominator * inv(integer_basis)``.
    scale = Fraction(denominator)
    return tuple(  # type: ignore[return-value]
        tuple(component * scale for component in row) for row in inverse_basis
    )


def _rewrite_shift_lattice(
    graph: PeriodicQuotientGraph,
) -> tuple[PeriodicQuotientGraph, tuple[ImageShift, ...]] | str:
    """Express shifts in the Hermite basis of the lattice they generate."""
    nonzero = [
        shift
        for _node, stubs in directed_adjacency(graph).items()
        for _neighbor, shift in stubs
        if shift != (0, 0, 0)
    ]
    if not nonzero:
        return graph, ()
    basis = list(integer_translation_basis(nonzero))
    rows = [list(vector) for vector in basis]

    def rank(vectors: list[list[int]]) -> int:
        return _integer_rank(vectors)

    for unit in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        if len(rows) >= 3:
            break
        if rank(rows + [unit]) > rank(rows):
            rows.append(unit)
    if len(rows) != 3:
        return "shift_lattice_completion"
    matrix = (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    )
    try:
        inverse = integer_matrix_inverse(matrix)
    except ValueError:
        return "shift_lattice_singular"
    edges: list[PeriodicQuotientEdge] = []
    try:
        for edge in graph.edges:
            edges.append(
                PeriodicQuotientEdge(
                    edge.edge_id,
                    edge.source,
                    edge.target,
                    as_integer_shift(matvec_f(edge.image_shift, inverse), edge.edge_id),
                )
            )
    except ValueError:
        return "shift_not_in_lattice"
    try:
        rewritten = PeriodicQuotientGraph(
            graph.nodes, tuple(edges), pbc=(True, True, True)
        )
    except ValueError:
        return "shift_lattice_invalid"
    return rewritten, tuple(tuple(row) for row in rows[: len(basis)])  # type: ignore[return-value]


def _integer_rank(vectors: list[list[int]]) -> int:
    """Exact rank by row reduction. Whole rows are swapped; columns are not."""
    if not vectors:
        return 0
    rows = [[Fraction(entry) for entry in vector] for vector in vectors]
    rank = 0
    width = max(len(row) for row in rows)
    for column in range(width):
        pivot = None
        for index in range(rank, len(rows)):
            if rows[index][column] != 0:
                pivot = index
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        for index in range(width):
            rows[rank][index] /= divisor
        for other in range(len(rows)):
            if other == rank or rows[other][column] == 0:
                continue
            factor = rows[other][column]
            for index in range(width):
                rows[other][index] -= factor * rows[rank][index]
        rank += 1
    return rank


def _lexicographic_key(
    graph: PeriodicQuotientGraph,
    periodic_rank: int,
    limits: NetAnalysisLimits,
) -> tuple[str, dict[str, object]] | str:
    """Lexicographically least Hermite form over labels and bundle orders.

    A spanning-tree edge is zeroed inside each bundle order, so the search
    quotients by integer gauges. Hermite form then quotients by ``GL(3, Z)``.
    The search enumerates those finite choices and no others. Exceeding the
    backtrack budget returns ``undetermined`` without a partial key.
    """
    nodes = list(graph.node_ids)
    if len(nodes) > limits.canonical_node_limit:
        return "canonical_node_limit"
    assignments = _label_assignments(graph)
    bundles_per_label = [
        _endpoint_bundles(graph, assignment) for assignment in assignments
    ]
    attempts = 0
    for bundles in bundles_per_label:
        bundle_ways = 1
        for bundle in bundles:
            bundle_ways *= math.factorial(len(bundle))
        attempts += bundle_ways
    if attempts > limits.backtrack_limit:
        return "backtrack_limit"
    best: tuple[tuple[int, int, ImageShift], ...] | None = None
    best_labels: tuple[str, ...] | None = None
    for assignment, bundles in zip(assignments, bundles_per_label, strict=True):
        tree = _tree_pairs(bundles)
        for ordering in _bundle_orderings(bundles):
            gauged = _tree_gauge(ordering, tree)
            reduced = hermite_form_shifts(tuple(shift for _i, _j, shift in gauged))
            keyed = tuple(
                (gauged[index][0], gauged[index][1], reduced[index])
                for index in range(len(gauged))
            )
            if best is None or keyed < best:
                best = keyed
                best_labels = assignment
    if best is None or best_labels is None:
        return "canonical_search_empty"
    body = ";".join(
        f"{left}>{right}@{shift[0]},{shift[1]},{shift[2]}"
        for left, right, shift in best
    )
    key = f"n{len(nodes)}|r{periodic_rank}|{body}"
    detail = {
        "labels": list(best_labels),
        "edges": [[left, right, list(shift)] for left, right, shift in best],
    }
    return key, detail


def _label_assignments(graph: PeriodicQuotientGraph) -> tuple[tuple[str, ...], ...]:
    """Permutations inside degree classes.

    Self-loop shifts are not part of the class key: a basis change rewrites
    them, and using them as a filter would drop a legitimate labeling.
    """
    adjacency = directed_adjacency(graph)
    signature: dict[str, tuple[int, int]] = {}
    for node_id in graph.node_ids:
        loop_stubs = sum(
            1 for neighbor, _shift in adjacency[node_id] if neighbor == node_id
        )
        signature[node_id] = (len(adjacency[node_id]), loop_stubs // 2)
    groups: list[list[str]] = []
    for node_id in sorted(graph.node_ids, key=lambda item: (signature[item], item)):
        if groups and signature[groups[-1][0]] == signature[node_id]:
            groups[-1].append(node_id)
        else:
            groups.append([node_id])
    assignments: list[tuple[str, ...]] = [()]
    for group in groups:
        grown: list[tuple[str, ...]] = []
        for prefix in assignments:
            for permutation in itertools.permutations(group):
                grown.append(prefix + permutation)
        assignments = grown
    return tuple(assignments)


def _endpoint_bundles(
    graph: PeriodicQuotientGraph,
    order: tuple[str, ...],
) -> tuple[tuple[tuple[int, int, ImageShift], ...], ...]:
    """Group edges by unordered labels, oriented from the smaller label."""
    label = {node_id: position for position, node_id in enumerate(order)}
    grouped: dict[tuple[int, int], list[tuple[int, int, ImageShift]]] = defaultdict(
        list
    )
    for edge in graph.edges:
        left = label[edge.source]
        right = label[edge.target]
        shift = edge.image_shift
        if right < left:
            left, right = right, left
            shift = neg(shift)
        grouped[(left, right)].append((left, right, shift))
    return tuple(tuple(grouped[pair]) for pair in sorted(grouped))


def _tree_pairs(
    bundles: tuple[tuple[tuple[int, int, ImageShift], ...], ...],
) -> frozenset[tuple[int, int]]:
    """Unique breadth-first tree on the simple graph of endpoint pairs."""
    if not bundles:
        return frozenset()
    node_count = 0
    neighbors: dict[int, set[int]] = defaultdict(set)
    for bundle in bundles:
        left, right, _shift = bundle[0]
        node_count = max(node_count, left, right)
        if left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    parent: dict[int, int | None] = {0: None}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(neighbors[node]):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    pairs = []
    for node, source in parent.items():
        if source is None:
            continue
        pairs.append((min(source, node), max(source, node)))
    if len(parent) != node_count + 1:
        # A quotient component is connected, so a missing node is a bug.
        raise ValueError("canonical tree did not reach every node")
    return frozenset(pairs)


def _bundle_orderings(
    bundles: tuple[tuple[tuple[int, int, ImageShift], ...], ...],
):
    if not bundles:
        yield ()
        return
    pools = [tuple(itertools.permutations(bundle)) for bundle in bundles]
    yield from itertools.product(*pools)


def _tree_gauge(
    ordering: tuple[tuple[tuple[int, int, ImageShift], ...], ...],
    tree_pairs: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int, ImageShift], ...]:
    """Zero the first edge of each tree bundle and rewrite the other shifts."""
    by_pair = {bundle[0][:2]: bundle for bundle in ordering}
    node_count = 0
    for pair in by_pair:
        node_count = max(node_count, pair[0], pair[1])
    potential: list[ImageShift] = [(0, 0, 0) for _ in range(node_count + 1)]
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in by_pair:
        if left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    queue = deque([0])
    seen = {0}
    while queue:
        node = queue.popleft()
        for neighbor in sorted(neighbors[node]):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
            pair = (min(node, neighbor), max(node, neighbor))
            if pair not in tree_pairs:
                continue
            chosen = by_pair[pair][0][2]
            step = chosen if node < neighbor else neg(chosen)
            potential[neighbor] = add(potential[node], step)
    gauged: list[tuple[int, int, ImageShift]] = []
    for bundle in ordering:
        for left, right, shift in bundle:
            gauged.append(
                (left, right, add(shift, sub(potential[left], potential[right])))
            )
    return tuple(gauged)


def _subgraph(
    graph: PeriodicQuotientGraph,
    node_ids: tuple[str, ...],
    edge_ids: tuple[str, ...],
) -> PeriodicQuotientGraph:
    wanted = set(edge_ids)
    edges = tuple(edge for edge in graph.edges if edge.edge_id in wanted)
    nodes = tuple(PeriodicQuotientNode(node_id) for node_id in node_ids)
    return PeriodicQuotientGraph(nodes, edges, pbc=graph.pbc)


def _census_to_json(
    census: tuple[tuple[tuple[int, ...], int], ...] | None,
) -> list[object] | None:
    if census is None:
        return None
    return [[list(sequence), count] for sequence, count in census]


__all__ = [
    "ComponentTopology",
    "NetEquivalence",
    "TopologyCharacterization",
    "characterize_periodic_net",
    "compare_periodic_nets",
    "verify_topology_witness",
]
