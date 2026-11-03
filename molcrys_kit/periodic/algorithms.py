"""Periodic connectivity of a quotient multigraph.

Finite-quotient components ignore image shifts. Infinite-cover components do
not: two cover vertices lie in one component when a walk's image shifts differ
by an element of the cycle-translation subgroup. Quotient cycle rank
(``edges - nodes + components``) counts finite loops and lattice windings
together. Periodic rank is the dimension of that translation subgroup and is
an integer from 0 to 3. A face-crossing tree and a zero-shift self-image edge
raise the cycle rank without raising the periodic rank.

The independent generators are exactly those selected by
:func:`molcrys_kit.chemistry.topology.analyze_periodic_topology`. The integer
basis is :func:`molcrys_kit.chemistry.topology.integer_translation_basis`
applied to every closed-walk translation, including combinations such as
``(2, 0, 0)`` and ``(3, 0, 0)`` that reduce to ``(1, 0, 0)``. A lone
``(2, 0, 0)`` stays ``(2, 0, 0)``. Fragment-chain ``winding_cycles`` is a
separate optimized path and is not called from here.

``periodic_rank`` is the maximum rank among quotient components. It is not the
rank of the sum of their subgroups: a chain along ``a`` and a separate chain
along ``b`` have periodic rank 1, not 2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from molcrys_kit.chemistry.models import ChemicalBond
from molcrys_kit.chemistry.topology import (
    analyze_periodic_topology,
    integer_translation_basis,
)

from ._lattice import ImageShift, add, det_square, sub
from .graph import PeriodicQuotientEdge, PeriodicQuotientGraph, PeriodicQuotientView


@dataclass(frozen=True)
class QuotientComponent:
    """One connected component of the finite quotient, ignoring image shifts."""

    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    cycle_rank: int

    def __post_init__(self) -> None:
        if self.cycle_rank < 0:
            raise ValueError("quotient cycle rank cannot be negative")
        if self.cycle_rank != len(self.edge_ids) - len(self.node_ids) + 1:
            raise ValueError("quotient cycle rank must be edges - nodes + 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "cycle_rank": int(self.cycle_rank),
        }


@dataclass(frozen=True)
class ClosedWalkTranslation:
    """Image translation of one cotree edge, oriented as stored on the graph.

    A zero translation is a finite quotient cycle. It contributes to the cycle
    rank and not to the periodic rank.
    """

    edge_id: str
    translation: ImageShift

    def to_dict(self) -> dict[str, object]:
        return {"edge_id": self.edge_id, "translation": list(self.translation)}


@dataclass(frozen=True)
class TranslationSubgroup:
    """Integer translation subgroup of one connected quotient component.

    ``basis`` is the column-style Hermite basis of every closed-walk
    translation. ``independent_generators`` is the subset selected by
    :func:`~molcrys_kit.chemistry.topology.analyze_periodic_topology`; those
    vectors span the same real dimension and are not a Hermite basis.
    ``multiplicity`` is the number of infinite-cover components of this
    quotient component. It is ``None`` when that number is infinite, which
    happens whenever the periodic rank is lower than the number of periodic
    axes.
    """

    rank: int
    basis: tuple[ImageShift, ...]
    independent_generators: tuple[ImageShift, ...]
    multiplicity: int | None

    def __post_init__(self) -> None:
        if self.rank not in {0, 1, 2, 3}:
            raise ValueError("periodic rank must be between zero and three")
        if (
            len(self.basis) != self.rank
            or len(self.independent_generators) != self.rank
        ):
            raise ValueError("generator count must equal periodic rank")
        if self.multiplicity is not None and self.multiplicity < 1:
            raise ValueError("cover multiplicity must be positive when finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "basis": [list(vector) for vector in self.basis],
            "independent_generators": [
                list(vector) for vector in self.independent_generators
            ],
            "multiplicity": self.multiplicity,
        }


@dataclass(frozen=True)
class CoverComponentFamily:
    """Translational copies of one quotient component in the infinite cover.

    The copies are not listed one by one. ``subgroup.multiplicity`` is their
    count when the translation subgroup has finite index in the periodic
    lattice, and ``None`` when there are infinitely many copies.
    """

    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    quotient_cycle_rank: int
    subgroup: TranslationSubgroup
    closed_walks: tuple[ClosedWalkTranslation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "quotient_cycle_rank": int(self.quotient_cycle_rank),
            "subgroup": self.subgroup.to_dict(),
            "closed_walks": [walk.to_dict() for walk in self.closed_walks],
        }


@dataclass(frozen=True)
class PeriodicGraphAnalysis:
    """Quotient components and infinite-cover families of one graph.

    ``periodic_rank`` is the largest component rank. ``quotient_cycle_rank``
    is the cyclomatic number of the finite quotient. They are equal only by
    coincidence.
    """

    quotient_components: tuple[QuotientComponent, ...]
    cover_families: tuple[CoverComponentFamily, ...]
    quotient_cycle_rank: int
    periodic_rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "quotient_components": [
                component.to_dict() for component in self.quotient_components
            ],
            "cover_families": [family.to_dict() for family in self.cover_families],
            "quotient_cycle_rank": int(self.quotient_cycle_rank),
            "periodic_rank": int(self.periodic_rank),
        }


def _as_graph(graph: PeriodicQuotientView) -> PeriodicQuotientGraph:
    if isinstance(graph, PeriodicQuotientGraph):
        return graph
    return PeriodicQuotientGraph(graph.nodes, graph.edges, pbc=graph.pbc)


def _component_records(
    graph: PeriodicQuotientGraph,
) -> tuple[tuple[tuple[str, ...], tuple[PeriodicQuotientEdge, ...]], ...]:
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

    for edge in graph.edges:
        if edge.source != edge.target:
            union(edge.source, edge.target)
    members: dict[str, list[str]] = defaultdict(list)
    for node_id in graph.node_ids:
        members[find(node_id)].append(node_id)
    incident: dict[str, list[PeriodicQuotientEdge]] = defaultdict(list)
    for edge in graph.edges:
        incident[find(edge.source)].append(edge)
    records = []
    for root, node_ids in members.items():
        ordered = tuple(sorted(node_ids))
        records.append((ordered, tuple(incident[root])))
    records.sort(key=lambda record: record[0])
    return tuple(records)


def quotient_components(graph: PeriodicQuotientView) -> tuple[QuotientComponent, ...]:
    """Return finite-quotient components, ignoring image shifts."""
    concrete = _as_graph(graph)
    components = []
    for node_ids, edges in _component_records(concrete):
        components.append(
            QuotientComponent(
                node_ids=node_ids,
                edge_ids=tuple(edge.edge_id for edge in edges),
                cycle_rank=len(edges) - len(node_ids) + 1,
            )
        )
    return tuple(components)


def quotient_cycle_rank(graph: PeriodicQuotientView) -> int:
    """Return ``edges - nodes + components`` for the finite quotient.

    This count includes zero-shift cycles. It is not the periodic rank.
    """
    concrete = _as_graph(graph)
    component_count = len(_component_records(concrete))
    return len(concrete.edges) - len(concrete.nodes) + component_count


def _closed_walks(
    node_ids: tuple[str, ...], edges: tuple[PeriodicQuotientEdge, ...]
) -> tuple[ClosedWalkTranslation, ...]:
    adjacency: dict[str, list[tuple[str, ImageShift, str]]] = {
        node_id: [] for node_id in node_ids
    }
    for edge in edges:
        adjacency[edge.source].append((edge.target, edge.image_shift, edge.edge_id))
        if edge.source != edge.target:
            adjacency[edge.target].append(
                (
                    edge.source,
                    (-edge.image_shift[0], -edge.image_shift[1], -edge.image_shift[2]),
                    edge.edge_id,
                )
            )
    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda item: (item[0], item[2], item[1]))
    potentials: dict[str, ImageShift] = {}
    tree: set[str] = set()
    for root in node_ids:
        if root in potentials:
            continue
        potentials[root] = (0, 0, 0)
        stack = [root]
        while stack:
            node_id = stack.pop()
            for neighbor, shift, edge_id in adjacency[node_id]:
                if edge_id in tree or neighbor == node_id:
                    continue
                if neighbor not in potentials:
                    potentials[neighbor] = add(potentials[node_id], shift)
                    tree.add(edge_id)
                    stack.append(neighbor)
    walks: list[ClosedWalkTranslation] = []
    for edge in edges:
        if edge.edge_id in tree:
            continue
        residual = sub(
            add(potentials[edge.source], edge.image_shift), potentials[edge.target]
        )
        walks.append(ClosedWalkTranslation(edge.edge_id, residual))
    return tuple(walks)


def _multiplicity(
    basis: tuple[ImageShift, ...], pbc: tuple[bool, bool, bool]
) -> int | None:
    axes = [index for index, flag in enumerate(pbc) if flag]
    dimension = len(axes)
    if len(basis) < dimension:
        return None
    if len(basis) > dimension:
        raise ValueError("translation rank exceeds the number of periodic axes")
    if dimension == 0:
        return 1
    minor = tuple(tuple(vector[axis] for axis in axes) for vector in basis)
    index = abs(det_square(minor))
    if index < 1:
        raise ValueError("a full-rank translation basis must have nonzero index")
    return index


def _family(
    node_ids: tuple[str, ...],
    edges: tuple[PeriodicQuotientEdge, ...],
    pbc: tuple[bool, bool, bool],
) -> CoverComponentFamily:
    walks = _closed_walks(node_ids, edges)
    basis = integer_translation_basis(walk.translation for walk in walks)
    bonds = tuple(
        ChemicalBond(edge.source, edge.target, atom2_image_shift=edge.image_shift)
        for edge in edges
    )
    topology = analyze_periodic_topology(node_ids, bonds, periodic_axes=pbc)
    if topology.rank != len(basis):
        raise ValueError(
            "periodic rank from cycle generators does not match the Hermite basis"
        )
    subgroup = TranslationSubgroup(
        rank=len(basis),
        basis=basis,
        independent_generators=topology.translation_generators,
        multiplicity=_multiplicity(basis, pbc),
    )
    return CoverComponentFamily(
        node_ids=node_ids,
        edge_ids=tuple(edge.edge_id for edge in edges),
        quotient_cycle_rank=len(edges) - len(node_ids) + 1,
        subgroup=subgroup,
        closed_walks=walks,
    )


def cover_component_families(
    graph: PeriodicQuotientView,
) -> tuple[CoverComponentFamily, ...]:
    """Return one infinite-cover family per finite-quotient component."""
    concrete = _as_graph(graph)
    return tuple(
        _family(node_ids, edges, concrete.pbc)
        for node_ids, edges in _component_records(concrete)
    )


def closed_walk_translations(
    graph: PeriodicQuotientView,
) -> tuple[ClosedWalkTranslation, ...]:
    """Return one image translation per cotree edge, including zeros."""
    return tuple(
        walk
        for family in cover_component_families(graph)
        for walk in family.closed_walks
    )


def translation_subgroups(
    graph: PeriodicQuotientView,
) -> tuple[TranslationSubgroup, ...]:
    """Return the translation subgroup of each quotient component."""
    return tuple(family.subgroup for family in cover_component_families(graph))


def analyze_periodic_graph(graph: PeriodicQuotientView) -> PeriodicGraphAnalysis:
    """Return quotient components, cover families, cycle rank, and periodic rank."""
    concrete = _as_graph(graph)
    records = _component_records(concrete)
    components = tuple(
        QuotientComponent(
            node_ids=node_ids,
            edge_ids=tuple(edge.edge_id for edge in edges),
            cycle_rank=len(edges) - len(node_ids) + 1,
        )
        for node_ids, edges in records
    )
    families = tuple(
        _family(node_ids, edges, concrete.pbc) for node_ids, edges in records
    )
    cycle_rank = len(concrete.edges) - len(concrete.nodes) + len(records)
    rank = max((family.subgroup.rank for family in families), default=0)
    return PeriodicGraphAnalysis(components, families, cycle_rank, rank)


def periodic_rank(graph: PeriodicQuotientView) -> int:
    """Return the largest periodic rank among quotient components.

    The result is 0 for an empty graph, a tree, or a finite molecule that
    only crosses a cell face. It is not the quotient cycle rank and it is not
    the rank of translations pooled across disconnected components.
    """
    return analyze_periodic_graph(graph).periodic_rank


def cover_multiplicity(graph: PeriodicQuotientView) -> int | None:
    """Return the number of infinite-cover components when that number is finite.

    The result is ``None`` when any quotient component has infinitely many
    translational copies. Empty graphs return 0.
    """
    families = cover_component_families(graph)
    total = 0
    for family in families:
        multiplicity = family.subgroup.multiplicity
        if multiplicity is None:
            return None
        total += multiplicity
    return total


__all__ = [
    "ClosedWalkTranslation",
    "CoverComponentFamily",
    "PeriodicGraphAnalysis",
    "QuotientComponent",
    "TranslationSubgroup",
    "analyze_periodic_graph",
    "closed_walk_translations",
    "cover_component_families",
    "cover_multiplicity",
    "periodic_rank",
    "quotient_components",
    "quotient_cycle_rank",
    "translation_subgroups",
]
