"""Immutable domain-neutral periodic quotient multigraph.

A node is one translational orbit. An undirected edge from ``source`` to
``target`` with integer image shift ``s`` joins the source representative to
the target representative displaced by ``s @ L`` (row-vector lattice). The
same edge may be written from ``target`` to ``source`` with shift ``-s``.
Parallel edges keep distinct ids. A self-image edge has ``source == target``;
a nonzero shift is a lattice winding and is not cancelled or primitive-reduced.

Stored orientation, after :class:`PeriodicQuotientGraph` normalizes an edge:

- distinct endpoints are ordered by node id, and the shift is rewritten so it
  still points at the stored target;
- a self-image shift is flipped so its first nonzero component is positive.
  ``(2, 0, 0)`` is unchanged. ``(-2, 0, 0)`` becomes ``(2, 0, 0)`` and still
  generates the sublattice ``2Z``, not ``Z``.

Node-image gauge ``g(v)`` replaces the representative of node ``v`` by the
image ``g(v)``. An edge shift transforms as
``s' = s + g(source) - g(target)`` in the orientation before normalization.
A self-image shift is invariant. A gauge that is the same on every node is an
origin change and does not alter shifts.

Basis convention matches :class:`molcrys_kit.structures.symmetry.LatticeBasisChange`:
``L_new = B @ L_old`` and ``s_new = s_old @ inv(B)``. ``B`` is an integer
3×3 matrix. Unimodular changes (``det ±1``) keep one quotient node per old
node. Supercells (``det`` any nonzero integer) are a separate replication.

PBC flags are part of the quotient, not of the optional embedding. A shift
component on a ``False`` axis is rejected. The embedding may carry Cartesian
positions and a row-vector cell; it is not consulted by connectivity or rank.

This module does not identify net names, interpenetration, or duality.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from molcrys_kit.constants.config import (
    MIN_PERIODIC_CELL_VOLUME_A3,
    PERIODIC_NEIGHBOR_TOLERANCE_A,
)

from ._lattice import ImageShift, exact_shift, undirected_shift

Vector3 = tuple[float, float, float]


def _as_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("pbc flags must be booleans")


def periodic_flags(value: Sequence[object]) -> tuple[bool, bool, bool]:
    """Return three Boolean periodic-boundary flags."""
    try:
        flags = tuple(value)
    except TypeError as exc:
        raise ValueError("pbc must be three booleans") from exc
    if len(flags) != 3:
        raise ValueError("pbc must be three booleans")
    try:
        return (
            _as_bool_flag(flags[0]),
            _as_bool_flag(flags[1]),
            _as_bool_flag(flags[2]),
        )
    except TypeError as exc:
        raise TypeError("pbc must be three booleans") from exc


def _finite_vector(value: Sequence[object], name: str) -> Vector3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return (float(array[0]), float(array[1]), float(array[2]))


def _finite_lattice(
    value: Sequence[Sequence[object]],
) -> tuple[Vector3, Vector3, Vector3]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("lattice must be a finite 3x3 matrix of row vectors")
    if abs(float(np.linalg.det(array))) <= MIN_PERIODIC_CELL_VOLUME_A3:
        raise ValueError("lattice vectors must be linearly independent")
    return tuple(tuple(float(component) for component in row) for row in array)  # type: ignore[return-value]


@dataclass(frozen=True, order=True)
class PeriodicQuotientNode:
    """One vertex of the finite quotient. Identity is ``node_id`` only."""

    node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"node_id": self.node_id}


@dataclass(frozen=True)
class PeriodicQuotientEdge:
    """One undirected quotient edge in a chosen orientation.

    ``image_shift`` is the lattice image of ``target`` relative to ``source``.
    The constructor does not reorder endpoints; :class:`PeriodicQuotientGraph`
    does that so a graph built from either orientation compares equal.
    """

    edge_id: str
    source: str
    target: str
    image_shift: ImageShift = (0, 0, 0)

    def __post_init__(self) -> None:
        if not isinstance(self.edge_id, str) or not self.edge_id:
            raise ValueError("edge_id must be a non-empty string")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("edge source must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("edge target must be a non-empty string")
        object.__setattr__(self, "image_shift", exact_shift(self.image_shift))

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "image_shift": list(self.image_shift),
        }


def reversed_edge(edge: PeriodicQuotientEdge) -> PeriodicQuotientEdge:
    """Return the opposite orientation of one undirected edge.

    Endpoints swap and the image shift is negated. A self-image edge keeps
    its endpoints. The edge id is unchanged. Passing either orientation to
    :class:`PeriodicQuotientGraph` stores the same normalized edge.
    """
    return PeriodicQuotientEdge(
        edge.edge_id,
        edge.target,
        edge.source,
        (-edge.image_shift[0], -edge.image_shift[1], -edge.image_shift[2]),
    )


def _node(value: PeriodicQuotientNode | str) -> PeriodicQuotientNode:
    if isinstance(value, PeriodicQuotientNode):
        return value
    if isinstance(value, str):
        return PeriodicQuotientNode(value)
    raise TypeError(
        "a quotient node must be a non-empty string or PeriodicQuotientNode, "
        f"got {type(value).__name__}"
    )


def _auto_edge_id(source: str, target: str, shift: ImageShift, occurrence: int) -> str:
    left, right, oriented = undirected_shift(source, target, shift)
    return (
        f"edge:{left}->{right}@{oriented[0]}_{oriented[1]}_{oriented[2]}#{occurrence}"
    )


def _prepare_edges(
    edges: Sequence[PeriodicQuotientEdge | Sequence[object]],
) -> tuple[tuple[PeriodicQuotientEdge, ...], tuple[str, ...]]:
    """Assign stable ids and return ``(edges, id_for_each_input_row)``.

    Rows are ``PeriodicQuotientEdge`` or ``(source, target, shift)`` or
    ``(edge_id, source, target, shift)``. Generated ids depend on the
    undirected endpoints and shift, plus an occurrence among identical input
    rows, so reversing an edge or reordering distinct edges keeps the ids.
    """
    parsed: list[tuple[str | None, str, str, ImageShift]] = []
    for index, edge in enumerate(edges):
        if isinstance(edge, PeriodicQuotientEdge):
            parsed.append((edge.edge_id, edge.source, edge.target, edge.image_shift))
            continue
        try:
            row = tuple(edge)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "a caller edge must be a PeriodicQuotientEdge or a tuple of "
                "endpoints and an image shift"
            ) from exc
        if len(row) == 3:
            source, target, shift = row
            edge_id = None
        elif len(row) == 4:
            edge_id, source, target, shift = row
            if not isinstance(edge_id, str) or not edge_id:
                raise ValueError(f"edge {index} has an empty edge id")
        else:
            raise ValueError(
                "a caller edge must be (source, target, shift) or "
                "(edge_id, source, target, shift)"
            )
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError(f"edge {index} endpoints must be strings")
        parsed.append((edge_id, source, target, exact_shift(shift)))

    occurrences: dict[tuple[str, str, ImageShift], int] = defaultdict(int)
    prepared: list[PeriodicQuotientEdge] = []
    input_ids: list[str] = []
    for edge_id, source, target, shift in parsed:
        if edge_id is None:
            left, right, oriented = undirected_shift(source, target, shift)
            key = (left, right, oriented)
            occurrence = occurrences[key]
            occurrences[key] += 1
            edge_id = _auto_edge_id(source, target, shift, occurrence)
        prepared.append(PeriodicQuotientEdge(edge_id, source, target, shift))
        input_ids.append(edge_id)
    return tuple(prepared), tuple(input_ids)


def _orient(edge: PeriodicQuotientEdge) -> PeriodicQuotientEdge:
    source, target, shift = undirected_shift(edge.source, edge.target, edge.image_shift)
    if source == edge.source and target == edge.target and shift == edge.image_shift:
        return edge
    return PeriodicQuotientEdge(edge.edge_id, source, target, shift)


def _validate_shifts(
    edges: Sequence[PeriodicQuotientEdge], pbc: tuple[bool, bool, bool]
) -> None:
    for edge in edges:
        for component, periodic in zip(edge.image_shift, pbc):
            if component and not periodic:
                raise ValueError(
                    f"image shift {edge.image_shift} on edge {edge.edge_id!r} "
                    "is nonzero on a non-periodic axis"
                )


@dataclass(frozen=True)
class PeriodicQuotientGraph:
    """Finite quotient of a periodic multigraph.

    Node and edge order are normalized. Equality therefore ignores the order
    in which equal nodes and edges were supplied. Domain metadata is not
    stored here; adapters keep it beside the graph.
    """

    nodes: tuple[PeriodicQuotientNode, ...]
    edges: tuple[PeriodicQuotientEdge, ...]
    pbc: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        nodes = tuple(_node(node) for node in self.nodes)
        node_ids = [node.node_id for node in nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("periodic quotient node ids must be unique")
        flags = periodic_flags(self.pbc)
        edges = tuple(_orient(edge) for edge in self.edges)
        edge_ids = [edge.edge_id for edge in edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("periodic quotient edge ids must be unique")
        known = set(node_ids)
        for edge in edges:
            if edge.source not in known or edge.target not in known:
                missing = edge.source if edge.source not in known else edge.target
                raise ValueError(
                    f"periodic edge {edge.edge_id!r} references unknown node {missing!r}"
                )
        _validate_shifts(edges, flags)
        object.__setattr__(
            self, "nodes", tuple(sorted(nodes, key=lambda node: node.node_id))
        )
        object.__setattr__(
            self,
            "edges",
            tuple(
                sorted(
                    edges,
                    key=lambda edge: (
                        edge.source,
                        edge.target,
                        edge.image_shift,
                        edge.edge_id,
                    ),
                )
            ),
        )
        object.__setattr__(self, "pbc", flags)

    @classmethod
    def from_edges(
        cls,
        nodes: Sequence[PeriodicQuotientNode | str],
        edges: Sequence[PeriodicQuotientEdge | Sequence[object]],
        *,
        pbc: Sequence[object] = (True, True, True),
    ) -> PeriodicQuotientGraph:
        """Build a quotient graph from node ids and caller-supplied edges.

        Missing edge ids are generated from the undirected endpoints, the
        exact image shift, and an occurrence index. See the module docstring
        for orientation, gauge, and basis conventions.
        """
        prepared, _ids = _prepare_edges(edges)
        return cls(tuple(nodes), prepared, pbc=periodic_flags(pbc))

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    def edge(self, edge_id: str) -> PeriodicQuotientEdge:
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge
        raise KeyError(edge_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "pbc": [bool(flag) for flag in self.pbc],
        }


@runtime_checkable
class PeriodicQuotientView(Protocol):
    """Read-only view implemented by :class:`PeriodicQuotientGraph`.

    Domain classes do not inherit this protocol. Adapters return the concrete
    graph, and algorithms accept any object with these attributes.
    """

    @property
    def nodes(self) -> tuple[PeriodicQuotientNode, ...]: ...

    @property
    def edges(self) -> tuple[PeriodicQuotientEdge, ...]: ...

    @property
    def pbc(self) -> tuple[bool, bool, bool]: ...


@dataclass(frozen=True)
class PeriodicEmbedding:
    """Optional geometry for quotient nodes, independent of connectivity.

    ``lattice_A`` uses row vectors, so Cartesian positions are
    ``fractional @ lattice_A``. A skew cell changes lengths and angles here
    and does not change integer image shifts on the quotient graph.
    """

    positions_A: tuple[tuple[str, Vector3], ...]
    lattice_A: tuple[Vector3, Vector3, Vector3] | None = None
    fractional: tuple[tuple[str, Vector3], ...] | None = None
    _index: dict[str, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        positions = tuple(
            (str(node_id), _finite_vector(vector, f"position of {node_id}"))
            for node_id, vector in self.positions_A
        )
        node_ids = [node_id for node_id, _vector in positions]
        if any(not node_id for node_id in node_ids):
            raise ValueError("embedding node ids must be non-empty strings")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("embedding node ids must be unique")
        lattice = None if self.lattice_A is None else _finite_lattice(self.lattice_A)
        fractional: tuple[tuple[str, Vector3], ...] | None
        if self.fractional is None:
            fractional = None
        else:
            fractional = tuple(
                (
                    str(node_id),
                    _finite_vector(vector, f"fractional coordinate of {node_id}"),
                )
                for node_id, vector in self.fractional
            )
            frac_ids = [node_id for node_id, _vector in fractional]
            if frac_ids != node_ids and set(frac_ids) != set(node_ids):
                raise ValueError("fractional coordinates must match embedding nodes")
            if frac_ids != node_ids:
                lookup = dict(fractional)
                fractional = tuple((node_id, lookup[node_id]) for node_id in node_ids)
        if lattice is None:
            if fractional is not None:
                raise ValueError("fractional coordinates require a lattice")
        else:
            matrix = np.asarray(lattice, dtype=float)
            if fractional is None:
                inverse = np.linalg.inv(matrix)
                fractional = tuple(
                    (
                        node_id,
                        tuple(
                            float(component)
                            for component in np.asarray(vector) @ inverse
                        ),
                    )
                    for node_id, vector in positions
                )
            else:
                for node_id, vector in positions:
                    predicted = (
                        np.asarray(dict(fractional)[node_id], dtype=float) @ matrix
                    )
                    if (
                        float(np.linalg.norm(predicted - np.asarray(vector)))
                        > PERIODIC_NEIGHBOR_TOLERANCE_A
                    ):
                        raise ValueError(
                            f"Cartesian and fractional coordinates disagree for {node_id}"
                        )
        order = sorted(range(len(positions)), key=lambda index: positions[index][0])
        positions = tuple(positions[index] for index in order)
        if fractional is not None:
            fractional = tuple(fractional[index] for index in order)
        object.__setattr__(self, "positions_A", positions)
        object.__setattr__(self, "lattice_A", lattice)
        object.__setattr__(self, "fractional", fractional)
        object.__setattr__(
            self,
            "_index",
            {node_id: index for index, (node_id, _vector) in enumerate(positions)},
        )

    def position(self, node_id: str) -> Vector3:
        index = None if self._index is None else self._index.get(node_id)
        if index is None:
            raise KeyError(node_id)
        return self.positions_A[index][1]

    def fractional_position(self, node_id: str) -> Vector3:
        if self.fractional is None or self._index is None:
            raise ValueError("fractional coordinates require a lattice")
        index = self._index.get(node_id)
        if index is None:
            raise KeyError(node_id)
        return self.fractional[index][1]

    def to_dict(self) -> dict[str, object]:
        return {
            "positions_A": [
                [node_id, list(vector)] for node_id, vector in self.positions_A
            ],
            "lattice_A": None
            if self.lattice_A is None
            else [list(row) for row in self.lattice_A],
            "fractional": None
            if self.fractional is None
            else [[node_id, list(vector)] for node_id, vector in self.fractional],
        }


__all__ = [
    "PeriodicEmbedding",
    "PeriodicQuotientEdge",
    "PeriodicQuotientGraph",
    "PeriodicQuotientNode",
    "PeriodicQuotientView",
    "reversed_edge",
]
