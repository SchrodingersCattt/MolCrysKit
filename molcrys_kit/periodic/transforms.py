"""Gauge, basis, origin, relabeling, and supercell transforms.

These operations return a new quotient graph. They do not mutate the input
and they do not primitive-reduce a nonprimitive winding. A supercell
replication uses the separator U+001F between a node id and its image, so
node ids and edge ids must not already contain that character.

An equivalent supercell preserves periodic rank and, for components whose
rank equals the number of periodic axes, the total number of infinite-cover
components. Integer components of the translation basis are written in the
supercell and are not the parent-cell integers. Map a supercell shift ``s``
back with ``s @ B`` where ``L_super = B @ L_parent``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from molcrys_kit.constants.symmetry_path import AFFINE_INTEGER_TOLERANCE

from ._lattice import (
    ImageShift,
    Matrix3,
    add,
    adjugate,
    det3,
    exact_shift,
    floor_div,
    sub,
    vecmul,
)
from .graph import (
    PeriodicEmbedding,
    PeriodicQuotientEdge,
    PeriodicQuotientGraph,
    PeriodicQuotientNode,
    _finite_lattice,
    _finite_vector,
)

_IMAGE_SEPARATOR = "\x1f"


def _integer_matrix(value: Sequence[Sequence[object]], *, unimodular: bool) -> Matrix3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("basis matrix must be a finite 3x3 matrix")
    rounded = np.rint(array)
    if float(np.max(np.abs(array - rounded))) > AFFINE_INTEGER_TOLERANCE:
        raise ValueError("basis matrix must be integer-valued")
    matrix: Matrix3 = tuple(
        tuple(int(component) for component in row) for row in rounded
    )  # type: ignore[assignment]
    determinant = det3(matrix)
    if determinant == 0:
        raise ValueError("basis matrix must be invertible")
    if unimodular and abs(determinant) != 1:
        raise ValueError("unimodular basis change must have determinant +1 or -1")
    return matrix


def _inverse_unimodular(matrix: Matrix3) -> Matrix3:
    determinant = det3(matrix)
    if abs(determinant) != 1:
        raise ValueError("unimodular basis change must have determinant +1 or -1")
    adjunct = adjugate(matrix)
    if determinant == 1:
        return adjunct
    return tuple(tuple(-component for component in row) for row in adjunct)  # type: ignore[return-value]


def _reject_nonperiodic(
    shift: ImageShift, pbc: tuple[bool, bool, bool], context: str
) -> None:
    for component, periodic in zip(shift, pbc):
        if component and not periodic:
            raise ValueError(
                f"{context} produces image shift {tuple(int(v) for v in shift)} "
                "on a non-periodic axis"
            )


def _gauges(
    graph: PeriodicQuotientGraph, gauges: Mapping[str, Sequence[int]]
) -> dict[str, ImageShift]:
    result = {node_id: (0, 0, 0) for node_id in graph.node_ids}
    for node_id, shift in gauges.items():
        if node_id not in result:
            raise ValueError(f"gauge references unknown node {node_id!r}")
        exact = exact_shift(shift, f"gauge of {node_id}")
        _reject_nonperiodic(exact, graph.pbc, f"gauge of {node_id!r}")
        result[str(node_id)] = exact
    return result


@dataclass(frozen=True)
class EmbeddedQuotientGraph:
    """A quotient graph and the embedding transform that accompanies it."""

    graph: PeriodicQuotientGraph
    embedding: PeriodicEmbedding | None = None


def _positions_from_fractional(
    fractional: Sequence[tuple[str, Sequence[float]]],
    lattice: Sequence[Sequence[float]],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    matrix = np.asarray(lattice, dtype=float)
    return tuple(
        (
            node_id,
            tuple(
                float(component)
                for component in np.asarray(vector, dtype=float) @ matrix
            ),
        )
        for node_id, vector in fractional
    )


def apply_node_gauge(
    graph: PeriodicQuotientGraph,
    gauges: Mapping[str, Sequence[int]],
    embedding: PeriodicEmbedding | None = None,
) -> EmbeddedQuotientGraph:
    """Move node representatives by integer images.

    The stored shift of an edge becomes ``s + g(source) - g(target)`` before
    orientation is normalized again. Self-image shifts are unchanged. A gauge
    omitted for a node is zero.
    """
    gauge_map = _gauges(graph, gauges)
    edges = tuple(
        PeriodicQuotientEdge(
            edge.edge_id,
            edge.source,
            edge.target,
            add(edge.image_shift, sub(gauge_map[edge.source], gauge_map[edge.target])),
        )
        for edge in graph.edges
    )
    gauged = PeriodicQuotientGraph(graph.nodes, edges, pbc=graph.pbc)
    if embedding is None:
        return EmbeddedQuotientGraph(gauged, None)
    if embedding.lattice_A is None:
        if any(any(component) for component in gauge_map.values()):
            raise ValueError("a node-image gauge of an embedding requires a lattice")
        return EmbeddedQuotientGraph(gauged, embedding)
    fractional = tuple(
        (
            node_id,
            tuple(
                float(component)
                for component in np.asarray(embedding.fractional_position(node_id))
                + np.asarray(gauge_map[node_id], dtype=float)
            ),
        )
        for node_id in graph.node_ids
    )
    positions = _positions_from_fractional(fractional, embedding.lattice_A)
    return EmbeddedQuotientGraph(
        gauged,
        PeriodicEmbedding(positions, embedding.lattice_A, fractional),
    )


def apply_unimodular_basis(
    graph: PeriodicQuotientGraph,
    matrix: Sequence[Sequence[object]],
    embedding: PeriodicEmbedding | None = None,
) -> EmbeddedQuotientGraph:
    """Rewrite image shifts for ``L_new = B @ L_old`` with ``det B = ±1``.

    Node ids are unchanged. Each shift transforms as ``s_new = s_old @ inv(B)``.
    Cartesian embedding coordinates are unchanged; fractional coordinates and
    the stored lattice change with ``B``.
    """
    change = _integer_matrix(matrix, unimodular=True)
    inverse = _inverse_unimodular(change)
    edges = []
    for edge in graph.edges:
        shift = vecmul(edge.image_shift, inverse)
        _reject_nonperiodic(shift, graph.pbc, "unimodular basis change")
        edges.append(
            PeriodicQuotientEdge(edge.edge_id, edge.source, edge.target, shift)
        )
    transformed = PeriodicQuotientGraph(graph.nodes, tuple(edges), pbc=graph.pbc)
    if embedding is None:
        return EmbeddedQuotientGraph(transformed, None)
    if embedding.lattice_A is None:
        raise ValueError("a basis change of an embedding requires a lattice")
    new_lattice = _finite_lattice(
        np.asarray(change, dtype=float) @ np.asarray(embedding.lattice_A, dtype=float)
    )
    inverse_array = np.asarray(inverse, dtype=float)
    fractional = tuple(
        (
            node_id,
            tuple(
                float(component)
                for component in np.asarray(embedding.fractional_position(node_id))
                @ inverse_array
            ),
        )
        for node_id in graph.node_ids
    )
    return EmbeddedQuotientGraph(
        transformed,
        PeriodicEmbedding(embedding.positions_A, new_lattice, fractional),
    )


def shift_embedding_origin(
    embedding: PeriodicEmbedding, origin: Sequence[float]
) -> PeriodicEmbedding:
    """Move the fractional origin by ``origin`` without changing the quotient.

    Stored fractional coordinates decrease by ``origin``. A uniform integer
    origin change is not a relative node gauge and does not alter edge shifts;
    apply it to the embedding only, or pass the same integer gauge for every
    node to :func:`apply_node_gauge`.
    """
    if embedding.lattice_A is None or embedding.fractional is None:
        raise ValueError(
            "an origin shift requires fractional coordinates and a lattice"
        )
    offset = np.asarray(_finite_vector(origin, "origin"), dtype=float)
    fractional = tuple(
        (
            node_id,
            tuple(float(component) for component in np.asarray(vector) - offset),
        )
        for node_id, vector in embedding.fractional
    )
    positions = _positions_from_fractional(fractional, embedding.lattice_A)
    return PeriodicEmbedding(positions, embedding.lattice_A, fractional)


def replace_lattice(
    embedding: PeriodicEmbedding, lattice: Sequence[Sequence[float]]
) -> PeriodicEmbedding:
    """Keep fractional coordinates and replace the row-vector cell.

    Use this for a skew cell that represents the same integer graph. Image
    shifts are not stored on the embedding and are left untouched.
    """
    if embedding.fractional is None:
        raise ValueError("replacing a lattice requires fractional coordinates")
    new_lattice = _finite_lattice(lattice)
    positions = _positions_from_fractional(embedding.fractional, new_lattice)
    return PeriodicEmbedding(positions, new_lattice, embedding.fractional)


def relabel_nodes(
    graph: PeriodicQuotientGraph,
    mapping: Mapping[str, str],
    embedding: PeriodicEmbedding | None = None,
) -> EmbeddedQuotientGraph:
    """Rename nodes. Image shifts are rewritten only to keep the stored orientation."""
    if set(mapping) != set(graph.node_ids):
        raise ValueError("relabel mapping must cover every node exactly once")
    images = []
    for node_id in graph.node_ids:
        renamed = mapping[node_id]
        if not isinstance(renamed, str) or not renamed:
            raise ValueError("relabeled node ids must be non-empty strings")
        images.append(renamed)
    if len(set(images)) != len(images):
        raise ValueError("relabel mapping must be bijective")
    edges = tuple(
        PeriodicQuotientEdge(
            edge.edge_id,
            mapping[edge.source],
            mapping[edge.target],
            edge.image_shift,
        )
        for edge in graph.edges
    )
    relabeled = PeriodicQuotientGraph(
        tuple(PeriodicQuotientNode(node_id) for node_id in images),
        edges,
        pbc=graph.pbc,
    )
    if embedding is None:
        return EmbeddedQuotientGraph(relabeled, None)
    fractional = None
    if embedding.fractional is not None:
        fractional = tuple(
            (mapping[node_id], vector) for node_id, vector in embedding.fractional
        )
    positions = tuple(
        (mapping[node_id], vector) for node_id, vector in embedding.positions_A
    )
    return EmbeddedQuotientGraph(
        relabeled,
        PeriodicEmbedding(positions, embedding.lattice_A, fractional),
    )


def _decompose(shift: ImageShift, matrix: Matrix3) -> tuple[ImageShift, ImageShift]:
    """Return ``(representative, supercell_shift)`` with ``shift = rep + n @ B``."""
    determinant = det3(matrix)
    numer = vecmul(shift, adjugate(matrix))
    supercell_shift = tuple(floor_div(component, determinant) for component in numer)
    represented = vecmul(supercell_shift, matrix)  # type: ignore[arg-type]
    return sub(shift, represented), supercell_shift  # type: ignore[return-value]


def _representatives(matrix: Matrix3) -> tuple[ImageShift, ...]:
    determinant = abs(det3(matrix))
    bounds = [sum(abs(matrix[row][col]) for row in range(3)) or 0 for col in range(3)]
    found: set[ImageShift] = set()
    for left in range(-bounds[0], bounds[0] + 1):
        for middle in range(-bounds[1], bounds[1] + 1):
            for right in range(-bounds[2], bounds[2] + 1):
                candidate = (left, middle, right)
                representative, supercell_shift = _decompose(candidate, matrix)
                if supercell_shift == (0, 0, 0):
                    found.add(representative)
    ordered = tuple(sorted(found))
    if len(ordered) != determinant:
        raise ValueError(
            "supercell coset enumeration found "
            f"{len(ordered)} representatives for determinant {determinant}"
        )
    return ordered


def _image_id(identifier: str, image: ImageShift) -> str:
    if _IMAGE_SEPARATOR in identifier:
        raise ValueError("ids replicated into a supercell must not contain U+001F")
    return f"{identifier}{_IMAGE_SEPARATOR}{image[0]},{image[1]},{image[2]}"


@dataclass(frozen=True)
class SupercellReplication:
    """Map from supercell node and edge ids back to the parent quotient.

    ``node_images`` rows are ``(supercell_node_id, parent_node_id, representative)``.
    ``edge_sources`` rows are ``(supercell_edge_id, parent_edge_id, source_representative)``.
    """

    matrix: Matrix3
    node_images: tuple[tuple[str, str, ImageShift], ...]
    edge_sources: tuple[tuple[str, str, ImageShift], ...]

    def parent_shift(self, shift: Sequence[int]) -> ImageShift:
        """Express a supercell image shift in the parent lattice."""
        return vecmul(exact_shift(shift), self.matrix)


@dataclass(frozen=True)
class SupercellQuotientGraph:
    """Parent quotient replicated onto an integer supercell."""

    graph: PeriodicQuotientGraph
    embedding: PeriodicEmbedding | None
    replication: SupercellReplication


def equivalent_supercell(
    graph: PeriodicQuotientGraph,
    matrix: Sequence[Sequence[object]],
    embedding: PeriodicEmbedding | None = None,
) -> SupercellQuotientGraph:
    """Replicate the quotient onto ``L_super = B @ L_parent``.

    ``B`` is any invertible integer 3×3 matrix. Each parent node becomes
    ``|det B|`` images. Edge shifts are integers in the supercell basis.
    Nonprimitive parent windings stay nonprimitive: a self-image ``(2, 0, 0)``
    does not become a primitive translation of the parent lattice.
    """
    change = _integer_matrix(matrix, unimodular=False)
    representatives = _representatives(change)
    node_images: list[tuple[str, str, ImageShift]] = []
    nodes: list[str] = []
    for node_id in graph.node_ids:
        for representative in representatives:
            replicated = _image_id(node_id, representative)
            nodes.append(replicated)
            node_images.append((replicated, node_id, representative))
    edges: list[PeriodicQuotientEdge] = []
    edge_sources: list[tuple[str, str, ImageShift]] = []
    for edge in graph.edges:
        for representative in representatives:
            arrival = add(representative, edge.image_shift)
            target_rep, supercell_shift = _decompose(arrival, change)
            _reject_nonperiodic(supercell_shift, graph.pbc, "supercell replication")
            edge_id = _image_id(edge.edge_id, representative)
            edges.append(
                PeriodicQuotientEdge(
                    edge_id,
                    _image_id(edge.source, representative),
                    _image_id(edge.target, target_rep),
                    supercell_shift,
                )
            )
            edge_sources.append((edge_id, edge.edge_id, representative))
    replicated_graph = PeriodicQuotientGraph(tuple(nodes), tuple(edges), pbc=graph.pbc)
    replicated_embedding = None
    if embedding is not None:
        if embedding.lattice_A is None or embedding.fractional is None:
            raise ValueError(
                "a supercell embedding requires fractional coordinates and a lattice"
            )
        parent_lattice = np.asarray(embedding.lattice_A, dtype=float)
        new_lattice = _finite_lattice(np.asarray(change, dtype=float) @ parent_lattice)
        inverse = np.linalg.inv(np.asarray(change, dtype=float))
        positions = []
        fractional = []
        for node_id in graph.node_ids:
            parent_fractional = np.asarray(embedding.fractional_position(node_id))
            for representative in representatives:
                child_fractional = (
                    parent_fractional + np.asarray(representative, dtype=float)
                ) @ inverse
                child_cartesian = (
                    parent_fractional + np.asarray(representative, dtype=float)
                ) @ parent_lattice
                child_id = _image_id(node_id, representative)
                positions.append(
                    (child_id, tuple(float(component) for component in child_cartesian))
                )
                fractional.append(
                    (
                        child_id,
                        tuple(float(component) for component in child_fractional),
                    )
                )
        replicated_embedding = PeriodicEmbedding(
            tuple(positions), new_lattice, tuple(fractional)
        )
    return SupercellQuotientGraph(
        replicated_graph,
        replicated_embedding,
        SupercellReplication(change, tuple(node_images), tuple(edge_sources)),
    )


__all__ = [
    "EmbeddedQuotientGraph",
    "SupercellQuotientGraph",
    "SupercellReplication",
    "apply_node_gauge",
    "apply_unimodular_basis",
    "equivalent_supercell",
    "relabel_nodes",
    "replace_lattice",
    "shift_embedding_origin",
]
