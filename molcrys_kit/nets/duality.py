"""Duality from an explicit tiling.

A dual net is built only when the caller supplies a closed face pairing on a
periodic primal: every face is a cover cycle, every face is crossed once,
lattice translates and reversals of one cycle are not counted as two faces,
every adjacency shift is the primitive normal of that face, every primal edge
lies in exactly two faces, and the quotient Euler characteristic is zero.
An open or nonperiodic face list is ``undetermined`` with reason
``incomplete_tiling``. A repeated face is ``duplicate_face``. A shift that is
not the face normal is ``inconsistent_adjacency_shift``. None of those is a
proved dual. Euler's count alone does not certify the surface. Coordination
numbers and coordination sequences do not establish duality. With no tiling
the status is ``not_assessed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from molcrys_kit.periodic import analyze_periodic_graph
from molcrys_kit.periodic._lattice import ImageShift, sub
from molcrys_kit.periodic.graph import PeriodicQuotientGraph

from .invariants import directed_adjacency
from .isomorphism import compare_periodic_nets
from .limits import NetAnalysisLimits


@dataclass(frozen=True)
class TilingFace:
    """One face as an ordered cover cycle.

    ``vertices`` are ``(node_id, image)`` pairs. The closing step from the
    last vertex back to the first must also be a primal edge.
    """

    face_id: str
    vertices: tuple[tuple[str, ImageShift], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "face_id": self.face_id,
            "vertices": [[node_id, list(image)] for node_id, image in self.vertices],
        }


@dataclass(frozen=True)
class TilingAdjacency:
    """A dual edge across ``face_id`` from ``source_tile`` to ``target_tile``."""

    face_id: str
    source_tile: str
    target_tile: str
    image_shift: ImageShift = (0, 0, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "face_id": self.face_id,
            "source_tile": self.source_tile,
            "target_tile": self.target_tile,
            "image_shift": list(self.image_shift),
        }


@dataclass(frozen=True)
class PeriodicTiling:
    """Caller-supplied face incidences. This module does not discover tiles."""

    tiles: tuple[str, ...]
    faces: tuple[TilingFace, ...]
    adjacencies: tuple[TilingAdjacency, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "tiles": list(self.tiles),
            "faces": [face.to_dict() for face in self.faces],
            "adjacencies": [adjacency.to_dict() for adjacency in self.adjacencies],
        }


@dataclass(frozen=True)
class DualityAssessment:
    """Relation of a validated dual graph to another periodic net."""

    status: str
    reason: str
    witness: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "witness": self.witness,
        }


def assess_duality(
    primal: PeriodicQuotientGraph,
    other: PeriodicQuotientGraph,
    tiling: PeriodicTiling | None,
    limits: NetAnalysisLimits | None = None,
) -> DualityAssessment:
    """Compare ``other`` with the dual of ``tiling`` on ``primal``."""
    if tiling is None:
        return DualityAssessment("not_assessed", "no_tiling", None)
    limits = limits or NetAnalysisLimits.from_config()
    failure = _validate(primal, tiling)
    if failure is not None:
        return DualityAssessment("undetermined", failure, {"kind": "invalid_tiling"})
    dual = _dual_graph(tiling)
    compared = compare_periodic_nets(dual, other, limits)
    witness = {
        "kind": "tiling_dual",
        "dual_graph": dual.to_dict(),
        "equivalence": compared.to_dict(),
    }
    if compared.status == "isomorphic" and compared.proof == "explicit_isomorphism":
        return DualityAssessment("dual", "explicit_isomorphism", witness)
    if compared.status == "non_isomorphic":
        return DualityAssessment("not_dual", compared.reason, witness)
    return DualityAssessment("undetermined", compared.reason, witness)


def _validate(primal: PeriodicQuotientGraph, tiling: PeriodicTiling) -> str | None:
    if not tiling.tiles or len(set(tiling.tiles)) != len(tiling.tiles):
        return "invalid_tiles"
    faces = {face.face_id: face for face in tiling.faces}
    if len(faces) != len(tiling.faces) or any(
        not face.face_id for face in tiling.faces
    ):
        return "invalid_faces"
    known = set(primal.node_ids)
    adjacency = directed_adjacency(primal)
    for face in tiling.faces:
        if len(face.vertices) < 3:
            return "face_too_short"
        cycle = list(face.vertices) + [face.vertices[0]]
        for (source, source_image), (target, target_image) in zip(cycle, cycle[1:]):
            if source not in known or target not in known:
                return "face_unknown_node"
            step = sub(target_image, source_image)
            if (target, step) not in adjacency[source]:
                return "face_not_a_cover_cycle"
    tiles = set(tiling.tiles)
    seen_faces: set[str] = set()
    for adjacency_record in tiling.adjacencies:
        if adjacency_record.face_id not in faces:
            return "adjacency_unknown_face"
        if adjacency_record.face_id in seen_faces:
            return "face_used_twice"
        seen_faces.add(adjacency_record.face_id)
        if (
            adjacency_record.source_tile not in tiles
            or adjacency_record.target_tile not in tiles
        ):
            return "adjacency_unknown_tile"
    if not tiling.adjacencies:
        return "tiling_has_no_incidences"
    if set(faces) != seen_faces:
        return "incomplete_tiling"
    incident = {
        tile
        for record in tiling.adjacencies
        for tile in (record.source_tile, record.target_tile)
    }
    if incident != tiles:
        return "incomplete_tiling"
    if analyze_periodic_graph(primal).periodic_rank < 1:
        return "incomplete_tiling"
    return _closed_surface(primal, tiling)


def _closed_surface(
    primal: PeriodicQuotientGraph, tiling: PeriodicTiling
) -> str | None:
    """Require a closed quotient surface: two faces on every edge, Euler zero.

    An edge that meets any other number of faces, including a single open
    face, is not a tiling certificate. Faces that are the same cycle up to
    reversal or lattice translation are not two incidences. An adjacency shift
    must be the primitive lattice normal of its face. This check does not
    prove that a three-dimensional cell fills space; that input stays
    ``incomplete_tiling``.
    """
    gluing = _oriented_gluing(tiling)
    if gluing is not None:
        return gluing
    counts: dict[tuple[str, str, ImageShift], int] = {}
    for face in tiling.faces:
        seen_in_face: set[tuple[str, str, ImageShift]] = set()
        steps = list(face.vertices) + [face.vertices[0]]
        for (source, source_image), (target, target_image) in zip(steps, steps[1:]):
            step = sub(target_image, source_image)
            seen_in_face.add(_undirected_edge(source, target, step))
        for key in seen_in_face:
            counts[key] = counts.get(key, 0) + 1
    primal_edges = {
        _undirected_edge(edge.source, edge.target, edge.image_shift)
        for edge in primal.edges
    }
    if set(counts) != primal_edges or any(count != 2 for count in counts.values()):
        return "incomplete_tiling"
    euler = (
        len(primal.nodes) - len(primal.edges) + len(tiling.faces) - len(tiling.tiles)
    )
    if euler != 0:
        return "incomplete_tiling"
    return None


def _undirected_edge(
    source: str, target: str, step: ImageShift
) -> tuple[str, str, ImageShift]:
    reverse = (target, source, (-step[0], -step[1], -step[2]))
    current = (source, target, step)
    if reverse < current:
        return reverse
    return current


def _oriented_gluing(tiling: PeriodicTiling) -> str | None:
    """Check distinct faces and shifts against oriented face normals.

    The normal is the right-hand primitive vector orthogonal to the boundary
    steps. Crossing that face moves by exactly that vector or its reverse.
    A tangential gauge, a doubled step, or a second copy of the same cycle is
    not a consistent gluing.
    """
    seen: dict[tuple[object, ...], str] = {}
    for face in tiling.faces:
        key = _canonical_unoriented_face(face.vertices)
        if key in seen:
            return "duplicate_face"
        seen[key] = face.face_id
    by_face = {record.face_id: record for record in tiling.adjacencies}
    for face in tiling.faces:
        normal = _oriented_normal(face.vertices)
        if normal is None:
            return "incomplete_tiling"
        shift = by_face[face.face_id].image_shift
        opposite = (-normal[0], -normal[1], -normal[2])
        if shift != normal and shift != opposite:
            return "inconsistent_adjacency_shift"
    return None


def _canonical_unoriented_face(
    vertices: tuple[tuple[str, ImageShift], ...],
) -> tuple[object, ...]:
    """Identity of one face orbit, ignoring direction and lattice translate."""
    origin = min(image for _node_id, image in vertices)
    gauged = tuple(sorted((node_id, sub(image, origin)) for node_id, image in vertices))
    cycle = list(vertices) + [vertices[0]]
    edges = []
    for (source, source_image), (target, target_image) in zip(cycle, cycle[1:]):
        step = sub(target_image, source_image)
        edges.append(_undirected_edge(source, target, step))
    return (gauged, tuple(sorted(edges)))


def _oriented_normal(
    vertices: tuple[tuple[str, ImageShift], ...],
) -> ImageShift | None:
    """Return the right-hand primitive normal, or ``None`` if it is not unique."""
    cycle = list(vertices) + [vertices[0]]
    steps = [
        sub(target_image, source_image)
        for (_source, source_image), (_target, target_image) in zip(cycle, cycle[1:])
    ]
    turns = [
        _cross(step, steps[(index + 1) % len(steps)])
        for index, step in enumerate(steps)
    ]
    turns = [turn for turn in turns if turn != (0, 0, 0)]
    if not turns:
        return None
    primitive = _primitive(turns[0])
    if primitive is None:
        return None
    sign = _integer_multiple(turns[0], primitive)
    if sign is None or sign == 0:
        return None
    normal = primitive if sign > 0 else (-primitive[0], -primitive[1], -primitive[2])
    for turn in turns:
        multiple = _integer_multiple(turn, normal)
        if multiple is None or multiple <= 0:
            return None
    for step in steps:
        if _dot(step, normal) != 0:
            return None
    return normal


def _primitive(vector: ImageShift) -> ImageShift | None:
    divisor = math.gcd(math.gcd(abs(vector[0]), abs(vector[1])), abs(vector[2]))
    if divisor == 0:
        return None
    reduced = (vector[0] // divisor, vector[1] // divisor, vector[2] // divisor)
    for component in reduced:
        if component < 0:
            return (-reduced[0], -reduced[1], -reduced[2])
        if component > 0:
            return reduced
    return None


def _integer_multiple(vector: ImageShift, basis: ImageShift) -> int | None:
    multiple: int | None = None
    for component, base in zip(vector, basis, strict=True):
        if base == 0:
            if component != 0:
                return None
            continue
        if component % base != 0:
            return None
        quotient = component // base
        if multiple is None:
            multiple = quotient
        elif quotient != multiple:
            return None
    return multiple


def _dot(left: ImageShift, right: ImageShift) -> int:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: ImageShift, right: ImageShift) -> ImageShift:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dual_graph(tiling: PeriodicTiling) -> PeriodicQuotientGraph:
    """Build the dual from shifts already checked against face normals."""
    edges = [
        (
            adjacency.face_id,
            adjacency.source_tile,
            adjacency.target_tile,
            adjacency.image_shift,
        )
        for adjacency in tiling.adjacencies
    ]
    return PeriodicQuotientGraph.from_edges(tiling.tiles, edges)


__all__ = [
    "DualityAssessment",
    "PeriodicTiling",
    "TilingAdjacency",
    "TilingFace",
    "assess_duality",
]
