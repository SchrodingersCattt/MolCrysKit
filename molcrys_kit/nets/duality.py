"""Duality from an explicit tiling.

A dual net is built only when the caller supplies a closed face pairing on a
periodic primal: every face is a cover cycle, every face is crossed once,
every primal edge lies in exactly two faces, and the quotient Euler
characteristic is zero. An open or nonperiodic face list is ``undetermined``
with reason ``incomplete_tiling``. It is not a proved dual. Coordination
numbers and coordination sequences do not establish duality. With no tiling
the status is ``not_assessed``.
"""

from __future__ import annotations

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
    face, is not a tiling certificate. This check does not prove that a
    three-dimensional cell fills space; that input stays ``incomplete_tiling``.
    """
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


def _dual_graph(tiling: PeriodicTiling) -> PeriodicQuotientGraph:
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
