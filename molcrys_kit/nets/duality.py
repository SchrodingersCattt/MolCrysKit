"""Duality from an explicit tiling.

A dual net is built only from a caller-supplied face pairing that passes one
of two certificates.

The space-filling certificate accepts a periodic tiling of Euclidean 3-space.
Each tile develops to a convex ball, opposite sides of every face meet at the
claimed lattice shift, the link of every edge is one cycle of at least three
faces, tile interiors are disjoint, and the tile volumes sum to the
translation cell. The witness records the primal face, the two tiles, and the
shift of the dual edge. An edge may lie in any number of face orbits other
than two. That count is not a reason to reject the tiling.

If that certificate does not apply, the closed-surface certificate still
accepts a quotient surface: every face is a cover cycle crossed once, lattice
translates and reversals of one cycle are one face, every adjacency shift is
the primitive normal of that face, every primal edge lies in exactly two face
orbits, and the quotient Euler characteristic is zero. Its witness kind is
``tiling_dual``. It does not claim that a three-dimensional cell fills space.

An open or nonperiodic face list is ``undetermined`` with reason
``incomplete_tiling``. A repeated face is ``duplicate_face``. A shift that is
not the face normal, and is not a same-cell gluing of two distinct tiles, is
``inconsistent_adjacency_shift``. None of those is a proved dual. Coordination
numbers, coordination sequences, voids, and prototype names do not establish
duality. With no tiling the status is ``not_assessed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from molcrys_kit.periodic import analyze_periodic_graph
from molcrys_kit.periodic._lattice import ImageShift, add, sub
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
    filling = _space_filling(primal, tiling, limits)
    if filling is None:
        surface = _closed_surface(primal, tiling)
        if surface is not None:
            return DualityAssessment(
                "undetermined", surface, {"kind": "invalid_tiling"}
            )
        return _compare_dual(tiling, other, limits, _surface_witness(tiling))
    if not filling.accepted:
        return DualityAssessment(
            "undetermined", filling.reason, {"kind": "invalid_tiling"}
        )
    return _compare_dual(tiling, other, limits, filling.witness)


def _compare_dual(tiling, other, limits, witness_body):
    dual = _dual_graph(tiling)
    compared = compare_periodic_nets(dual, other, limits)
    witness = {
        **witness_body,
        "dual_graph": dual.to_dict(),
        "equivalence": compared.to_dict(),
    }
    if compared.status == "isomorphic" and compared.proof == "explicit_isomorphism":
        return DualityAssessment("dual", "explicit_isomorphism", witness)
    if compared.status == "non_isomorphic":
        return DualityAssessment("not_dual", compared.reason, witness)
    return DualityAssessment("undetermined", compared.reason, witness)


def _surface_witness(tiling: PeriodicTiling) -> dict[str, object]:
    return {"kind": "tiling_dual", "incidence": _incidence_rows(tiling)}


def _incidence_rows(tiling: PeriodicTiling) -> list[dict[str, object]]:
    faces = {face.face_id: face for face in tiling.faces}
    rows = []
    for adjacency in tiling.adjacencies:
        face = faces[adjacency.face_id]
        rows.append(
            {
                "face_id": adjacency.face_id,
                "dual_edge_id": adjacency.face_id,
                "source_tile": adjacency.source_tile,
                "target_tile": adjacency.target_tile,
                "image_shift": list(adjacency.image_shift),
                "primal_vertices": [
                    [node_id, list(image)] for node_id, image in face.vertices
                ],
            }
        )
    return rows


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
    return None


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
    """Build the dual: one node per tile and one edge per face adjacency."""
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


@dataclass(frozen=True)
class _Filling:
    """A finished space-filling certificate, or a refusal that blocks fallback."""

    accepted: bool
    reason: str
    witness: dict[str, object]


def _space_filling(
    primal: PeriodicQuotientGraph,
    tiling: PeriodicTiling,
    limits: NetAnalysisLimits,
) -> _Filling | None:
    """Certify a convex periodic tiling of Euclidean 3-space.

    ``None`` means this input is not a space-filling tiling under the rules
    below, and the closed-surface certificate may still apply. A returned
    refusal is final: the surface test must not turn it into a dual.
    """
    if _duplicate_face(tiling) is not None:
        return _Filling(False, "duplicate_face", {})
    shift_failure = _shift_rule(tiling)
    if shift_failure == "inconsistent_adjacency_shift":
        return _Filling(False, shift_failure, {})
    if shift_failure is not None:
        return None
    analysis = analyze_periodic_graph(primal)
    if analysis.periodic_rank != 3 or len(analysis.quotient_components) != 1:
        return None
    family = analysis.cover_families[0]
    if family.subgroup.rank != 3:
        return None
    positions = _node_positions(primal)
    if isinstance(positions, str):
        return _Filling(False, "tiling_placement_undetermined", {})
    assembled = _assemble_tiles(tiling, limits.tiling_assembly_limit)
    if isinstance(assembled, str):
        if assembled in {"tiling_assembly_capped", "tiling_assembly_ambiguous"}:
            return _Filling(False, assembled, {})
        return None
    cell = abs(det_basis(family.subgroup.basis))
    euler = (
        len(primal.nodes) - len(primal.edges) + len(tiling.faces) - len(tiling.tiles)
    )
    if euler != 0:
        return None
    choices = _assembly_choices(assembled)
    if isinstance(choices, str):
        return _Filling(False, choices, {})
    accepted: list[dict[str, object]] = []
    for choice in choices:
        placed = _place_tiles(choice, tiling)
        if isinstance(placed, str):
            continue
        if _convex_tiles(placed, positions) is not None:
            continue
        links = _edge_links(primal, placed)
        if links is None or any(
            item["faces"] < 3 or not item["cycle"] for item in links
        ):
            continue
        vertex_links = _vertex_links(primal, placed)
        if vertex_links is None:
            continue
        volumes = _tile_volumes(placed, positions)
        if any(volume <= 0 for _tile, volume in volumes):
            continue
        total = sum((volume for _tile, volume in volumes), Fraction(0))
        if total != Fraction(cell):
            continue
        if not _interiors_disjoint(placed, positions, family.subgroup.basis):
            continue
        accepted.append(
            {
                "kind": "space_filling_tiling",
                "incidence": _incidence_rows(tiling),
                "euler_characteristic": euler,
                "cell_volume": str(Fraction(cell)),
                "tile_volumes": [
                    {"tile_id": tile_id, "volume": str(volume)}
                    for tile_id, volume in volumes
                ],
                "edge_links": links,
                "vertex_links": vertex_links,
                "quotient_edge_face_orbits": _orbit_face_counts(tiling),
            }
        )
        break
    if accepted:
        return _Filling(True, "space_filling_tiling", accepted[0])
    return None


def _duplicate_face(tiling: PeriodicTiling) -> str | None:
    seen: dict[tuple[object, ...], str] = {}
    for face in tiling.faces:
        key = _canonical_unoriented_face(face.vertices)
        if key in seen:
            return "duplicate_face"
        seen[key] = face.face_id
    return None


def _shift_rule(tiling: PeriodicTiling) -> str | None:
    """Reject a face with no lattice plane, or a null self-gluing.

    A zero shift is allowed when two distinct tiles share a face inside one
    cell. A zero shift from a tile to itself has no separation to check.
    Other shifts are accepted here and checked against the developed faces.
    """
    by_face = {record.face_id: record for record in tiling.adjacencies}
    for face in tiling.faces:
        if _oriented_normal(face.vertices) is None:
            return "incomplete_tiling"
        adjacency = by_face[face.face_id]
        if (
            adjacency.image_shift == (0, 0, 0)
            and adjacency.source_tile == adjacency.target_tile
        ):
            return "inconsistent_adjacency_shift"
    return None


def _node_positions(primal: PeriodicQuotientGraph):
    from .isomorphism import _barycentric_positions

    solved = _barycentric_positions(primal)
    if isinstance(solved, str):
        return solved
    return {node_id: solved[index] for index, node_id in enumerate(primal.node_ids)}


_Point = tuple[str, ImageShift]
_Cycle = tuple[_Point, ...]


@dataclass(frozen=True)
class _Slot:
    face_id: str
    side: str
    shift: ImageShift
    base: _Cycle


@dataclass(frozen=True)
class _PlacedFace:
    face_id: str
    side: str
    cycle: _Cycle


@dataclass(frozen=True)
class _PlacedTile:
    tile_id: str
    faces: tuple[_PlacedFace, ...]


def _boundary_cycle(
    vertices: tuple[tuple[str, ImageShift], ...],
    side: str,
    shift: ImageShift,
    translation: ImageShift,
    normal: ImageShift,
) -> _Cycle:
    """Orient one side so the listed right-hand normal leaves the source tile.

    ``shift`` and ``normal`` are not required to be parallel. A unimodular
    change of basis rewrites shifts as vectors and normals as cross products,
    so equality of those two presentations is not an invariant. The placement
    check is what ties ``shift`` to the developed faces.
    """
    del shift, normal
    ordered = vertices if side == "source" else tuple(reversed(vertices))
    return tuple((node_id, add(image, translation)) for node_id, image in ordered)


def _assembly_choices(
    assembled: dict[str, list[dict[tuple[str, str], _Cycle]]],
) -> list[dict[str, dict[tuple[str, str], _Cycle]]] | str:
    import itertools

    lists = [assembled[tile_id] for tile_id in assembled]
    count = 1
    for item in lists:
        count *= len(item)
    if count > 8:
        return "tiling_assembly_ambiguous"
    return [
        dict(zip(assembled, combo, strict=True)) for combo in itertools.product(*lists)
    ]


def _assemble_tiles(
    tiling: PeriodicTiling, limit: int
) -> dict[str, list[dict[tuple[str, str], _Cycle]]] | str:
    faces = {face.face_id: face for face in tiling.faces}
    normals = {face.face_id: _oriented_normal(face.vertices) for face in tiling.faces}
    assembled: dict[str, dict[tuple[str, str], _Cycle]] = {}
    for tile_id in tiling.tiles:
        slots: list[_Slot] = []
        for adjacency in tiling.adjacencies:
            normal = normals[adjacency.face_id]
            if normal is None:
                return "incomplete_tiling"
            vertices = faces[adjacency.face_id].vertices
            if adjacency.source_tile == tile_id:
                slots.append(
                    _Slot(
                        adjacency.face_id,
                        "source",
                        adjacency.image_shift,
                        _boundary_cycle(
                            vertices, "source", adjacency.image_shift, (0, 0, 0), normal
                        ),
                    )
                )
            if adjacency.target_tile == tile_id:
                slots.append(
                    _Slot(
                        adjacency.face_id,
                        "target",
                        adjacency.image_shift,
                        _boundary_cycle(
                            vertices, "target", adjacency.image_shift, (0, 0, 0), normal
                        ),
                    )
                )
        if len(slots) < 4:
            return "incomplete_tiling"
        solution = _assemble_boundary(slots, limit)
        if isinstance(solution, str):
            return solution
        assembled[tile_id] = solution
    return assembled


def _assemble_boundary(
    slots: list[_Slot], limit: int
) -> list[dict[tuple[str, str], _Cycle]] | str:
    solutions: list[dict[tuple[str, str], _Cycle]] = []
    nodes = 0

    def tally(placed: dict[int, _Cycle]):
        counts: dict[tuple[_Point, _Point], int] = {}
        forward: dict[tuple[_Point, _Point], tuple[_Point, _Point]] = {}
        for cycle in placed.values():
            for edge in _cycle_edges(cycle):
                key = _sorted_edge(edge)
                counts[key] = counts.get(key, 0) + 1
                forward[key] = edge
        return counts, forward

    def search(placed: dict[int, _Cycle]) -> str | None:
        nonlocal nodes
        nodes += 1
        if nodes > limit:
            return "tiling_assembly_capped"
        if len(solutions) > 4:
            return "tiling_assembly_ambiguous"
        counts, forward = tally(placed)
        if any(count > 2 for count in counts.values()):
            return None
        if len(placed) == len(slots):
            if any(count != 2 for count in counts.values()):
                return None
            vertices = {point for cycle in placed.values() for point in cycle}
            euler = len(vertices) - len(counts) + len(slots)
            if euler != 2:
                return None
            solutions.append(
                {
                    (slots[index].face_id, slots[index].side): cycle
                    for index, cycle in placed.items()
                }
            )
            return None
        exposed = [key for key, count in counts.items() if count == 1]
        if not exposed:
            return None
        want_src, want_dst = forward[exposed[0]][1], forward[exposed[0]][0]
        hits: list[tuple[int, _Cycle]] = []
        for index, slot in enumerate(slots):
            if index in placed:
                continue
            seen: set[ImageShift] = set()
            for src, dst in _cycle_edges(slot.base):
                if src[0] != want_src[0] or dst[0] != want_dst[0]:
                    continue
                translation = sub(want_src[1], src[1])
                if add(dst[1], translation) != want_dst[1]:
                    continue
                if translation in seen:
                    continue
                seen.add(translation)
                cycle = tuple(
                    (node_id, add(image, translation)) for node_id, image in slot.base
                )
                if any(
                    frozenset(cycle) == frozenset(other) for other in placed.values()
                ):
                    continue
                hits.append((index, cycle))
        for index, cycle in hits:
            placed[index] = cycle
            failure = search(placed)
            del placed[index]
            if failure is not None:
                return failure
        return None

    failure = search({0: slots[0].base})
    if failure is not None:
        return failure
    if not solutions:
        return "incomplete_tiling"
    return solutions


def _place_tiles(
    assembled: dict[str, dict[tuple[str, str], _Cycle]],
    tiling: PeriodicTiling,
) -> tuple[_PlacedTile, ...] | str:
    """Put every tile in one frame and require each shift to match that frame."""
    origin: dict[str, ImageShift] = {tiling.tiles[0]: (0, 0, 0)}
    pending = [tiling.tiles[0]]
    checked: set[str] = set()
    while pending or len(checked) < len(tiling.adjacencies):
        if not pending:
            missing = [
                adjacency.face_id
                for adjacency in tiling.adjacencies
                if adjacency.face_id not in checked
            ]
            if not missing:
                break
            return "incomplete_tiling"
        current = pending.pop()
        for adjacency in tiling.adjacencies:
            if adjacency.face_id in checked:
                continue
            if current not in {adjacency.source_tile, adjacency.target_tile}:
                continue
            source_cycle = assembled[adjacency.source_tile][
                (adjacency.face_id, "source")
            ]
            target_cycle = assembled[adjacency.target_tile][
                (adjacency.face_id, "target")
            ]
            relation = _translation_between(target_cycle, source_cycle)
            if relation is None:
                return "inconsistent_adjacency_shift"
            source_known = adjacency.source_tile in origin
            target_known = adjacency.target_tile in origin
            if source_known and target_known:
                predicted = sub(
                    add(relation, origin[adjacency.source_tile]),
                    origin[adjacency.target_tile],
                )
                if predicted != adjacency.image_shift:
                    return "inconsistent_adjacency_shift"
                checked.add(adjacency.face_id)
                continue
            if source_known and not target_known:
                origin[adjacency.target_tile] = sub(
                    add(relation, origin[adjacency.source_tile]),
                    adjacency.image_shift,
                )
                pending.append(adjacency.target_tile)
                checked.add(adjacency.face_id)
                continue
            if target_known and not source_known:
                # relation + origin_source - origin_target == shift
                # origin_source == shift + origin_target - relation
                origin[adjacency.source_tile] = sub(
                    add(adjacency.image_shift, origin[adjacency.target_tile]),
                    relation,
                )
                pending.append(adjacency.source_tile)
                checked.add(adjacency.face_id)
                continue
    if len(origin) != len(tiling.tiles) or len(checked) != len(tiling.adjacencies):
        return "incomplete_tiling"
    placed: list[_PlacedTile] = []
    for tile_id in tiling.tiles:
        faces = []
        for (face_id, side), cycle in sorted(assembled[tile_id].items()):
            moved = tuple(
                (node_id, add(image, origin[tile_id])) for node_id, image in cycle
            )
            faces.append(_PlacedFace(face_id, side, moved))
        placed.append(_PlacedTile(tile_id, tuple(faces)))
    return tuple(placed)


def _translation_between(source: _Cycle, target: _Cycle) -> ImageShift | None:
    """Return ``D`` such that translating ``source`` by ``D`` yields ``target``."""
    if len(source) != len(target):
        return None
    anchor = source[0]
    found: set[ImageShift] = set()
    target_set = frozenset(target)
    for node_id, image in target:
        if node_id != anchor[0]:
            continue
        delta = sub(image, anchor[1])
        moved = frozenset((node, add(img, delta)) for node, img in source)
        if moved == target_set:
            found.add(delta)
    if len(found) != 1:
        return None
    return next(iter(found))


def _cycle_edges(cycle: _Cycle) -> tuple[tuple[_Point, _Point], ...]:
    return tuple(zip(cycle, cycle[1:] + cycle[:1], strict=True))


def _sorted_edge(edge: tuple[_Point, _Point]) -> tuple[_Point, _Point]:
    if edge[1] < edge[0]:
        return (edge[1], edge[0])
    return edge


def _convex_tiles(placed: tuple[_PlacedTile, ...], positions) -> str | None:
    for tile in placed:
        if _tile_is_convex(tile, positions):
            continue
        return "nonconvex_tile"
    return None


def _tile_is_convex(tile: _PlacedTile, positions) -> bool:
    points = _unique_points(tile, positions)
    if len(points) < 4:
        return False
    centroid = _centroid(points)
    for face in tile.faces:
        located = [_locate(point, positions, (0, 0, 0)) for point in face.cycle]
        normal = _polygon_normal(located)
        if normal is None or normal == (Fraction(0), Fraction(0), Fraction(0)):
            return False
        origin = located[0]
        signs = {
            _sign(_fdot(_fsub(point, origin), normal))
            for point in points
            if _fdot(_fsub(point, origin), normal) != 0
        }
        if signs != {-1}:
            return False
        inward = _fdot(_fsub(centroid, origin), normal)
        if inward >= 0:
            return False
    return True


def _edge_links(primal: PeriodicQuotientGraph, placed: tuple[_PlacedTile, ...]):
    rows = []
    for edge in primal.edges:
        representative = _representative_cover_edge(edge, placed)
        if representative is None:
            return None
        incidences = _incidences_on_edge(representative, placed)
        tiles = {(item[0], item[1]) for item in incidences}
        # Bipartite vertices: tile copies and geometric faces.
        geometric: dict[frozenset[_Point], set[tuple[str, ImageShift]]] = {}
        tile_faces: dict[tuple[str, ImageShift], set[frozenset[_Point]]] = {}
        for tile_id, translation, _face_id, geometry in incidences:
            geometric.setdefault(geometry, set()).add((tile_id, translation))
            tile_faces.setdefault((tile_id, translation), set()).add(geometry)
        if any(len(owners) != 2 for owners in geometric.values()):
            return None
        if any(len(members) != 2 for members in tile_faces.values()):
            return None
        if not _bipartite_cycle(geometric, tile_faces):
            return None
        rows.append(
            {
                "edge_id": edge.edge_id,
                "faces": len(geometric),
                "tiles": len(tiles),
                "cycle": True,
            }
        )
    rows.sort(key=lambda item: str(item["edge_id"]))
    return rows


def _representative_cover_edge(edge, placed: tuple[_PlacedTile, ...]):
    wanted = _undirected_edge(edge.source, edge.target, edge.image_shift)
    for tile in placed:
        for face in tile.faces:
            for src, dst in _cycle_edges(face.cycle):
                step = sub(dst[1], src[1])
                if _undirected_edge(src[0], dst[0], step) == wanted:
                    return (src, dst)
    return None


def _incidences_on_edge(representative, placed: tuple[_PlacedTile, ...]):
    src_rep, dst_rep = representative
    found = []
    seen = set()
    for tile in placed:
        for face in tile.faces:
            for src, dst in _cycle_edges(face.cycle):
                for left, right, want_left, want_right in (
                    (src, dst, src_rep, dst_rep),
                    (src, dst, dst_rep, src_rep),
                ):
                    if left[0] != want_left[0] or right[0] != want_right[0]:
                        continue
                    translation = sub(want_left[1], left[1])
                    if add(right[1], translation) != want_right[1]:
                        continue
                    geometry = frozenset(
                        (node_id, add(image, translation))
                        for node_id, image in face.cycle
                    )
                    key = (tile.tile_id, translation, face.face_id, geometry)
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(key)
    return found


def _bipartite_cycle(geometric, tile_faces) -> bool:
    if not geometric:
        return False
    graph: dict[object, set[object]] = {}
    for geometry, owners in geometric.items():
        graph.setdefault(("f", geometry), set())
        for owner in owners:
            graph.setdefault(("t", owner), set()).add(("f", geometry))
            graph[("f", geometry)].add(("t", owner))
    if any(len(neighbors) != 2 for neighbors in graph.values()):
        return False
    start = next(iter(graph))
    seen = {start}
    previous = None
    current = start
    while True:
        nxt = next(neighbor for neighbor in graph[current] if neighbor != previous)
        if nxt == start:
            return len(seen) == len(graph)
        if nxt in seen:
            return False
        seen.add(nxt)
        previous, current = current, nxt


def _vertex_links(primal: PeriodicQuotientGraph, placed: tuple[_PlacedTile, ...]):
    rows = []
    for node_id in primal.node_ids:
        copies = _tiles_at_vertex((node_id, (0, 0, 0)), placed)
        if not copies:
            return None
        link_vertices: set[tuple[_Point, _Point]] = set()
        link_edges: dict[tuple[_Point, _Point], int] = {}
        for _tile_id, _translation, faces in copies:
            corners = []
            for face in faces:
                corner = _corner_at(face, node_id, (0, 0, 0))
                if corner is None:
                    continue
                left, right = corner
                key = _sorted_edge((left, right))
                link_vertices.add(left)
                link_vertices.add(right)
                link_edges[key] = link_edges.get(key, 0) + 1
                corners.append(key)
            if len(corners) < 3:
                return None
        if not link_edges or any(count != 2 for count in link_edges.values()):
            return None
        euler = len(link_vertices) - len(link_edges) + len(copies)
        if euler != 2 or not _link_connected(link_edges):
            return None
        rows.append(
            {
                "node_id": node_id,
                "edges": len(link_vertices),
                "corners": len(link_edges),
                "tiles": len(copies),
                "euler_characteristic": euler,
            }
        )
    return rows


def _tiles_at_vertex(vertex: _Point, placed: tuple[_PlacedTile, ...]):
    found = []
    seen = set()
    for tile in placed:
        for face in tile.faces:
            for node_id, image in face.cycle:
                if node_id != vertex[0]:
                    continue
                translation = sub(vertex[1], image)
                key = (tile.tile_id, translation)
                if key in seen:
                    continue
                moved = []
                contains = False
                for placed_face in tile.faces:
                    cycle = tuple(
                        (item_node, add(item_image, translation))
                        for item_node, item_image in placed_face.cycle
                    )
                    if vertex in cycle:
                        contains = True
                    moved.append(cycle)
                if not contains:
                    continue
                seen.add(key)
                found.append((tile.tile_id, translation, tuple(moved)))
    return found


def _corner_at(cycle: _Cycle, node_id: str, image: ImageShift):
    size = len(cycle)
    for index, point in enumerate(cycle):
        if point != (node_id, image):
            continue
        return cycle[(index - 1) % size], cycle[(index + 1) % size]
    return None


def _link_connected(edges: dict[tuple[_Point, _Point], int]) -> bool:
    if not edges:
        return False
    neighbors: dict[_Point, set[_Point]] = {}
    for left, right in edges:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    start = next(iter(neighbors))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in neighbors[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(neighbors)


def _tile_volumes(placed, positions) -> list[tuple[str, Fraction]]:
    rows = []
    for tile in placed:
        total = Fraction(0)
        for face in tile.faces:
            located = [_locate(point, positions, (0, 0, 0)) for point in face.cycle]
            origin = located[0]
            for index in range(1, len(located) - 1):
                total += _fdot(origin, _fcross(located[index], located[index + 1]))
        rows.append((tile.tile_id, total / 6))
    return rows


def _interiors_disjoint(placed, positions, basis: tuple[ImageShift, ...]) -> bool:
    window = _neighbor_window(placed, positions, basis)
    bodies = [(tile, (0, 0, 0)) for tile in placed]
    translated = []
    for tile in placed:
        for vector in window:
            translated.append((tile, vector))
    for left_tile, left_shift in bodies:
        for right_tile, right_shift in translated:
            if not _pair_key_fresh(left_tile, right_tile, right_shift):
                continue
            if not _separated(
                left_tile, left_shift, right_tile, right_shift, positions
            ):
                return False
    return True


def _pair_key_fresh(left_tile, right_tile, right_shift) -> bool:
    """Skip comparing a representative tile with itself."""
    return not (left_tile.tile_id == right_tile.tile_id and right_shift == (0, 0, 0))


def _neighbor_window(placed, positions, basis: tuple[ImageShift, ...]):
    span = 1
    for tile in placed:
        points = _unique_points(tile, positions)
        for axis in range(3):
            values = [point[axis] for point in points]
            width = max(values) - min(values)
            span = max(span, int(width) + 1)
    vectors = []
    coefficients = range(-span, span + 1)
    for i in coefficients:
        for j in coefficients:
            for k in coefficients:
                if (i, j, k) == (0, 0, 0):
                    continue
                vectors.append(_combine_basis(basis, i, j, k))
    # Also the zero vector so two distinct representatives in one cell are compared.
    vectors.append((0, 0, 0))
    return vectors


def _combine_basis(basis, i, j, k) -> ImageShift:
    first, second, third = basis
    return (
        i * first[0] + j * second[0] + k * third[0],
        i * first[1] + j * second[1] + k * third[1],
        i * first[2] + j * second[2] + k * third[2],
    )


def _separated(left, left_shift, right, right_shift, positions) -> bool:
    return _outside(left, left_shift, right, right_shift, positions) or _outside(
        right, right_shift, left, left_shift, positions
    )


def _outside(host, host_shift, other, other_shift, positions) -> bool:
    other_points = [
        _locate(point, positions, other_shift) for point in _tile_points(other)
    ]
    for face in host.faces:
        located = [_locate(point, positions, host_shift) for point in face.cycle]
        normal = _polygon_normal(located)
        if normal is None:
            return False
        origin = located[0]
        if all(_fdot(_fsub(point, origin), normal) >= 0 for point in other_points):
            return True
    return False


def _orbit_face_counts(tiling: PeriodicTiling) -> list[int]:
    counts: dict[tuple[str, str, ImageShift], int] = {}
    for face in tiling.faces:
        seen: set[tuple[str, str, ImageShift]] = set()
        steps = list(face.vertices) + [face.vertices[0]]
        for (source, source_image), (target, target_image) in zip(steps, steps[1:]):
            seen.add(_undirected_edge(source, target, sub(target_image, source_image)))
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.values())


def _tile_points(tile: _PlacedTile) -> list[_Point]:
    points = []
    seen = set()
    for face in tile.faces:
        for point in face.cycle:
            if point not in seen:
                seen.add(point)
                points.append(point)
    return points


def _unique_points(
    tile: _PlacedTile, positions
) -> list[tuple[Fraction, Fraction, Fraction]]:
    return [_locate(point, positions, (0, 0, 0)) for point in _tile_points(tile)]


def _centroid(points):
    count = Fraction(len(points))
    return (
        sum((point[0] for point in points), Fraction(0)) / count,
        sum((point[1] for point in points), Fraction(0)) / count,
        sum((point[2] for point in points), Fraction(0)) / count,
    )


def _locate(point: _Point, positions, translation: ImageShift):
    base = positions[point[0]]
    image = point[1]
    return (
        base[0] + image[0] + translation[0],
        base[1] + image[1] + translation[1],
        base[2] + image[2] + translation[2],
    )


def _polygon_normal(points):
    if len(points) < 3:
        return None
    origin = points[0]
    total = (Fraction(0), Fraction(0), Fraction(0))
    for index in range(1, len(points) - 1):
        total = _fadd(
            total,
            _fcross(_fsub(points[index], origin), _fsub(points[index + 1], origin)),
        )
    return total


def _fadd(left, right):
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _fsub(left, right):
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _fdot(left, right) -> Fraction:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _fcross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _sign(value: Fraction) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def det_basis(basis: tuple[ImageShift, ...]) -> int:
    if len(basis) != 3:
        return 0
    return _dot(basis[0], _cross(basis[1], basis[2]))


__all__ = [
    "DualityAssessment",
    "PeriodicTiling",
    "TilingAdjacency",
    "TilingFace",
    "assess_duality",
]
