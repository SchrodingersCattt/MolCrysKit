"""Duality follows a validated tiling, never a coordination sequence."""

from __future__ import annotations

from molcrys_kit.nets.duality import (
    PeriodicTiling,
    TilingAdjacency,
    TilingFace,
    assess_duality,
)
from molcrys_kit.nets.limits import NetAnalysisLimits
from molcrys_kit.periodic import (
    PeriodicQuotientGraph,
    apply_node_gauge,
    apply_unimodular_basis,
    equivalent_supercell,
)
from molcrys_kit.periodic._lattice import adjugate, det3, vecmul


def _graph(nodes, edges):
    return PeriodicQuotientGraph.from_edges(nodes, edges)


def _sql():
    return _graph(("a",), [("ax", "a", "a", (1, 0, 0)), ("ay", "a", "a", (0, 1, 0))])


def _pcu():
    return _graph(
        ("a",),
        [
            ("ax", "a", "a", (1, 0, 0)),
            ("ay", "a", "a", (0, 1, 0)),
            ("az", "a", "a", (0, 0, 1)),
        ],
    )


def _chain():
    return _graph(("a",), [("wind", "a", "a", (1, 0, 0))])


def _add(left, right):
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _lattice_face(face_id, origin, along, across):
    """Square whose right-hand normal is ``along × across``."""
    return TilingFace(
        face_id,
        (
            ("a", origin),
            ("a", _add(origin, along)),
            ("a", _add(origin, _add(along, across))),
            ("a", _add(origin, across)),
        ),
    )


def _cube_tiling(shifts=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    """Quotient of the primitive cubic tiling.

    Three face orbits meet two-to-an-edge. Each adjacency shift is the
    right-hand normal of that face, so the dual is again the cubic net.
    """
    return PeriodicTiling(
        ("t",),
        (
            _lattice_face("fx", (0, 0, 0), (0, 1, 0), (0, 0, 1)),
            _lattice_face("fy", (0, 0, 0), (0, 0, 1), (1, 0, 0)),
            _lattice_face("fz", (0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ),
        (
            TilingAdjacency("fx", "t", "t", shifts[0]),
            TilingAdjacency("fy", "t", "t", shifts[1]),
            TilingAdjacency("fz", "t", "t", shifts[2]),
        ),
    )


def test_cube_tiling_is_dual_to_its_net_and_not_to_a_chain():
    primal = _pcu()
    tiling = _cube_tiling()
    dual = assess_duality(primal, primal, tiling)
    assert dual.status == "dual", dual.reason
    assert dual.witness is not None
    assert dual.witness["kind"] == "space_filling_tiling"
    assert dual.witness["tile_volumes"] == [{"tile_id": "t", "volume": "1"}]
    assert dual.witness["quotient_edge_face_orbits"] == [2, 2, 2]
    assert {row["faces"] for row in dual.witness["edge_links"]} == {4}
    assert dual.witness["vertex_links"][0]["euler_characteristic"] == 2
    incidence = dual.witness["incidence"]
    assert {row["face_id"] for row in incidence} == {"fx", "fy", "fz"}
    assert dual.witness["equivalence"]["proof"] == "explicit_isomorphism"
    other = assess_duality(primal, _chain(), tiling)
    assert other.status == "not_dual"


def test_duplicate_square_with_forged_shifts_is_not_a_proved_dual():
    """The same square listed twice is not a closed surface.

    Shifts ``(1, 0, 0)`` and ``(2, 0, 0)`` are not the face normal. The
    one-node graph with those two loops is rank 1. It must not be certified
    as dual to the rank-2 square lattice.
    """
    square = (
        ("a", (0, 0, 0)),
        ("a", (1, 0, 0)),
        ("a", (1, 1, 0)),
        ("a", (0, 1, 0)),
    )
    tiling = PeriodicTiling(
        ("t",),
        (TilingFace("fx", square), TilingFace("fy", square)),
        (
            TilingAdjacency("fx", "t", "t", (1, 0, 0)),
            TilingAdjacency("fy", "t", "t", (2, 0, 0)),
        ),
    )
    other = _graph(
        ("t",),
        [("e1", "t", "t", (1, 0, 0)), ("e2", "t", "t", (2, 0, 0))],
    )
    assessment = assess_duality(_sql(), other, tiling)
    assert assessment.status == "undetermined"
    assert assessment.status != "dual"
    assert assessment.reason == "duplicate_face"


def test_adjacency_shift_must_match_the_face_normal():
    primal = _pcu()
    tiling = _cube_tiling(((1, 0, 0), (2, 0, 0), (0, 0, 1)))
    other = _graph(
        ("t",),
        [
            ("ex", "t", "t", (1, 0, 0)),
            ("ey", "t", "t", (2, 0, 0)),
            ("ez", "t", "t", (0, 0, 1)),
        ],
    )
    assessment = assess_duality(primal, other, tiling)
    assert assessment.status == "undetermined"
    assert assessment.status != "dual"
    assert assessment.reason == "inconsistent_adjacency_shift"


def test_open_face_list_is_not_a_proved_dual():
    primal = _graph(
        ("a", "b", "c", "d"),
        [
            ("ab", "a", "b", (0, 0, 0)),
            ("bc", "b", "c", (0, 0, 0)),
            ("cd", "c", "d", (0, 0, 0)),
            ("da", "d", "a", (0, 0, 0)),
        ],
    )
    face = TilingFace(
        "open",
        (
            ("a", (0, 0, 0)),
            ("b", (0, 0, 0)),
            ("c", (0, 0, 0)),
            ("d", (0, 0, 0)),
        ),
    )
    tiling = PeriodicTiling(
        ("u", "v"),
        (face,),
        (TilingAdjacency("open", "u", "v", (0, 0, 0)),),
    )
    other = _graph(("u", "v"), [("uv", "u", "v", (0, 0, 0))])
    assessment = assess_duality(primal, other, tiling)
    assert assessment.status == "undetermined"
    assert assessment.reason == "incomplete_tiling"
    assert assessment.status != "dual"


def test_missing_tiling_is_not_assessed_and_a_broken_cycle_is_undetermined():
    primal = _sql()
    assert assess_duality(primal, primal, None).status == "not_assessed"
    broken = PeriodicTiling(
        ("t",),
        (
            TilingFace(
                "bad",
                (
                    ("a", (0, 0, 0)),
                    ("a", (2, 0, 0)),
                    ("a", (2, 1, 0)),
                    ("a", (0, 1, 0)),
                ),
            ),
        ),
        (TilingAdjacency("bad", "t", "t", (1, 0, 0)),),
    )
    assessment = assess_duality(primal, primal, broken)
    assert assessment.status == "undetermined"
    assert assessment.reason == "face_not_a_cover_cycle"


def test_capped_isomorphism_leaves_duality_undetermined():
    primal = _pcu()
    limits = NetAnalysisLimits.from_config().to_dict()
    limits["backtrack_limit"] = 0
    assessment = assess_duality(
        primal, primal, _cube_tiling(), NetAnalysisLimits(**limits)
    )
    assert assessment.status == "undetermined"


def _face(face_id, vertices):
    return TilingFace(face_id, tuple(("a", vertex) for vertex in vertices))


def _prism_tiling():
    """Two triangular prisms that fill one cubic cell.

    The diagonal face is shared inside the cell, so its adjacency shift is
    zero while its normal is ``(1, 1, 0)``. Every quotient edge lies in three
    face orbits. The closed-surface rule, which demands two orbits and a
    normal shift, does not apply.
    """
    return PeriodicTiling(
        ("A", "B"),
        (
            _face("triA", ((0, 0, 0), (1, 0, 0), (0, 1, 0))),
            _face("triB", ((1, 0, 0), (1, 1, 0), (0, 1, 0))),
            _face("wallX", ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
            _face("wallY", ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
            _face("cut", ((1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1))),
        ),
        (
            TilingAdjacency("triA", "A", "A", (0, 0, 1)),
            TilingAdjacency("triB", "B", "B", (0, 0, 1)),
            TilingAdjacency("wallX", "A", "B", (-1, 0, 0)),
            TilingAdjacency("wallY", "A", "B", (0, -1, 0)),
            TilingAdjacency("cut", "A", "B", (0, 0, 0)),
        ),
    )


def _prism_skeleton():
    return _graph(
        ("a",),
        [
            ("ex", "a", "a", (1, 0, 0)),
            ("ey", "a", "a", (0, 1, 0)),
            ("ez", "a", "a", (0, 0, 1)),
            ("ed", "a", "a", (1, -1, 0)),
        ],
    )


def _prism_dual():
    return _graph(
        ("A", "B"),
        [
            ("zA", "A", "A", (0, 0, 1)),
            ("zB", "B", "B", (0, 0, 1)),
            ("x", "A", "B", (-1, 0, 0)),
            ("y", "A", "B", (0, -1, 0)),
            ("c", "A", "B", (0, 0, 0)),
        ],
    )


def _transform_shift(shift, matrix):
    determinant = det3(matrix)
    scaled = vecmul(shift, adjugate(matrix))
    return tuple(component // determinant for component in scaled)


def _transform_tiling(tiling, matrix):
    faces = tuple(
        TilingFace(
            face.face_id,
            tuple(
                (node_id, _transform_shift(image, matrix))
                for node_id, image in face.vertices
            ),
        )
        for face in tiling.faces
    )
    adjacencies = tuple(
        TilingAdjacency(
            adjacency.face_id,
            adjacency.source_tile,
            adjacency.target_tile,
            _transform_shift(adjacency.image_shift, matrix),
        )
        for adjacency in tiling.adjacencies
    )
    return PeriodicTiling(tiling.tiles, faces, adjacencies)


def _gauge_tiling(tiling, gauge):
    faces = tuple(
        TilingFace(
            face.face_id,
            tuple((node_id, _add(image, gauge)) for node_id, image in face.vertices),
        )
        for face in tiling.faces
    )
    return PeriodicTiling(tiling.tiles, faces, tiling.adjacencies)


def test_prism_tiling_is_dual_with_a_recoverable_incidence():
    primal = _prism_skeleton()
    tiling = _prism_tiling()
    dual = assess_duality(primal, _prism_dual(), tiling)
    assert dual.status == "dual", dual.reason
    assert dual.witness["kind"] == "space_filling_tiling"
    assert dual.witness["quotient_edge_face_orbits"] == [3, 3, 3, 3]
    assert min(row["faces"] for row in dual.witness["edge_links"]) >= 3
    assert {row["volume"] for row in dual.witness["tile_volumes"]} == {"1/2"}
    by_face = {row["face_id"]: row for row in dual.witness["incidence"]}
    assert by_face["cut"]["image_shift"] == [0, 0, 0]
    assert by_face["cut"]["source_tile"] == "A"
    assert by_face["cut"]["target_tile"] == "B"
    assert by_face["cut"]["dual_edge_id"] == "cut"
    assert dual.witness["equivalence"]["proof"] == "explicit_isomorphism"
    assert dual.witness["equivalence"]["status"] == "isomorphic"
    different = assess_duality(primal, _prism_skeleton(), tiling)
    assert different.status == "not_dual"
    assert different.status != "dual"
    missing = assess_duality(primal, _prism_dual(), None)
    assert missing.status == "not_assessed"


def test_prism_duality_is_stable_under_reordering_gauge_basis_and_supercell():
    primal = _prism_skeleton()
    tiling = _prism_tiling()
    dual = _prism_dual()
    matrix = ((1, 1, 0), (0, 1, 0), (0, 0, 1))
    reordered = PeriodicTiling(
        tiling.tiles,
        tuple(reversed(tiling.faces)),
        tuple(reversed(tiling.adjacencies)),
    )
    gauged_graph = apply_node_gauge(primal, {"a": (2, -1, 3)}).graph
    gauged_tiling = _gauge_tiling(tiling, (2, -1, 3))
    changed_graph = apply_unimodular_basis(primal, matrix).graph
    changed_tiling = _transform_tiling(tiling, matrix)
    changed_dual = apply_unimodular_basis(dual, matrix).graph
    supercell = equivalent_supercell(dual, ((2, 0, 0), (0, 1, 0), (0, 0, 1))).graph
    cases = (
        (primal, dual, reordered),
        (gauged_graph, dual, gauged_tiling),
        (changed_graph, changed_dual, changed_tiling),
        (primal, supercell, tiling),
    )
    for left, right, faces in cases:
        assessment = assess_duality(left, right, faces)
        assert assessment.status == "dual", assessment.reason
        assert assessment.witness["kind"] == "space_filling_tiling"


def test_unfilled_prism_and_capped_assembly_are_not_dual():
    primal = _prism_skeleton()
    tiling = _prism_tiling()
    faces = tuple(face for face in tiling.faces if face.face_id != "cut")
    adjacencies = tuple(
        adjacency for adjacency in tiling.adjacencies if adjacency.face_id != "cut"
    )
    incomplete = PeriodicTiling(tiling.tiles, faces, adjacencies)
    assessment = assess_duality(primal, _prism_dual(), incomplete)
    assert assessment.status == "undetermined"
    assert assessment.reason == "incomplete_tiling"
    assert assessment.status != "dual"
    limits = NetAnalysisLimits.from_config().to_dict()
    limits["tiling_assembly_limit"] = 1
    capped = assess_duality(_pcu(), _pcu(), _cube_tiling(), NetAnalysisLimits(**limits))
    assert capped.status == "undetermined"
    assert capped.reason == "tiling_assembly_capped"
    assert capped.status != "dual"
