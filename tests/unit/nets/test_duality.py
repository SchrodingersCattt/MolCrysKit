"""Duality follows a validated tiling, never a coordination sequence."""

from __future__ import annotations

from molcrys_kit.nets.duality import (
    PeriodicTiling,
    TilingAdjacency,
    TilingFace,
    assess_duality,
)
from molcrys_kit.nets.limits import NetAnalysisLimits
from molcrys_kit.periodic import PeriodicQuotientGraph


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
