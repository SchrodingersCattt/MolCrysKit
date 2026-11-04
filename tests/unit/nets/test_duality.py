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


def _chain():
    return _graph(("a",), [("wind", "a", "a", (1, 0, 0))])


def _square_face(face_id):
    return TilingFace(
        face_id,
        (
            ("a", (0, 0, 0)),
            ("a", (1, 0, 0)),
            ("a", (1, 1, 0)),
            ("a", (0, 1, 0)),
        ),
    )


def _self_tiling():
    return PeriodicTiling(
        ("t",),
        (_square_face("fx"), _square_face("fy")),
        (
            TilingAdjacency("fx", "t", "t", (1, 0, 0)),
            TilingAdjacency("fy", "t", "t", (0, 1, 0)),
        ),
    )


def test_square_tiling_is_dual_to_its_net_and_not_to_a_chain():
    primal = _sql()
    tiling = _self_tiling()
    dual = assess_duality(primal, primal, tiling)
    assert dual.status == "dual", dual.reason
    assert dual.witness is not None
    other = assess_duality(primal, _chain(), tiling)
    assert other.status == "not_dual"


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
    primal = _sql()
    limits = NetAnalysisLimits.from_config().to_dict()
    limits["backtrack_limit"] = 0
    assessment = assess_duality(
        primal, primal, _self_tiling(), NetAnalysisLimits(**limits)
    )
    assert assessment.status == "undetermined"
