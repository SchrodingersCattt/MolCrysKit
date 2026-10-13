from __future__ import annotations

import numpy as np

from molcrys_kit.chemistry import (
    BondOrderOptions,
    infer_bond_orders_array,
    infer_bond_orders_batch,
)


def test_array_adapter_uses_entity_solver_and_does_not_mutate_inputs() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    original = positions.copy()
    result = infer_bond_orders_array(
        positions,
        [8, 1, 1],
        bonds=((0, 1), (0, 2)),
        options=BondOrderOptions(),
    )
    assert result.entity.status.value in {"inferred", "provisional"}
    assert all(bond.order == 1.0 for bond in result.entity.bonds)
    np.testing.assert_array_equal(positions, original)
    payload = result.to_dict()
    assert payload["entity"]["bonds"][0]["order"] == 1.0
    assert "diagnostics" in payload


def test_batch_adapter_preserves_frame_order() -> None:
    frames = [
        (np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]), [1, 1], ((0, 1),)),
        (
            np.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            [8, 1, 1],
            ((0, 1), (0, 2)),
        ),
    ]
    results = infer_bond_orders_batch(frames)
    assert len(results) == 2
    assert [len(result.entity.atoms) for result in results] == [2, 3]
    assert all(result.entity.status.value == "inferred" for result in results)
