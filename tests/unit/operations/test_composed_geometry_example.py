"""The project-neutral path composition example is executable."""

import runpy
from pathlib import Path

import numpy as np


def test_composed_geometry_example():
    example = (
        Path(__file__).resolve().parents[3] / "examples" / "composed_geometry_path.py"
    )
    namespace = runpy.run_path(str(example))
    result, composed, restraint = namespace["build_example"]()
    assert result.converged
    assert len(result.images) == 5
    assert restraint.name == "coupled coordinate"
    for original, refined in zip(composed.images, result.images):
        np.testing.assert_array_equal(original.positions[:3], refined.positions[:3])
    assert result.restraint_diagnostics[0]["applied_to_objective"] == [
        False,
        True,
        True,
        True,
        False,
    ]
