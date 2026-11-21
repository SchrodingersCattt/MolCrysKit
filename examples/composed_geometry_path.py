"""Compose a rotating subset with coupled, caller-defined path coordinates.

This synthetic example is geometric only and does not represent a particular
material, reaction mechanism, electronic state, or physical energy surface.
"""

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation

from molcrys_kit.operations import (
    DistanceCoordinate,
    LinearCombinationCoordinate,
    PathLayer,
    PathRestraint,
    SignedPlaneDistanceCoordinate,
    compose_path_images,
    refine_idpp_path,
)


def build_example():
    """Return a path result and its explicitly supplied restraint schedule."""
    fractions = np.linspace(0.0, 1.0, 5)
    base = []
    layer = []
    for fraction in fractions:
        frame = Atoms(
            "CHHHO",
            positions=[
                [0, 0, 0],
                [1.1, 0, 0],
                [0, 1.1, 0],
                [0.1, 0.1, 1.0],
                [3.0 - fraction, 0.5 * fraction, 0],
            ],
            cell=np.eye(3) * 12,
            pbc=False,
        )
        fragment = frame[[0, 1, 2]]
        angle = 0.4 * np.sin(np.pi * fraction)
        fragment.positions[:] = Rotation.from_rotvec([0, 0, angle]).apply(
            fragment.positions
        )
        base.append(frame)
        layer.append(fragment)

    composed = compose_path_images(
        base, [PathLayer((0, 1, 2), layer, anchor_index=0, name="moving subset")]
    )
    shape = SignedPlaneDistanceCoordinate(0, 1, 2, 3)
    separation = DistanceCoordinate(0, 4)
    coordinate = LinearCombinationCoordinate(((1.0, shape), (0.2, separation)))
    initial = [
        coordinate.value_and_gradient(frame.positions, frame.cell, frame.pbc)[0]
        for frame in composed.images
    ]
    targets = np.asarray(initial) + 0.05 * np.sin(np.pi * fractions)
    restraint = PathRestraint(
        coordinate, targets=targets, weights=[0, 3, 3, 3, 0], name="coupled coordinate"
    )
    result = refine_idpp_path(
        composed.images,
        prescribed_atom_indices=(0, 1, 2),
        restraints=[restraint],
        max_iterations=1000,
        gradient_tolerance=1e-5,
    )
    return result, composed, restraint


if __name__ == "__main__":
    result, _, _ = build_example()
    print("images:", len(result.images), "all converged:", result.converged)
    print("restraint diagnostics:", result.restraint_diagnostics)
