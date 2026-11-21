"""Behavior and derivatives for reusable geometric path restraints."""

import numpy as np
import pytest
from ase import Atoms
from scipy.spatial.transform import Rotation

from molcrys_kit.operations import (
    AngleCoordinate,
    DistanceCoordinate,
    LinearCombinationCoordinate,
    PathRestraint,
    PeriodicPathConfig,
    SignedPlaneDistanceCoordinate,
    SignedVolumeCoordinate,
    interpolate_periodic_paths,
    refine_idpp_path,
)
from molcrys_kit.structures.crystal import MolecularCrystal


CELL = np.array([[8.0, 0.2, 0.0], [1.3, 7.0, 0.4], [0.2, 0.1, 9.0]])
PBC = np.array([True, True, False])
POSITIONS = np.array(
    [[1.0, 1.5, 2.0], [2.1, 1.8, 2.3], [0.7, 2.4, 2.4], [1.2, 1.1, 3.1]]
)


@pytest.mark.parametrize(
    "coordinate",
    [
        DistanceCoordinate(0, 1),
        AngleCoordinate(1, 0, 2),
        SignedVolumeCoordinate(0, 1, 2, 3),
        SignedPlaneDistanceCoordinate(0, 1, 2, 3),
        LinearCombinationCoordinate(
            (
                (1.7, DistanceCoordinate(0, 1)),
                (-0.8, SignedVolumeCoordinate(0, 1, 2, 3)),
            )
        ),
    ],
)
def test_coordinate_gradients_and_global_rigid_transform(coordinate):
    value, gradient = coordinate.value_and_gradient(POSITIONS, CELL, PBC)
    numerical = np.zeros_like(POSITIONS)
    step = 1e-6
    for i in range(len(POSITIONS)):
        for j in range(3):
            plus, minus = POSITIONS.copy(), POSITIONS.copy()
            plus[i, j] += step
            minus[i, j] -= step
            numerical[i, j] = (
                coordinate.value_and_gradient(plus, CELL, PBC)[0]
                - coordinate.value_and_gradient(minus, CELL, PBC)[0]
            ) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-12)
    rotation = Rotation.from_rotvec([0.4, -0.3, 0.7]).as_matrix()
    moved = POSITIONS @ rotation.T + [2.3, -1.0, 0.7]
    rotated_cell = CELL @ rotation.T
    actual, rotated_gradient = coordinate.value_and_gradient(moved, rotated_cell, PBC)
    assert actual == pytest.approx(value, abs=1e-11)
    np.testing.assert_allclose(rotated_gradient, gradient @ rotation.T, atol=1e-10)


def test_wrapping_and_explicit_images_are_equivalent():
    shifts = np.array([[1, -1, 0], [-2, 1, 0], [0, -1, 0], [1, 0, 0]])
    wrapped = POSITIONS + shifts @ CELL
    coordinates = [
        DistanceCoordinate(0, 1),
        AngleCoordinate(1, 0, 2),
        SignedVolumeCoordinate(0, 1, 2, 3),
        SignedPlaneDistanceCoordinate(0, 1, 2, 3),
    ]
    for coordinate in coordinates:
        original, gradient = coordinate.value_and_gradient(POSITIONS, CELL, PBC)
        actual, shifted_gradient = coordinate.value_and_gradient(wrapped, CELL, PBC)
        assert actual == pytest.approx(original, abs=1e-11)
        np.testing.assert_allclose(shifted_gradient, gradient, atol=1e-11)
    base = DistanceCoordinate(0, 1, (1, 0, 0))
    corrected = DistanceCoordinate(
        0, 1, tuple((np.array([1, 0, 0]) + shifts[0] - shifts[1]).tolist())
    )
    assert base.value_and_gradient(POSITIONS, CELL, PBC)[0] == pytest.approx(
        corrected.value_and_gradient(wrapped, CELL, PBC)[0], abs=1e-11
    )


def test_invalid_or_singular_coordinates_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        DistanceCoordinate(0, 0)
    with pytest.raises(ValueError, match="integer triples"):
        DistanceCoordinate(0, 1, (0.5, 0, 0))
    with pytest.raises(ValueError, match="non-periodic"):
        DistanceCoordinate(0, 1, (0, 0, 1)).value_and_gradient(POSITIONS, CELL, PBC)
    with pytest.raises(ValueError, match="outside"):
        DistanceCoordinate(0, 99).value_and_gradient(POSITIONS, CELL, PBC)
    with pytest.raises(ValueError, match="collinear"):
        AngleCoordinate(0, 1, 2).value_and_gradient(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.zeros((3, 3)),
            np.zeros(3, bool),
        )
    with pytest.raises(ValueError, match="overlapping"):
        DistanceCoordinate(0, 1).value_and_gradient(np.zeros((2, 3)), CELL, PBC)


def test_idpp_restraint_changes_only_free_atoms_and_keeps_endpoints():
    original = Atoms(
        "NCHH",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0], [0.1, 0.1, 1.0]],
        pbc=False,
    )
    frames = [original.copy() for _ in range(5)]
    before = [a.positions.copy() for a in frames]
    coordinate = SignedVolumeCoordinate(0, 1, 2, 3)
    initial_volume = coordinate.value_and_gradient(
        original.positions, original.cell, original.pbc
    )[0]
    target = PathRestraint(
        coordinate, targets=0.0, weights=[0, 100, 100, 100, 0], name="shape"
    )
    result = refine_idpp_path(
        frames,
        prescribed_atom_indices=(0, 1, 2),
        restraints=[target],
        max_iterations=1000,
        gradient_tolerance=1e-5,
    )
    assert result.converged
    for k, image in enumerate(result.images):
        np.testing.assert_allclose(image.positions[:3], before[k][:3], atol=0, rtol=0)
        np.testing.assert_array_equal(frames[k].positions, before[k])
    np.testing.assert_array_equal(result.images[0].positions, before[0])
    np.testing.assert_array_equal(result.images[-1].positions, before[-1])
    volume = coordinate.value_and_gradient(
        result.images[2].positions, original.cell, original.pbc
    )[0]
    assert abs(volume) < abs(initial_volume) * 0.1
    diagnostic = result.restraint_diagnostics[0]
    assert diagnostic["name"] == "shape"
    assert diagnostic["values"][0] is None and diagnostic["values"][-1] is None
    assert diagnostic["values"][2] == pytest.approx(volume)
    assert diagnostic["residuals"][2] == pytest.approx(volume)
    assert diagnostic["weighted_penalties"][2] == pytest.approx(100 * volume**2)
    assert diagnostic["applied_to_objective"] == [False, True, True, True, False]


def test_zero_weight_and_invalid_schedules():
    atoms = Atoms("HH", positions=[[0, 0, 0], [1.2, 0, 0]], pbc=False)
    frames = [atoms.copy() for _ in range(3)]
    coordinate = DistanceCoordinate(0, 1)
    plain = refine_idpp_path(frames)
    disabled = refine_idpp_path(
        frames, restraints=[PathRestraint(coordinate, 2.0, 0.0)]
    )
    for a, b in zip(plain.images, disabled.images):
        np.testing.assert_array_equal(a.positions, b.positions)
    with pytest.raises(ValueError, match="one value per image"):
        refine_idpp_path(frames, restraints=[PathRestraint(coordinate, [1, 2], 1)])
    with pytest.raises(ValueError, match="non-negative"):
        refine_idpp_path(frames, restraints=[PathRestraint(coordinate, 1, -1)])
    with pytest.raises(TypeError, match="PathRestraint"):
        refine_idpp_path(frames, restraints=["not a restraint"])


def test_periodic_path_uses_the_same_generic_restraint():
    atoms = Atoms(
        "HH", positions=[[0, 0, 0], [1.2, 0, 0]], cell=np.eye(3) * 10, pbc=True
    )
    crystal = MolecularCrystal.from_ase_atoms(atoms)
    config = PeriodicPathConfig(n_images=3, max_candidates=1)
    restraint = PathRestraint(
        DistanceCoordinate(0, 1), targets=2.0, weights=[0, 10, 0], name="separation"
    )
    result = interpolate_periodic_paths(
        crystal,
        crystal,
        product_index_by_reactant=[0, 1],
        config=config,
        restraints=[restraint],
    )
    path = result.candidates[0]
    assert path.images[1].get_distance(0, 1, mic=True) > 1.8
    np.testing.assert_allclose(path.images[0].positions, crystal.to_ase().positions)
    np.testing.assert_allclose(path.images[-1].positions, crystal.to_ase().positions)
    assert result.diagnostics["restraint_count"] == 1
    assert path.diagnostics["restraints"][0]["name"] == "separation"


def test_coupled_coordinate_can_move_independent_atoms_together():
    atoms = Atoms(
        "NCHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.1, 0.1, 1.0],
            [2.7, 2.3, 0.5],
        ],
        pbc=False,
    )
    images = [atoms.copy() for _ in range(3)]
    shape = SignedVolumeCoordinate(0, 1, 2, 3)
    separation = DistanceCoordinate(0, 4)
    coordinate = LinearCombinationCoordinate(((1.0, shape), (1.0, separation)))
    before_shape = shape.value_and_gradient(atoms.positions, atoms.cell, atoms.pbc)[0]
    before_distance = separation.value_and_gradient(
        atoms.positions, atoms.cell, atoms.pbc
    )[0]
    target = before_shape + before_distance - 0.8
    result = refine_idpp_path(
        images,
        prescribed_atom_indices=[0, 1, 2],
        restraints=[PathRestraint(coordinate, target, [0.0, 50.0, 0.0])],
        max_iterations=1000,
        gradient_tolerance=1e-5,
    )
    assert result.converged
    middle = result.images[1]
    after_shape = shape.value_and_gradient(middle.positions, middle.cell, middle.pbc)[0]
    after_distance = separation.value_and_gradient(
        middle.positions, middle.cell, middle.pbc
    )[0]
    assert abs(after_shape - before_shape) > 0.01
    assert abs(after_distance - before_distance) > 0.01
    assert abs(after_shape + after_distance - target) < abs(
        before_shape + before_distance - target
    )


@pytest.mark.parametrize(
    "coordinate_type", [SignedVolumeCoordinate, SignedPlaneDistanceCoordinate]
)
def test_oriented_coordinate_with_explicit_images_tracks_atom_rewrapping(
    coordinate_type,
):
    wraps = np.array([[1, -1, 0], [-2, 1, 0], [0, -1, 0], [1, 0, 0]])
    explicit = np.array([[1, 0, 0], [0, 0, 0], [0, -1, 0]])
    adjusted = explicit + wraps[0] - wraps[1:]
    original = coordinate_type(0, 1, 2, 3, tuple(map(tuple, explicit)))
    rewrapped = coordinate_type(0, 1, 2, 3, tuple(map(tuple, adjusted)))
    a, ga = original.value_and_gradient(POSITIONS, CELL, PBC)
    b, gb = rewrapped.value_and_gradient(POSITIONS + wraps @ CELL, CELL, PBC)
    assert a == pytest.approx(b, abs=1e-10)
    np.testing.assert_allclose(ga, gb, atol=1e-10)


def test_user_supplied_coordinate_bad_gradient_is_rejected():
    class BrokenCoordinate:
        def value_and_gradient(self, positions, cell, pbc):
            return 1.0, np.zeros(3)

    atoms = Atoms("HH", positions=[[0, 0, 0], [1.2, 0, 0]], pbc=False)
    with pytest.raises(ValueError, match="invalid gradient"):
        refine_idpp_path(
            [atoms.copy() for _ in range(3)],
            restraints=[PathRestraint(BrokenCoordinate(), 0.0, 1.0)],
        )


def test_soft_restraint_residual_remains_visible_when_all_atoms_are_fixed():
    atoms = Atoms(
        "HH", positions=[[0, 0, 0], [1.2, 0, 0]], cell=np.eye(3) * 10, pbc=True
    )
    crystal = MolecularCrystal.from_ase_atoms(atoms)
    config = PeriodicPathConfig(n_images=3, max_candidates=1, fixed_atom_indices=(0, 1))
    restraint = PathRestraint(
        DistanceCoordinate(0, 1), targets=2.0, weights=1.0, name="unreachable"
    )
    result = interpolate_periodic_paths(
        crystal,
        crystal,
        product_index_by_reactant=[0, 1],
        config=config,
        restraints=[restraint],
    )
    report = result.candidates[0].diagnostics["restraints"][0]
    assert result.candidates[0].diagnostics["accepted"]
    assert report["values"] == pytest.approx([1.2, 1.2, 1.2])
    assert report["residuals"] == pytest.approx([-0.8, -0.8, -0.8])
    assert report["applied_to_objective"] == [False, True, False]


def test_signed_plane_distance_tracks_orientation_and_rejects_degenerate_plane():
    coordinate = SignedPlaneDistanceCoordinate(0, 1, 2, 3)
    value, gradient = coordinate.value_and_gradient(POSITIONS, CELL, PBC)
    reversed_value, _ = SignedPlaneDistanceCoordinate(0, 2, 1, 3).value_and_gradient(
        POSITIONS, CELL, PBC
    )
    assert reversed_value == pytest.approx(-value)
    assert np.linalg.norm(gradient) > 0
    planar = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    assert coordinate.value_and_gradient(planar, CELL, (False, False, False))[
        0
    ] == pytest.approx(0.0)
    planar[3] = [2.0, 0.0, 0.0]
    planar[2] = [1.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="collinear plane atoms"):
        coordinate.value_and_gradient(planar, CELL, (False, False, False))


def test_plane_distance_can_be_used_in_a_path_restraint():
    atoms = Atoms(
        "NCHH",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0], [0.1, 0.1, 1.0]],
        pbc=False,
    )
    coordinate = SignedPlaneDistanceCoordinate(0, 1, 2, 3)
    original = coordinate.value_and_gradient(atoms.positions, atoms.cell, atoms.pbc)[0]
    frames = [atoms.copy() for _ in range(3)]
    result = refine_idpp_path(
        frames,
        prescribed_atom_indices=(0, 1, 2),
        restraints=[PathRestraint(coordinate, targets=0.0, weights=[0.0, 100.0, 0.0])],
        max_iterations=1000,
        gradient_tolerance=1e-5,
    )
    assert result.converged
    height = coordinate.value_and_gradient(
        result.images[1].positions, atoms.cell, atoms.pbc
    )[0]
    assert abs(height) < abs(original) * 0.1


def test_idpp_rejects_multidimensional_atom_ids():
    atoms = Atoms("HH", positions=[[0, 0, 0], [1.2, 0, 0]], pbc=False)
    atoms.set_array("atom_id", np.array([[1, 1], [2, 2]]))
    with pytest.raises(ValueError, match="one scalar per atom"):
        refine_idpp_path([atoms.copy() for _ in range(3)])
