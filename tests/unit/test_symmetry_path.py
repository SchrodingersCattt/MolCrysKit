"""Strict rigid crystallographic path planning and generation."""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.constants.symmetry_path import (
    RIGID_MASS_WEIGHTED_RMSD_TOLERANCE_ANGSTROM,
)
from molcrys_kit.operations import (
    CollectiveConstraint,
    RigidReachabilityError,
    SymmetryPathConfig,
    build_symmetry_path_plan,
    generate_collective_symmetry_path,
    interpolate_symmetry_path,
    transform_crystal_fractional,
)
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal
from molcrys_kit.structures.symmetry import FractionalAffineOperation
from molcrys_kit.utils.geometry import get_rotation_matrix


def molecule(symbols="OHH", positions=None):
    if positions is None:
        positions = [[0, 0, 0], [0.95, 0, 0], [-0.24, 0.92, 0]]
    return CrystalMolecule(
        Atoms(symbols, positions=positions, pbc=False),
        check_pbc=False,
    )


def crystal(molecules, lattice=None):
    lattice = np.diag([10.0, 10.0, 10.0]) if lattice is None else lattice
    return MolecularCrystal(lattice, molecules, metadata={"tag": "start"})


def rigid_target(source, rotation, translation):
    source_com = source.get_center_of_mass()
    positions = (
        (source.get_positions() - source_com) @ rotation.T + source_com + translation
    )
    return molecule(source.get_chemical_symbols(), positions)


def pair_distances(mol):
    positions = mol.get_positions()
    return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)


def test_fractional_transform_preserves_whole_molecule_geometry():
    source = crystal([molecule()])
    operation = FractionalAffineOperation(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0.5, 0]
    )
    target = transform_crystal_fractional(source, operation)
    np.testing.assert_allclose(
        pair_distances(target.molecules[0]), pair_distances(source.molecules[0])
    )
    assert target.metadata["fractional_affine_operation"]["translation"] == [
        0.0,
        0.5,
        0.0,
    ]


def test_fractional_transform_selects_one_nearest_image_for_whole_molecule():
    source_molecule = molecule(
        positions=[[9.8, 1.0, 1.0], [10.75, 1.0, 1.0], [9.56, 1.92, 1.0]]
    )
    source = crystal([source_molecule])
    translated_identity = FractionalAffineOperation(np.eye(3), [1, 0, 0])
    target = transform_crystal_fractional(source, translated_identity)
    np.testing.assert_allclose(
        target.molecules[0].get_positions(), source_molecule.get_positions()
    )
    np.testing.assert_allclose(
        pair_distances(target.molecules[0]), pair_distances(source_molecule)
    )


def test_strict_rigid_plan_preserves_all_internal_distances_and_endpoints():
    source_molecule = molecule()
    rotation = get_rotation_matrix([0, 0, 1], np.pi / 2)
    target_molecule = rigid_target(source_molecule, rotation, [2.0, 1.0, 0.5])
    start = crystal([source_molecule])
    target = crystal([target_molecule])
    operation = FractionalAffineOperation(np.eye(3), [0.2, 0.1, 0.05])
    plan = build_symmetry_path_plan(
        start,
        operation,
        target=target,
        config=SymmetryPathConfig(n_images=5),
    )
    frames = interpolate_symmetry_path(plan)
    np.testing.assert_allclose(
        frames[0].molecules[0].get_positions(), source_molecule.get_positions()
    )
    np.testing.assert_allclose(
        frames[-1].molecules[0].get_positions(), target_molecule.get_positions()
    )
    reference = pair_distances(source_molecule)
    for frame in frames:
        np.testing.assert_allclose(
            pair_distances(frame.molecules[0]), reference, atol=1e-10
        )
        assert (
            abs(
                np.linalg.det(
                    plan.provenance.correspondence.molecule_matches[0].proper_rotation
                )
                - 1.0
            )
            < 1e-10
        )


def test_global_assignment_handles_reordered_identical_molecules():
    first = molecule(positions=np.asarray(molecule().get_positions()) + [1, 1, 1])
    second = molecule(positions=np.asarray(molecule().get_positions()) + [7, 1, 1])
    start = crystal([first, second])
    target = crystal([second, first])
    plan = build_symmetry_path_plan(
        start,
        FractionalAffineOperation(np.eye(3), [0, 0, 0]),
        target=target,
        config=SymmetryPathConfig(n_images=3),
    )
    mapping = {
        match.source_molecule_index: match.target_molecule_index
        for match in plan.provenance.correspondence.molecule_matches
    }
    assert mapping == {0: 1, 1: 0}
    frames = interpolate_symmetry_path(plan)
    np.testing.assert_allclose(
        frames[-1].molecules[0].get_positions(),
        target.molecules[1].get_positions(),
    )
    np.testing.assert_allclose(
        frames[-1].molecules[1].get_positions(),
        target.molecules[0].get_positions(),
    )


def test_deformed_endpoint_is_rejected_before_images_are_generated():
    source_molecule = molecule()
    deformed = source_molecule.get_positions().copy()
    deformed[1, 0] += 0.20
    start = crystal([source_molecule])
    target = crystal([molecule(positions=deformed)])
    with pytest.raises(RigidReachabilityError, match="not rigid-reachable"):
        build_symmetry_path_plan(
            start,
            FractionalAffineOperation(np.eye(3), [0, 0, 0]),
            target=target,
        )


def test_default_rmsd_threshold_is_centralized_and_configurable():
    assert (
        SymmetryPathConfig().tolerance.mass_weighted_rmsd_angstrom
        == RIGID_MASS_WEIGHTED_RMSD_TOLERANCE_ANGSTROM
    )


def test_unresolved_partial_occupancy_is_rejected_by_default():
    source_molecule = molecule()
    source_molecule.set_array("occupancy", np.array([1.0, 0.5, 0.5]))
    start = crystal([source_molecule])
    with pytest.raises(ValueError, match="resolve disorder"):
        build_symmetry_path_plan(start, FractionalAffineOperation(np.eye(3), [0, 0, 0]))


def test_improper_operation_allowed_when_atom_permutation_has_proper_realization():
    # Linear symmetric molecule: inversion exchanges equivalent terminal atoms.
    source = molecule("HOH", [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    start = crystal([source])
    inversion = FractionalAffineOperation(-np.eye(3), [0.5, 0.5, 0.5])
    frames = generate_collective_symmetry_path(
        start,
        inversion,
        config=SymmetryPathConfig(n_images=3, max_isomorphisms=128),
    )
    assert len(frames) == 3
    np.testing.assert_allclose(
        pair_distances(frames[1].molecules[0]), pair_distances(source)
    )


def test_equivariant_mode_requires_explicit_orbit_api():
    start = crystal([molecule()])
    with pytest.raises(NotImplementedError, match="orbit"):
        build_symmetry_path_plan(
            start,
            FractionalAffineOperation(np.eye(3), [0, 0, 0]),
            config=SymmetryPathConfig(
                collective=CollectiveConstraint.SYMMETRY_EQUIVARIANT
            ),
        )


def test_provenance_is_json_serializable():
    source = crystal([molecule()])
    operation = FractionalAffineOperation(np.eye(3), [0.1, 0.0, 0.0])
    plan = build_symmetry_path_plan(source, operation)
    import json

    json.dumps(plan.provenance.to_dict())
