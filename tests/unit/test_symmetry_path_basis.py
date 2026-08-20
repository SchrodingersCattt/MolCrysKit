"""Basis-independent symmetry operation and path invariants."""

import numpy as np
from ase import Atoms

from molcrys_kit.operations import transform_crystal_fractional
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal
from molcrys_kit.structures.symmetry import (
    FractionalAffineOperation,
    LatticeBasisChange,
)


def test_basis_conjugation_preserves_cartesian_transformed_structure():
    lattice = np.array([[5.0, 0.0, 0.0], [0.7, 6.0, 0.0], [0.2, 0.4, 7.0]])
    fractional = np.array([[0.1, 0.2, 0.3], [0.15, 0.2, 0.3]])
    molecule = CrystalMolecule(
        Atoms("HH", positions=fractional @ lattice, pbc=False),
        check_pbc=False,
    )
    crystal = MolecularCrystal(lattice, [molecule])
    inversion = FractionalAffineOperation(-np.eye(3), [0.25, 0.5, 0.75])
    change = LatticeBasisChange([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    transformed_old = transform_crystal_fractional(crystal, inversion)

    new_lattice = change.transform_lattice(lattice)
    new_fractional = change.old_to_new_fractional(fractional)
    new_molecule = CrystalMolecule(
        Atoms("HH", positions=new_fractional @ new_lattice, pbc=False),
        check_pbc=False,
    )
    new_crystal = MolecularCrystal(new_lattice, [new_molecule])
    transformed_new = transform_crystal_fractional(
        new_crystal, change.transform_operation(inversion)
    )

    # Physical coordinates may differ by one whole-cell image; MIC distances
    # and the transformed molecular shape must be identical.
    np.testing.assert_allclose(
        transformed_old.molecules[0].get_all_distances(),
        transformed_new.molecules[0].get_all_distances(),
    )
    np.testing.assert_allclose(
        transformed_old.molecules[0].get_center_of_mass(),
        transformed_new.molecules[0].get_center_of_mass(),
        atol=1.0e-10,
    )
