"""Valence-constrained chemistry perception tests."""

import numpy as np
import pytest
from ase import Atoms

from molcrys_kit.chemistry import (
    BondKind,
    ChemistryIndeterminateError,
    InferenceStatus,
    infer_chemistry,
)
from molcrys_kit.structures import CrystalMolecule, MolecularCrystal


def _crystal(symbols, positions) -> MolecularCrystal:
    molecule = CrystalMolecule(
        Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float)),
        check_pbc=False,
    )
    return MolecularCrystal(np.eye(3) * 20.0, [molecule], pbc=(False, False, False))


@pytest.mark.parametrize(
    "symbols,positions,expected_order",
    [
        ("C2H6", [[-0.77, 0, 0], [0.77, 0, 0], [-1.1, 1, 0], [-1.1, -0.5, 0.87], [-1.1, -0.5, -0.87], [1.1, -1, 0], [1.1, 0.5, 0.87], [1.1, 0.5, -0.87]], 1.0),
        ("C2H4", [[-0.67, 0, 0], [0.67, 0, 0], [-1.2, 0.93, 0], [-1.2, -0.93, 0], [1.2, 0.93, 0], [1.2, -0.93, 0]], 2.0),
        ("C2H2", [[-0.60, 0, 0], [0.60, 0, 0], [-1.66, 0, 0], [1.66, 0, 0]], 3.0),
    ],
)
def test_carbon_bond_orders_are_solved_from_valence(symbols, positions, expected_order):
    crystal = _crystal(symbols, positions)
    result = infer_chemistry(crystal)
    entity = result.components[0]
    carbon_bond = next(
        bond
        for bond in entity.bonds
        if {entity.atoms[0].atom_id, entity.atoms[1].atom_id}
        == {bond.atom1_id, bond.atom2_id}
    )
    assert carbon_bond.order == expected_order
    assert entity.net_charge == 0
    assert result.status is InferenceStatus.INFERRED


def test_metal_nonmetal_edge_is_coordination_not_covalent_order() -> None:
    crystal = _crystal("ZnN", [[0, 0, 0], [2.0, 0, 0]])
    result = infer_chemistry(crystal)
    bond = result.components[0].bonds[0]
    assert bond.kind is BondKind.COORDINATION
    assert bond.order is None


def test_ambiguous_valence_is_visible_and_strict_mode_is_atomic() -> None:
    crystal = _crystal("CO", [[0, 0, 0], [1.30, 0, 0]])
    before = crystal.chemistry
    with pytest.raises(ChemistryIndeterminateError):
        infer_chemistry(crystal, strict=True)
    assert crystal.chemistry is before

    result = infer_chemistry(crystal)
    assert result.status in {InferenceStatus.PROVISIONAL, InferenceStatus.INDETERMINATE}
    assert result.warnings
