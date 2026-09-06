"""Chemistry-domain model and stable crystal mapping tests."""

from dataclasses import FrozenInstanceError

import pytest

from molcrys_kit.chemistry import (
    ChemicalAtom,
    ChemistryIndeterminateError,
    InferenceStatus,
    annotate_chemistry,
)


def test_chemical_atom_is_immutable() -> None:
    atom = ChemicalAtom(atom_id="m0:a0", element="C")
    with pytest.raises(FrozenInstanceError):
        atom.element = "N"  # type: ignore[misc]


def test_annotation_uses_stable_site_ids_and_preserves_copy(crystal_single_water) -> None:
    result = annotate_chemistry(crystal_single_water)

    assert result.status is InferenceStatus.PROVISIONAL
    assert result.atom_ids_by_global_index == tuple(
        record.site_id for record in crystal_single_water.get_site_records()
    )
    assert len(result.components) == 1
    entity = result.components[0]
    assert [atom.element for atom in entity.atoms] == ["O", "H", "H"]
    assert len(entity.bonds) == 2
    assert all(bond.order is None for bond in entity.bonds)
    assert crystal_single_water.molecules[0].chemical_entity is entity

    copied = crystal_single_water.copy()
    assert copied.chemistry is result
    assert copied.molecules[0].chemical_entity is entity
    assert [record.site_id for record in copied.get_site_records()] == [
        record.site_id for record in crystal_single_water.get_site_records()
    ]


def test_strict_annotation_rejects_unresolved_connectivity(crystal_single_water) -> None:
    with pytest.raises(ChemistryIndeterminateError, match="Bond orders"):
        annotate_chemistry(crystal_single_water, strict=True)
    assert crystal_single_water.chemistry is None


def test_repeated_input_atom_ids_are_disambiguated(cubic_lattice_10) -> None:
    import numpy as np
    from ase import Atoms

    from molcrys_kit.constants.config import KEY_ATOM_ID
    from molcrys_kit.structures import CrystalMolecule, MolecularCrystal

    first = CrystalMolecule(Atoms("H", positions=[[0.0, 0.0, 0.0]]), check_pbc=False)
    second = CrystalMolecule(Atoms("H", positions=[[2.0, 0.0, 0.0]]), check_pbc=False)
    first.set_array(KEY_ATOM_ID, np.asarray(["shared"]))
    second.set_array(KEY_ATOM_ID, np.asarray(["shared"]))

    crystal = MolecularCrystal(cubic_lattice_10, [first, second])
    ids = [record.site_id for record in crystal.get_site_records()]
    assert ids[0] == "shared"
    assert ids[1].startswith("shared~m1:a0")
    assert len(ids) == len(set(ids))
