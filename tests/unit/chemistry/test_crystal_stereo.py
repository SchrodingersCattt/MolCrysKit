from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from molcrys_kit.chemistry import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    CrystalChemistry,
    CrystalStereoClass,
    CrystalStereoIndeterminateError,
    Embedding,
    EntityRelationship,
    FiniteChemicalEntity,
    InferenceStatus,
    StereoDescriptor,
    StereoKind,
    StereoReport,
    analyze_crystal_stereochemistry,
    assign_stereochemistry,
    classify_entity_relationship,
)
from molcrys_kit.structures.symmetry import (
    CrystalSymmetry,
    FractionalAffineOperation,
    identity_operation,
)


def _tetrahedral_entity(
    entity_id: str, *, mirror: bool = False
) -> FiniteChemicalEntity:
    prefix = entity_id.replace(":", "_")
    ids = [f"{prefix}:{label}" for label in ("C", "Br", "Cl", "F", "H")]
    atoms = tuple(
        ChemicalAtom(
            atom_id=atom_id, element=element, formal_charge=0, implicit_hydrogens=0
        )
        for atom_id, element in zip(ids, ("C", "Br", "Cl", "F", "H"))
    )
    bonds = tuple(
        ChemicalBond(ids[0], atom_id, order=1.0, kind=BondKind.COVALENT)
        for atom_id in ids[1:]
    )
    directions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    ) / np.sqrt(3.0)
    coordinates = np.vstack([np.zeros(3), directions])
    if mirror:
        coordinates[:, 0] *= -1.0
    embedding = Embedding(
        tuple((atom_id, tuple(position)) for atom_id, position in zip(ids, coordinates))
    )
    return FiniteChemicalEntity(
        entity_id=entity_id,
        atoms=atoms,
        bonds=bonds,
        embedding=embedding,
        net_charge=0,
        status=InferenceStatus.INFERRED,
    )


def _chemistry(*entities) -> CrystalChemistry:
    atom_ids = tuple(atom.atom_id for entity in entities for atom in entity.atoms)
    return CrystalChemistry(
        components=tuple(entities),
        atom_ids_by_global_index=atom_ids,
        status=InferenceStatus.INFERRED,
    )


def _manual_report(entity, descriptors) -> StereoReport:
    return StereoReport(
        entity_id=entity.entity_id,
        descriptors=tuple(
            StereoDescriptor(
                kind=StereoKind.TETRAHEDRAL,
                center_atom_id=atom_id,
                descriptor=value,
                cip_order=(),
                status=InferenceStatus.INFERRED,
                reason="golden descriptor",
                rules_applied=("golden",),
            )
            for atom_id, value in descriptors
        ),
        status=InferenceStatus.INFERRED,
        evidence=(),
    )


def _symmetric_two_center_entity(entity_id: str) -> FiniteChemicalEntity:
    ids = {
        name: f"{entity_id}:{name}"
        for name in ("left", "right", "cl1", "cl2", "me1", "me2")
    }
    atoms = (
        ChemicalAtom(ids["left"], "C", formal_charge=0, implicit_hydrogens=1),
        ChemicalAtom(ids["right"], "C", formal_charge=0, implicit_hydrogens=1),
        ChemicalAtom(ids["cl1"], "Cl", formal_charge=0, implicit_hydrogens=0),
        ChemicalAtom(ids["cl2"], "Cl", formal_charge=0, implicit_hydrogens=0),
        ChemicalAtom(ids["me1"], "C", formal_charge=0, implicit_hydrogens=3),
        ChemicalAtom(ids["me2"], "C", formal_charge=0, implicit_hydrogens=3),
    )
    pairs = (
        ("left", "right"),
        ("left", "cl1"),
        ("right", "cl2"),
        ("left", "me1"),
        ("right", "me2"),
    )
    bonds = tuple(
        ChemicalBond(ids[left], ids[right], order=1.0, kind=BondKind.COVALENT)
        for left, right in pairs
    )
    return FiniteChemicalEntity(
        entity_id=entity_id,
        atoms=atoms,
        bonds=bonds,
        status=InferenceStatus.INFERRED,
    )


def test_entity_relationship_distinguishes_same_mirror_and_constitution() -> None:
    right = _tetrahedral_entity("right")
    same = _tetrahedral_entity("same")
    mirror = _tetrahedral_entity("mirror", mirror=True)
    different = FiniteChemicalEntity(
        entity_id="different",
        atoms=(ChemicalAtom("different:C", "C"),),
        bonds=(),
    )

    assert (
        classify_entity_relationship(right, same)
        is EntityRelationship.SAME_STEREOISOMER
    )
    assert classify_entity_relationship(right, mirror) is EntityRelationship.MIRROR
    assert (
        classify_entity_relationship(right, different)
        is EntityRelationship.DIFFERENT_CONSTITUTION
    )


def test_crystal_classifies_equal_mirror_counts_as_racemic_not_conglomerate() -> None:
    right = _tetrahedral_entity("right")
    mirror = _tetrahedral_entity("mirror", mirror=True)

    report = analyze_crystal_stereochemistry(_chemistry(right, mirror))

    assert report.classification is CrystalStereoClass.RACEMIC_CRYSTAL
    assert report.enantiomer_counts[0].count == 1
    assert report.enantiomer_counts[0].mirror_count == 1
    assert "conglomerate" not in report.reason.lower()


def test_crystal_classifies_repeated_one_handed_entities_as_enantiopure() -> None:
    first = _tetrahedral_entity("first")
    second = _tetrahedral_entity("second")

    report = analyze_crystal_stereochemistry(_chemistry(first, second))

    assert report.classification is CrystalStereoClass.ENANTIOPURE
    assert report.enantiomer_counts[0].count == 2
    assert report.enantiomer_counts[0].mirror_entity_id is None


def test_internal_descriptor_inversion_identifies_meso_graph() -> None:
    entity = _symmetric_two_center_entity("meso")
    ids = {atom.atom_id.rsplit(":", 1)[-1]: atom.atom_id for atom in entity.atoms}
    stereo = _manual_report(entity, ((ids["left"], "R"), (ids["right"], "S")))

    report = analyze_crystal_stereochemistry(
        _chemistry(entity),
        stereo_reports={entity.entity_id: stereo},
    )

    assert report.classification is CrystalStereoClass.MESO_ACHIRAL
    assert report.entities[0].is_internal_mirror is True


def test_non_mirror_stereoisomer_is_not_counted_as_racemate() -> None:
    meso = _symmetric_two_center_entity("meso")
    rr = _symmetric_two_center_entity("rr")
    meso_ids = {atom.atom_id.rsplit(":", 1)[-1]: atom.atom_id for atom in meso.atoms}
    rr_ids = {atom.atom_id.rsplit(":", 1)[-1]: atom.atom_id for atom in rr.atoms}
    meso_report = _manual_report(
        meso, ((meso_ids["left"], "R"), (meso_ids["right"], "S"))
    )
    rr_report = _manual_report(rr, ((rr_ids["left"], "R"), (rr_ids["right"], "R")))

    relationship = classify_entity_relationship(meso, rr, meso_report, rr_report)

    assert relationship is EntityRelationship.STEREOISOMER


def test_absolute_structure_values_and_symmetry_category_remain_separate() -> None:
    entity = _tetrahedral_entity("one")
    structure = SimpleNamespace(
        chemistry=_chemistry(entity),
        metadata={
            "crystal_symmetry": CrystalSymmetry((identity_operation(),)),
            "cif_chemistry": {
                "absolute_structure": {
                    "flack": {
                        "raw": "0.06(3)",
                        "value": 0.06,
                        "standard_uncertainty": 0.03,
                    },
                    "details": "Parsons quotients",
                }
            },
        },
    )

    report = analyze_crystal_stereochemistry(structure)

    assert report.classification is CrystalStereoClass.ENANTIOPURE
    assert report.symmetry_category == "Sohncke (proper operations only)"
    assert report.absolute_structure[0].raw == "0.06(3)"
    assert report.absolute_structure[0].standard_uncertainty == pytest.approx(0.03)
    assert report.absolute_structure_details == "Parsons quotients"


def test_improper_operation_marks_non_sohncke_without_changing_stereo_count() -> None:
    entity = _tetrahedral_entity("one")
    inversion = FractionalAffineOperation(-np.eye(3), np.zeros(3))
    structure = SimpleNamespace(
        chemistry=_chemistry(entity),
        metadata={
            "crystal_symmetry": CrystalSymmetry((identity_operation(), inversion))
        },
    )

    report = analyze_crystal_stereochemistry(structure)

    assert report.symmetry_category == "non-Sohncke (contains improper operation)"
    assert report.classification is CrystalStereoClass.ENANTIOPURE


def test_strict_mode_rejects_current_partial_stereogenic_family_scope() -> None:
    entity = _tetrahedral_entity("one")

    with pytest.raises(CrystalStereoIndeterminateError):
        analyze_crystal_stereochemistry(_chemistry(entity), strict=True)


def test_coordinate_mirror_flips_the_assigned_descriptor() -> None:
    right = assign_stereochemistry(_tetrahedral_entity("right"))
    mirror = assign_stereochemistry(_tetrahedral_entity("mirror", mirror=True))

    assert right.descriptors[0].descriptor in {"R", "S"}
    assert {right.descriptors[0].descriptor, mirror.descriptors[0].descriptor} == {
        "R",
        "S",
    }
