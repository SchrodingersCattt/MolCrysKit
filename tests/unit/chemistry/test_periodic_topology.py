"""Periodic quotient-graph dimensionality tests."""

import numpy as np
from ase import Atoms

from molcrys_kit.chemistry import (
    ChemicalBond,
    FiniteChemicalEntity,
    PeriodicChemicalEntity,
    analyze_periodic_topology,
    annotate_chemistry,
    infer_chemistry,
)
from molcrys_kit.io.cif import identify_molecules
from molcrys_kit.structures import MolecularCrystal


def _bond(left: str, right: str, shift) -> ChemicalBond:
    return ChemicalBond(left, right, atom2_image_shift=shift)


def test_face_crossing_tree_edge_remains_finite() -> None:
    topology = analyze_periodic_topology(
        ("a", "b"),
        (_bond("a", "b", (1, 0, 0)),),
    )
    assert topology.rank == 0
    assert topology.translation_generators == ()


def test_cycle_translation_rank_distinguishes_1d_2d_and_3d() -> None:
    for expected_rank, shifts in [
        (1, [(1, 0, 0)]),
        (2, [(1, 0, 0), (0, 1, 0)]),
        (3, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ]:
        topology = analyze_periodic_topology(
            ("a",),
            tuple(_bond("a", "a", shift) for shift in shifts),
        )
        assert topology.rank == expected_rank
        assert topology.translation_generators == tuple(shifts)


def test_dependent_cycle_translations_do_not_inflate_rank() -> None:
    topology = analyze_periodic_topology(
        ("a",),
        (
            _bond("a", "a", (1, 0, 0)),
            _bond("a", "a", (2, 0, 0)),
            _bond("a", "a", (-3, 0, 0)),
        ),
    )
    assert topology.rank == 1
    assert topology.translation_generators == ((1, 0, 0),)


def test_annotation_emits_periodic_entity_and_perception_preserves_it() -> None:
    cell = np.diag([1.45, 10.0, 10.0])
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=(True, False, False))
    molecules = identify_molecules(atoms, bond_thresholds={("C", "C"): 1.6})
    crystal = MolecularCrystal(cell, molecules, pbc=(True, False, False))

    annotated = annotate_chemistry(crystal)
    assert isinstance(annotated.components[0], PeriodicChemicalEntity)
    assert annotated.component_dimensions == (1,)
    assert not annotated.is_molecular_crystal

    perceived = infer_chemistry(crystal)
    assert isinstance(perceived.components[0], PeriodicChemicalEntity)
    assert perceived.components[0].periodic_rank == 1


def test_annotation_keeps_unwrapped_finite_molecule_0d() -> None:
    cell = np.diag([10.0, 10.0, 10.0])
    atoms = Atoms(
        "C2",
        scaled_positions=[[0.95, 0.5, 0.5], [0.05, 0.5, 0.5]],
        cell=cell,
        pbc=True,
    )
    molecules = identify_molecules(atoms, bond_thresholds={("C", "C"): 1.2})
    crystal = MolecularCrystal(cell, molecules)

    chemistry = annotate_chemistry(crystal)
    assert isinstance(chemistry.components[0], FiniteChemicalEntity)
    assert chemistry.component_dimensions == (0,)
    assert chemistry.is_molecular_crystal
