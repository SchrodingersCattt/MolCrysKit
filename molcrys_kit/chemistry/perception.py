"""Self-contained bond-order and formal-charge perception.

This module treats coordinate-derived connectivity as evidence rather than
truth. It classifies edge semantics, solves bounded valence constraints, ranks
chemically valid bond-order assignments by observed lengths, and retains
equally plausible alternatives instead of silently choosing one resonance or
charge representation.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import prod

import numpy as np

from ..constants import is_metal_element
from ..constants.config import CHEMISTRY_PERCEPTION_CONFIG
from .annotation import ChemistryIndeterminateError, annotate_chemistry
from .models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    CrystalChemistry,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
)


_VALENCES: dict[str, tuple[int, ...]] = {
    "H": (1,),
    "D": (1,),
    "B": (3, 4),
    "C": (4,),
    "N": (2, 3, 4),
    "O": (1, 2, 3),
    "F": (1,),
    "Si": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "Se": (2, 4, 6),
    "Cl": (0, 1),
    "Br": (0, 1),
    "I": (0, 1),
}

_VALENCE_FOR_EXPLICIT_CHARGE: dict[tuple[str, int], tuple[int, ...]] = {
    ("B", -1): (4,),
    ("C", -1): (3,),
    ("C", 1): (3,),
    ("N", -1): (2,),
    ("N", 1): (4,),
    ("O", -1): (1,),
    ("O", 1): (3,),
    ("F", -1): (0,),
    ("Cl", -1): (0,),
    ("Br", -1): (0,),
    ("I", -1): (0,),
}

_CHARGE_BY_VALENCE: dict[str, dict[int, int]] = {
    "B": {3: 0, 4: -1},
    "C": {3: 0, 4: 0},
    "N": {2: -1, 3: 0, 4: 1},
    "O": {1: -1, 2: 0, 3: 1},
    "F": {0: -1, 1: 0},
    "Cl": {0: -1, 1: 0},
    "Br": {0: -1, 1: 0},
    "I": {0: -1, 1: 0},
    "Si": {4: 0},
    "P": {3: 0, 5: 0},
    "S": {2: 0, 4: 0, 6: 0},
    "Se": {2: 0, 4: 0, 6: 0},
    "H": {1: 0},
    "D": {1: 0},
}

_EXPECTED_LENGTHS: dict[tuple[str, str], dict[int, float]] = {
    ("C", "C"): {1: 1.54, 2: 1.34, 3: 1.20},
    ("C", "N"): {1: 1.47, 2: 1.28, 3: 1.16},
    ("C", "O"): {1: 1.43, 2: 1.23},
    ("N", "N"): {1: 1.45, 2: 1.25, 3: 1.10},
    ("N", "O"): {1: 1.40, 2: 1.21},
    ("O", "O"): {1: 1.48, 2: 1.21},
}

def infer_chemistry(
    structure,
    *,
    strict: bool = False,
    max_candidates: int = 64,
) -> CrystalChemistry:
    """Infer edge semantics, bond orders, and atomic formal charges.

    The input crystal is annotated in place only after the complete result has
    been constructed. ``strict=True`` therefore fails atomically when any
    component remains provisional or indeterminate.
    """
    previous_chemistry = structure.chemistry
    previous_entities = tuple(
        molecule.chemical_entity for molecule in structure.molecules
    )
    if previous_chemistry is None:
        base = annotate_chemistry(structure)
        # Annotation currently provides the public graph-to-atom-id bridge and
        # attaches its provisional result. Restore the caller's state while
        # perception is running so strict failure remains atomic.
        structure._chemistry = previous_chemistry
        for molecule, previous_entity in zip(structure.molecules, previous_entities):
            molecule.chemical_entity = previous_entity
    else:
        base = previous_chemistry
    components: list[FiniteChemicalEntity] = []
    all_alternatives: list[tuple[FiniteChemicalEntity, ...]] = []
    warnings: list[str] = []

    for component in base.components:
        if not isinstance(component, FiniteChemicalEntity):
            warnings.append(
                f"{component.entity_id}: bond-order perception for this entity type is unavailable"
            )
            continue
        perceived, alternatives = _perceive_finite_entity(
            component,
            max_candidates=max_candidates,
        )
        components.append(perceived)
        all_alternatives.append(alternatives)
        warnings.extend(perceived.warnings)

    status = _combined_status(components)
    if strict and status in {InferenceStatus.PROVISIONAL, InferenceStatus.INDETERMINATE}:
        detail = warnings[0] if warnings else "chemistry perception is indeterminate"
        raise ChemistryIndeterminateError(detail)

    result = CrystalChemistry(
        components=tuple(components),
        atom_ids_by_global_index=base.atom_ids_by_global_index,
        status=status,
        evidence=tuple(evidence for component in components for evidence in component.evidence),
        warnings=tuple(dict.fromkeys(warnings)),
        alternatives=tuple(all_alternatives),
    )
    for molecule, entity in zip(structure.molecules, components):
        molecule.chemical_entity = entity
    structure._chemistry = result
    return result


def _perceive_finite_entity(
    entity: FiniteChemicalEntity,
    *,
    max_candidates: int,
) -> tuple[FiniteChemicalEntity, tuple[FiniteChemicalEntity, ...]]:
    atom_by_id = {atom.atom_id: atom for atom in entity.atoms}
    classified = tuple(_classify_bond(bond, atom_by_id) for bond in entity.bonds)
    order_candidates, search_exhaustive = _solve_bond_orders(
        entity.atoms,
        classified,
        entity.embedding,
        max_candidates=max_candidates,
    )
    evidence = Evidence(
        source=EvidenceSource.INFERRED,
        method="bounded_valence_and_bond_length_solver",
    )

    if not order_candidates:
        fallback = tuple(
            replace(bond, order=1.0, evidence=bond.evidence + (evidence,))
            if bond.kind is BondKind.COVALENT
            else bond
            for bond in classified
        )
        reason = (
            "no complete valence-consistent bond-order assignment"
            if search_exhaustive
            else "bond-order search space exceeds the exhaustive-search limit"
        )
        warning = f"{entity.entity_id}: {reason}; single-bond fallback is provisional"
        perceived = _entity_with_orders(
            entity,
            fallback,
            status=InferenceStatus.INDETERMINATE,
            warnings=(warning,),
            evidence=evidence,
        )
        return perceived, ()

    candidate_entities = tuple(
        _entity_with_orders(
            entity,
            candidate,
            status=InferenceStatus.INFERRED,
            warnings=(),
            evidence=evidence,
        )
        for _, candidate in order_candidates
    )
    best_score = order_candidates[0][0]
    competitive = tuple(
        candidate_entity
        for (score, _), candidate_entity in zip(order_candidates, candidate_entities)
        if score - best_score
        <= CHEMISTRY_PERCEPTION_CONFIG["COMPETITIVE_SCORE_GAP"]
    )
    if len(competitive) == 1:
        return competitive[0], ()

    warning = (
        f"{entity.entity_id}: {len(competitive)} bond-order assignments are chemically "
        "competitive; showing the deterministic best candidate"
    )
    best = replace(
        competitive[0],
        status=InferenceStatus.PROVISIONAL,
        warnings=(warning,),
    )
    return best, competitive[1:]


def _classify_bond(
    bond: ChemicalBond,
    atom_by_id: dict[str, ChemicalAtom],
) -> ChemicalBond:
    left_metal = is_metal_element(atom_by_id[bond.atom1_id].element)
    right_metal = is_metal_element(atom_by_id[bond.atom2_id].element)
    if left_metal and right_metal:
        kind = BondKind.METALLIC
    elif left_metal or right_metal:
        kind = BondKind.COORDINATION
    else:
        kind = BondKind.COVALENT
    return replace(bond, kind=kind)


def _solve_bond_orders(
    atoms: tuple[ChemicalAtom, ...],
    bonds: tuple[ChemicalBond, ...],
    embedding,
    *,
    max_candidates: int,
) -> tuple[list[tuple[float, tuple[ChemicalBond, ...]]], bool]:
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    covalent_indices = [
        index for index, bond in enumerate(bonds) if bond.kind is BondKind.COVALENT
    ]
    choices = [
        _bond_order_choices(bonds[index], atom_by_id, embedding)
        for index in covalent_indices
    ]
    if not choices:
        return [(0.0, bonds)], True
    if (
        prod(len(options) for options in choices)
        > CHEMISTRY_PERCEPTION_CONFIG["MAX_EXHAUSTIVE_ASSIGNMENTS"]
    ):
        return [], False

    candidates: list[tuple[float, tuple[ChemicalBond, ...]]] = []
    for assignment in product(*choices):
        valence_sums = {atom.atom_id: 0 for atom in atoms}
        for bond_index, order in zip(covalent_indices, assignment):
            bond = bonds[bond_index]
            valence_sums[bond.atom1_id] += order
            valence_sums[bond.atom2_id] += order
        if not _valid_valence_sums(atoms, valence_sums):
            continue
        assigned = list(bonds)
        for bond_index, order in zip(covalent_indices, assignment):
            assigned[bond_index] = replace(assigned[bond_index], order=float(order))
        score = _assignment_score(tuple(assigned), atom_by_id, embedding)
        candidates.append((score, tuple(assigned)))
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda item: (item[0], _order_signature(item[1])))
    return candidates, True


def _bond_order_choices(bond, atom_by_id, embedding) -> tuple[int, ...]:
    left = atom_by_id[bond.atom1_id].element
    right = atom_by_id[bond.atom2_id].element
    if left in {"H", "D", "F", "Cl", "Br", "I"} or right in {
        "H",
        "D",
        "F",
        "Cl",
        "Br",
        "I",
    }:
        return (1,)
    pair = tuple(sorted((left, right)))
    max_order = 3 if pair in {("C", "C"), ("C", "N"), ("N", "N")} else 2
    if embedding is not None:
        distance = _bond_distance(bond, embedding)
        if distance > CHEMISTRY_PERCEPTION_CONFIG["SINGLE_BOND_DISTANCE"]:
            max_order = 1
        elif distance > CHEMISTRY_PERCEPTION_CONFIG["TRIPLE_BOND_DISTANCE"]:
            max_order = min(max_order, 2)
    return tuple(range(1, max_order + 1))


def _valid_valence_sums(atoms, sums: dict[str, int]) -> bool:
    for atom in atoms:
        if is_metal_element(atom.element):
            continue
        allowed = _allowed_valences(atom)
        if allowed is not None and sums[atom.atom_id] not in allowed:
            return False
    return True


def _allowed_valences(atom: ChemicalAtom) -> tuple[int, ...] | None:
    if atom.formal_charge is not None:
        explicit = _VALENCE_FOR_EXPLICIT_CHARGE.get((atom.element, atom.formal_charge))
        if explicit is not None:
            return explicit
    return _VALENCES.get(atom.element)


def _assignment_score(bonds, atom_by_id, embedding) -> float:
    if embedding is None:
        return 0.0
    score = 0.0
    for bond in bonds:
        if bond.kind is not BondKind.COVALENT or bond.order is None:
            continue
        pair = tuple(sorted((atom_by_id[bond.atom1_id].element, atom_by_id[bond.atom2_id].element)))
        expected = _EXPECTED_LENGTHS.get(pair, {}).get(int(bond.order))
        if expected is None:
            continue
        score += (
            (_bond_distance(bond, embedding) - expected)
            / CHEMISTRY_PERCEPTION_CONFIG["BOND_LENGTH_SIGMA"]
        ) ** 2
    return float(score)


def _bond_distance(bond: ChemicalBond, embedding) -> float:
    left = np.asarray(embedding.position(bond.atom1_id), dtype=float)
    right = np.asarray(embedding.position(bond.atom2_id), dtype=float)
    return float(np.linalg.norm(right - left))


def _entity_with_orders(entity, bonds, *, status, warnings, evidence):
    valence_sums = {atom.atom_id: 0 for atom in entity.atoms}
    for bond in bonds:
        if bond.kind is BondKind.COVALENT and bond.order is not None:
            valence_sums[bond.atom1_id] += int(bond.order)
            valence_sums[bond.atom2_id] += int(bond.order)
    atoms = tuple(
        replace(
            atom,
            formal_charge=(
                atom.formal_charge
                if atom.formal_charge is not None
                else _CHARGE_BY_VALENCE.get(atom.element, {}).get(valence_sums[atom.atom_id])
            ),
            evidence=atom.evidence + (evidence,),
        )
        for atom in entity.atoms
    )
    charges = [atom.formal_charge for atom in atoms]
    return replace(
        entity,
        atoms=atoms,
        bonds=tuple(
            replace(bond, evidence=bond.evidence + (evidence,)) for bond in bonds
        ),
        net_charge=sum(charges) if charges and all(value is not None for value in charges) else None,
        status=status,
        evidence=entity.evidence + (evidence,),
        warnings=warnings,
    )


def _order_signature(bonds) -> tuple[float, ...]:
    return tuple(0.0 if bond.order is None else bond.order for bond in bonds)


def _combined_status(components) -> InferenceStatus:
    statuses = {component.status for component in components}
    if not components or InferenceStatus.INDETERMINATE in statuses:
        return InferenceStatus.INDETERMINATE
    if InferenceStatus.PROVISIONAL in statuses:
        return InferenceStatus.PROVISIONAL
    return InferenceStatus.INFERRED


__all__ = ["infer_chemistry"]
