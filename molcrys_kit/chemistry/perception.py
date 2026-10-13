"""Pure bond-order and charge perception plus the legacy crystal adapter."""

from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
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
    PeriodicChemicalEntity,
)
from .solver import order_signature, search_bond_orders
from .valence import (
    CHARGE_BY_VALENCE,
    VALENCES,
    VALENCES_FOR_CHARGE,
    allowed_valences,
    inferred_charge,
)

_VALENCES = VALENCES
_VALENCE_FOR_EXPLICIT_CHARGE = VALENCES_FOR_CHARGE
_CHARGE_BY_VALENCE = CHARGE_BY_VALENCE
_EXPECTED_LENGTHS = {
    ("B", "B"): {1: 1.70, 2: 1.55, 3: 1.40},
    ("B", "C"): {1: 1.58, 2: 1.40, 3: 1.25},
    ("C", "C"): {1: 1.54, 2: 1.34, 3: 1.20},
    ("C", "N"): {1: 1.47, 2: 1.28, 3: 1.16},
    ("C", "O"): {1: 1.43, 2: 1.23},
    ("C", "P"): {1: 1.84, 2: 1.65},
    ("C", "S"): {1: 1.82, 2: 1.60},
    ("N", "N"): {1: 1.45, 2: 1.25, 3: 1.10},
    ("N", "O"): {1: 1.40, 2: 1.21},
    ("O", "O"): {1: 1.48, 2: 1.21},
}


@dataclass(frozen=True, slots=True)
class BondOrderOptions:
    mode: str = "auto"
    strict: bool = False
    max_candidates: int = 64
    max_search_states: int = 100_000
    infer_charges: bool = True

    def __post_init__(self):
        if self.mode not in {"auto", "valence", "geometry"}:
            raise ValueError("mode must be 'auto', 'valence', or 'geometry'")
        if self.max_candidates < 1 or self.max_search_states < 1:
            raise ValueError("search limits must be positive")


@dataclass(frozen=True, slots=True)
class BondOrderResult:
    entity: Any
    status: InferenceStatus
    bond_evidence: tuple[Evidence, ...] = ()
    resonance_classes: tuple[tuple[int, ...], ...] = ()
    alternatives: tuple[Any, ...] = ()
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self):
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "status": self.status.value,
            "entity": _json_value(self.entity),
            "bond_evidence": _json_value(self.bond_evidence),
            "resonance_classes": [list(x) for x in self.resonance_classes],
            "alternatives": [_json_value(x) for x in self.alternatives],
            "diagnostics": _json_value(dict(self.diagnostics)),
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True)


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {k: _json_value(getattr(value, k)) for k in value.__dataclass_fields__}
    return value


def infer_bond_orders(entity, *, total_charge=None, cell=None, options=None):
    """Infer covalent orders without mutating an entity."""
    options = options or BondOrderOptions()
    if total_charge is not None and (not isinstance(total_charge, (int, np.integer))):
        raise ValueError("total_charge must be an integer or None")
    if not isinstance(entity, (FiniteChemicalEntity, PeriodicChemicalEntity)):
        result = BondOrderResult(
            entity,
            InferenceStatus.INDETERMINATE,
            diagnostics={"reason": "unsupported_entity_type"},
        )
        if options.strict:
            raise ChemistryIndeterminateError(
                "bond-order perception unavailable for this entity type"
            )
        return result
    if total_charge is None:
        total_charge = (
            entity.net_charge
            if isinstance(entity, FiniteChemicalEntity)
            else entity.net_charge_per_repeat
        )
    perceived, alternatives, diagnostics, evidence = _infer_entity(
        entity, total_charge, cell, options
    )
    result = BondOrderResult(
        perceived,
        perceived.status,
        evidence,
        _resonance_classes((perceived, *alternatives)),
        alternatives,
        diagnostics,
    )
    if options.strict and perceived.status in {
        InferenceStatus.PROVISIONAL,
        InferenceStatus.INDETERMINATE,
    }:
        raise ChemistryIndeterminateError(
            perceived.warnings[0]
            if perceived.warnings
            else "bond-order perception unresolved"
        )
    return result


def _infer_entity(entity, total_charge, cell, options):
    atom_by_id = {a.atom_id: a for a in entity.atoms}
    bonds = tuple(_classify_bond(b, atom_by_id) for b in entity.bonds)
    variable = []
    choices = {}
    for i, bond in enumerate(bonds):
        if bond.kind is BondKind.COVALENT and bond.order is None:
            domain = _bond_order_choices(bond, atom_by_id)
            if domain:
                variable.append(i)
                choices[i] = domain
    unresolved = [
        a.atom_id
        for a in entity.atoms
        if allowed_valences(a) is None and not is_metal_element(a.element)
    ]
    metal_edges = sum(1 for bond in bonds if bond.kind in {BondKind.COORDINATION, BondKind.METALLIC})

    def score(assign):
        return _assignment_score(bonds, assign, atom_by_id, entity.embedding, cell)

    outcome = search_bond_orders(
        entity.atoms,
        bonds,
        variable,
        choices,
        score=score,
        total_charge=total_charge,
        max_search_states=options.max_search_states,
        max_candidates=options.max_candidates,
    )
    evidence = Evidence(
        source=EvidenceSource.INFERRED,
        method="propagating_valence_and_geometry_solver",
        detail="Residual-valence propagation with bounded deterministic search.",
    )
    diagnostics = {
        "search_states": outcome.states,
        "search_complete": outcome.complete,
        "truncated": not outcome.complete,
        "candidates_found": len(outcome.candidates),
        "candidates_rejected": outcome.rejected,
        "max_search_states": options.max_search_states,
        "max_candidates": options.max_candidates,
        "candidate_scores": tuple(float(score) for score, _ in outcome.candidates),
        "unresolved_atoms": tuple(unresolved),
    }
    if not outcome.candidates:
        warning = (
            f"{entity.entity_id}: no complete valence-consistent bond-order assignment"
            + ("; search budget exhausted" if not outcome.complete else "")
        )
        return (
            _entity_with_orders(
                entity,
                bonds,
                status=InferenceStatus.INDETERMINATE,
                warnings=(warning,),
                evidence=evidence,
            ),
            (),
            diagnostics,
            (evidence,),
        )
    candidates = outcome.candidates
    entities = tuple(
        _entity_with_orders(
            entity, b, status=InferenceStatus.INFERRED, warnings=(), evidence=evidence
        )
        for _, b in candidates
    )
    warnings = []
    if not outcome.complete:
        warnings.append(
            f"{entity.entity_id}: search budget exhausted; result is bounded"
        )
    if unresolved:
        warnings.append(
            f"{entity.entity_id}: unsupported atom valence states remain indeterminate"
        )
    if metal_edges:
        warnings.append(
            f"{entity.entity_id}: {metal_edges} metal edge(s) retained as coordination/metallic evidence"
        )
    gap = float(CHEMISTRY_PERCEPTION_CONFIG.get("COMPETITIVE_SCORE_GAP", 1.0))
    competitive = [
        e for (s, _), e in zip(candidates, entities) if s - candidates[0][0] <= gap
    ]
    # ``max_candidates`` is an output-retention limit.  It must not change the
    # selected best assignment or resolution status; retained alternatives are
    # a view onto the complete bounded search result.
    status = InferenceStatus.PROVISIONAL if warnings else InferenceStatus.INFERRED
    return (
        replace(competitive[0], status=status, warnings=tuple(warnings)),
        tuple(competitive[1:]),
        diagnostics,
        (evidence,),
    )


def _classify_bond(bond, atom_by_id):
    if bond.kind is not BondKind.UNKNOWN:
        return bond
    lm = is_metal_element(atom_by_id[bond.atom1_id].element)
    rm = is_metal_element(atom_by_id[bond.atom2_id].element)
    return replace(
        bond,
        kind=BondKind.METALLIC
        if lm and rm
        else BondKind.COORDINATION
        if lm or rm
        else BondKind.COVALENT,
    )


def _bond_order_choices(bond, atom_by_id):
    left, right = atom_by_id[bond.atom1_id].element, atom_by_id[bond.atom2_id].element
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
    if bond.aromatic:
        return (1, 2)
    return (1, 2, 3) if pair in {("C", "C"), ("C", "N"), ("N", "N")} else (1, 2)


def _assignment_score(bonds, assignment, atom_by_id, embedding, cell):
    if embedding is None:
        return 0.0
    score = 0.0
    for i, order in assignment.items():
        b = bonds[i]
        pair = tuple(
            sorted((atom_by_id[b.atom1_id].element, atom_by_id[b.atom2_id].element))
        )
        expected = _EXPECTED_LENGTHS.get(pair, {}).get(int(order))
        if expected is not None:
            sigma = float(CHEMISTRY_PERCEPTION_CONFIG.get("BOND_LENGTH_SIGMA", 0.08))
            score += ((_bond_distance(b, embedding, cell) - expected) / sigma) ** 2
    return float(score)


def _bond_distance(bond, embedding, cell=None):
    if hasattr(embedding, "position"):
        left = np.asarray(embedding.position(bond.atom1_id), dtype=float)
        right = np.asarray(embedding.position(bond.atom2_id), dtype=float)
    else:
        positions = {i: np.asarray(p, dtype=float) for i, p in embedding.coordinates_A}
        left = np.asarray(positions[bond.atom1_id], dtype=float)
        right = np.asarray(positions[bond.atom2_id], dtype=float)
    if cell is not None and any(bond.atom2_image_shift):
        right = right + np.asarray(bond.atom2_image_shift, dtype=float) @ np.asarray(
            cell, dtype=float
        )
        return float(np.linalg.norm(right - left))
    if hasattr(embedding, "vector"):
        try:
            vector = embedding.vector(
                bond.atom1_id, bond.atom2_id, bond.atom2_image_shift
            )
        except TypeError:
            vector = embedding.vector(
                bond.atom1_id, bond.atom2_id, bond.atom2_image_shift, cell
            )
        return float(np.linalg.norm(np.asarray(vector, dtype=float)))
    return float(np.linalg.norm(right - left))


def _entity_with_orders(entity, bonds, *, status, warnings, evidence):
    sums = {a.atom_id: 0 for a in entity.atoms}
    for b in bonds:
        if b.kind is BondKind.COVALENT and b.order is not None:
            value = int(round(b.order))
            sums[b.atom1_id] += value
            sums[b.atom2_id] += value
            if b.atom1_id == b.atom2_id:
                sums[b.atom1_id] += value
    atoms = tuple(
        replace(
            a,
            formal_charge=a.formal_charge
            if a.formal_charge is not None
            else inferred_charge(a, sums[a.atom_id]),
            evidence=a.evidence + (evidence,),
        )
        for a in entity.atoms
    )
    charges = [a.formal_charge for a in atoms]
    charge = sum(charges) if charges and all(v is not None for v in charges) else None
    warnings = tuple(warnings)
    if charge is None and status is InferenceStatus.INFERRED:
        status = InferenceStatus.PROVISIONAL
        warnings += ("Atomic formal charges remain unresolved.",)
    updated = tuple(replace(b, evidence=b.evidence + (evidence,)) for b in bonds)
    kw = dict(
        atoms=atoms,
        bonds=updated,
        status=status,
        warnings=warnings,
        evidence=entity.evidence + (evidence,),
    )
    kw[
        "net_charge_per_repeat"
        if isinstance(entity, PeriodicChemicalEntity)
        else "net_charge"
    ] = charge
    return replace(entity, **kw)


def _resonance_classes(entities):
    groups = {}
    for i, e in enumerate(entities):
        # Aromatic 1/2 Kekule choices are equivalent; retain numeric orders on
        # bonds but canonicalize them into one resonance class key.
        key = tuple(
            1.5 if b.aromatic else (0.0 if b.order is None else float(b.order))
            for b in e.bonds
        )
        groups.setdefault(key, []).append(i)
    return tuple(tuple(v) for _, v in sorted(groups.items(), key=lambda x: x[0]))


def infer_chemistry(structure, *, strict=False, max_candidates=64):
    previous = structure.chemistry
    previous_entities = tuple(m.chemical_entity for m in structure.molecules)
    if previous is None:
        base = annotate_chemistry(structure)
        structure._set_chemistry(previous)
        for m, e in zip(structure.molecules, previous_entities):
            m.chemical_entity = e
    else:
        base = previous
    components = []
    alternatives = []
    warnings = []
    unsupported = False
    for component in base.components:
        if not isinstance(component, (FiniteChemicalEntity, PeriodicChemicalEntity)):
            unsupported = True
            warnings.append(
                f"{component.entity_id}: bond-order perception for this entity type is unavailable"
            )
            continue
        result = infer_bond_orders(
            component,
            cell=getattr(structure, "lattice", None),
            options=BondOrderOptions(max_candidates=max_candidates),
        )
        components.append(result.entity)
        alternatives.append(tuple(result.alternatives))
        warnings.extend(result.entity.warnings)
    status = (
        InferenceStatus.INDETERMINATE if unsupported else _combined_status(components)
    )
    if strict and status in {
        InferenceStatus.PROVISIONAL,
        InferenceStatus.INDETERMINATE,
    }:
        raise ChemistryIndeterminateError(
            warnings[0] if warnings else "chemistry perception is indeterminate"
        )
    result = CrystalChemistry(
        components=tuple(components),
        atom_ids_by_global_index=base.atom_ids_by_global_index,
        status=status,
        evidence=tuple(ev for c in components for ev in c.evidence),
        warnings=tuple(dict.fromkeys(warnings)),
        alternatives=tuple(alternatives),
    )
    for m, e in zip(structure.molecules, components):
        m.chemical_entity = e
    structure._set_chemistry(result)
    return result


def _combined_status(components):
    statuses = {c.status for c in components}
    if not components or InferenceStatus.INDETERMINATE in statuses:
        return InferenceStatus.INDETERMINATE
    return (
        InferenceStatus.PROVISIONAL
        if InferenceStatus.PROVISIONAL in statuses
        else InferenceStatus.INFERRED
    )


def _perceive_finite_entity(entity, *, max_candidates, cell=None):
    result = infer_bond_orders(
        entity, cell=cell, options=BondOrderOptions(max_candidates=max_candidates)
    )
    return result.entity, tuple(result.alternatives)


__all__ = [
    "BondOrderOptions",
    "BondOrderResult",
    "infer_bond_orders",
    "infer_chemistry",
]
