"""Coordinate stereochemistry using a self-contained CIP digraph engine.

The implementation in this module does not call an external chemistry engine.
It constructs hierarchical ligand digraphs from stable atom identities and
applies IUPAC Blue Book 2013 sequence rules in rule order. This first public
slice assigns ordinary tetrahedral ``R``/``S`` centers using sequence rules
1a, 1b, and 2, including multiple-bond and saturated-ring duplicate nodes.
Cases requiring rules 3--5 remain explicitly indeterminate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from ..constants.config import STEREOCHEMISTRY_CONFIG
from .models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Embedding,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
    PeriodicChemicalEntity,
)


class StereoKind(str, Enum):
    """Supported stereogenic-unit families."""

    TETRAHEDRAL = "tetrahedral"


@dataclass(frozen=True)
class StereoDescriptor:
    """One auditable atom- or bond-level stereochemical conclusion."""

    kind: StereoKind
    center_atom_id: str
    descriptor: str | None
    cip_order: tuple[str, ...]
    status: InferenceStatus
    reason: str
    rules_applied: tuple[str, ...]
    signed_volume: float | None = None

    @property
    def is_assigned(self) -> bool:
        return self.descriptor is not None


@dataclass(frozen=True)
class StereoReport:
    """Stereochemical conclusions for one chemical entity and embedding."""

    entity_id: str
    descriptors: tuple[StereoDescriptor, ...]
    status: InferenceStatus
    evidence: tuple[Evidence, ...]
    warnings: tuple[str, ...] = ()

    def for_atom(self, atom_id: str) -> StereoDescriptor | None:
        """Return the descriptor record for one stable atom identity."""
        return next(
            (item for item in self.descriptors if item.center_atom_id == atom_id),
            None,
        )


@dataclass(frozen=True)
class _DigraphNode:
    element: str
    isotope: int | None
    duplicate_origin_depth: int | None
    children: tuple["_DigraphNode", ...] = ()


class _CIPIndeterminate(ValueError):
    pass


def assign_stereochemistry(
    entity: FiniteChemicalEntity | PeriodicChemicalEntity,
    embedding: Embedding | None = None,
) -> StereoReport:
    """Assign coordinate-supported tetrahedral ``R``/``S`` descriptors.

    Near-planar geometry, unresolved bond orders, periodic-image ligands, or
    ligand ties that may require sequence rules 3--5 are reported as
    indeterminate rather than forced into an ``R``/``S`` label.
    """
    coordinates = embedding or entity.embedding
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    adjacency = _adjacency(entity.bonds)
    evidence = Evidence(
        source=EvidenceSource.INFERRED,
        method="IUPAC_Blue_Book_2013_P-92_rules_1a_1b_2",
        detail="Hierarchical digraph with multiple-bond and ring duplicate nodes.",
    )
    descriptors: list[StereoDescriptor] = []
    warnings: list[str] = []

    for center in entity.atoms:
        incident = adjacency.get(center.atom_id, ())
        explicit_ligands = [
            (other_id, bond)
            for other_id, bond in incident
            if bond.kind not in {BondKind.IONIC, BondKind.METALLIC}
        ]
        implicit_count = int(center.implicit_hydrogens or 0)
        if len(explicit_ligands) + implicit_count != 4:
            continue
        descriptor = _assign_tetrahedral_center(
            center,
            explicit_ligands,
            implicit_count,
            atoms,
            adjacency,
            coordinates,
        )
        descriptors.append(descriptor)
        if descriptor.status is InferenceStatus.INDETERMINATE:
            warnings.append(f"{center.atom_id}: {descriptor.reason}")

    status = (
        InferenceStatus.INDETERMINATE
        if any(item.status is InferenceStatus.INDETERMINATE for item in descriptors)
        else InferenceStatus.INFERRED
    )
    return StereoReport(
        entity_id=entity.entity_id,
        descriptors=tuple(descriptors),
        status=status,
        evidence=(evidence,),
        warnings=tuple(warnings),
    )


def _assign_tetrahedral_center(
    center: ChemicalAtom,
    explicit_ligands: list[tuple[str, ChemicalBond]],
    implicit_count: int,
    atoms: dict[str, ChemicalAtom],
    adjacency: dict[str, tuple[tuple[str, ChemicalBond], ...]],
    embedding: Embedding | None,
) -> StereoDescriptor:
    rules = ("P-92.2.1 (1a)", "P-92.2.2 (1b)", "P-92.3 (2)")
    if any(any(bond.atom2_image_shift) for _, bond in explicit_ligands):
        return _indeterminate(center.atom_id, rules, "periodic-image CIP ligands are not expanded yet")
    if implicit_count > 1:
        return _indeterminate(center.atom_id, rules, "multiple indistinguishable implicit hydrogens")

    ligand_ids = [atom_id for atom_id, _ in explicit_ligands]
    if implicit_count:
        ligand_ids.append(f"{center.atom_id}:implicit-H")
    try:
        priorities = [
            (
                _implicit_hydrogen_priority()
                if ligand_id.endswith(":implicit-H")
                else _ligand_priority(
                    center.atom_id,
                    ligand_id,
                    atoms,
                    adjacency,
                )
            )
            for ligand_id in ligand_ids
        ]
    except _CIPIndeterminate as exc:
        return _indeterminate(center.atom_id, rules, str(exc))

    if len(set(priorities)) != 4:
        tied = tuple(
            ligand_id
            for ligand_id, priority in zip(ligand_ids, priorities)
            if priorities.count(priority) > 1
        )
        return StereoDescriptor(
            kind=StereoKind.TETRAHEDRAL,
            center_atom_id=center.atom_id,
            descriptor=None,
            cip_order=tied,
            status=InferenceStatus.INDETERMINATE,
            reason="ligands remain tied after sequence rules 1a, 1b, and 2",
            rules_applied=rules,
        )

    ranked = sorted(
        zip(ligand_ids, priorities),
        key=lambda item: item[1],
        reverse=True,
    )
    cip_order = tuple(ligand_id for ligand_id, _ in ranked)
    if embedding is None:
        return _indeterminate(center.atom_id, rules, "3D embedding is unavailable", cip_order)

    try:
        center_position = np.asarray(embedding.position(center.atom_id), dtype=float)
        vectors = [
            _ligand_vector(ligand_id, center.atom_id, center_position, cip_order, embedding)
            for ligand_id in cip_order
        ]
    except (KeyError, ValueError) as exc:
        return _indeterminate(center.atom_id, rules, str(exc), cip_order)

    signed_volume = _normalized_tetrahedral_volume(vectors)
    threshold = STEREOCHEMISTRY_CONFIG["MIN_NORMALIZED_TETRAHEDRAL_VOLUME"]
    if not np.isfinite(signed_volume) or abs(signed_volume) < threshold:
        return StereoDescriptor(
            kind=StereoKind.TETRAHEDRAL,
            center_atom_id=center.atom_id,
            descriptor=None,
            cip_order=cip_order,
            status=InferenceStatus.INDETERMINATE,
            reason="tetrahedral coordinates are planar or numerically degenerate",
            rules_applied=rules,
            signed_volume=float(signed_volume),
        )
    return StereoDescriptor(
        kind=StereoKind.TETRAHEDRAL,
        center_atom_id=center.atom_id,
        descriptor="R" if signed_volume < 0.0 else "S",
        cip_order=cip_order,
        status=InferenceStatus.INFERRED,
        reason="assigned from CIP order and signed tetrahedral volume",
        rules_applied=rules,
        signed_volume=float(signed_volume),
    )


def _indeterminate(center_id, rules, reason, cip_order=()) -> StereoDescriptor:
    return StereoDescriptor(
        kind=StereoKind.TETRAHEDRAL,
        center_atom_id=center_id,
        descriptor=None,
        cip_order=tuple(cip_order),
        status=InferenceStatus.INDETERMINATE,
        reason=reason,
        rules_applied=rules,
    )


def _adjacency(bonds: tuple[ChemicalBond, ...]):
    result: dict[str, list[tuple[str, ChemicalBond]]] = {}
    for bond in bonds:
        result.setdefault(bond.atom1_id, []).append((bond.atom2_id, bond))
        result.setdefault(bond.atom2_id, []).append((bond.atom1_id, bond))
    return {atom_id: tuple(items) for atom_id, items in result.items()}


def _ligand_priority(center_id, ligand_id, atoms, adjacency):
    counter = [0]
    root = _build_digraph(
        ligand_id,
        parent_id=center_id,
        path=(center_id, ligand_id),
        atoms=atoms,
        adjacency=adjacency,
        depth=1,
        counter=counter,
    )
    rule_1a = _canonical_breadth_code(root, (_atomic_number,))
    rule_1b = _canonical_breadth_code(root, (_atomic_number, _duplicate_priority))
    rule_2 = _canonical_breadth_code(
        root,
        (_atomic_number, _duplicate_priority, _mass_number),
    )
    return (rule_1a, rule_1b, rule_2)


def _implicit_hydrogen_priority():
    root = _DigraphNode("H", None, None)
    return (
        _canonical_breadth_code(root, (_atomic_number,)),
        _canonical_breadth_code(root, (_atomic_number, _duplicate_priority)),
        _canonical_breadth_code(root, (_atomic_number, _duplicate_priority, _mass_number)),
    )


def _build_digraph(
    atom_id,
    *,
    parent_id,
    path,
    atoms,
    adjacency,
    depth,
    counter,
) -> _DigraphNode:
    counter[0] += 1
    if counter[0] > STEREOCHEMISTRY_CONFIG["MAX_CIP_DIGRAPH_NODES"]:
        raise _CIPIndeterminate("CIP digraph exceeds the configured node limit")
    atom = atoms[atom_id]
    children: list[_DigraphNode] = []
    for neighbor_id, bond in adjacency.get(atom_id, ()):
        multiplicity = _bond_multiplicity(bond)
        if neighbor_id == parent_id:
            for _ in range(multiplicity - 1):
                parent = atoms[parent_id]
                children.append(_duplicate_node(parent, depth - 1))
            continue
        neighbor = atoms[neighbor_id]
        if neighbor_id in path:
            origin_depth = path.index(neighbor_id)
            children.append(_duplicate_node(neighbor, origin_depth))
        else:
            children.append(
                _build_digraph(
                    neighbor_id,
                    parent_id=atom_id,
                    path=(*path, neighbor_id),
                    atoms=atoms,
                    adjacency=adjacency,
                    depth=depth + 1,
                    counter=counter,
                )
            )
        for _ in range(multiplicity - 1):
            children.append(_duplicate_node(neighbor, depth + 1))
    children.extend(
        _DigraphNode("H", None, None) for _ in range(int(atom.implicit_hydrogens or 0))
    )
    return _DigraphNode(atom.element, atom.isotope, None, tuple(children))


def _duplicate_node(atom: ChemicalAtom, origin_depth: int) -> _DigraphNode:
    return _DigraphNode(atom.element, atom.isotope, origin_depth)


def _bond_multiplicity(bond: ChemicalBond) -> int:
    if bond.aromatic:
        raise _CIPIndeterminate("mancude aromatic duplicate-node averaging is not implemented")
    if bond.order is None:
        raise _CIPIndeterminate("bond order is unresolved")
    rounded = round(bond.order)
    if abs(bond.order - rounded) > 1.0e-8 or rounded not in {1, 2, 3}:
        raise _CIPIndeterminate(f"unsupported CIP bond order {bond.order}")
    return int(rounded)


def _canonical_breadth_code(root: _DigraphNode, selectors) -> tuple:
    @lru_cache(maxsize=None)
    def subtree_key(node: _DigraphNode):
        child_keys = sorted(
            (subtree_key(child) for child in node.children),
            reverse=True,
        )
        return tuple(selector(node) for selector in selectors) + (tuple(child_keys),)

    levels = []
    frontier = [root]
    while frontier:
        levels.append(tuple(tuple(selector(node) for selector in selectors) for node in frontier))
        next_frontier = []
        for node in frontier:
            next_frontier.extend(
                sorted(node.children, key=subtree_key, reverse=True)
            )
        frontier = next_frontier
    return tuple(levels)


def _normal_element(element: str) -> str:
    return "H" if element in {"D", "T"} else element


def _atomic_number(node: _DigraphNode) -> int:
    return int(atomic_numbers[_normal_element(node.element)])


def _duplicate_priority(node: _DigraphNode) -> int:
    if node.duplicate_origin_depth is None:
        return 0
    return STEREOCHEMISTRY_CONFIG["MAX_CIP_DIGRAPH_NODES"] - node.duplicate_origin_depth


def _mass_number(node: _DigraphNode) -> float:
    if node.isotope is not None:
        return float(node.isotope)
    if node.element == "D":
        return 2.0
    if node.element == "T":
        return 3.0
    return float(atomic_masses[_atomic_number(node)])


def _ligand_vector(ligand_id, center_id, center_position, cip_order, embedding):
    if not ligand_id.endswith(":implicit-H"):
        return np.asarray(embedding.position(ligand_id), dtype=float) - center_position
    explicit = [
        np.asarray(embedding.position(other_id), dtype=float) - center_position
        for other_id in cip_order
        if other_id != ligand_id
    ]
    unit_vectors = [vector / np.linalg.norm(vector) for vector in explicit]
    direction = -np.sum(unit_vectors, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError(f"implicit hydrogen direction at {center_id} is indeterminate")
    return direction / norm


def _normalized_tetrahedral_volume(vectors) -> float:
    reference = vectors[3]
    edges = [np.asarray(vector, dtype=float) - reference for vector in vectors[:3]]
    norms = [float(np.linalg.norm(edge)) for edge in edges]
    if any(norm <= 1.0e-12 for norm in norms):
        return 0.0
    return float(np.linalg.det(np.vstack(edges)) / np.prod(norms))


__all__ = [
    "StereoDescriptor",
    "StereoKind",
    "StereoReport",
    "assign_stereochemistry",
]
