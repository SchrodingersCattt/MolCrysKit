"""Graph aromaticity authority.

A ring is aromatic when it is a chordless 5-, 6-, or 7-membered cycle of
C/N/O/S and its pi-electron count is 4n+2. Geometry can rank Kekulé forms.
It does not grant or revoke this flag. Thiophene sulfur and furan/pyrrole
heteroatoms contribute a lone pair; pyridine nitrogen, pyridinium N–H+,
and carbon contribute one pi electron.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants.config import CHEMISTRY_PERCEPTION_CONFIG
from .models import BondKind, ChemicalAtom, ChemicalBond, EvidenceSource

_HYDROGEN = frozenset({"H", "D"})
_PI_ELEMENTS = frozenset({"C", "N", "O", "S"})


@dataclass(frozen=True, slots=True)
class AromaticityAssignment:
    """Bond flags plus implicit hydrogens required by the aromatic valence."""

    bonds: tuple[ChemicalBond, ...]
    implicit_hydrogens: dict[str, int]
    ring_count: int


def apply_aromaticity(
    atoms: tuple[ChemicalAtom, ...] | list[ChemicalAtom],
    bonds: tuple[ChemicalBond, ...] | list[ChemicalBond],
) -> AromaticityAssignment:
    """Return bonds with inferred aromatic flags and solver-local implicit H."""
    atoms = tuple(atoms)
    bonds = tuple(bonds)
    aromatic_bonds, ring_count = _aromatic_bond_indices(atoms, bonds)
    updated = []
    for index, bond in enumerate(bonds):
        if index in aromatic_bonds and not bond.aromatic and not _locked(bond):
            updated.append(
                bond.__class__(
                    atom1_id=bond.atom1_id,
                    atom2_id=bond.atom2_id,
                    order=bond.order,
                    kind=bond.kind,
                    aromatic=True,
                    atom2_image_shift=bond.atom2_image_shift,
                    stereochemistry=bond.stereochemistry,
                    evidence=bond.evidence,
                )
            )
        else:
            updated.append(bond)
    updated_bonds = tuple(updated)
    return AromaticityAssignment(
        updated_bonds,
        _implicit_hydrogens(atoms, updated_bonds),
        ring_count,
    )


def _locked(bond: ChemicalBond) -> bool:
    """A user-confirmed non-aromatic bond is not reclassified."""
    return (not bond.aromatic) and any(
        evidence.source is EvidenceSource.USER_CONFIRMED for evidence in bond.evidence
    )


def _aromatic_bond_indices(atoms, bonds) -> tuple[set[int], int]:
    adjacency: dict[str, list[tuple[str, int]]] = {atom.atom_id: [] for atom in atoms}
    for index, bond in enumerate(bonds):
        if bond.kind not in {BondKind.COVALENT, BondKind.UNKNOWN}:
            continue
        if bond.atom1_id == bond.atom2_id:
            continue
        adjacency.setdefault(bond.atom1_id, []).append((bond.atom2_id, index))
        adjacency.setdefault(bond.atom2_id, []).append((bond.atom1_id, index))
    # Saturation uses the full coordination number, including hydrogen leaves
    # that the ring search is allowed to ignore.
    degree = {atom_id: len(neighbors) for atom_id, neighbors in adjacency.items()}
    # Degree-1 atoms cannot lie on a ring. Stripping them keeps chain alkanes
    # from consuming the ring-search budget before a distant ring is seen.
    ring_graph = _without_leaves(adjacency)
    minimum = int(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_MIN_RING_SIZE"])
    maximum = int(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_MAX_RING_SIZE"])
    limit = int(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_RING_SEARCH_LIMIT"])
    rings = _chordless_rings(ring_graph, minimum, maximum, limit)
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    selected: set[int] = set()
    kept = 0
    for atom_ids, bond_indices in rings:
        if any(_locked(bonds[index]) for index in bond_indices):
            continue
        electrons = [
            _pi_electrons(atom_by_id[atom_id], degree.get(atom_id, 0))
            for atom_id in atom_ids
        ]
        if any(value is None for value in electrons):
            continue
        total = sum(electrons)
        modulus = int(CHEMISTRY_PERCEPTION_CONFIG["HUCKEL_ELECTRON_MODULUS"])
        residue = int(CHEMISTRY_PERCEPTION_CONFIG["HUCKEL_ELECTRON_RESIDUE"])
        if total % modulus != residue:
            continue
        selected.update(bond_indices)
        kept += 1
    return selected, kept


def _pi_electrons(atom: ChemicalAtom, degree: int) -> int | None:
    """Return the Hückel contribution, or None when the atom is saturated.

    A carbon with four sigma neighbors (drawn atoms or a stored hydrogen
    count) has no p orbital, so a ring that contains it is not aromatic.
    Unknown hydrogen counts stay eligible only while the drawn degree is at
    most three; that is the benzene drawing, not cyclohexane. A positive
    carbon contributes an empty p orbital (tropylium). A negative carbon
    contributes a lone pair (cyclopentadienyl).
    """
    if atom.element not in _PI_ELEMENTS:
        return None
    explicit = atom.explicit_hydrogens
    stored_h = int(explicit or 0) + int(atom.implicit_hydrogens or 0)
    if degree > 3:
        return None
    if atom.element == "C":
        if stored_h and degree + stored_h >= 4:
            return None
        if atom.formal_charge is not None and int(atom.formal_charge) > 0:
            return 0
        if atom.formal_charge is not None and int(atom.formal_charge) < 0:
            return 2
        return 1
    if atom.element == "N":
        # Pyridinium N–H+ keeps one pi electron: protonation uses the lone
        # pair, not the p contribution. Neutral or anionic N–H (pyrrole)
        # donates that lone pair. Bare pyridine nitrogen donates one.
        if atom.formal_charge is not None and int(atom.formal_charge) > 0:
            return 1
        if degree >= 3 or stored_h > 0:
            return 2
        return 1
    if atom.element in {"O", "S"}:
        if degree > 2:
            return None
        return 2
    return None


def _implicit_hydrogens(atoms, bonds) -> dict[str, int]:
    """One implicit H on a divalent aromatic carbon with no hydrogen neighbor."""
    neighbors: dict[str, list[str]] = {atom.atom_id: [] for atom in atoms}
    aromatic_atoms: set[str] = set()
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    for bond in bonds:
        if bond.kind not in {BondKind.COVALENT, BondKind.UNKNOWN}:
            continue
        neighbors.setdefault(bond.atom1_id, []).append(bond.atom2_id)
        neighbors.setdefault(bond.atom2_id, []).append(bond.atom1_id)
        if bond.aromatic:
            aromatic_atoms.add(bond.atom1_id)
            aromatic_atoms.add(bond.atom2_id)
    implied: dict[str, int] = {}
    for atom_id in aromatic_atoms:
        atom = atom_by_id.get(atom_id)
        if (
            atom is None
            or atom.element != "C"
            or atom.explicit_hydrogens is not None
            or atom.implicit_hydrogens is not None
        ):
            continue
        attached = neighbors.get(atom_id, [])
        if len(attached) != 2:
            continue
        if any(
            atom_by_id[other].element in _HYDROGEN
            for other in attached
            if other in atom_by_id
        ):
            continue
        implied[atom_id] = 1
    return implied


def _without_leaves(
    adjacency: dict[str, list[tuple[str, int]]],
) -> dict[str, list[tuple[str, int]]]:
    pending = {node: list(neighbors) for node, neighbors in adjacency.items()}
    while True:
        leaves = [node for node, neighbors in pending.items() if len(neighbors) < 2]
        if not leaves:
            break
        for node in leaves:
            for neighbor, bond_index in pending.pop(node):
                if neighbor not in pending:
                    continue
                pending[neighbor] = [
                    item for item in pending[neighbor] if item != (node, bond_index)
                ]
    return pending


def _chordless_rings(adjacency, minimum: int, maximum: int, limit: int):
    rings = []
    seen: set[frozenset[int]] = set()
    expanded = 0
    for start in sorted(adjacency):
        # (node, atoms, bond indices)
        stack = [(start, [start], [])]
        while stack:
            node, atoms, bond_ids = stack.pop()
            expanded += 1
            if expanded > limit:
                return rings
            for neighbor, bond_index in adjacency[node]:
                if bond_ids and bond_index == bond_ids[-1]:
                    continue
                if neighbor == start and minimum <= len(atoms) <= maximum:
                    ring_bonds = bond_ids + [bond_index]
                    if len(ring_bonds) != len(atoms):
                        continue
                    key = frozenset(ring_bonds)
                    if key in seen or not _chordless(atoms, adjacency, key):
                        continue
                    seen.add(key)
                    rings.append((tuple(atoms), tuple(ring_bonds)))
                    continue
                if neighbor in atoms or len(atoms) >= maximum:
                    continue
                stack.append((neighbor, atoms + [neighbor], bond_ids + [bond_index]))
    return rings


def _chordless(atoms: list[str], adjacency, ring_bonds: frozenset[int]) -> bool:
    members = set(atoms)
    for atom in atoms:
        for _neighbor, bond_index in adjacency[atom]:
            if bond_index in ring_bonds:
                continue
            if _neighbor in members:
                return False
    return True


__all__ = ["AromaticityAssignment", "apply_aromaticity"]
