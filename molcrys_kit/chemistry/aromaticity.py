"""Graph aromaticity authority.

A ring is aromatic when it is a chordless 5-, 6-, or 7-membered cycle of
C/N/O/S (or the fused 10-membered 5/7 perimeter) and its pi-electron count is
4n+2. Geometry can rank Kekulé forms.
It does not grant or revoke this flag. Thiophene sulfur and furan/pyrrole
heteroatoms contribute a lone pair; pyridine nitrogen, pyridinium N–H+,
and carbon contribute one pi electron.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    uncertain_hydrogen_atoms: tuple[str, ...] = ()


def apply_aromaticity(
    atoms: tuple[ChemicalAtom, ...] | list[ChemicalAtom],
    bonds: tuple[ChemicalBond, ...] | list[ChemicalBond],
    *,
    embedding=None,
    cell=None,
) -> AromaticityAssignment:
    """Return bonds with inferred aromatic flags and solver-local implicit H."""
    atoms = tuple(atoms)
    bonds = tuple(bonds)
    aromatic_bonds, ring_count = _aromatic_bond_indices(
        atoms, bonds, embedding=embedding, cell=cell
    )
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
        _unspecified_saturated_ring_atoms(atoms, bonds, embedding, cell),
    )


def _locked(bond: ChemicalBond) -> bool:
    """Only an explicit user non-aromatic assertion blocks aromaticity.

    A confirmed Kekulé single/double order is an order authority, not an
    aromaticity authority.  The latter is represented by ``aromatic:false``
    (or the legacy human-readable ``not aromatic`` detail).
    """
    if bond.aromatic:
        return False
    for evidence in bond.evidence:
        if evidence.source is not EvidenceSource.USER_CONFIRMED:
            continue
        detail = (evidence.detail or "").strip().lower().replace(" ", "")
        if "aromatic:false" in detail or "notaromatic" in detail:
            return True
    return False


def _aromatic_bond_indices(atoms, bonds, *, embedding=None, cell=None) -> tuple[set[int], int]:
    # Each adjacency entry carries the image shift incurred when traversing
    # the edge in that direction.  This lets ring discovery reject quotient
    # cycles with non-zero winding through a periodic cell.
    adjacency: dict[str, list[tuple[str, int, tuple[int, int, int]]]] = {
        atom.atom_id: [] for atom in atoms
    }
    for index, bond in enumerate(bonds):
        if bond.kind not in {BondKind.COVALENT, BondKind.UNKNOWN}:
            continue
        if bond.atom1_id == bond.atom2_id:
            continue
        shift = tuple(int(value) for value in bond.atom2_image_shift)
        adjacency.setdefault(bond.atom1_id, []).append((bond.atom2_id, index, shift))
        adjacency.setdefault(bond.atom2_id, []).append(
            (bond.atom1_id, index, tuple(-value for value in shift))
        )
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
        if not _geometry_supports_unspecified_carbon_ring(
            atom_ids, bond_indices, atoms, bonds, embedding, cell
        ):
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
        fused_chords = (
            _fused_chord_indices(atom_ids, ring_graph, frozenset(bond_indices))
            if len(atom_ids) == 10
            and _fused_five_seven_circuit(atom_ids, ring_graph, frozenset(bond_indices))
            else set()
        )
        if any(_locked(bonds[index]) for index in fused_chords):
            continue
        selected.update(bond_indices)
        if fused_chords:
            # The shared fusion edge is part of the conjugated azulene
            # circuit even though it is a chord of the outer perimeter.
            selected.update(
                _fused_chord_indices(atom_ids, ring_graph, frozenset(bond_indices))
            )
        kept += 1
    return selected, kept


def _geometry_supports_unspecified_carbon_ring(
    atom_ids, bond_indices, atoms, bonds, embedding, cell
) -> bool:
    """Refuse a geometry-confirmed saturated all-carbon ring.

    A six-carbon graph with all hydrogen fields absent is otherwise compatible
    with both benzene and cyclohexane.  When coordinates are available, long
    C--C edges provide decisive evidence against aromaticity.  Without
    coordinates the historical graph-only behavior is retained and the caller
    can expose the unresolved hydrogen metadata as provisional.
    """
    if embedding is None:
        return True
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    ring_atoms = [atom_by_id[atom_id] for atom_id in atom_ids]
    if not ring_atoms or not all(atom.element == "C" for atom in ring_atoms):
        return True
    if any(
        atom.explicit_hydrogens is not None or atom.implicit_hydrogens is not None
        for atom in ring_atoms
    ):
        return True
    maximum = float(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_MAX_UNSPECIFIED_CC_BOND_LENGTH"])
    for index in bond_indices:
        bond = bonds[index]
        try:
            vector = embedding.vector(bond.atom1_id, bond.atom2_id, bond.atom2_image_shift)
        except (AttributeError, KeyError, ValueError, TypeError):
            try:
                left = embedding.position(bond.atom1_id)
                right = embedding.position(bond.atom2_id)
                if cell is not None and any(bond.atom2_image_shift):
                    right = tuple(
                        float(right[axis])
                        + sum(
                            int(bond.atom2_image_shift[row]) * float(cell[row][axis])
                            for row in range(3)
                        )
                        for axis in range(3)
                    )
                vector = tuple(float(r) - float(l) for l, r in zip(left, right))
            except (AttributeError, KeyError, ValueError, TypeError):
                return True
        if math.sqrt(sum(float(value) ** 2 for value in vector)) > maximum:
            return False
    return True


def _unspecified_saturated_ring_atoms(atoms, bonds, embedding, cell):
    """Return atoms in a geometry-confirmed ring with unknown H inventory."""
    if embedding is None:
        return ()
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    explicit_h_edge_atoms: set[str] = set()
    adjacency: dict[str, list[tuple[str, ChemicalBond]]] = {
        atom.atom_id: [] for atom in atoms
    }
    for bond in bonds:
        if bond.kind not in {BondKind.COVALENT, BondKind.UNKNOWN}:
            continue
        if bond.atom1_id == bond.atom2_id:
            continue
        left = atom_by_id.get(bond.atom1_id)
        right = atom_by_id.get(bond.atom2_id)
        if left is not None and left.element in _HYDROGEN:
            explicit_h_edge_atoms.add(bond.atom2_id)
        if right is not None and right.element in _HYDROGEN:
            explicit_h_edge_atoms.add(bond.atom1_id)
        adjacency.setdefault(bond.atom1_id, []).append((bond.atom2_id, bond))
        adjacency.setdefault(bond.atom2_id, []).append((bond.atom1_id, bond))
    pending = {atom_id for atom_id, neighbors in adjacency.items() if len(neighbors) < 2}
    while pending:
        atom_id = pending.pop()
        for neighbor, _bond in adjacency.pop(atom_id, []):
            adjacency[neighbor] = [
                item for item in adjacency.get(neighbor, ()) if item[0] != atom_id
            ]
            if len(adjacency[neighbor]) < 2:
                pending.add(neighbor)
    if not adjacency:
        return ()
    maximum = float(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_MAX_UNSPECIFIED_CC_BOND_LENGTH"])
    uncertain = []
    for atom_id, neighbors in adjacency.items():
        atom = atom_by_id[atom_id]
        if (
            atom.element != "C"
            or atom.explicit_hydrogens is not None
            or atom.implicit_hydrogens is not None
            or atom_id in explicit_h_edge_atoms
        ):
            continue
        if not all(
            atom_by_id.get(other) is not None
            and atom_by_id[other].element == "C"
            for other, _ in neighbors
        ):
            continue
        long_edge = False
        for _other, bond in neighbors:
            try:
                vector = embedding.vector(bond.atom1_id, bond.atom2_id, bond.atom2_image_shift)
                if bond.atom1_id != atom_id:
                    vector = tuple(-float(value) for value in vector)
                length = math.sqrt(sum(float(value) ** 2 for value in vector))
            except (AttributeError, KeyError, ValueError, TypeError):
                continue
            if length > maximum:
                long_edge = True
                break
        if long_edge:
            uncertain.append(atom_id)
    # Only report a boundary when an actual ring survived leaf stripping.
    return tuple(sorted(uncertain)) if len(uncertain) >= 3 else ()


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
    adjacency: dict[str, list[tuple[str, int, tuple[int, int, int]]]],
) -> dict[str, list[tuple[str, int, tuple[int, int, int]]]]:
    pending = {node: list(neighbors) for node, neighbors in adjacency.items()}
    while True:
        leaves = [node for node, neighbors in pending.items() if len(neighbors) < 2]
        if not leaves:
            break
        for node in leaves:
            for neighbor, bond_index, shift in pending.pop(node):
                if neighbor not in pending:
                    continue
                pending[neighbor] = [
                    item for item in pending[neighbor] if item[:2] != (node, bond_index)
                ]
    return pending


def _chordless_rings(adjacency, minimum: int, maximum: int, limit: int):
    rings = []
    seen: set[frozenset[int]] = set()
    expanded = 0
    for start in sorted(adjacency):
        # (node, atoms, bond indices, cumulative image shift)
        stack = [(start, [start], [], (0, 0, 0))]
        while stack:
            node, atoms, bond_ids, winding = stack.pop()
            expanded += 1
            if expanded > limit:
                return rings
            for neighbor, bond_index, shift in adjacency[node]:
                if bond_ids and bond_index == bond_ids[-1]:
                    continue
                if neighbor == start and minimum <= len(atoms) <= maximum:
                    ring_bonds = bond_ids + [bond_index]
                    if len(ring_bonds) != len(atoms):
                        continue
                    total_winding = tuple(
                        winding[axis] + shift[axis] for axis in range(3)
                    )
                    if total_winding != (0, 0, 0):
                        continue
                    key = frozenset(ring_bonds)
                    if key in seen or (
                        not _chordless(atoms, adjacency, key)
                        and not _fused_five_seven_circuit(atoms, adjacency, key)
                    ):
                        continue
                    seen.add(key)
                    rings.append((tuple(atoms), tuple(ring_bonds)))
                    continue
                if neighbor in atoms or len(atoms) >= maximum:
                    continue
                next_winding = tuple(
                    winding[axis] + shift[axis] for axis in range(3)
                )
                stack.append(
                    (neighbor, atoms + [neighbor], bond_ids + [bond_index], next_winding)
                )
    return rings


def _chordless(atoms: list[str], adjacency, ring_bonds: frozenset[int]) -> bool:
    members = set(atoms)
    for atom in atoms:
        for _neighbor, bond_index, _shift in adjacency[atom]:
            if bond_index in ring_bonds:
                continue
            if _neighbor in members:
                return False
    return True


def _fused_five_seven_circuit(atoms, adjacency, ring_bonds: frozenset[int]) -> bool:
    """Recognize azulene's ten-membered perimeter around one shared edge."""
    if len(atoms) != 10:
        return False
    members = set(atoms)
    chords = _fused_chord_indices(atoms, adjacency, ring_bonds)
    if len(chords) != 1:
        return False
    chord = next(iter(chords))
    endpoints = []
    for position, atom in enumerate(atoms):
        if any(
            neighbor in members and bond_index == chord
            for neighbor, bond_index, _shift in adjacency[atom]
        ):
            endpoints.append(position)
    if len(endpoints) != 2:
        return False
    distance = abs(endpoints[0] - endpoints[1])
    path_lengths = {distance, len(atoms) - distance}
    return path_lengths == {4, 6}


def _fused_chord_indices(atoms, adjacency, ring_bonds: frozenset[int]) -> set[int]:
    members = set(atoms)
    return {
        bond_index
        for atom in atoms
        for neighbor, bond_index, _shift in adjacency[atom]
        if neighbor in members and bond_index not in ring_bonds
    }


__all__ = ["AromaticityAssignment", "apply_aromaticity"]
