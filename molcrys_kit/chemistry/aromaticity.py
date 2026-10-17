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

import numpy as np

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
        _unspecified_saturated_ring_atoms(
            atoms,
            bonds,
            embedding,
            cell,
            aromatic_bond_indices=aromatic_bonds,
        ),
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
        fused_circuit = (
            len(atom_ids) == 10
            and _fused_five_seven_circuit(
                atom_ids, ring_graph, frozenset(bond_indices)
            )
        )
        # Only azulene's fused five/seven perimeter is allowed above the
        # ordinary seven-membered aromatic ring limit.  A chordless C8--C10
        # cycle has the right Hückel residue on paper, but it is not the
        # fused circuit this graph authority is meant to recognize.
        if len(atom_ids) > 7 and not fused_circuit:
            continue
        fused_chords = (
            _fused_chord_indices(atom_ids, ring_graph, frozenset(bond_indices))
            if fused_circuit
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
    """Refuse a geometry-confirmed saturated ring with omitted H data.

    A ring graph with all hydrogen fields absent is otherwise compatible with
    both an aromatic and a saturated interpretation.  Coordinates therefore
    go through the ring-level planarity, sp2-angle, and bond-metric authority;
    an ambiguous geometry is deliberately left unresolved.  Without
    coordinates the historical graph-only behavior is retained and the caller
    can expose the unresolved hydrogen metadata as provisional.
    """
    if embedding is None:
        return True
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    ring_atoms = [atom_by_id[atom_id] for atom_id in atom_ids]
    if not ring_atoms:
        return True
    if any(
        atom.explicit_hydrogens is not None or atom.implicit_hydrogens is not None
        for atom in ring_atoms
    ):
        return True
    # A single edge is not a reliable aromaticity authority.  Thermal motion,
    # heteroatom substitution, and the two N--O bond populations all overlap
    # at the old pair-specific cutoffs.  Evaluate the complete ring instead:
    # planar/sp2 geometry and a ring-wide bond metric must agree before we
    # infer aromaticity.  ``ambiguous`` intentionally abstains.
    return _ring_geometry_state(
        atom_ids, bond_indices, atom_by_id, bonds, embedding, cell
    ) == "aromatic"


def _unspecified_saturated_ring_atoms(
    atoms,
    bonds,
    embedding,
    cell,
    *,
    aromatic_bond_indices=frozenset(),
):
    """Return atoms in a geometry-confirmed ring with unknown H inventory."""
    if embedding is None:
        return ()
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    aromatic_atoms = {
        atom_id
        for index in aromatic_bond_indices
        for atom_id in (bonds[index].atom1_id, bonds[index].atom2_id)
    }
    explicit_h_edge_atoms: set[str] = set()
    adjacency: dict[str, list[tuple[str, int, ChemicalBond]]] = {
        atom.atom_id: [] for atom in atoms
    }
    for index, bond in enumerate(bonds):
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
        adjacency.setdefault(bond.atom1_id, []).append((bond.atom2_id, index, bond))
        adjacency.setdefault(bond.atom2_id, []).append((bond.atom1_id, index, bond))
    if not adjacency:
        return ()

    # A branch between two rings survives leaf stripping even though it is not
    # part of either cycle.  Remove graph bridges first so one long connecting
    # edge cannot make an otherwise unambiguous ring look saturated.
    ring_indices = _non_bridge_indices(adjacency)
    ring_adjacency: dict[str, list[tuple[str, int, ChemicalBond]]] = {
        atom_id: [item for item in neighbors if item[1] in ring_indices]
        for atom_id, neighbors in adjacency.items()
    }
    minimum = int(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_MIN_RING_SIZE"])
    maximum = int(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_MAX_RING_SIZE"])
    limit = int(CHEMISTRY_PERCEPTION_CONFIG["AROMATIC_RING_SEARCH_LIMIT"])
    ring_graph = {
        atom_id: [
            (
                other,
                index,
                # Hydrogen-boundary discovery is a local saturation check.
                # Unlike aromatic ring authority, it must still report an
                # H-unspecified quotient cycle when a periodic image shift
                # prevents treating that cycle as a finite aromatic ring.
                (0, 0, 0),
            )
            for other, index, _bond in neighbors
        ]
        for atom_id, neighbors in ring_adjacency.items()
    }
    # Keep the boundary local to each chordless cycle.  A fused aromatic ring
    # can share a 2-edge-connected component with a saturated heterocycle, so
    # a component-wide carbon test either over-abstains or misses the boundary.
    rings = _chordless_rings(ring_graph, minimum, maximum, limit)
    uncertain: set[str] = set()
    for atom_ids, bond_indices in rings:
        ring_atoms = [atom_by_id.get(atom_id) for atom_id in atom_ids]
        if any(atom is None for atom in ring_atoms):
            continue
        if any(
            atom.explicit_hydrogens is not None or atom.implicit_hydrogens is not None
            for atom in ring_atoms
        ):
            continue
        state = _ring_geometry_state(
            atom_ids, bond_indices, atom_by_id, bonds, embedding, cell
        )
        if state == "aromatic":
            continue
        uncertain.update(
            atom_id
            for atom_id in atom_ids
            if atom_id not in aromatic_atoms
            if atom_id not in explicit_h_edge_atoms
            and atom_by_id[atom_id].explicit_hydrogens is None
            and atom_by_id[atom_id].implicit_hydrogens is None
        )
    # Only report a boundary when an actual ring has at least three unresolved
    # atoms.  The threshold prevents a single uncertain substituent from
    # turning an otherwise solvable entity into a local abstention.
    return tuple(sorted(uncertain)) if len(uncertain) >= 3 else ()


def _ring_geometry_state(
    atom_ids, bond_indices, atom_by_id, bonds, embedding, cell
) -> str:
    """Classify an H-unspecified ring from whole-ring geometry.

    Returns ``"aromatic"``, ``"saturated"`` or ``"ambiguous"``.  The
    ambiguous band is explicit: a ring whose mean bond metric lies between the
    aromatic and saturated limits is left unresolved even when one edge happens
    to fall on one side of an element-pair reference.  This is intentionally a
    conservative local decision used only when all ring hydrogen metadata are
    absent; explicit H counts and source bond assertions retain precedence.
    """
    metrics = _ring_geometry_metrics(
        atom_ids, bond_indices, atom_by_id, bonds, embedding, cell
    )
    if metrics is None:
        # Preserve the historical graph-only behaviour when an embedding is
        # incomplete.  The caller already ensures ``embedding`` is present.
        return "aromatic"
    planarity, angle_deviation, bond_metric = metrics
    config = CHEMISTRY_PERCEPTION_CONFIG
    planar = planarity <= float(config["AROMATIC_RING_PLANARITY_MAX"])
    sp2 = angle_deviation <= float(
        config["AROMATIC_RING_SP2_ANGLE_DEVIATION_MAX_DEG"]
    )
    aromatic_limit = float(config["AROMATIC_RING_BOND_METRIC_AROMATIC_MAX"])
    saturated_limit = float(config["AROMATIC_RING_BOND_METRIC_SATURATED_MIN"])
    if planar and sp2 and bond_metric <= aromatic_limit:
        return "aromatic"
    if (not planar or not sp2) and bond_metric >= aromatic_limit:
        return "saturated"
    if bond_metric >= saturated_limit:
        return "saturated"
    return "ambiguous"


def _ring_geometry_metrics(
    atom_ids, bond_indices, atom_by_id, bonds, embedding, cell
):
    """Return normalized planarity, sp2-angle deviation, and bond metric."""
    if embedding is None or len(atom_ids) < 3:
        return None
    try:
        # Walk the cycle in the order returned by ``_chordless_rings``.  Using
        # bond vectors (rather than only atom coordinates) preserves periodic
        # image shifts in a zero-winding cycle.
        points = [
            np.asarray(embedding.position(atom_ids[0]), dtype=float)
        ]
        edge_lengths: list[float] = []
        for position, index in enumerate(bond_indices[:-1]):
            left_id = atom_ids[position]
            right_id = atom_ids[position + 1]
            bond = bonds[index]
            vector = np.asarray(_embedding_vector(embedding, bond, cell), dtype=float)
            if bond.atom1_id == left_id and bond.atom2_id == right_id:
                oriented = vector
            elif bond.atom1_id == right_id and bond.atom2_id == left_id:
                oriented = -vector
            else:
                # A malformed ring record should not make chemistry
                # indeterminate; fall back to the direct embedding vector.
                oriented = np.asarray(
                    embedding.vector(left_id, right_id), dtype=float
                )
            length = float(np.linalg.norm(oriented))
            if not math.isfinite(length) or length <= 1e-8:
                return None
            edge_lengths.append(length)
            points.append(points[-1] + oriented)
        # The closing edge is needed for the metric and angle calculation but
        # must not add a duplicate point to the plane fit.
        last_index = bond_indices[-1]
        last_bond = bonds[last_index]
        left_id = atom_ids[-1]
        right_id = atom_ids[0]
        vector = np.asarray(_embedding_vector(embedding, last_bond, cell), dtype=float)
        if last_bond.atom1_id == left_id and last_bond.atom2_id == right_id:
            oriented = vector
        elif last_bond.atom1_id == right_id and last_bond.atom2_id == left_id:
            oriented = -vector
        else:
            oriented = np.asarray(embedding.vector(left_id, right_id), dtype=float)
        closing_length = float(np.linalg.norm(oriented))
        if not math.isfinite(closing_length) or closing_length <= 1e-8:
            return None
        edge_lengths.append(closing_length)
        points = np.asarray(points, dtype=float)
        if len(points) != len(atom_ids):
            return None
        center = points.mean(axis=0)
        centered = points - center
        covariance = centered.T @ centered / float(len(points))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        deviations = centered @ normal
        mean_edge = float(np.mean(edge_lengths))
        planarity = float(np.sqrt(np.mean(deviations * deviations)) / mean_edge)

        angles = []
        for index, point in enumerate(points):
            previous = points[(index - 1) % len(points)] - point
            following = points[(index + 1) % len(points)] - point
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
            if denominator <= 1e-12:
                return None
            cosine = float(np.dot(previous, following) / denominator)
            cosine = max(-1.0, min(1.0, cosine))
            angles.append(math.degrees(math.acos(cosine)))
        # Interior angle of a regular n-gon (the angle between the two
        # in-ring sigma vectors), which is the local sp2 reference for a
        # planar aromatic ring.
        ideal = 180.0 - 360.0 / float(len(points))
        angle_deviation = float(np.mean([abs(value - ideal) for value in angles]))

        references = CHEMISTRY_PERCEPTION_CONFIG.get(
            "AROMATIC_RING_SINGLE_BOND_LENGTHS", {}
        )
        ratios = []
        for index, length in enumerate(edge_lengths):
            bond = bonds[bond_indices[index]]
            left = atom_by_id.get(bond.atom1_id)
            right = atom_by_id.get(bond.atom2_id)
            if left is None or right is None:
                continue
            pair = tuple(sorted((left.element, right.element)))
            reference = references.get(pair)
            if reference is None or float(reference) <= 0:
                continue
            ratios.append(length / float(reference))
        if not ratios:
            return None
        bond_metric = float(np.mean(ratios))
        if not all(math.isfinite(value) for value in (planarity, angle_deviation, bond_metric)):
            return None
        return planarity, angle_deviation, bond_metric
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, FloatingPointError):
        return None


def _embedding_vector(embedding, bond, cell=None):
    """Return an edge vector, including an image shift when a cell is given."""
    try:
        return tuple(
            float(value)
            for value in embedding.vector(
                bond.atom1_id, bond.atom2_id, bond.atom2_image_shift
            )
        )
    except (AttributeError, KeyError, ValueError, TypeError):
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
        return tuple(
            float(right_value) - float(left_value)
            for left_value, right_value in zip(left, right)
        )


def _non_bridge_indices(adjacency):
    """Return edge indices that participate in an undirected cycle."""
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    bridges: set[int] = set()
    tick = 0

    def visit(node, parent_edge=None):
        nonlocal tick
        tick += 1
        discovery[node] = low[node] = tick
        for neighbor, edge_index, _bond in adjacency.get(node, ()):
            if edge_index == parent_edge:
                continue
            if neighbor not in discovery:
                visit(neighbor, edge_index)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(edge_index)
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(adjacency):
        if node not in discovery:
            visit(node)
    return {
        edge_index
        for neighbors in adjacency.values()
        for _neighbor, edge_index, _bond in neighbors
        if edge_index not in bridges
    }


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
