"""Normalize representations that are the same chemical graph.

Normalization covers a Kekulé six-membered ring whose bond pattern is the
alternating single/double form of a 6π aromatic ring, and the standard
Kekulé forms of furan, thiophene, and pyrrole. Quinones, cyclohexadienes,
and charged or radical rings are left unchanged. Atom identity, isotopes,
charges, radical counts, and specified stereo are not rewritten.
"""

from __future__ import annotations

from dataclasses import replace

from .models import FiniteChemicalEntity


def _heavy_adjacency(entity: FiniteChemicalEntity):
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    adjacency = {atom_id: [] for atom_id in atoms}
    for bond in entity.bonds:
        adjacency[bond.atom1_id].append((bond.atom2_id, bond))
        adjacency[bond.atom2_id].append((bond.atom1_id, bond))
    heavy = {atom_id for atom_id, atom in atoms.items() if atom.element != "H"}
    return atoms, {
        atom_id: [(neighbor, bond) for neighbor, bond in adjacency[atom_id] if neighbor in heavy]
        for atom_id in heavy
    }


def _cycles(graph, size: int) -> list[tuple[str, ...]]:
    cycles = []
    seen = set()

    def walk(start, current, path):
        if len(path) == size:
            if any(neighbor == start for neighbor, _ in graph[current]):
                ring = path
                rotations = []
                for direction in (ring, tuple(reversed(ring))):
                    rotations.extend(direction[index:] + direction[:index] for index in range(size))
                key = min(rotations)
                if key not in seen:
                    seen.add(key)
                    cycles.append(key)
            return
        for neighbor, _ in graph[current]:
            if neighbor not in path and neighbor > start:
                walk(start, neighbor, (*path, neighbor))

    for start in sorted(graph):
        walk(start, start, (start,))
    return cycles


def _six_cycles(graph) -> list[tuple[str, ...]]:
    return _cycles(graph, 6)


def _alternating_kekule(orders: tuple[float, ...]) -> bool:
    if len(orders) != 6:
        return False
    if any(abs(order - 1.0) > 1e-8 and abs(order - 2.0) > 1e-8 for order in orders):
        return False
    return all(abs(orders[index] - orders[(index + 2) % 6]) <= 1e-8 for index in range(6)) and abs(
        orders[0] - orders[1]
    ) > 1e-8


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-8


def _five_heteroarene_edges(atoms, cycle, graph) -> set[frozenset[str]] | None:
    """Standard Kekulé furan, thiophene, or pyrrole: heteroatom between two single bonds."""
    ring_atoms = [atoms[atom_id] for atom_id in cycle]
    if any((atom.formal_charge or 0) or atom.radical_electrons for atom in ring_atoms):
        return None
    elements = [atom.element for atom in ring_atoms]
    if elements.count("C") != 4:
        return None
    heteroatoms = [element for element in elements if element != "C"]
    if len(heteroatoms) != 1 or heteroatoms[0] not in {"O", "S", "N"}:
        return None
    start = next(index for index, atom in enumerate(ring_atoms) if atom.element != "C")
    ordered = cycle[start:] + cycle[:start]
    orders = []
    edges = []
    for index, atom_id in enumerate(ordered):
        nxt = ordered[(index + 1) % 5]
        bond = next(bond for neighbor, bond in graph[atom_id] if neighbor == nxt)
        if bond.aromatic or _close(float(bond.order or 0.0), 1.5):
            return None
        orders.append(float(bond.order or 0.0))
        edges.append(frozenset((atom_id, nxt)))
    expected = (1.0, 2.0, 1.0, 2.0, 1.0)
    # The other direction from the heteroatom starts with the closing bond.
    reverse = (orders[4], orders[3], orders[2], orders[1], orders[0])
    if any(
        all(_close(order, target) for order, target in zip(candidate, expected))
        for candidate in (tuple(orders), reverse)
    ):
        return set(edges)
    return None


def normalize_kekule_aromatic(entity: FiniteChemicalEntity) -> FiniteChemicalEntity:
    """Rewrite proven Kekulé aromatic sextets as aromatic bonds."""
    if not isinstance(entity, FiniteChemicalEntity) or not entity.bonds:
        return entity
    atoms, graph = _heavy_adjacency(entity)
    aromatic_edges: set[frozenset[str]] = set()
    for cycle in _cycles(graph, 5):
        edges = _five_heteroarene_edges(atoms, cycle, graph)
        if edges:
            aromatic_edges.update(edges)
    for cycle in _six_cycles(graph):
        ring_atoms = [atoms[atom_id] for atom_id in cycle]
        if any((atom.formal_charge or 0) or atom.radical_electrons for atom in ring_atoms):
            continue
        elements = [atom.element for atom in ring_atoms]
        carbon_count = elements.count("C")
        nitrogen_count = elements.count("N")
        if not (
            carbon_count == 6
            or (carbon_count == 5 and nitrogen_count == 1)
        ):
            continue
        orders = []
        edges = []
        valid = True
        for index, atom_id in enumerate(cycle):
            nxt = cycle[(index + 1) % 6]
            bond = next(bond for neighbor, bond in graph[atom_id] if neighbor == nxt)
            if bond.aromatic or abs((bond.order or 0.0) - 1.5) <= 1e-8:
                valid = False
                break
            orders.append(float(bond.order or 0.0))
            edges.append(frozenset((atom_id, nxt)))
        if not valid or not _alternating_kekule(tuple(orders)):
            continue
        aromatic_edges.update(edges)
    if not aromatic_edges:
        return entity
    bonds = []
    changed = False
    for bond in entity.bonds:
        edge = frozenset((bond.atom1_id, bond.atom2_id))
        if edge in aromatic_edges and not bond.aromatic:
            changed = True
            bonds.append(
                replace(bond, order=1.5, aromatic=True)
            )
        else:
            bonds.append(bond)
    if not changed:
        return entity
    return replace(entity, bonds=tuple(bonds))


__all__ = ["normalize_kekule_aromatic"]
