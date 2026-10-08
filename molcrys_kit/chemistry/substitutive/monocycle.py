"""Monocyclic substitutive naming.

This module covers the parent forms needed by the current chemistry API:
cycloalkanes, isolated cycloalkenes, benzene/phenol (the legacy benzene
recognizer still handles carbon substituents), and simple heterocyclic
skeletal replacement names.
"""

from __future__ import annotations

from collections import Counter

from ..models import BondKind, FiniteChemicalEntity
from .acyclic import alkane_stem


def _adjacency(entity):
    values = {atom.atom_id: [] for atom in entity.atoms}
    for bond in entity.bonds:
        values[bond.atom1_id].append((bond.atom2_id, bond))
        values[bond.atom2_id].append((bond.atom1_id, bond))
    return values


def _cycles(entity):
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    adjacency = _adjacency(entity)
    heavy = {a for a, atom in atoms.items() if atom.element != "H"}
    if not heavy:
        return []
    graph = {a: [(n, b) for n, b in adjacency[a] if n in heavy] for a in heavy}
    if sum(len(v) for v in graph.values()) // 2 != len(heavy):
        return []
    found = set()

    def walk(start, current, path):
        if len(path) > len(heavy):
            return
        for neighbor, bond in graph[current]:
            if neighbor == start and len(path) >= 3:
                values = tuple(path)
                rotations = []
                for seq in (values, tuple(reversed(values))):
                    rotations.extend(seq[i:] + seq[:i] for i in range(len(seq)))
                found.add(min(rotations))
            elif neighbor not in path and neighbor >= start:
                walk(start, neighbor, (*path, neighbor))

    for start in sorted(heavy):
        walk(start, start, (start,))
    return [(cycle, graph) for cycle in sorted(found)]


def _result(name, preferred=True, *trace):
    return name, preferred, *trace


def _ring_name(cycle, graph, atoms):
    n = len(cycle)
    edges = []
    for index, atom_id in enumerate(cycle):
        next_id = cycle[(index + 1) % n]
        bond = next((b for neighbor, b in graph[atom_id] if neighbor == next_id), None)
        if bond is None:
            return None
        edges.append(bond)
    elements = [atoms[a].element for a in cycle]
    aromatic = all(b.aromatic or b.order == 1.5 for b in edges)
    if aromatic and n == 6 and all(element == "C" for element in elements):
        # Carbon-substituted benzene is deliberately left to the established
        # benzene/phenol recognizer, which also preserves its retained names.
        return "benzene", True
    if aromatic and n == 6 and any(element != "C" for element in elements):
        hetero = [(index, element) for index, element in enumerate(elements) if element != "C"]
        if all(element in {"N", "O", "S"} for _, element in hetero):
            # Skeleton replacement numbering: place the first heteroatom at 1,
            # then choose the direction with the lowest locant sequence.
            start = next(index for index, element in enumerate(elements) if element != "C")
            options = []
            for direction in (1, -1):
                ordered = tuple(cycle[(start + direction * i) % n] for i in range(n))
                prefixes = []
                for locant, atom_id in enumerate(ordered, 1):
                    element = atoms[atom_id].element
                    if element != "C":
                        prefixes.append((locant, {"N": "aza", "O": "oxa", "S": "thia"}[element]))
                options.append((tuple(loc for loc, _ in prefixes), tuple(prefix for _, prefix in prefixes), prefixes))
            prefixes = min(options, key=lambda item: (item[0], item[1]))[2]
            return "-".join(f"{loc}-{prefix}" for loc, prefix in prefixes) + "benzene", False
    if any(element not in {"C"} for element in elements):
        if any(element not in {"C", "N", "O", "S"} for element in elements):
            return None
        # Saturated skeletal replacement, with oxa/aza locants.  Prefer an
        # oxygen start when present (the conventional C1COCCN1 spelling is
        # therefore 1-oxa-4-azacyclohexane).
        starts = [i for i, e in enumerate(elements) if e == "O"] or [i for i, e in enumerate(elements) if e != "C"]
        options = []
        for start in starts:
            for direction in (1, -1):
                ordered = tuple(cycle[(start + direction * i) % n] for i in range(n))
                prefixes = [
                    (locant, {"N": "aza", "O": "oxa", "S": "thia"}[atoms[atom_id].element])
                    for locant, atom_id in enumerate(ordered, 1)
                    if atoms[atom_id].element != "C"
                ]
                options.append((tuple(loc for loc, _ in prefixes), tuple(prefix for _, prefix in prefixes), prefixes))
        prefixes = min(options, key=lambda item: (item[0], item[1]))[2]
        stem = alkane_stem(n)
        if stem is None:
            return None
        return "-".join(f"{loc}-{prefix}" for loc, prefix in prefixes) + f"cyclo{stem}ane", False
    # All-carbon monocycle.  Aromatic and substituted forms are delegated to
    # the legacy benzene family; isolated unsaturation gets the -ene suffix.
    if aromatic:
        return "benzene", True
    if any(b.order == 2.0 for b in edges):
        if sum(b.order == 2.0 for b in edges) == 1 and all(b.order in {1.0, 2.0} for b in edges):
            stem = alkane_stem(n)
            return (f"cyclo{stem}ene", True) if stem else None
        return None
    if all(b.order == 1.0 for b in edges):
        stem = alkane_stem(n)
        return (f"cyclo{stem}ane", True) if stem else None
    return None


def name_monocycle(entity: FiniteChemicalEntity):
    """Return a naming tuple for an isolated monocycle, or ``None``."""
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    for cycle, graph in _cycles(entity):
        if len(cycle) != len([atom for atom in entity.atoms if atom.element != "H"]):
            continue
        value = _ring_name(cycle, graph, atoms)
        if value is None:
            continue
        name, preferred = value
        if name == "benzene":
            # If there are outside heavy substituents, let naming.py's mature
            # benzene-family implementation choose retained phenol numbering.
            adjacency = _adjacency(entity)
            if any(
                neighbor not in cycle and atoms[neighbor].element != "H"
                for atom_id in cycle
                for neighbor, _ in adjacency[atom_id]
            ):
                continue
        return _result(name, preferred, "Identify the isolated monocyclic parent.", "Apply cycloalkane, cycloalkene, or skeletal-replacement nomenclature.")
    return None


__all__ = ["name_monocycle"]
