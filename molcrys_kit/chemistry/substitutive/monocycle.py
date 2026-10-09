"""Monocyclic substitutive naming.

This module covers the parent forms needed by the current chemistry API:
cycloalkanes, isolated cycloalkenes, benzene/phenol (the legacy benzene
recognizer still handles carbon substituents), and simple heterocyclic
skeletal replacement names.
"""

from __future__ import annotations

from ..models import FiniteChemicalEntity
from .acyclic import alkane_stem


def _ring_functional_name(entity):
    """Name the small set of senior groups attached to a six-member parent."""
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    adjacency = _adjacency(entity)
    for cycle, graph in _cycles(entity):
        if len(cycle) != 6:
            continue
        edges = [next((bond for neighbor, bond in graph[a] if neighbor == cycle[(i + 1) % 6]), None) for i, a in enumerate(cycle)]
        if any(bond is None for bond in edges):
            continue
        aromatic = all(bond.aromatic or bond.order == 1.5 for bond in edges)
        saturated = all(bond.order == 1.0 for bond in edges)
        if not (aromatic or saturated) or any(atoms[a].element != "C" for a in cycle):
            continue
        ring = set(cycle)
        outside = [(a, n, bond) for a in cycle for n, bond in adjacency[a] if n not in ring and atoms[n].element != "H"]
        if aromatic:
            for ring_atom in cycle:
                for oxygen, link in adjacency[ring_atom]:
                    if atoms[oxygen].element != "O" or link.order != 1.0:
                        continue
                    carbonyls = [n for n, edge in adjacency[oxygen] if n != ring_atom and atoms[n].element == "C" and edge.order == 1.0]
                    for carbonyl in carbonyls:
                        carbonyl_edges = adjacency[carbonyl]
                        if sum(atoms[n].element == "O" and edge.order == 2.0 for n, edge in carbonyl_edges) != 1:
                            continue
                        if any(
                            n != oxygen
                            and atoms[n].element != "C"
                            and not (atoms[n].element == "O" and edge.order == 2.0)
                            for n, edge in carbonyl_edges
                        ):
                            continue
                        chain = _acyl_chain(entity, carbonyl)
                        if chain is None:
                            continue
                        stem = alkane_stem(len(chain))
                        if stem is None:
                            continue
                        # A phenyl ester recognizer must account for the entire
                        # aromatic substituent set before returning its name.
                        if any(n != oxygen for _, n, _ in outside):
                            continue
                        return (f"phenyl {stem}anoate", False, "Name the phenyl alcohol component and the acid-derived ester parent.")
        # A carbonyl group attached to benzene is the senior parent suffix.
        for ring_atom, carbonyl, link in outside:
            if atoms[carbonyl].element != "C" or link.order != 1.0:
                continue
            carbonyl_edges = adjacency[carbonyl]
            oxygens = [(n, b) for n, b in carbonyl_edges if atoms[n].element == "O"]
            if sum(b.order == 2.0 for _, b in oxygens) != 1:
                continue
            if aromatic and sum(b.order == 1.0 for _, b in oxygens) == 1:
                hydroxyl = next((n for n, b in oxygens if b.order == 1.0), None)
                if (
                    hydroxyl is not None
                    and _hydrogen_count(entity, hydroxyl) > 0
                    and len(outside) == 1
                    and outside[0][1] == carbonyl
                ):
                    return ("benzenecarboxylic acid", False, "Select benzenecarboxylic acid as the senior group.")
            if saturated:
                nitrogens = [n for n, b in carbonyl_edges if atoms[n].element == "N" and b.order == 1.0]
                if nitrogens:
                    return ("cyclohexanecarboxamide", False, "Use the ring carboxamide suffix.")
            # Phenyl ethanoate: the acid side is acetyl and the oxygen joins
            # the aromatic parent.
            if aromatic:
                ester_o = next((n for n, b in carbonyl_edges if atoms[n].element == "O" and b.order == 1.0 and n != ring_atom), None)
                if ester_o is not None and len(outside) == 1 and outside[0][1] == carbonyl:
                    alkyl = [n for n, b in adjacency[ester_o] if n != carbonyl and atoms[n].element == "C"]
                    if alkyl:
                        chain = _acyl_chain(entity, carbonyl)
                        if chain is not None:
                            stem = alkane_stem(len(chain))
                            if stem is not None:
                                return (f"phenyl {stem}anoate", False, "Name the phenyl alcohol component and the acid-derived ester parent.")
        if saturated:
            for ring_atom in cycle:
                for neighbor, bond in adjacency[ring_atom]:
                    if atoms[neighbor].element == "O" and bond.order == 2.0:
                        return ("cyclohexanone", False, "Apply the ketone suffix to the cyclohexane parent.")
    return None


def _acyl_chain(entity, carbonyl):
    """Return a fully accounted-for unbranched acyl carbon chain."""
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    adjacency = _adjacency(entity)
    chain = [carbonyl]
    previous = None
    current = carbonyl
    while True:
        candidates = [
            neighbor
            for neighbor, bond in adjacency[current]
            if neighbor != previous
            and atoms[neighbor].element == "C"
            and bond.order == 1.0
        ]
        if len(candidates) > 1:
            return None
        if not candidates:
            break
        previous, current = current, candidates[0]
        if current in chain:
            return None
        chain.append(current)
    chain_set = set(chain)
    for atom_id in chain:
        for neighbor, bond in adjacency[atom_id]:
            if atoms[neighbor].element == "H" or neighbor in chain_set:
                continue
            # The carbonyl oxygen(s) and ester oxygen are functional-group
            # attachments; the carbon chain itself must have no hidden branch.
            if atom_id == carbonyl and atoms[neighbor].element == "O":
                continue
            return None
    return tuple(chain)


def _hydrogen_count(entity, atom_id):
    atom = next(atom for atom in entity.atoms if atom.atom_id == atom_id)
    adjacency = _adjacency(entity)
    explicit = sum(next(candidate for candidate in entity.atoms if candidate.atom_id == n).element == "H" for n, _ in adjacency[atom_id])
    return explicit + (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)


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
        # Skeletal-replacement saturated parents cannot describe a ring
        # that contains ordinary C=C bonds.  Fail closed until the
        # unsaturated heterocycle grammar is implemented.
        if not all(b.order == 1.0 for b in edges):
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
    functional = _ring_functional_name(entity)
    if functional is not None:
        return functional
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
