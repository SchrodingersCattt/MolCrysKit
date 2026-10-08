"""Retained parent names and their reversible parsers.

The naming dispatcher keeps these established parent forms in the substitutive
namespace while the conversion helpers use the same constructors for inverse
round trips.
"""

from __future__ import annotations

from collections import Counter
import re

from ..models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
    DEFAULT_VALENCE,
)
from ..systematic_name import NamingParseError
from .acyclic import alkane_stem, HALOGEN_PREFIX

def _adjacency(entity):
    values = {atom.atom_id: [] for atom in entity.atoms}
    for bond in entity.bonds:
        values[bond.atom1_id].append((bond.atom2_id, bond))
        values[bond.atom2_id].append((bond.atom1_id, bond))
    return values

def _atom(entity, atom_id):
    return next(atom for atom in entity.atoms if atom.atom_id == atom_id)

def _element_counts(entity):
    counts = Counter(atom.element for atom in entity.atoms)
    counts["H"] += sum(
        (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)
        for atom in entity.atoms
        if atom.element != "H"
    )
    return +counts

def _single_heavy_center(entity, element):
    return [atom.element for atom in entity.atoms if atom.element != "H"] == [element]

def _hydrogen_count(entity, atom_id):
    atom = _atom(entity, atom_id)
    explicit_neighbors = sum(
        _atom(entity, neighbor).element == "H"
        for neighbor, _ in _adjacency(entity)[atom_id]
    )
    return explicit_neighbors + (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)

def _normal_valence(entity):
    adjacency = _adjacency(entity)
    for atom in entity.atoms:
        if atom.element not in DEFAULT_VALENCE:
            continue
        bond_sum = sum(bond.order or 0.0 for _, bond in adjacency[atom.atom_id])
        bond_sum += (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)
        if abs(bond_sum - DEFAULT_VALENCE[atom.element]) > 1e-8:
            return False
    return True

def _all_bonds(entity, orders):
    return all(
        bond.kind in {BondKind.COVALENT, BondKind.UNKNOWN}
        and bond.order in orders
        for bond in entity.bonds
    )

def _unbranched_carbon_chain(entity, carbon_ids):
    if not carbon_ids:
        return None
    carbon_set = set(carbon_ids)
    adjacency = _adjacency(entity)
    carbon_neighbors = {
        atom_id: [neighbor for neighbor, _ in adjacency[atom_id] if neighbor in carbon_set]
        for atom_id in carbon_ids
    }
    if any(len(values) > 2 for values in carbon_neighbors.values()):
        return None
    if len(carbon_ids) == 1:
        return tuple(carbon_ids)
    endpoints = [atom_id for atom_id, values in carbon_neighbors.items() if len(values) == 1]
    if len(endpoints) != 2:
        return None
    chain = []
    previous = None
    current = min(endpoints)
    while current is not None:
        chain.append(current)
        candidates = [value for value in carbon_neighbors[current] if value != previous]
        previous, current = current, (candidates[0] if candidates else None)
    return tuple(chain) if len(chain) == len(carbon_ids) else None

def _six_carbon_ring(entity):
    adjacency = _adjacency(entity)
    carbons = {atom.atom_id for atom in entity.atoms if atom.element == "C"}
    cycles = set()

    def walk(start, current, path):
        if len(path) == 6:
            if any(neighbor == start for neighbor, _ in adjacency[current]):
                cycle = tuple(path)
                rotations = []
                for values in (cycle, tuple(reversed(cycle))):
                    rotations.extend(values[index:] + values[:index] for index in range(6))
                cycles.add(min(rotations))
            return
        for neighbor, _ in adjacency[current]:
            if neighbor in carbons and neighbor not in path:
                walk(start, neighbor, (*path, neighbor))

    for start in carbons:
        walk(start, start, (start,))
    for cycle in sorted(cycles):
        ring_edges = []
        valid = True
        for index, atom_id in enumerate(cycle):
            next_id = cycle[(index + 1) % 6]
            bond = next(
                (bond for neighbor, bond in adjacency[atom_id] if neighbor == next_id),
                None,
            )
            if bond is None:
                valid = False
                break
            ring_edges.append(bond)
        if not valid:
            continue
        aromatic = all(bond.aromatic or bond.order == 1.5 for bond in ring_edges)
        alternating = sorted(bond.order for bond in ring_edges) == [1.0] * 3 + [2.0] * 3
        if aromatic or alternating:
            return cycle
    return None

def _acyclic_acyl_chain(entity, carbonyl_id, excluded):
    adjacency = _adjacency(entity)
    chain = [carbonyl_id]
    previous = None
    current = carbonyl_id
    while True:
        candidates = [
            neighbor
            for neighbor, bond in adjacency[current]
            if neighbor != previous
            and neighbor not in excluded
            and _atom(entity, neighbor).element == "C"
            and bond.order == 1.0
        ]
        if len(candidates) > 1:
            return ()
        if not candidates:
            return tuple(chain)
        previous, current = current, candidates[0]
        chain.append(current)

def _ring_hydroxy_positions(entity, ring, attachment):
    substituents = {
        atom_id: ("hydroxy",)
        for atom_id in ring
        if any(
            _atom(entity, neighbor).element == "O"
            and bond.order == 1.0
            and _hydrogen_count(entity, neighbor) == 1
            for neighbor, bond in _adjacency(entity)[atom_id]
            if neighbor not in ring
        )
    }
    numbering = _best_ring_numbering(ring, substituents, fixed_one=attachment)
    return sorted(numbering[atom_id] for atom_id in substituents)

def _best_ring_numbering(ring, substituents, fixed_one=None):
    candidates = []
    for direction in (tuple(ring), tuple(reversed(ring))):
        for offset in range(6):
            ordered = direction[offset:] + direction[:offset]
            if fixed_one is not None and ordered[0] != fixed_one:
                continue
            numbering = {atom_id: index + 1 for index, atom_id in enumerate(ordered)}
            locants = tuple(
                sorted(
                    (numbering[atom_id], name)
                    for atom_id, names in substituents.items()
                    for name in names
                )
            )
            candidates.append((tuple(value[0] for value in locants), locants, numbering))
    return min(candidates, key=lambda value: (value[0], value[1]))[2]

def _is_methyl(entity, atom_id, parent_id):
    adjacency = _adjacency(entity)
    heavy = [
        neighbor
        for neighbor, _ in adjacency[atom_id]
        if _atom(entity, neighbor).element != "H"
    ]
    return heavy == [parent_id] and _hydrogen_count(entity, atom_id) == 3

def _locanted_prefix(locants, prefix):
    locant_text = ",".join(map(str, locants))
    multiplier = {1: "", 2: "di", 3: "tri"}.get(len(locants), f"{len(locants)}-")
    return f"{locant_text}-{multiplier}{prefix}"

def _prefix_string(prefixes):
    if not prefixes:
        return ""
    grouped = {}
    for locant, name in prefixes:
        grouped.setdefault(name, []).append(locant)
    parts = [
        _locanted_prefix(sorted(locants), name)
        for name, locants in sorted(grouped.items())
    ]
    return "-".join(parts)


def name_hydride(entity):
    counts = _element_counts(entity)
    if counts == Counter({"O": 1, "H": 2}) and _single_heavy_center(entity, "O"):
        return (
            "water",
            True,
            "Recognize the retained parent-hydride name for H2O.",
        )
    if counts == Counter({"N": 1, "H": 3}) and _single_heavy_center(entity, "N"):
        return (
            "azane",
            True,
            "Select the parent-hydride name for the nitrogen hydride NH3.",
        )
    if (
        counts == Counter({"N": 1, "H": 4})
        and _single_heavy_center(entity, "N")
        and (entity.net_charge or 0) == 1
    ):
        return (
            "azanium",
            False,
            "Recognize the charged parent-hydride name for NH4+.",
        )
    return None

def name_anilide(entity):
    adjacency = _adjacency(entity)
    ring = _six_carbon_ring(entity)
    if ring is None:
        return None
    for carbonyl in entity.atoms:
        if carbonyl.element != "C":
            continue
        double_o = [
            neighbor
            for neighbor, bond in adjacency[carbonyl.atom_id]
            if _atom(entity, neighbor).element == "O" and bond.order == 2.0
        ]
        nitrogens = [
            neighbor
            for neighbor, bond in adjacency[carbonyl.atom_id]
            if _atom(entity, neighbor).element == "N" and bond.order == 1.0
        ]
        if len(double_o) != 1 or len(nitrogens) != 1:
            continue
        nitrogen = nitrogens[0]
        ring_attachments = [
            neighbor
            for neighbor, bond in adjacency[nitrogen]
            if neighbor in ring and bond.order == 1.0
        ]
        if len(ring_attachments) != 1:
            continue
        acyl_carbons = _acyclic_acyl_chain(entity, carbonyl.atom_id, set(ring))
        stem = alkane_stem(len(acyl_carbons))
        if stem is None:
            continue
        hydroxy_positions = _ring_hydroxy_positions(
            entity,
            ring,
            ring_attachments[0],
        )
        if not hydroxy_positions:
            continue
        parent = {1: "formamide", 2: "acetamide"}.get(
            len(acyl_carbons), stem + "anamide"
        )
        prefix = _locanted_prefix(hydroxy_positions, "hydroxy") + "phenyl"
        return (
            f"N-({prefix}){parent}",
            True,
            "Select the carboxamide as the senior characteristic group.",
            "Use the retained amide parent name where permitted.",
            "Name the N-bound substituted phenyl group and assign its lowest ring locants.",
        )
    return None

def name_benzene_family(entity):
    ring = _six_carbon_ring(entity)
    if ring is None:
        return None
    adjacency = _adjacency(entity)
    substituents = {}
    for ring_atom in ring:
        outside = [
            (neighbor, bond)
            for neighbor, bond in adjacency[ring_atom]
            if neighbor not in ring and _atom(entity, neighbor).element != "H"
        ]
        names = []
        for neighbor, bond in outside:
            atom = _atom(entity, neighbor)
            if atom.element == "O" and bond.order == 1.0 and _hydrogen_count(entity, neighbor) == 1:
                names.append("hydroxy")
            elif atom.element in HALOGEN_PREFIX and len(adjacency[neighbor]) == 1:
                names.append(HALOGEN_PREFIX[atom.element])
            elif atom.element == "C" and _is_methyl(entity, neighbor, ring_atom):
                names.append("methyl")
            else:
                return None
        if names:
            substituents[ring_atom] = tuple(sorted(names))
    if not substituents:
        return (
            "benzene",
            True,
            "Recognize the six-member monocyclic aromatic hydrocarbon parent.",
        )
    hydroxy_sites = [atom_id for atom_id, names in substituents.items() if "hydroxy" in names]
    if len(hydroxy_sites) == 1:
        numbering = _best_ring_numbering(ring, substituents, fixed_one=hydroxy_sites[0])
        prefixes = [
            (locant, name)
            for atom_id, locant in numbering.items()
            for name in substituents.get(atom_id, ())
            if name != "hydroxy"
        ]
        name = _prefix_string(prefixes) + "phenol"
    else:
        numbering = _best_ring_numbering(ring, substituents)
        prefixes = [
            (locant, name)
            for atom_id, locant in numbering.items()
            for name in substituents.get(atom_id, ())
        ]
        name = _prefix_string(prefixes) + "benzene"
    return (
        name,
        True,
        "Select benzene or phenol as the retained parent hydride.",
        "Choose the ring numbering that gives the lowest locant sequence.",
        "Cite detachable prefixes alphabetically.",
    )

# --- reverse parsers for the retained parents ---------------------------------

def _evidence() -> tuple[Evidence, ...]:
    return (Evidence(EvidenceSource.IUPAC_NAME, "self_contained_iupac_subset_parser"),)


def _p_atom(atom_id: str, element: str, hydrogens: int | None = None, *, formal_charge: int | None = None) -> ChemicalAtom:
    return ChemicalAtom(atom_id=atom_id, element=element, formal_charge=formal_charge, implicit_hydrogens=hydrogens if hydrogens else None, evidence=_evidence())


def _p_bond(left: str, right: str, order: float, *, aromatic: bool = False) -> ChemicalBond:
    return ChemicalBond(atom1_id=left, atom2_id=right, order=order, kind=BondKind.COVALENT, aromatic=aromatic, evidence=_evidence())


def _p_entity(name: str, atoms: list[ChemicalAtom], bonds: list[ChemicalBond]) -> FiniteChemicalEntity:
    return FiniteChemicalEntity(entity_id=f"iupac:{name}", atoms=tuple(atoms), bonds=tuple(bonds), net_charge=sum(atom.formal_charge or 0 for atom in atoms), status=InferenceStatus.EXPLICIT, evidence=_evidence())


def _p_carbon_chain(count: int, *, name: str, terminal_group: str | None = None):
    atoms: list[ChemicalAtom] = []
    bonds: list[ChemicalBond] = []
    for index in range(count):
        atom_id = f"C{index + 1}"
        if index == 0 and terminal_group == "carbonyl":
            hydrogens = 1 if count == 1 else 0
        elif count == 1:
            hydrogens = 4
        elif index in {0, count - 1}:
            hydrogens = 3
        else:
            hydrogens = 2
        atoms.append(_p_atom(atom_id, "C", hydrogens))
        if index:
            bonds.append(_p_bond(f"C{index}", atom_id, 1.0))
    return atoms, bonds


_PREFIX_PATTERN = re.compile(r"(?P<locants>\d+(?:,\d+)*)-(?:(?P<multiplier>di|tri|\d+-)?(?P<prefix>fluoro|chloro|bromo|iodo|methyl|hydroxy))")


def parse_prefixes(text: str):
    if not text:
        return []
    values = []
    position = 0
    while position < len(text):
        match = _PREFIX_PATTERN.match(text, position)
        if match is None:
            raise NamingParseError(f"unsupported prefix syntax in {text!r}")
        locants = tuple(int(value) for value in match.group("locants").split(","))
        prefix, multiplier = match.group("prefix"), match.group("multiplier")
        numeric = multiplier[:-1] if multiplier and multiplier.endswith("-") else multiplier
        expected = {None: 1, "di": 2, "tri": 3}.get(multiplier)
        if expected is None:
            try:
                expected = int(numeric)
            except (TypeError, ValueError) as exc:
                raise NamingParseError(f"unsupported multiplier in {text!r}") from exc
            if expected < 4:
                raise NamingParseError(f"numeric prefix multiplier must be at least four in {text!r}")
        if len(locants) != expected:
            raise NamingParseError(f"{multiplier or 'single'}-{prefix} requires {expected} locant(s)")
        values.extend((locant, prefix) for locant in locants)
        position = match.end()
        if position < len(text):
            if text[position] != "-":
                raise NamingParseError(f"unsupported prefix separator in {text!r}")
            position += 1
    if len({locant for locant, _ in values}) != len(values):
        raise NamingParseError("a ring locant may occur only once")
    if any(not 1 <= locant <= 6 for locant, _ in values):
        raise NamingParseError("benzene locants must be between 1 and 6")
    return values


def parse_parent_hydride(name: str):
    if name == "water":
        return _p_entity(name, [_p_atom("O1", "O", 2)], [])
    if name == "azane":
        return _p_entity(name, [_p_atom("N1", "N", 3)], [])
    if name == "azanium":
        return _p_entity(name, [_p_atom("N1", "N", 4, formal_charge=1)], [])
    return None


def parse_benzene(name: str):
    if name.endswith("phenol"):
        base, prefix_text = "phenol", name[:-len("phenol")].rstrip("-")
    elif name.endswith("benzene"):
        base, prefix_text = "benzene", name[:-len("benzene")].rstrip("-")
    else:
        return None
    prefixes = parse_prefixes(prefix_text)
    hydroxy_count = sum(prefix == "hydroxy" for _, prefix in prefixes)
    if base == "phenol":
        if hydroxy_count:
            raise NamingParseError("phenol already supplies the position-one hydroxy group")
        substituents = [(1, "hydroxy"), *prefixes]
    else:
        if hydroxy_count == 1:
            raise NamingParseError("one hydroxy substituent must be named as phenol")
        substituents = prefixes
    atoms = [_p_atom(f"C{index}", "C") for index in range(1, 7)]
    bonds = [_p_bond(f"C{index}", f"C{index % 6 + 1}", 1.5, aromatic=True) for index in range(1, 7)]
    for locant, prefix in substituents:
        ring_id = f"C{locant}"
        ring_atom = next(atom for atom in atoms if atom.atom_id == ring_id)
        atoms[atoms.index(ring_atom)] = _p_atom(ring_id, "C", 0)
        if prefix == "hydroxy":
            atoms.append(_p_atom(f"O{locant}", "O", 1))
            bonds.append(_p_bond(ring_id, f"O{locant}", 1.0))
        elif prefix == "methyl":
            atoms.append(_p_atom(f"M{locant}", "C", 3))
            bonds.append(_p_bond(ring_id, f"M{locant}", 1.0))
        else:
            element = next(element for element, value in HALOGEN_PREFIX.items() if value == prefix)
            atoms.append(_p_atom(f"X{locant}", element))
            bonds.append(_p_bond(ring_id, f"X{locant}", 1.0))
    return _p_entity(name, atoms, bonds)


_ANILIDE_PATTERN = re.compile(r"^n-\((?P<phenyl>[^()]+)phenyl\)(?P<parent>[a-z]+amide)$")


def _stem_count(stem: str) -> int | None:
    from ..naming import ALKANE_STEMS
    for count, value in sorted(ALKANE_STEMS.items(), key=lambda item: len(item[1]), reverse=True):
        if value == stem:
            return count
    return None


def parse_anilide(name: str):
    match = _ANILIDE_PATTERN.fullmatch(name)
    if match is None:
        return None
    phenyl, parent = match.group("phenyl"), match.group("parent")
    if parent == "formamide":
        acyl_count = 1
    elif parent == "acetamide":
        acyl_count = 2
    elif parent.endswith("anamide"):
        acyl_count = _stem_count(parent[:-len("anamide")])
        if acyl_count is None or acyl_count < 3:
            return None
    else:
        return None
    prefixes = parse_prefixes(phenyl)
    if not prefixes or any(prefix != "hydroxy" for _, prefix in prefixes):
        raise NamingParseError("anilide phenyl groups require hydroxy substituents")
    acyl_atoms, acyl_bonds = _p_carbon_chain(acyl_count, name=name, terminal_group="carbonyl")
    ring_atoms = [_p_atom(f"R{index}", "C") for index in range(1, 7)]
    ring_bonds = [_p_bond(f"R{index}", f"R{index % 6 + 1}", 1.5, aromatic=True) for index in range(1, 7)]
    ring_atoms[0] = _p_atom("R1", "C", 0)
    atoms = [*acyl_atoms, *ring_atoms, _p_atom("O1", "O"), _p_atom("N1", "N", 1)]
    bonds = [*acyl_bonds, *ring_bonds, _p_bond("C1", "O1", 2.0), _p_bond("C1", "N1", 1.0), _p_bond("N1", "R1", 1.0)]
    for locant, _ in prefixes:
        ring_id = f"R{locant}"
        ring_atom = next(atom for atom in atoms if atom.atom_id == ring_id)
        atoms[atoms.index(ring_atom)] = _p_atom(ring_id, "C", 0)
        oxygen_id = f"OH{locant}"
        atoms.append(_p_atom(oxygen_id, "O", 1))
        bonds.append(_p_bond(ring_id, oxygen_id, 1.0))
    return _p_entity(name, atoms, bonds)


__all__ = ["name_hydride", "name_anilide", "name_benzene_family", "parse_parent_hydride", "parse_benzene", "parse_anilide", "parse_prefixes"]
