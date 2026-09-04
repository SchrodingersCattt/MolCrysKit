"""Standards-traced chemical naming without external naming engines.

The implementation deliberately separates names covered by implemented rules
from deterministic composition descriptions. Unsupported structures are never
presented as though a preferred IUPAC name had been established.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from .models import (
    BondKind,
    ChemicalEntity,
    CrystalChemistry,
    FiniteChemicalEntity,
    InferenceStatus,
    MulticomponentEntity,
    PeriodicChemicalEntity,
    PolymerChemicalEntity,
)


class NamingKind(str, Enum):
    """Semantic strength of a returned name string."""

    PREFERRED_IUPAC_NAME = "preferred_iupac_name"
    GENERAL_IUPAC_NAME = "general_iupac_name"
    IUPAC_COMPOSITION_DESCRIPTION = "iupac_composition_description"


@dataclass(frozen=True)
class NamingResult:
    """A name or composition description with scope and rule provenance."""

    name: str
    kind: NamingKind
    nomenclature: str
    standard: str
    version: str
    status: InferenceStatus
    preferred: bool | None
    rule_trace: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    source: str = "molcrys_kit_self_contained"


class NamingIndeterminateError(ValueError):
    """Raised when strict naming would return a provisional description."""


# Explicit straight-chain alkane stems currently implemented for C1-C12.
ALKANE_STEMS = {
    1: "meth",
    2: "eth",
    3: "prop",
    4: "but",
    5: "pent",
    6: "hex",
    7: "hept",
    8: "oct",
    9: "non",
    10: "dec",
    11: "undec",
    12: "dodec",
}
HALOGEN_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}


def name_entity(entity: ChemicalEntity, *, strict: bool = False) -> NamingResult:
    """Name an entity within the explicitly implemented IUPAC rule scope.

    This is the general one-way naming API: non-strict mode may return a
    deterministic composition description when no preferred name is
    established.  Call ``smiles_to_iupac(..., strict=True)`` for the narrower
    API that additionally requires a round-trip through ``iupac_to_smiles``.
    """
    if isinstance(entity, FiniteChemicalEntity):
        result = _name_finite(entity)
    elif isinstance(entity, PeriodicChemicalEntity):
        result = _periodic_description(entity)
    elif isinstance(entity, PolymerChemicalEntity):
        result = _name_polymer(entity)
    elif isinstance(entity, MulticomponentEntity):
        result = _name_multicomponent(entity)
    else:
        raise TypeError(f"unsupported chemical entity: {type(entity).__name__}")
    if strict and result.status in {
        InferenceStatus.PROVISIONAL,
        InferenceStatus.INDETERMINATE,
    }:
        raise NamingIndeterminateError(result.warnings[0] if result.warnings else result.name)
    return result


def name_crystal(structure_or_chemistry, *, strict: bool = False) -> NamingResult:
    """Return a crystal name or deterministic stoichiometric description."""
    if isinstance(structure_or_chemistry, CrystalChemistry):
        chemistry = structure_or_chemistry
    else:
        chemistry = getattr(structure_or_chemistry, "chemistry", None)
        if chemistry is None:
            from .perception import infer_chemistry

            chemistry = infer_chemistry(structure_or_chemistry)
    if not chemistry.components:
        result = NamingResult(
            name="empty crystal chemistry model",
            kind=NamingKind.IUPAC_COMPOSITION_DESCRIPTION,
            nomenclature="IUPAC compositional nomenclature",
            standard="Red Book",
            version="2005",
            status=InferenceStatus.INDETERMINATE,
            preferred=None,
            rule_trace=("No chemical components are available for naming.",),
            warnings=("crystal contains no named chemical components",),
        )
    else:
        component_results = [name_entity(component) for component in chemistry.components]
        counts = Counter(result.name for result in component_results)
        if len(counts) == 1:
            component = component_results[0]
            result = NamingResult(
                name=component.name,
                kind=component.kind,
                nomenclature=component.nomenclature,
                standard=component.standard,
                version=component.version,
                status=_combined_naming_status(component_results, chemistry.status),
                preferred=component.preferred,
                rule_trace=(
                    *component.rule_trace,
                    "Equivalent unit-cell entities were collapsed by generated name.",
                ),
                warnings=tuple(
                    dict.fromkeys(
                        warning
                        for item in component_results
                        for warning in item.warnings
                    )
                ),
            )
        else:
            ordered = sorted(counts.items())
            description = " · ".join(
                name if count == 1 else f"{count}({name})"
                for name, count in ordered
            )
            result = NamingResult(
                name=description,
                kind=NamingKind.IUPAC_COMPOSITION_DESCRIPTION,
                nomenclature="IUPAC compositional nomenclature",
                standard="Red Book",
                version="2005",
                status=InferenceStatus.PROVISIONAL,
                preferred=None,
                rule_trace=(
                    "Name each chemical component independently.",
                    "Combine components in deterministic lexical order with unit-cell counts.",
                ),
                warnings=(
                    "a unique salt, adduct, or solvate name is not established; showing a deterministic composition description",
                ),
            )
    if strict and result.status in {
        InferenceStatus.PROVISIONAL,
        InferenceStatus.INDETERMINATE,
    }:
        raise NamingIndeterminateError(result.warnings[0] if result.warnings else result.name)
    return result


def _name_finite(entity: FiniteChemicalEntity) -> NamingResult:
    for recognizer in (
        _name_hydride,
        _name_hydrocarbon,
        _name_alcohol,
        _name_carboxylic_acid,
        _name_anilide,
        _name_benzene_family,
    ):
        value = recognizer(entity)
        if value is not None:
            return _organic_result(entity, *value)
    formula = _formula(entity)
    return NamingResult(
        name=f"molecular entity {formula}",
        kind=NamingKind.IUPAC_COMPOSITION_DESCRIPTION,
        nomenclature="IUPAC compositional nomenclature",
        standard="Blue Book / Red Book",
        version="2013 / 2005",
        status=InferenceStatus.INDETERMINATE,
        preferred=None,
        rule_trace=(
            "Determine the Hill formula from the explicit chemical graph.",
            "Stop before substitutive naming because the required rule family is outside the implemented coverage matrix.",
        ),
        warnings=(
            "no implemented rule family establishes a unique IUPAC name; composition description shown",
        ),
    )


def _organic_result(entity, name, preferred, *trace):
    source_status = entity.status
    status = (
        source_status
        if source_status in {InferenceStatus.EXPLICIT, InferenceStatus.CONFIRMED}
        else InferenceStatus.PROVISIONAL
    )
    warnings = ()
    if status is InferenceStatus.PROVISIONAL:
        warnings = ("name depends on provisional or inferred chemical connectivity",)
    return NamingResult(
        name=name,
        kind=(
            NamingKind.PREFERRED_IUPAC_NAME
            if preferred
            else NamingKind.GENERAL_IUPAC_NAME
        ),
        nomenclature="IUPAC substitutive nomenclature",
        standard="Blue Book",
        version="2013",
        status=status,
        preferred=preferred,
        rule_trace=trace,
        warnings=warnings,
    )


def _name_hydride(entity):
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
    return None


def _name_hydrocarbon(entity):
    if set(_element_counts(entity)) - {"C", "H"}:
        return None
    carbons = [atom.atom_id for atom in entity.atoms if atom.element == "C"]
    chain = _unbranched_carbon_chain(entity, carbons)
    if chain is None or not _all_bonds(entity, {1.0}):
        return None
    if not _normal_valence(entity):
        return None
    stem = ALKANE_STEMS.get(len(chain))
    if stem is None:
        return None
    return (
        stem + "ane",
        True,
        "Select the longest unbranched saturated carbon parent.",
        "Apply the acyclic hydrocarbon suffix -ane.",
    )


def _name_alcohol(entity):
    heavy = [atom for atom in entity.atoms if atom.element != "H"]
    oxygens = [atom for atom in heavy if atom.element == "O"]
    if len(oxygens) != 1 or any(atom.element not in {"C", "O"} for atom in heavy):
        return None
    oxygen = oxygens[0]
    adjacency = _adjacency(entity)
    carbon_neighbors = [
        neighbor
        for neighbor, bond in adjacency[oxygen.atom_id]
        if _atom(entity, neighbor).element == "C" and bond.order == 1.0
    ]
    if len(carbon_neighbors) != 1 or _hydrogen_count(entity, oxygen.atom_id) != 1:
        return None
    carbons = [atom.atom_id for atom in heavy if atom.element == "C"]
    chain = _unbranched_carbon_chain(entity, carbons)
    if chain is None or not _all_bonds(entity, {1.0}) or not _normal_valence(entity):
        return None
    stem = ALKANE_STEMS.get(len(chain))
    if stem is None:
        return None
    positions = [
        chain.index(carbon_neighbors[0]) + 1,
        tuple(reversed(chain)).index(carbon_neighbors[0]) + 1,
    ]
    locant = min(positions)
    if len(chain) == 1:
        name = "methanol"
    elif len(chain) == 2:
        name = "ethanol"
    else:
        name = f"{stem}an-{locant}-ol"
    return (
        name,
        True,
        "Select the longest carbon chain containing the hydroxy-bearing carbon.",
        "Number the chain to give the hydroxy suffix the lowest locant.",
        "Apply the suffix -ol.",
    )


def _name_carboxylic_acid(entity):
    adjacency = _adjacency(entity)
    carboxyl = None
    for atom in entity.atoms:
        if atom.element != "C":
            continue
        oxygens = [
            (neighbor, bond)
            for neighbor, bond in adjacency[atom.atom_id]
            if _atom(entity, neighbor).element == "O"
        ]
        double = [neighbor for neighbor, bond in oxygens if bond.order == 2.0]
        hydroxy = [
            neighbor
            for neighbor, bond in oxygens
            if bond.order == 1.0 and _hydrogen_count(entity, neighbor) == 1
        ]
        if len(double) == 1 and len(hydroxy) == 1:
            carboxyl = atom.atom_id
            break
    if carboxyl is None:
        return None
    if any(atom.element not in {"C", "H", "O"} for atom in entity.atoms):
        return None
    carbons = [atom.atom_id for atom in entity.atoms if atom.element == "C"]
    chain = _unbranched_carbon_chain(entity, carbons)
    if chain is None or carboxyl not in {chain[0], chain[-1]}:
        return None
    if sum(atom.element == "O" for atom in entity.atoms) != 2:
        return None
    stem = ALKANE_STEMS.get(len(chain))
    if stem is None:
        return None
    return (
        stem + "anoic acid",
        True,
        "Select the chain containing the carboxylic acid characteristic atom.",
        "Assign the carboxyl carbon locant one and apply the suffix -oic acid.",
    )


def _name_anilide(entity):
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
        stem = ALKANE_STEMS.get(len(acyl_carbons))
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


def _name_benzene_family(entity):
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


def _periodic_description(entity):
    return NamingResult(
        name=f"{entity.periodic_rank}-dimensional periodic entity {_formula(entity)}",
        kind=NamingKind.IUPAC_COMPOSITION_DESCRIPTION,
        nomenclature="IUPAC additive/compositional nomenclature",
        standard="Red Book",
        version="2005",
        status=InferenceStatus.INDETERMINATE,
        preferred=None,
        rule_trace=(
            "Preserve periodic dimensionality and repeat composition.",
            "Stop before additive network naming because coordination descriptors are incomplete.",
        ),
        warnings=("a unique IUPAC network name is not established",),
    )


def _name_polymer(entity):
    if len(entity.repeat_units) == 1:
        repeat = name_entity(entity.repeat_units[0])
        return NamingResult(
            name=f"poly({repeat.name})",
            kind=NamingKind.GENERAL_IUPAC_NAME,
            nomenclature="IUPAC structure-based polymer nomenclature",
            standard="Purple Book",
            version="2008",
            status=InferenceStatus.PROVISIONAL,
            preferred=False,
            rule_trace=(
                "Name the single constitutional repeating unit.",
                "Enclose the repeating-unit name in poly(...).",
            ),
            warnings=(
                "polymer end groups and typed connection descriptors are incomplete",
                *repeat.warnings,
            ),
        )
    return NamingResult(
        name=f"polymer with {len(entity.repeat_units)} repeat-unit types",
        kind=NamingKind.IUPAC_COMPOSITION_DESCRIPTION,
        nomenclature="IUPAC polymer nomenclature",
        standard="Purple Book",
        version="2008",
        status=InferenceStatus.INDETERMINATE,
        preferred=None,
        rule_trace=("Retain the count of distinct repeat-unit records.",),
        warnings=("a unique structure-based polymer name is not established",),
    )


def _name_multicomponent(entity):
    named = [(name_entity(component), count) for component, count in entity.components]
    description = " · ".join(
        result.name if count == 1 else f"{count}({result.name})"
        for result, count in named
    )
    return NamingResult(
        name=description,
        kind=NamingKind.IUPAC_COMPOSITION_DESCRIPTION,
        nomenclature="IUPAC compositional nomenclature",
        standard="Red Book",
        version="2005",
        status=InferenceStatus.PROVISIONAL,
        preferred=None,
        rule_trace=(
            "Name each component independently.",
            "Preserve declared component stoichiometry and order.",
        ),
        warnings=("salt/adduct/solvate class is not established from composition alone",),
    )


def _combined_naming_status(results, chemistry_status):
    if any(result.status is InferenceStatus.INDETERMINATE for result in results):
        return InferenceStatus.INDETERMINATE
    if chemistry_status in {InferenceStatus.EXPLICIT, InferenceStatus.CONFIRMED} and all(
        result.status in {InferenceStatus.EXPLICIT, InferenceStatus.CONFIRMED}
        for result in results
    ):
        return chemistry_status
    return InferenceStatus.PROVISIONAL


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


def _formula(entity):
    counts = _element_counts(entity)
    order = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
    order.extend(sorted(symbol for symbol in counts if symbol not in order))
    return "".join(
        symbol + ("" if counts[symbol] == 1 else str(counts[symbol]))
        for symbol in order
    )


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
    expected = {"H": 1.0, "C": 4.0, "N": 3.0, "O": 2.0}
    for atom in entity.atoms:
        if atom.element not in expected:
            continue
        bond_sum = sum(bond.order or 0.0 for _, bond in adjacency[atom.atom_id])
        bond_sum += (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)
        if abs(bond_sum - expected[atom.element]) > 1e-8:
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


__all__ = [
    "ALKANE_STEMS",
    "HALOGEN_PREFIX",
    "NamingIndeterminateError",
    "NamingKind",
    "NamingResult",
    "name_crystal",
    "name_entity",
]
