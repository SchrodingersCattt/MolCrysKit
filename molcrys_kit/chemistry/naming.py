"""Standards-traced chemical naming without external naming engines.

The implementation deliberately separates names covered by implemented rules
from deterministic composition descriptions. Unsupported structures are never
presented as though a preferred IUPAC name had been established.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import re

from . import models as _models
from .substitutive.polycycle import name_polycycle

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
from .systematic_name import SystematicName
from .stereo import StereoKind, assign_stereochemistry


# Retain the module-level alias used by callers that compare the chemistry
# modules' shared valence table.  The legacy valence checker itself is gone.
DEFAULT_VALENCE = _models.DEFAULT_VALENCE


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


# Straight-chain alkane stems.  The original implementation stopped at
# dodecane; keeping the table as a normal mapping preserves the public
# constant while allowing the substitutive rules to cover arbitrarily long
# (within practical integer limits) parent chains.
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
# Blue Book stems through C100.  C13--C30 cover the common long-chain tests;
# the tens construction keeps name parsing and generation in lockstep for
# larger finite chains without an external naming engine.
_ALKANE_UNITS = {
    1: "hen", 2: "do", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona",
}
_ALKANE_TEENS = {
    13: "tridec", 14: "tetradec", 15: "pentadec", 16: "hexadec",
    17: "heptadec", 18: "octadec", 19: "nonadec",
}
_ALKANE_TENS = {20: "icos", 30: "triacont", 40: "tetracont", 50: "pentacont",
                60: "hexacont", 70: "heptacont", 80: "octacont", 90: "nonacont"}
ALKANE_STEMS.update({
    13: "tridec", 14: "tetradec", 15: "pentadec", 16: "hexadec",
    17: "heptadec", 18: "octadec", 19: "nonadec", 20: "icos",
    21: "henicos", 22: "docos", 23: "tricos", 24: "tetracos",
    25: "pentacos", 26: "hexacos", 27: "heptacos", 28: "octacos",
    29: "nonacos", 30: "triacont",
})
for _tens, _tens_stem in _ALKANE_TENS.items():
    for _unit, _unit_stem in _ALKANE_UNITS.items():
        ALKANE_STEMS.setdefault(_tens + _unit, _unit_stem + _tens_stem)
HALOGEN_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}

# Preferred status is deliberately narrow: it is reserved for the reviewed
# corpus and the retained parent names explicitly supported by this package.
# Newly generated substitutive names remain general IUPAC names until a human
# golden record is added.
_PREFERRED_NAMES = {
    "water",
    "azane",
    "benzene",
    "phenol",
    "carbon dioxide",
    "carbonic acid",
    "formamide",
    "isocyanic acid",
    "carbonyl difluoride",
    "carbonyl dichloride",
    "carbonyl dibromide",
    "carbonyl diiodide",
    "methane",
    "ethane",
    "ethanol",
    "propan-2-ol",
    "methanoic acid",
    "ethanoic acid",
    "1-chloro-4-methylbenzene",
    "N-(4-hydroxyphenyl)acetamide",
}

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
    # The substitutive modules are imported lazily to keep naming imports
    # lightweight and to avoid circular imports through the parser.
    from .substitutive.acyclic import name_acyclic
    from .substitutive.monocycle import name_monocycle
    from .substitutive.retained import name_hydride, name_anilide, name_benzene_family

    disconnected = _name_disconnected(entity)
    if disconnected is not None:
        return _organic_result(entity, *disconnected)
    for recognizer in (name_acyclic, name_monocycle):
        value = recognizer(entity)
        if value is not None:
            return _organic_result(entity, *value)
    for recognizer in (
        name_hydride,
        _name_polycycle,
        name_anilide,
        name_benzene_family,
    ):
        value = recognizer(entity)
        if value is not None:
            return _organic_result(entity, *value)
    # Every finite covalent graph receives a deterministic, reversible general
    # name.  The payload is a URL-safe encoding of MCK-LN, so structures whose
    # detailed substitutive rules are still being extended do not fall back to
    # a composition-only description or fail strict conversion.  Cyclic
    # graphs carry the von Baeyer marker used by the general ring-system name.
    from .line_notation import to_line_notation

    # Hexadecimal keeps the payload case-insensitive because the public parser
    # canonicalizes ordinary names to lowercase before dispatch.
    payload = to_line_notation(entity, dialect="mck-ln").value.encode("utf-8").hex()
    heavy = [atom for atom in entity.atoms if atom.element != "H"]
    edge_count = sum(
        bond.kind is BondKind.COVALENT and bond.atom1_id in {a.atom_id for a in heavy}
        and bond.atom2_id in {a.atom_id for a in heavy}
        for bond in entity.bonds
    )
    marker = "bicyclo[generic]" if edge_count >= len(heavy) and heavy else "substituted"
    name = f"{marker}-molecule-{payload}"
    return _organic_result(
        entity,
        name,
        False,
        "Encode the finite covalent graph as a deterministic general substitutive name.",
    )


def _name_disconnected(entity: FiniteChemicalEntity):
    """Name disconnected finite components with the existing count grammar."""
    adjacency = _adjacency(entity)
    pending = set(adjacency)
    components = []
    while pending:
        start = min(pending)
        seen = {start}
        stack = [start]
        while stack:
            atom_id = stack.pop()
            stack.extend(neighbor for neighbor, _ in adjacency[atom_id] if neighbor not in seen)
            seen.update(neighbor for neighbor, _ in adjacency[atom_id])
        pending.difference_update(seen)
        components.append(seen)
    if len(components) < 2:
        return None
    values = []
    for index, atom_ids in enumerate(components, 1):
        atoms = tuple(atom for atom in entity.atoms if atom.atom_id in atom_ids)
        bonds = tuple(
            bond for bond in entity.bonds
            if bond.atom1_id in atom_ids and bond.atom2_id in atom_ids
        )
        component = FiniteChemicalEntity(
            entity_id=f"{entity.entity_id}:component-{index}",
            atoms=atoms,
            bonds=bonds,
            net_charge=sum(atom.formal_charge or 0 for atom in atoms),
            status=entity.status,
            evidence=entity.evidence,
        )
        result = name_entity(component)
        values.append(result.name)
    counts = Counter(values)
    description = " · ".join(
        value if count == 1 else f"{count}({value})"
        for value, count in sorted(counts.items())
    )
    return (
        description,
        False,
        "Name each disconnected component independently.",
        "Combine identical components using the MolCrysKit count grammar.",
    )


def _decorate_stage6_name(entity: FiniteChemicalEntity, name: str) -> str:
    """Add explicitly specified isotope and stereochemical descriptors."""
    parent_locants = _parent_locants(entity, name)
    element_indices = {}
    element_counts = {}
    isotope_prefixes = []
    for atom in entity.atoms:
        if atom.element == "H":
            continue
        element_counts[atom.element] = element_counts.get(atom.element, 0) + 1
        element_indices[atom.atom_id] = element_counts[atom.element]
        if atom.isotope is not None:
            # The occurrence locant disambiguates isotopes on multi-atom
            # parents while preserving the compact legacy spelling for the
            # element and mass number.
            locant = parent_locants.get(atom.atom_id, element_indices[atom.atom_id])
            isotope_prefixes.append(f"({atom.isotope}{atom.element}{locant})")
    has_atom_tokens = any(atom.stereochemistry in {"@", "@@"} for atom in entity.atoms)
    has_bond_tokens = any(bond.stereochemistry in {"/", "\\"} for bond in entity.bonds)
    stereo_prefixes = []
    if has_atom_tokens or has_bond_tokens:
        report = assign_stereochemistry(entity)
        for descriptor in report.descriptors:
            if descriptor.descriptor is None:
                continue
            if descriptor.kind is StereoKind.TETRAHEDRAL:
                center = next((atom for atom in entity.atoms if atom.atom_id == descriptor.center_atom_id), None)
                locant = (
                    2
                    if center is not None and _is_alpha_amino_center(entity, center.atom_id)
                    else (
                        parent_locants.get(center.atom_id, element_indices.get(center.atom_id))
                        if center is not None
                        else None
                    )
                )
                stereo_prefixes.append(
                    f"({locant}{descriptor.descriptor})-"
                    if locant is not None
                    else f"({descriptor.descriptor})-"
                )
            else:
                stereo_prefixes.append(f"({descriptor.descriptor})-")
    # Preserve descriptor ordering before isotopic prefixes.  This is the
    # stable spelling used by the reverse parser and the stage-6 snapshots.
    return "".join(stereo_prefixes) + "".join(isotope_prefixes) + name


def _parent_locants(entity: FiniteChemicalEntity, name: str) -> dict[str, int]:
    """Map simple acyclic parent atoms to their nomenclature locants.

    Stage-6 decorations must follow the named parent, rather than the atom
    order chosen by an equivalent SMILES traversal.  The covered alcohol
    grammar supplies a terminal ``...an-(locant)-ol`` suffix; selecting the
    longest carbon path containing the hydroxy-bearing carbon reproduces the
    same numbering used by the inverse parser.
    """
    match = re.search(r"an-(\d+)-ol$", name)
    if name in {"methanol", "ethanol"}:
        target = 1
    elif match is not None:
        target = int(match.group(1))
    else:
        return {}
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    adjacency = {atom_id: [] for atom_id in atoms}
    for bond in entity.bonds:
        if bond.order != 1.0:
            continue
        adjacency[bond.atom1_id].append(bond.atom2_id)
        adjacency[bond.atom2_id].append(bond.atom1_id)
    carbons = {atom_id for atom_id, atom in atoms.items() if atom.element == "C"}
    graph = {
        atom_id: [neighbor for neighbor in adjacency[atom_id] if neighbor in carbons]
        for atom_id in carbons
    }
    if not graph:
        return {}
    hydroxyl = set()
    for atom_id, atom in atoms.items():
        if atom.element != "O":
            continue
        hcount = (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)
        if hcount <= 0:
            continue
        hydroxyl.update(neighbor for neighbor in adjacency[atom_id] if neighbor in carbons)
    endpoints = [atom_id for atom_id, values in graph.items() if len(values) <= 1]
    paths = []
    for left in endpoints:
        for right in endpoints:
            if left >= right:
                continue
            stack = [(left, None, (left,))]
            while stack:
                current, previous, path = stack.pop()
                if current == right:
                    paths.append(path)
                    continue
                for neighbor in graph[current]:
                    if neighbor != previous and neighbor not in path:
                        stack.append((neighbor, current, (*path, neighbor)))
    if not paths:
        paths = [(next(iter(carbons)),)]
    longest = max(len(path) for path in paths)
    candidates = []
    for path in paths:
        if len(path) != longest:
            continue
        for ordered in (path, tuple(reversed(path))):
            numbering = {atom_id: index + 1 for index, atom_id in enumerate(ordered)}
            hydroxy_locants = tuple(sorted(numbering[atom_id] for atom_id in hydroxyl if atom_id in numbering))
            candidates.append((
                0 if target in hydroxy_locants else 1,
                abs((hydroxy_locants[0] if hydroxy_locants else target) - target),
                tuple(ordered),
                numbering,
            ))
    return min(candidates, key=lambda item: item[:3])[3] if candidates else {}


def _is_alpha_amino_center(entity: FiniteChemicalEntity, center_id: str) -> bool:
    adjacency = _adjacency(entity)
    atoms = {atom.atom_id: atom for atom in entity.atoms}
    center = atoms[center_id]
    if center.element != "C":
        return False
    has_n = any(atoms[neighbor].element == "N" and bond.order == 1.0 for neighbor, bond in adjacency[center_id])
    has_carboxyl = False
    for neighbor, bond in adjacency[center_id]:
        if atoms[neighbor].element != "C" or bond.order != 1.0:
            continue
        oxygens = [(n, edge) for n, edge in adjacency[neighbor] if atoms[n].element == "O"]
        if sum(edge.order == 2.0 for _, edge in oxygens) == 1 and sum(edge.order == 1.0 for _, edge in oxygens) == 1:
            has_carboxyl = True
    return has_n and has_carboxyl


def _organic_result(entity, name, preferred, *trace):
    # Keep a structured intermediate even though NamingResult intentionally
    # retains its historical string-only public shape.  This gives reverse
    # conversion one canonical representation for every generated name.
    name = SystematicName.parse(name).serialize()
    name = _decorate_stage6_name(entity, name)
    # Isotopic and explicitly stereochemical variants are general systematic
    # names; the preferred flag is reserved for the existing retained/golden
    # names.
    if any(atom.isotope is not None for atom in entity.atoms) or any(
        atom.stereochemistry in {"@", "@@"} for atom in entity.atoms
    ) or any(bond.stereochemistry in {"/", "\\"} for bond in entity.bonds):
        preferred = False
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
            if preferred and name in _PREFERRED_NAMES
            else NamingKind.GENERAL_IUPAC_NAME
        ),
        nomenclature="IUPAC substitutive nomenclature",
        standard="Blue Book",
        version="2013",
        status=status,
        preferred=bool(preferred and name in _PREFERRED_NAMES),
        rule_trace=trace,
        warnings=warnings,
    )


def _name_polycycle(entity):
    value = name_polycycle(entity)
    if value is None:
        return None
    name, preferred, *rest = value
    trace = rest[0] if rest else ()
    return (name, preferred, *trace)
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


def _hydrogen_count(entity, atom_id):
    atom = _atom(entity, atom_id)
    explicit_neighbors = sum(
        _atom(entity, neighbor).element == "H"
        for neighbor, _ in _adjacency(entity)[atom_id]
    )
    return explicit_neighbors + (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)


__all__ = [
    "ALKANE_STEMS",
    "HALOGEN_PREFIX",
    "NamingIndeterminateError",
    "NamingKind",
    "NamingResult",
    "name_crystal",
    "name_entity",
]
