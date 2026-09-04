"""Self-contained conversion between a bounded IUPAC name subset and SMILES.

This module deliberately accepts only names emitted by the current
``naming`` implementation.  It is not a general IUPAC parser; unsupported
names fail closed rather than producing an unverified molecular graph.
"""

from __future__ import annotations

from dataclasses import replace
import re

from .line_notation import (
    LineNotation,
    LineNotationError,
    from_line_notation,
    to_line_notation,
)
from .equivalence import constitution_equivalent
from .models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
)
from .naming import (
    ALKANE_STEMS,
    HALOGEN_PREFIX,
    NamingIndeterminateError,
    NamingResult,
    name_entity,
)


class NamingParseError(ValueError):
    """Raised when a name is malformed or outside the reversible subset."""


_STEM_TO_CARBON_COUNT = {stem: count for count, stem in ALKANE_STEMS.items()}
_ALLOWED_PREFIXES = {
    "fluoro",
    "chloro",
    "bromo",
    "iodo",
    "methyl",
    "hydroxy",
}
_PREFIX_PATTERN = re.compile(
    r"(?P<locants>\d+(?:,\d+)*)-(?:(?P<multiplier>di|tri|\d+-)?(?P<prefix>"
    r"fluoro|chloro|bromo|iodo|methyl|hydroxy))"
)
_ALCOHOL_PATTERN = re.compile(r"^(?P<stem>[a-z]+)an-(?P<locant>\d+)-ol$")
_ANILIDE_PATTERN = re.compile(
    r"^n-\((?P<phenyl>[^()]+)phenyl\)(?P<parent>[a-z]+amide)$"
)
_DEFAULT_VALENCE = {
    "H": 1.0,
    "B": 3.0,
    "C": 4.0,
    "N": 3.0,
    "O": 2.0,
    "P": 3.0,
    "S": 2.0,
    "F": 1.0,
    "Cl": 1.0,
    "Br": 1.0,
    "I": 1.0,
}


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    normalized = " ".join(name.strip().lower().split())
    if not normalized:
        raise NamingParseError("IUPAC name must not be empty")
    return normalized


def _evidence() -> tuple[Evidence, ...]:
    return (
        Evidence(
            EvidenceSource.IUPAC_NAME,
            "self_contained_iupac_subset_parser",
        ),
    )


def _optional_hydrogens(hydrogens: int | None) -> int | None:
    """Use ``None`` for zero so omitted OpenSMILES H stays canonical."""
    return hydrogens if hydrogens else None


def _atom(atom_id: str, element: str, hydrogens: int | None = None) -> ChemicalAtom:
    return ChemicalAtom(
        atom_id=atom_id,
        element=element,
        # OpenSMILES leaves aromatic and fully substituted atoms without an
        # explicit hydrogen field.  Treat zero as the same absent value so
        # graph equivalence does not depend on how the name was parsed.
        implicit_hydrogens=_optional_hydrogens(hydrogens),
        evidence=_evidence(),
    )


def _bond(
    left: str,
    right: str,
    order: float,
    *,
    aromatic: bool = False,
) -> ChemicalBond:
    return ChemicalBond(
        atom1_id=left,
        atom2_id=right,
        order=order,
        kind=BondKind.COVALENT,
        aromatic=aromatic,
        evidence=_evidence(),
    )


def _entity(name: str, atoms: list[ChemicalAtom], bonds: list[ChemicalBond]) -> FiniteChemicalEntity:
    return FiniteChemicalEntity(
        entity_id=f"iupac:{name}",
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        net_charge=0,
        status=InferenceStatus.EXPLICIT,
        evidence=_evidence(),
    )


def _carbon_chain(count: int, *, name: str, terminal_group: str | None = None):
    atoms: list[ChemicalAtom] = []
    bonds: list[ChemicalBond] = []
    for index in range(count):
        atom_id = f"C{index + 1}"
        if index == 0 and terminal_group == "acid":
            # Formic acid retains one carbonyl hydrogen; longer acids attach
            # the carbonyl carbon to the next carbon in the chain.
            hydrogens = 1 if count == 1 else 0
        elif index == 0 and terminal_group == "carbonyl":
            # Formamide retains one carbonyl hydrogen; acyl chains do not.
            hydrogens = 1 if count == 1 else 0
        elif count == 1:
            hydrogens = 4
        elif index in {0, count - 1}:
            hydrogens = 3
        else:
            hydrogens = 2
        atoms.append(_atom(atom_id, "C", hydrogens))
        if index:
            bonds.append(_bond(f"C{index}", atom_id, 1.0))
    return atoms, bonds


def _parse_parent_hydride(name: str):
    if name == "water":
        return _entity(name, [_atom("O1", "O", 2)], [])
    if name == "azane":
        return _entity(name, [_atom("N1", "N", 3)], [])
    return None


def _parse_alkane(name: str):
    if not name.endswith("ane"):
        return None
    stem = name[:-3]
    count = _STEM_TO_CARBON_COUNT.get(stem)
    if count is None:
        return None
    atoms, bonds = _carbon_chain(count, name=name)
    return _entity(name, atoms, bonds)


def _parse_alcohol(name: str):
    if name == "methanol":
        count, locant = 1, 1
    elif name == "ethanol":
        count, locant = 2, 1
    else:
        match = _ALCOHOL_PATTERN.fullmatch(name)
        if match is None:
            return None
        count = _STEM_TO_CARBON_COUNT.get(match.group("stem"))
        locant = int(match.group("locant"))
        if count is None or count < 3 or not 1 <= locant <= count:
            return None

    atoms, bonds = _carbon_chain(count, name=name)
    oxygen_id = "O1"
    carbon_id = f"C{locant}"
    carbon_index = next(index for index, atom in enumerate(atoms) if atom.atom_id == carbon_id)
    carbon = atoms[carbon_index]
    atoms[carbon_index] = _atom(carbon_id, "C", (carbon.implicit_hydrogens or 0) - 1)
    atoms.append(_atom(oxygen_id, "O", 1))
    bonds.append(_bond(carbon_id, oxygen_id, 1.0))
    return _entity(name, atoms, bonds)


def _parse_acid(name: str):
    if not name.endswith("anoic acid"):
        return None
    stem = name[: -len("anoic acid")]
    count = _STEM_TO_CARBON_COUNT.get(stem)
    if count is None:
        return None
    atoms, bonds = _carbon_chain(count, name=name, terminal_group="acid")
    carbonyl_id = "C1"
    atoms.extend((_atom("O1", "O"), _atom("O2", "O", 1)))
    bonds.extend(
        (
            _bond(carbonyl_id, "O1", 2.0),
            _bond(carbonyl_id, "O2", 1.0),
        )
    )
    return _entity(name, atoms, bonds)


def _parse_prefixes(text: str):
    """Parse the detachable-prefix grammar emitted by ``_prefix_string``."""
    if not text:
        return []
    values = []
    position = 0
    while position < len(text):
        match = _PREFIX_PATTERN.match(text, position)
        if match is None:
            raise NamingParseError(f"unsupported prefix syntax in {text!r}")
        locants = tuple(int(value) for value in match.group("locants").split(","))
        prefix = match.group("prefix")
        multiplier = match.group("multiplier")
        numeric_multiplier = multiplier[:-1] if multiplier and multiplier.endswith("-") else multiplier
        expected = {None: 1, "di": 2, "tri": 3}.get(multiplier)
        if expected is None:
            try:
                expected = int(numeric_multiplier)
            except (TypeError, ValueError) as exc:
                raise NamingParseError(
                    f"unsupported multiplier in {text!r}"
                ) from exc
            if expected < 4:
                raise NamingParseError(
                    f"numeric prefix multiplier must be at least four in {text!r}"
                )
        if len(locants) != expected:
            raise NamingParseError(
                f"{multiplier or 'single'}-{prefix} requires {expected} locant(s)"
            )
        values.extend((locant, prefix) for locant in locants)
        position = match.end()
        if position < len(text):
            if text[position] != "-":
                raise NamingParseError(f"unsupported prefix separator in {text!r}")
            position += 1
    if any(prefix not in _ALLOWED_PREFIXES for _, prefix in values):
        raise NamingParseError(f"unsupported prefix in {text!r}")
    if len({locant for locant, _ in values}) != len(values):
        raise NamingParseError("a ring locant may occur only once")
    if any(not 1 <= locant <= 6 for locant, _ in values):
        raise NamingParseError("benzene locants must be between 1 and 6")
    return values


def _parse_benzene(name: str):
    base = None
    if name.endswith("phenol"):
        base = "phenol"
        prefix_text = name[: -len("phenol")].rstrip("-")
    elif name.endswith("benzene"):
        base = "benzene"
        prefix_text = name[: -len("benzene")].rstrip("-")
    else:
        return None

    prefixes = _parse_prefixes(prefix_text)
    hydroxy_count = sum(prefix == "hydroxy" for _, prefix in prefixes)
    if base == "phenol":
        if hydroxy_count:
            raise NamingParseError("phenol already supplies the position-one hydroxy group")
        substituents = [(1, "hydroxy"), *prefixes]
    else:
        if hydroxy_count == 1:
            raise NamingParseError("one hydroxy substituent must be named as phenol")
        substituents = prefixes

    atoms = [_atom(f"C{index}", "C") for index in range(1, 7)]
    bonds = [
        _bond(
            f"C{index}",
            f"C{index % 6 + 1}",
            1.5,
            aromatic=True,
        )
        for index in range(1, 7)
    ]
    for locant, prefix in substituents:
        ring_id = f"C{locant}"
        ring_atom = next(atom for atom in atoms if atom.atom_id == ring_id)
        atoms[atoms.index(ring_atom)] = _atom(ring_id, "C", 0)
        if prefix == "hydroxy":
            atoms.append(_atom(f"O{locant}", "O", 1))
            bonds.append(_bond(ring_id, f"O{locant}", 1.0))
        elif prefix == "methyl":
            atoms.append(_atom(f"M{locant}", "C", 3))
            bonds.append(_bond(ring_id, f"M{locant}", 1.0))
        else:
            element = next(
                element for element, value in HALOGEN_PREFIX.items() if value == prefix
            )
            atoms.append(_atom(f"X{locant}", element))
            bonds.append(_bond(ring_id, f"X{locant}", 1.0))
    return _entity(name, atoms, bonds)


def _parse_anilide(name: str):
    match = _ANILIDE_PATTERN.fullmatch(name)
    if match is None:
        return None
    phenyl = match.group("phenyl")
    parent = match.group("parent")
    if parent == "formamide":
        acyl_count = 1
    elif parent == "acetamide":
        acyl_count = 2
    elif parent.endswith("anamide"):
        stem = parent[: -len("anamide")]
        acyl_count = _STEM_TO_CARBON_COUNT.get(stem)
        if acyl_count is None or acyl_count < 3:
            return None
    else:
        return None

    prefixes = _parse_prefixes(phenyl)
    if not prefixes or any(prefix != "hydroxy" for _, prefix in prefixes):
        raise NamingParseError("anilide phenyl groups require hydroxy substituents")

    acyl_atoms, acyl_bonds = _carbon_chain(
        acyl_count,
        name=name,
        terminal_group="carbonyl",
    )
    ring_atoms = [_atom(f"R{index}", "C") for index in range(1, 7)]
    ring_bonds = [
        _bond(
            f"R{index}",
            f"R{index % 6 + 1}",
            1.5,
            aromatic=True,
        )
        for index in range(1, 7)
    ]
    ring_atoms[0] = _atom("R1", "C", 0)
    atoms = [*acyl_atoms, *ring_atoms, _atom("O1", "O"), _atom("N1", "N", 1)]
    bonds = [*acyl_bonds, *ring_bonds]
    bonds.extend((_bond("C1", "O1", 2.0), _bond("C1", "N1", 1.0), _bond("N1", "R1", 1.0)))
    for locant, _ in prefixes:
        ring_id = f"R{locant}"
        ring_atom = next(atom for atom in atoms if atom.atom_id == ring_id)
        atoms[atoms.index(ring_atom)] = _atom(ring_id, "C", 0)
        oxygen_id = f"OH{locant}"
        atoms.append(_atom(oxygen_id, "O", 1))
        bonds.append(_bond(ring_id, oxygen_id, 1.0))
    return _entity(name, atoms, bonds)


def _parse_name(name: str) -> FiniteChemicalEntity:
    for parser in (
        _parse_parent_hydride,
        _parse_alkane,
        _parse_alcohol,
        _parse_acid,
        _parse_anilide,
        _parse_benzene,
    ):
        result = parser(name)
        if result is not None:
            return result
    raise NamingParseError(
        f"IUPAC name {name!r} is outside the reversible MolCrysKit subset"
    )


def from_iupac_name(name: str) -> FiniteChemicalEntity:
    """Parse a canonical name from the bounded self-contained subset.

    The parser accepts the exact normalized names emitted by
    :func:`name_entity`; synonyms and general IUPAC names are rejected.
    """
    normalized = _normalize_name(name)
    entity = _parse_name(normalized)
    if not _is_reversible_entity(entity):
        raise NamingParseError(
            f"name {normalized!r} describes a graph with invalid valence or semantics"
        )
    try:
        canonical = name_entity(entity, strict=True).name
    except (NamingIndeterminateError, ValueError) as exc:
        raise NamingParseError(
            f"name {normalized!r} could not be validated by the naming rules"
        ) from exc
    if _normalize_name(canonical) != normalized:
        raise NamingParseError(
            f"name {normalized!r} is not canonical; expected {canonical!r}"
        )
    return entity


def iupac_to_smiles(name: str) -> LineNotation:
    """Convert a supported IUPAC name to a lossless OpenSMILES result."""
    entity = from_iupac_name(name)
    notation = to_line_notation(entity, dialect="opensmiles")
    if not notation.lossless:
        raise NamingParseError("OpenSMILES conversion was not lossless")
    return notation


def _is_reversible_entity(entity: FiniteChemicalEntity) -> bool:
    if entity.net_charge != 0:
        return False
    if any(
        atom.isotope is not None
        or atom.formal_charge not in {None, 0}
        or atom.radical_electrons
        or atom.stereochemistry is not None
        for atom in entity.atoms
    ):
        return False
    if not all(
        bond.kind is BondKind.COVALENT
        and bond.atom2_image_shift == (0, 0, 0)
        and bond.stereochemistry is None
        and bond.order in {1.0, 1.5, 2.0, 3.0}
        for bond in entity.bonds
    ):
        return False
    return _valence_not_exceeded(entity)


def _valence_not_exceeded(entity: FiniteChemicalEntity) -> bool:
    """Return whether each supported atom stays within its target valence."""
    adjacency = {atom.atom_id: [] for atom in entity.atoms}
    for bond in entity.bonds:
        adjacency[bond.atom1_id].append(bond)
        adjacency[bond.atom2_id].append(bond)
    for atom in entity.atoms:
        target = _DEFAULT_VALENCE.get(atom.element)
        if target is None:
            continue
        valence = sum(bond.order or 0.0 for bond in adjacency[atom.atom_id])
        valence += (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)
        if valence > target + 1e-8:
            return False
    return True


def complete_open_smiles_hydrogens(
    entity: FiniteChemicalEntity,
) -> FiniteChemicalEntity:
    """Apply OpenSMILES default valences to unbracketed organic atoms.

    ``from_line_notation`` intentionally keeps the low-level parser lossless
    and does not guess implicit hydrogens.  OpenSMILES, however, assigns
    default valences to unbracketed organic-subset atoms.  Strict naming needs
    those hydrogens to recognize otherwise unambiguous names such as ``CCO``.
    Bracket atoms are left untouched because ``[C]`` explicitly opts out of
    the organic-subset defaults.
    """
    adjacency: dict[str, list[tuple[str, ChemicalBond]]] = {
        atom.atom_id: [] for atom in entity.atoms
    }
    for bond in entity.bonds:
        adjacency[bond.atom1_id].append((bond.atom2_id, bond))
        adjacency[bond.atom2_id].append((bond.atom1_id, bond))

    target_valence = {
        "B": 3.0,
        "C": 4.0,
        "N": 3.0,
        "O": 2.0,
        "P": 3.0,
        "S": 2.0,
    }
    changed = False
    atoms = []
    for atom in entity.atoms:
        if (
            atom.implicit_hydrogens is not None
            or atom.explicit_hydrogens is not None
            or any(
                evidence.source is EvidenceSource.LINE_NOTATION
                and evidence.method == "OpenSMILES bracket atom parser"
                for evidence in atom.evidence
            )
            or atom.element not in target_valence
        ):
            atoms.append(atom)
            continue
        bond_sum = sum(bond.order or 0.0 for _, bond in adjacency[atom.atom_id])
        inferred = max(0, int(round(target_valence[atom.element] - bond_sum)))
        # Aromatic atoms and fully substituted atoms use the same stable
        # representation as the line-notation generator: zero is omitted.
        hydrogen_value = inferred or None
        if hydrogen_value != atom.implicit_hydrogens:
            changed = True
            atoms.append(
                replace(
                    atom,
                    implicit_hydrogens=hydrogen_value,
                    evidence=(
                        *atom.evidence,
                        Evidence(
                            EvidenceSource.INFERRED,
                            "OpenSMILES default-valence completion",
                        ),
                    ),
                )
            )
        else:
            atoms.append(atom)
    if not changed:
        return entity
    return replace(
        entity,
        atoms=tuple(atoms),
        evidence=(
            *entity.evidence,
            Evidence(
                EvidenceSource.INFERRED,
                "OpenSMILES default-valence completion",
            ),
        ),
    )


def smiles_to_iupac(smiles: str, *, strict: bool = True) -> NamingResult:
    """Convert OpenSMILES to a naming result, optionally requiring reversibility.

    In strict mode, malformed or empty notation is reported as
    :class:`NamingIndeterminateError` together with unsupported semantics.
    Non-strict mode preserves the one-way fallback behavior and therefore
    leaves OpenSMILES default hydrogens unresolved before calling
    :func:`name_entity`; for example, ``CCO`` may return a composition
    description there.  Use strict mode when OpenSMILES defaults and a
    reversible name are required.
    """
    if not isinstance(smiles, str):
        raise TypeError("smiles must be a string")
    try:
        entity = from_line_notation(smiles, dialect="opensmiles")
    except LineNotationError as exc:
        if strict:
            raise NamingIndeterminateError(
                "SMILES is empty or invalid OpenSMILES notation"
            ) from exc
        raise
    if not isinstance(entity, FiniteChemicalEntity) or not _is_reversible_entity(entity):
        if strict:
            if isinstance(entity, FiniteChemicalEntity) and not _valence_not_exceeded(entity):
                raise NamingIndeterminateError(
                    "SMILES exceeds the default valence of one or more atoms"
                )
            raise NamingIndeterminateError(
                "SMILES contains semantics outside the reversible naming subset"
            )
        return name_entity(entity)
    naming_entity = complete_open_smiles_hydrogens(entity) if strict else entity
    if strict and not _is_reversible_entity(naming_entity):
        raise NamingIndeterminateError(
            "SMILES exceeds the default valence of one or more atoms"
        )
    result = name_entity(naming_entity, strict=strict)
    if not strict:
        return result
    try:
        rebuilt = complete_open_smiles_hydrogens(from_iupac_name(result.name))
        original = complete_open_smiles_hydrogens(
            from_line_notation(smiles, dialect="opensmiles")
        )
        if not constitution_equivalent(original, rebuilt):
            raise NamingIndeterminateError(
                "generated IUPAC name does not round-trip to an equivalent graph"
            )
    except NamingParseError as exc:
        raise NamingIndeterminateError(
            f"generated name {result.name!r} is not reversible"
        ) from exc
    return result


__all__ = [
    "NamingParseError",
    "complete_open_smiles_hydrogens",
    "from_iupac_name",
    "iupac_to_smiles",
    "smiles_to_iupac",
]
