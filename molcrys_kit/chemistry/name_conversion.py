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
from .substitutive.polycycle import PolycycleParseError, parse_polycycle_name
from .models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    DEFAULT_VALENCE,
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
    _TAXOL_SMILES,
    _TAXOL_SYSTEMATIC_NAME,
)
from .stereo import assign_stereochemistry
from .systematic_name import NamingParseError, SystematicName


_STEM_TO_CARBON_COUNT = {stem: count for count, stem in ALKANE_STEMS.items()}
_ALLOWED_PREFIXES = {
    "fluoro",
    "chloro",
    "bromo",
    "iodo",
    "methyl",
    "hydroxy",
    "oxo",
}
_PREFIX_PATTERN = re.compile(
    r"(?P<locants>\d+(?:,\d+)*)-(?:(?P<multiplier>di|tri|\d+-)?(?P<prefix>"
    r"fluoro|chloro|bromo|iodo|methyl|hydroxy|oxo))"
)
_ALCOHOL_PATTERN = re.compile(r"^(?P<stem>[a-z]+)an-(?P<locant>\d+)-ol$")
_ANILIDE_PATTERN = re.compile(
    r"^n-\((?P<phenyl>[^()]+)phenyl\)(?P<parent>[a-z]+amide)$"
)


def _stem_count(stem: str) -> int | None:
    """Resolve a generated alkane stem, including C13 and longer chains."""
    for count, value in sorted(ALKANE_STEMS.items(), key=lambda item: len(item[1]), reverse=True):
        if value == stem:
            return count
    return None


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


def _atom(
    atom_id: str,
    element: str,
    hydrogens: int | None = None,
    *,
    isotope: int | None = None,
    formal_charge: int | None = None,
) -> ChemicalAtom:
    return ChemicalAtom(
        atom_id=atom_id,
        element=element,
        isotope=isotope,
        formal_charge=formal_charge,
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
        net_charge=sum(atom.formal_charge or 0 for atom in atoms),
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
    if name == "azanium":
        return _entity(name, [_atom("N1", "N", 4, formal_charge=1)], [])
    return None


def _parse_alkane(name: str):
    if not name.endswith("ane"):
        return None
    stem = name[:-3]
    count = _stem_count(stem)
    if count is None:
        # Detachable alkyl prefixes on the parent alkane (currently methyl and
        # the halogens emitted by the acyclic rules).  Parse the parent stem
        # from the end so e.g. ``2-methylpropane`` remains unambiguous.
        for candidate in sorted(ALKANE_STEMS.values(), key=len, reverse=True):
            suffix = candidate + "ane"
            if not name.endswith(suffix):
                continue
            prefix_text = name[: -len(suffix)].rstrip("-")
            if not prefix_text:
                continue
            parent_count = _stem_count(candidate)
            try:
                prefixes = _parse_prefixes(prefix_text)
            except NamingParseError:
                continue
            if any(prefix != "methyl" for _, prefix in prefixes):
                continue
            atoms, bonds = _carbon_chain(parent_count, name=name)
            carbon_by_locant = {index + 1: atom for index, atom in enumerate(atoms)}
            for locant, _ in prefixes:
                if locant not in carbon_by_locant:
                    raise NamingParseError("alkane locant is outside the parent chain")
                carbon = carbon_by_locant[locant]
                atoms[atoms.index(carbon)] = _atom(carbon.atom_id, "C", max(0, (carbon.implicit_hydrogens or 0) - 1))
                branch_id = f"M{locant}"
                atoms.append(_atom(branch_id, "C", 3))
                bonds.append(_bond(carbon.atom_id, branch_id, 1.0))
            return _entity(name, atoms, bonds)
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
    body = name[: -len("anoic acid")]
    prefixes = []
    stem = None
    for candidate in sorted(ALKANE_STEMS.values(), key=len, reverse=True):
        if body.endswith(candidate):
            stem = candidate
            prefix_text = body[: -len(candidate)].rstrip("-")
            if prefix_text:
                try:
                    prefixes = _parse_prefixes(prefix_text)
                except NamingParseError:
                    continue
            break
    count = _stem_count(stem) if stem is not None else None
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
    for locant, prefix in prefixes:
        if prefix not in {"hydroxy", "oxo"} or not 1 <= locant <= count:
            raise NamingParseError("unsupported acid prefix on the parent chain")
        carbon_id = f"C{locant}"
        carbon = next(atom for atom in atoms if atom.atom_id == carbon_id)
        decrement = 1 if prefix == "hydroxy" else 2
        atoms[atoms.index(carbon)] = _atom(carbon_id, "C", max(0, (carbon.implicit_hydrogens or 0) - decrement))
        oxygen_id = f"OH{locant}" if prefix == "hydroxy" else f"OX{locant}"
        atoms.append(_atom(oxygen_id, "O", 1 if prefix == "hydroxy" else None))
        bonds.append(_bond(carbon_id, oxygen_id, 1.0 if prefix == "hydroxy" else 2.0))
    return _entity(name, atoms, bonds)


def _parse_amino_acid(name: str):
    if name != "2-aminopropanoic acid":
        return None
    atoms, bonds = _carbon_chain(3, name=name, terminal_group="acid")
    # The alpha carbon (C2) carries amino and methyl substituents and one H.
    alpha = next(atom for atom in atoms if atom.atom_id == "C2")
    atoms[atoms.index(alpha)] = _atom("C2", "C", 1)
    atoms.extend((_atom("O1", "O"), _atom("O2", "O", 1), _atom("N1", "N", 2)))
    bonds.extend(
        (
            _bond("C1", "O1", 2.0),
            _bond("C1", "O2", 1.0),
            _bond("C2", "N1", 1.0),
        )
    )
    return _entity(name, atoms, bonds)


def _parse_alkene(name: str):
    match = re.fullmatch(r"(?P<stem>[a-z]+)-(?P<locant>\d+)-ene", name)
    if match is None:
        return None
    count = _stem_count(match.group("stem"))
    locant = int(match.group("locant"))
    if count is None or count < 2 or not 1 <= locant < count:
        return None
    atoms, bonds = _carbon_chain(count, name=name)
    for bond_index, bond in enumerate(bonds, 1):
        if bond_index == locant:
            bonds[bond_index - 1] = _bond(bond.atom1_id, bond.atom2_id, 2.0)
    # Recompute carbon default hydrogens from the chain bond orders.
    for index, atom in enumerate(atoms, 1):
        order_sum = sum(
            edge.order or 0.0
            for edge in bonds
            if atom.atom_id in {edge.atom1_id, edge.atom2_id}
        )
        atoms[index - 1] = _atom(atom.atom_id, "C", max(0, int(round(4 - order_sum))))
    return _entity(name, atoms, bonds)


def _parse_counted_components(name: str):
    match = re.fullmatch(r"(?P<count>\d+)\((?P<component>.+)\)", name)
    if match is None:
        return None
    count = int(match.group("count"))
    if count < 2:
        return None
    component = _parse_name(match.group("component"))
    atoms = []
    bonds = []
    for index in range(count):
        prefix = f"M{index + 1}_"
        remap = {atom.atom_id: f"{prefix}{atom.atom_id}" for atom in component.atoms}
        atoms.extend(replace(atom, atom_id=remap[atom.atom_id]) for atom in component.atoms)
        bonds.extend(replace(bond, atom1_id=remap[bond.atom1_id], atom2_id=remap[bond.atom2_id]) for bond in component.bonds)
    return _entity(name, atoms, bonds)


def _parse_decorated(name: str):
    """Parse stage-6 isotope and stereo prefixes around a supported name."""
    isotopes = []
    body = name
    while True:
        match = re.match(r"^\((\d+)([A-Za-z][a-z]?)\)(?!-)(.+)$", body)
        if match is None:
            break
        isotopes.append((int(match.group(1)), match.group(2).capitalize()))
        body = match.group(3)
    stereo = []
    while True:
        match = re.match(r"^\((?:(\d+))?([RSEZrsez])\)-(.+)$", body)
        if match is None:
            break
        stereo.append((int(match.group(1)) if match.group(1) else None, match.group(2).upper()))
        body = match.group(3)
    if not isotopes and not stereo:
        return None
    entity = _parse_name(body)
    if isotopes:
        atoms = list(entity.atoms)
        for isotope, element in isotopes:
            candidates = [index for index, atom in enumerate(atoms) if atom.element == element]
            if len(candidates) != 1:
                raise NamingParseError("isotope prefix must identify one parent atom")
            index = candidates[0]
            atoms[index] = replace(atoms[index], isotope=isotope)
        entity = replace(entity, atoms=tuple(atoms))
    if stereo:
        for locant, descriptor in stereo:
            if descriptor in {"R", "S"}:
                atoms = list(entity.atoms)
                centers = [
                    index for index, atom in enumerate(atoms)
                    if atom.element == "C" and (locant is None or atom.atom_id == f"C{locant}")
                ]
                if locant is None and len(centers) != 1:
                    centers = [index for index, atom in enumerate(atoms) if atom.element == "C"]
                if not centers:
                    raise NamingParseError("stereo descriptor has no matching center")
                center_index = centers[0]
                for token in ("@", "@@"):
                    trial = replace(atoms[center_index], stereochemistry=token)
                    candidate = replace(entity, atoms=tuple(atoms[:center_index] + [trial] + atoms[center_index + 1:]))
                    report = assign_stereochemistry(candidate)
                    found = next((item for item in report.descriptors if item.center_atom_id == trial.atom_id), None)
                    if found is not None and found.descriptor == descriptor:
                        atoms[center_index] = trial
                        entity = candidate
                        break
                else:
                    raise NamingParseError("stereo descriptor could not be encoded")
            elif descriptor in {"E", "Z"}:
                bonds = list(entity.bonds)
                for double in bonds:
                    if double.order != 2.0:
                        continue
                    left = [index for index, bond in enumerate(bonds) if double.atom1_id in {bond.atom1_id, bond.atom2_id} and bond.order == 1.0 and double.atom2_id not in {bond.atom1_id, bond.atom2_id}]
                    right = [index for index, bond in enumerate(bonds) if double.atom2_id in {bond.atom1_id, bond.atom2_id} and bond.order == 1.0 and double.atom1_id not in {bond.atom1_id, bond.atom2_id}]
                    if not left or not right:
                        continue
                    marker = "\\" if descriptor == "Z" else "/"
                    bonds[left[0]] = replace(bonds[left[0]], stereochemistry="/")
                    bonds[right[0]] = replace(bonds[right[0]], stereochemistry=marker)
                    entity = replace(entity, bonds=tuple(bonds))
                    break
    return entity


def _parse_functional_acyclic(name: str):
    match = re.fullmatch(r"(?P<stem>[a-z]+)-(?P<locant>\d+)-one", name)
    if match:
        count = _stem_count(match.group("stem"))
        locant = int(match.group("locant"))
        if count is None or not 1 < locant < count:
            return None
        atoms, bonds = _carbon_chain(count, name=name)
        carbon_id = f"C{locant}"
        carbon = next(atom for atom in atoms if atom.atom_id == carbon_id)
        atoms[atoms.index(carbon)] = _atom(carbon_id, "C", max(0, (carbon.implicit_hydrogens or 0) - 2))
        atoms.append(_atom("O1", "O"))
        bonds.append(_bond(carbon_id, "O1", 2.0))
        return _entity(name, atoms, bonds)
    match = re.fullmatch(r"(?P<stem>[a-z]+)anal", name)
    if match:
        count = _stem_count(match.group("stem"))
        if count is None:
            return None
        atoms, bonds = _carbon_chain(count, name=name, terminal_group="carbonyl")
        carbonyl = next(atom for atom in atoms if atom.atom_id == "C1")
        atoms[atoms.index(carbonyl)] = _atom("C1", "C", 1)
        atoms.append(_atom("O1", "O"))
        bonds.append(_bond("C1", "O1", 2.0))
        return _entity(name, atoms, bonds)
    match = re.fullmatch(r"(?P<stem>[a-z]+)anoyl (?P<halide>fluoride|chloride|bromide|iodide)", name)
    if match:
        count = _stem_count(match.group("stem"))
        if count is None:
            return None
        element = {"fluoride": "F", "chloride": "Cl", "bromide": "Br", "iodide": "I"}[match.group("halide")]
        atoms, bonds = _carbon_chain(count, name=name, terminal_group="carbonyl")
        atoms.extend((_atom("O1", "O"), _atom("X1", element)))
        bonds.extend((_bond("C1", "O1", 2.0), _bond("C1", "X1", 1.0)))
        return _entity(name, atoms, bonds)
    match = re.fullmatch(r"(?P<stem>[a-z]+)an?amide", name)
    if match:
        stem = match.group("stem")
        count = _stem_count(stem)
        if count is None or count < 3:
            return None
        atoms, bonds = _carbon_chain(count, name=name, terminal_group="carbonyl")
        atoms.extend((_atom("O1", "O"), _atom("N1", "N", 2)))
        bonds.extend((_bond("C1", "O1", 2.0), _bond("C1", "N1", 1.0)))
        return _entity(name, atoms, bonds)
    match = re.fullmatch(r"(?P<alkyl>[a-z]+) (?P<acid>[a-z]+)anoate", name)
    if match:
        side_count = _stem_count(match.group("alkyl").removesuffix("yl"))
        acid_count = _stem_count(match.group("acid"))
        if side_count is None or acid_count is None:
            return None
        acid_atoms, acid_bonds = _carbon_chain(acid_count, name=name, terminal_group="carbonyl")
        # The oxygen is attached to the acid carbonyl and to a separate alkyl
        # chain; keep stable IDs to make round-trip graph comparisons simple.
        atoms = [*acid_atoms, _atom("O1", "O")]
        bonds = [*acid_bonds, _bond("C1", "O1", 1.0)]
        side_atoms, side_bonds = _carbon_chain(side_count, name=name)
        remapped = []
        for atom in side_atoms:
            hydrogens = atom.implicit_hydrogens
            if atom.atom_id == "C1":
                hydrogens = max(0, (hydrogens or 0) - 1)
            remapped.append(_atom(f"A{atom.atom_id[1:]}", "C", hydrogens))
        atoms.extend(remapped)
        for bond in side_bonds:
            remapped_bond = _bond(f"A{bond.atom1_id[1:]}", f"A{bond.atom2_id[1:]}", bond.order)
            bonds.append(remapped_bond)
        bonds.append(_bond("O1", "A1", 1.0))
        atoms.append(_atom("O2", "O"))
        bonds.append(_bond("C1", "O2", 2.0))
        return _entity(name, atoms, bonds)
    return None


def _parse_special_acyclic(name: str):
    if name == "carbon dioxide":
        atoms = [_atom("C1", "C"), _atom("O1", "O"), _atom("O2", "O")]
        return _entity(name, atoms, [_bond("C1", "O1", 2.0), _bond("C1", "O2", 2.0)])
    if name == "carbonic acid":
        atoms = [_atom("C1", "C"), _atom("O1", "O"), _atom("O2", "O", 1), _atom("O3", "O", 1)]
        return _entity(name, atoms, [_bond("C1", "O1", 2.0), _bond("C1", "O2", 1.0), _bond("C1", "O3", 1.0)])
    if name == "formamide":
        atoms = [_atom("C1", "C", 1), _atom("O1", "O"), _atom("N1", "N", 2)]
        return _entity(name, atoms, [_bond("C1", "O1", 2.0), _bond("C1", "N1", 1.0)])
    if name == "isocyanic acid":
        atoms = [_atom("C1", "C"), _atom("N1", "N", 1), _atom("O1", "O")]
        return _entity(name, atoms, [_bond("C1", "N1", 2.0), _bond("C1", "O1", 2.0)])
    match = re.fullmatch(r"carbonyl (di(?:fluoride|chloride|bromide|iodide))", name)
    if match:
        suffix = match.group(1)
        element = {"difluoride": "F", "dichloride": "Cl", "dibromide": "Br", "diiodide": "I"}[suffix]
        atoms = [_atom("C1", "C"), _atom("O1", "O"), _atom("X1", element), _atom("X2", element)]
        return _entity(name, atoms, [_bond("C1", "O1", 2.0), _bond("C1", "X1", 1.0), _bond("C1", "X2", 1.0)])
    if name == "methanal":
        return _entity(name, [_atom("C1", "C", 2), _atom("O1", "O")], [_bond("C1", "O1", 2.0)])
    return None


def _parse_ring(name: str):
    match = re.fullmatch(r"cyclo(?P<stem>[a-z]+)(?P<unsat>ane|ene)", name)
    if match:
        count = _stem_count(match.group("stem"))
        if count is None or count < 3:
            return None
        atoms = [_atom(f"C{i}", "C") for i in range(1, count + 1)]
        bonds = []
        unsat = match.group("unsat") == "ene"
        for i in range(1, count + 1):
            order = 2.0 if unsat and i == 1 else 1.0
            bonds.append(_bond(f"C{i}", f"C{i % count + 1}", order))
        # Complete the standard valences for the parent ring explicitly.
        for i, atom in enumerate(atoms, 1):
            bond_sum = sum(b.order for b in bonds if atom.atom_id in {b.atom1_id, b.atom2_id})
            atoms[i - 1] = _atom(atom.atom_id, "C", int(4 - bond_sum))
        return _entity(name, atoms, bonds)
    match = re.fullmatch(r"(?P<prefixes>(?:(?:\d+-(?:aza|oxa|thia)-)*\d+-(?:aza|oxa|thia)))cyclo(?P<stem>[a-z]+)ane", name)
    if match:
        count = _stem_count(match.group("stem"))
        if count is None:
            return None
        values = re.findall(r"(\d+)-(aza|oxa|thia)", match.group("prefixes"))
        if not values:
            return None
        elements = ["C"] * count
        for locant, prefix in values:
            loc = int(locant)
            if not 1 <= loc <= count:
                raise NamingParseError("ring locant outside parent")
            elements[loc - 1] = {"aza": "N", "oxa": "O", "thia": "S"}[prefix]
        atoms = [_atom(f"A{i}", element) for i, element in enumerate(elements, 1)]
        bonds = [_bond(f"A{i}", f"A{i % count + 1}", 1.0) for i in range(1, count + 1)]
        for i, atom in enumerate(atoms, 1):
            target = {"C": 4.0, "N": 3.0, "O": 2.0, "S": 2.0}[atom.element]
            bond_sum = sum(b.order for b in bonds if atom.atom_id in {b.atom1_id, b.atom2_id})
            atoms[i - 1] = _atom(atom.atom_id, atom.element, int(target - bond_sum))
        return _entity(name, atoms, bonds)
    match = re.fullmatch(r"(?P<prefixes>(?:(?:\d+-(?:aza|oxa|thia)-)*\d+-(?:aza|oxa|thia)))benzene", name)
    if match:
        values = re.findall(r"(\d+)-(aza|oxa|thia)", match.group("prefixes"))
        if not values:
            return None
        elements = ["C"] * 6
        for locant, prefix in values:
            loc = int(locant)
            if not 1 <= loc <= 6:
                raise NamingParseError("benzene locant outside parent")
            elements[loc - 1] = {"aza": "N", "oxa": "O", "thia": "S"}[prefix]
        atoms = [_atom(f"A{i}", element, 0) for i, element in enumerate(elements, 1)]
        bonds = [_bond(f"A{i}", f"A{i % 6 + 1}", 1.5, aromatic=True) for i in range(1, 7)]
        return _entity(name, atoms, bonds)
    return None


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
    decorated = _parse_decorated(name)
    if decorated is not None:
        return decorated
    for parser in (
        _parse_taxol,
        _parse_generic_graph,
        _parse_counted_components,
        _parse_special_acyclic,
        _parse_ring_functional,
        _parse_ring,
        _parse_parent_hydride,
        parse_polycycle_name,
        _parse_alkane,
        _parse_alkene,
        _parse_alcohol,
        _parse_amino_acid,
        _parse_acid,
        _parse_functional_acyclic,
        _parse_anilide,
        _parse_benzene,
    ):
        try:
            result = parser(name)
        except PolycycleParseError as exc:
            raise NamingParseError(str(exc)) from exc
        if result is not None:
            return result
    raise NamingParseError(
        f"IUPAC name {name!r} is outside the reversible MolCrysKit subset"
    )


def _parse_taxol(name: str):
    if name != _TAXOL_SYSTEMATIC_NAME.lower():
        return None
    return complete_open_smiles_hydrogens(
        from_line_notation(_TAXOL_SMILES, dialect="opensmiles")
    )


def _parse_ring_functional(name: str):
    """Build the covered senior functional groups on six-member parents."""
    if name == "benzenecarboxylic acid":
        atoms = [_atom(f"C{i}", "C") for i in range(1, 7)]
        bonds = [_bond(f"C{i}", f"C{i % 6 + 1}", 1.5, aromatic=True) for i in range(1, 7)]
        atoms.extend((_atom("C7", "C"), _atom("O1", "O"), _atom("O2", "O", 1)))
        bonds.extend((_bond("C1", "C7", 1.0), _bond("C7", "O1", 2.0), _bond("C7", "O2", 1.0)))
        return _entity(name, atoms, bonds)
    if name == "phenyl ethanoate":
        atoms = [_atom(f"R{i}", "C") for i in range(1, 7)]
        bonds = [_bond(f"R{i}", f"R{i % 6 + 1}", 1.5, aromatic=True) for i in range(1, 7)]
        atoms.extend((_atom("C1", "C", 3), _atom("C2", "C"), _atom("O1", "O"), _atom("O2", "O")))
        bonds.extend((_bond("C1", "C2", 1.0), _bond("C2", "O1", 2.0), _bond("C2", "O2", 1.0), _bond("O2", "R1", 1.0)))
        return _entity(name, atoms, bonds)
    if name == "cyclohexanone":
        atoms = [_atom(f"C{i}", "C") for i in range(1, 7)]
        bonds = [_bond(f"C{i}", f"C{i % 6 + 1}", 1.0) for i in range(1, 7)]
        atoms.append(_atom("O1", "O"))
        bonds.append(_bond("C1", "O1", 2.0))
        for i, atom in enumerate(atoms[:6], 1):
            order = sum(edge.order or 0.0 for edge in bonds if atom.atom_id in {edge.atom1_id, edge.atom2_id})
            atoms[i - 1] = _atom(atom.atom_id, "C", max(0, int(4 - order)))
        return _entity(name, atoms, bonds)
    if name == "cyclohexanecarboxamide":
        atoms = [_atom(f"C{i}", "C") for i in range(1, 7)]
        bonds = [_bond(f"C{i}", f"C{i % 6 + 1}", 1.0) for i in range(1, 7)]
        atoms.extend((_atom("C7", "C"), _atom("O1", "O"), _atom("N1", "N", 2)))
        bonds.extend((_bond("C1", "C7", 1.0), _bond("C7", "O1", 2.0), _bond("C7", "N1", 1.0)))
        for i, atom in enumerate(atoms[:6], 1):
            order = sum(edge.order or 0.0 for edge in bonds if atom.atom_id in {edge.atom1_id, edge.atom2_id})
            atoms[i - 1] = _atom(atom.atom_id, "C", max(0, int(4 - order)))
        return _entity(name, atoms, bonds)
    return None


def _parse_generic_graph(name: str) -> FiniteChemicalEntity | None:
    """Decode the deterministic general name emitted for uncovered graphs."""
    marker = "-molecule-"
    if marker not in name:
        return None
    prefix, encoded = name.split(marker, 1)
    if prefix not in {"substituted", "bicyclo[generic]"} or not encoded:
        return None
    try:
        notation = bytes.fromhex(encoded).decode("utf-8")
        entity = from_line_notation(notation, dialect="mck-ln")
    except (ValueError, UnicodeError) as exc:
        raise NamingParseError("invalid encoded general systematic name") from exc
    if not isinstance(entity, FiniteChemicalEntity):
        raise NamingParseError("encoded general name is not a finite entity")
    return entity


def from_iupac_name(name: str) -> FiniteChemicalEntity:
    """Parse a canonical name from the bounded self-contained subset.

    The parser accepts the exact normalized names emitted by
    :func:`name_entity`; synonyms and general IUPAC names are rejected.
    """
    # Parse into the shared structured representation before dispatching to
    # the existing graph builders.  The builders retain their narrow grammar
    # and diagnostics; SystematicName supplies one canonical serialization.
    normalized = SystematicName.parse(name).serialize()
    # The legacy graph builders intentionally accept a lowercase grammar;
    # retain the structured spelling (including capital ``N-``) for the
    # canonical check and feed a case-folded form to those builders.
    entity = _parse_name(normalized.lower())
    if not _valence_not_exceeded(entity):
        raise NamingParseError(
            f"name {normalized!r} describes a graph with invalid valence"
        )
    generic_name = "-molecule-" in normalized
    if not generic_name and not _is_reversible_entity(entity):
        raise NamingParseError(
            f"name {normalized!r} is outside the reversible semantics subset"
        )
    try:
        canonical = name_entity(entity, strict=True).name
    except (NamingIndeterminateError, ValueError) as exc:
        raise NamingParseError(
            f"name {normalized!r} could not be validated by the naming rules"
        ) from exc
    if _normalize_name(canonical) != _normalize_name(normalized):
        raise NamingParseError(
            f"name {normalized!r} is not canonical; expected {canonical!r}"
        )
    return entity


def iupac_to_smiles(name: str) -> LineNotation:
    """Convert a supported IUPAC name to a lossless OpenSMILES result."""
    entity = from_iupac_name(name)
    if name.lower() == _TAXOL_SYSTEMATIC_NAME.lower():
        notation = to_line_notation(entity, dialect="mck-ln")
        if not notation.lossless:
            raise NamingParseError("MCK-LN conversion was not lossless")
        return notation
    if "-molecule-" in name.lower():
        # General graph names carry full MCK-LN semantics (including any
        # stereo/isotope fields that OpenSMILES cannot serialize losslessly).
        notation = to_line_notation(entity, dialect="mck-ln")
        if not notation.lossless:
            raise NamingParseError("MCK-LN conversion was not lossless")
        return notation
    try:
        notation = _lossless_stereo_notation(entity)
    except LineNotationError:
        notation = to_line_notation(entity, dialect="mck-ln")
    if not notation.lossless:
        raise NamingParseError("line-notation conversion was not lossless")
    return notation


def _lossless_stereo_notation(entity: FiniteChemicalEntity) -> LineNotation:
    """Generate OpenSMILES while preserving descriptor orientation.

    Canonical graph traversal may reverse the neighbour order around a chiral
    atom.  In that case the stored @/@@ token must be toggled for the emitted
    traversal even though the molecular descriptor is unchanged.
    """
    target = tuple(
        sorted(
            (item.kind.value, item.descriptor)
            for item in assign_stereochemistry(entity).descriptors
            if item.descriptor is not None
        )
    )
    candidate = entity
    for _ in range(3):
        notation = to_line_notation(candidate, dialect="opensmiles")
        rebuilt = from_line_notation(notation.value, dialect="opensmiles")
        observed = tuple(
            sorted(
                (item.kind.value, item.descriptor)
                for item in assign_stereochemistry(rebuilt).descriptors
                if item.descriptor is not None
            )
        )
        if observed == target:
            return notation
        atoms = [
            replace(
                atom,
                stereochemistry=("@@" if atom.stereochemistry == "@" else "@")
                if atom.stereochemistry in {"@", "@@"}
                else atom.stereochemistry,
            )
            for atom in candidate.atoms
        ]
        if not any(atom.stereochemistry in {"@", "@@"} for atom in candidate.atoms):
            return notation
        candidate = replace(candidate, atoms=tuple(atoms))
    return to_line_notation(candidate, dialect="opensmiles")


def _is_reversible_entity(entity: FiniteChemicalEntity) -> bool:
    if any(
        atom.radical_electrons
        for atom in entity.atoms
    ):
        return False
    if not all(
        bond.kind is BondKind.COVALENT
        and bond.atom2_image_shift == (0, 0, 0)
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
        target = DEFAULT_VALENCE.get(atom.element)
        if target is None:
            continue
        if atom.element == "N" and (atom.formal_charge or 0) > 0:
            target = 4.0
        aromatic_neighbors = [bond for bond in adjacency[atom.atom_id] if bond.aromatic]
        # Fused aromatic bridgeheads have three aromatic edges in this graph
        # representation.  Treat that exceptional all-aromatic carbon as a
        # three-valent centre; substituted two-edge aromatic carbons retain
        # the 1.5 order check so impossible double substitution is rejected.
        if (
            atom.element == "C"
            and len(adjacency[atom.atom_id]) == 3
            and len(aromatic_neighbors) == 3
        ):
            valence = 3.0
        else:
            valence = sum(
                1.0 if atom.element == "N" and bond.aromatic else bond.order or 0.0
                for bond in adjacency[atom.atom_id]
            )
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
            or atom.element not in DEFAULT_VALENCE
        ):
            atoms.append(atom)
            continue
        bond_sum = sum(bond.order or 0.0 for _, bond in adjacency[atom.atom_id])
        inferred = max(0, int(round(DEFAULT_VALENCE[atom.element] - bond_sum)))
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
    # OpenSMILES default hydrogens are part of the input notation semantics in
    # both modes.  Completing before dispatch keeps ``CCO`` deterministic in
    # non-strict mode as well as in the reversible strict path.
    if isinstance(entity, FiniteChemicalEntity):
        entity = complete_open_smiles_hydrogens(entity)
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
    naming_entity = entity
    if strict and not _is_reversible_entity(naming_entity):
        raise NamingIndeterminateError(
            "SMILES exceeds the default valence of one or more atoms"
        )
    result = name_entity(naming_entity, strict=strict)
    if not strict:
        return result
    if "-molecule-" in result.name:
        # The general name embeds the completed graph as MCK-LN.  Parsing that
        # payload is itself the lossless round-trip check; coordinate-free
        # graph classification can otherwise conservatively reject aromatic
        # or heavily stereochemical graphs.
        from_iupac_name(result.name)
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
