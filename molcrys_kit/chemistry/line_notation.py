"""Self-contained linear notation parsing and generation.

OpenSMILES is used for finite covalent graphs that it can represent without
MolCrysKit-specific semantics. MCK-LN 1 is a transparent, versioned extension
for periodic edges, coordination semantics, polymers, multicomponent entities,
stable identities, and embeddings. Neither implementation calls an external
chemistry engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import quote, unquote

from .models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    ChemicalEntity,
    Embedding,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
    MulticomponentEntity,
    PeriodicChemicalEntity,
    PolymerChemicalEntity,
)


@dataclass(frozen=True)
class LineNotation:
    """One generated notation plus an explicit fidelity declaration."""

    value: str
    dialect: str
    version: str
    lossless: bool
    warnings: tuple[str, ...] = ()


class LineNotationError(ValueError):
    """Raised for invalid or unsupported linear notation."""


_ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
_AROMATIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}
_BOND_FROM_TOKEN = {
    "-": (1.0, False, None),
    "=": (2.0, False, None),
    "#": (3.0, False, None),
    ":": (1.5, True, None),
    "/": (1.0, False, "/"),
    "\\": (1.0, False, "\\"),
}
_BOND_TO_TOKEN = {1.0: "", 2.0: "=", 3.0: "#", 1.5: ":"}
_BRACKET_ATOM = re.compile(
    r"^(?P<isotope>\d+)?(?P<element>[A-Z][a-z]?|[bcnops])"
    r"(?P<stereo>@@?|@TH[12]|@AL[12]|@SP[123]|@TB\d+|@OH\d+)?"
    r"(?P<hydrogen>H\d*)?(?P<charge>\+\+?|--?|[+-]\d+)?(?::\d+)?$"
)


def to_line_notation(
    entity: ChemicalEntity,
    dialect: str = "auto",
) -> LineNotation:
    """Generate deterministic OpenSMILES or the lossless MCK-LN 1 extension."""
    requested = dialect.strip().lower().replace("_", "-")
    if requested == "auto":
        requested = (
            "opensmiles"
            if isinstance(entity, FiniteChemicalEntity)
            and _can_use_opensmiles(entity)
            else "mck-ln"
        )
    if requested in {"smiles", "opensmiles", "open-smiles"}:
        if not isinstance(entity, FiniteChemicalEntity):
            raise LineNotationError("OpenSMILES represents only finite entities")
        unsupported = _opensmiles_unsupported(entity)
        if unsupported:
            raise LineNotationError(
                "OpenSMILES would discard: " + ", ".join(unsupported)
            )
        return LineNotation(
            value=_write_opensmiles(entity),
            dialect="OpenSMILES",
            version="1.0",
            lossless=True,
        )
    if requested in {"mck", "mck-ln", "mck-ln1", "mck-ln-1"}:
        return LineNotation(
            value=_write_mck(entity),
            dialect="MCK-LN",
            version="1",
            lossless=True,
        )
    if requested in {"bigsmiles", "big-smiles"}:
        raise LineNotationError(
            "BigSMILES generation requires typed bonding descriptors; "
            "use MCK-LN 1 for the current polymer model"
        )
    raise LineNotationError(f"unknown line-notation dialect: {dialect}")


def from_line_notation(
    value: str,
    dialect: str = "auto",
) -> ChemicalEntity:
    """Parse OpenSMILES or MCK-LN 1 without an external chemistry engine."""
    text = value.strip()
    if not text:
        raise LineNotationError("line notation must not be empty")
    requested = dialect.strip().lower().replace("_", "-")
    if requested == "auto":
        requested = "mck-ln" if text.startswith("MCK-LN1|") else "opensmiles"
    if requested in {"mck", "mck-ln", "mck-ln1", "mck-ln-1"}:
        return _read_mck(text)
    if requested in {"smiles", "opensmiles", "open-smiles"}:
        return _read_opensmiles(text)
    raise LineNotationError(f"unknown line-notation dialect: {dialect}")


def _opensmiles_unsupported(entity: FiniteChemicalEntity) -> list[str]:
    unsupported = []
    if any(atom.oxidation_state is not None for atom in entity.atoms):
        unsupported.append("oxidation states")
    if any(atom.radical_electrons for atom in entity.atoms):
        unsupported.append("explicit radical-electron counts")
    if any(atom.explicit_hydrogens is not None for atom in entity.atoms):
        unsupported.append("explicit/implicit hydrogen distinction")
    if any(atom.stereochemistry is not None for atom in entity.atoms):
        unsupported.append("stored atom stereo tokens")
    if any(
        bond.kind not in {BondKind.COVALENT, BondKind.UNKNOWN}
        for bond in entity.bonds
    ):
        unsupported.append("non-covalent bond semantics")
    if any(bond.atom2_image_shift != (0, 0, 0) for bond in entity.bonds):
        unsupported.append("periodic image shifts")
    if any(bond.stereochemistry is not None for bond in entity.bonds):
        unsupported.append("stored bond stereo tokens")
    if any(
        bond.order is None or bond.order not in {1.0, 1.5, 2.0, 3.0}
        for bond in entity.bonds
    ):
        unsupported.append("unknown or nonstandard bond orders")
    return unsupported


def _can_use_opensmiles(entity: FiniteChemicalEntity) -> bool:
    return not _opensmiles_unsupported(entity)


def _atom_signature(atom: ChemicalAtom) -> tuple:
    return (
        atom.element,
        atom.isotope or 0,
        atom.formal_charge if atom.formal_charge is not None else 999,
        atom.implicit_hydrogens if atom.implicit_hydrogens is not None else 999,
    )


def _bond_signature(bond: ChemicalBond) -> tuple:
    return (bond.order or 0.0, bool(bond.aromatic))


def _canonical_colors(entity: FiniteChemicalEntity) -> dict[str, int]:
    adjacency = _adjacency(entity)
    initial = {atom.atom_id: _atom_signature(atom) for atom in entity.atoms}
    unique = {value: index for index, value in enumerate(sorted(set(initial.values())))}
    colors = {atom_id: unique[value] for atom_id, value in initial.items()}
    for _ in range(max(1, len(entity.atoms))):
        signatures = {
            atom_id: (
                colors[atom_id],
                tuple(
                    sorted(
                        (_bond_signature(bond), colors[neighbor])
                        for neighbor, bond in adjacency[atom_id]
                    )
                ),
            )
            for atom_id in colors
        }
        palette = {
            value: index for index, value in enumerate(sorted(set(signatures.values())))
        }
        refined = {atom_id: palette[value] for atom_id, value in signatures.items()}
        if refined == colors:
            break
        colors = refined
    return colors


def _write_opensmiles(entity: FiniteChemicalEntity) -> str:
    if not entity.atoms:
        raise LineNotationError("an empty entity has no OpenSMILES representation")
    adjacency = _adjacency(entity)
    colors = _canonical_colors(entity)
    components = _connected_components(tuple(adjacency), adjacency)
    rendered = []
    for component in components:
        candidates = [
            _render_component(entity, adjacency, colors, root)
            for root in component
        ]
        rendered.append(min(candidates))
    return ".".join(sorted(rendered))


def _render_component(entity, adjacency, colors, root: str) -> str:
    atom_by_id = {atom.atom_id: atom for atom in entity.atoms}
    parent: dict[str, str | None] = {root: None}
    parent_bond: dict[str, ChemicalBond] = {}
    order = []

    def neighbor_key(item):
        neighbor, bond = item
        return (
            colors[neighbor],
            _atom_signature(atom_by_id[neighbor]),
            _bond_signature(bond),
            len(adjacency[neighbor]),
        )

    def visit(atom_id: str) -> None:
        order.append(atom_id)
        for neighbor, bond in sorted(adjacency[atom_id], key=neighbor_key):
            if neighbor in parent:
                continue
            parent[neighbor] = atom_id
            parent_bond[neighbor] = bond
            visit(neighbor)

    visit(root)
    position = {atom_id: index for index, atom_id in enumerate(order)}
    tree_edges = {
        frozenset((atom_id, parent_id))
        for atom_id, parent_id in parent.items()
        if parent_id is not None
    }
    extra_edges = []
    for bond in entity.bonds:
        edge = frozenset((bond.atom1_id, bond.atom2_id))
        if edge not in tree_edges:
            endpoints = tuple(sorted(edge, key=position.__getitem__))
            extra_edges.append((position[endpoints[0]], position[endpoints[1]], endpoints, bond))
    ring_marks: dict[str, list[tuple[int, ChemicalBond, bool]]] = {
        atom_id: [] for atom_id in order
    }
    for ring_number, (_, _, endpoints, bond) in enumerate(sorted(extra_edges), start=1):
        ring_marks[endpoints[0]].append((ring_number, bond, True))
        ring_marks[endpoints[1]].append((ring_number, bond, False))

    children = {atom_id: [] for atom_id in order}
    for atom_id, parent_id in parent.items():
        if parent_id is not None:
            children[parent_id].append(atom_id)
    for atom_id in children:
        children[atom_id].sort(
            key=lambda child: (
                colors[child],
                _atom_signature(atom_by_id[child]),
                _bond_signature(parent_bond[child]),
                len(adjacency[child]),
            )
        )

    def render(atom_id: str) -> str:
        text = _smiles_atom(
            atom_by_id[atom_id],
            aromatic=any(bond.aromatic for _, bond in adjacency[atom_id]),
        )
        for number, bond, first in sorted(ring_marks[atom_id]):
            if first:
                text += _smiles_bond(bond)
            text += str(number) if number < 10 else f"%{number}"
        atom_children = children[atom_id]
        for child in atom_children[1:]:
            text += f"({_smiles_bond(parent_bond[child])}{render(child)})"
        if atom_children:
            child = atom_children[0]
            text += _smiles_bond(parent_bond[child]) + render(child)
        return text

    return render(root)


def _smiles_atom(atom: ChemicalAtom, *, aromatic: bool = False) -> str:
    simple = (
        atom.element in _ORGANIC
        and atom.isotope is None
        and atom.formal_charge in {None, 0}
        and atom.implicit_hydrogens is None
    )
    if simple:
        if aromatic and atom.element.lower() in _AROMATIC:
            return atom.element.lower()
        return atom.element
    isotope = "" if atom.isotope is None else str(atom.isotope)
    hydrogens = ""
    if atom.implicit_hydrogens:
        hydrogens = "H" if atom.implicit_hydrogens == 1 else f"H{atom.implicit_hydrogens}"
    charge = ""
    if atom.formal_charge:
        sign = "+" if atom.formal_charge > 0 else "-"
        magnitude = abs(atom.formal_charge)
        charge = sign if magnitude == 1 else f"{sign}{magnitude}"
    return f"[{isotope}{atom.element}{hydrogens}{charge}]"


def _smiles_bond(bond: ChemicalBond) -> str:
    if bond.aromatic:
        return ":"
    try:
        return _BOND_TO_TOKEN[float(bond.order)]
    except (KeyError, TypeError) as exc:
        raise LineNotationError("unsupported OpenSMILES bond order") from exc


def _adjacency(entity: FiniteChemicalEntity):
    adjacency = {atom.atom_id: [] for atom in entity.atoms}
    for bond in entity.bonds:
        adjacency[bond.atom1_id].append((bond.atom2_id, bond))
        adjacency[bond.atom2_id].append((bond.atom1_id, bond))
    return adjacency


def _connected_components(atom_ids, adjacency):
    pending = set(atom_ids)
    components = []
    while pending:
        start = min(pending)
        stack = [start]
        component = set()
        while stack:
            atom_id = stack.pop()
            if atom_id in component:
                continue
            component.add(atom_id)
            stack.extend(neighbor for neighbor, _ in adjacency[atom_id])
        pending.difference_update(component)
        components.append(tuple(component))
    return tuple(components)


def _read_opensmiles(text: str) -> FiniteChemicalEntity:
    atoms: list[ChemicalAtom] = []
    bonds: list[ChemicalBond] = []
    aromatic_atoms: dict[str, bool] = {}
    branch_stack: list[str | None] = []
    rings: dict[int, tuple[str, tuple[float, bool, str | None] | None]] = {}
    current: str | None = None
    pending_bond: tuple[float, bool, str | None] | None = None
    index = 0

    def add_atom(token: str, bracketed: bool) -> None:
        nonlocal current, pending_bond
        atom, aromatic = _parse_smiles_atom(token, bracketed, len(atoms))
        atoms.append(atom)
        aromatic_atoms[atom.atom_id] = aromatic
        if current is not None:
            bond_data = pending_bond
            if bond_data is None:
                bond_data = (
                    (1.5, True, None)
                    if aromatic_atoms[current] and aromatic
                    else (1.0, False, None)
                )
            bonds.append(_parsed_bond(current, atom.atom_id, bond_data))
        current = atom.atom_id
        pending_bond = None

    while index < len(text):
        char = text[index]
        if char.isspace():
            raise LineNotationError("whitespace is not valid inside OpenSMILES")
        if char in _BOND_FROM_TOKEN:
            if pending_bond is not None:
                raise LineNotationError("two consecutive bond tokens")
            pending_bond = _BOND_FROM_TOKEN[char]
            index += 1
            continue
        if char == "(":
            if current is None:
                raise LineNotationError("branch has no parent atom")
            branch_stack.append(current)
            index += 1
            continue
        if char == ")":
            if not branch_stack:
                raise LineNotationError("unmatched closing branch")
            current = branch_stack.pop()
            index += 1
            continue
        if char == ".":
            if pending_bond is not None:
                raise LineNotationError("bond token before disconnected component")
            current = None
            index += 1
            continue
        if char.isdigit() or char == "%":
            if current is None:
                raise LineNotationError("ring closure has no atom")
            if char == "%":
                digits = text[index + 1 : index + 3]
                if len(digits) != 2 or not digits.isdigit():
                    raise LineNotationError("percent ring labels require two digits")
                ring = int(digits)
                index += 3
            else:
                ring = int(char)
                index += 1
            if ring not in rings:
                rings[ring] = (current, pending_bond)
            else:
                other, first_bond = rings.pop(ring)
                bond_data = pending_bond or first_bond
                if pending_bond is not None and first_bond is not None and pending_bond != first_bond:
                    raise LineNotationError("conflicting ring-closure bond tokens")
                if bond_data is None:
                    bond_data = (
                        (1.5, True, None)
                        if aromatic_atoms[current] and aromatic_atoms[other]
                        else (1.0, False, None)
                    )
                bonds.append(_parsed_bond(other, current, bond_data))
            pending_bond = None
            continue
        if char == "[":
            end = text.find("]", index + 1)
            if end < 0:
                raise LineNotationError("unclosed bracket atom")
            add_atom(text[index + 1 : end], True)
            index = end + 1
            continue
        token = None
        if text.startswith("Cl", index) or text.startswith("Br", index):
            token = text[index : index + 2]
        elif char in "BCNOPSFIbcnops":
            token = char
        if token is None:
            raise LineNotationError(f"unsupported OpenSMILES token at position {index}")
        add_atom(token, False)
        index += len(token)

    if branch_stack:
        raise LineNotationError("unclosed branch")
    if rings:
        raise LineNotationError("unclosed ring label")
    if pending_bond is not None:
        raise LineNotationError("trailing bond token")
    if not atoms:
        raise LineNotationError("OpenSMILES contains no atoms")
    evidence = Evidence(EvidenceSource.LINE_NOTATION, "OpenSMILES parser")
    return FiniteChemicalEntity(
        entity_id="line:0",
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        net_charge=sum(atom.formal_charge or 0 for atom in atoms),
        status=InferenceStatus.EXPLICIT,
        evidence=(evidence,),
    )


def _parse_smiles_atom(token: str, bracketed: bool, index: int):
    if bracketed:
        match = _BRACKET_ATOM.fullmatch(token)
        if match is None:
            raise LineNotationError(f"unsupported bracket atom: [{token}]")
        raw_element = match.group("element")
        isotope = int(match.group("isotope")) if match.group("isotope") else None
        hydrogen = match.group("hydrogen")
        implicit_h = None
        if hydrogen:
            implicit_h = 1 if hydrogen == "H" else int(hydrogen[1:])
        charge = _parse_charge(match.group("charge"))
        stereo = match.group("stereo")
    else:
        raw_element = token
        isotope = None
        implicit_h = None
        charge = None
        stereo = None
    aromatic = raw_element in _AROMATIC
    element = _AROMATIC.get(raw_element, raw_element)
    return (
        ChemicalAtom(
            atom_id=f"line:a{index}",
            element=element,
            isotope=isotope,
            formal_charge=charge,
            implicit_hydrogens=implicit_h,
            stereochemistry=stereo,
            evidence=(
                Evidence(EvidenceSource.LINE_NOTATION, "OpenSMILES atom parser"),
            ),
        ),
        aromatic,
    )


def _parse_charge(token: str | None) -> int | None:
    if not token:
        return None
    sign = 1 if token[0] == "+" else -1
    suffix = token[1:]
    if suffix.isdigit():
        return sign * int(suffix)
    return sign * len(token)


def _parsed_bond(left, right, data):
    order, aromatic, stereo = data
    return ChemicalBond(
        atom1_id=left,
        atom2_id=right,
        order=order,
        kind=BondKind.COVALENT,
        aromatic=aromatic,
        stereochemistry=stereo,
        evidence=(Evidence(EvidenceSource.LINE_NOTATION, "OpenSMILES bond parser"),),
    )


def _encode(value) -> str:
    if value is None:
        return "_"
    return quote(str(value), safe="")


def _decode(value: str):
    return None if value == "_" else unquote(value)


def _write_mck(entity: ChemicalEntity) -> str:
    if isinstance(entity, FiniteChemicalEntity):
        fields = ["type=finite", f"id={_encode(entity.entity_id)}"]
        fields.extend(_mck_graph_fields(entity.atoms, entity.bonds, entity.embedding))
        fields.extend((f"charge={_encode(entity.net_charge)}", f"status={entity.status.value}"))
    elif isinstance(entity, PeriodicChemicalEntity):
        fields = ["type=periodic", f"id={_encode(entity.entity_id)}"]
        fields.extend(_mck_graph_fields(entity.atoms, entity.bonds, entity.embedding))
        generators = ";".join(",".join(map(str, vector)) for vector in entity.translation_generators)
        fields.extend(
            (
                f"rank={entity.periodic_rank}",
                f"generators={generators}",
                f"charge={_encode(entity.net_charge_per_repeat)}",
                f"status={entity.status.value}",
            )
        )
    elif isinstance(entity, PolymerChemicalEntity):
        repeats = ";".join(_encode(_write_mck(unit)) for unit in entity.repeat_units)
        connections = ";".join(_encode(value) for value in entity.connections)
        fields = [
            "type=polymer",
            f"id={_encode(entity.entity_id)}",
            f"repeats={repeats}",
            f"connections={connections}",
            f"status={entity.status.value}",
        ]
    elif isinstance(entity, MulticomponentEntity):
        components = ";".join(
            f"{count_}~{_encode(_write_mck(component))}"
            for component, count_ in entity.components
        )
        fields = [
            "type=multicomponent",
            f"id={_encode(entity.entity_id)}",
            f"components={components}",
            f"status={entity.status.value}",
        ]
    else:
        raise LineNotationError(f"unsupported chemical entity: {type(entity).__name__}")
    return "MCK-LN1|" + "|".join(fields)


def _mck_graph_fields(atoms, bonds, embedding):
    atom_index = {atom.atom_id: index for index, atom in enumerate(atoms)}
    atom_rows = []
    for atom in atoms:
        atom_rows.append(
            "~".join(
                _encode(value)
                for value in (
                    atom.atom_id,
                    atom.element,
                    atom.label,
                    atom.isotope,
                    atom.formal_charge,
                    atom.radical_electrons,
                    atom.explicit_hydrogens,
                    atom.implicit_hydrogens,
                    atom.oxidation_state,
                    atom.stereochemistry,
                )
            )
        )
    bond_rows = []
    for bond in bonds:
        bond_rows.append(
            "~".join(
                _encode(value)
                for value in (
                    atom_index[bond.atom1_id],
                    atom_index[bond.atom2_id],
                    bond.order,
                    bond.kind.value,
                    int(bond.aromatic),
                    ",".join(map(str, bond.atom2_image_shift)),
                    bond.stereochemistry,
                )
            )
        )
    fields = [f"atoms={';'.join(atom_rows)}", f"bonds={';'.join(bond_rows)}"]
    if embedding is None:
        fields.extend(("coord=_", "xyz=_"))
    else:
        coordinates = ";".join(
            f"{atom_index[atom_id]}~{','.join(format(value, '.17g') for value in vector)}"
            for atom_id, vector in embedding.coordinates_A
        )
        fields.extend((f"coord={_encode(embedding.coordinate_system)}", f"xyz={coordinates}"))
    return fields


def _read_mck(text: str) -> ChemicalEntity:
    if not text.startswith("MCK-LN1|"):
        raise LineNotationError("MCK-LN notation must begin with MCK-LN1|")
    fields = {}
    for field in text[len("MCK-LN1|") :].split("|"):
        if "=" not in field:
            raise LineNotationError("invalid MCK-LN field")
        key, value = field.split("=", 1)
        if key in fields:
            raise LineNotationError(f"duplicate MCK-LN field: {key}")
        fields[key] = value
    kind = fields.get("type")
    entity_id = _required_decoded(fields, "id")
    status = _status(fields.get("status", "explicit"))
    evidence = (Evidence(EvidenceSource.LINE_NOTATION, "MCK-LN 1 parser"),)
    if kind in {"finite", "periodic"}:
        atoms, bonds, embedding = _read_mck_graph(fields)
        if kind == "finite":
            return FiniteChemicalEntity(
                entity_id=entity_id,
                atoms=atoms,
                bonds=bonds,
                embedding=embedding,
                net_charge=_optional_int(fields.get("charge", "_")),
                status=status,
                evidence=evidence,
            )
        generators = tuple(
            tuple(int(value) for value in row.split(","))
            for row in fields.get("generators", "").split(";")
            if row
        )
        return PeriodicChemicalEntity(
            entity_id=entity_id,
            atoms=atoms,
            bonds=bonds,
            periodic_rank=int(fields["rank"]),
            translation_generators=generators,
            embedding=embedding,
            net_charge_per_repeat=_optional_int(fields.get("charge", "_")),
            status=status,
            evidence=evidence,
        )
    if kind == "polymer":
        repeats = tuple(
            _read_mck(_required_text(_decode(value), "repeat unit"))
            for value in fields.get("repeats", "").split(";")
            if value
        )
        if not all(isinstance(value, FiniteChemicalEntity) for value in repeats):
            raise LineNotationError("polymer repeat units must be finite")
        connections = tuple(
            _required_text(_decode(value), "connection")
            for value in fields.get("connections", "").split(";")
            if value
        )
        return PolymerChemicalEntity(
            entity_id=entity_id,
            repeat_units=repeats,
            connections=connections,
            status=status,
            evidence=evidence,
        )
    if kind == "multicomponent":
        components = []
        for row in fields.get("components", "").split(";"):
            if not row:
                continue
            raw_count, encoded = row.split("~", 1)
            components.append((
                _read_mck(_required_text(_decode(encoded), "component")),
                int(raw_count),
            ))
        return MulticomponentEntity(
            entity_id=entity_id,
            components=tuple(components),
            status=status,
            evidence=evidence,
        )
    raise LineNotationError(f"unsupported MCK-LN entity type: {kind}")


def _read_mck_graph(fields):
    atoms = []
    for row in fields.get("atoms", "").split(";"):
        if not row:
            continue
        values = [_decode(value) for value in row.split("~")]
        if len(values) != 10:
            raise LineNotationError("MCK-LN atom rows require ten fields")
        atoms.append(
            ChemicalAtom(
                atom_id=_required_text(values[0], "atom id"),
                element=_required_text(values[1], "element"),
                label=values[2],
                isotope=_int_or_none(values[3]),
                formal_charge=_int_or_none(values[4]),
                radical_electrons=int(values[5] or 0),
                explicit_hydrogens=_int_or_none(values[6]),
                implicit_hydrogens=_int_or_none(values[7]),
                oxidation_state=_int_or_none(values[8]),
                stereochemistry=values[9],
            )
        )
    bonds = []
    for row in fields.get("bonds", "").split(";"):
        if not row:
            continue
        values = [_decode(value) for value in row.split("~")]
        if len(values) != 7:
            raise LineNotationError("MCK-LN bond rows require seven fields")
        left = atoms[int(_required_text(values[0], "left atom index"))]
        right = atoms[int(_required_text(values[1], "right atom index"))]
        shift = tuple(int(value) for value in _required_text(values[5], "shift").split(","))
        bonds.append(
            ChemicalBond(
                atom1_id=left.atom_id,
                atom2_id=right.atom_id,
                order=_float_or_none(values[2]),
                kind=BondKind(_required_text(values[3], "bond kind")),
                aromatic=bool(int(values[4] or 0)),
                atom2_image_shift=shift,
                stereochemistry=values[6],
            )
        )
    embedding = None
    if fields.get("xyz", "_") != "_":
        coordinates = []
        for row in fields["xyz"].split(";"):
            atom_index, vector = row.split("~", 1)
            coordinates.append(
                (atoms[int(atom_index)].atom_id, tuple(float(value) for value in vector.split(",")))
            )
        embedding = Embedding(
            coordinates_A=tuple(coordinates),
            coordinate_system=_required_decoded(fields, "coord"),
        )
    return tuple(atoms), tuple(bonds), embedding


def _required_decoded(fields, key):
    if key not in fields:
        raise LineNotationError(f"missing MCK-LN field: {key}")
    return _required_text(_decode(fields[key]), key)


def _required_text(value, label):
    if value is None or value == "":
        raise LineNotationError(f"missing {label}")
    return value


def _int_or_none(value):
    return None if value is None else int(value)


def _float_or_none(value):
    return None if value is None else float(value)


def _optional_int(value):
    decoded = _decode(value)
    return None if decoded is None else int(decoded)


def _status(value):
    try:
        return InferenceStatus(value)
    except ValueError as exc:
        raise LineNotationError(f"invalid inference status: {value}") from exc


__all__ = [
    "LineNotation",
    "LineNotationError",
    "from_line_notation",
    "to_line_notation",
]
