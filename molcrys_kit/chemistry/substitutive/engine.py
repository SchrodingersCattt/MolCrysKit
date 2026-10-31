"""Composable substitutive grammar shared by forward and inverse naming.

Legacy recognizers still own the narrow parents they already spell. This
module is the coverage beyond that ceiling: multiplied suffixes, unsaturation
on functional parents, heteroatom linkers, radicals and ions, and finite
conjugates built from the same prefix operations. Graph construction for a
name uses the same locant and suffix rules as generation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re

from ..models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
)
from ..radicals import bracket_radical_electrons, is_bracket_atom, observed_bracket_valence
from .acyclic import (
    HALOGEN_PREFIX,
    _adjacency,
    _atom_map,
    _hcount,
    _heavy_adjacency,
    _is_double,
    _is_single,
    _prefix_string,
    alkane_stem,
)


_SIMPLE = (
    "acetamido",
    "amino",
    "bromo",
    "butyl",
    "chloro",
    "ethyl",
    "fluoro",
    "formamido",
    "hydroxy",
    "iodo",
    "methyl",
    "oxo",
    "phenoxy",
    "phenyl",
    "propyl",
    "sulfanyl",
)
_MULTIPLIERS = {
    1: "",
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
}
_MULTIPLIER_WORDS = {value: key for key, value in _MULTIPLIERS.items() if value}
_HW_SIZE = {"oxirane": 3, "oxetane": 4, "oxolane": 5, "oxane": 6, "oxepine": 7}
_HW_STEM = {3: "oxiran", 4: "oxetan", 5: "oxolan", 6: "oxan"}


@dataclass(frozen=True)
class _Pref:
    sort_name: str
    locant: int
    rendered: str


def _trace(*items: str) -> tuple[str, ...]:
    return items


def _result(name: str, locants: dict[str, int], *trace: str):
    return name, locants, trace


def _legacy_covers(entity: FiniteChemicalEntity) -> bool:
    from .acyclic import name_acyclic
    from .monocycle import name_monocycle
    from .polycycle import name_polycycle
    from .retained import name_anilide, name_benzene_family, name_hydride

    for recognizer in (
        name_acyclic,
        name_monocycle,
        name_hydride,
        name_polycycle,
        name_anilide,
        name_benzene_family,
    ):
        if recognizer(entity) is not None:
            return True
    return False


def _blocked(entity: FiniteChemicalEntity) -> bool:
    """Refuse graphs whose bracket valence is neither closed-shell nor a radical deficit."""
    atoms = _atom_map(entity)
    adjacency = _adjacency(entity)
    for atom in entity.atoms:
        if not is_bracket_atom(atom):
            continue
        bond_sum = sum(bond.order or 0.0 for _, bond in adjacency[atom.atom_id])
        inferred = bracket_radical_electrons(
            atom.element,
            atom.formal_charge,
            observed_bracket_valence(atom, bond_sum),
        )
        if inferred is None or atom.radical_electrons != inferred:
            return True
        if atoms[atom.atom_id].element == "H":
            continue
    return False


def _neighbors(adj, atoms, atom_id, blocked=None):
    return [
        neighbor
        for neighbor, _bond in adj[atom_id]
        if neighbor != blocked and atoms[neighbor].element != "H"
    ]


def _bond(adj, left, right):
    return next((bond for neighbor, bond in adj[left] if neighbor == right), None)


def _components(nodes, graph):
    pending = set(nodes)
    found = []
    while pending:
        start = min(pending)
        stack = [start]
        part = set()
        while stack:
            node = stack.pop()
            if node in part:
                continue
            part.add(node)
            stack.extend(neighbor for neighbor in graph.get(node, ()) if neighbor not in part)
        pending -= part
        found.append(part)
    return found


def _path(graph, start, end):
    queue = deque([(start, (start,))])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for neighbor in graph.get(current, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, (*path, neighbor)))
    return ()


def _carbon_graph(atoms, adj):
    carbons = {atom_id for atom_id, atom in atoms.items() if atom.element == "C"}
    return carbons, {
        atom_id: [neighbor for neighbor in _neighbors(adj, atoms, atom_id) if neighbor in carbons]
        for atom_id in carbons
    }


def _is_tree(component, graph):
    edges = sum(len([neighbor for neighbor in graph[node] if neighbor in component]) for node in component) // 2
    return edges == len(component) - 1


def _h(entity, atom_id):
    return _hcount(entity, atom_id)


def _alkyl_word(count: int) -> str:
    return {1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl"}.get(count) or f"{alkane_stem(count)}yl"


def _alkane_word(count: int) -> str:
    stem = alkane_stem(count)
    if stem is None:
        return ""
    return {1: "methane", 2: "ethane"}.get(count, f"{stem}ane")


def _multiplier(count: int) -> str:
    return _MULTIPLIERS.get(count, f"{count}-")


def _alphabet_key(text: str) -> str:
    """Alphabetical order of a compound prefix.

    P-14.5 cites a simple prefix under the parent substituent (``trihydroxy``
    under hydroxy). A parenthesized compound prefix is cited under the first
    letter of its complete name, multiplier included, so
    ``(2,4-dimethylphenoxy)`` belongs under ``d``. Locants are ignored.
    """
    bare = re.sub(r"[()\[\]]", "", str(text)).lower()
    return re.sub(r"^\d+(?:,\d+)*-", "", bare)


def _join_prefs(items: list[_Pref]) -> str:
    ordered = sorted(items, key=lambda item: (item.sort_name, item.locant, item.rendered))
    return "-".join(item.rendered for item in ordered)


def _group_simple(pairs: list[tuple[int, str]], *, omit_locants: bool = False) -> list[_Pref]:
    grouped: dict[str, list[int]] = {}
    for locant, name in pairs:
        grouped.setdefault(name, []).append(locant)
    prefs = []
    for name, locants in grouped.items():
        locants = tuple(sorted(locants))
        if omit_locants:
            rendered = f"{_multiplier(len(locants))}{name}"
        else:
            rendered = _prefix_string([(locant, name) for locant in locants])
        prefs.append(_Pref(name, locants[0], rendered))
    return prefs


def _rings(graph, limit=8):
    cycles = []
    seen = set()

    def walk(start, current, path):
        if len(cycles) > 64 or len(path) > limit:
            return
        for neighbor in graph.get(current, ()):
            if neighbor == start and len(path) >= 3:
                ring = path
                keys = []
                for direction in (ring, tuple(reversed(ring))):
                    keys.extend(direction[index:] + direction[:index] for index in range(len(direction)))
                key = min(keys)
                if key not in seen:
                    seen.add(key)
                    cycles.append(key)
            elif neighbor not in path and neighbor > start and len(path) < limit:
                walk(start, neighbor, (*path, neighbor))

    for start in sorted(graph):
        walk(start, start, (start,))
    return cycles


def _aromatic_six(entity, atoms, adj, root):
    carbons, graph = _carbon_graph(atoms, adj)
    if root not in carbons:
        return None
    for cycle in _rings({node: [n for n in graph[node]] for node in carbons}, limit=6):
        if len(cycle) != 6 or root not in cycle:
            continue
        bonds = []
        ok = True
        for index, atom_id in enumerate(cycle):
            bond = _bond(adj, atom_id, cycle[(index + 1) % 6])
            if bond is None or atoms[atom_id].element != "C":
                ok = False
                break
            bonds.append(bond)
        if not ok:
            continue
        aromatic = all(bond.aromatic or abs((bond.order or 0.0) - 1.5) < 1e-8 for bond in bonds)
        orders = tuple(float(bond.order or 0.0) for bond in bonds)
        alternating = (
            not aromatic
            and all(abs(order - 1.0) < 1e-8 or abs(order - 2.0) < 1e-8 for order in orders)
            and all(abs(orders[index] - orders[(index + 2) % 6]) < 1e-8 for index in range(6))
            and abs(orders[0] - orders[1]) > 1e-8
        )
        if aromatic or alternating:
            outside = []
            for atom_id in cycle:
                outside.extend(
                    neighbor
                    for neighbor in _neighbors(adj, atoms, atom_id)
                    if neighbor not in cycle
                )
            return cycle, outside
    return None


def _phenyl_text(entity, atoms, adj, root, blocked):
    found = _aromatic_six(entity, atoms, adj, root)
    if found is None:
        return None
    cycle, outside = found
    if any(neighbor != blocked for neighbor in outside):
        # Substituents are allowed; outside atoms must be simple on the ring.
        pass
    substituents = {}
    for atom_id in cycle:
        for neighbor in _neighbors(adj, atoms, atom_id):
            if neighbor in cycle or neighbor == blocked:
                continue
            atom = atoms[neighbor]
            bond = _bond(adj, atom_id, neighbor)
            word = None
            if atom.element == "O" and _is_single(bond) and _h(entity, neighbor) > 0 and not (atom.formal_charge or atom.radical_electrons):
                word = "hydroxy"
            elif atom.element == "C" and _h(entity, neighbor) == 3 and _neighbors(adj, atoms, neighbor) == [atom_id]:
                word = "methyl"
            elif atom.element in HALOGEN_PREFIX and _neighbors(adj, atoms, neighbor) == [atom_id]:
                word = HALOGEN_PREFIX[atom.element]
            else:
                return None
            substituents.setdefault(atom_id, []).append(word)
    best = None
    ring = list(cycle)
    # Rotate so root is locant 1, both directions, lowest substituent locants.
    for direction in (ring, list(reversed(ring))):
        if root not in direction:
            continue
        start = direction.index(root)
        ordered = direction[start:] + direction[:start]
        pairs = []
        for locant, atom_id in enumerate(ordered, 1):
            for word in substituents.get(atom_id, ()):
                pairs.append((locant, word))
        rendered = _prefix_string(pairs)
        text = f"{rendered}phenyl" if rendered else "phenyl"
        score = (tuple(sorted(locant for locant, _word in pairs)), text)
        if best is None or score < best[0]:
            best = (score, text)
    return None if best is None else best[1]


def _dienyl_text(entity, atoms, adj, root, blocked):
    carbons, graph = _carbon_graph(atoms, adj)
    for cycle in _rings(graph, limit=6):
        if len(cycle) != 6 or root not in cycle:
            continue
        if any(atoms[atom_id].element != "C" for atom_id in cycle):
            continue
        doubles = []
        ok = True
        for index, atom_id in enumerate(cycle):
            bond = _bond(adj, atom_id, cycle[(index + 1) % 6])
            if bond is None or bond.aromatic:
                ok = False
                break
            if _is_double(bond):
                doubles.append(index)
            elif not _is_single(bond):
                ok = False
                break
        if not ok or len(doubles) != 2:
            continue
        outside = [
            neighbor
            for atom_id in cycle
            for neighbor in _neighbors(adj, atoms, atom_id)
            if neighbor not in cycle
        ]
        if outside != [blocked]:
            continue
        ring = list(cycle)
        best = None
        for direction in (ring, list(reversed(ring))):
            start = direction.index(root)
            ordered = direction[start:] + direction[:start]
            ene = []
            for index in range(6):
                bond = _bond(adj, ordered[index], ordered[(index + 1) % 6])
                if _is_double(bond):
                    ene.append(index + 1)
            ene = tuple(sorted(ene))
            if best is None or ene < best[0]:
                best = (ene, ene)
        if best is None:
            continue
        locants = ",".join(str(item) for item in best[1])
        return f"cyclohexa-{locants}-dien-1-yl"
    return None


def _is_carboxyl_carbon(entity, atoms, adj, carbon):
    if atoms[carbon].element != "C":
        return None
    oxygens = [
        (neighbor, bond)
        for neighbor, bond in adj[carbon]
        if atoms[neighbor].element == "O"
    ]
    doubles = [neighbor for neighbor, bond in oxygens if _is_double(bond)]
    hydroxys = [
        neighbor
        for neighbor, bond in oxygens
        if _is_single(bond) and _h(entity, neighbor) > 0 and not (atoms[neighbor].formal_charge or atoms[neighbor].radical_electrons)
    ]
    if len(doubles) == 1 and len(hydroxys) == 1:
        return doubles[0], hydroxys[0]
    return None


def _is_methyl(entity, atoms, adj, carbon, parent):
    return (
        atoms[carbon].element == "C"
        and _h(entity, carbon) == 3
        and _neighbors(adj, atoms, carbon) == [parent]
    )


def _chain_from(atoms, adj, start, blocked):
    carbons, graph = _carbon_graph(atoms, adj)
    if start not in carbons:
        return None
    component = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == blocked or node in component or node not in carbons:
            continue
        component.add(node)
        stack.extend(
            neighbor
            for neighbor in graph[node]
            if neighbor not in component and neighbor != blocked
        )
    if not _is_tree(component, graph):
        return None
    # Unbranched chain beginning at start, not walking through blocked.
    chain = [start]
    previous = blocked if blocked in graph.get(start, ()) else None
    current = start
    guard = 0
    while guard < 16:
        guard += 1
        nxt = [
            neighbor
            for neighbor in graph[current]
            if neighbor != previous and neighbor in component
        ]
        if len(nxt) != 1:
            break
        previous, current = current, nxt[0]
        chain.append(current)
    if set(chain) != component:
        return None
    return tuple(chain)


def _fragment(entity, atoms, adj, root, blocked, depth=0):
    """Name the substituent whose first atom is ``root`` and whose parent is ``blocked``."""
    if depth > 6 or root not in atoms:
        return None
    atom = atoms[root]
    if atom.radical_electrons or (atom.formal_charge or 0):
        return None
    bond = _bond(adj, blocked, root) if blocked in adj else None
    others = _neighbors(adj, atoms, root, blocked)
    if atom.element in HALOGEN_PREFIX and not others:
        return ("simple", HALOGEN_PREFIX[atom.element])
    if atom.element == "O" and bond is not None and _is_double(bond):
        return ("simple", "oxo")
    if atom.element == "O" and bond is not None and _is_single(bond) and _h(entity, root) > 0 and not others:
        return ("simple", "hydroxy")
    if atom.element == "S" and _h(entity, root) > 0 and not others:
        return ("simple", "sulfanyl")
    if atom.element == "N" and _h(entity, root) == 2 and not others:
        return ("simple", "amino")
    if atom.element == "O" and len(others) == 1:
        phenyl = _phenyl_text(entity, atoms, adj, others[0], root)
        if phenyl is not None and _h(entity, root) == 0:
            text = phenyl[:-6] + "phenoxy" if phenyl.endswith("phenyl") else None
            if text is not None:
                return ("complex", text, text not in {"phenoxy"})
    if atom.element == "S" and len(others) == 1 and _h(entity, root) == 0:
        other = others[0]
        phenyl = _phenyl_text(entity, atoms, adj, other, root)
        if phenyl == "phenyl":
            return ("complex", "phenylsulfanyl", True)
        dienyl = _dienyl_text(entity, atoms, adj, other, root)
        if dienyl is not None:
            return ("complex", f"{dienyl}sulfanyl", True)
        carbamoyl = _carbamoyl_text(entity, atoms, adj, other, root)
        if carbamoyl is not None:
            return ("complex", f"[({carbamoyl})sulfanyl]", False)
    if atom.element == "N":
        named = _amino_text(entity, atoms, adj, root, blocked, depth)
        if named is not None:
            return named
    if atom.element == "C" and _is_methyl(entity, atoms, adj, root, blocked):
        return ("simple", "methyl")
    if atom.element == "C":
        phenyl = _phenyl_text(entity, atoms, adj, root, blocked)
        if phenyl is not None:
            return ("complex", phenyl, phenyl != "phenyl")
        chain = _chain_from(atoms, adj, root, blocked)
        if chain is not None and all(
            atoms[carbon].element == "C" and not _extra_on_alkyl(entity, atoms, adj, carbon, chain, blocked)
            for carbon in chain
        ):
            return ("simple", _alkyl_word(len(chain)))
        yl = _yl_text(entity, atoms, adj, root, blocked, depth)
        if yl is not None:
            return ("complex", yl, True)
    return None


def _extra_on_alkyl(entity, atoms, adj, carbon, chain, blocked):
    for neighbor in _neighbors(adj, atoms, carbon):
        if neighbor in chain or neighbor == blocked:
            continue
        if atoms[neighbor].element in HALOGEN_PREFIX and _neighbors(adj, atoms, neighbor) == [carbon]:
            continue
        return True
    return False


def _amino_text(entity, atoms, adj, nitrogen, blocked, depth):
    others = _neighbors(adj, atoms, nitrogen, blocked)
    if len(others) == 1 and _h(entity, nitrogen) == 1:
        acyl = _acyl_on_nitrogen(entity, atoms, adj, others[0], nitrogen)
        if acyl == 2:
            return ("simple", "acetamido")
        if acyl == 1:
            return ("simple", "formamido")
        carboxymethyl = _fragment(entity, atoms, adj, others[0], nitrogen, depth + 1)
        if carboxymethyl == ("simple", "carboxymethyl") or (
            isinstance(carboxymethyl, tuple) and carboxymethyl[0] == "complex" and carboxymethyl[1] == "carboxymethyl"
        ):
            return ("complex", "carboxymethylamino", True)
        yl = _yl_text(entity, atoms, adj, others[0], nitrogen, depth + 1)
        if yl is not None:
            return ("complex", f"[[{yl}]amino]", False)
    if not others and _h(entity, nitrogen) == 2:
        return ("simple", "amino")
    return None


def _acyl_on_nitrogen(entity, atoms, adj, carbon, nitrogen):
    if atoms[carbon].element != "C":
        return None
    oxy = [neighbor for neighbor, bond in adj[carbon] if atoms[neighbor].element == "O" and _is_double(bond)]
    if len(oxy) != 1:
        return None
    rest = [neighbor for neighbor in _neighbors(adj, atoms, carbon) if neighbor != nitrogen and neighbor not in oxy]
    if not rest and _h(entity, carbon) == 1:
        return 1
    if len(rest) == 1 and _is_methyl(entity, atoms, adj, rest[0], carbon):
        return 2
    return None


def _carbamoyl_text(entity, atoms, adj, carbon, blocked):
    if atoms[carbon].element != "C":
        return None
    oxy = [neighbor for neighbor, bond in adj[carbon] if atoms[neighbor].element == "O" and _is_double(bond)]
    nitrogens = [neighbor for neighbor in _neighbors(adj, atoms, carbon, blocked) if atoms[neighbor].element == "N"]
    if len(oxy) != 1 or len(nitrogens) != 1:
        return None
    if [neighbor for neighbor in _neighbors(adj, atoms, carbon, blocked) if neighbor not in oxy] != nitrogens:
        return None
    nitrogen = nitrogens[0]
    bits = []
    for neighbor in _neighbors(adj, atoms, nitrogen, carbon):
        if atoms[neighbor].element == "O" and _h(entity, neighbor) > 0 and not _neighbors(adj, atoms, neighbor, nitrogen):
            bits.append("hydroxy")
        elif _is_methyl(entity, atoms, adj, neighbor, nitrogen):
            bits.append("methyl")
        else:
            word = _alkyl_if_plain(entity, atoms, adj, neighbor, nitrogen)
            if word is None:
                return None
            bits.append(word)
    if _h(entity, nitrogen) == 2 and not bits:
        return "carbamoyl"
    bits = sorted(set(bits))
    body = "-".join(f"N-{bit}" if bit != "hydroxy" else "N-hydroxy" for bit in bits)
    # identical substituents use N,N-
    if bits == ["methyl"]:
        count = sum(_is_methyl(entity, atoms, adj, neighbor, nitrogen) for neighbor in _neighbors(adj, atoms, nitrogen, carbon))
        if count == 2:
            body = "N,N-dimethyl"
        elif count == 1:
            body = "N-methyl"
    if "hydroxy" in bits and "methyl" in bits:
        body = "N-hydroxy-N-methyl"
    return f"{body}carbamoyl"


def _alkyl_if_plain(entity, atoms, adj, root, blocked):
    chain = _chain_from(atoms, adj, root, blocked)
    if chain is None:
        return None
    if any(_extra_on_alkyl(entity, atoms, adj, carbon, chain, blocked) for carbon in chain):
        # halogens are allowed by _extra returning False when only halogens; True means foreign
        return None
    halogens = []
    for carbon in chain:
        for neighbor in _neighbors(adj, atoms, carbon):
            if neighbor in chain or neighbor == blocked:
                continue
            halogens.append(atoms[neighbor].element)
    if halogens:
        return None
    return _alkyl_word(len(chain))


def _yl_text(entity, atoms, adj, attachment, blocked, depth):
    if depth > 5 or atoms[attachment].element != "C":
        return None
    if _is_carboxymethyl(entity, atoms, adj, attachment, blocked):
        stereo = _local_stereo(entity, attachment)
        return f"({stereo})-carboxymethyl" if stereo else "carboxymethyl"
    carbons, graph = _carbon_graph(atoms, adj)
    if attachment not in carbons:
        return None
    component = set()
    stack = [attachment]
    while stack:
        node = stack.pop()
        if node == blocked or node in component or node not in carbons:
            continue
        component.add(node)
        stack.extend(
            neighbor
            for neighbor in graph[node]
            if neighbor not in component and neighbor != blocked
        )
    if not _is_tree(component, graph):
        return None
    nodes = list(component)
    paths = [(nodes[0],)] if len(nodes) == 1 else []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            path = _path(graph, left, right)
            if path and set(path) <= component:
                paths.append(path)
    best = None
    for path in paths:
        if attachment not in path:
            continue
        for ordered in (tuple(path), tuple(reversed(path))):
            numbering = {atom_id: locant for locant, atom_id in enumerate(ordered, 1)}
            simple = []
            complex_prefs = []
            ok = True
            consumed = set(ordered)
            for atom_id in ordered:
                for neighbor in _neighbors(adj, atoms, atom_id):
                    if neighbor in numbering or neighbor == blocked:
                        continue
                    classified = _fragment(entity, atoms, adj, neighbor, atom_id, depth + 1)
                    if classified is None:
                        ok = False
                        break
                    if classified[0] == "simple":
                        simple.append((numbering[atom_id], classified[1]))
                    else:
                        _kind, text, paren = classified
                        rendered = f"{numbering[atom_id]}-({text})" if paren and not str(text).startswith("[") else (
                            f"{numbering[atom_id]}-{text}" if str(text).startswith("[") or not paren else f"{numbering[atom_id]}-({text})"
                        )
                        sort_name = "amino" if str(text).endswith("amino") else str(text).replace("(", "").replace(")", "")
                        complex_prefs.append(_Pref(sort_name, numbering[atom_id], rendered))
                    consumed.add(neighbor)
                    _consume_fragment(atoms, adj, neighbor, atom_id, consumed)
            if not ok:
                continue
            # Double bonds inside the chain are ene locants.
            ene = []
            for index, atom_id in enumerate(ordered[:-1]):
                bond = _bond(adj, atom_id, ordered[index + 1])
                if bond is not None and _is_double(bond):
                    ene.append(index + 1)
            locant = numbering[attachment]
            prefix = _join_prefs([*_group_simple(simple), *complex_prefs])
            stem = alkane_stem(len(ordered))
            if stem is None:
                continue
            head = stem if not ene or stem.endswith("a") else (stem + "a" if len(ene) > 1 else stem)
            if len(ene) > 1:
                ene_text = f"{head}-{','.join(map(str, ene))}-diene"
                body = ene_text
            elif len(ene) == 1:
                body = f"{stem}-{ene[0]}-ene"
            else:
                body = f"{stem}ane" if len(ordered) > 1 or prefix else "methane"
            # propan-2-yl style: the yl locant replaces the final e when needed.
            if len(ordered) == 1:
                parent = "methyl"
            else:
                parent = f"{stem}an-{locant}-yl" if not ene else f"{body[:-1]}-{locant}-yl" if False else _yl_parent(stem, len(ordered), locant, ene)
            stereo = _local_stereo(entity, attachment)
            text = f"{prefix}{parent}" if prefix == "" or parent.startswith(prefix) else (
                f"{prefix}{parent}" if not prefix else f"{prefix}{'' if parent[:1].isdigit() else ''}{parent}"
            )
            if prefix and not parent[:1].isdigit():
                text = f"{prefix}{parent}" if prefix.endswith("-") else f"{prefix}-{parent}" if False else f"{prefix}{parent}"
            # prefix renderer already contains hyphens; concatenate directly when the
            # parent does not need an extra separator (2-amino + propan-2-yl).
            text = f"{prefix}{parent}" if prefix else parent
            if stereo:
                text = f"({locant}{stereo})-{text}"
            score = (
                -len(ordered),
                tuple(sorted(loc for loc, _name in simple)),
                locant,
                text,
            )
            if best is None or score < best[0]:
                best = (score, text, consumed, component)
    if best is None:
        return None
    _score, text, consumed, component = best
    if not component <= consumed and not component <= consumed | {blocked}:
        # Allow the attachment's blocked parent to sit outside the fragment.
        missing = component - consumed
        if missing:
            return None
    return text


def _yl_parent(stem: str, count: int, locant: int, ene: list[int]) -> str:
    if count == 1:
        return "methyl"
    if not ene:
        return f"{stem}an-{locant}-yl"
    if len(ene) == 1:
        return f"{stem}-{ene[0]}-en-{locant}-yl"
    head = stem if stem.endswith("a") else stem + "a"
    return f"{head}-{','.join(map(str, ene))}-{_multiplier(len(ene))}en-{locant}-yl"


def _is_carboxymethyl(entity, atoms, adj, attachment, blocked):
    if atoms[attachment].element != "C":
        return False
    others = _neighbors(adj, atoms, attachment, blocked)
    if len(others) != 1:
        return False
    return _is_carboxyl_carbon(entity, atoms, adj, others[0]) is not None and _h(entity, attachment) == 2


def _consume_fragment(atoms, adj, root, blocked, consumed):
    stack = [root]
    while stack:
        node = stack.pop()
        if node in consumed:
            continue
        consumed.add(node)
        stack.extend(neighbor for neighbor in _neighbors(adj, atoms, node, blocked) if neighbor not in consumed)


def _local_stereo(entity, atom_id):
    atom = next(item for item in entity.atoms if item.atom_id == atom_id)
    if atom.stereochemistry not in {"@", "@@"}:
        return None
    from ..stereo import assign_stereochemistry

    report = assign_stereochemistry(entity)
    found = next((item for item in report.descriptors if item.center_atom_id == atom_id), None)
    if found is None or found.descriptor not in {"R", "S"}:
        return None
    return found.descriptor


def _heavy_set(atoms):
    return {atom_id for atom_id, atom in atoms.items() if atom.element != "H"}


def _analyze_small(entity, atoms, adj):
    heavy = [atom for atom in entity.atoms if atom.element != "H"]
    charges = [atom.formal_charge or 0 for atom in entity.atoms]
    if len(entity.atoms) == 1 and entity.atoms[0].element == "H" and (entity.atoms[0].formal_charge or 0) == 1:
        return _result("hydron", {}, *_trace("Name the free proton as hydron."))
    if len(heavy) == 1:
        atom = heavy[0]
        if atom.element == "O" and atom.radical_electrons == 1 and _h(entity, atom.atom_id) == 1 and not (atom.formal_charge or 0):
            return _result("hydroxyl", {atom.atom_id: 1}, *_trace("Name the oxygen-centred radical as hydroxyl."))
        if atom.element == "Cl" and not atom.radical_electrons and not (atom.formal_charge or 0) and _h(entity, atom.atom_id) == 1:
            return _result("hydrogen chloride", {}, *_trace("Name hydrogen chloride from the neutral hydrogen halide."))
        if atom.element == "C" and atom.radical_electrons == 1 and _h(entity, atom.atom_id) == 3 and not (atom.formal_charge or 0):
            return _result("methyl", {atom.atom_id: 1}, *_trace("Name the carbon-centred radical as methyl."))
    if len(heavy) == 2 and {atom.element for atom in heavy} == {"O"}:
        left, right = heavy
        bond = _bond(adj, left.atom_id, right.atom_id)
        if bond is not None and _is_double(bond) and not any(atom.radical_electrons or (atom.formal_charge or 0) for atom in heavy):
            return _result("dioxygen", {}, *_trace("Name the dioxygen molecule."))
        if bond is not None and _is_single(bond):
            radicals = {atom.atom_id: atom.radical_electrons for atom in heavy}
            charge = {atom.atom_id: atom.formal_charge or 0 for atom in heavy}
            if sorted(radicals.values()) == [0, 1] and sorted(charge.values()) == [-1, 0] and sum(charges) == -1:
                return _result("superoxide", {}, *_trace("Name superoxide with its charge and radical electron separated."))
    return None


def _plain_halogen_alkyl(entity, atoms, adj, carbon, blocked=None):
    """Return a halogenated methyl label when the carbon carries only H and halogens."""
    if atoms[carbon].element != "C" or (atoms[carbon].formal_charge or 0):
        return None
    halogens = []
    for neighbor in _neighbors(adj, atoms, carbon):
        if neighbor == blocked:
            continue
        if atoms[neighbor].element not in HALOGEN_PREFIX or _neighbors(adj, atoms, neighbor) != [carbon]:
            return None
        halogens.append(HALOGEN_PREFIX[atoms[neighbor].element])
    if not halogens:
        return "methyl" if _h(entity, carbon) == 3 else None
    grouped: dict[str, int] = {}
    for name in halogens:
        grouped[name] = grouped.get(name, 0) + 1
    parts = [f"{_multiplier(count)}{name}" for name, count in sorted(grouped.items())]
    return "".join(parts) + "methyl"


def _analyze_peroxide(entity, atoms, adj):
    for oxygen, atom in atoms.items():
        if atom.element != "O":
            continue
        partners = _neighbors(adj, atoms, oxygen)
        if len(partners) != 1 or atoms[partners[0]].element != "O":
            continue
        inner = partners[0]
        inner_partners = _neighbors(adj, atoms, inner)
        if oxygen not in inner_partners or len(inner_partners) != 2:
            continue
        carbon = next(item for item in inner_partners if item != oxygen)
        if atoms[carbon].element != "C":
            continue
        terminal_h = _h(entity, oxygen)
        radical = atom.radical_electrons
        if atom.formal_charge or atoms[inner].formal_charge or atoms[inner].radical_electrons:
            continue
        if radical == 1 and terminal_h == 0:
            kind = "peroxyl"
        elif radical == 0 and terminal_h == 1:
            kind = "hydroperoxide"
        else:
            continue
        label = _plain_halogen_alkyl(entity, atoms, adj, carbon, blocked=inner)
        alkyl_chain = None
        if label is None:
            chain = _chain_from(atoms, adj, carbon, inner)
            if chain is None or any(_extra_on_alkyl(entity, atoms, adj, item, chain, inner) for item in chain):
                continue
            label = _alkyl_word(len(chain))
            alkyl_chain = chain
        represented = {oxygen, inner, carbon}
        if alkyl_chain:
            represented.update(alkyl_chain)
        represented.update(
            neighbor
            for neighbor in _neighbors(adj, atoms, carbon)
            if atoms[neighbor].element in HALOGEN_PREFIX
        )
        if represented != _heavy_set(atoms):
            continue
        if kind == "peroxyl":
            shown = (
                f"{label}peroxyl"
                if label in {"methyl", "ethyl", "propyl", "butyl"}
                else f"({label})peroxyl"
            )
            return _result(shown, {}, *_trace("Name the alkyl peroxyl radical."))
        return _result(f"{label} hydroperoxide", {}, *_trace("Name the alkyl hydroperoxide."))
    return None


def _analyze_isocyanate(entity, atoms, adj):
    for carbon, atom in atoms.items():
        if atom.element != "C":
            continue
        partners = _neighbors(adj, atoms, carbon)
        if len(partners) != 2:
            continue
        nitrogen = next((item for item in partners if atoms[item].element == "N"), None)
        oxygen = next((item for item in partners if atoms[item].element == "O"), None)
        if nitrogen is None or oxygen is None:
            continue
        if not _is_double(_bond(adj, carbon, nitrogen)) or not _is_double(_bond(adj, carbon, oxygen)):
            continue
        alkyls = [item for item in _neighbors(adj, atoms, nitrogen) if item != carbon]
        if len(alkyls) != 1:
            continue
        chain = _chain_from(atoms, adj, alkyls[0], nitrogen)
        if chain is None or any(_extra_on_alkyl(entity, atoms, adj, item, chain, nitrogen) for item in chain):
            continue
        represented = {carbon, nitrogen, oxygen, *chain}
        if represented != _heavy_set(atoms):
            continue
        return _result(
            f"{_alkyl_word(len(chain))} isocyanate",
            {},
            *_trace("Name the alkyl isocyanate from the N=C=O connectivity."),
        )
    return None


def _analyze_formamide(entity, atoms, adj):
    for carbon, atom in atoms.items():
        if atom.element != "C" or _h(entity, carbon) < 1:
            continue
        partners = _neighbors(adj, atoms, carbon)
        oxygen = [item for item in partners if atoms[item].element == "O" and _is_double(_bond(adj, carbon, item))]
        nitrogen = [item for item in partners if atoms[item].element == "N" and _is_single(_bond(adj, carbon, item))]
        if len(oxygen) != 1 or len(nitrogen) != 1 or len(partners) != 2:
            continue
        subs = []
        represented = {carbon, oxygen[0], nitrogen[0]}
        ok = True
        for neighbor in _neighbors(adj, atoms, nitrogen[0], carbon):
            if atoms[neighbor].element == "O" and _h(entity, neighbor) > 0:
                return None
            if atoms[neighbor].element == "C" and _h(entity, neighbor) == 2 and any(
                atoms[other].element == "O" and _h(entity, other) > 0
                for other in _neighbors(adj, atoms, neighbor, nitrogen[0])
            ):
                hydroxy = next(
                    other
                    for other in _neighbors(adj, atoms, neighbor, nitrogen[0])
                    if atoms[other].element == "O"
                )
                subs.append("hydroxymethyl")
                represented.update({neighbor, hydroxy})
            else:
                chain = _chain_from(atoms, adj, neighbor, nitrogen[0])
                if chain is None or any(_extra_on_alkyl(entity, atoms, adj, item, chain, nitrogen[0]) for item in chain):
                    ok = False
                    break
                subs.append(_alkyl_word(len(chain)))
                represented.update(chain)
        if not ok or represented != _heavy_set(atoms):
            continue
        if not subs:
            continue
        subs_sorted = sorted(subs)
        if subs_sorted == ["methyl", "methyl"]:
            body = "N,N-dimethyl"
        elif subs_sorted == ["ethyl", "ethyl"]:
            body = "N,N-diethyl"
        elif subs_sorted == ["hydroxymethyl", "methyl"]:
            body = "N-(hydroxymethyl)-N-methyl"
        else:
            rendered = []
            for name in subs_sorted:
                rendered.append(f"N-({name})" if name == "hydroxymethyl" else f"N-{name}")
            body = "-".join(rendered)
        return _result(f"{body}formamide", {}, *_trace("Name the N-substituted formamide."))
    return None


def _analyze_sulfur_link(entity, atoms, adj):
    for sulfur, atom in atoms.items():
        if atom.element != "S" or atom.radical_electrons or (atom.formal_charge or 0):
            continue
        carbons = [neighbor for neighbor in _neighbors(adj, atoms, sulfur) if atoms[neighbor].element == "C"]
        oxygens = [neighbor for neighbor in _neighbors(adj, atoms, sulfur) if atoms[neighbor].element == "O"]
        if len(carbons) != 2 or any(not _is_single(_bond(adj, sulfur, item)) for item in carbons):
            continue
        if any(not _is_double(_bond(adj, sulfur, item)) for item in oxygens):
            continue
        if len(oxygens) == 0:
            kind = "sulfanyl"
        elif len(oxygens) == 1:
            kind = "sulfinyl"
        elif len(oxygens) == 2:
            kind = "sulfonyl"
        else:
            continue
        chains = []
        ok = True
        for carbon in carbons:
            chain = _chain_from(atoms, adj, carbon, sulfur)
            if chain is None or any(_extra_on_alkyl(entity, atoms, adj, item, chain, sulfur) for item in chain):
                ok = False
                break
            chains.append(chain)
        if not ok:
            continue
        represented = {sulfur, *oxygens, *(atom_id for chain in chains for atom_id in chain)}
        if represented != _heavy_set(atoms):
            continue
        chains = sorted(chains, key=len)
        left, right = chains[0], chains[1]
        if kind == "sulfanyl":
            prefix = f"{_alkyl_word(len(left))}sulfanyl"
        else:
            prefix = f"{_alkane_word(len(left))}{kind}"
        parent = _alkane_word(len(right))
        if not prefix or not parent:
            continue
        return _result(
            f"({prefix}){parent}",
            {},
            *_trace("Name the symmetrical sulfur link by substitutive prefix."),
        )
    return None


def _senior_groups(entity, atoms, adj):
    groups = []
    for atom_id, atom in atoms.items():
        if atom.element != "C":
            continue
        carboxyl = _is_carboxyl_carbon(entity, atoms, adj, atom_id)
        if carboxyl is not None:
            groups.append(("acid", 80, atom_id, set(carboxyl)))
            continue
        oxy = [neighbor for neighbor, bond in adj[atom_id] if atoms[neighbor].element == "O" and _is_double(bond)]
        carbons = [neighbor for neighbor in _neighbors(adj, atoms, atom_id) if atoms[neighbor].element == "C"]
        if len(oxy) == 1 and _h(entity, atom_id) >= 1 and len(carbons) <= 1:
            groups.append(("aldehyde", 50, atom_id, {oxy[0]}))
        elif len(oxy) == 1 and len(carbons) == 2 and _h(entity, atom_id) == 0:
            groups.append(("ketone", 40, atom_id, {oxy[0]}))
    for atom_id, atom in atoms.items():
        if atom.element != "O" or atom.radical_electrons or (atom.formal_charge or 0):
            continue
        partners = _neighbors(adj, atoms, atom_id)
        if len(partners) == 1 and atoms[partners[0]].element == "C" and _h(entity, atom_id) > 0 and _is_single(_bond(adj, atom_id, partners[0])):
            carbon = partners[0]
            if _is_carboxyl_carbon(entity, atoms, adj, carbon) is None:
                groups.append(("alcohol", 30, carbon, {atom_id}))
    return groups


def _hydrogens_known(atom) -> bool:
    """Bracket omission means zero hydrogens; an unbracketed unset count is missing data.

    A radical atom built from a name has already been valence-completed, and a
    zero hydrogen count is stored as ``None`` so it matches OpenSMILES.
    """
    if atom.explicit_hydrogens is not None or atom.implicit_hydrogens is not None:
        return True
    if is_bracket_atom(atom):
        return True
    return bool(atom.radical_electrons) and any(
        evidence.source is EvidenceSource.IUPAC_NAME for evidence in atom.evidence
    )


def _analyze_carbon_radical(entity, atoms, adj):
    centres = [atom_id for atom_id, atom in atoms.items() if atom.element == "C" and atom.radical_electrons == 1]
    if len(centres) == 1 and not _hydrogens_known(atoms[centres[0]]):
        return None
    if len(centres) != 1:
        return None
    if any(atom.radical_electrons and atom.atom_id not in centres for atom in atoms.values()):
        return None
    carbon = centres[0]
    others = _neighbors(adj, atoms, carbon)
    halogens = []
    alkyl = []
    for neighbor in others:
        if atoms[neighbor].element in HALOGEN_PREFIX and _neighbors(adj, atoms, neighbor) == [carbon]:
            halogens.append(HALOGEN_PREFIX[atoms[neighbor].element])
        elif atoms[neighbor].element == "C":
            alkyl.append(neighbor)
        else:
            return None
    if len(alkyl) > 1:
        return None
    represented = {carbon, *others}
    if alkyl:
        chain = _chain_from(atoms, adj, alkyl[0], carbon)
        if chain is None or any(_extra_on_alkyl(entity, atoms, adj, item, chain, carbon) for item in chain):
            return None
        represented.update(chain)
        if halogens:
            return None
        label = _alkyl_word(len(chain) + 1)
    else:
        grouped: dict[str, int] = {}
        for name in halogens:
            grouped[name] = grouped.get(name, 0) + 1
        label = "".join(f"{_multiplier(count)}{name}" for name, count in sorted(grouped.items())) + "methyl"
    if represented != _heavy_set(atoms):
        return None
    return _result(label, {carbon: 1}, *_trace("Name the carbon-centred radical without adding hydrogen."))


def _analyze_chain(entity, atoms, adj):
    if any(atom.radical_electrons for atom in atoms.values()):
        return None
    groups = _senior_groups(entity, atoms, adj)
    if not groups and not any(atom.element in HALOGEN_PREFIX for atom in atoms.values()):
        return None
    carbons, graph = _carbon_graph(atoms, adj)
    components = _components(carbons, graph)
    trees = [component for component in components if _is_tree(component, graph)]
    if not trees:
        return None
    senior = max(item[1] for item in groups) if groups else 0
    principal_kind = next(kind for kind, rank, _anchor, _extra in groups if rank == senior) if groups else "halide"
    anchors = {anchor for kind, rank, anchor, _extra in groups if rank == senior and kind == principal_kind}
    candidates = []
    for component in trees:
        nodes = list(component)
        paths = [(nodes[0],)] if len(nodes) == 1 else []
        for index, left in enumerate(nodes):
            for right in nodes[index + 1 :]:
                path = _path({node: [n for n in graph[node] if n in component] for node in component}, left, right)
                if path:
                    paths.append(path)
        ranked = []
        for path in paths:
            hits = [anchor for anchor in anchors if anchor in path and anchor in component]
            target_hits = [anchor for anchor in anchors if anchor in component]
            ranked.append((len(hits), len(path), path, len(target_hits)))
        if not ranked:
            continue
        best_hits = max(item[0] for item in ranked)
        best_length = max(item[1] for item in ranked if item[0] == best_hits)
        for _hits, _length, path, target_count in ranked:
            if _hits != best_hits or _length != best_length or (target_count and _hits != target_count):
                continue
            for ordered in (tuple(path), tuple(reversed(path))):
                if anchors and anchors <= set(ordered):
                    pass
                numbering = {atom_id: locant for locant, atom_id in enumerate(ordered, 1)}
                principal_locants = tuple(sorted(numbering[anchor] for anchor in anchors if anchor in numbering))
                if anchors and len(principal_locants) != len([anchor for anchor in anchors if anchor in component]):
                    continue
                ene = []
                for index, atom_id in enumerate(ordered[:-1]):
                    bond = _bond(adj, atom_id, ordered[index + 1])
                    if bond is not None and _is_double(bond):
                        ene.append(numbering[atom_id])
                simple = []
                complex_prefs = []
                consumed = set(ordered)
                for kind, _rank, anchor, extra in groups:
                    if anchor in numbering and kind == principal_kind:
                        consumed.update(extra)
                ok = True
                for atom_id in ordered:
                    for neighbor in _neighbors(adj, atoms, atom_id):
                        if neighbor in consumed or neighbor in numbering:
                            continue
                        # Principal-group atoms are already consumed.
                        classified = _fragment(entity, atoms, adj, neighbor, atom_id)
                        if classified is None:
                            ok = False
                            break
                        if classified[0] == "simple":
                            simple.append((numbering[atom_id], classified[1]))
                        else:
                            text = classified[1]
                            paren = classified[2]
                            if str(text).startswith("["):
                                rendered = f"{numbering[atom_id]}-{text}"
                            elif paren:
                                rendered = f"{numbering[atom_id]}-({text})"
                            else:
                                rendered = f"{numbering[atom_id]}-{text}"
                            bare = re.sub(r"[()\[\]]", "", str(text)).lower()
                            sort_name = "amino" if bare.endswith("amino") else bare
                            complex_prefs.append(_Pref(sort_name, numbering[atom_id], rendered))
                        stack = [neighbor]
                        seen = set()
                        while stack:
                            node = stack.pop()
                            if node in seen:
                                continue
                            seen.add(node)
                            consumed.add(node)
                            stack.extend(item for item in _neighbors(adj, atoms, node) if item not in numbering)
                if not ok:
                    continue
                if not _heavy_set(atoms) <= consumed:
                    continue
                # Principal groups expressed as suffixes are not also prefixes.
                if principal_kind == "alcohol":
                    simple = [(locant, name) for locant, name in simple if name != "hydroxy" or locant not in principal_locants]
                if principal_kind == "ketone":
                    simple = [(locant, name) for locant, name in simple if name != "oxo" or locant not in principal_locants]
                if principal_kind == "aldehyde":
                    simple = [(locant, name) for locant, name in simple if not (name == "oxo" and locant in principal_locants)]
                stem = alkane_stem(len(ordered))
                if stem is None:
                    continue
                omit = len(ordered) == 1 or (
                    len(ordered) == 2
                    and len(simple) == 1
                    and not complex_prefs
                    and simple[0][0] == 1
                )
                prefix = _join_prefs([*_group_simple(simple, omit_locants=omit), *complex_prefs])
                parent = _render_parent(stem, len(ordered), principal_kind, principal_locants, tuple(ene))
                if parent == "ethanol" and prefix:
                    parent = "ethan-1-ol"
                if parent is None:
                    continue
                name = f"{prefix}{parent}" if prefix else parent
                score = (
                    -len(principal_locants),
                    -len(ordered),
                    -len(ene),
                    principal_locants,
                    tuple(ene),
                    tuple(sorted(locant for locant, _name in simple)),
                    name,
                )
                candidates.append((score, name, numbering))
    if not candidates:
        return None
    _score, name, numbering = min(candidates, key=lambda item: item[0])
    return _result(name, numbering, *_trace(
        "Select the senior characteristic group and the principal chain.",
        "Number the chain by suffix locants, then unsaturation, then prefixes.",
    ))


def _render_parent(stem, count, kind, locants, ene):
    locants = tuple(locants)
    ene = tuple(ene)
    if kind == "halide":
        return _alkane_word(count)
    if kind == "acid":
        if len(locants) == 2 and set(locants) == {1, count}:
            if not ene:
                return f"{stem}anedioic acid"
            if len(ene) == 1:
                return f"{stem}-{ene[0]}-enedioic acid"
            head = stem if stem.endswith("a") else stem + "a"
            return f"{head}-{','.join(map(str, ene))}-diene" + "dioic acid" if len(ene) == 2 else None
        if len(locants) == 1 and locants[0] == 1:
            if not ene:
                return f"{stem}anoic acid"
            if len(ene) == 1:
                return f"{stem}-{ene[0]}-enoic acid"
            head = stem if stem.endswith("a") else stem + "a"
            ending = {2: "dienoic acid", 3: "trienoic acid"}.get(len(ene))
            if ending is None:
                return None
            return f"{head}-{','.join(map(str, ene))}-{ending}"
        return None
    if kind == "aldehyde":
        if len(locants) == 1 and locants[0] == 1 and not ene:
            return "methanal" if count == 1 else f"{stem}anal"
        if len(locants) == 2 and set(locants) == {1, count} and not ene:
            return f"{stem}anedial"
        return None
    if kind == "ketone":
        if ene:
            return None
        if len(locants) == 1:
            return f"{stem}an-{locants[0]}-one"
        return f"{stem}ane-{','.join(map(str, locants))}-{_multiplier(len(locants))}one"
    if kind == "alcohol":
        if ene:
            return None
        if len(locants) == 1:
            if count == 1:
                return "methanol"
            if count == 2 and locants[0] == 1:
                return "ethanol"
            return f"{stem}an-{locants[0]}-ol"
        return f"{stem}ane-{','.join(map(str, locants))}-{_multiplier(len(locants))}ol"
    return None


def _analyze_rings(entity, atoms, adj):
    sulfate = _analyze_sulfate(entity, atoms, adj)
    if sulfate is not None:
        return sulfate
    phenoxy = _analyze_phenoxide(entity, atoms, adj)
    if phenoxy is not None:
        return phenoxy
    hw = _analyze_heterocycle(entity, atoms, adj)
    if hw is not None:
        return hw
    carbocycle = _analyze_carbocycle(entity, atoms, adj)
    if carbocycle is not None:
        return carbocycle
    return None


def _analyze_sulfate(entity, atoms, adj):
    sulfurs = [atom_id for atom_id, atom in atoms.items() if atom.element == "S"]
    if len(sulfurs) != 1:
        return None
    sulfur = sulfurs[0]
    doubles = [neighbor for neighbor, bond in adj[sulfur] if atoms[neighbor].element == "O" and _is_double(bond)]
    singles = [neighbor for neighbor, bond in adj[sulfur] if atoms[neighbor].element == "O" and _is_single(bond)]
    if len(doubles) != 2 or len(singles) != 2:
        return None
    charged = [item for item in singles if (atoms[item].formal_charge or 0) == -1 and not atoms[item].radical_electrons]
    linkers = [item for item in singles if item not in charged]
    if len(charged) != 1 or len(linkers) != 1 or _h(entity, linkers[0]) != 0:
        return None
    aryl = _neighbors(adj, atoms, linkers[0], sulfur)
    if len(aryl) != 1:
        return None
    phenyl = _phenyl_text(entity, atoms, adj, aryl[0], linkers[0])
    if phenyl is None:
        return None
    represented = {sulfur, *doubles, *singles}
    _consume_fragment(atoms, adj, aryl[0], linkers[0], represented)
    represented.add(linkers[0])
    if represented != _heavy_set(atoms):
        return None
    if sum(atom.formal_charge or 0 for atom in entity.atoms) != -1:
        return None
    return _result(f"{phenyl} sulfate", {}, *_trace("Name the aryl sulfate monoester anion."))


def _analyze_phenoxide(entity, atoms, adj):
    aromatic = None
    for atom_id, atom in atoms.items():
        if atom.element == "C":
            found = _aromatic_six(entity, atoms, adj, atom_id)
            if found is not None:
                aromatic = found[0]
                break
    if aromatic is None:
        return None
    ring = set(aromatic)
    oxygens = []
    for atom_id in aromatic:
        for neighbor in _neighbors(adj, atoms, atom_id):
            if neighbor in ring:
                continue
            atom = atoms[neighbor]
            if atom.element != "O" or not _is_single(_bond(adj, atom_id, neighbor)):
                return None
            if _neighbors(adj, atoms, neighbor, atom_id):
                return None
            oxygens.append((atom_id, neighbor, atom))
    if not oxygens:
        return None
    if _heavy_set(atoms) != ring | {neighbor for _carbon, neighbor, _atom in oxygens}:
        return None
    radicals = [(carbon, oxygen) for carbon, oxygen, atom in oxygens if atom.radical_electrons == 1 and not (atom.formal_charge or 0)]
    anions = [(carbon, oxygen) for carbon, oxygen, atom in oxygens if (atom.formal_charge or 0) == -1 and atom.radical_electrons == 0]
    hydroxys = [(carbon, oxygen) for carbon, oxygen, atom in oxygens if _h(entity, oxygen) > 0 and not atom.radical_electrons and not (atom.formal_charge or 0)]
    if len(radicals) + len(anions) + len(hydroxys) != len(oxygens):
        return None
    if len(radicals) == 1 and not anions:
        return _phenoxy_name(entity, aromatic, radicals[0][0], hydroxys, "phenoxyl", "hydroxy")
    if len(anions) == 1 and not radicals:
        return _phenoxy_name(entity, aromatic, anions[0][0], hydroxys, "phenolate", "hydroxy")
    if len(radicals) == 1 and len(anions) == 1 and not hydroxys:
        return _phenoxy_name(entity, aromatic, radicals[0][0], anions, "phenoxyl", "oxido")
    if len(anions) >= 2 and not radicals and not hydroxys:
        return _benzene_olate(aromatic, [carbon for carbon, _oxygen in anions])
    return None


def _phenoxy_name(entity, ring, root, substituents, suffix, word):
    best = None
    for direction in (list(ring), list(reversed(ring))):
        start = direction.index(root)
        ordered = direction[start:] + direction[:start]
        numbering = {atom_id: locant for locant, atom_id in enumerate(ordered, 1)}
        locants = tuple(sorted(numbering[carbon] for carbon, _oxygen in substituents))
        if best is None or locants < best[0]:
            best = (locants, numbering)
    locants, numbering = best
    prefix = _prefix_string([(locant, word) for locant in locants])
    return _result(
        f"{prefix}{suffix}",
        numbering,
        *_trace("Number the phenolic parent from the senior oxygen substituent."),
    )


def _benzene_olate(ring, carbons):
    best = None
    for direction in (list(ring), list(reversed(ring))):
        for start in range(6):
            ordered = direction[start:] + direction[:start]
            if ordered[0] not in carbons:
                continue
            numbering = {atom_id: locant for locant, atom_id in enumerate(ordered, 1)}
            locants = tuple(sorted(numbering[carbon] for carbon in carbons))
            if best is None or locants < best[0]:
                best = (locants, numbering)
    if best is None:
        return None
    locants, numbering = best
    return _result(
        f"benzene-{','.join(map(str, locants))}-{_multiplier(len(locants))}olate",
        numbering,
        *_trace("Name the phenylene dianion as a benzene diolate."),
    )


def _unique_cycle(atoms, adj):
    carbons, graph = _carbon_graph(atoms, adj)
    # Include heteroatoms of degree >= 2 in the ring graph.
    nodes = {
        atom_id
        for atom_id, atom in atoms.items()
        if atom.element != "H" and len(_neighbors(adj, atoms, atom_id)) >= 2
    }
    ring_graph = {atom_id: [neighbor for neighbor in _neighbors(adj, atoms, atom_id) if neighbor in nodes] for atom_id in nodes}
    cycles = _rings(ring_graph, limit=8)
    return cycles


def _analyze_heterocycle(entity, atoms, adj):
    cycles = [cycle for cycle in _unique_cycle(atoms, adj) if any(atoms[atom_id].element == "O" for atom_id in cycle)]
    if len(cycles) != 1:
        return None
    cycle = cycles[0]
    if sum(atoms[atom_id].element == "O" for atom_id in cycle) != 1:
        return None
    if any(atoms[atom_id].element not in {"C", "O"} for atom_id in cycle):
        return None
    ring_bonds = [_bond(adj, cycle[index], cycle[(index + 1) % len(cycle)]) for index in range(len(cycle))]
    if any(bond is None for bond in ring_bonds):
        return None
    oxygen = next(atom_id for atom_id in cycle if atoms[atom_id].element == "O")
    doubles = [index for index, bond in enumerate(ring_bonds) if _is_double(bond)]
    if len(cycle) == 7 and len(doubles) == 3 and _heavy_set(atoms) == set(cycle):
        return _result("oxepine", {}, *_trace("Use the retained oxepine name for the unsaturated oxacycle."))
    if any(bond.aromatic or not _is_single(bond) for bond in ring_bonds):
        return None
    # Substituents on ring carbons.
    substituents = []
    for atom_id in cycle:
        if atom_id == oxygen:
            continue
        for neighbor in _neighbors(adj, atoms, atom_id):
            if neighbor in cycle:
                continue
            substituents.append((atom_id, neighbor))
    consumed = set(cycle)
    simple = []
    complex_prefs = []
    carboxyls = []
    carbonyls = []
    for atom_id, neighbor in substituents:
        carboxyl = _is_carboxyl_carbon(entity, atoms, adj, neighbor) if atoms[neighbor].element == "C" else None
        if carboxyl is not None and _neighbors(adj, atoms, neighbor) == [atom_id, *list(carboxyl)] or (
            carboxyl is not None and set(_neighbors(adj, atoms, neighbor)) <= {atom_id, *carboxyl}
        ):
            carboxyls.append(atom_id)
            consumed.update({neighbor, *carboxyl})
            continue
        if atoms[neighbor].element == "O" and _is_double(_bond(adj, atom_id, neighbor)):
            carbonyls.append(atom_id)
            consumed.add(neighbor)
            continue
        classified = _fragment(entity, atoms, adj, neighbor, atom_id)
        if classified is None:
            return None
        # Locants are assigned after numbering; store the ring atom.
        if classified[0] == "simple":
            simple.append((atom_id, classified[1]))
        else:
            complex_prefs.append((atom_id, classified[1], classified[2]))
        _consume_fragment(atoms, adj, neighbor, atom_id, consumed)
    if consumed != _heavy_set(atoms):
        return None
    size = len(cycle)
    stem = _HW_STEM.get(size)
    if stem is None:
        return None
    ring = list(cycle)
    start = ring.index(oxygen)
    best = None
    for direction in (ring[start:] + ring[:start], list(reversed(ring))[list(reversed(ring)).index(oxygen) :] + list(reversed(ring))[: list(reversed(ring)).index(oxygen)]):
        if direction[0] != oxygen:
            continue
        numbering = {atom_id: locant for locant, atom_id in enumerate(direction, 1)}
        acid_locants = tuple(sorted(numbering[atom_id] for atom_id in carboxyls))
        carbonyl_locants = tuple(sorted(numbering[atom_id] for atom_id in carbonyls))
        hydroxy_atoms = [atom_id for atom_id, word in simple if word == "hydroxy"]
        other_simple = [(numbering[atom_id], word) for atom_id, word in simple if word != "hydroxy"]
        hydroxy_locants = tuple(sorted(numbering[atom_id] for atom_id in hydroxy_atoms))
        rendered_complex = []
        for atom_id, text, paren in complex_prefs:
            locant = numbering[atom_id]
            if str(text).startswith("["):
                rendered = f"{locant}-{text}"
            elif paren:
                rendered = f"{locant}-({text})"
            else:
                rendered = f"{locant}-{text}"
            sort_name = _alphabet_key(text)
            rendered_complex.append(_Pref(sort_name, locant, rendered))
        score = (
            acid_locants or (99,),
            carbonyl_locants or (99,),
            hydroxy_locants,
            tuple(item.rendered for item in rendered_complex),
        )
        if best is None or score < best[0]:
            best = (score, numbering, acid_locants, carbonyl_locants, hydroxy_locants, other_simple, rendered_complex)
    if best is None:
        return None
    _score, numbering, acid_locants, carbonyl_locants, hydroxy_locants, other_simple, rendered_complex = best
    if len(acid_locants) == 1:
        hydroxy_prefix = _group_simple([(locant, "hydroxy") for locant in hydroxy_locants])
        prefix = _join_prefs([*hydroxy_prefix, *_group_simple(other_simple), *rendered_complex])
        return _result(
            f"{prefix}{stem}e-{acid_locants[0]}-carboxylic acid",
            numbering,
            *_trace("Select the heterocyclic carboxylic acid and number from the ring oxygen."),
        )
    if len(carbonyl_locants) == 1 and not acid_locants and not hydroxy_locants and not other_simple and not rendered_complex:
        return _result(
            f"{stem}-{carbonyl_locants[0]}-one",
            numbering,
            *_trace("Name the lactone by the Hantzsch–Widman parent."),
        )
    if hydroxy_locants and not acid_locants and not carbonyl_locants and not other_simple and not rendered_complex:
        if len(hydroxy_locants) == 1:
            return _result(
                f"{stem}-{hydroxy_locants[0]}-ol",
                numbering,
                *_trace("Name the cyclic hemiacetal or alcohol on the Hantzsch–Widman parent."),
            )
    return None


def _analyze_carbocycle(entity, atoms, adj):
    cycles = [
        cycle
        for cycle in _unique_cycle(atoms, adj)
        if all(atoms[atom_id].element == "C" for atom_id in cycle)
    ]
    if len(cycles) != 1:
        return None
    cycle = cycles[0]
    ring_bonds = [_bond(adj, cycle[index], cycle[(index + 1) % len(cycle)]) for index in range(len(cycle))]
    if any(bond is None or bond.aromatic for bond in ring_bonds):
        return None
    doubles = [index for index, bond in enumerate(ring_bonds) if _is_double(bond)]
    if any(not _is_single(bond) and not _is_double(bond) for bond in ring_bonds):
        return None
    substituents = []
    for atom_id in cycle:
        for neighbor in _neighbors(adj, atoms, atom_id):
            if neighbor in cycle:
                continue
            substituents.append((atom_id, neighbor))
    carbonyls = []
    hydroxys = []
    for atom_id, neighbor in substituents:
        if atoms[neighbor].element == "O" and _is_double(_bond(adj, atom_id, neighbor)):
            carbonyls.append(atom_id)
        elif atoms[neighbor].element == "O" and _h(entity, neighbor) > 0 and _is_single(_bond(adj, atom_id, neighbor)):
            hydroxys.append(atom_id)
        else:
            return None
    if set(cycle) | {neighbor for _atom, neighbor in substituents} != _heavy_set(atoms):
        return None
    size = len(cycle)
    stem = alkane_stem(size)
    if stem is None:
        return None
    if not doubles and len(hydroxys) == 1 and not carbonyls:
        return _result(f"cyclo{stem}anol", {hydroxys[0]: 1}, *_trace("Name the cycloalkanol with the hydroxy locant implied at position 1."))
    if not doubles and len(carbonyls) == 1 and not hydroxys:
        return _result(f"cyclo{stem}anone", {carbonyls[0]: 1}, *_trace("Name the cycloalkanone."))
    if len(doubles) == 2 and (hydroxys or carbonyls):
        kind = "one" if carbonyls else "ol"
        anchors = carbonyls or hydroxys
        best = None
        for direction in (list(cycle), list(reversed(cycle))):
            for start in range(size):
                ordered = direction[start:] + direction[:start]
                numbering = {atom_id: locant for locant, atom_id in enumerate(ordered, 1)}
                principal = tuple(sorted(numbering[atom_id] for atom_id in anchors))
                ene = []
                for index in range(size):
                    bond = _bond(adj, ordered[index], ordered[(index + 1) % size])
                    if _is_double(bond):
                        ene.append(index + 1)
                ene = tuple(sorted(min(item, size) for item in ene))
                # ene locant is the lower endpoint; index+1 is the start of the bond in this direction.
                score = (principal, ene)
                if best is None or score < best[0]:
                    best = (score, numbering, principal, ene)
        if best is None:
            return None
        _score, numbering, principal, ene = best
        head = stem if stem.endswith("a") else stem + "a"
        suffix = "ol" if kind == "ol" else "one"
        return _result(
            f"cyclo{head}-{','.join(map(str, ene))}-diene-{','.join(map(str, principal))}-{_multiplier(len(principal))}{suffix}",
            numbering,
            *_trace("Number the cyclohexadiene so the senior groups, then the double bonds, get the lowest locants."),
        )
    return None


def _analyze(entity: FiniteChemicalEntity):
    if not isinstance(entity, FiniteChemicalEntity) or _blocked(entity):
        return None
    from ..representation import normalize_kekule_aromatic

    entity = normalize_kekule_aromatic(entity)
    atoms = _atom_map(entity)
    adj = _heavy_adjacency(entity)
    for analyzer in (
        _analyze_small,
        _analyze_peroxide,
        _analyze_isocyanate,
        _analyze_formamide,
        _analyze_sulfur_link,
        _analyze_carbon_radical,
        _analyze_chain,
        _analyze_rings,
    ):
        found = analyzer(entity, atoms, adj)
        if found is not None:
            return found
    return None


def name_extended(entity: FiniteChemicalEntity):
    """Return a naming tuple, or ``None`` when this grammar does not apply."""
    if _legacy_covers(entity):
        return None
    found = _analyze(entity)
    if found is None:
        return None
    name, _locants, trace = found
    return (name, False, *trace)


def parent_numbering(entity: FiniteChemicalEntity):
    """Locants of the parent chosen by this grammar, when legacy naming did not run."""
    if _legacy_covers(entity):
        return None
    found = _analyze(entity)
    if found is None:
        return None
    return found[1]


def naming_gap(entity: FiniteChemicalEntity) -> str:
    """A short family hint for a structure this grammar still does not name."""
    if any(atom.radical_electrons or (atom.formal_charge or 0) for atom in entity.atoms):
        return "charge or radical state is outside the implemented suffix set"
    atoms = _atom_map(entity)
    adj = _heavy_adjacency(entity)
    _carbons, graph = _carbon_graph(atoms, adj)
    if any(not _is_tree(component, graph) for component in _components(_carbons, graph) if component):
        return "ring substitution pattern is outside the implemented grammar"
    return "acyclic functional-group combination is outside the implemented parent grammar"


# --- inverse construction ----------------------------------------------------


class _Build:
    def __init__(self):
        self.atoms: list[ChemicalAtom] = []
        self.bonds: list[ChemicalBond] = []
        self.counter = 0
        self.stereo: list[tuple[str, str]] = []

    def atom(self, atom_id, element, charge=None, radical=0):
        self.atoms.append(
            ChemicalAtom(
                atom_id=atom_id,
                element=element,
                formal_charge=charge,
                radical_electrons=radical,
                evidence=(Evidence(EvidenceSource.IUPAC_NAME, "substitutive_grammar"),),
            )
        )
        return atom_id

    def bond(self, left, right, order=1.0, aromatic=False):
        self.bonds.append(
            ChemicalBond(
                atom1_id=left,
                atom2_id=right,
                order=order,
                kind=BondKind.COVALENT,
                aromatic=aromatic,
                evidence=(Evidence(EvidenceSource.IUPAC_NAME, "substitutive_grammar"),),
            )
        )

    def fresh(self, element, charge=None, radical=0):
        self.counter += 1
        return self.atom(f"X{self.counter}", element, charge, radical)

    def entity(self, name: str) -> FiniteChemicalEntity:
        _complete_valence(self.atoms, self.bonds)
        entity = FiniteChemicalEntity(
            entity_id=f"iupac:{name}",
            atoms=tuple(self.atoms),
            bonds=tuple(self.bonds),
            net_charge=sum(atom.formal_charge or 0 for atom in self.atoms),
            status=InferenceStatus.EXPLICIT,
            evidence=(Evidence(EvidenceSource.IUPAC_NAME, "substitutive_grammar"),),
        )
        return _apply_recorded_stereo(entity, self.stereo)


def _complete_valence(atoms: list[ChemicalAtom], bonds: list[ChemicalBond]) -> None:
    adjacency: dict[str, float] = {atom.atom_id: 0.0 for atom in atoms}
    for bond in bonds:
        adjacency[bond.atom1_id] += float(bond.order or 0.0)
        adjacency[bond.atom2_id] += float(bond.order or 0.0)
    for index, atom in enumerate(atoms):
        charge = atom.formal_charge or 0
        bond_sum = adjacency[atom.atom_id]
        if atom.element == "C":
            target = 3 if charge else 4
        elif atom.element == "N":
            target = 4 if charge == 1 else 3
        elif atom.element == "O":
            target = 1 if charge == -1 else 3 if charge == 1 else 2
        elif atom.element == "S":
            target = next((value for value in (2, 4, 6) if bond_sum <= value + 1e-8), 6)
        elif atom.element in HALOGEN_PREFIX:
            target = 1
        else:
            continue
        hydrogens = int(round(target - bond_sum - atom.radical_electrons))
        if hydrogens < 0:
            hydrogens = 0
        atoms[index] = ChemicalAtom(
            atom_id=atom.atom_id,
            element=atom.element,
            formal_charge=atom.formal_charge,
            radical_electrons=atom.radical_electrons,
            implicit_hydrogens=hydrogens or None,
            stereochemistry=atom.stereochemistry,
            evidence=atom.evidence,
        )


def _apply_recorded_stereo(entity: FiniteChemicalEntity, stereo: list[tuple[str, str]]):
    from ..stereo import assign_stereochemistry

    for atom_id, descriptor in stereo:
        if descriptor not in {"R", "S"}:
            continue
        atoms = list(entity.atoms)
        index = next(position for position, atom in enumerate(atoms) if atom.atom_id == atom_id)
        for token in ("@", "@@"):
            trial = ChemicalAtom(
                atom_id=atoms[index].atom_id,
                element=atoms[index].element,
                formal_charge=atoms[index].formal_charge,
                radical_electrons=atoms[index].radical_electrons,
                implicit_hydrogens=atoms[index].implicit_hydrogens,
                stereochemistry=token,
                evidence=atoms[index].evidence,
            )
            candidate_atoms = atoms[:index] + [trial] + atoms[index + 1 :]
            candidate = FiniteChemicalEntity(
                entity_id=entity.entity_id,
                atoms=tuple(candidate_atoms),
                bonds=entity.bonds,
                net_charge=entity.net_charge,
                status=entity.status,
                evidence=entity.evidence,
            )
            report = assign_stereochemistry(candidate)
            found = next((item for item in report.descriptors if item.center_atom_id == atom_id), None)
            if found is not None and found.descriptor == descriptor:
                entity = candidate
                break
    return entity


def _stem_count(stem: str) -> int | None:
    from ..naming import ALKANE_STEMS

    for count, value in ALKANE_STEMS.items():
        if value == stem:
            return count
    return None


def _read_matching(text: str, index: int, opener: str, closer: str):
    if text[index] != opener:
        return None
    depth = 0
    for position in range(index, len(text)):
        if text[position] == opener:
            depth += 1
        elif text[position] == closer:
            depth -= 1
            if depth == 0:
                return text[index + 1 : position], position + 1
    return None


def _read_prefixes(text: str):
    words = sorted(_SIMPLE, key=len, reverse=True)
    word_pattern = re.compile(rf"(di|tri|tetra|penta|hexa)?({'|'.join(words)})")
    prefixes = []
    while text[:1].isdigit():
        locant_match = re.match(r"(\d+(?:,\d+)*)-", text)
        if locant_match is None:
            return None
        locants = tuple(int(item) for item in locant_match.group(1).split(","))
        rest = text[locant_match.end() :]
        if rest[:1] in {"(", "["}:
            opener = rest[0]
            closer = ")" if opener == "(" else "]"
            matched = _read_matching(rest, 0, opener, closer)
            if matched is None:
                return None
            inner, end = matched
            prefixes.append((locants, inner if opener == "[" else inner, True))
            text = rest[end:]
        else:
            word_match = word_pattern.match(rest)
            if word_match is None:
                return None
            multiplier = word_match.group(1)
            word = word_match.group(2)
            expected = _MULTIPLIER_WORDS.get(multiplier, 1)
            if len(locants) != expected:
                return None
            prefixes.append((locants, word, False))
            text = rest[word_match.end() :]
        if text.startswith("-") and len(text) > 1 and text[1].isdigit():
            text = text[1:]
    return prefixes, text


def _add_chain(build: _Build, count: int, orders: dict[int, float] | None = None):
    orders = orders or {}
    for locant in range(1, count + 1):
        build.atom(f"C{locant}", "C")
    for locant in range(1, count):
        build.bond(f"C{locant}", f"C{locant + 1}", orders.get(locant, 1.0))


def _attach_oxo(build: _Build, locant: int):
    oxygen = build.fresh("O")
    build.bond(f"C{locant}", oxygen, 2.0)


def _attach_hydroxy(build: _Build, locant: int):
    oxygen = build.fresh("O")
    build.bond(f"C{locant}", oxygen, 1.0)


def _attach_simple(build: _Build, locant: int, word: str):
    anchor = f"C{locant}"
    if word == "hydroxy":
        _attach_hydroxy(build, locant)
    elif word == "oxo":
        _attach_oxo(build, locant)
    elif word == "amino":
        nitrogen = build.fresh("N")
        build.bond(anchor, nitrogen)
    elif word == "sulfanyl":
        sulfur = build.fresh("S")
        build.bond(anchor, sulfur)
    elif word in {"fluoro", "chloro", "bromo", "iodo"}:
        element = {name: symbol for symbol, name in HALOGEN_PREFIX.items()}[word]
        halogen = build.fresh(element)
        build.bond(anchor, halogen)
    elif word == "methyl":
        carbon = build.fresh("C")
        build.bond(anchor, carbon)
    elif word == "acetamido":
        nitrogen = build.fresh("N")
        carbonyl = build.fresh("C")
        oxygen = build.fresh("O")
        methyl = build.fresh("C")
        build.bond(anchor, nitrogen)
        build.bond(nitrogen, carbonyl)
        build.bond(carbonyl, oxygen, 2.0)
        build.bond(carbonyl, methyl)
    elif word == "phenoxy":
        oxygen = build.fresh("O")
        build.bond(anchor, oxygen)
        _attach_phenyl(build, oxygen, ())
    elif word == "phenyl":
        _attach_phenyl(build, anchor, ())
    elif word == "phenylsulfanyl":
        sulfur = build.fresh("S")
        build.bond(anchor, sulfur)
        _attach_phenyl(build, sulfur, ())
    else:
        return False
    return True


def _attach_phenyl(build: _Build, anchor: str, substituents: list[tuple[int, str]]):
    ids = []
    for _index in range(6):
        ids.append(build.fresh("C"))
    for index in range(6):
        build.bond(ids[index], ids[(index + 1) % 6], 1.5, aromatic=True)
    build.bond(anchor, ids[0])
    for locant, word in substituents:
        if not 1 <= locant <= 6:
            continue
        _attach_to_atom(build, ids[locant - 1], word)


def _attach_to_atom(build: _Build, anchor: str, word: str):
    if word == "hydroxy":
        oxygen = build.fresh("O")
        build.bond(anchor, oxygen)
    elif word == "methyl":
        carbon = build.fresh("C")
        build.bond(anchor, carbon)
    elif word in {"fluoro", "chloro", "bromo", "iodo"}:
        element = {name: symbol for symbol, name in HALOGEN_PREFIX.items()}[word]
        build.bond(anchor, build.fresh(element))


def _parse_parent(build: _Build, parent: str):
    acid = re.fullmatch(r"([a-z]+)anoic acid", parent)
    if acid:
        count = _stem_count(acid.group(1))
        if count is None:
            return None
        _add_chain(build, count)
        oxygen = build.fresh("O")
        hydroxy = build.fresh("O")
        build.bond("C1", oxygen, 2.0)
        build.bond("C1", hydroxy, 1.0)
        return {"kind": "acid", "count": count}
    diacid = re.fullmatch(r"([a-z]+)anedioic acid", parent)
    if diacid:
        count = _stem_count(diacid.group(1))
        if count is None or count < 2:
            return None
        _add_chain(build, count)
        for locant in (1, count):
            build.bond(f"C{locant}", build.fresh("O"), 2.0)
            build.bond(f"C{locant}", build.fresh("O"), 1.0)
        return {"kind": "diacid", "count": count}
    enoic = re.fullmatch(r"([a-z]+)-(\d+)-enoic acid", parent)
    if enoic:
        count = _stem_count(enoic.group(1))
        locant = int(enoic.group(2))
        if count is None or not 1 <= locant < count:
            return None
        _add_chain(build, count, {locant: 2.0})
        build.bond("C1", build.fresh("O"), 2.0)
        build.bond("C1", build.fresh("O"), 1.0)
        return {"kind": "acid", "count": count}
    dienoic = re.fullmatch(r"([a-z]+)a-(\d+(?:,\d+)+)-dienoic acid", parent)
    if dienoic:
        count = _stem_count(dienoic.group(1))
        locants = tuple(int(item) for item in dienoic.group(2).split(","))
        if count is None:
            return None
        _add_chain(build, count, {locant: 2.0 for locant in locants})
        build.bond("C1", build.fresh("O"), 2.0)
        build.bond("C1", build.fresh("O"), 1.0)
        return {"kind": "acid", "count": count}
    dienedioic = re.fullmatch(r"([a-z]+)a-(\d+(?:,\d+)+)-dienedioic acid", parent)
    if dienedioic:
        count = _stem_count(dienedioic.group(1))
        locants = tuple(int(item) for item in dienedioic.group(2).split(","))
        if count is None:
            return None
        _add_chain(build, count, {locant: 2.0 for locant in locants})
        for locant in (1, count):
            build.bond(f"C{locant}", build.fresh("O"), 2.0)
            build.bond(f"C{locant}", build.fresh("O"), 1.0)
        return {"kind": "diacid", "count": count}
    ketone = re.fullmatch(r"([a-z]+)an-(\d+)-one", parent)
    if ketone:
        count = _stem_count(ketone.group(1))
        locant = int(ketone.group(2))
        if count is None or not 1 <= locant <= count:
            return None
        _add_chain(build, count)
        _attach_oxo(build, locant)
        return {"kind": "ketone", "count": count}
    dione = re.fullmatch(r"([a-z]+)ane-(\d+(?:,\d+)+)-(di|tri|tetra)one", parent)
    if dione:
        count = _stem_count(dione.group(1))
        locants = tuple(int(item) for item in dione.group(2).split(","))
        if count is None or _MULTIPLIER_WORDS.get(dione.group(3)) != len(locants):
            return None
        _add_chain(build, count)
        for locant in locants:
            _attach_oxo(build, locant)
        return {"kind": "ketone", "count": count}
    alcohol = re.fullmatch(r"([a-z]+)an-(\d+)-ol", parent)
    if alcohol:
        count = _stem_count(alcohol.group(1))
        locant = int(alcohol.group(2))
        if count is None:
            return None
        _add_chain(build, count)
        _attach_hydroxy(build, locant)
        return {"kind": "alcohol", "count": count}
    diol = re.fullmatch(r"([a-z]+)ane-(\d+(?:,\d+)+)-(di|tri|tetra)ol", parent)
    if diol:
        count = _stem_count(diol.group(1))
        locants = tuple(int(item) for item in diol.group(2).split(","))
        if count is None or _MULTIPLIER_WORDS.get(diol.group(3)) != len(locants):
            return None
        _add_chain(build, count)
        for locant in locants:
            _attach_hydroxy(build, locant)
        return {"kind": "alcohol", "count": count}
    aldehyde = re.fullmatch(r"([a-z]+)anal", parent)
    if aldehyde:
        count = _stem_count(aldehyde.group(1))
        if count is None:
            return None
        _add_chain(build, count)
        _attach_oxo(build, 1)
        return {"kind": "aldehyde", "count": count}
    dial = re.fullmatch(r"([a-z]+)anedial", parent)
    if dial:
        count = _stem_count(dial.group(1))
        if count is None:
            return None
        _add_chain(build, count)
        _attach_oxo(build, 1)
        _attach_oxo(build, count)
        return {"kind": "aldehyde", "count": count}
    alkane = re.fullmatch(r"([a-z]+)ane", parent)
    if alkane and parent not in {"methane"}:
        count = _stem_count(alkane.group(1))
        if count is None:
            return None
        _add_chain(build, count)
        return {"kind": "alkane", "count": count}
    if parent == "methane":
        _add_chain(build, 1)
        return {"kind": "alkane", "count": 1}
    return None


def _expand_prefixes(prefixes):
    pairs = []
    for locants, word, compound in prefixes:
        if compound:
            for locant in locants:
                pairs.append((locant, word, True))
        else:
            for locant in locants:
                pairs.append((locant, word, False))
    return pairs


def parse_extended(name: str):
    """Parse one grammar-owned name into a finite chemical graph."""
    if not isinstance(name, str) or not name:
        return None
    special = _parse_special_name(name)
    if special is not None:
        return special
    read = _read_prefixes(name)
    if read is None:
        return _parse_decorated_parent(name)
    prefixes, parent = read
    if not parent:
        return None
    build = _Build()
    parsed = _parse_parent(build, parent)
    if parsed is None:
        # Heterocycle and functional-class parents carry their own prefixes.
        return _parse_decorated_parent(name)
    for locant, word, compound in _expand_prefixes(prefixes):
        if compound:
            if not _attach_complex(build, f"C{locant}", word):
                return None
        elif not _attach_simple(build, locant, word):
            return None
    return build.entity(name)


def _attach_complex(build: _Build, anchor: str, text: str) -> bool:
    if text == "phenylsulfanyl":
        sulfur = build.fresh("S")
        build.bond(anchor, sulfur)
        _attach_phenyl(build, sulfur, ())
        return True
    if text.endswith("sulfanyl") and text.startswith("cyclohexa-"):
        sulfur = build.fresh("S")
        build.bond(anchor, sulfur)
        return _attach_dienyl(build, sulfur, text[: -len("sulfanyl")])
    if text.endswith("sulfanyl") and "carbamoyl" in text:
        sulfur = build.fresh("S")
        build.bond(anchor, sulfur)
        return _attach_carbamoyl(build, sulfur, text[: -len("sulfanyl")])
    if text == "carboxymethylamino":
        nitrogen = build.fresh("N")
        methylene = build.fresh("C")
        carbonyl = build.fresh("C")
        oxo = build.fresh("O")
        hydroxy = build.fresh("O")
        build.bond(anchor, nitrogen)
        build.bond(nitrogen, methylene)
        build.bond(methylene, carbonyl)
        build.bond(carbonyl, oxo, 2.0)
        build.bond(carbonyl, hydroxy)
        return True
    if text.endswith("]amino") and text.startswith("["):
        inner = text[1:-6]
        nitrogen = build.fresh("N")
        build.bond(anchor, nitrogen)
        return _attach_yl(build, nitrogen, inner)
    if text.endswith("phenoxy"):
        body = text[: -len("phenoxy")]
        oxygen = build.fresh("O")
        build.bond(anchor, oxygen)
        return _attach_substituted_phenyl(build, oxygen, body)
    if text.endswith("phenyl"):
        return _attach_substituted_phenyl(build, anchor, text[: -len("phenyl")])
    return False


def _attach_substituted_phenyl(build: _Build, anchor: str, prefix: str) -> bool:
    substituents = []
    if prefix:
        read = _read_prefixes(prefix)
        if read is None:
            return False
        parsed, rest = read
        if rest:
            return False
        substituents = [(locant, word) for locants, word, compound in parsed if not compound for locant in locants]
    _attach_phenyl(build, anchor, substituents)
    return True


def _attach_dienyl(build: _Build, anchor: str, text: str) -> bool:
    match = re.fullmatch(r"cyclohexa-(\d+(?:,\d+)+)-dien-1-yl", text)
    if match is None:
        return False
    ene = {int(item) for item in match.group(1).split(",")}
    ids = [build.fresh("C") for _index in range(6)]
    build.bond(anchor, ids[0])
    for index in range(6):
        order = 2.0 if (index + 1) in ene else 1.0
        build.bond(ids[index], ids[(index + 1) % 6], order)
    return True


def _attach_carbamoyl(build: _Build, anchor: str, text: str) -> bool:
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    match = re.fullmatch(r"(n-hydroxy-n-methyl|n-methyl|n,n-dimethyl)?carbamoyl", text.lower())
    if match is None:
        return False
    carbon = build.fresh("C")
    oxygen = build.fresh("O")
    nitrogen = build.fresh("N")
    build.bond(anchor, carbon)
    build.bond(carbon, oxygen, 2.0)
    build.bond(carbon, nitrogen)
    kind = match.group(1)
    if kind == "n-hydroxy-n-methyl":
        build.bond(nitrogen, build.fresh("O"))
        build.bond(nitrogen, build.fresh("C"))
    elif kind == "n-methyl":
        build.bond(nitrogen, build.fresh("C"))
    elif kind == "n,n-dimethyl":
        build.bond(nitrogen, build.fresh("C"))
        build.bond(nitrogen, build.fresh("C"))
    return True


def _attach_yl(build: _Build, anchor: str, text: str) -> bool:
    stereo = None
    body = text
    match = re.match(r"^\((\d*)([RrSs])\)-(.+)$", body)
    if match:
        stereo = match.group(2).upper()
        body = match.group(3)
    read = _read_prefixes(body) if body[:1].isdigit() else ([], body)
    if read is None:
        return False
    prefixes, parent = read
    yl = re.fullmatch(r"([a-z]+)an-(\d+)-yl", parent)
    if yl is None:
        return False
    count = _stem_count(yl.group(1))
    attachment = int(yl.group(2))
    if count is None or not 1 <= attachment <= count:
        return False
    ids = {}
    for locant in range(1, count + 1):
        ids[locant] = build.fresh("C")
    for locant in range(1, count):
        build.bond(ids[locant], ids[locant + 1])
    build.bond(anchor, ids[attachment])
    if stereo:
        build.stereo.append((ids[attachment], stereo))
    for locants, word, compound in prefixes:
        for locant in locants:
            target = ids[locant]
            if compound:
                if word.lower() == "carboxymethylamino":
                    nitrogen = build.fresh("N")
                    methylene = build.fresh("C")
                    carbonyl = build.fresh("C")
                    build.bond(target, nitrogen)
                    build.bond(nitrogen, methylene)
                    build.bond(methylene, carbonyl)
                    build.bond(carbonyl, build.fresh("O"), 2.0)
                    build.bond(carbonyl, build.fresh("O"))
                elif not _attach_complex(build, target, word):
                    return False
            elif word == "oxo":
                build.bond(target, build.fresh("O"), 2.0)
            elif word == "sulfanyl":
                build.bond(target, build.fresh("S"))
            elif word == "hydroxy":
                build.bond(target, build.fresh("O"))
            elif word == "amino":
                build.bond(target, build.fresh("N"))
            else:
                return False
    return True


def _parse_special_name(name: str):
    build = _Build()
    if name == "hydron":
        build.atom("H1", "H", charge=1)
        return build.entity(name)
    if name == "hydroxyl":
        build.atom("O1", "O", radical=1)
        return build.entity(name)
    if name == "hydrogen chloride":
        build.atom("Cl1", "Cl")
        return build.entity(name)
    if name == "methyl":
        build.atom("C1", "C", radical=1)
        return build.entity(name)
    if name == "dioxygen":
        build.atom("O1", "O")
        build.atom("O2", "O")
        build.bond("O1", "O2", 2.0)
        return build.entity(name)
    if name == "superoxide":
        build.atom("O1", "O", charge=-1)
        build.atom("O2", "O", radical=1)
        build.bond("O1", "O2", 1.0)
        return build.entity(name)
    radical = re.fullmatch(r"(dichloro|trichloro)?methyl", name)
    if radical and name != "methyl":
        build.atom("C1", "C", radical=1)
        count = {"dichloro": 2, "trichloro": 3}[radical.group(1)]
        for _index in range(count):
            build.bond("C1", build.fresh("Cl"))
        return build.entity(name)
    peroxide = re.fullmatch(
        r"(\(trichloromethyl\)|methyl|ethyl|trichloromethyl)(peroxyl| hydroperoxide)",
        name,
    )
    if peroxide:
        label, kind = peroxide.group(1).strip("()"), peroxide.group(2)
        carbon = build.atom("C1", "C")
        inner = build.fresh("O")
        outer = build.fresh("O", radical=1 if kind == "peroxyl" else 0)
        build.bond(carbon, inner)
        build.bond(inner, outer)
        if label == "ethyl":
            build.bond(carbon, build.fresh("C"))
        if label == "trichloromethyl":
            for _index in range(3):
                build.bond(carbon, build.fresh("Cl"))
        return build.entity(name)
    isocyanate = re.fullmatch(r"(methyl|ethyl) isocyanate", name)
    if isocyanate:
        count = 1 if isocyanate.group(1) == "methyl" else 2
        _add_chain(build, count)
        carbon = build.fresh("C")
        nitrogen = build.fresh("N")
        oxygen = build.fresh("O")
        build.bond("C1", nitrogen)
        build.bond(nitrogen, carbon, 2.0)
        build.bond(carbon, oxygen, 2.0)
        return build.entity(name)
    formamide = re.fullmatch(
        r"n,n-dimethylformamide|n,n-diethylformamide|n-methylformamide|n-ethylformamide|"
        r"n-\(hydroxymethyl\)formamide|n-\(hydroxymethyl\)-n-methylformamide",
        name.lower(),
    )
    if formamide:
        carbon = build.atom("C1", "C")
        oxygen = build.fresh("O")
        nitrogen = build.fresh("N")
        build.bond(carbon, oxygen, 2.0)
        build.bond(carbon, nitrogen)
        lowered = name.lower()
        if "n,n-dimethyl" in lowered:
            build.bond(nitrogen, build.fresh("C"))
            build.bond(nitrogen, build.fresh("C"))
        elif "n,n-diethyl" in lowered:
            for _index in range(2):
                first = build.fresh("C")
                second = build.fresh("C")
                build.bond(nitrogen, first)
                build.bond(first, second)
        elif "n-(hydroxymethyl)-n-methyl" in lowered:
            methylene = build.fresh("C")
            build.bond(nitrogen, methylene)
            build.bond(methylene, build.fresh("O"))
            build.bond(nitrogen, build.fresh("C"))
        elif "n-(hydroxymethyl)" in lowered:
            methylene = build.fresh("C")
            build.bond(nitrogen, methylene)
            build.bond(methylene, build.fresh("O"))
        elif "n-ethyl" in lowered:
            first = build.fresh("C")
            second = build.fresh("C")
            build.bond(nitrogen, first)
            build.bond(first, second)
        else:
            build.bond(nitrogen, build.fresh("C"))
        return build.entity(name)
    sulfur = re.fullmatch(
        r"\(((?:methyl|ethyl)sulfanyl|(?:methane|ethane)(?:sulfinyl|sulfonyl))\)(methane|ethane)",
        name,
    )
    if sulfur:
        inner, parent = sulfur.groups()
        if inner.endswith("sulfanyl"):
            left_count = {"methyl": 1, "ethyl": 2}[inner[: -len("sulfanyl")]]
            kind_name = "sulfide"
        elif inner.endswith("sulfinyl"):
            left_count = {"methane": 1, "ethane": 2}[inner[: -len("sulfinyl")]]
            kind_name = "sulfinyl"
        else:
            left_count = {"methane": 1, "ethane": 2}[inner[: -len("sulfonyl")]]
            kind_name = "sulfonyl"
        right_count = {"methane": 1, "ethane": 2}[parent]
        _add_chain(build, right_count)
        sulfur_id = build.fresh("S")
        build.bond("C1", sulfur_id)
        current = sulfur_id
        for _index in range(left_count):
            carbon = build.fresh("C")
            build.bond(current, carbon)
            current = carbon
        if kind_name == "sulfinyl":
            build.bond(sulfur_id, build.fresh("O"), 2.0)
        elif kind_name == "sulfonyl":
            build.bond(sulfur_id, build.fresh("O"), 2.0)
            build.bond(sulfur_id, build.fresh("O"), 2.0)
        return build.entity(name)
    halo = re.fullmatch(r"(di|tri|tetra)?(fluoro|chloro|bromo|iodo)methane", name)
    if halo:
        build.atom("C1", "C")
        count = _MULTIPLIER_WORDS.get(halo.group(1), 1)
        element = {value: key for key, value in HALOGEN_PREFIX.items()}[halo.group(2)]
        for _index in range(count):
            build.bond("C1", build.fresh(element))
        return build.entity(name)
    methanol = re.fullmatch(r"(di|tri|tetra)?(fluoro|chloro|bromo|iodo)methanol", name)
    if methanol:
        build.atom("C1", "C")
        count = _MULTIPLIER_WORDS.get(methanol.group(1), 1)
        element = {value: key for key, value in HALOGEN_PREFIX.items()}[methanol.group(2)]
        for _index in range(count):
            build.bond("C1", build.fresh(element))
        build.bond("C1", build.fresh("O"))
        return build.entity(name)
    if name == "ethyl":
        build.atom("C1", "C", radical=1)
        build.bond("C1", build.fresh("C"))
        return build.entity(name)
    haloethane = re.fullmatch(r"(fluoro|chloro|bromo|iodo)ethane", name)
    if haloethane:
        build.atom("C1", "C")
        build.atom("C2", "C")
        build.bond("C1", "C2")
        element = {value: key for key, value in HALOGEN_PREFIX.items()}[haloethane.group(1)]
        build.bond("C1", build.fresh(element))
        return build.entity(name)
    return None


def _parse_decorated_parent(name: str):
    """Parents that are not plain alkane suffixes, including rings and salts."""
    build = _Build()
    sulfate = re.fullmatch(r"([A-Za-z0-9,\-]+) sulfate", name)
    if sulfate:
        aryl = sulfate.group(1)
        sulfur = build.fresh("S")
        build.bond(sulfur, build.fresh("O"), 2.0)
        build.bond(sulfur, build.fresh("O"), 2.0)
        build.bond(sulfur, build.fresh("O", charge=-1))
        linker = build.fresh("O")
        build.bond(sulfur, linker)
        if not _attach_substituted_phenyl(build, linker, aryl[: -len("phenyl")] if aryl.endswith("phenyl") else None):
            return None
        return build.entity(name)
    phenoxyl = re.fullmatch(r"(?:(\d+(?:,\d+)*)-(hydroxy|oxido))?phenoxyl", name)
    if phenoxyl:
        return _parse_phenolic(build, name, phenoxyl.group(1), phenoxyl.group(2), "phenoxyl")
    phenolate = re.fullmatch(r"(?:(\d+(?:,\d+)*)-hydroxy)?phenolate", name)
    if phenolate:
        return _parse_phenolic(build, name, phenolate.group(1), "hydroxy", "phenolate")
    diolate = re.fullmatch(r"benzene-(\d+(?:,\d+)+)-(di|tri)olate", name)
    if diolate:
        locants = [int(item) for item in diolate.group(1).split(",")]
        ids = [build.fresh("C") for _index in range(6)]
        for index in range(6):
            build.bond(ids[index], ids[(index + 1) % 6], 1.5, aromatic=True)
        for locant in locants:
            build.bond(ids[locant - 1], build.fresh("O", charge=-1))
        return build.entity(name)
    oxepine = name == "oxepine"
    if oxepine:
        ids = [build.atom("O1", "O")] + [build.atom(f"C{index}", "C") for index in range(2, 8)]
        orders = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        for index in range(7):
            build.bond(ids[index], ids[(index + 1) % 7], orders[index])
        return build.entity(name)
    hw_acid = re.fullmatch(
        r"(.*?)(oxirane|oxetane|oxolane|oxane)-(\d+)-carboxylic acid",
        name,
    )
    if hw_acid:
        prefix, parent_name, locant = hw_acid.groups()
        if prefix.endswith("-"):
            prefix = prefix[:-1]
        size = {"oxirane": 3, "oxetane": 4, "oxolane": 5, "oxane": 6}[parent_name]
        return _parse_hw(build, name, size, int(locant), prefix or None, "acid")
    hw_fun = re.fullmatch(r"(oxiran|oxetan|oxolan|oxan)-(\d+)-(ol|one)", name)
    if hw_fun:
        stem, locant, kind = hw_fun.groups()
        size = {"oxiran": 3, "oxetan": 4, "oxolan": 5, "oxan": 6}[stem]
        return _parse_hw(build, name, size, int(locant), None, kind)
    cycloalcohol = re.fullmatch(r"cyclo([a-z]+)anol", name)
    if cycloalcohol:
        count = _stem_count(cycloalcohol.group(1))
        if count is None:
            return None
        _add_cycle(build, count, {})
        _attach_hydroxy(build, 1)
        return build.entity(name)
    cycloketone = re.fullmatch(r"cyclo([a-z]+)anone", name)
    if cycloketone and name != "cyclohexanone":
        count = _stem_count(cycloketone.group(1))
        if count is None:
            return None
        _add_cycle(build, count, {})
        _attach_oxo(build, 1)
        return build.entity(name)
    diene = re.fullmatch(
        r"cyclo([a-z]+)a-(\d+(?:,\d+)+)-diene-(\d+(?:,\d+)+)-(di|tri)(ol|one)",
        name,
    )
    if diene:
        count = _stem_count(diene.group(1))
        ene = {int(item) for item in diene.group(2).split(",")}
        locants = [int(item) for item in diene.group(3).split(",")]
        kind = diene.group(5)
        if count is None:
            return None
        _add_cycle(build, count, {locant: 2.0 for locant in ene})
        for locant in locants:
            if kind == "ol":
                _attach_hydroxy(build, locant)
            else:
                _attach_oxo(build, locant)
        return build.entity(name)
    return None


def _parse_phenolic(build, name, locant_text, word, suffix):
    ids = [build.fresh("C") for _index in range(6)]
    for index in range(6):
        build.bond(ids[index], ids[(index + 1) % 6], 1.5, aromatic=True)
    if suffix == "phenoxyl":
        build.bond(ids[0], build.fresh("O", radical=1))
    else:
        build.bond(ids[0], build.fresh("O", charge=-1))
    if locant_text:
        for locant in (int(item) for item in locant_text.split(",")):
            if word == "oxido":
                build.bond(ids[locant - 1], build.fresh("O", charge=-1))
            else:
                build.bond(ids[locant - 1], build.fresh("O"))
    return build.entity(name)


def _add_cycle(build: _Build, count: int, orders: dict[int, float]):
    for locant in range(1, count + 1):
        build.atom(f"C{locant}", "C")
    for locant in range(1, count):
        build.bond(f"C{locant}", f"C{locant + 1}", orders.get(locant, 1.0))
    build.bond(f"C{count}", "C1", orders.get(count, 1.0))


def _parse_hw(build, name, size, locant, prefix, kind):
    build.atom("O1", "O")
    for index in range(2, size + 1):
        build.atom(f"C{index}", "C")
    build.bond("O1", "C2")
    for index in range(2, size):
        build.bond(f"C{index}", f"C{index + 1}")
    build.bond(f"C{size}", "O1")
    if kind == "acid":
        carbon = build.fresh("C")
        build.bond(f"C{locant}", carbon)
        build.bond(carbon, build.fresh("O"), 2.0)
        build.bond(carbon, build.fresh("O"))
    elif kind == "ol":
        build.bond(f"C{locant}", build.fresh("O"))
    elif kind == "one":
        build.bond(f"C{locant}", build.fresh("O"), 2.0)
    if prefix:
        read = _read_prefixes(prefix + ("-" if prefix[-1].isdigit() else ""))
        # prefix is already without a trailing hyphen before the parent; reader
        # expects the prefix text alone.
        read = _read_prefixes(prefix if prefix.endswith("-") else prefix)
        if read is None:
            # The caller passes the prefix without forcing a final hyphen.
            read = _read_prefixes(prefix + "-") if False else _read_prefixes(prefix)
        if read is None:
            return None
        prefixes, rest = read
        if rest not in {"", "-"}:
            return None
        for locants, word, compound in prefixes:
            for item in locants:
                anchor = f"C{item}"
                if compound:
                    if not _attach_complex(build, anchor, word):
                        return None
                elif not _attach_simple(build, item, word):
                    return None
    return build.entity(name)


__all__ = ["name_extended", "naming_gap", "parent_numbering", "parse_extended"]
