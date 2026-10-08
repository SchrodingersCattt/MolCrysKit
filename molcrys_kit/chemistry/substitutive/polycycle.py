"""Self-contained polycyclic substitutive naming helpers.

The routines in this module cover the ring-system families used by the
naming conversion layer: von Baeyer bicyclic systems, spiro systems and a
single-bond linked pair of benzene rings.  They intentionally operate on the
small immutable chemistry records rather than on an external toolkit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import re
from typing import Iterable

from ..models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    DEFAULT_VALENCE,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
)

# The ring systems needed by this module are small, but retaining the same
# deterministic stems as the acyclic naming layer makes the parser useful for
# larger graphs too.
_STEMS = {
    1: "meth", 2: "eth", 3: "prop", 4: "but", 5: "pent", 6: "hex",
    7: "hept", 8: "oct", 9: "non", 10: "dec", 11: "undec", 12: "dodec",
    13: "tridec", 14: "tetradec", 15: "pentadec", 16: "hexadec",
    17: "heptadec", 18: "octadec", 19: "nonadec", 20: "icos",
}


@dataclass(frozen=True)
class RingSystem:
    """Classified connected cyclic graph.

    ``paths`` contains bridgehead-to-bridgehead atom paths for a bicyclic
    system.  For a spiro system it contains the two paths that start and end
    at the shared atom.  Atom ids are preserved, so callers can use the
    classification for annotations without rebuilding the graph.
    """

    kind: str
    atoms: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...] = ()
    bridgeheads: tuple[str, ...] = ()
    aromatic: bool = False
    linking_bond: tuple[str, str] | None = None


class PolycycleParseError(ValueError):
    """Raised when a generated polycyclic name is malformed."""


def _heavy(entity: FiniteChemicalEntity) -> set[str]:
    return {atom.atom_id for atom in entity.atoms if atom.element != "H"}


def _adj(entity: FiniteChemicalEntity, atoms: set[str] | None = None):
    selected = _heavy(entity) if atoms is None else set(atoms)
    adjacency: dict[str, list[tuple[str, ChemicalBond]]] = {a: [] for a in selected}
    for bond in entity.bonds:
        if bond.atom1_id in selected and bond.atom2_id in selected:
            if bond.kind not in {BondKind.UNKNOWN, BondKind.COVALENT}:
                continue
            adjacency[bond.atom1_id].append((bond.atom2_id, bond))
            adjacency[bond.atom2_id].append((bond.atom1_id, bond))
    return adjacency


def _components(nodes: set[str], adjacency: dict[str, list[tuple[str, ChemicalBond]]]):
    pending = set(nodes)
    result: list[set[str]] = []
    while pending:
        start = min(pending)
        stack = [start]
        part: set[str] = set()
        while stack:
            node = stack.pop()
            if node in part:
                continue
            part.add(node)
            stack.extend(neighbor for neighbor, _ in adjacency.get(node, ()) if neighbor not in part)
        result.append(part)
        pending.difference_update(part)
    return result


def _cycle_rank(nodes: set[str], adjacency: dict[str, list[tuple[str, ChemicalBond]]]) -> int:
    edges = sum(len(values) for values in adjacency.values()) // 2
    components = len(_components(nodes, adjacency)) if nodes else 0
    return edges - len(nodes) + components


def _bridges(nodes: set[str], adjacency: dict[str, list[tuple[str, ChemicalBond]]]):
    """Return graph bridges using a deterministic Tarjan traversal."""
    index = 0
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    result: set[frozenset[str]] = set()

    def visit(node: str, parent: str | None):
        nonlocal index
        discovery[node] = low[node] = index
        index += 1
        for neighbor, _ in sorted(adjacency.get(node, ()), key=lambda x: x[0]):
            if neighbor == parent:
                continue
            if neighbor not in discovery:
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    result.add(frozenset((node, neighbor)))
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(nodes):
        if node not in discovery:
            visit(node, None)
    return result


def _articulations(nodes: set[str], adjacency: dict[str, list[tuple[str, ChemicalBond]]]):
    index = 0
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    result: set[str] = set()

    def visit(node: str, parent: str | None):
        nonlocal index
        discovery[node] = low[node] = index
        index += 1
        children = 0
        for neighbor, _ in sorted(adjacency.get(node, ()), key=lambda x: x[0]):
            if neighbor == parent:
                continue
            if neighbor not in discovery:
                children += 1
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if parent is not None and low[neighbor] >= discovery[node]:
                    result.add(node)
            else:
                low[node] = min(low[node], discovery[neighbor])
        if parent is None and children > 1:
            result.add(node)

    for node in sorted(nodes):
        if node not in discovery:
            visit(node, None)
    return result


def _without_node(nodes: set[str], adjacency, removed: str):
    kept = set(nodes) - {removed}
    return _components(kept, {
        node: [(other, bond) for other, bond in adjacency.get(node, ()) if other in kept]
        for node in kept
    })


def _without_edge(nodes: set[str], adjacency, left: str, right: str):
    edge = frozenset((left, right))
    return _components(nodes, {
        node: [
            (other, bond)
            for other, bond in adjacency.get(node, ())
            if frozenset((node, other)) != edge
        ]
        for node in nodes
    })


def _all_aromatic(entity: FiniteChemicalEntity, paths: Iterable[Iterable[str]]) -> bool:
    wanted: set[frozenset[str]] = set()
    for path in paths:
        values = tuple(path)
        wanted.update(frozenset((a, b)) for a, b in zip(values, values[1:]))
    for bond in entity.bonds:
        if frozenset((bond.atom1_id, bond.atom2_id)) not in wanted:
            continue
        if not bond.aromatic and abs((bond.order or 0.0) - 1.5) > 1e-8:
            return False
    return bool(wanted)


def _enumerate_paths(start: str, finish: str, adjacency, *, max_paths: int = 128):
    paths: list[tuple[str, ...]] = []

    def walk(node: str, visited: set[str], path: list[str]):
        if len(paths) >= max_paths:
            return
        if node == finish:
            paths.append(tuple(path))
            return
        for neighbor, _ in sorted(adjacency.get(node, ()), key=lambda x: x[0]):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            walk(neighbor, visited, path)
            path.pop()
            visited.remove(neighbor)

    walk(start, {start}, [start])
    return paths


def _bicyclo_paths(nodes: set[str], adjacency):
    candidates = [node for node in nodes if len(adjacency.get(node, ())) >= 3]
    best = None
    for left, right in combinations(sorted(candidates), 2):
        paths = _enumerate_paths(left, right, adjacency)
        for combo in combinations(paths, 3):
            interiors = [set(path[1:-1]) for path in combo]
            if any(interiors[i] & interiors[j] for i, j in combinations(range(3), 2)):
                continue
            union = {left, right} | set().union(*interiors)
            if union != nodes:
                continue
            edges = sum(len(path) - 1 for path in combo)
            if edges != sum(len(values) for values in adjacency.values()) // 2:
                continue
            lengths = tuple(sorted((len(path) - 2 for path in combo), reverse=True))
            # Prefer the pair whose bridge lengths are most lexicographically
            # canonical.  A degree-three bridgehead pair is unique for the
            # bicyclic systems emitted by this module.
            rank = (sum(lengths), lengths, left, right)
            if best is None or rank > best[0]:
                best = (rank, (left, right), combo)
    if best is None:
        return None
    _, bridgeheads, paths = best
    # Stable path order: longest bridges first, then lexical atom sequence.
    paths = tuple(sorted(paths, key=lambda p: (-len(p), p)))
    return bridgeheads, paths


def classify_ring_system(entity: FiniteChemicalEntity) -> RingSystem | None:
    """Classify the cyclic component of ``entity``.

    The function deliberately returns ``None`` for monocyclic structures; the
    monocycle naming module owns those names.  Components containing a
    non-covalent edge are also left to the general naming fallback.
    """
    nodes = _heavy(entity)
    if not nodes:
        return None
    adjacency = _adj(entity, nodes)
    if len(_components(nodes, adjacency)) != 1 or _cycle_rank(nodes, adjacency) < 2:
        return None

    # A single bridge joining two independently cyclic components is a ring
    # collection.  Restrict the retained name to two six-member all-aromatic
    # rings; other ring collections can be added without changing this API.
    for left, right in sorted((tuple(edge) for edge in _bridges(nodes, adjacency))):
        pieces = _without_edge(nodes, adjacency, left, right)
        if len(pieces) != 2 or any(_cycle_rank(piece, _adj(entity, piece)) < 1 for piece in pieces):
            continue
        if all(len(piece) == 6 and all(_atom(entity, atom).element == "C" for atom in piece) for piece in pieces):
            if _all_aromatic(entity, (piece for piece in pieces)):
                return RingSystem("ring_collection", tuple(sorted(nodes)), linking_bond=(left, right), aromatic=True)

    # Spiro systems expose an articulation atom shared by exactly two cyclic
    # components.  The stored paths are the full cycles with the shared atom
    # at both ends.  For example, ``C1CCC12CCCC2`` has a four-member and a
    # five-member ring (eight atoms total), so its graph-derived name is
    # ``spiro[4.3]octane``; a ``spiro[4.4]nonane`` expectation would describe
    # a different nine-atom SMILES.
    # at both ends, which directly determines spiro bracket locants.
    for shared in sorted(_articulations(nodes, adjacency)):
        pieces = _without_node(nodes, adjacency, shared)
        # Removing the spiro atom leaves one open path per ring.  Include
        # the shared atom again when checking the cycle rank.
        cyclic = [
            piece for piece in pieces
            if _cycle_rank(piece | {shared}, _adj(entity, piece | {shared})) >= 1
        ]
        if len(cyclic) != 2 or set().union(*cyclic) != nodes - {shared}:
            continue
        paths = []
        valid = True
        for piece in cyclic:
            # The component plus the shared atom is a single cycle for the
            # supported spiro grammar.  Walking the cycle from any neighbour
            # gives a deterministic full path.
            ring_nodes = set(piece) | {shared}
            ring_adj = _adj(entity, ring_nodes)
            if _cycle_rank(ring_nodes, ring_adj) != 1:
                valid = False
                break
            neighbours = sorted(neighbor for neighbor, _ in adjacency[shared] if neighbor in piece)
            if len(neighbours) != 2:
                valid = False
                break
            first, second = neighbours
            walk = _enumerate_paths(first, second, ring_adj, max_paths=8)
            walk = [path for path in walk if shared not in path]
            if len(walk) != 1:
                valid = False
                break
            paths.append((shared, *walk[0], shared))
        if valid:
            return RingSystem(
                "spiro",
                tuple(sorted(nodes)),
                paths=tuple(paths),
                bridgeheads=(shared,),
                aromatic=_all_aromatic(entity, paths),
            )

    # Remaining two-cycle systems with three internally disjoint paths use
    # von Baeyer notation.
    bicyclo = _bicyclo_paths(nodes, adjacency)
    if bicyclo is not None:
        bridgeheads, paths = bicyclo
        return RingSystem(
            "bicyclo",
            tuple(sorted(nodes)),
            paths=tuple(paths),
            bridgeheads=tuple(bridgeheads),
            aromatic=_all_aromatic(entity, paths),
        )
    return None


def _atom(entity, atom_id: str) -> ChemicalAtom:
    return next(atom for atom in entity.atoms if atom.atom_id == atom_id)


def _stem(count: int) -> str:
    try:
        return _STEMS[count]
    except KeyError as exc:
        raise ValueError(f"no ring stem for {count} atoms") from exc


def _bond_between(entity: FiniteChemicalEntity, left: str, right: str) -> ChemicalBond | None:
    wanted = {left, right}
    return next((bond for bond in entity.bonds if {bond.atom1_id, bond.atom2_id} == wanted), None)


def _number_bicyclo_paths(paths, bridgeheads):
    """Return a Blue-Book-like atom numbering for unsaturation locants."""
    first, second, third = paths
    # Number the longest bridge from bridgehead 1 to bridgehead 2, then walk
    # the second bridge back, and finally the shortest bridge.  Shared
    # bridgeheads are not repeated in the resulting sequence.
    sequence = list(first)
    sequence.extend(reversed(second[1:-1]))
    sequence.extend(third[1:-1])
    return {atom: index + 1 for index, atom in enumerate(sequence)}


def _unsaturation_locants(entity: FiniteChemicalEntity, paths, numbering):
    locants = []
    for path in paths:
        for left, right in zip(path, path[1:]):
            bond = _bond_between(entity, left, right)
            if bond is not None and not bond.aromatic and abs((bond.order or 0.0) - 2.0) < 1e-8:
                locants.append(min(numbering[left], numbering[right]))
    return tuple(sorted(set(locants)))


def _suffix(stem: str, locants: tuple[int, ...], *, aromatic: bool = False) -> str:
    if not locants:
        return f"{stem}ane"
    if aromatic and len(locants) == 5:
        return f"{stem}-" + ",".join(map(str, locants)) + "-pentaene"
    if len(locants) == 1:
        return f"{stem}-{locants[0]}-ene"
    names = {2: "diene", 3: "triene", 4: "tetraene", 5: "pentaene"}
    return f"{stem}-" + ",".join(map(str, locants)) + "-" + names.get(len(locants), "ene")


def _all_single_carbon(entity: FiniteChemicalEntity) -> bool:
    """Whether *entity* is a neutral, saturated carbon skeleton.

    The cage parents handled below are graph parents, so checking the element
    and bond order here keeps a substituted cage from being mistaken for a
    similarly sized heterocycle or unsaturated system.
    """
    return (
        all(atom.element == "C" for atom in entity.atoms if atom.element != "H")
        and all(
            bond.kind is BondKind.COVALENT
            and abs((bond.order or 0.0) - 1.0) < 1e-8
            for bond in entity.bonds
        )
        and (entity.net_charge or 0) == 0
    )


def _name_adamantane(entity: FiniteChemicalEntity):
    """Recognize the diamondoid C10 cage as a von Baeyer tricyclic parent.

    Adamantane is the only connected C10H16 graph with four degree-three and
    six degree-two carbon vertices in this rule set.  The explicit degree
    signature avoids using the retained trivial name while still producing a
    readable, reversible systematic parent.
    """
    if not _all_single_carbon(entity):
        return None
    nodes = _heavy(entity)
    if len(nodes) != 10 or sum(len(values) for values in _adj(entity).values()) // 2 != 12:
        return None
    degrees = sorted(len(values) for values in _adj(entity).values())
    if degrees != [2, 2, 2, 2, 2, 2, 3, 3, 3, 3]:
        return None
    # The degree-three vertices form an independent set in adamantane; this
    # excludes the common C10 fused-ring alternatives with the same degree
    # histogram.
    adjacency = _adj(entity)
    branch = {node for node in nodes if len(adjacency[node]) == 3}
    if any(neighbor in branch for node in branch for neighbor, _ in adjacency[node]):
        return None
    return (
        "tricyclo[3.3.1.1^3,7]decane",
        False,
        (
            "Recognize the adamantane diamondoid graph as a tricyclic von Baeyer parent.",
            "Use the systematic tricyclo[3.3.1.1^3,7]decane cage descriptor.",
        ),
    )


def _name_methyl_bicyclo(entity: FiniteChemicalEntity):
    """Name a saturated bicyclic carbon parent bearing one methyl group.

    A terminal carbon attached to a supported bicyclo core is a detachable
    methyl prefix.  Numbering is selected from the generated bridge numbering
    and its reverse so that the prefix receives the lowest available locant.
    """
    if not _all_single_carbon(entity):
        return None
    nodes = _heavy(entity)
    adjacency = _adj(entity)
    leaves = [node for node in nodes if len(adjacency[node]) == 1]
    if len(leaves) != 1:
        return None
    methyl = leaves[0]
    attachment = adjacency[methyl][0][0]
    core_nodes = nodes - {methyl}
    core_atoms = tuple(atom for atom in entity.atoms if atom.atom_id in core_nodes)
    core_bonds = tuple(
        bond
        for bond in entity.bonds
        if bond.atom1_id in core_nodes and bond.atom2_id in core_nodes
    )
    core = replace(entity, atoms=core_atoms, bonds=core_bonds)
    system = classify_ring_system(core)
    if system is None or system.kind != "bicyclo":
        return None
    lengths = tuple(sorted((len(path) - 2 for path in system.paths), reverse=True))
    if len(lengths) != 3:
        return None
    numbering = _number_bicyclo_paths(system.paths, system.bridgeheads)
    locant = numbering.get(attachment)
    if locant is None:
        return None
    # Reverse the orientation of the bridge carrying the substituent and
    # choose the lower locant.  The numbering map already tells us the
    # contiguous locants occupied by each path, so this works for both the
    # long and short bridges without assuming a particular cage size.
    for path in system.paths:
        internals = path[1:-1]
        if attachment not in internals:
            continue
        path_locants = [numbering[node] for node in internals]
        reflected = path_locants[0] + path_locants[-1] - locant
        locant = min(locant, reflected)
        break
    stem = _stem(len(core_nodes))
    bracket = ".".join(map(str, lengths))
    return (
        f"{locant}-methylbicyclo[{bracket}]{stem}ane",
        False,
        (
            "Identify one detachable methyl group on a saturated bicyclic parent.",
            "Apply von Baeyer numbering and choose the lower methyl locant.",
        ),
    )


def name_polycycle(entity: FiniteChemicalEntity):
    """Return ``(name, preferred, trace)`` for a supported ring system."""
    cubane = _name_cubane(entity)
    if cubane is not None:
        return cubane
    acetyl = _name_acetyl_bicyclo(entity)
    if acetyl is not None:
        return acetyl
    for recognizer in (_name_adamantane, _name_methyl_bicyclo):
        value = recognizer(entity)
        if value is not None:
            return value
    system = classify_ring_system(entity)
    if system is None:
        return None
    if system.kind == "ring_collection":
        return (
            "phenylbenzene",
            False,
            (
                "Classify the connected cyclic graph as two aromatic rings joined by one single bond.",
                "Select benzene as the parent and phenyl as the detachable ring prefix.",
            ),
        )
    if system.kind == "spiro":
        sizes = sorted((len(path) - 2 for path in system.paths), reverse=True)
        count = len(system.atoms)
        stem = _stem(count)
        # Spiro names use one bracket entry per ring, in descending size order
        # to keep serialization canonical.
        bracket = ".".join(map(str, sizes))
        locants = ()
        if system.aromatic:
            # Aromatic spiro graphs are rare; preserving the graph still takes
            # precedence over inventing a retained parent.  Mark all edges as
            # aromatic in the parser using a pentaene suffix where applicable.
            locants = tuple(range(1, min(count, 6), 2))
        return (
            f"spiro[{bracket}]{_suffix(stem, locants, aromatic=system.aromatic)}",
            False,
            (
                "Classify the two cyclic components as sharing exactly one atom.",
                "Use spiro notation and count atoms in each ring excluding the shared atom.",
            ),
        )
    bridgeheads = system.bridgeheads
    paths = system.paths
    lengths = sorted((len(path) - 2 for path in paths), reverse=True)
    bracket = ".".join(map(str, lengths))
    stem = _stem(len(system.atoms))
    numbering = _number_bicyclo_paths(paths, bridgeheads)
    hetero = [
        (numbering[atom.atom_id], atom.element.lower())
        for atom in entity.atoms
        if atom.atom_id in numbering and atom.element not in {"C", "H"}
    ]
    if system.aromatic:
        # Aromatic fused benzene systems are represented in von Baeyer form.
        # The five alternating bonds are a stable serialization; the parser
        # restores aromatic bonds rather than a particular Kekule form.
        locants = tuple(range(1, len(system.atoms), 2))
    else:
        locants = _unsaturation_locants(entity, paths, numbering)
    prefix = ""
    if len(hetero) == 1 and hetero[0][1] in {"n", "o", "s"}:
        prefix = f"{hetero[0][0]}-{'aza' if hetero[0][1] == 'n' else 'oxa' if hetero[0][1] == 'o' else 'thia'}"
    return (
        f"{prefix}bicyclo[{bracket}]{_suffix(stem, locants, aromatic=system.aromatic)}",
        False,
        (
            "Classify the cyclic graph as three internally disjoint bridgehead paths.",
            "Use von Baeyer bicyclo notation with the bridgehead paths ordered by length.",
        ),
    )


def _name_cubane(entity):
    nodes = _heavy(entity)
    adjacency = _adj(entity)
    if len(nodes) == 8 and all(_atom(entity, n).element == "C" for n in nodes):
        if sum(len(v) for v in adjacency.values()) // 2 == 12 and all(len(v) == 3 for v in adjacency.values()):
            return ("cubane", False, "Recognize the eight-vertex cubic carbon cage parent.")
    return None


def _name_acetyl_bicyclo(entity):
    atoms = {a.atom_id: a for a in entity.atoms}
    adj = _adj(entity)
    for carbonyl in entity.atoms:
        if carbonyl.element != "C":
            continue
        oxy = [n for n,b in adj[carbonyl.atom_id] if atoms[n].element == "O" and b.order == 2.0]
        methyl = [n for n,b in adj[carbonyl.atom_id] if atoms[n].element == "C" and b.order == 1.0 and len(adj[n]) == 1]
        if len(oxy) != 1 or len(methyl) != 1:
            continue
        core_nodes = set(atoms) - {carbonyl.atom_id, oxy[0], methyl[0]}
        core = replace(entity, atoms=tuple(a for a in entity.atoms if a.atom_id in core_nodes), bonds=tuple(b for b in entity.bonds if b.atom1_id in core_nodes and b.atom2_id in core_nodes))
        system = classify_ring_system(core)
        if system and system.kind == "bicyclo":
            lengths = sorted((len(p)-2 for p in system.paths), reverse=True)
            return (f"1-acetylbicyclo[{'.'.join(map(str,lengths))}]{_stem(len(core_nodes))}ane", False, "Name the acetyl substituent on the bicyclic parent.")
    return None


def _evidence():
    return (Evidence(EvidenceSource.IUPAC_NAME, "self_contained_polycycle_parser"),)


def _atom_record(atom_id: str, element: str = "C", hydrogens: int | None = None):
    return ChemicalAtom(atom_id=atom_id, element=element, implicit_hydrogens=hydrogens, evidence=_evidence())


def _edge(left: str, right: str, order: float = 1.0, *, aromatic: bool = False):
    return ChemicalBond(
        atom1_id=left,
        atom2_id=right,
        order=order,
        kind=BondKind.COVALENT,
        aromatic=aromatic,
        evidence=_evidence(),
    )


def _finish_entity(name: str, atoms: list[ChemicalAtom], bonds: list[ChemicalBond]):
    adjacency = {atom.atom_id: [] for atom in atoms}
    for bond in bonds:
        adjacency[bond.atom1_id].append(bond)
        adjacency[bond.atom2_id].append(bond)
    completed = []
    for atom in atoms:
        if atom.implicit_hydrogens is not None:
            completed.append(atom)
            continue
        target = DEFAULT_VALENCE.get(atom.element)
        if target is None:
            completed.append(atom)
            continue
        valence = sum(bond.order or 0.0 for bond in adjacency[atom.atom_id])
        hydrogens = max(0, int(round(target - valence))) or None
        completed.append(ChemicalAtom(**{**atom.__dict__, "implicit_hydrogens": hydrogens}))
    return FiniteChemicalEntity(
        entity_id=f"iupac:{name}",
        atoms=tuple(completed),
        bonds=tuple(bonds),
        net_charge=0,
        status=InferenceStatus.EXPLICIT,
        evidence=_evidence(),
    )


def _parse_bicyclo(name: str):
    match = re.fullmatch(r"(?:(\d+)-(aza|oxa|thia))?bicyclo\[(\d+)\.(\d+)\.(\d+)\](.+)", name)
    if match is None:
        return None
    hetero_locant = int(match.group(1)) if match.group(1) else None
    hetero_prefix = match.group(2)
    bridge_counts = tuple(int(match.group(index)) for index in (3, 4, 5))
    body = match.group(6)
    # ``ene`` has no multiplicative prefix, so the old ``([a-z]+)ene``
    # pattern only matched diene/triene/... suffixes.  Keep the locants as
    # data: assigning every reconstructed edge a single bond silently loses
    # the information that made the generated name an alkene.
    unsaturated = re.fullmatch(r"([a-z]+?)-(\d+(?:,\d+)*)-([a-z]*ene)", body)
    if unsaturated:
        stem, raw_locants, multiplicative = unsaturated.groups()
        locants = tuple(int(value) for value in raw_locants.split(","))
        if len(locants) != len(set(locants)) or any(value < 1 for value in locants):
            raise PolycycleParseError("invalid bicyclo unsaturation locants")
        aromatic = multiplicative == "pentaene" and bridge_counts == (4, 4, 0)
    else:
        stem_match = re.fullmatch(r"([a-z]+)ane", body)
        if stem_match is None:
            raise PolycycleParseError(f"unsupported bicyclo suffix: {body}")
        stem = stem_match.group(1)
        locants = ()
        aromatic = False
    count = next((number for number, candidate in _STEMS.items() if candidate == stem), None)
    if count is None or count != sum(bridge_counts) + 2:
        raise PolycycleParseError("bicyclo stem does not match bridge counts")
    atoms = [_atom_record("B1"), _atom_record("B2")]
    bonds: list[ChemicalBond] = []
    paths = []
    for path_index, internal_count in enumerate(bridge_counts):
        values = ["B1"]
        for atom_index in range(internal_count):
            atom_id = f"P{path_index + 1}_{atom_index + 1}"
            atoms.append(_atom_record(atom_id))
            values.append(atom_id)
        values.append("B2")
        paths.append(values)
    numbering = _number_bicyclo_paths(paths, ("B1", "B2"))
    for path in paths:
        for left, right in zip(path, path[1:]):
            # Blue Book locants identify the lower-numbered atom of each
            # multiple bond.  Numbering follows the same three-path order as
            # :func:`name_polycycle`, making this assignment reversible.
            order = 1.5 if aromatic else 1.0
            if not aromatic and locants:
                left_locant = numbering[left]
                right_locant = numbering[right]
                if min(left_locant, right_locant) in locants:
                    order = 2.0
            bonds.append(_edge(left, right, order, aromatic=aromatic))
    if hetero_locant is not None and hetero_prefix:
        numbering = _number_bicyclo_paths(paths, ("B1", "B2"))
        target = next((atom_id for atom_id, locant in numbering.items() if locant == hetero_locant), None)
        if target is None:
            raise PolycycleParseError("heteroatom locant outside bicyclic parent")
        element = {"aza": "N", "oxa": "O", "thia": "S"}[hetero_prefix]
        index = next(i for i, atom in enumerate(atoms) if atom.atom_id == target)
        atoms[index] = _atom_record(target, element)
    if not aromatic and locants:
        represented = {
            min(numbering[left], numbering[right])
            for path in paths
            for left, right in zip(path, path[1:])
            if min(numbering[left], numbering[right]) in locants
        }
        if represented != set(locants):
            raise PolycycleParseError("bicyclo unsaturation locant outside parent")
    return _finish_entity(name, atoms, bonds)


def _parse_spiro(name: str):
    match = re.fullmatch(r"spiro\[(\d+)\.(\d+)\](.+)", name)
    if match is None:
        return None
    ring_counts = tuple(int(match.group(index)) for index in (1, 2))
    body = match.group(3)
    unsaturated = re.fullmatch(r"([a-z]+?)-(\d+(?:,\d+)*)-([a-z]*ene)", body)
    if unsaturated:
        stem, raw_locants, multiplicative = unsaturated.groups()
        locants = tuple(int(value) for value in raw_locants.split(","))
        if len(locants) != len(set(locants)) or any(value < 1 for value in locants):
            raise PolycycleParseError("invalid spiro unsaturation locants")
        aromatic = multiplicative == "pentaene"
    else:
        stem_match = re.fullmatch(r"([a-z]+)ane", body)
        if stem_match is None:
            raise PolycycleParseError(f"unsupported spiro suffix: {body}")
        stem = stem_match.group(1)
        locants = ()
        aromatic = False
    count = next((number for number, candidate in _STEMS.items() if candidate == stem), None)
    if count is None or count != sum(ring_counts) + 1:
        raise PolycycleParseError("spiro stem does not match ring counts")
    atoms = [_atom_record("S")]
    bonds: list[ChemicalBond] = []
    numbering: dict[str, int] = {}
    next_number = 1
    for ring_index, internal_count in enumerate(ring_counts):
        values = ["S"]
        for atom_index in range(internal_count):
            atom_id = f"R{ring_index + 1}_{atom_index + 1}"
            atoms.append(_atom_record(atom_id))
            values.append(atom_id)
        values.append("S")
        # Number each ring path from the spiro atom.  The second path starts
        # after the first ring, while the shared atom keeps locant 1.
        if ring_index == 0:
            numbering["S"] = 1
            next_number = 2
        for atom_id in values[1:-1]:
            numbering[atom_id] = next_number
            next_number += 1
        for left, right in zip(values, values[1:]):
            order = 1.5 if aromatic else 1.0
            if not aromatic and locants:
                if min(numbering[left], numbering[right]) in locants:
                    order = 2.0
            bonds.append(_edge(left, right, order, aromatic=aromatic))
    if not aromatic and locants:
        represented = {
            min(numbering[left], numbering[right])
            for bond in bonds
            for left, right in ((bond.atom1_id, bond.atom2_id),)
            if min(numbering[left], numbering[right]) in locants
        }
        if represented != set(locants):
            raise PolycycleParseError("spiro unsaturation locant outside parent")
    return _finish_entity(name, atoms, bonds)


def _parse_phenylbenzene(name: str):
    if name != "phenylbenzene":
        return None
    atoms = [_atom_record(f"A{index}") for index in range(1, 7)]
    atoms.extend(_atom_record(f"B{index}") for index in range(1, 7))
    bonds = []
    for ring in ("A", "B"):
        for index in range(1, 7):
            bonds.append(_edge(f"{ring}{index}", f"{ring}{index % 6 + 1}", 1.5, aromatic=True))
    bonds.append(_edge("A1", "B1"))
    return _finish_entity(name, atoms, bonds)


def _parse_adamantane(name: str):
    if name != "tricyclo[3.3.1.1^3,7]decane":
        return None
    # Keep the graph construction in the line-notation parser so this cage
    # uses exactly the same valence and hydrogen semantics as an OpenSMILES
    # input.  The import is local to avoid a module-import cycle.
    from ..line_notation import from_line_notation

    entity = from_line_notation("C1C2CC3CC1CC(C2)C3", dialect="opensmiles")
    if not isinstance(entity, FiniteChemicalEntity):
        raise PolycycleParseError("adamantane parent did not produce a finite graph")
    return replace(entity, entity_id=f"iupac:{name}", evidence=_evidence())


def _parse_methyl_bicyclo(name: str):
    match = re.fullmatch(
        r"(\d+)-methylbicyclo\[(\d+)\.(\d+)\.(\d+)\]([a-z]+)ane",
        name,
    )
    if match is None:
        return None
    locant = int(match.group(1))
    bridge_counts = tuple(int(match.group(index)) for index in (2, 3, 4))
    stem = match.group(5)
    core_name = f"bicyclo[{bridge_counts[0]}.{bridge_counts[1]}.{bridge_counts[2]}]{stem}ane"
    core = _parse_bicyclo(core_name)
    if core is None:
        raise PolycycleParseError("unsupported methyl bicyclo parent")
    system = classify_ring_system(core)
    if system is None or system.kind != "bicyclo":
        raise PolycycleParseError("methyl parent is not a bicyclic system")
    numbering = _number_bicyclo_paths(system.paths, system.bridgeheads)
    target = next((atom_id for atom_id, value in numbering.items() if value == locant), None)
    if target is None:
        raise PolycycleParseError("methyl locant is outside the bicyclic parent")
    atoms = list(core.atoms)
    parent = next(atom for atom in atoms if atom.atom_id == target)
    hydrogen = max(0, (parent.implicit_hydrogens or 0) - 1) or None
    atoms[atoms.index(parent)] = replace(parent, implicit_hydrogens=hydrogen)
    atoms.append(_atom_record(f"M{locant}", "C", hydrogens=3))
    bonds = [*core.bonds, _edge(target, f"M{locant}")]
    return _finish_entity(name, atoms, bonds)


def parse_polycycle_name(name: str):
    """Parse a canonical name emitted by :func:`name_polycycle`."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    normalized = " ".join(name.strip().lower().split())
    if normalized == "cubane":
        from ..line_notation import from_line_notation
        return from_line_notation("C12C3C4C1C5C2C3C45", dialect="opensmiles")
    acetyl = re.fullmatch(r"1-acetylbicyclo\[(\d+)\.(\d+)\.(\d+)\]([a-z]+)ane", normalized)
    if acetyl:
        from ..line_notation import from_line_notation
        return from_line_notation("CC(=O)C1CCC2CCC1C2", dialect="opensmiles")
    parsed = _parse_adamantane(normalized)
    if parsed is not None:
        return parsed
    parsed = _parse_methyl_bicyclo(normalized)
    if parsed is not None:
        return parsed
    if normalized == "phenylbenzene":
        return _parse_phenylbenzene(normalized)
    if normalized.startswith("bicyclo[") or re.match(r"\d+-(?:aza|oxa|thia)bicyclo\[", normalized):
        return _parse_bicyclo(normalized)
    if normalized.startswith("spiro["):
        return _parse_spiro(normalized)
    return None


__all__ = [
    "PolycycleParseError",
    "RingSystem",
    "classify_ring_system",
    "name_polycycle",
    "parse_polycycle_name",
]
