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


def _number_spiro_paths(paths, orientations=None):
    """Return Blue-Book numbering for a monospiro parent.

    ``paths`` are the two cycles emitted by :func:`classify_ring_system`,
    with the shared atom at both ends.  Spiro descriptors cite the number of
    *non-spiro* atoms in ascending ring-size order.  Numbering starts at the
    atom next to the spiro atom in the smaller ring, runs around that ring to
    the spiro atom, and then continues around the larger ring.  The shared
    atom therefore receives ``small_ring_size + 1`` rather than locant 1.

    ``orientations`` is an optional pair of booleans used by the name builder
    to select which direction is numbered in each ring.  The parser leaves it
    at the default (the orientation encoded by the generated name).
    """
    ordered = tuple(sorted(paths, key=lambda path: (len(path) - 2, path)))
    if len(ordered) != 2:
        raise ValueError("a monospiro parent needs exactly two ring paths")
    if orientations is None:
        orientations = (False, False)
    if len(orientations) != 2:
        raise ValueError("spiro orientations must contain two flags")

    numbering: dict[str, int] = {}
    cursor = 1
    for index, path in enumerate(ordered):
        internals = tuple(path[1:-1])
        if orientations[index]:
            internals = tuple(reversed(internals))
        for atom_id in internals:
            numbering[atom_id] = cursor
            cursor += 1
        if index == 0:
            # The common atom is counted once, between the two ring walks.
            numbering[path[0]] = cursor
            cursor += 1
    return numbering


def _best_spiro_numbering(entity: FiniteChemicalEntity, paths):
    """Choose the lowest useful heteroatom/unsaturation locants for a spiro.

    Each ring can be traversed in either direction.  The Blue Book priority
    for this small supported subset is replacement heteroatoms first and
    multiple-bond locants second; atom IDs are only a deterministic tie-break.
    """
    ordered = tuple(sorted(paths, key=lambda path: (len(path) - 2, path)))
    candidates = []
    for first_reversed in (False, True):
        for second_reversed in (False, True):
            orientations = (first_reversed, second_reversed)
            numbering = _number_spiro_paths(ordered, orientations)
            hetero = tuple(
                sorted(
                    numbering[atom.atom_id]
                    for atom in entity.atoms
                    if atom.atom_id in numbering and atom.element not in {"C", "H"}
                )
            )
            unsaturation = _unsaturation_locants(entity, ordered, numbering)
            # Prefer lower heteroatom locants when present; otherwise lower
            # double-bond locants.  The final sequence keeps output stable for
            # fully symmetric systems.
            score = (
                (0, hetero) if hetero else (1, unsaturation),
                unsaturation,
                tuple(numbering[atom] for path in ordered for atom in path[1:-1]),
            )
            candidates.append((score, numbering))
    return min(candidates, key=lambda value: value[0])[1]


def _hetero_prefixes_for_paths(entity: FiniteChemicalEntity, paths, numbering):
    """Return locanted skeletal-replacement prefixes, or ``None`` if unsupported."""
    prefixes = []
    for atom in entity.atoms:
        if atom.atom_id not in numbering or atom.element in {"C", "H"}:
            continue
        base = {"N": "aza", "O": "oxa", "S": "thia"}.get(atom.element)
        if base is None:
            return None
        charge = atom.formal_charge or 0
        if charge:
            if atom.element != "N" or charge != 1:
                return None
            hydrogens = (atom.explicit_hydrogens or 0) + (atom.implicit_hydrogens or 0)
            base = "azanium" if hydrogens else "azonia"
        prefixes.append((numbering[atom.atom_id], base))
    return tuple(sorted(prefixes))


def _serialize_hetero_prefixes(prefixes) -> str:
    return "-".join(f"{locant}-{prefix}" for locant, prefix in prefixes)


_RING_WORDS = {
    2: "bi", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa",
    7: "hepta", 8: "octa", 9: "nona", 10: "deca", 11: "undeca", 12: "dodeca",
}
_WORD_RINGS = {word: count for count, word in _RING_WORDS.items()}
_HETERO_PREFIX = {"N": "aza", "O": "oxa", "S": "thia"}
_PREFIX_ELEMENT = {"aza": "N", "oxa": "O", "thia": "S"}


def _edge_key(left: str, right: str) -> frozenset[str]:
    return frozenset((left, right))


def _two_core_nodes(entity: FiniteChemicalEntity) -> set[str]:
    """Return atoms that survive deletion of every acyclic pendant."""
    nodes = set(_heavy(entity))
    adjacency = _adj(entity, nodes)
    core = set(nodes)
    changed = True
    while changed:
        changed = False
        for node in list(core):
            degree = sum(other in core for other, _ in adjacency[node])
            if degree < 2:
                core.remove(node)
                changed = True
    return core


def _induced_entity(entity: FiniteChemicalEntity, nodes: set[str]) -> FiniteChemicalEntity:
    return replace(
        entity,
        atoms=tuple(atom for atom in entity.atoms if atom.atom_id in nodes),
        bonds=tuple(
            bond
            for bond in entity.bonds
            if bond.atom1_id in nodes and bond.atom2_id in nodes
        ),
    )


def _is_unbranched_carbon_chain(entity: FiniteChemicalEntity, nodes: set[str], root: str) -> bool:
    if root not in nodes or any(_atom(entity, node).element != "C" for node in nodes):
        return False
    adjacency = _adj(entity, nodes)
    degrees = {node: len(adjacency[node]) for node in nodes}
    if any(degree > 2 for degree in degrees.values()):
        return False
    if len(nodes) == 1:
        return degrees[root] == 0
    if degrees[root] != 1 or sum(degree == 1 for degree in degrees.values()) != 2:
        return False
    for node in nodes:
        for _, bond in adjacency[node]:
            if bond.aromatic or abs((bond.order or 0.0) - 1.0) > 1e-8:
                return False
    return True


def _alkyl_word(count: int) -> str:
    stem = _stem(count)
    if count == 1:
        return "methyl"
    if count == 2:
        return "ethyl"
    if count == 3:
        return "propyl"
    if count == 4:
        return "butyl"
    return f"{stem}yl"


def _acyl_word(count: int) -> str:
    return f"{_stem(count)}anoyl"


def _fragment_word(entity: FiniteChemicalEntity, nodes: set[str], root: str) -> str | None:
    """Name one pendant as an alkyl or alkanoyl prefix."""
    if _atom(entity, root).element != "C":
        return None
    adjacency = _adj(entity)
    neighbours = [(other, bond) for other, bond in adjacency[root] if other in nodes]
    doubles = [
        other for other, bond in neighbours
        if not bond.aromatic and abs((bond.order or 0.0) - 2.0) < 1e-8
    ]
    singles = [
        other for other, bond in neighbours
        if not bond.aromatic and abs((bond.order or 0.0) - 1.0) < 1e-8
    ]
    if len(neighbours) != len(doubles) + len(singles):
        return None
    if len(doubles) == 1 and _atom(entity, doubles[0]).element == "O" and len(singles) <= 1:
        oxygen = doubles[0]
        if any(other != root and other in nodes for other, _ in adjacency[oxygen]):
            return None
        chain = set(nodes) - {root, oxygen}
        if singles:
            if set(singles) != {next(iter(chain), None)} and not (
                len(singles) == 1 and singles[0] in chain and _is_unbranched_carbon_chain(entity, chain, singles[0])
            ):
                return None
            if not _is_unbranched_carbon_chain(entity, chain, singles[0]):
                return None
        elif chain:
            return None
        return _acyl_word(1 + len(chain))
    if doubles or not _is_unbranched_carbon_chain(entity, nodes, root):
        return None
    return _alkyl_word(len(nodes))


def _substituent_specs(entity: FiniteChemicalEntity, core: set[str]):
    """Return ``(attachment, prefix)`` pairs, or ``None`` when one will not name."""
    extra = _heavy(entity) - core
    if not extra:
        return []
    adjacency = _adj(entity)
    pendant = {
        node: [(other, bond) for other, bond in adjacency[node] if other in extra]
        for node in extra
    }
    specs = []
    for component in _components(extra, pendant):
        links = []
        for node in component:
            links.extend(other for other, _ in adjacency[node] if other in core)
        if len(links) != 1:
            return None
        attachment = links[0]
        root = next(
            node
            for node in component
            if any(other == attachment for other, _ in adjacency[node])
        )
        word = _fragment_word(entity, set(component), root)
        if word is None:
            return None
        specs.append((attachment, word))
    return specs


def _lowest_bicyclo_locant(system: RingSystem, numbering: dict[str, int], attachment: str) -> int:
    locant = numbering[attachment]
    for path in system.paths:
        internals = path[1:-1]
        if attachment not in internals:
            continue
        path_locants = [numbering[node] for node in internals]
        reflected = path_locants[0] + path_locants[-1] - locant
        return min(locant, reflected)
    return locant


def _simple_cycles(nodes: set[str], adjacency) -> list[tuple[str, ...]]:
    neighbours = {node: sorted(other for other, _ in adjacency.get(node, ())) for node in nodes}
    cycles: list[tuple[str, ...]] = []

    def walk(start: str, current: str, path: list[str], seen: set[str]):
        if len(cycles) > 2500:
            return
        for nxt in neighbours[current]:
            if nxt == start and len(path) >= 3:
                cycles.append(tuple(path))
                continue
            if nxt <= start or nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            walk(start, nxt, path, seen)
            path.pop()
            seen.remove(nxt)

    for start in sorted(nodes):
        if len(cycles) > 2500:
            break
        walk(start, start, [start], {start})
    return cycles


def _exterior_paths(start: str, goal: str, ring_nodes: set[str], ring_edges: set[frozenset[str]], adjacency):
    paths: list[tuple[str, ...]] = []
    if any(
        other == goal and _edge_key(start, goal) not in ring_edges
        for other, _ in adjacency.get(start, ())
    ):
        paths.append(())

    def walk(node: str, seen: set[str], interior: list[str]):
        if len(paths) > 32 or len(interior) > 8:
            return
        for nxt, _ in sorted(adjacency.get(node, ()), key=lambda item: item[0]):
            if nxt == goal:
                paths.append(tuple(interior))
                continue
            if nxt in ring_nodes or nxt in seen:
                continue
            seen.add(nxt)
            interior.append(nxt)
            walk(nxt, seen, interior)
            interior.pop()
            seen.remove(nxt)

    for other, _ in adjacency.get(start, ()):
        if other in ring_nodes:
            continue
        walk(other, {start, other}, [other])
    # The length-zero chord and the positive paths are both useful; duplicate
    # interiors are removed so each bridge is scored once.
    unique: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for path in paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique.append(path)
    return unique


def _bridge_candidates(numbering: dict[str, int], adjacency, used: set[frozenset[str]]):
    included = set(numbering)
    found = []
    seen: set[tuple] = set()

    def consider(path: list[str]):
        if len(path) < 2 or path[0] not in numbering or path[-1] not in numbering:
            return
        forward = tuple(path)
        reverse = tuple(reversed(path))
        key = forward if forward <= reverse else reverse
        if key in seen:
            return
        seen.add(key)
        start, finish = path[0], path[-1]
        high = start if numbering[start] >= numbering[finish] else finish
        interior = path[1:-1] if high == start else list(reversed(path[1:-1]))
        low = min(numbering[start], numbering[finish])
        high_locant = max(numbering[start], numbering[finish])
        found.append((low, high_locant, tuple(interior), path))

    def walk(start: str, node: str, path: list[str], seen_nodes: set[str]):
        if len(found) > 64 or len(path) > 10:
            return
        for nxt, _ in adjacency.get(node, ()):
            edge = _edge_key(node, nxt)
            if edge in used:
                continue
            if nxt in included:
                if nxt != start:
                    consider(path + [nxt])
                continue
            if nxt in seen_nodes:
                continue
            seen_nodes.add(nxt)
            path.append(nxt)
            walk(start, nxt, path, seen_nodes)
            path.pop()
            seen_nodes.remove(nxt)

    for start in included:
        walk(start, start, [start], {start})
    found.sort(key=lambda item: (item[0], item[1], len(item[2]), tuple(item[3])))
    return found


def _select_von_baeyer(entity: FiniteChemicalEntity, attachments: tuple[str, ...] = ()):
    """Choose a deterministic von Baeyer parent for a bridged core of rank >= 3."""
    nodes = set(_heavy(entity))
    if len(nodes) > 16 or (entity.net_charge or 0) != 0:
        return None
    adjacency = _adj(entity, nodes)
    if _cycle_rank(nodes, adjacency) < 3 or len(_components(nodes, adjacency)) != 1:
        return None
    if any(bond.aromatic for bond in entity.bonds if bond.atom1_id in nodes and bond.atom2_id in nodes):
        return None
    cycles = _simple_cycles(nodes, adjacency)
    if len(cycles) > 2500:
        return None
    by_length: dict[int, list[tuple[str, ...]]] = {}
    for cycle in cycles:
        by_length.setdefault(len(cycle), []).append(cycle)
    best = None
    for length in sorted(by_length, reverse=True):
        for cycle in by_length[length]:
            ring_nodes = set(cycle)
            for left in range(length):
                for right in range(left + 1, length):
                    forward = list(cycle[left + 1:right])
                    backward = list(cycle[right + 1:]) + list(cycle[:left])
                    if len(forward) >= len(backward):
                        orientations = ((cycle[left], forward, cycle[right], backward),)
                    else:
                        orientations = ((cycle[right], backward, cycle[left], list(reversed(forward))),)
                    # Equal arms still need both directions; unequal arms need
                    # the opposite bridgehead as locant 1 as well.
                    expanded = []
                    for h1, long_arc, h2, short_arc in orientations:
                        expanded.append((h1, long_arc, h2, short_arc))
                        expanded.append((h2, list(reversed(long_arc)), h1, list(reversed(short_arc))))
                    for h1, long_arc, h2, short_arc in expanded:
                        if len(long_arc) < len(short_arc):
                            continue
                        ring_seq = [h1, *long_arc, h2, *short_arc]
                        ring_edges = {
                            _edge_key(a, b) for a, b in zip(ring_seq, ring_seq[1:] + [h1])
                        }
                        for interior in _exterior_paths(h1, h2, ring_nodes, ring_edges, adjacency):
                            bridge_seq = [h1, *interior, h2]
                            main_edges = {
                                _edge_key(a, b) for a, b in zip(bridge_seq, bridge_seq[1:])
                            }
                            if main_edges & ring_edges:
                                continue
                            numbering: dict[str, int] = {}
                            cursor = 1
                            numbering[h1] = cursor
                            for node in long_arc:
                                cursor += 1
                                numbering[node] = cursor
                            cursor += 1
                            numbering[h2] = cursor
                            for node in short_arc:
                                cursor += 1
                                numbering[node] = cursor
                            for node in interior:
                                cursor += 1
                                numbering[node] = cursor
                            used = set(ring_edges) | set(main_edges)
                            secondaries = []
                            covered = True
                            while True:
                                pending = _bridge_candidates(numbering, adjacency, used)
                                if not pending:
                                    break
                                low, high, extra, path = pending[0]
                                for node in extra:
                                    if node in numbering:
                                        covered = False
                                        break
                                    cursor += 1
                                    numbering[node] = cursor
                                if not covered:
                                    break
                                for a, b in zip(path, path[1:]):
                                    used.add(_edge_key(a, b))
                                secondaries.append((len(extra), low, high))
                            edge_count = sum(len(values) for values in adjacency.values()) // 2
                            if not covered or set(numbering) != nodes or len(used) != edge_count:
                                continue
                            parts = [str(len(long_arc)), str(len(short_arc)), str(len(interior))]
                            parts.extend(f"{size}^{low},{high}" for size, low, high in secondaries)
                            bracket = ".".join(parts)
                            ring_count = len(secondaries) + 2
                            if ring_count not in _RING_WORDS:
                                continue
                            try:
                                stem = _stem(len(nodes))
                            except ValueError:
                                continue
                            hetero = []
                            blocked = False
                            for atom in entity.atoms:
                                if atom.atom_id not in numbering or atom.element in {"C", "H"}:
                                    continue
                                prefix = _HETERO_PREFIX.get(atom.element)
                                if prefix is None:
                                    blocked = True
                                    break
                                hetero.append((numbering[atom.atom_id], prefix))
                            if blocked:
                                continue
                            hetero.sort()
                            hetero_text = "-".join(f"{locant}-{prefix}" for locant, prefix in hetero)
                            locants = _locants_for_edges(entity, used, numbering)
                            suffix = _suffix(stem, locants)
                            name = f"{_RING_WORDS[ring_count]}cyclo[{bracket}]{suffix}"
                            if hetero_text:
                                name = f"{hetero_text}{name}"
                            attachment_locants = tuple(sorted(numbering[node] for node in attachments if node in numbering))
                            score = (
                                -length,
                                -len(interior),
                                abs(len(long_arc) - len(short_arc)),
                                tuple((low, high) for _, low, high in secondaries),
                                bracket,
                                attachment_locants,
                                name,
                            )
                            if best is None or score < best[0]:
                                best = (score, name, dict(numbering))
        if best is not None and -best[0][0] == length:
            break
    if best is None:
        return None
    return best[1], best[2]


def _von_baeyer_result(entity: FiniteChemicalEntity, attachments: tuple[str, ...] = ()):
    # Substituent locants depend on the selected parent orientation.  Until
    # that orientation is carried through the reverse parser, fail closed for
    # substituted rank-3 systems rather than emitting a mismatched name.
    if attachments:
        return None
    selected = _select_von_baeyer(entity, attachments)
    if selected is None:
        return None
    name, _numbering = selected
    # The selector works on graph candidates, while the reverse parser uses a
    # fixed von Baeyer numbering.  Normalize the selected spelling through the
    # parser once so a generated rank-3 name is canonical and self-reversible.
    parsed = _parse_von_baeyer(name)
    if parsed is not None:
        canonical_selected = _select_von_baeyer(parsed, attachments=())
        if canonical_selected is not None and canonical_selected[0] != name:
            name = canonical_selected[0]
    return (
        name,
        False,
        (
            "Select the main ring and main bridge by the von Baeyer seniority rules.",
            "Cite the remaining bridges with the locants of their endpoints.",
        ),
    )


def _with_substituent_prefix(core: FiniteChemicalEntity, named, specs, attachments: tuple[str, ...]):
    system = classify_ring_system(core)
    if system is not None and system.kind == "bicyclo":
        numbering = _number_bicyclo_paths(system.paths, system.bridgeheads)
        locants = [
            (_lowest_bicyclo_locant(system, numbering, atom), word)
            for atom, word in specs
        ]
    else:
        selected = _select_von_baeyer(core, attachments)
        if selected is None:
            return None
        numbering = selected[1]
        try:
            locants = [(numbering[atom], word) for atom, word in specs]
        except KeyError:
            return None
    locants.sort()
    prefix = "-".join(f"{locant}-{word}" for locant, word in locants)
    return (
        prefix + ("-" if named[0][:1].isdigit() else "") + named[0],
        False,
        (
            "Detach acyclic substituents from the ring core.",
            "Name each substituent as an alkyl or alkanoyl prefix at its parent locant.",
        ),
    )


def name_polycycle(entity: FiniteChemicalEntity):
    """Return ``(name, preferred, trace)`` for a supported ring system."""
    core_nodes = _two_core_nodes(entity)
    heavy = _heavy(entity)
    specs: list[tuple[str, str]] = []
    core = entity
    if heavy != core_nodes:
        if len(core_nodes) < 3:
            return None
        found = _substituent_specs(entity, core_nodes)
        if found is None:
            return None
        specs = found
        core = _induced_entity(entity, core_nodes)
    attachments = tuple(atom for atom, _word in specs)
    named = _name_bare_polycycle(core, attachments)
    if named is None or not specs:
        return named
    return _with_substituent_prefix(core, named, specs, attachments)


def _name_bare_polycycle(entity: FiniteChemicalEntity, attachments: tuple[str, ...] = ()):
    system = classify_ring_system(entity)
    if system is None:
        return _von_baeyer_result(entity, attachments)
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
        # The bracket entries are the counts of non-spiro atoms and are
        # cited in ascending order (Blue Book SP-1.1).  Numbering follows the
        # same small-ring-first order, with the shared atom between the two
        # walks.
        ordered_paths = tuple(sorted(system.paths, key=lambda path: (len(path) - 2, path)))
        sizes = [len(path) - 2 for path in ordered_paths]
        count = len(system.atoms)
        stem = _stem(count)
        bracket = ".".join(map(str, sizes))
        numbering = _best_spiro_numbering(entity, ordered_paths)
        locants = _unsaturation_locants(entity, ordered_paths, numbering)
        if system.aromatic:
            # Aromatic spiro graphs are rare; preserving the graph still takes
            # precedence over inventing a retained parent.  Mark all edges as
            # aromatic in the parser using a pentaene suffix where applicable.
            locants = tuple(range(1, min(count, 6), 2))
        hetero = _hetero_prefixes_for_paths(entity, ordered_paths, numbering)
        if hetero is None:
            return None
        hetero_text = _serialize_hetero_prefixes(hetero)
        return (
            f"{hetero_text}spiro[{bracket}]{_suffix(stem, locants, aromatic=system.aromatic)}",
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
    hetero = _hetero_prefixes_for_paths(entity, paths, numbering)
    if hetero is None:
        return None
    if system.aromatic:
        # Aromatic fused benzene systems are represented in von Baeyer form.
        # The five alternating bonds are a stable serialization; the parser
        # restores aromatic bonds rather than a particular Kekule form.
        locants = tuple(range(1, len(system.atoms), 2))
    else:
        locants = _unsaturation_locants(entity, paths, numbering)
    prefix = _serialize_hetero_prefixes(hetero)
    return (
        f"{prefix}bicyclo[{bracket}]{_suffix(stem, locants, aromatic=system.aromatic)}",
        False,
        (
            "Classify the cyclic graph as three internally disjoint bridgehead paths.",
            "Use von Baeyer bicyclo notation with the bridgehead paths ordered by length.",
        ),
    )


def _locants_for_edges(entity: FiniteChemicalEntity, edges: set[frozenset[str]], numbering: dict[str, int]):
    locants = []
    for edge in edges:
        left, right = tuple(edge)
        bond = _bond_between(entity, left, right)
        if bond is not None and not bond.aromatic and abs((bond.order or 0.0) - 2.0) < 1e-8:
            locants.append(min(numbering[left], numbering[right]))
    return tuple(sorted(set(locants)))


def _evidence():
    return (Evidence(EvidenceSource.IUPAC_NAME, "self_contained_polycycle_parser"),)


def _atom_record(
    atom_id: str,
    element: str = "C",
    hydrogens: int | None = None,
    *,
    formal_charge: int | None = None,
):
    return ChemicalAtom(
        atom_id=atom_id,
        element=element,
        implicit_hydrogens=hydrogens,
        formal_charge=formal_charge,
        evidence=_evidence(),
    )


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
        net_charge=sum(atom.formal_charge or 0 for atom in completed),
        status=InferenceStatus.EXPLICIT,
        evidence=_evidence(),
    )


_HETERO_PREFIXES = ("aza", "oxa", "thia", "azonia", "azanium")
_HETERO_PATTERN = r"azanium|azonia|thia|oxa|aza"
_HETERO_TEXT = rf"(?:(?:\d+-(?:{_HETERO_PATTERN})(?:-\d+-(?:{_HETERO_PATTERN}))*)?)"
_HETERO_TEXT_PATTERN = rf"(?P<hetero>(?:\d+-(?:{_HETERO_PATTERN})(?:-\d+-(?:{_HETERO_PATTERN}))*)?)"


def _hetero_spec(prefix: str):
    if prefix == "aza":
        return "N", None, None
    if prefix == "oxa":
        return "O", None, None
    if prefix == "thia":
        return "S", None, None
    if prefix == "azonia":
        return "N", 1, None
    if prefix == "azanium":
        return "N", 1, 1
    raise PolycycleParseError(f"unsupported skeletal-replacement prefix: {prefix}")


def _parse_hetero_text(text: str):
    if not text:
        return ()
    pattern = rf"(?:(\d+)-({_HETERO_PATTERN})-)*(\d+)-({_HETERO_PATTERN})"
    match = re.fullmatch(pattern, text)
    if match is None:
        raise PolycycleParseError("malformed skeletal-replacement prefix")
    values = []
    for locant, prefix in re.findall(rf"(\d+)-({_HETERO_PATTERN})", text):
        values.append((int(locant), prefix))
    return tuple(values)


def _apply_hetero_specs(atoms, hetero_text: str, numbering: dict[str, int], total: int):
    for locant, prefix in _parse_hetero_text(hetero_text):
        if not 1 <= locant <= total:
            raise PolycycleParseError("heteroatom locant outside polycyclic parent")
        target = next((atom_id for atom_id, value in numbering.items() if value == locant), None)
        if target is None:
            raise PolycycleParseError("heteroatom locant outside polycyclic parent")
        element, charge, hydrogens = _hetero_spec(prefix)
        index = next(i for i, atom in enumerate(atoms) if atom.atom_id == target)
        atoms[index] = _atom_record(target, element, hydrogens, formal_charge=charge)


def _parse_bicyclo(name: str):
    match = re.fullmatch(
        _HETERO_TEXT_PATTERN + r"bicyclo"
        r"\[(\d+)\.(\d+)\.(\d+)\](.+)",
        name,
    )
    if match is None:
        return None
    hetero_text = match.group("hetero") or ""
    bridge_counts = tuple(int(match.group(index)) for index in (2, 3, 4))
    body = match.group(5)
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
    # A locant denotes the consecutive parent bond beginning at that atom.
    # Bridgehead/ring-closure edges can share the same lower endpoint but are
    # not the numbered parent edge; accepting them all would duplicate one
    # alkene and overvalence the bridgehead.
    edge_by_locant = {}
    if not aromatic and locants:
        for locant in locants:
            candidates = []
            for path in paths:
                for left, right in zip(path, path[1:]):
                    left_locant, right_locant = numbering[left], numbering[right]
                    if (
                        min(left_locant, right_locant) == locant
                        and abs(left_locant - right_locant) == 1
                    ):
                        candidates.append((left, right))
            if len(candidates) != 1:
                raise PolycycleParseError(
                    "bicyclo unsaturation locant does not identify one parent bond"
                )
            edge_by_locant[locant] = frozenset(candidates[0])
    for path in paths:
        for left, right in zip(path, path[1:]):
            # Blue Book locants identify the lower-numbered atom of each
            # multiple bond.  Numbering follows the same three-path order as
            # :func:`name_polycycle`, making this assignment reversible.
            order = 1.5 if aromatic else 1.0
            if not aromatic and locants:
                edge = frozenset((left, right))
                if edge in edge_by_locant.values():
                    order = 2.0
            bonds.append(_edge(left, right, order, aromatic=aromatic))
    _apply_hetero_specs(
        atoms,
        hetero_text,
        _number_bicyclo_paths(paths, ("B1", "B2")),
        count,
    )
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
    match = re.fullmatch(
        _HETERO_TEXT_PATTERN + r"spiro"
        r"\[(\d+)\.(\d+)\](.+)",
        name,
    )
    if match is None:
        return None
    hetero_text = match.group("hetero") or ""
    ring_counts = tuple(int(match.group(index)) for index in (2, 3))
    body = match.group(4)
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
    # The descriptor is already in ascending ring-size order.  Number the
    # smaller ring first, then the shared spiro atom, then the larger ring.
    # This mirrors :func:`_number_spiro_paths` and makes heteroatom and ene
    # locants independently reversible.
    numbering: dict[str, int] = {}
    next_number = 1
    ring_values = []
    for ring_index, internal_count in enumerate(ring_counts):
        values = ["S"]
        for atom_index in range(internal_count):
            atom_id = f"R{ring_index + 1}_{atom_index + 1}"
            atoms.append(_atom_record(atom_id))
            values.append(atom_id)
        values.append("S")
        ring_values.append(values)
        for atom_id in values[1:-1]:
            numbering[atom_id] = next_number
            next_number += 1
        if ring_index == 0:
            numbering["S"] = next_number
            next_number += 1
    edge_by_locant = {}
    if not aromatic and locants:
        for locant in locants:
            candidates = []
            for values in ring_values:
                for left, right in zip(values, values[1:]):
                    left_locant, right_locant = numbering[left], numbering[right]
                    if (
                        min(left_locant, right_locant) == locant
                        and abs(left_locant - right_locant) == 1
                    ):
                        candidates.append((left, right))
            if len(candidates) != 1:
                raise PolycycleParseError(
                    "spiro unsaturation locant does not identify one parent bond"
                )
            edge_by_locant[locant] = frozenset(candidates[0])
    for values in ring_values:
        for left, right in zip(values, values[1:]):
            order = 1.5 if aromatic else 1.0
            if not aromatic and locants:
                if frozenset((left, right)) in edge_by_locant.values():
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
    _apply_hetero_specs(atoms, hetero_text, numbering, count)
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


def _stem_number(stem: str) -> int | None:
    for number, candidate in _STEMS.items():
        if candidate == stem:
            return number
    return None


def _split_ring_prefixes(name: str):
    match = re.search(
        rf"{_HETERO_TEXT}(?:bi|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca)cyclo\[|"
        rf"{_HETERO_TEXT}spiro\[|phenylbenzene",
        name,
    )
    if match is None:
        return [], name
    prefix_text = name[: match.start()].strip("-")
    if not prefix_text:
        return [], name[match.start() :]
    found = re.findall(r"(\d+)-([a-z]+)", prefix_text)
    if "-".join(f"{locant}-{word}" for locant, word in found) != prefix_text:
        return [], name
    return [(int(locant), word) for locant, word in found], name[match.start() :]


def _interpret_prefix(word: str):
    if word.endswith("anoyl"):
        stem, kind = word[: -len("anoyl")], "acyl"
    elif word.endswith("yl"):
        stem, kind = word[: -len("yl")], "alkyl"
    else:
        return None
    count = _stem_number(stem)
    if count is None:
        return None
    return kind, count


def _parent_locants(entity: FiniteChemicalEntity) -> dict[int, str] | None:
    numbered = {
        int(atom.atom_id[1:]): atom.atom_id
        for atom in entity.atoms
        if re.fullmatch(r"N\d+", atom.atom_id)
    }
    if numbered:
        return numbered
    system = classify_ring_system(entity)
    if system is None or system.kind != "bicyclo":
        return None
    numbering = _number_bicyclo_paths(system.paths, system.bridgeheads)
    locants = {locant: atom_id for atom_id, locant in numbering.items()}
    # ``name_polycycle`` uses the lowest locant on a symmetric bridge.  When
    # reversing that convention, choose the reflected (higher numbered) atom;
    # it is the canonical representative and remains constitutionally
    # equivalent under the bridge symmetry.
    for path in system.paths:
        internals = path[1:-1]
        if not internals:
            continue
        values = [numbering[node] for node in internals]
        for node, value in zip(internals, values):
            reflected = values[0] + values[-1] - value
            low = min(value, reflected)
            candidate = node if value >= reflected else internals[values.index(reflected)]
            locants[low] = candidate
    return locants


def _attach_substituents(entity: FiniteChemicalEntity, prefixes, name: str):
    locants = _parent_locants(entity)
    if locants is None:
        raise PolycycleParseError("substituent locant is outside the ring parent")
    atoms = list(entity.atoms)
    bonds = list(entity.bonds)
    for index, (locant, word) in enumerate(prefixes):
        interpreted = _interpret_prefix(word)
        target = locants.get(locant) if interpreted is not None else None
        if interpreted is None or target is None:
            raise PolycycleParseError("unsupported ring substituent prefix")
        kind, count = interpreted
        parent = next(atom for atom in atoms if atom.atom_id == target)
        hydrogen = max(0, (parent.implicit_hydrogens or 0) - 1) or None
        atoms[atoms.index(parent)] = replace(parent, implicit_hydrogens=hydrogen)
        if kind == "alkyl":
            previous = target
            for offset in range(count):
                atom_id = f"Y{index}_{offset}"
                atoms.append(_atom_record(atom_id, "C"))
                bonds.append(_edge(previous, atom_id))
                previous = atom_id
            continue
        carbonyl = f"Y{index}_0"
        atoms.append(_atom_record(carbonyl, "C"))
        bonds.append(_edge(target, carbonyl))
        atoms.append(_atom_record(f"O{index}", "O"))
        bonds.append(_edge(carbonyl, f"O{index}", 2.0))
        previous = carbonyl
        for offset in range(count - 1):
            atom_id = f"Y{index}_{offset + 1}"
            atoms.append(_atom_record(atom_id, "C"))
            bonds.append(_edge(previous, atom_id))
            previous = atom_id
    return _finish_entity(name, atoms, bonds)


def _parse_von_baeyer(name: str):
    match = re.fullmatch(
        rf"(?:((?:\d+-(?:{_HETERO_PATTERN})-)*\d+-(?:{_HETERO_PATTERN})))?"
        r"(bi|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca)cyclo"
        r"\[([0-9^.,]+)\]"
        r"([a-z]+?)"
        r"(ane|-(?:\d+(?:,\d+)*)-(?:di|tri|tetra|penta)?ene)",
        name,
    )
    if match is None:
        return None
    hetero_text, ring_word, body, stem, suffix = match.groups()
    if ring_word == "bi" and "^" not in body:
        return None
    parts = body.split(".")
    if len(parts) < 3 or any(not part for part in parts[:3]) or any(not part.isdigit() for part in parts[:3]):
        return None
    main = tuple(int(part) for part in parts[:3])
    secondaries = []
    for part in parts[3:]:
        secondary = re.fullmatch(r"(\d+)\^(\d+),(\d+)", part)
        if secondary is None:
            return None
        length, low, high = (int(secondary.group(index)) for index in (1, 2, 3))
        if low > high:
            low, high = high, low
        secondaries.append((length, low, high))
    if _WORD_RINGS[ring_word] != 2 + len(secondaries):
        raise PolycycleParseError("von Baeyer ring count does not match the bridges")
    total = 2 + sum(main) + sum(length for length, _, _ in secondaries)
    if _stem_number(stem) != total:
        raise PolycycleParseError("von Baeyer stem does not match the bridge atom count")
    atoms = [_atom_record(f"N{index}") for index in range(1, total + 1)]
    bonds: list[ChemicalBond] = []
    a, b, c = main
    second = 2 + a
    ring_end = second + b

    def link(left: int, right: int, order: float = 1.0):
        bonds.append(_edge(f"N{left}", f"N{right}", order))

    for locant in range(1, second):
        link(locant, locant + 1)
    for locant in range(second, ring_end):
        link(locant, locant + 1)
    if b:
        link(ring_end, 1)
    elif a:
        link(second, 1)
    if c == 0:
        if a and b:
            link(1, second)
    else:
        first = ring_end + 1
        last = ring_end + c
        link(1, first)
        for locant in range(first, last):
            link(locant, locant + 1)
        link(last, second)
    cursor = ring_end + c + 1
    for length, low, high in secondaries:
        if not 1 <= low < high <= total:
            raise PolycycleParseError("von Baeyer bridge locant is outside the parent")
        if length == 0:
            link(low, high)
            continue
        link(high, cursor)
        for locant in range(cursor, cursor + length - 1):
            link(locant, locant + 1)
        link(cursor + length - 1, low)
        cursor += length
    numbering = {f"N{index}": index for index in range(1, total + 1)}
    _apply_hetero_specs(atoms, hetero_text or "", numbering, total)
    unsaturated = ()
    if suffix != "ane":
        raw = re.fullmatch(r"-(\d+(?:,\d+)*)-(?:di|tri|tetra|penta)?ene", suffix)
        if raw is None:
            raise PolycycleParseError("unsupported von Baeyer unsaturation")
        unsaturated = tuple(int(value) for value in raw.group(1).split(","))
        represented = set()
        for bond in bonds:
            left = int(bond.atom1_id[1:])
            right = int(bond.atom2_id[1:])
            locant = min(left, right)
            if locant in unsaturated:
                bond_index = bonds.index(bond)
                bonds[bond_index] = _edge(bond.atom1_id, bond.atom2_id, 2.0)
                represented.add(locant)
        if represented != set(unsaturated):
            raise PolycycleParseError("von Baeyer unsaturation locant is outside the parent")
    return _finish_entity(name, atoms, bonds)


def _parse_ring_parent(name: str):
    if name == "phenylbenzene":
        return _parse_phenylbenzene(name)
    if name.startswith("spiro[") or re.match(
        rf"\d+-(?:{_HETERO_PATTERN})(?:-\d+-(?:{_HETERO_PATTERN}))*spiro\[",
        name,
    ):
        return _parse_spiro(name)
    if name.startswith("bicyclo[") or re.match(
        rf"\d+-(?:{_HETERO_PATTERN})(?:-\d+-(?:{_HETERO_PATTERN}))*bicyclo\[",
        name,
    ):
        return _parse_bicyclo(name)
    return _parse_von_baeyer(name)


def parse_polycycle_name(name: str):
    """Parse a canonical name emitted by :func:`name_polycycle`."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    normalized = " ".join(name.strip().lower().split())
    prefixes, parent = _split_ring_prefixes(normalized)
    parsed = _parse_ring_parent(parent)
    if parsed is None:
        return None
    if not prefixes:
        return parsed
    return _attach_substituents(parsed, prefixes, normalized)


__all__ = [
    "PolycycleParseError",
    "RingSystem",
    "classify_ring_system",
    "name_polycycle",
    "parse_polycycle_name",
]
