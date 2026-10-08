"""Self-contained substitutive naming for finite acyclic organic graphs.

The functions in this module intentionally return the small tuple consumed by
``naming._organic_result``.  Keeping graph perception here makes the naming
dispatcher easy to extend while preserving the existing public result type.
"""

from __future__ import annotations

from collections import Counter, deque

from ..models import BondKind, FiniteChemicalEntity


HALOGENS = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}
HALOGEN_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}


def _stems():
    # Import lazily to avoid a naming -> substitutive import cycle.
    from ..naming import ALKANE_STEMS
    return ALKANE_STEMS


def alkane_stem(count: int) -> str | None:
    """Return a Blue Book alkane stem, extending the shared table on demand."""
    stems = _stems()
    if count in stems:
        return stems[count]
    # A deterministic extension for chains beyond the precomputed C100 table.
    # The common acceptance range is covered by the table; this branch avoids a
    # hard length refusal while retaining valid compositional spelling.
    if count < 1:
        return None
    units = {1: "hen", 2: "do", 3: "tri", 4: "tetra", 5: "penta",
             6: "hexa", 7: "hepta", 8: "octa", 9: "nona"}
    tens = {20: "icos", 30: "triacont", 40: "tetracont", 50: "pentacont",
            60: "hexacont", 70: "heptacont", 80: "octacont", 90: "nonacont"}
    if 13 <= count <= 19:
        stem = {13: "tridec", 14: "tetradec", 15: "pentadec", 16: "hexadec",
                17: "heptadec", 18: "octadec", 19: "nonadec"}[count]
    elif 20 <= count <= 99:
        ten, unit = divmod(count, 10)
        base = tens.get(ten * 10)
        stem = base if unit == 0 else units[unit] + base if base else None
    else:
        stem = None
    if stem is not None:
        stems[count] = stem
    return stem


def _adjacency(entity):
    result = {atom.atom_id: [] for atom in entity.atoms}
    for bond in entity.bonds:
        result[bond.atom1_id].append((bond.atom2_id, bond))
        result[bond.atom2_id].append((bond.atom1_id, bond))
    return result


def _atom_map(entity):
    return {atom.atom_id: atom for atom in entity.atoms}


def _heavy_adjacency(entity):
    atoms = _atom_map(entity)
    adjacency = _adjacency(entity)
    return {
        atom_id: [(neighbor, bond) for neighbor, bond in values if atoms[neighbor].element != "H"]
        for atom_id, values in adjacency.items() if atoms[atom_id].element != "H"
    }


def _hcount(entity, atom_id):
    atoms = _atom_map(entity)
    return sum(
        atoms[n].element == "H" for n, _ in _adjacency(entity)[atom_id]
    ) + (atoms[atom_id].explicit_hydrogens or 0) + (atoms[atom_id].implicit_hydrogens or 0)


def _bond_between(adjacency, left, right):
    return next((bond for neighbor, bond in adjacency[left] if neighbor == right), None)


def _is_single(bond):
    return bond is not None and bond.kind in {BondKind.COVALENT, BondKind.UNKNOWN} and bond.order == 1.0


def _is_double(bond):
    return bond is not None and bond.order == 2.0


def _tree_path(adjacency, start, end):
    queue = deque([(start, (start,))])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, (*path, neighbor)))
    return ()


def _carbon_parent(entity, *, required=None):
    """Find a longest acyclic carbon path and its parent-side numbering."""
    atoms = _atom_map(entity)
    adjacency = _heavy_adjacency(entity)
    carbons = {a for a, atom in atoms.items() if atom.element == "C"}
    if not carbons or any(
        not _is_single(bond)
        for a in carbons
        for neighbor, bond in adjacency[a]
        if neighbor in carbons
    ):
        return None
    if any(sum(neighbor in carbons for neighbor, _ in adjacency[a]) > 2 for a in carbons):
        # Branches can still be present; only reject non-tree carbon graphs.
        pass
    carbon_graph = {a: [n for n, _ in adjacency[a] if n in carbons] for a in carbons}
    if required is not None and required in carbon_graph:
        component = {required}
        pending = [required]
        while pending:
            current = pending.pop()
            for neighbor in carbon_graph[current]:
                if neighbor not in component:
                    component.add(neighbor)
                    pending.append(neighbor)
        carbon_graph = {a: [n for n in carbon_graph[a] if n in component] for a in component}
        carbons = component
    edges = sum(len(v) for v in carbon_graph.values()) // 2
    if edges != len(carbons) - 1:
        return None
    endpoints = [a for a, values in carbon_graph.items() if len(values) <= 1]
    if len(carbons) == 1:
        paths = [(next(iter(carbons)),)]
    else:
        candidates = []
        for left in endpoints:
            for right in endpoints:
                if left < right:
                    path = _tree_path(carbon_graph, left, right)
                    if required is None or required in path:
                        candidates.append(path)
        if not candidates:
            return None
        longest = max(len(p) for p in candidates)
        paths = [p for p in candidates if len(p) == longest]
    # Substituent locants are the first numbering criterion.  If several paths
    # tie, lexical atom ids give a stable result independent of parser order.
    scored = []
    for path in paths:
        for ordered in (path, tuple(reversed(path))):
            numbering = {atom_id: index + 1 for index, atom_id in enumerate(ordered)}
            branch_locs = []
            for atom_id in ordered:
                branch_locs.extend(
                    numbering[atom_id]
                    for neighbor in carbon_graph[atom_id]
                    if neighbor not in numbering
                )
            required_locant = numbering.get(required, 0) if required is not None else 0
            scored.append((required_locant, tuple(sorted(branch_locs)), tuple(ordered), numbering, carbon_graph))
    chosen = min(scored, key=lambda item: (item[0], item[1], tuple(item[2])))
    return chosen[1], chosen[2], chosen[3], chosen[4]


def _prefix_string(prefixes):
    grouped = {}
    for locant, prefix in prefixes:
        grouped.setdefault(prefix, []).append(locant)
    words = []
    for prefix in sorted(grouped):
        locants = sorted(grouped[prefix])
        multiplier = {1: "", 2: "di", 3: "tri"}.get(len(locants), f"{len(locants)}-")
        words.append(f"{','.join(map(str, locants))}-{multiplier}{prefix}")
    return "-".join(words)


def _result(name, *trace):
    return name, True, *trace


def _special_patterns(entity):
    atoms = _atom_map(entity)
    adjacency = _heavy_adjacency(entity)
    counts = Counter(atom.element for atom in atoms.values() if atom.element != "H")
    heavy = set(atoms) - {a for a, atom in atoms.items() if atom.element == "H"}
    if counts == Counter({"C": 1, "O": 2}):
        carbon = next(a for a in heavy if atoms[a].element == "C")
        if all(_is_double(_bond_between(adjacency, carbon, n)) for n, _ in adjacency[carbon]):
            return _result("carbon dioxide", "Recognize the retained carbon dioxide name.")
    if counts == Counter({"C": 1, "O": 3}):
        carbon = next(a for a in heavy if atoms[a].element == "C")
        oxygens = adjacency[carbon]
        if sum(_is_double(b) for _, b in oxygens) == 1 and sum(_is_single(b) and _hcount(entity, n) > 0 for n, b in oxygens) == 2:
            return _result("carbonic acid", "Recognize the retained carbonic acid pattern.")
    if counts == Counter({"C": 1, "N": 1, "O": 1}):
        carbon = next(a for a in heavy if atoms[a].element == "C")
        neighbours = adjacency[carbon]
        if len(neighbours) == 2 and any(atoms[n].element == "N" and _is_single(b) for n, b in neighbours) and any(atoms[n].element == "O" and _is_double(b) for n, b in neighbours):
            return _result("formamide", "Recognize the retained formamide pattern.")
        if len(neighbours) == 2 and any(atoms[n].element == "N" and _is_double(b) for n, b in neighbours) and any(atoms[n].element == "O" and _is_double(b) for n, b in neighbours):
            return _result("isocyanic acid", "Recognize the retained isocyanic acid pattern.")
    if counts.get("C") == 1 and counts.get("O") == 1 and sum(counts.get(x, 0) for x in HALOGENS) == 2 and sum(counts.values()) == 4:
        carbon = next(a for a, atom in atoms.items() if atom.element == "C")
        neighbours = adjacency[carbon]
        halogens = [atoms[n].element for n, b in neighbours if atoms[n].element in HALOGENS and _is_single(b)]
        if len(halogens) == 2 and sum(_is_double(b) for _, b in neighbours) == 1:
            prefix = {"F": "difluoride", "Cl": "dichloride", "Br": "dibromide", "I": "diiodide"}.get(halogens[0])
            if prefix and halogens[0] == halogens[1]:
                return _result(f"carbonyl {prefix}", "Recognize the retained carbonyl dihalide pattern.")
    return None


def _name_acid(entity):
    atoms = _atom_map(entity)
    adjacency = _heavy_adjacency(entity)
    for carbon, atom in atoms.items():
        if atom.element != "C":
            continue
        oxygens = [(n, b) for n, b in adjacency[carbon] if atoms[n].element == "O"]
        doubles = [n for n, b in oxygens if _is_double(b)]
        hydroxys = [n for n, b in oxygens if _is_single(b) and _hcount(entity, n) > 0]
        if len(doubles) != 1 or len(hydroxys) != 1:
            continue
        parent = _carbon_parent(entity, required=carbon)
        if parent is None or parent[1][0] != carbon:
            continue
        ordered, numbering = parent[1], parent[2]
        # Every non-parent heavy atom must be the acid OH or a substituent OH.
        prefixes = []
        valid = True
        for c in ordered:
            for n, b in adjacency[c]:
                if n in numbering or n in doubles or n == hydroxys[0]:
                    continue
                if atoms[n].element == "O" and _is_single(b) and _hcount(entity, n) > 0:
                    prefixes.append((numbering[c], "hydroxy"))
                else:
                    valid = False
        if not valid:
            continue
        stem = alkane_stem(len(ordered))
        if stem is None:
            continue
        prefix = _prefix_string(prefixes)
        return _result(f"{prefix if prefix else ''}{stem}anoic acid", "Select the carboxylic acid parent chain.", "Assign hydroxy groups as detachable prefixes.")
    return None


def _name_alcohol(entity):
    atoms = _atom_map(entity)
    adjacency = _heavy_adjacency(entity)
    hydroxys = [
        (n, c) for n, atom in atoms.items() if atom.element == "O"
        for c, bond in adjacency[n] if _is_single(bond) and atoms[c].element == "C" and _hcount(entity, n) > 0
    ]
    if len(hydroxys) != 1:
        return None
    parent = _carbon_parent(entity, required=hydroxys[0][1])
    if parent is None:
        return None
    ordered, numbering = parent[1], parent[2]
    # Keep this stage's parent handling deliberately conservative for oxygen
    # and halogen substituents; the existing benzene rules cover aromatic cases.
    if any(atoms[a].element not in {"C", "O"} for a in atoms):
        return None
    locant = numbering[hydroxys[0][1]]
    stem = alkane_stem(len(ordered))
    if stem is None:
        return None
    if len(ordered) == 1:
        name = "methanol"
    elif len(ordered) == 2 and locant == 1:
        name = "ethanol"
    else:
        name = f"{stem}an-{locant}-ol"
    return _result(name, "Select the longest carbon chain containing the hydroxy-bearing carbon.", "Assign the hydroxy suffix the lowest locant.")


def _name_hydrocarbon(entity):
    atoms = _atom_map(entity)
    if any(atom.element not in {"C", "H"} for atom in atoms.values()):
        return None
    parent = _carbon_parent(entity)
    if parent is None:
        return None
    ordered, numbering, carbon_graph = parent[1], parent[2], parent[3]
    if any(not _is_single(_bond_between(_heavy_adjacency(entity), a, b)) for a in ordered for b in carbon_graph[a] if numbering.get(b, 0) == numbering.get(a, 0) + 1):
        return None
    prefixes = []
    for c in ordered:
        for branch in carbon_graph[c]:
            if branch in numbering:
                continue
            if len(carbon_graph[branch]) == 1:
                prefixes.append((numbering[c], "methyl"))
            else:
                return None
    stem = alkane_stem(len(ordered))
    if stem is None:
        return None
    prefix = _prefix_string(prefixes)
    return _result(f"{prefix if prefix else ''}{stem}ane", "Select the longest carbon parent chain.", "Number substituents to obtain the lowest locant sequence.")


def _acyl_parent(entity, carbonyl, excluded):
    atoms = _atom_map(entity)
    adjacency = _heavy_adjacency(entity)
    carbons = {a for a, atom in atoms.items() if atom.element == "C" and a not in excluded}
    graph = {a: [n for n, b in adjacency[a] if n in carbons and _is_single(b)] for a in carbons}
    if carbonyl not in graph:
        return None
    chain = [carbonyl]
    previous = None
    current = carbonyl
    while True:
        values = [n for n in graph[current] if n != previous]
        if not values:
            break
        if len(values) != 1:
            return None
        previous, current = current, values[0]
        chain.append(current)
    return chain


def _name_carbonyl_derivative(entity):
    atoms = _atom_map(entity)
    adjacency = _heavy_adjacency(entity)
    for carbonyl, atom in atoms.items():
        if atom.element != "C":
            continue
        oxygens = [(n, b) for n, b in adjacency[carbonyl] if atoms[n].element == "O" and _is_double(b)]
        if len(oxygens) != 1:
            continue
        single = [(n, b) for n, b in adjacency[carbonyl] if _is_single(b)]
        # Acyl halides.
        halides = [n for n, b in single if atoms[n].element in HALOGENS]
        if len(halides) == 1 and all(atoms[n].element in {"C", *HALOGENS} for n, _ in single):
            chain = _acyl_parent(entity, carbonyl, set())
            if chain is None:
                continue
            stem = alkane_stem(len(chain))
            if stem is not None:
                return _result(f"{stem}anoyl {HALOGENS[atoms[halides[0]].element]}", "Select the acyl chain and name the acid halide.")
        # Amides (the C2 retained name is required by the existing anilide API).
        nitrogens = [n for n, b in single if atoms[n].element == "N"]
        if len(nitrogens) == 1 and all(atoms[n].element in {"C", "N"} for n, _ in single):
            # N-substituted amides (for example the retained
            # N-(4-hydroxyphenyl)acetamide case) belong to the established
            # benzene/anilide recognizer and must not be collapsed to a plain
            # acetamide here.
            nitrogen = nitrogens[0]
            if any(neighbor != carbonyl for neighbor, _ in adjacency[nitrogen]):
                continue
            chain = _acyl_parent(entity, carbonyl, set())
            if chain is None:
                continue
            stem = alkane_stem(len(chain))
            if stem is None:
                continue
            parent = {1: "formamide", 2: "acetamide"}.get(len(chain), f"{stem}anamide")
            return _result(parent, "Select the carboxamide as the senior characteristic group.")
        # Esters: C(=O)-O-R.  The alcohol-side chain is named as an alkyl
        # prefix; this handles methyl ethanoate and its longer analogues.
        ester_o = [n for n, b in single if atoms[n].element == "O"]
        if len(ester_o) == 1 and all(atoms[n].element in {"C", "O"} for n, _ in single):
            oxygen = ester_o[0]
            alkyl = [n for n, b in adjacency[oxygen] if n != carbonyl and atoms[n].element == "C"]
            if len(alkyl) != 1:
                continue
            side = _carbon_parent(entity, required=alkyl[0])
            chain = _acyl_parent(entity, carbonyl, {oxygen})
            if side is None or chain is None:
                continue
            side_stem = alkane_stem(len(side[1]))
            acid_stem = alkane_stem(len(chain))
            if side_stem and acid_stem:
                alkyl_name = side_stem + ("yl" if len(side_stem) > 1 else "yl")
                return _result(f"{alkyl_name} {acid_stem}anoate", "Name the alcohol-derived alkyl group.", "Name the acid-derived ester parent.")
    return None


def name_acyclic(entity: FiniteChemicalEntity):
    """Return a naming tuple for acyclic structures, or ``None``."""
    if not isinstance(entity, FiniteChemicalEntity):
        return None
    heavy = [atom for atom in entity.atoms if atom.element != "H"]
    if not heavy:
        return None
    graph = _heavy_adjacency(entity)
    if len(heavy) > 1 and sum(len(values) for values in graph.values()) // 2 >= len(heavy):
        return None
    for recognizer in (_special_patterns, _name_carbonyl_derivative, _name_acid, _name_alcohol, _name_hydrocarbon):
        result = recognizer(entity)
        if result is not None:
            return result
    # Aldehydes are only introduced here for the one-carbon methanal pattern;
    # larger aldehydes are handled by later parent-functional-group stages.
    atoms = _atom_map(entity)
    adjacency = _heavy_adjacency(entity)
    if len(atoms) == 2 and {a.element for a in atoms.values()} == {"C", "O"}:
        carbon = next(a for a, atom in atoms.items() if atom.element == "C")
        oxygen = next(a for a, atom in atoms.items() if atom.element == "O")
        if _is_double(_bond_between(adjacency, carbon, oxygen)):
            return _result("methanal", "Select the one-carbon aldehyde parent.")
    return None


__all__ = ["alkane_stem", "name_acyclic"]
