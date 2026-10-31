"""Disorder policy for molecule identity and site-centred shells.

Alternative orientations of one crystallographic site are mutually exclusive.
Bond perception can still fuse them into a single component when they share an
ordered atom (disorder group 0) or when only partial occupancies distinguish
them. The fused component is not a molecule that exists in either orientation,
and a later moiety match then finds no ligand shell.

This module selects chemically allowed realizations without resolving a full
disorder ensemble:

* ``major`` keeps, within each exclusive family, the component with the
  largest occupancy sum. Ties break by atom count, then by component label
  (the greater label wins), then by the smaller atom index.
* ``exclusive`` emits one realization per alternative. Other families stay on
  their major component, so the result is one state per alternative rather
  than the full Cartesian product.
* ``as_refined`` keeps every refined site, including a fused component.

Labelled CIF assemblies (``_atom_site_disorder_assembly``) are exclusive by
definition: different disorder groups in one assembly cannot be occupied
together, even when those groups are far apart.

Unlabelled disorder groups are not coupled by the PART label alone. Atoms that
share a label are split into local clusters (within
``DISORDER_CONFIG["SYMMETRY_SITE_RADIUS"]``). Clusters of different labels
compete only when they approach within
``DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"]``. A distant second site therefore
keeps its own majority orientation.

Group-0 atoms that differ only by occupancy are alternatives only when the
occupancy values sum to at most one and some pair lies well below a covalent
bond (``OCCUPANCY_ALTERNATIVE_BOND_FRACTION`` of the bonding threshold).
Ordinary bonded neighbours with unequal partial occupancy stay together.
Equal occupancies use the same short-contact test inside the class; the
tie-break above picks the atom with the smaller index.

``find_polyhedra`` defaults to ``major`` for the central site, and shell
vertices always use the major ligand component. ``identify_molecules`` and
the structure readers default to ``as_refined`` so a general identification
does not drop refined atoms.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from ...constants.config import (
    DISORDER_CONFIG,
    KEY_ASSEMBLY,
    KEY_DISORDER_GROUP,
    KEY_OCCUPANCY,
    PARTIAL_OCCUPANCY_TOLERANCE,
)

DISORDER_POLICIES = ("major", "exclusive", "as_refined")
DEFAULT_DISORDER_POLICY = "major"
_BLANK_ASSEMBLY = {"", ".", "?"}


@dataclass(frozen=True)
class DisorderRealization:
    """One chemically allowed subset of a molecule or atom list.

    ``mask`` is True for atoms that belong to this realization. ``component_ids``
    lists the exclusive components that were selected; it is empty when no
    choice was required. ``occupancy_sum`` is the sum of refined occupancies
    inside the mask. ``drops_atoms`` is True when at least one refined site was
    left out. ``competes_for_site`` is True when this realization carries
    partial occupancy or a disorder group, so a site-centred search may prefer
    it over another realization of the same moiety at the same position.
    """

    mask: np.ndarray
    component_ids: tuple[str, ...]
    occupancy_sum: float
    drops_atoms: bool
    competes_for_site: bool


@dataclass(frozen=True)
class _Component:
    component_id: str
    label: str
    indices: np.ndarray
    occupancy_sum: float
    n_atoms: int


def normalize_disorder_policy(policy: str) -> str:
    """Return a canonical policy name, or raise ``ValueError``."""
    if not isinstance(policy, str):
        raise TypeError(
            f"disorder_policy must be a string, got {type(policy).__name__}."
        )
    canonical = policy.strip().lower()
    if canonical not in DISORDER_POLICIES:
        raise ValueError(
            f"disorder_policy must be one of {DISORDER_POLICIES}, got {policy!r}."
        )
    return canonical


def disorder_realizations(
    atoms,
    policy: str = DEFAULT_DISORDER_POLICY,
    *,
    spatial_alternatives: bool = True,
) -> list[DisorderRealization]:
    """Return the realizations of ``atoms`` allowed by ``policy``.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` or ``CrystalMolecule``. Occupancy, disorder group, and
        assembly arrays are read when present.
    policy
        ``major``, ``exclusive``, or ``as_refined``.
    spatial_alternatives
        When True, unlabelled disorder groups and partial-occupancy classes
        that approach within the disorder clash threshold form an exclusive
        family. When False, only a labelled disorder assembly creates a
        choice. Atom-level shell searches use False so group numbers that
        repeat at unrelated sites are not coupled across the crystal.
    """
    canonical = normalize_disorder_policy(policy)
    n_atoms = len(atoms)
    occupancies, groups, assemblies = _metadata(atoms, n_atoms)
    full = np.ones(n_atoms, dtype=bool)
    competes = _competes(occupancies, groups)
    if n_atoms == 0 or canonical == "as_refined":
        return [
            _realization(full, (), occupancies, drops_atoms=False, competes=competes)
        ]

    families = _exclusive_families(
        occupancies,
        groups,
        assemblies,
        _positions(atoms, n_atoms),
        _symbols(atoms, n_atoms),
        spatial_alternatives=spatial_alternatives,
    )
    if not families:
        return [
            _realization(full, (), occupancies, drops_atoms=False, competes=competes)
        ]

    winners = [_winner(family) for family in families]
    if canonical == "major":
        mask, component_ids = _mask_for(n_atoms, families, winners)
        return [
            _realization(
                mask,
                component_ids,
                occupancies,
                drops_atoms=not bool(mask.all()),
                competes=True,
            )
        ]

    realizations: list[DisorderRealization] = []
    seen: set[bytes] = set()

    def _add(chosen: Sequence[_Component]) -> None:
        mask, component_ids = _mask_for(n_atoms, families, chosen)
        key = np.packbits(mask).tobytes()
        if key in seen:
            return
        seen.add(key)
        realizations.append(
            _realization(
                mask,
                component_ids,
                occupancies,
                drops_atoms=not bool(mask.all()),
                competes=True,
            )
        )

    _add(winners)
    for family_index, family in enumerate(families):
        for component in family:
            if component.component_id == winners[family_index].component_id:
                continue
            chosen = list(winners)
            chosen[family_index] = component
            _add(chosen)
    return realizations


def _realization(
    mask: np.ndarray,
    component_ids: tuple[str, ...],
    occupancies: np.ndarray,
    *,
    drops_atoms: bool,
    competes: bool,
) -> DisorderRealization:
    if mask.shape[0] != occupancies.shape[0]:
        raise ValueError("disorder mask does not match the atom count.")
    kept = occupancies[mask] if mask.any() else np.array([], dtype=float)
    return DisorderRealization(
        mask=np.asarray(mask, dtype=bool).copy(),
        component_ids=tuple(component_ids),
        occupancy_sum=float(kept.sum()) if kept.size else 0.0,
        drops_atoms=bool(drops_atoms),
        competes_for_site=bool(competes),
    )


def _metadata(atoms, n_atoms: int):
    occupancies = _numeric_column(
        atoms,
        KEY_OCCUPANCY,
        n_atoms,
        default=1.0,
    )
    groups = _text_column(atoms, KEY_DISORDER_GROUP, n_atoms, default="0")
    assemblies = np.array(
        [
            _assembly_name(value)
            for value in _text_column(atoms, KEY_ASSEMBLY, n_atoms, default="")
        ],
        dtype=object,
    )
    return occupancies, groups, assemblies


def _numeric_column(atoms, key: str, n_atoms: int, *, default: float) -> np.ndarray:
    raw = _raw_column(atoms, key)
    values = np.full(n_atoms, float(default), dtype=float)
    if raw is None:
        return values
    source = np.asarray(raw, dtype=object).reshape(-1)
    if source.shape[0] != n_atoms:
        return values
    for index, item in enumerate(source):
        try:
            number = float(item)
        except (TypeError, ValueError):
            number = float(default)
        if not np.isfinite(number):
            number = float(default)
        values[index] = number
    return values


def _text_column(atoms, key: str, n_atoms: int, *, default: str) -> np.ndarray:
    raw = _raw_column(atoms, key)
    values = np.array([default] * n_atoms, dtype=object)
    if raw is None:
        return values
    source = np.asarray(raw, dtype=object).reshape(-1)
    if source.shape[0] != n_atoms:
        return values
    for index, item in enumerate(source):
        if item is None:
            values[index] = default
            continue
        values[index] = str(item).strip() or default
    return values


def _raw_column(atoms, key: str):
    arrays = getattr(atoms, "arrays", None)
    if arrays is not None and key in arrays:
        return arrays[key]
    getter = getattr(atoms, "get_array", None)
    if getter is None:
        return None
    try:
        return getter(key)
    except (KeyError, ValueError, AttributeError):
        return None


def _positions(atoms, n_atoms: int) -> np.ndarray:
    getter = getattr(atoms, "get_positions", None)
    if getter is None:
        return np.zeros((n_atoms, 3), dtype=float)
    positions = np.asarray(getter(), dtype=float)
    if positions.shape != (n_atoms, 3):
        return np.zeros((n_atoms, 3), dtype=float)
    return positions


def _assembly_name(value: str) -> str:
    text = str(value).strip()
    if text in _BLANK_ASSEMBLY:
        return ""
    return text


def _is_nonzero_group(label: str) -> bool:
    try:
        return int(float(label)) != 0
    except (TypeError, ValueError):
        return bool(label) and label not in _BLANK_ASSEMBLY


def _competes(occupancies: np.ndarray, groups: Iterable[str]) -> bool:
    partial = occupancies < (1.0 - PARTIAL_OCCUPANCY_TOLERANCE)
    nonzero = np.array([_is_nonzero_group(str(label)) for label in groups], dtype=bool)
    if nonzero.shape[0] != partial.shape[0]:
        return bool(np.any(partial))
    return bool(np.any(partial | nonzero))


def _symbols(atoms, n_atoms: int) -> list[str]:
    getter = getattr(atoms, "get_chemical_symbols", None)
    if getter is None:
        return ["X"] * n_atoms
    symbols = [str(symbol) for symbol in getter()]
    if len(symbols) != n_atoms:
        return ["X"] * n_atoms
    return symbols


def _exclusive_families(
    occupancies: np.ndarray,
    groups: np.ndarray,
    assemblies: np.ndarray,
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    spatial_alternatives: bool,
) -> list[list[_Component]]:
    named: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    blank_groups: dict[str, list[int]] = defaultdict(list)
    occupancy_classes: dict[str, list[int]] = defaultdict(list)
    decimals = int(DISORDER_CONFIG["OCCUPANCY_KEY_DECIMALS"])
    full_occupancy = 1.0 - PARTIAL_OCCUPANCY_TOLERANCE

    for index in range(occupancies.shape[0]):
        assembly = str(assemblies[index])
        group = str(groups[index])
        if assembly:
            named[assembly][group].append(index)
            continue
        if _is_nonzero_group(group):
            blank_groups[group].append(index)
            continue
        if float(occupancies[index]) < full_occupancy:
            label = f"{round(float(occupancies[index]), decimals):.{decimals}f}"
            occupancy_classes[label].append(index)

    families: list[list[_Component]] = []
    for assembly in sorted(named):
        members = named[assembly]
        if len(members) <= 1:
            continue
        families.append(
            [
                _component(
                    f"assembly:{assembly}:group:{label}",
                    label,
                    indices,
                    occupancies,
                )
                for label, indices in members.items()
            ]
        )

    if not spatial_alternatives:
        return families

    clash = float(DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"])
    site_radius = float(DISORDER_CONFIG["SYMMETRY_SITE_RADIUS"])
    blank_components: list[_Component] = []
    for label, indices in blank_groups.items():
        clusters = _spatial_clusters(indices, positions, site_radius)
        for cluster in clusters:
            site_tag = "" if len(clusters) == 1 else f":site:{int(cluster.min())}"
            blank_components.append(
                _component(
                    f"group:{label}{site_tag}",
                    label,
                    cluster,
                    occupancies,
                )
            )
    families.extend(
        _overlapping_families(
            blank_components,
            positions,
            clash,
            require_different_labels=True,
        )
    )
    occupancy_components = [
        _component(f"occupancy:{label}", label, indices, occupancies)
        for label, indices in occupancy_classes.items()
    ]
    families.extend(
        _occupancy_families(occupancy_components, positions, symbols, occupancies)
    )
    return families


def _component(
    component_id: str,
    label: str,
    indices: Sequence[int],
    occupancies: np.ndarray,
) -> _Component:
    chosen = np.asarray(list(indices), dtype=int)
    return _Component(
        component_id=component_id,
        label=str(label),
        indices=chosen,
        occupancy_sum=float(occupancies[chosen].sum()) if chosen.size else 0.0,
        n_atoms=int(chosen.size),
    )


def _overlapping_families(
    components: Sequence[_Component],
    positions: np.ndarray,
    cutoff: float,
    *,
    require_different_labels: bool = False,
) -> list[list[_Component]]:
    if len(components) <= 1:
        return []
    parent = list(range(len(components)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in combinations(range(len(components)), 2):
        if require_different_labels and components[left].label == components[right].label:
            continue
        distance = _min_distance(
            positions,
            components[left].indices,
            components[right].indices,
        )
        if distance <= cutoff:
            _union(left, right)

    clusters: dict[int, list[_Component]] = defaultdict(list)
    for index, component in enumerate(components):
        clusters[_find(index)].append(component)
    return [members for members in clusters.values() if len(members) > 1]


def _spatial_clusters(
    indices: Sequence[int],
    positions: np.ndarray,
    cutoff: float,
) -> list[np.ndarray]:
    """Split ``indices`` into clusters whose members meet within ``cutoff``."""
    chosen = np.asarray(list(indices), dtype=int)
    if chosen.size <= 1:
        return [chosen]
    from scipy.spatial import cKDTree

    parent = np.arange(chosen.size)

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in cKDTree(positions[chosen]).query_pairs(float(cutoff)):
        _union(int(left), int(right))
    buckets: dict[int, list[int]] = defaultdict(list)
    for local, global_index in enumerate(chosen):
        buckets[_find(local)].append(int(global_index))
    return [np.asarray(members, dtype=int) for members in buckets.values()]


def _occupancy_families(
    components: Sequence[_Component],
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
) -> list[list[_Component]]:
    """Exclusive families for group-0 partial occupancies.

    Different occupancy values compete only when they sum to at most one and
    a pair sits well below a covalent bond. One occupancy value is split only
    where atoms themselves sit that close; bonded neighbours are left intact.
    """
    if not components:
        return []
    parent = list(range(len(components)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in combinations(range(len(components)), 2):
        try:
            left_occ = float(components[left].label)
            right_occ = float(components[right].label)
        except ValueError:
            continue
        if left_occ + right_occ > 1.0 + PARTIAL_OCCUPANCY_TOLERANCE:
            continue
        if _has_alternative_contact(
            positions,
            symbols,
            components[left].indices,
            components[right].indices,
        ):
            _union(left, right)

    clusters: dict[int, list[_Component]] = defaultdict(list)
    for index, component in enumerate(components):
        clusters[_find(index)].append(component)
    families: list[list[_Component]] = []
    consumed: set[int] = set()
    for members in clusters.values():
        if len(members) <= 1:
            continue
        families.append(members)
        consumed.update(id(member) for member in members)
    for component in components:
        if id(component) in consumed:
            continue
        families.extend(
            _equal_occupancy_splits(component, positions, symbols, occupancies)
        )
    return families


def _equal_occupancy_splits(
    component: _Component,
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
) -> list[list[_Component]]:
    """Split one occupancy class at short alternative-site contacts."""
    indices = component.indices
    if indices.size < 2:
        return []
    edges = _alternative_edges(indices, positions, symbols)
    if not edges:
        return []
    parent = list(range(indices.size))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        _union(left, right)
    buckets: dict[int, list[int]] = defaultdict(list)
    for local, global_index in enumerate(indices):
        buckets[_find(local)].append(int(global_index))
    families: list[list[_Component]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        families.append(
            [
                _component(
                    f"occupancy:{component.label}:atom:{atom}",
                    component.label,
                    [atom],
                    occupancies,
                )
                for atom in members
            ]
        )
    return families


def _alternative_cutoff(symbol_i: str, symbol_j: str) -> float:
    """Distance below which a partial pair is an alternative site, not a bond."""
    from ...constants import get_atomic_radius, has_atomic_radius, is_metal_element
    from ..interactions.bonding import get_bonding_threshold

    radius_i = get_atomic_radius(symbol_i) if has_atomic_radius(symbol_i) else 0.5
    radius_j = get_atomic_radius(symbol_j) if has_atomic_radius(symbol_j) else 0.5
    bond = get_bonding_threshold(
        radius_i,
        radius_j,
        is_metal_element(symbol_i),
        is_metal_element(symbol_j),
    )
    fraction = float(DISORDER_CONFIG["OCCUPANCY_ALTERNATIVE_BOND_FRACTION"])
    return float(bond) * fraction


def _alternative_edges(
    indices: np.ndarray,
    positions: np.ndarray,
    symbols: Sequence[str],
) -> list[tuple[int, int]]:
    """Local index pairs closer than the alternative-site cutoff."""
    if indices.size < 2:
        return []
    from scipy.spatial import cKDTree

    points = positions[indices]
    local_symbols = [symbols[int(index)] for index in indices]
    search = max(
        _alternative_cutoff(left, right)
        for left in set(local_symbols)
        for right in set(local_symbols)
    )
    if search <= 0:
        return []
    edges: list[tuple[int, int]] = []
    for left, right in cKDTree(points).query_pairs(search):
        distance = float(np.linalg.norm(points[left] - points[right]))
        cutoff = _alternative_cutoff(local_symbols[left], local_symbols[right])
        if distance <= cutoff:
            edges.append((int(left), int(right)))
    return edges


def _has_alternative_contact(
    positions: np.ndarray,
    symbols: Sequence[str],
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    """True when some cross pair is an alternative site rather than a bond."""
    if left.size == 0 or right.size == 0:
        return False
    from scipy.spatial import cKDTree

    left_symbols = {symbols[int(index)] for index in left}
    right_symbols = {symbols[int(index)] for index in right}
    search = max(
        _alternative_cutoff(left_symbol, right_symbol)
        for left_symbol in left_symbols
        for right_symbol in right_symbols
    )
    if search <= 0:
        return False
    neighbours = cKDTree(positions[right]).query_ball_point(positions[left], search)
    right_positions = positions[right]
    for local_left, hits in enumerate(neighbours):
        if not hits:
            continue
        origin = positions[left[local_left]]
        left_symbol = symbols[int(left[local_left])]
        for local_right in hits:
            distance = float(np.linalg.norm(origin - right_positions[local_right]))
            cutoff = _alternative_cutoff(left_symbol, symbols[int(right[local_right])])
            if distance <= cutoff:
                return True
    return False


def _min_distance(
    positions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """Nearest distance between two index sets without an ``n × m`` matrix."""
    if left.size == 0 or right.size == 0:
        return float("inf")
    from scipy.spatial import cKDTree

    if left.size > right.size:
        left, right = right, left
    distances, _ = cKDTree(positions[right]).query(positions[left], k=1)
    return float(np.min(distances))


def _winner(family: Sequence[_Component]) -> _Component:
    """Highest occupancy sum, then more atoms, then the greater label.

    The label tie-break matches the historical assembly mask, which used the
    disorder-group text as the last key of ``max``. A remaining tie keeps the
    component whose lowest atom index is smaller, so equal occupancies resolve
    without depending on dict order.
    """
    return max(
        family,
        key=lambda component: (
            component.occupancy_sum,
            component.n_atoms,
            component.label,
            -int(component.indices.min()) if component.indices.size else 0,
        ),
    )


def _mask_for(
    n_atoms: int,
    families: Sequence[Sequence[_Component]],
    chosen: Sequence[_Component],
) -> tuple[np.ndarray, tuple[str, ...]]:
    mask = np.ones(n_atoms, dtype=bool)
    component_ids = []
    for family, selected in zip(families, chosen):
        component_ids.append(selected.component_id)
        for component in family:
            if component.component_id == selected.component_id:
                continue
            mask[component.indices] = False
    if not mask.any():
        mask[:] = True
        return mask, ()
    return mask, tuple(component_ids)


__all__ = [
    "DEFAULT_DISORDER_POLICY",
    "DISORDER_POLICIES",
    "DisorderRealization",
    "disorder_realizations",
    "normalize_disorder_policy",
]
