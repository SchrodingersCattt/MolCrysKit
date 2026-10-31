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
  (the greater label wins). An orientation that recorded its geometry then
  breaks the remaining tie by the tighter angle spread, the smaller shell
  radius, and that geometry. Otherwise the smaller atom index wins.
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

Group-0 atoms that differ by occupancy are alternatives when the values sum
to at most one, plus ``OCCUPANCY_SUM_TOLERANCE``, and they cannot be one
shell: a short contact well below a covalent bond, a bond angle at a shared
ordered atom below ``OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG``, or a union that
would exceed that atom's maximum coordination. A bonded chain of unequal
partial occupancies does none of those and stays intact.

Equal occupancies are split into whole orientations around the ordered atom
they share. Each orientation is one regular shell: pairwise angles at least
``OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG``, the same element composition, and a
normalized radius spread (distance divided by that pair's bond cutoff) no
wider than ``OCCUPANCY_ORIENTATION_RADIUS_STD_FRACTION``. A shifted copy is
kept when its angle spread is within ``OCCUPANCY_ORIENTATION_STD_SLACK_DEG``
of the best shell. A mixed shell that shares atoms with a tighter orientation
is not. The remaining tie prefers the tighter shell, then the smaller radius,
then geometry, so the choice does not follow atom order. Metal–ligand
contacts are not covalent valence. Two partial atoms are exclusive when their
occupancies sum to at most one plus ``OCCUPANCY_SUM_TOLERANCE``, even if they
lie within a bond of each other. Above that sum they must coexist. A class
that cannot be split raises ``UnresolvedDisorderWarning``. One regular
orientation is kept when it can be found; otherwise the class stays whole.

``find_polyhedra`` defaults to ``major`` for the central site, and shell
vertices always use the major ligand component. ``identify_molecules`` and
the structure readers default to ``as_refined`` so a general identification
does not drop refined atoms.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from .diagnostics import UnresolvedDisorderWarning

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
    # Order-independent tie key. Empty means fall back to the lowest atom index.
    geometry_key: tuple = field(default_factory=tuple)
    selectable: bool = True
    # Angle spread and mean substituent radius. Used only when geometry_key is set.
    angle_spread: float = 0.0
    mean_radius: float = 0.0


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
    minimum_image: bool = False,
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
        choice.
    minimum_image
        When True, bond and clash distances use the periodic minimum image.
        Atom-level shell masks set this. Unwrapped molecules leave it false
        so a long molecule is not folded back into the cell.
    """
    canonical = normalize_disorder_policy(policy)
    n_atoms = len(atoms)
    occupancies, groups, assemblies = _metadata(atoms, n_atoms)
    metric = _Metric.from_atoms(atoms) if minimum_image else _Metric()
    notes: list[str] = []
    full = np.ones(n_atoms, dtype=bool)
    competes = _competes(occupancies, groups)
    if n_atoms == 0 or canonical == "as_refined":
        return [
            _realization(full, (), occupancies, drops_atoms=False, competes=competes)
        ]

    positions = _positions(atoms, n_atoms)
    symbols = _symbols(atoms, n_atoms)
    contacts = _Contacts.build(positions, symbols, metric)
    families = _exclusive_families(
        occupancies,
        groups,
        assemblies,
        positions,
        symbols,
        metric,
        contacts,
        notes,
        spatial_alternatives=spatial_alternatives,
    )
    if notes:
        warnings.warn(" ".join(notes), UnresolvedDisorderWarning, stacklevel=2)
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
            if not component.selectable:
                continue
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
    metric: "_Metric",
    contacts: "_Contacts",
    notes: list[str],
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
        clusters = _spatial_clusters(indices, positions, site_radius, metric)
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
            metric,
            require_different_labels=True,
        )
    )
    # Negative PART labels are SHELX's mark for a symmetry-generated disorder
    # shell. The copies share one group, so they are split into orientations
    # around the ordered centre instead of being kept as one over-complete set.
    # A labelled assembly with several groups stays exact: each group is
    # already one alternative. A single negative group, labelled or not, is
    # the symmetry shell.
    ordered = _ordered_indices(occupancies, groups)
    for component in blank_components:
        if _group_number(component.label) >= 0:
            continue
        families.extend(
            _orientation_families(
                component,
                positions,
                symbols,
                occupancies,
                ordered,
                metric,
                contacts,
                notes,
            )
        )
    for assembly in sorted(named):
        members = named[assembly]
        if len(members) != 1:
            continue
        label, indices = next(iter(members.items()))
        if _group_number(label) >= 0:
            continue
        families.extend(
            _orientation_families(
                _component(f"group:{label}", label, indices, occupancies),
                positions,
                symbols,
                occupancies,
                ordered,
                metric,
                contacts,
                notes,
            )
        )
    occupancy_components = [
        _component(f"occupancy:{label}", label, indices, occupancies)
        for label, indices in occupancy_classes.items()
    ]
    families.extend(
        _occupancy_families(
            occupancy_components,
            positions,
            symbols,
            occupancies,
            groups,
            metric,
            contacts,
            notes,
        )
    )
    return families


def _group_number(label: str) -> int:
    try:
        return int(float(label))
    except (TypeError, ValueError):
        return 0


def _component(
    component_id: str,
    label: str,
    indices: Sequence[int],
    occupancies: np.ndarray,
    geometry_key: tuple = (),
    selectable: bool = True,
    angle_spread: float = 0.0,
    mean_radius: float = 0.0,
) -> _Component:
    chosen = np.asarray(list(indices), dtype=int)
    return _Component(
        component_id=component_id,
        label=str(label),
        indices=chosen,
        occupancy_sum=float(occupancies[chosen].sum()) if chosen.size else 0.0,
        n_atoms=int(chosen.size),
        geometry_key=tuple(geometry_key),
        selectable=selectable,
        angle_spread=float(angle_spread),
        mean_radius=float(mean_radius),
    )


def _overlapping_families(
    components: Sequence[_Component],
    positions: np.ndarray,
    cutoff: float,
    metric: "_Metric",
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
            metric,
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
    metric: "_Metric",
) -> list[np.ndarray]:
    """Split ``indices`` into clusters whose members meet within ``cutoff``."""
    chosen = np.asarray(list(indices), dtype=int)
    if chosen.size <= 1:
        return [chosen]
    points = positions[chosen]
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

    for left, right in _pairs_within(points, float(cutoff), metric):
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
    groups: np.ndarray,
    metric: "_Metric",
    contacts: "_Contacts",
    notes: list[str],
) -> list[list[_Component]]:
    """Exclusive families for group-0 partial occupancies.

    Distinct occupancy values compete when they are complementary and cannot
    occupy one shell. Each value is split into whole orientations first, so a
    superimposed shell cannot win just because it contains more atoms.
    """
    if not components:
        return []
    ordered = _ordered_indices(occupancies, groups)
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
        if not _classes_compete(
            components[left],
            components[right],
            positions,
            symbols,
            ordered,
            metric,
            contacts,
        ):
            continue
        _union(left, right)

    clusters: dict[int, list[_Component]] = defaultdict(list)
    for index, component in enumerate(components):
        clusters[_find(index)].append(component)
    families: list[list[_Component]] = []
    consumed: set[int] = set()
    for members in clusters.values():
        if len(members) <= 1:
            continue
        expanded: list[_Component] = []
        for member in members:
            expanded.extend(
                _expand_orientations(
                    member,
                    positions,
                    symbols,
                    occupancies,
                    ordered,
                    metric,
                    contacts,
                    notes,
                )
            )
        if len(expanded) > 1:
            families.append(expanded)
        consumed.update(id(member) for member in members)
    for component in components:
        if id(component) in consumed:
            continue
        families.extend(
            _orientation_families(
                component,
                positions,
                symbols,
                occupancies,
                ordered,
                metric,
                contacts,
                notes,
            )
        )
    return families


def _expand_orientations(
    component: _Component,
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    ordered: Sequence[int],
    metric: "_Metric",
    contacts: "_Contacts",
    notes: list[str],
) -> list[_Component]:
    """Replace one class by its orientations, or keep the class."""
    found = _orientation_families(
        component,
        positions,
        symbols,
        occupancies,
        ordered,
        metric,
        contacts,
        notes,
    )
    if not found:
        return [component]
    expanded: list[_Component] = []
    for family in found:
        expanded.extend(family)
    return expanded or [component]


def _ordered_indices(occupancies: np.ndarray, groups: np.ndarray) -> list[int]:
    full = 1.0 - PARTIAL_OCCUPANCY_TOLERANCE
    return [
        index
        for index in range(occupancies.shape[0])
        if float(occupancies[index]) >= full and not _is_nonzero_group(str(groups[index]))
    ]


def _classes_compete(
    left: _Component,
    right: _Component,
    positions: np.ndarray,
    symbols: Sequence[str],
    ordered: Sequence[int],
    metric: "_Metric",
    contacts: "_Contacts",
) -> bool:
    try:
        left_occ = float(left.label)
        right_occ = float(right.label)
    except ValueError:
        return False
    tolerance = float(DISORDER_CONFIG["OCCUPANCY_SUM_TOLERANCE"])
    if left_occ + right_occ > 1.0 + tolerance:
        return False
    if _shared_center_conflict(
        left.indices,
        right.indices,
        positions,
        symbols,
        ordered,
        metric,
        contacts,
    ):
        return True
    return _has_alternative_contact(
        positions, symbols, left.indices, right.indices, metric
    )


def _shared_center_conflict(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    symbols: Sequence[str],
    ordered: Sequence[int],
    metric: "_Metric",
    contacts: "_Contacts",
) -> bool:
    """True when two partial sets cannot both form one covalent shell.

    Metal–ligand contacts are coordination, not covalent valence, so they do
    not use up the centre's covalent slots. Callers only reach this when the
    occupancies sum to at most one plus the occupancy tolerance, so a close
    pair is exclusive even if the two atoms are within a covalent distance.
    """
    if not ordered:
        return False
    from ...constants.config import DEFAULT_MAX_COORDINATION, MAX_COORDINATION_NUMBERS

    minimum_angle = float(DISORDER_CONFIG["OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG"])
    ordered_set = set(ordered)
    left_set = {int(index) for index in left}
    right_set = {int(index) for index in right}
    centers: list[int] = []
    seen: set[int] = set()
    for atom in left_set:
        for neigh in contacts.covalent[atom]:
            if neigh in seen or neigh not in ordered_set:
                continue
            seen.add(neigh)
            if any(other in right_set for other in contacts.covalent[neigh]):
                centers.append(neigh)
    for center in centers:
        bonded_left = [index for index in contacts.covalent[center] if index in left_set]
        bonded_right = [index for index in contacts.covalent[center] if index in right_set]
        if not bonded_left or not bonded_right:
            continue
        other = [
            index
            for index in contacts.covalent[center]
            if index in ordered_set and index != center
        ]
        limit = int(MAX_COORDINATION_NUMBERS.get(symbols[center], DEFAULT_MAX_COORDINATION))
        if len(bonded_left) + len(bonded_right) + len(other) > limit:
            return True
        for left_index in bonded_left:
            for right_index in bonded_right:
                angle = _bond_angle_deg(positions, center, left_index, right_index, metric)
                if angle < minimum_angle:
                    return True
    return False


def _orientation_families(
    component: _Component,
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    ordered: Sequence[int],
    metric: "_Metric",
    contacts: "_Contacts",
    notes: list[str],
) -> list[list[_Component]]:
    """Split one occupancy class into regular shells around ordered atoms."""
    if component.indices.size < 2:
        return []
    if not ordered:
        return _unanchored_families(
            component, positions, symbols, occupancies, metric, contacts, notes
        )
    from math import comb

    from ...constants.config import DEFAULT_MAX_COORDINATION, MAX_COORDINATION_NUMBERS

    families: list[list[_Component]] = []
    claimed: set[int] = set()
    cap = int(DISORDER_CONFIG["OCCUPANCY_MAX_ORIENTATION_COMBINATIONS"])
    slack = float(DISORDER_CONFIG["OCCUPANCY_ORIENTATION_STD_SLACK_DEG"])
    ordered_set = set(ordered)
    component_set = {int(index) for index in component.indices}
    for center in ordered:
        metal = _is_metal_symbol(symbols[center])
        neighbours = contacts.shell[center] if metal else contacts.covalent[center]
        pool = [index for index in neighbours if index in component_set and index not in claimed]
        if len(pool) < 2:
            continue
        other = [index for index in neighbours if index in ordered_set and index != center]
        slots = int(MAX_COORDINATION_NUMBERS.get(symbols[center], DEFAULT_MAX_COORDINATION)) - len(other)
        if slots < 1 or len(pool) <= slots:
            if not _has_acute_unbonded_pair(
                positions, center, pool, symbols, occupancies, metric
            ):
                continue
        if len(pool) < slots or comb(len(pool), slots) > cap:
            _note_unresolved(notes, kept_whole=True)
            continue
        target = _target_composition(pool, slots, symbols)
        scored: list[tuple[float, tuple[int, ...]]] = []
        for combo in combinations(pool, slots):
            chosen = tuple(sorted(int(index) for index in combo))
            if target is not None and _composition(chosen, symbols) != target:
                continue
            spread = _shell_angle_std(positions, center, chosen, symbols, metric)
            if spread is None:
                continue
            scored.append((spread, chosen))
        if not scored:
            _note_unresolved(notes, kept_whole=True)
            continue
        cutoff = _accepted_spread_cutoff([item[0] for item in scored], slack)
        ranked = sorted(
            scored,
            key=lambda item: (
                _spread_bucket(item[0]),
                _relative_geometry_key(positions, center, item[1], metric),
            ),
        )
        accepted: list[tuple[float, tuple[int, ...]]] = []
        used: set[int] = set()
        for spread, combo in ranked:
            # Compare on the same scale as the cutoff. A perfect shell can
            # report a spread of ~1e-14, which is still the best bucket.
            if _spread_bucket(spread) > cutoff:
                continue
            if used.intersection(combo):
                continue
            accepted.append((spread, combo))
            used.update(combo)
        if len(accepted) < 2:
            if len(accepted) == 1:
                _note_unresolved(notes, kept_whole=False)
            else:
                _note_unresolved(notes, kept_whole=True)
                continue
        members = [
            _component(
                f"occupancy:{component.label}:orientation:{combo[0]}",
                component.label,
                combo,
                occupancies,
                geometry_key=_relative_geometry_key(positions, center, combo, metric),
                angle_spread=spread,
                mean_radius=_mean_shell_radius(positions, center, combo, metric),
            )
            for spread, combo in accepted
        ]
        leftover = [index for index in pool if index not in used]
        if leftover:
            members.append(
                _component(
                    f"occupancy:{component.label}:excluded",
                    component.label,
                    leftover,
                    occupancies,
                    selectable=False,
                )
            )
        families.append(members)
        claimed.update(pool)
    if component_set - claimed and _has_internal_alternative(component.indices, symbols, positions, metric, contacts):
        if not families:
            _note_unresolved(notes, kept_whole=True)
    return families


def _note_unresolved(notes: list[str], *, kept_whole: bool) -> None:
    if kept_whole:
        message = (
            "Equal-occupancy substituents could not be split into regular "
            "orientations; the class was kept whole."
        )
    else:
        message = (
            "Equal-occupancy substituents produced only one regular "
            "orientation; the remaining atoms were left out."
        )
    if message not in notes:
        notes.append(message)


def _spread_bucket(value: float) -> float:
    """Angle spreads compared at 0.0001°. Noise below that is one shell."""
    return round(float(value), 4)


def _accepted_spread_cutoff(spreads: Sequence[float], slack: float) -> float:
    """Keep shells within the slack of the best, and at least one gap wide.

    A shifted copy can sit just past the cluster gap (about 6°) and still be
    the other orientation. Mixed shells inside that window share atoms with a
    tighter shell, so the disjoint selection drops them.
    """
    best = min(_spread_bucket(value) for value in spreads)
    gap = float(DISORDER_CONFIG["OCCUPANCY_ORIENTATION_STD_GAP_DEG"])
    return best + max(float(slack), gap)


@lru_cache(maxsize=None)
def _is_metal_symbol(symbol: str) -> bool:
    from ...constants import is_metal_element

    return bool(is_metal_element(symbol))


def _is_covalent(
    left: int,
    right: int,
    positions: np.ndarray,
    symbols: Sequence[str],
    metric: "_Metric",
) -> bool:
    """A covalent contact. Metal–ligand pairs are coordination and do not count."""
    if _is_metal_symbol(symbols[left]) or _is_metal_symbol(symbols[right]):
        return False
    return _pair_distance(positions, left, right, metric) <= _bond_cutoff(
        symbols[left], symbols[right]
    )


@lru_cache(maxsize=None)
def _bond_cutoff(symbol_i: str, symbol_j: str) -> float:
    from ...constants import get_atomic_radius, has_atomic_radius, is_metal_element
    from ..interactions.bonding import get_bonding_threshold

    radius_i = get_atomic_radius(symbol_i) if has_atomic_radius(symbol_i) else 0.5
    radius_j = get_atomic_radius(symbol_j) if has_atomic_radius(symbol_j) else 0.5
    return float(
        get_bonding_threshold(
            radius_i,
            radius_j,
            is_metal_element(symbol_i),
            is_metal_element(symbol_j),
        )
    )


def _bond_angle_deg(
    positions: np.ndarray,
    center: int,
    left: int,
    right: int,
    metric: "_Metric",
) -> float:
    first = metric.vector(positions[left] - positions[center])
    second = metric.vector(positions[right] - positions[center])
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _has_acute_unbonded_pair(
    positions: np.ndarray,
    center: int,
    indices: Sequence[int],
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
) -> bool:
    """True when two substituents cannot be one shell.

    A pair whose occupancies sum to more than one plus the tolerance must
    coexist, so a bond between them is not an alternative. At or below that
    sum the angle rule applies even when the atoms are within a bond length.
    """
    minimum = float(DISORDER_CONFIG["OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG"])
    tolerance = float(DISORDER_CONFIG["OCCUPANCY_SUM_TOLERANCE"])
    for left, right in combinations(indices, 2):
        left_index = int(left)
        right_index = int(right)
        occupancy_sum = float(occupancies[left_index] + occupancies[right_index])
        if occupancy_sum > 1.0 + tolerance and _is_covalent(
            left_index, right_index, positions, symbols, metric
        ):
            continue
        if _bond_angle_deg(positions, center, left_index, right_index, metric) < minimum:
            return True
    return False


def _shell_angle_std(
    positions: np.ndarray,
    center: int,
    indices: Sequence[int],
    symbols: Sequence[str],
    metric: "_Metric",
) -> float | None:
    minimum = float(DISORDER_CONFIG["OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG"])
    normalized = [
        _pair_distance(positions, int(center), int(index), metric)
        / _bond_cutoff(symbols[center], symbols[int(index)])
        for index in indices
    ]
    if len(normalized) >= 2:
        fraction = float(DISORDER_CONFIG["OCCUPANCY_ORIENTATION_RADIUS_STD_FRACTION"])
        if float(np.std(normalized)) > fraction:
            return None
    angles: list[float] = []
    for left, right in combinations(indices, 2):
        angle = _bond_angle_deg(positions, center, int(left), int(right), metric)
        if angle < minimum:
            return None
        angles.append(angle)
    if len(angles) < 2:
        return 0.0
    return float(np.std(angles))


def _composition(indices: Sequence[int], symbols: Sequence[str]) -> tuple:
    counts: dict[str, int] = {}
    for index in indices:
        symbol = symbols[int(index)]
        counts[symbol] = counts.get(symbol, 0) + 1
    return tuple(sorted(counts.items()))


def _target_composition(
    pool: Sequence[int], slots: int, symbols: Sequence[str]
) -> tuple | None:
    """Element counts of one orientation, when the pool is k identical shells."""
    if slots < 1 or len(pool) < slots or len(pool) % slots != 0:
        return None
    copies = len(pool) // slots
    counts: dict[str, int] = {}
    for index in pool:
        symbol = symbols[int(index)]
        counts[symbol] = counts.get(symbol, 0) + 1
    if any(count % copies != 0 for count in counts.values()):
        return None
    return tuple(sorted((symbol, count // copies) for symbol, count in counts.items()))


def _has_internal_alternative(
    indices: Sequence[int],
    symbols: Sequence[str],
    positions: np.ndarray,
    metric: "_Metric",
    contacts: "_Contacts",
) -> bool:
    """True when the class contains a same-element pair too close to coexist."""
    chosen = {int(index) for index in indices}
    for index in chosen:
        for other in contacts.covalent[index]:
            if other not in chosen or other <= index:
                continue
            if symbols[index] != symbols[other]:
                continue
            distance = _pair_distance(positions, index, other, metric)
            if distance <= _alternative_cutoff(symbols[index], symbols[other]):
                return True
    return False


def _unanchored_families(
    component: _Component,
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
    contacts: "_Contacts",
    notes: list[str],
) -> list[list[_Component]]:
    """Split a fully partial copy when every atom has one alternative image.

    With no ordered anchor the covalent-shell enumeration does not apply. A
    rigid duplicate is a matching of same-element pairs closer than the
    alternative cutoff. Anything else is left whole and warned.
    """
    indices = [int(index) for index in component.indices]
    index_set = set(indices)
    candidates: list[tuple[float, int, int]] = []
    for index in indices:
        for other in contacts.covalent[index]:
            if other not in index_set or other <= index:
                continue
            if symbols[index] != symbols[other]:
                continue
            distance = _pair_distance(positions, index, other, metric)
            if distance <= _alternative_cutoff(symbols[index], symbols[other]):
                candidates.append((distance, index, other))
    candidates.sort()
    partner: dict[int, int] = {}
    for _distance, left, right in candidates:
        if left in partner or right in partner:
            continue
        partner[left] = right
        partner[right] = left
    if len(partner) != len(indices):
        _note_unresolved(notes, kept_whole=True)
        return []
    pairs = [(index, partner[index]) for index in indices if index < partner[index]]
    reference = metric.vector(positions[pairs[0][1]] - positions[pairs[0][0]])
    if float(np.linalg.norm(reference)) < 1e-8:
        _note_unresolved(notes, kept_whole=True)
        return []
    side_a: list[int] = []
    side_b: list[int] = []
    for left, right in pairs:
        shift = metric.vector(positions[right] - positions[left])
        # A rigid translation gives one common shift. Opposite endpoints are
        # flipped so each side is one copy. A rotation or a broken matching
        # points somewhere else and is left unresolved.
        if float(np.dot(shift, reference)) <= 0.0:
            left, right = right, left
            shift = -shift
        cosine = float(np.dot(shift, reference) / (np.linalg.norm(shift) * np.linalg.norm(reference)))
        if cosine < 0.5:
            _note_unresolved(notes, kept_whole=True)
            return []
        side_a.append(left)
        side_b.append(right)

    def _centroid_key(chosen: Sequence[int]) -> tuple:
        centroid = positions[list(chosen)].mean(axis=0)
        return tuple(-round(float(value), 4) for value in centroid)

    return [
        [
            _component(
                f"occupancy:{component.label}:orientation:{min(side_a)}",
                component.label,
                side_a,
                occupancies,
                geometry_key=_centroid_key(side_a),
            ),
            _component(
                f"occupancy:{component.label}:orientation:{min(side_b)}",
                component.label,
                side_b,
                occupancies,
                geometry_key=_centroid_key(side_b),
            ),
        ]
    ]


def _mean_shell_radius(
    positions: np.ndarray,
    center: int,
    indices: Sequence[int],
    metric: "_Metric",
) -> float:
    if not indices:
        return 0.0
    distances = [
        _pair_distance(positions, int(center), int(index), metric) for index in indices
    ]
    return float(np.mean(distances))


def _relative_geometry_key(
    positions: np.ndarray,
    center: int,
    indices: Sequence[int],
    metric: "_Metric",
) -> tuple:
    """Order-independent placement key for equal shells."""
    vectors = []
    for index in indices:
        delta = metric.vector(positions[int(index)] - positions[int(center)])
        vectors.append(tuple(float(value) for value in np.round(delta, 4)))
    vectors.sort()
    return tuple(vectors)


def _pair_distance(
    positions: np.ndarray,
    left: int,
    right: int,
    metric: "_Metric",
) -> float:
    return float(np.linalg.norm(metric.vector(positions[right] - positions[left])))


@lru_cache(maxsize=None)
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


def _has_alternative_contact(
    positions: np.ndarray,
    symbols: Sequence[str],
    left: np.ndarray,
    right: np.ndarray,
    metric: "_Metric",
) -> bool:
    """True when some cross pair is an alternative site rather than a bond."""
    if left.size == 0 or right.size == 0:
        return False
    left_symbols = {symbols[int(index)] for index in left}
    right_symbols = {symbols[int(index)] for index in right}
    search = max(
        _alternative_cutoff(left_symbol, right_symbol)
        for left_symbol in left_symbols
        for right_symbol in right_symbols
    )
    if search <= 0:
        return False
    if metric.enabled():
        for left_index in left:
            left_symbol = symbols[int(left_index)]
            for right_index in right:
                distance = _pair_distance(positions, int(left_index), int(right_index), metric)
                if distance > search:
                    continue
                cutoff = _alternative_cutoff(left_symbol, symbols[int(right_index)])
                if distance <= cutoff:
                    return True
        return False
    from scipy.spatial import cKDTree

    neighbours = cKDTree(positions[right]).query_ball_point(positions[left], search)
    for local_left, hits in enumerate(neighbours):
        if not hits:
            continue
        left_symbol = symbols[int(left[local_left])]
        origin = positions[left[local_left]]
        for local_right in hits:
            distance = float(np.linalg.norm(origin - positions[right[local_right]]))
            cutoff = _alternative_cutoff(left_symbol, symbols[int(right[local_right])])
            if distance <= cutoff:
                return True
    return False


def _min_distance(
    positions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    metric: "_Metric",
) -> float:
    """Nearest distance between two index sets, with the periodic image when set."""
    if left.size == 0 or right.size == 0:
        return float("inf")
    if not metric.enabled():
        from scipy.spatial import cKDTree

        if left.size > right.size:
            left, right = right, left
        distances, _ = cKDTree(positions[right]).query(positions[left], k=1)
        return float(np.min(distances))
    best = float("inf")
    for left_index in left:
        for right_index in right:
            best = min(best, _pair_distance(positions, int(left_index), int(right_index), metric))
    return best


def _winner(family: Sequence[_Component]) -> _Component:
    """Highest occupancy sum, then more atoms, then the greater label.

    The label tie-break matches the historical assembly mask. A remaining tie
    uses the orientation's geometry when that was recorded, so the choice does
    not follow input atom order. Otherwise the lowest atom index wins.
    """

    def _tie(component: _Component) -> tuple:
        if component.geometry_key:
            # Smaller angle spread, then the shell closer to the centre, then
            # the placement itself. None of these follow atom order.
            return (
                -_spread_bucket(component.angle_spread),
                -round(float(component.mean_radius), 4),
                component.geometry_key,
            )
        if component.indices.size:
            return (-int(component.indices.min()),)
        return (0,)

    selectable = [component for component in family if component.selectable]
    if not selectable:
        selectable = list(family)
    return max(
        selectable,
        key=lambda component: (
            component.occupancy_sum,
            component.n_atoms,
            component.label,
            _tie(component),
        ),
    )


@dataclass
class _Contacts:
    """Covalent and metal–ligand neighbours, built once per realization."""

    covalent: list[list[int]]
    shell: list[list[int]]

    @classmethod
    def build(
        cls, positions: np.ndarray, symbols: Sequence[str], metric: "_Metric"
    ) -> "_Contacts":
        n_atoms = len(symbols)
        covalent: list[list[int]] = [[] for _ in range(n_atoms)]
        shell: list[list[int]] = [[] for _ in range(n_atoms)]
        if n_atoms < 2:
            return cls(covalent=covalent, shell=shell)
        present = set(symbols)
        max_cutoff = 0.0
        for left in present:
            for right in present:
                max_cutoff = max(max_cutoff, _bond_cutoff(left, right))
        if max_cutoff <= 0:
            return cls(covalent=covalent, shell=shell)
        for left, right in _pairs_within(positions, max_cutoff, metric):
            if _is_covalent(left, right, positions, symbols, metric):
                covalent[left].append(right)
                covalent[right].append(left)
                shell[left].append(right)
                shell[right].append(left)
                continue
            if _is_metal_symbol(symbols[left]) == _is_metal_symbol(symbols[right]):
                continue
            if _pair_distance(positions, left, right, metric) <= _bond_cutoff(
                symbols[left], symbols[right]
            ):
                shell[left].append(right)
                shell[right].append(left)
        return cls(covalent=covalent, shell=shell)


@dataclass(frozen=True)
class _Metric:
    """Optional periodic minimum image for bond and clash distances."""

    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)

    @classmethod
    def from_atoms(cls, atoms) -> "_Metric":
        cell = None
        getter = getattr(atoms, "get_cell", None)
        if getter is not None:
            candidate = np.asarray(getter(), dtype=float)
            if candidate.shape == (3, 3):
                cell = candidate
        pbc_get = getattr(atoms, "get_pbc", None)
        if pbc_get is None:
            pbc = (False, False, False)
        else:
            values = tuple(bool(flag) for flag in pbc_get())
            pbc = values if len(values) == 3 else (False, False, False)
        return cls(cell=cell, pbc=pbc)

    def enabled(self) -> bool:
        if self.cell is None or self.cell.shape != (3, 3) or not any(self.pbc):
            return False
        return abs(float(np.linalg.det(self.cell))) > 1e-8

    def vector(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if not self.enabled():
            return delta
        try:
            fractional = delta @ np.linalg.inv(self.cell)
        except np.linalg.LinAlgError:
            return delta
        fractional = np.array(fractional, dtype=float, copy=True)
        for axis, periodic in enumerate(self.pbc):
            if periodic:
                fractional[axis] -= np.round(fractional[axis])
        return fractional @ self.cell


def _pairs_within(
    points: np.ndarray,
    cutoff: float,
    metric: _Metric,
) -> list[tuple[int, int]]:
    """Index pairs no farther than ``cutoff``, including one periodic shell."""
    from scipy.spatial import cKDTree

    if len(points) < 2:
        return []
    pairs: set[tuple[int, int]] = set()
    tree = cKDTree(points)
    for left, right in tree.query_pairs(cutoff):
        pairs.add((int(left), int(right)))
    if not metric.enabled():
        return sorted(pairs)
    for shift in _image_offsets(metric):
        shifted = points + shift
        neighbours = tree.query_ball_point(shifted, cutoff)
        for left, hits in enumerate(neighbours):
            for right in hits:
                if int(left) == int(right):
                    continue
                pair = tuple(sorted((int(left), int(right))))
                pairs.add(pair)  # type: ignore[arg-type]
    return sorted(pairs)


def _image_offsets(metric: _Metric) -> list[np.ndarray]:
    offsets = []
    ranges = [(-1, 0, 1) if periodic else (0,) for periodic in metric.pbc]
    for ia in ranges[0]:
        for ib in ranges[1]:
            for ic in ranges[2]:
                if ia == ib == ic == 0:
                    continue
                offsets.append(ia * metric.cell[0] + ib * metric.cell[1] + ic * metric.cell[2])
    return offsets


def _mask_for(
    n_atoms: int,
    families: Sequence[Sequence[_Component]],
    chosen: Sequence[_Component],
) -> tuple[np.ndarray, tuple[str, ...]]:
    mask = np.ones(n_atoms, dtype=bool)
    component_ids = []
    for family, selected in zip(families, chosen):
        if not selected.selectable:
            continue
        component_ids.append(selected.component_id)
        for component in family:
            if component.component_id == selected.component_id:
                continue
            mask[component.indices] = False
        mask[selected.indices] = True
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
