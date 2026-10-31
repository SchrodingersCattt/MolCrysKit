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
is not. The element maximum coordination is an upper bound on that shell;
equal occupancy ``1/k`` uses ``|pool| / k`` when that is smaller. With no ordered anchor, the choice is between whole connected copies. A
centre plus its nearest atoms is not a copy. Image pairing and the intra-copy
bond graph are used first. A centre-plus-shell split is kept only when each
piece is one full copy and the pieces cover the class. Otherwise the class
stays whole and a warning is recorded. The remaining tie prefers the tighter shell, then the smaller
radius, then geometry, so the choice does not follow atom order. Metal–ligand
contacts are not covalent valence. Two partial atoms are exclusive when their
occupancies sum to at most one plus ``OCCUPANCY_SUM_TOLERANCE``, even if they
lie within a bond of each other. Above that sum they must coexist.
``UnresolvedDisorderWarning`` is emitted once per call, with a count and a
short site summary. One regular orientation is kept when it can be found;
otherwise the class stays whole.

``find_polyhedra`` defaults to ``major`` for the central site, and shell
vertices always use the major ligand component. ``identify_molecules`` and
the structure readers default to ``as_refined`` so a general identification
does not drop refined atoms.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from itertools import combinations
from typing import Iterable, Iterator, Sequence

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
# Notes collected by ``grouped_disorder_warnings``. ``None`` means each
# ``disorder_realizations`` call warns on its own.
_DISORDER_NOTES: ContextVar[list[str] | None] = ContextVar(
    "molcrys_disorder_notes", default=None
)


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
    if _structure_has_disorder(occupancies, groups):
        if not families:
            _note_overcoordination(full, contacts, symbols, notes)
        else:
            preview_mask, _preview_ids = _mask_for(
                n_atoms, families, [_winner(family) for family in families]
            )
            _note_overcoordination(preview_mask, contacts, symbols, notes)
    _emit_notes(notes)
    if not families:
        return [
            _realization(full, (), occupancies, drops_atoms=False, competes=competes)
        ]

    winners = [_winner(family) for family in families]
    major_mask, _major_ids = _mask_for(n_atoms, families, winners)
    if canonical == "major":
        return [
            _realization(
                major_mask,
                _major_ids,
                occupancies,
                drops_atoms=not bool(major_mask.all()),
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
    """Replace one class by its orientations, or keep the class.

    Unequal occupancy classes that already fit a centre are left whole. Only
    an over-complete class is expanded, which is what keeps a superimposed
    shell from winning on atom count.
    """
    if not _class_is_overcomplete(component, ordered, contacts, symbols):
        return [component]
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
        upper = int(MAX_COORDINATION_NUMBERS.get(symbols[center], DEFAULT_MAX_COORDINATION)) - len(other)
        if upper < 1:
            continue
        if len(pool) <= upper and not _has_acute_unbonded_pair(
            positions, center, pool, symbols, occupancies, metric
        ):
            continue
        accepted = _best_orientation_split(
            positions,
            center,
            pool,
            symbols,
            occupancies,
            metric,
            upper,
            cap,
            slack,
        )
        if accepted is None:
            _note_unresolved(notes, kept_whole=True, label=f"{component.label}@{center}")
            continue
        if len(accepted) < 2:
            _note_unresolved(notes, kept_whole=False, label=f"{component.label}@{center}")
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
        used = {index for _spread, combo in accepted for index in combo}
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
    if component_set - claimed and _has_internal_alternative(
        component.indices, symbols, positions, metric, contacts
    ):
        if not families:
            _note_unresolved(notes, kept_whole=True, label=component.label)
    return _join_linked_orientation_families(
        families, positions, symbols, occupancies, metric, contacts
    )


def _join_linked_orientation_families(
    families: list[list[_Component]],
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
    contacts: "_Contacts",
) -> list[list[_Component]]:
    """Pair orientation choices at the two ends of one connected molecule.

    Independent sites are not linked by a bond and stay separate, so
    ``exclusive`` still flips one site at a time. A cage whose bridges connect
    the ends becomes one family of pure copies instead of a mix of halves.
    """
    pending = list(families)
    changed = True
    while changed and len(pending) >= 2:
        changed = False
        linked_pairs = _linked_family_pairs(pending, contacts)
        for left_index, right_index in linked_pairs:
            joint = _paired_family(
                pending[left_index],
                pending[right_index],
                positions,
                symbols,
                occupancies,
                metric,
                contacts,
            )
            if joint is None:
                continue
            pending = [
                family
                for index, family in enumerate(pending)
                if index not in (left_index, right_index)
            ]
            pending.append(joint)
            changed = True
            break
    return pending


def _linked_family_pairs(
    families: Sequence[Sequence[_Component]],
    contacts: "_Contacts",
) -> list[tuple[int, int]]:
    """Family index pairs that share a covalent bond. Independent sites are absent."""
    owner: dict[int, int] = {}
    for family_index, family in enumerate(families):
        for shell in _family_shells(family):
            for index in shell.indices:
                owner[int(index)] = family_index
    pairs: set[tuple[int, int]] = set()
    for index, family_index in owner.items():
        for other in contacts.covalent[index]:
            other_family = owner.get(int(other))
            if other_family is None or other_family == family_index:
                continue
            pairs.add((min(family_index, other_family), max(family_index, other_family)))
    return sorted(pairs)


def _family_shells(family: Sequence[_Component]) -> list[_Component]:
    return [component for component in family if component.selectable and component.indices.size]


def _mean_nearest_distance(
    left: Sequence[int],
    right: Sequence[int],
    positions: np.ndarray,
    metric: "_Metric",
) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("inf")
    total = 0.0
    for left_index in left:
        total += min(
            _pair_distance(positions, int(left_index), int(right_index), metric)
            for right_index in right
        )
    return total / len(left)


def _cross_bond_count(
    left: Sequence[int],
    right: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    metric: "_Metric",
) -> int:
    count = 0
    for left_index in left:
        for right_index in right:
            if _is_real_bond(positions, int(left_index), int(right_index), symbols, metric):
                count += 1
    return count


def _paired_family(
    left_family: Sequence[_Component],
    right_family: Sequence[_Component],
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
    contacts: "_Contacts",
) -> list[_Component] | None:
    del contacts
    left_shells = _family_shells(left_family)
    right_shells = _family_shells(right_family)
    if len(left_shells) < 2 or len(left_shells) != len(right_shells):
        return None
    linked = False
    for left in left_shells:
        for right in right_shells:
            if _cross_bond_count(left.indices, right.indices, positions, symbols, metric):
                linked = True
                break
        if linked:
            break
    if not linked:
        return None
    used: set[int] = set()
    pairs: list[tuple[_Component, _Component]] = []
    for left in left_shells:
        # The matching half is the one sitting nearest this half. A mixed
        # twist sits farther away and would break the bridges.
        ranked = sorted(
            (
                (
                    _mean_nearest_distance(left.indices, right.indices, positions, metric),
                    index,
                    right,
                )
                for index, right in enumerate(right_shells)
                if index not in used
            ),
            key=lambda item: item[0],
        )
        if not ranked:
            return None
        _distance, index, right = ranked[0]
        used.add(index)
        pairs.append((left, right))
    if len(used) != len(right_shells):
        return None
    members: list[_Component] = []
    for left, right in pairs:
        indices = [int(index) for index in left.indices] + [int(index) for index in right.indices]
        members.append(
            _component(
                f"{left.component_id}+{right.component_id}",
                left.label,
                indices,
                occupancies,
                geometry_key=left.geometry_key or right.geometry_key,
                angle_spread=max(left.angle_spread, right.angle_spread),
                mean_radius=max(left.mean_radius, right.mean_radius),
            )
        )
    return members


def _note_unresolved(notes: list[str], *, kept_whole: bool, label: str = "occupancy") -> None:
    state = "kept whole" if kept_whole else "one orientation kept"
    entry = f"{label} ({state})"
    if entry not in notes:
        notes.append(entry)


def _warn_unresolved(notes: list[str]) -> None:
    """One warning for these notes, with a count and a short site summary."""
    unique: list[str] = []
    for note in notes:
        if note not in unique:
            unique.append(note)
    if not unique:
        return
    preview = "; ".join(unique[:6])
    extra = f" (+{len(unique) - 6} more)" if len(unique) > 6 else ""
    warnings.warn(
        f"unresolved disorder at {len(unique)} site(s): {preview}{extra}.",
        UnresolvedDisorderWarning,
        stacklevel=3,
    )


def _emit_notes(notes: list[str]) -> None:
    """Warn now, or store the notes for ``grouped_disorder_warnings``."""
    buffer = _DISORDER_NOTES.get()
    if buffer is None:
        _warn_unresolved(notes)
        return
    for note in notes:
        if note not in buffer:
            buffer.append(note)


def _structure_has_disorder(occupancies: np.ndarray, groups: np.ndarray) -> bool:
    """True when some atom is partial or carries a non-zero disorder group."""
    if np.any(occupancies < (1.0 - PARTIAL_OCCUPANCY_TOLERANCE)):
        return True
    return any(_is_nonzero_group(str(label)) for label in groups)


def collapse_disorder_warnings(func):
    """Emit at most one disorder warning for a public structure call."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with grouped_disorder_warnings():
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def grouped_disorder_warnings() -> Iterator[None]:
    """Collect disorder warnings and emit one when the public call returns.

    ``disorder_realizations`` warns once per call. Shell searches and molecule
    identification call it once per molecule; this groups those into a single
    ``UnresolvedDisorderWarning``.
    """
    if _DISORDER_NOTES.get() is not None:
        yield
        return
    buffer: list[str] = []
    token = _DISORDER_NOTES.set(buffer)
    try:
        yield
    finally:
        _DISORDER_NOTES.reset(token)
        _warn_unresolved(buffer)


def _note_overcoordination(
    mask: np.ndarray,
    contacts: "_Contacts",
    symbols: Sequence[str],
    notes: list[str],
) -> None:
    """Warn when a kept atom still has more covalent neighbours than it can."""
    from ...constants.config import DEFAULT_MAX_COORDINATION, MAX_COORDINATION_NUMBERS

    kept = np.flatnonzero(mask)
    if kept.size == 0:
        return
    over: dict[str, int] = {}
    for index in kept:
        limit = int(MAX_COORDINATION_NUMBERS.get(symbols[int(index)], DEFAULT_MAX_COORDINATION))
        # Hydrogens are often closer than a bond to several neighbours in an
        # unresolved pile. The heavy atom they over-coordinate is the signal.
        if limit <= 1:
            continue
        degree = sum(1 for other in contacts.covalent[int(index)] if mask[other])
        if degree > limit:
            over[symbols[int(index)]] = over.get(symbols[int(index)], 0) + 1
    if not over:
        return
    summary = ",".join(f"{element}:{count}" for element, count in sorted(over.items()))
    _note_unresolved(notes, kept_whole=True, label=f"over-coordinated {summary}")


def _class_is_overcomplete(
    component: _Component,
    ordered: Sequence[int],
    contacts: "_Contacts",
    symbols: Sequence[str],
) -> bool:
    """True when some ordered centre sees more of this class than it can bond."""
    from ...constants.config import DEFAULT_MAX_COORDINATION, MAX_COORDINATION_NUMBERS

    component_set = {int(index) for index in component.indices}
    ordered_set = set(ordered)
    for center in ordered:
        metal = _is_metal_symbol(symbols[center])
        neighbours = contacts.shell[center] if metal else contacts.covalent[center]
        pool = sum(1 for index in neighbours if index in component_set)
        other = sum(1 for index in neighbours if index in ordered_set and index != center)
        slots = int(MAX_COORDINATION_NUMBERS.get(symbols[center], DEFAULT_MAX_COORDINATION)) - other
        if pool > max(slots, 0):
            return True
    return False


def _candidate_slots(pool: Sequence[int], occupancies: np.ndarray, upper: int) -> list[int]:
    """Shell sizes to try. The element maximum is only an upper bound.

    Equal occupancy ``1/k`` on a pool that divides evenly uses ``|pool| / k``
    when that is smaller than the element maximum. A cage nitrogen then keeps
    three bridges instead of looking for a fourth substituent.
    """
    sizes = [int(upper)]
    occupancy = float(min(float(occupancies[int(index)]) for index in pool))
    if occupancy <= 0:
        return sizes
    copies = int(round(1.0 / occupancy))
    if copies >= 2 and len(pool) % copies == 0:
        derived = len(pool) // copies
        if 1 <= derived < upper:
            sizes.append(derived)
    return sizes


def _best_orientation_split(
    positions: np.ndarray,
    center: int,
    pool: Sequence[int],
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
    upper: int,
    cap: int,
    slack: float,
) -> list[tuple[float, tuple[int, ...]]] | None:
    """Disjoint regular shells. Prefer two or more; else the one best shell."""
    from math import comb

    best: list[tuple[float, tuple[int, ...]]] | None = None
    for slots in _candidate_slots(pool, occupancies, upper):
        if len(pool) < slots:
            continue
        if comb(len(pool), slots) > cap:
            scored = _score_by_direction(positions, center, pool, symbols, slots, metric)
        else:
            scored = _score_combinations(
                positions, center, pool, symbols, slots, metric
            )
        if not scored:
            continue
        accepted = _greedy_shells(positions, center, scored, metric, slack)
        if len(accepted) >= 2:
            return accepted
        if accepted and best is None:
            best = accepted
    return best


def _greedy_shells(
    positions: np.ndarray,
    center: int,
    scored: Sequence[tuple[float, tuple[int, ...]]],
    metric: "_Metric",
    slack: float,
) -> list[tuple[float, tuple[int, ...]]]:
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
        if _spread_bucket(spread) > cutoff:
            continue
        if used.intersection(combo):
            continue
        accepted.append((spread, combo))
        used.update(combo)
    return accepted


def _score_combinations(
    positions: np.ndarray,
    center: int,
    pool: Sequence[int],
    symbols: Sequence[str],
    slots: int,
    metric: "_Metric",
) -> list[tuple[float, tuple[int, ...]]]:
    """Score every substituent combination with one vectorised shell pass."""
    combos = list(combinations(pool, slots))
    if not combos:
        return []
    target = _target_composition(pool, slots, symbols)
    if target is not None:
        combos = [combo for combo in combos if _composition(combo, symbols) == target]
    if not combos:
        return []
    chosen = np.asarray(combos, dtype=int)
    return _score_shell_array(positions, center, chosen, symbols, metric)


def _score_shell_array(
    positions: np.ndarray,
    center: int,
    chosen: np.ndarray,
    symbols: Sequence[str],
    metric: "_Metric",
) -> list[tuple[float, tuple[int, ...]]]:
    """``chosen`` is ``(n_shells, slots)`` global atom indices."""
    if chosen.size == 0:
        return []
    vectors = metric.vectors(positions[chosen] - positions[int(center)])
    distances = np.linalg.norm(vectors, axis=2)
    cutoffs = np.array(
        [
            [_bond_cutoff(symbols[int(center)], symbols[int(index)]) for index in row]
            for row in chosen
        ],
        dtype=float,
    )
    normalized = distances / np.maximum(cutoffs, 1e-8)
    fraction = float(DISORDER_CONFIG["OCCUPANCY_ORIENTATION_RADIUS_STD_FRACTION"])
    ok = np.std(normalized, axis=1) <= fraction
    slots = chosen.shape[1]
    pair_angles: list[np.ndarray] = []
    minimum = float(DISORDER_CONFIG["OCCUPANCY_MIN_SIMULTANEOUS_ANGLE_DEG"])
    for left, right in combinations(range(slots), 2):
        first = vectors[:, left, :]
        second = vectors[:, right, :]
        denom = np.maximum(distances[:, left] * distances[:, right], 1e-12)
        cosine = np.clip(np.sum(first * second, axis=1) / denom, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        ok &= angles >= minimum
        pair_angles.append(angles)
    if pair_angles:
        spreads = np.std(np.stack(pair_angles, axis=1), axis=1)
    else:
        spreads = np.zeros(chosen.shape[0], dtype=float)
    scored: list[tuple[float, tuple[int, ...]]] = []
    for row in np.flatnonzero(ok):
        combo = tuple(sorted(int(index) for index in chosen[row]))
        scored.append((float(spreads[row]), combo))
    return scored


def _score_by_direction(
    positions: np.ndarray,
    center: int,
    pool: Sequence[int],
    symbols: Sequence[str],
    slots: int,
    metric: "_Metric",
) -> list[tuple[float, tuple[int, ...]]]:
    """Group atoms on the same ray, then score one representative per vertex.

    A special-position shell can pile many symmetry copies on four directions.
    Enumerating atoms is unnecessary once those directions are the vertices.
    """
    clusters = _direction_clusters(positions, center, pool, metric)
    if len(clusters) < slots:
        return []
    representatives = []
    for cluster in clusters:
        vectors = metric.vectors(positions[np.asarray(cluster)] - positions[int(center)])
        norms = np.linalg.norm(vectors, axis=1)
        units = vectors / np.maximum(norms[:, None], 1e-8)
        mean = units.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm < 1e-8:
            representatives.append(int(cluster[0]))
            continue
        mean /= mean_norm
        cosine = units @ mean
        representatives.append(int(cluster[int(np.argmax(cosine))]))
    if len(representatives) == slots:
        return _score_combinations(
            positions, center, representatives, symbols, slots, metric
        )
    from math import comb

    cap = int(DISORDER_CONFIG["OCCUPANCY_MAX_ORIENTATION_COMBINATIONS"])
    if comb(len(representatives), slots) > cap:
        return []
    return _score_combinations(
        positions, center, representatives, symbols, slots, metric
    )


def _direction_clusters(
    positions: np.ndarray,
    center: int,
    pool: Sequence[int],
    metric: "_Metric",
) -> list[list[int]]:
    limit = float(DISORDER_CONFIG["OCCUPANCY_SAME_VERTEX_ANGLE_DEG"])
    vectors = metric.vectors(positions[np.asarray(pool, dtype=int)] - positions[int(center)])
    norms = np.linalg.norm(vectors, axis=1)
    units = vectors / np.maximum(norms[:, None], 1e-8)
    clusters: list[list[int]] = []
    seeds: list[np.ndarray] = []
    for local, index in enumerate(pool):
        placed = False
        for cluster_index, seed in enumerate(seeds):
            cosine = float(np.clip(np.dot(units[local], seed), -1.0, 1.0))
            if float(np.degrees(np.arccos(cosine))) <= limit:
                clusters[cluster_index].append(int(index))
                placed = True
                break
        if not placed:
            clusters.append([int(index)])
            seeds.append(units[local])
    return clusters


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
    """Split a fully partial class into whole connected copies.

    A copy is every atom of one orientation, not a centre plus its nearest
    neighbours. Image pairing and bond-graph components are tried first.
    A centre-plus-shell split is kept only when each piece is exactly one
    full copy and the pieces cover the class. Otherwise the class stays whole
    and a warning is recorded.
    """
    indices = [int(index) for index in component.indices]
    if len(indices) < 2:
        return []
    occupancy = float(min(float(occupancies[int(index)]) for index in indices))
    copies = int(round(1.0 / occupancy)) if occupancy > 0 else 1
    if copies < 2 or len(indices) % copies != 0:
        if _has_internal_alternative(indices, symbols, positions, metric, contacts):
            _note_unresolved(notes, kept_whole=True, label=component.label)
        return []
    full = len(indices) // copies
    pieces = _image_copies(indices, positions, symbols, metric)
    if not _covers_whole_copies(pieces, indices, full):
        pieces = _bonded_copy_components(indices, positions, symbols, metric, full, copies)
    if not _covers_whole_copies(pieces, indices, full):
        pieces = _hub_shell_copies(
            indices, positions, symbols, occupancies, metric, contacts, full
        )
    if not _covers_whole_copies(pieces, indices, full):
        _note_unresolved(notes, kept_whole=True, label=component.label)
        return []
    return [_orientation_members(component, pieces, positions, occupancies)]


def _covers_whole_copies(
    pieces: Sequence[Sequence[int]] | None,
    indices: Sequence[int],
    full: int,
) -> bool:
    """True when every piece is one full copy and nothing is left over."""
    if not pieces or full < 1:
        return False
    seen: set[int] = set()
    for piece in pieces:
        chosen = [int(index) for index in piece]
        if len(chosen) != full or seen.intersection(chosen):
            return False
        seen.update(chosen)
    return seen == {int(index) for index in indices}


def _image_copies(
    indices: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    metric: "_Metric",
) -> list[list[int]] | None:
    """Pair each atom with its same-element image when the shift is one direction.

    A mirror or a short translation gives that pairing and keeps the whole
    chain on each side. A rotation does not, and is left for another strategy.
    """
    if len(indices) % 2 != 0:
        return None
    partner: dict[int, int] = {}
    for index in indices:
        best_distance = float("inf")
        best = None
        cutoff = 0.0
        for other in indices:
            if int(other) == int(index) or symbols[int(other)] != symbols[int(index)]:
                continue
            distance = _pair_distance(positions, int(index), int(other), metric)
            limit = _alternative_cutoff(symbols[int(index)], symbols[int(other)])
            if distance <= limit and distance < best_distance:
                best_distance = distance
                best = int(other)
                cutoff = limit
        if best is None or best_distance > cutoff:
            return None
        partner[int(index)] = best
    if any(partner.get(other) != index for index, other in partner.items()):
        return None
    pairs = [(index, partner[index]) for index in indices if index < partner[index]]
    if len(pairs) * 2 != len(indices):
        return None
    reference = metric.vector(positions[pairs[0][1]] - positions[pairs[0][0]])
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm < 1e-8:
        return None
    side_a: list[int] = []
    side_b: list[int] = []
    for left, right in pairs:
        shift = metric.vector(positions[right] - positions[left])
        if float(np.dot(shift, reference)) < 0.0:
            left, right = right, left
            shift = -shift
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm < 1e-8:
            return None
        cosine = float(np.dot(shift, reference) / (shift_norm * reference_norm))
        if cosine < 0.5:
            return None
        side_a.append(int(left))
        side_b.append(int(right))
    if _composition(side_a, symbols) != _composition(side_b, symbols):
        return None
    if not _side_is_connected(side_a, positions, symbols, metric):
        return None
    if not _side_is_connected(side_b, positions, symbols, metric):
        return None
    return [side_a, side_b]


def _side_is_connected(
    side: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    metric: "_Metric",
) -> bool:
    if len(side) <= 1:
        return True
    members = [int(index) for index in side]
    parent = {index: index for index in members}

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in combinations(members, 2):
        if not _is_real_bond(positions, left, right, symbols, metric):
            continue
        parent[_find(right)] = _find(left)
    return len({_find(index) for index in members}) == 1


def _bonded_copy_components(
    indices: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    metric: "_Metric",
    full: int,
    copies: int,
) -> list[list[int]] | None:
    """Connected pieces of the intra-copy bond graph, when they are the copies."""
    members = [int(index) for index in indices]
    parent = {index: index for index in members}

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in combinations(members, 2):
        if not _is_real_bond(positions, left, right, symbols, metric):
            continue
        parent[_find(right)] = _find(left)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index in members:
        buckets[_find(index)].append(index)
    pieces = list(buckets.values())
    if len(pieces) != copies or any(len(piece) != full for piece in pieces):
        return None
    composition = _composition(pieces[0], symbols)
    if any(_composition(piece, symbols) != composition for piece in pieces[1:]):
        return None
    return pieces


def _hub_shell_copies(
    indices: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
    contacts: "_Contacts",
    full: int,
) -> list[list[int]] | None:
    """One centre plus a regular shell, only when that is the entire copy.

    A tetrahedron is a centre and four arms. A chain is not: the shell would
    drop every atom past the first bond, so this returns nothing.
    """
    hubs = _partial_hubs(indices, symbols, occupancies, contacts)
    if len(hubs) < 2 or full <= len(hubs):
        return None
    slots = full - 1
    if slots < 1 or (len(indices) - len(hubs)) != slots * len(hubs):
        return None
    arms = [int(index) for index in indices if int(index) not in set(hubs)]
    cap = int(DISORDER_CONFIG["OCCUPANCY_MAX_ORIENTATION_COMBINATIONS"])
    slack = float(DISORDER_CONFIG["OCCUPANCY_ORIENTATION_STD_SLACK_DEG"])
    from math import comb

    offers: list[tuple[float, int, tuple[int, ...]]] = []
    for hub in hubs:
        pool = [
            arm
            for arm in arms
            if _is_real_bond(positions, int(arm), int(hub), symbols, metric)
            or _pair_distance(positions, int(arm), int(hub), metric)
            <= _alternative_cutoff(symbols[int(arm)], symbols[int(hub)])
        ]
        # Arms of an overlapped centre sit on both centres. Use every arm then.
        if _hubs_are_one_site(positions, hubs, symbols, metric):
            pool = list(arms)
        if len(pool) < slots or comb(len(pool), slots) > cap:
            continue
        for spread, combo in _score_combinations(
            positions, int(hub), pool, symbols, slots, metric
        ):
            offers.append((spread, int(hub), combo))
    offers.sort(key=lambda item: (_spread_bucket(item[0]), item[1], item[2]))
    used_hubs: set[int] = set()
    used_arms: set[int] = set()
    pieces: list[list[int]] = []
    best = offers[0][0] if offers else 0.0
    for spread, hub, combo in offers:
        if hub in used_hubs or used_arms.intersection(combo):
            continue
        if pieces and _spread_bucket(spread) > _spread_bucket(best) + slack:
            continue
        used_hubs.add(hub)
        used_arms.update(combo)
        pieces.append([hub, *combo])
    if not _covers_whole_copies(pieces, indices, full):
        return None
    return pieces


def _partial_hubs(
    indices: Sequence[int],
    symbols: Sequence[str],
    occupancies: np.ndarray,
    contacts: "_Contacts",
) -> list[int]:
    """Centre atoms of a repeated motif, not every atom of a partial chain.

    The centre element is the rarest one whose count is a multiple of
    ``round(1 / occupancy)``. A chain of three carbons at occupancy 0.6 does
    not qualify and stays intact.
    """
    if not indices:
        return []
    occupancy = float(min(float(occupancies[int(index)]) for index in indices))
    copies = int(round(1.0 / occupancy)) if occupancy > 0 else 1
    if copies < 2:
        return []
    counts: dict[str, list[int]] = {}
    for index in indices:
        counts.setdefault(symbols[index], []).append(int(index))
    repeated = [
        (len(members), symbol)
        for symbol, members in counts.items()
        if len(members) >= copies and len(members) % copies == 0
    ]
    if not repeated:
        return []
    smallest = min(count for count, _symbol in repeated)
    candidates = [symbol for count, symbol in repeated if count == smallest]

    def _degree(symbol: str) -> float:
        members = counts[symbol]
        member_set = set(members)
        return sum(len(set(contacts.covalent[index]) - member_set) for index in members) / len(members)

    chosen = max(candidates, key=_degree)
    return counts[chosen]


def _hubs_are_one_site(
    positions: np.ndarray,
    hubs: Sequence[int],
    symbols: Sequence[str],
    metric: "_Metric",
) -> bool:
    if len(hubs) < 2:
        return False
    parent = list(range(len(hubs)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in combinations(range(len(hubs)), 2):
        distance = _pair_distance(positions, hubs[left], hubs[right], metric)
        cutoff = _alternative_cutoff(symbols[hubs[left]], symbols[hubs[right]])
        if distance <= cutoff:
            parent[_find(right)] = _find(left)
    return len({_find(index) for index in range(len(hubs))}) == 1


def _match_hub_shells(
    component: _Component,
    indices: Sequence[int],
    hubs: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
    notes: list[str],
) -> list[list[_Component]] | None:
    """Give each separated centre the regular shell of arms bonded to it.

    Nearest-centre assignment pulls an arm onto the wrong centre when the
    copies are about a bond length apart. Scoring a shell around each centre
    keeps the arms that actually form that centre's orientation.
    """
    hub_set = {int(hub) for hub in hubs}
    arms = [int(index) for index in indices if int(index) not in hub_set]
    if len(hubs) < 2 or len(arms) < len(hubs) or len(arms) % len(hubs) != 0:
        return None
    slots = len(arms) // len(hubs)
    cap = int(DISORDER_CONFIG["OCCUPANCY_MAX_ORIENTATION_COMBINATIONS"])
    slack = float(DISORDER_CONFIG["OCCUPANCY_ORIENTATION_STD_SLACK_DEG"])
    from math import comb

    offers: list[tuple[float, int, tuple[int, ...]]] = []
    for hub in hubs:
        pool = [
            arm
            for arm in arms
            if _is_real_bond(positions, int(arm), int(hub), symbols, metric)
        ]
        if len(pool) < slots or comb(len(pool), slots) > cap:
            continue
        for spread, combo in _score_combinations(
            positions, int(hub), pool, symbols, slots, metric
        ):
            offers.append((spread, int(hub), combo))
    offers.sort(key=lambda item: (_spread_bucket(item[0]), item[1]))
    used_hubs: set[int] = set()
    used_arms: set[int] = set()
    pieces: list[list[int]] = []
    for _spread, hub, combo in offers:
        if hub in used_hubs or used_arms.intersection(combo):
            continue
        # A shell far worse than the best one is a mix, not the other copy.
        if pieces and _spread_bucket(_spread) > _spread_bucket(offers[0][0]) + slack:
            continue
        used_hubs.add(hub)
        used_arms.update(combo)
        pieces.append([hub, *combo])
    if len(pieces) < 2:
        return None
    if len(pieces) < len(hubs):
        _note_unresolved(notes, kept_whole=False, label=component.label)
    return [_orientation_members(component, pieces, positions, occupancies, exclude=indices)]


def _is_real_bond(
    positions: np.ndarray,
    left: int,
    right: int,
    symbols: Sequence[str],
    metric: "_Metric",
) -> bool:
    distance = _pair_distance(positions, left, right, metric)
    cutoff = _bond_cutoff(symbols[left], symbols[right])
    alternative = _alternative_cutoff(symbols[left], symbols[right])
    return alternative < distance <= cutoff


def _assign_arms_to_hubs(
    indices: Sequence[int],
    hubs: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    metric: "_Metric",
) -> dict[int, list[int]]:
    """Each arm goes to the nearest hub that makes a real bond, not an overlap."""
    groups = {int(hub): [int(hub)] for hub in hubs}
    hub_set = set(groups)
    for index in indices:
        if int(index) in hub_set:
            continue
        best_distance = float("inf")
        best_hub = None
        for hub in hubs:
            distance = _pair_distance(positions, int(index), int(hub), metric)
            cutoff = _bond_cutoff(symbols[int(index)], symbols[int(hub)])
            alternative = _alternative_cutoff(symbols[int(index)], symbols[int(hub)])
            if not (alternative < distance <= cutoff):
                continue
            if distance < best_distance:
                best_distance = distance
                best_hub = int(hub)
        if best_hub is not None:
            groups[best_hub].append(int(index))
    return groups


def _groups_with_shared_composition(
    groups: dict[int, list[int]],
    symbols: Sequence[str],
) -> list[list[int]]:
    populated = [members for members in groups.values() if len(members) >= 2]
    if len(populated) < 2:
        return populated
    counts: dict[tuple, list[list[int]]] = {}
    for members in populated:
        counts.setdefault(_composition(members, symbols), []).append(members)
    best = max(counts.values(), key=len)
    if len(best) < 2:
        return []
    return best


def _largest_hub_piece(groups: dict[int, list[int]]) -> list[int]:
    if not groups:
        return []
    return max(groups.values(), key=len)


def _shells_around_overlapped_hubs(
    component: _Component,
    indices: Sequence[int],
    hubs: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
    occupancies: np.ndarray,
    metric: "_Metric",
    notes: list[str],
) -> list[list[_Component]] | None:
    """One centre site, several arm orientations (rotation or cube disorder)."""
    from ...constants.config import DEFAULT_MAX_COORDINATION, MAX_COORDINATION_NUMBERS

    hub_set = {int(hub) for hub in hubs}
    arms = [int(index) for index in indices if int(index) not in hub_set]
    if len(arms) < 2:
        return None
    upper = int(MAX_COORDINATION_NUMBERS.get(symbols[int(hubs[0])], DEFAULT_MAX_COORDINATION))
    cap = int(DISORDER_CONFIG["OCCUPANCY_MAX_ORIENTATION_COMBINATIONS"])
    slack = float(DISORDER_CONFIG["OCCUPANCY_ORIENTATION_STD_SLACK_DEG"])
    accepted = _best_orientation_split(
        positions,
        int(hubs[0]),
        arms,
        symbols,
        occupancies,
        metric,
        upper,
        cap,
        slack,
    )
    if not accepted:
        return None
    if len(accepted) < 2:
        _note_unresolved(notes, kept_whole=False, label=component.label)
    pieces: list[list[int]] = []
    for offset, (_spread, combo) in enumerate(accepted):
        hub = int(hubs[offset % len(hubs)])
        pieces.append([hub, *combo])
    return [_orientation_members(component, pieces, positions, occupancies, exclude=indices)]


def _orientation_members(
    component: _Component,
    pieces: Sequence[Sequence[int]],
    positions: np.ndarray,
    occupancies: np.ndarray,
    exclude: Sequence[int] | None = None,
) -> list[_Component]:
    def _centroid_key(chosen: Sequence[int]) -> tuple:
        centroid = positions[list(chosen)].mean(axis=0)
        return tuple(-round(float(value), 4) for value in centroid)

    members = [
        _component(
            f"occupancy:{component.label}:orientation:{min(piece)}",
            component.label,
            piece,
            occupancies,
            geometry_key=_centroid_key(piece),
        )
        for piece in pieces
        if piece
    ]
    if exclude is None:
        return members
    used = {int(index) for piece in pieces for index in piece}
    leftover = [int(index) for index in exclude if int(index) not in used]
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
    return members


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
        pair_list = _pairs_within(positions, max_cutoff, metric)
        if not pair_list:
            return cls(covalent=covalent, shell=shell)
        pairs = np.asarray(pair_list, dtype=int)
        left = pairs[:, 0]
        right = pairs[:, 1]
        distances = np.linalg.norm(metric.vectors(positions[right] - positions[left]), axis=1)
        present_list = list(present)
        symbol_index = {symbol: index for index, symbol in enumerate(present_list)}
        cutoff = np.zeros((len(present_list), len(present_list)), dtype=float)
        metal = np.array([_is_metal_symbol(symbol) for symbol in present_list], dtype=bool)
        for row, left_symbol in enumerate(present_list):
            for column, right_symbol in enumerate(present_list):
                cutoff[row, column] = _bond_cutoff(left_symbol, right_symbol)
        left_symbol = np.fromiter((symbol_index[symbols[int(index)]] for index in left), dtype=int)
        right_symbol = np.fromiter((symbol_index[symbols[int(index)]] for index in right), dtype=int)
        thresholds = cutoff[left_symbol, right_symbol]
        close = distances <= thresholds
        one_metal = metal[left_symbol] ^ metal[right_symbol]
        neither_metal = ~(metal[left_symbol] | metal[right_symbol])
        for index in np.flatnonzero(close & neither_metal):
            first = int(left[index])
            second = int(right[index])
            covalent[first].append(second)
            covalent[second].append(first)
            shell[first].append(second)
            shell[second].append(first)
        for index in np.flatnonzero(close & one_metal):
            first = int(left[index])
            second = int(right[index])
            shell[first].append(second)
            shell[second].append(first)
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
        cached = getattr(self, "_enabled_cached", None)
        if cached is not None:
            return bool(cached)
        ok = (
            self.cell is not None
            and self.cell.shape == (3, 3)
            and any(self.pbc)
            and abs(float(np.linalg.det(self.cell))) > 1e-8
        )
        object.__setattr__(self, "_enabled_cached", ok)
        return ok

    def vector(self, delta: np.ndarray) -> np.ndarray:
        return self.vectors(np.asarray(delta, dtype=float))

    def vectors(self, deltas: np.ndarray) -> np.ndarray:
        """Minimum-image vectors. ``deltas`` is ``(..., 3)``."""
        deltas = np.asarray(deltas, dtype=float)
        if not self.enabled():
            return deltas
        transform = getattr(self, "_inverse_cell", None)
        if transform is None:
            try:
                transform = np.linalg.inv(self.cell)
            except np.linalg.LinAlgError:
                return deltas
            object.__setattr__(self, "_inverse_cell", transform)
        fractional = np.array(deltas @ transform, dtype=float, copy=True)
        for axis, periodic in enumerate(self.pbc):
            if periodic:
                fractional[..., axis] -= np.round(fractional[..., axis])
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
