"""Coordination-polyhedron network on one ordered molecular replica.

``find_polyhedra`` still returns one shell per centre. This module connects
those shells. Two centres share a contact when a ligand vertex of one is the
same molecule and disorder realization as a ligand vertex of the other, and
the integer image shifts differ by the contact's relative translation.

Image shifts are integers ``(na, nb, nc)`` on the input cell. With lattice
rows ``L``,

    position = reference_centre + image_shift @ L

A shared vertex therefore satisfies ``left.image_shift == right.image_shift + S``.
The contact ``(i, j, S)`` is the same edge as ``(j, i, -S)``.

Each connected component keeps every non-tree closure, including a self-image
edge ``(i, i, S != 0)`` and parallel edges with different shifts. Those finite
vectors generate the translational subgroup. ``integer_translation_basis``
returns the nonzero columns of their column-style Hermite form. The basis is
written in the input cell: a supercell has the same rank and need not repeat
the same integers.

Shell-boundary checks start from the original shells and are not chained.
``drop_farthest_tier`` removes the farthest complete distance tier.
``add_next_tier`` appends the next complete tier that sits inside the search
radius by at least ``SHELL_TIER_DISTANCE_TOLERANCE_A``. If the selected shell
cuts through one distance tier, ``add_next_tier`` is ``indeterminate`` with
reason ``split_distance_tier`` instead of reporting those neighbours as a new
tier. If any centre cannot take that step, the whole difference is
``insufficient`` or ``indeterminate`` and carries no partial contact or
coordination diff.

The first public entry accepts only ``pbc=(True, True, True)``. ``exclusive``
returns one network per local alternative and does not assemble a joint
configuration of the crystal. Species labels are assigned on the masked graph
of that replica, in the ``formula_k`` style used by
:class:`~molcrys_kit.analysis.stoichiometry.StoichiometryAnalyzer`.

Centres are the geometric centroid, centre of mass, or heavy-atom centroid
of one unwrapped molecular image. Atoms stored across a cell face are joined
along bonds before the average, with the first active atom kept fixed. A
wrapped molecule that cannot be unwrapped raises ``ValueError``. An already
contiguous molecule is averaged as stored. Shifting the crystal origin and
wrapping molecules back into the cell keeps component rank, contact
relations, and the Cartesian image-shift geometry.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np

from ..chemistry.topology import integer_translation_basis
from ..constants.config import (
    DISORDER_CONFIG,
    KEY_OCCUPANCY,
    PARTIAL_OCCUPANCY_TOLERANCE,
    PERIODIC_NEIGHBOR_TOLERANCE_A,
    SHELL_TIER_DISTANCE_TOLERANCE_A,
)
from ..utils.graph import graph_invariant
from .disorder.policy import (
    collapse_disorder_warnings,
    disorder_realizations,
    normalize_disorder_policy,
)
from .formula_moiety import heavy_signature
from .packing_shell import (
    DEFAULT_CENTROID_OFFSET_FRAC,
    DEFAULT_MOLECULAR_SEARCH_CUTOFF,
    _MOLECULE_CENTRE_KINDS,
    _collapse_competing_sites,
    _format_molecule_formula,
    _lattice_translations,
    _molecule_centre,
    _molecule_formula_symbols,
    _parse_moiety_signature,
    detect_coordination_number,
)
from ..structures.polyhedra import _ordered_face_edges

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - optional dependency
    ConvexHull = None


_FULL_PBC = (True, True, True)
_FACE_MERGE_DEG = 1.0
# Same absolute plane tolerance used when merged hull triangles are one facet.
# A pair inside the angle window but outside this distance is tolerance-sensitive.
_HULL_PLANE_TOLERANCE = 1e-6


def _require_full_pbc(pbc) -> tuple[bool, bool, bool]:
    try:
        flags = tuple(bool(flag) for flag in pbc)
    except TypeError as exc:
        raise ValueError("pbc must be three periodic-boundary flags.") from exc
    if len(flags) != 3 or flags != _FULL_PBC:
        raise ValueError(
            "polyhedron networks require pbc=(True, True, True) so a "
            f"non-periodic direction is not copied as a lattice image; got {flags}."
        )
    return flags


def _as_lattice(lattice) -> np.ndarray:
    array = np.asarray(lattice, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("lattice must be a finite 3x3 matrix of row vectors.")
    if abs(float(np.linalg.det(array))) <= 0.0:
        raise ValueError("lattice vectors must be linearly independent.")
    return array


def _image_shift(value) -> tuple[int, int, int]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("an image shift must be three finite components.")
    rounded = np.rint(array)
    if float(np.max(np.abs(array - rounded))) > PERIODIC_NEIGHBOR_TOLERANCE_A:
        raise ValueError("an image shift must be an integer lattice translation.")
    return tuple(int(component) for component in rounded)


def _position(value) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("a position must be three finite Cartesian components.")
    return (float(array[0]), float(array[1]), float(array[2]))


def _add(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _neg(shift: tuple[int, int, int]) -> tuple[int, int, int]:
    return (-shift[0], -shift[1], -shift[2])


def _canonical_shift(shift: tuple[int, int, int]) -> tuple[int, int, int]:
    first = next((component for component in shift if component), 0)
    if first < 0:
        return _neg(shift)
    return shift


def _cycle_sort_key(shift: tuple[int, int, int]) -> tuple[int, int, int, int, int]:
    axis = next(index for index, component in enumerate(shift) if component)
    return (sum(component * component for component in shift), axis, *shift)


def _shift_list(shift: tuple[int, int, int]) -> list[int]:
    return [int(component) for component in shift]


def _point_list(position: tuple[float, float, float]) -> list[float]:
    return [float(component) for component in position]


@dataclass(frozen=True, order=True)
class CentreId:
    """Molecule index and disorder realization inside one replica."""

    molecule_index: int
    realization_index: int

    def to_dict(self) -> dict[str, int]:
        return {
            "molecule_index": int(self.molecule_index),
            "realization_index": int(self.realization_index),
        }


@dataclass(frozen=True, order=True)
class SharedVertex:
    """Ligand identity in the left centre's image frame."""

    molecule_index: int
    realization_index: int
    image_shift: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule_index": int(self.molecule_index),
            "realization_index": int(self.realization_index),
            "image_shift": _shift_list(self.image_shift),
        }


@dataclass(frozen=True, order=True)
class ShellVertex:
    """One ligand image kept in a coordination shell."""

    molecule_index: int
    realization_index: int
    image_shift: tuple[int, int, int]
    position_A: tuple[float, float, float]
    distance_A: float
    formula: str
    species_id: str
    disorder_component_ids: tuple[str, ...] = ()
    occupancy_sum: float | None = None
    competes_for_site: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule_index": int(self.molecule_index),
            "realization_index": int(self.realization_index),
            "image_shift": _shift_list(self.image_shift),
            "position_A": _point_list(self.position_A),
            "distance_A": float(self.distance_A),
            "formula": self.formula,
            "species_id": self.species_id,
            "disorder_component_ids": list(self.disorder_component_ids),
            "occupancy_sum": None
            if self.occupancy_sum is None
            else float(self.occupancy_sum),
            "competes_for_site": self.competes_for_site,
        }

    @property
    def identity(self) -> SharedVertex:
        return SharedVertex(
            self.molecule_index, self.realization_index, self.image_shift
        )


@dataclass(frozen=True, order=True)
class ShellCandidate:
    """Ligand image inside the search radius and outside the chosen shell."""

    molecule_index: int
    realization_index: int
    image_shift: tuple[int, int, int]
    position_A: tuple[float, float, float]
    distance_A: float
    formula: str
    species_id: str
    disorder_component_ids: tuple[str, ...] = ()
    occupancy_sum: float | None = None
    competes_for_site: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule_index": int(self.molecule_index),
            "realization_index": int(self.realization_index),
            "image_shift": _shift_list(self.image_shift),
            "position_A": _point_list(self.position_A),
            "distance_A": float(self.distance_A),
            "formula": self.formula,
            "species_id": self.species_id,
            "disorder_component_ids": list(self.disorder_component_ids),
            "occupancy_sum": None
            if self.occupancy_sum is None
            else float(self.occupancy_sum),
            "competes_for_site": self.competes_for_site,
        }


@dataclass(frozen=True)
class CoordinationShell:
    """Molecule-level coordination shell used as one network centre."""

    molecule_index: int
    realization_index: int
    formula: str
    species_id: str
    center_position_A: tuple[float, float, float]
    vertices: tuple[ShellVertex, ...]
    criterion: str
    primary_gap_cn: int
    enclosure_expanded: bool
    search_cutoff_A: float | None
    centre_role: str | None = None
    center_kind: str | None = None
    centre_disorder_policy: str | None = None
    ligand_disorder_policy: str | None = None
    outside_candidates: tuple[ShellCandidate, ...] | None = None
    hard_cutoff_A: float | None = None
    enclosed: bool | None = None
    rejection_reason: str | None = None
    disorder_component_ids: tuple[str, ...] = ()
    occupancy_sum: float | None = None
    competes_for_site: bool | None = None

    @property
    def centre(self) -> CentreId:
        return CentreId(int(self.molecule_index), int(self.realization_index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule_index": int(self.molecule_index),
            "realization_index": int(self.realization_index),
            "centre_role": self.centre_role,
            "formula": self.formula,
            "species_id": self.species_id,
            "center_kind": self.center_kind,
            "center_position_A": _point_list(self.center_position_A),
            "centre_disorder_policy": self.centre_disorder_policy,
            "ligand_disorder_policy": self.ligand_disorder_policy,
            "vertices": [vertex.to_dict() for vertex in self.vertices],
            "outside_candidates": (
                None
                if self.outside_candidates is None
                else [candidate.to_dict() for candidate in self.outside_candidates]
            ),
            "criterion": self.criterion,
            "primary_gap_cn": int(self.primary_gap_cn),
            "enclosure_expanded": bool(self.enclosure_expanded),
            "search_cutoff_A": (
                None if self.search_cutoff_A is None else float(self.search_cutoff_A)
            ),
            "hard_cutoff_A": None
            if self.hard_cutoff_A is None
            else float(self.hard_cutoff_A),
            "enclosed": self.enclosed,
            "rejection_reason": self.rejection_reason,
            "disorder_component_ids": list(self.disorder_component_ids),
            "occupancy_sum": None
            if self.occupancy_sum is None
            else float(self.occupancy_sum),
            "competes_for_site": self.competes_for_site,
        }


@dataclass(frozen=True)
class SharedContact:
    """One periodic contact between two coordination centres."""

    left: CentreId
    right: CentreId
    right_image_shift: tuple[int, int, int]
    relation: str
    ambiguity: str | None
    shared_vertices: tuple[SharedVertex, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "right_image_shift": _shift_list(self.right_image_shift),
            "relation": self.relation,
            "ambiguity": self.ambiguity,
            "shared_vertices": [vertex.to_dict() for vertex in self.shared_vertices],
        }

    @property
    def key(self) -> tuple[CentreId, CentreId, tuple[int, int, int]]:
        return (self.left, self.right, self.right_image_shift)


@dataclass(frozen=True, order=True)
class UnperturbedCentre:
    """Centre that blocked a shell-boundary perturbation."""

    molecule_index: int
    realization_index: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule_index": int(self.molecule_index),
            "realization_index": int(self.realization_index),
            "reason": self.reason,
        }


@dataclass(frozen=True, order=True)
class CoordinationSnapshot:
    """Coordination number of one centre before and after a boundary step."""

    molecule_index: int
    realization_index: int
    before: int
    after: int

    def to_dict(self) -> dict[str, int]:
        return {
            "molecule_index": int(self.molecule_index),
            "realization_index": int(self.realization_index),
            "before": int(self.before),
            "after": int(self.after),
        }


@dataclass(frozen=True, order=True)
class ContactChange:
    """Contact key whose shared geometry changed but whose endpoints did not."""

    left: CentreId
    right: CentreId
    right_image_shift: tuple[int, int, int]
    before_relation: str
    after_relation: str
    before_ambiguity: str | None
    after_ambiguity: str | None
    before_shared_vertices: tuple[SharedVertex, ...] = ()
    after_shared_vertices: tuple[SharedVertex, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "right_image_shift": _shift_list(self.right_image_shift),
            "before_relation": self.before_relation,
            "after_relation": self.after_relation,
            "before_ambiguity": self.before_ambiguity,
            "after_ambiguity": self.after_ambiguity,
            "before_shared_vertices": [
                vertex.to_dict() for vertex in self.before_shared_vertices
            ],
            "after_shared_vertices": [
                vertex.to_dict() for vertex in self.after_shared_vertices
            ],
        }


@dataclass(frozen=True)
class BoundaryShift:
    """Contact and coordination difference for one shell-boundary step."""

    kind: str
    status: str
    reason: str | None
    unperturbed_centres: tuple[UnperturbedCentre, ...]
    added: tuple[SharedContact, ...] | None
    removed: tuple[SharedContact, ...] | None
    changed: tuple[ContactChange, ...] | None
    coordination: tuple[CoordinationSnapshot, ...] | None
    distance_tolerance_A: float = SHELL_TIER_DISTANCE_TOLERANCE_A

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
            "unperturbed_centres": [
                item.to_dict() for item in self.unperturbed_centres
            ],
            "added": None
            if self.added is None
            else [item.to_dict() for item in self.added],
            "removed": (
                None
                if self.removed is None
                else [item.to_dict() for item in self.removed]
            ),
            "changed": (
                None
                if self.changed is None
                else [item.to_dict() for item in self.changed]
            ),
            "coordination": (
                None
                if self.coordination is None
                else [item.to_dict() for item in self.coordination]
            ),
            "distance_tolerance_A": float(self.distance_tolerance_A),
        }


@dataclass(frozen=True)
class ShellBoundary:
    """Independent farthest-tier and next-tier differences of one network."""

    drop_farthest_tier: BoundaryShift
    add_next_tier: BoundaryShift

    def to_dict(self) -> dict[str, Any]:
        return {
            "drop_farthest_tier": self.drop_farthest_tier.to_dict(),
            "add_next_tier": self.add_next_tier.to_dict(),
        }


@dataclass(frozen=True)
class PolyhedronComponent:
    """One connected component and its integer translation basis."""

    centres: tuple[CentreId, ...]
    rank: int
    translation_generators: tuple[tuple[int, int, int], ...]
    cycle_translations: tuple[tuple[int, int, int], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "centres": [centre.to_dict() for centre in self.centres],
            "rank": int(self.rank),
            "translation_generators": [
                _shift_list(generator) for generator in self.translation_generators
            ],
        }
        if self.cycle_translations is not None:
            payload["cycle_translations"] = [
                _shift_list(translation) for translation in self.cycle_translations
            ]
        return payload


@dataclass(frozen=True)
class PolyhedronNetwork:
    """Shells, contacts, and translation components on one replica."""

    shells: tuple[CoordinationShell, ...]
    contacts: tuple[SharedContact, ...]
    components: tuple[PolyhedronComponent, ...]
    lattice_A: tuple[tuple[float, float, float], ...]
    pbc: tuple[bool, bool, bool]
    boundary: ShellBoundary | None
    center_kind: str | None
    centre_disorder_policy: str | None
    ligand_disorder_policy: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shells": [shell.to_dict() for shell in self.shells],
            "contacts": [contact.to_dict() for contact in self.contacts],
            "components": [component.to_dict() for component in self.components],
            "lattice_A": [list(row) for row in self.lattice_A],
            "pbc": [bool(flag) for flag in self.pbc],
            "boundary": None if self.boundary is None else self.boundary.to_dict(),
            "center_kind": self.center_kind,
            "centre_disorder_policy": self.centre_disorder_policy,
            "ligand_disorder_policy": self.ligand_disorder_policy,
        }


@dataclass(frozen=True)
class PolyhedronNetworkReport:
    """Networks produced for one disorder policy.

    ``exclusive`` stores one locally conditioned network per alternative.
    Those networks are not a simultaneous configuration of the crystal.
    """

    networks: tuple[PolyhedronNetwork, ...]
    configuration_status: str
    centre_disorder: str
    ligand_disorder: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "networks": [network.to_dict() for network in self.networks],
            "configuration_status": self.configuration_status,
            "centre_disorder": self.centre_disorder,
            "ligand_disorder": self.ligand_disorder,
            "warnings": list(self.warnings),
        }


@dataclass
class _BoundaryModel:
    failed: bool
    dimension: int
    non_vertices: set[int]
    edges: set[tuple[int, int]]
    faces: set[frozenset[int]]
    loose_edges: set[tuple[int, int]] | None = None
    loose_faces: set[frozenset[int]] | None = None


_FAILED_MODEL = _BoundaryModel(True, -1, set(), set(), set())


def _unanimous(values: Iterable[Any]) -> Any:
    unique = set(values)
    if len(unique) == 1:
        return unique.pop()
    return None


def _lattice_rows(lattice: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(component) for component in row) for row in lattice)


def _observe_reference(
    references: dict[tuple[int, int], np.ndarray],
    lattice: np.ndarray,
    molecule_index: int,
    realization_index: int,
    image_shift: tuple[int, int, int],
    position: tuple[float, float, float],
) -> None:
    shift = np.asarray(image_shift, dtype=float)
    reference = np.asarray(position, dtype=float) - shift @ lattice
    key = (int(molecule_index), int(realization_index))
    known = references.get(key)
    if known is None:
        references[key] = reference
        return
    if float(np.linalg.norm(reference - known)) > PERIODIC_NEIGHBOR_TOLERANCE_A:
        raise ValueError(
            "shell position does not equal the molecule centre plus "
            "image_shift @ lattice for molecule "
            f"{molecule_index} realization {realization_index}."
        )


def _normalize_shell(shell: CoordinationShell) -> CoordinationShell:
    """Store image shifts and positions as plain tuples."""
    vertices = tuple(
        replace(
            vertex,
            molecule_index=int(vertex.molecule_index),
            realization_index=int(vertex.realization_index),
            image_shift=_image_shift(vertex.image_shift),
            position_A=_position(vertex.position_A),
            distance_A=float(vertex.distance_A),
        )
        for vertex in shell.vertices
    )
    outside = None
    if shell.outside_candidates is not None:
        outside = tuple(
            replace(
                candidate,
                molecule_index=int(candidate.molecule_index),
                realization_index=int(candidate.realization_index),
                image_shift=_image_shift(candidate.image_shift),
                position_A=_position(candidate.position_A),
                distance_A=float(candidate.distance_A),
            )
            for candidate in shell.outside_candidates
        )
    return replace(
        shell,
        molecule_index=int(shell.molecule_index),
        realization_index=int(shell.realization_index),
        center_position_A=_position(shell.center_position_A),
        vertices=vertices,
        outside_candidates=outside,
        search_cutoff_A=(
            None if shell.search_cutoff_A is None else float(shell.search_cutoff_A)
        ),
        hard_cutoff_A=None
        if shell.hard_cutoff_A is None
        else float(shell.hard_cutoff_A),
    )


def _validate_shells(shells: Sequence[CoordinationShell], lattice: np.ndarray) -> None:
    seen: set[tuple[int, int]] = set()
    references: dict[tuple[int, int], np.ndarray] = {}
    for shell in shells:
        if not isinstance(shell, CoordinationShell):
            raise TypeError(
                "polyhedron_network_from_shells expects CoordinationShell records, "
                f"got {type(shell).__name__}."
            )
        identity = (int(shell.molecule_index), int(shell.realization_index))
        if identity in seen:
            raise ValueError(
                "duplicate coordination centre "
                f"molecule {identity[0]} realization {identity[1]}."
            )
        seen.add(identity)
        centre_position = _position(shell.center_position_A)
        _observe_reference(
            references,
            lattice,
            shell.molecule_index,
            shell.realization_index,
            (0, 0, 0),
            centre_position,
        )
        for vertex in shell.vertices:
            shift = _image_shift(vertex.image_shift)
            position = _position(vertex.position_A)
            _observe_reference(
                references,
                lattice,
                vertex.molecule_index,
                vertex.realization_index,
                shift,
                position,
            )
        if shell.outside_candidates is None:
            continue
        for candidate in shell.outside_candidates:
            shift = _image_shift(candidate.image_shift)
            position = _position(candidate.position_A)
            _observe_reference(
                references,
                lattice,
                candidate.molecule_index,
                candidate.realization_index,
                shift,
                position,
            )


def _vertex_distances(shell: CoordinationShell) -> list[float]:
    centre = np.asarray(shell.center_position_A, dtype=float)
    return [
        float(np.linalg.norm(np.asarray(vertex.position_A, dtype=float) - centre))
        for vertex in shell.vertices
    ]


def _candidate_distance(shell: CoordinationShell, candidate: ShellCandidate) -> float:
    centre = np.asarray(shell.center_position_A, dtype=float)
    return float(np.linalg.norm(np.asarray(candidate.position_A, dtype=float) - centre))


def _distance_tiers(distances: Sequence[float]) -> list[list[int]]:
    """Group sorted distances that stay within one tier tolerance of the tier start."""
    if not distances:
        return []
    order = sorted(
        range(len(distances)), key=lambda index: (float(distances[index]), index)
    )
    tiers = [[order[0]]]
    anchor = float(distances[order[0]])
    for index in order[1:]:
        distance = float(distances[index])
        if distance - anchor <= SHELL_TIER_DISTANCE_TOLERANCE_A:
            tiers[-1].append(index)
            continue
        tiers.append([index])
        anchor = distance
    return tiers


def _affine_dimension(points: np.ndarray) -> int:
    if len(points) == 0:
        return -1
    centred = points - points.mean(axis=0)
    if len(points) == 1 or not np.any(np.abs(centred) > PERIODIC_NEIGHBOR_TOLERANCE_A):
        return 0
    singular = np.linalg.svd(centred, compute_uv=False)
    return int(np.sum(singular > PERIODIC_NEIGHBOR_TOLERANCE_A))


def _model_1d(points: np.ndarray) -> _BoundaryModel:
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    projection = centred @ axes[0]
    low = float(projection.min())
    high = float(projection.max())
    tolerance = PERIODIC_NEIGHBOR_TOLERANCE_A
    low_indices = [
        index
        for index, value in enumerate(projection)
        if abs(float(value) - low) <= tolerance
    ]
    high_indices = [
        index
        for index, value in enumerate(projection)
        if abs(float(value) - high) <= tolerance
    ]
    corners = set(low_indices) | set(high_indices)
    edges: set[tuple[int, int]] = set()
    if low_indices and high_indices and min(low_indices) != min(high_indices):
        edges.add(tuple(sorted((min(low_indices), min(high_indices)))))
    return _BoundaryModel(
        False,
        1,
        set(range(len(points))) - corners,
        edges,
        set(),
    )


def _model_2d(points: np.ndarray) -> _BoundaryModel:
    if ConvexHull is None:
        return _FAILED_MODEL
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    plane = centred @ axes[:2].T
    try:
        hull = ConvexHull(plane)
    except Exception:
        return _FAILED_MODEL
    ordered = [int(index) for index in hull.vertices]
    edges = {
        tuple(sorted((ordered[index], ordered[(index + 1) % len(ordered)])))
        for index in range(len(ordered))
    }
    faces = {frozenset(ordered)} if len(ordered) >= 3 else set()
    return _BoundaryModel(
        False,
        2,
        set(range(len(points))) - set(ordered),
        edges,
        faces,
    )


def _simplex_frame(points: np.ndarray, simplex, equation) -> tuple | None:
    indices = tuple(int(index) for index in simplex)
    origin = points[indices[0]]
    normal = np.cross(
        points[indices[1]] - origin,
        points[indices[2]] - origin,
    )
    scale = float(np.linalg.norm(normal))
    if scale < 1e-12:
        return None
    normal = normal / scale
    hull_normal = np.asarray(equation[:3], dtype=float)
    hull_scale = float(np.linalg.norm(hull_normal))
    if hull_scale >= 1e-12 and float(np.dot(normal, hull_normal / hull_scale)) < 0.0:
        normal = -normal
    return normal, origin, indices


def _plane_gap(points: np.ndarray, left, right) -> float:
    normal, origin, _indices = left
    return max(abs(float(np.dot(normal, points[index] - origin))) for index in right[2])


def _grouped_hull_faces(
    points: np.ndarray,
    hull,
    *,
    merge_parallel: bool,
) -> list[tuple[np.ndarray, tuple[int, ...]]] | None:
    """Group hull triangles with a translation-invariant plane test.

    ``merge_parallel`` joins every pair whose normals lie within the face
    angle, ignoring plane separation. The strict grouping also requires the
    vertices to sit on one plane. The two results disagree exactly when the
    angle tolerance would merge a shallow, non-coplanar pair.
    """
    frames = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        frame = _simplex_frame(points, simplex, equation)
        if frame is not None:
            frames.append(frame)
    parent = list(range(len(frames)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    cos_tol = math.cos(math.radians(_FACE_MERGE_DEG))
    for left, right in combinations(range(len(frames)), 2):
        if float(np.dot(frames[left][0], frames[right][0])) < cos_tol:
            continue
        gap = max(
            _plane_gap(points, frames[left], frames[right]),
            _plane_gap(points, frames[right], frames[left]),
        )
        if gap <= _HULL_PLANE_TOLERANCE or merge_parallel:
            union(left, right)
    grouped: dict[int, set[int]] = defaultdict(set)
    normals: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, (normal, _origin, indices) in enumerate(frames):
        root = find(index)
        grouped[root].update(indices)
        normals[root].append(normal)
    faces = []
    for root, vertices in grouped.items():
        normal = np.mean(np.vstack(normals[root]), axis=0)
        faces.append((normal, tuple(sorted(vertices))))
    return faces


def _edges_and_faces(
    points: np.ndarray,
    grouped: list[tuple[np.ndarray, tuple[int, ...]]],
) -> tuple[set[tuple[int, int]], set[frozenset[int]]] | None:
    edges: set[tuple[int, int]] = set()
    faces: set[frozenset[int]] = set()
    for normal, vertices in grouped:
        faces.add(frozenset(vertices))
        try:
            face_edges = _ordered_face_edges(points, normal, vertices)
        except Exception:
            return None
        edges.update(tuple(sorted(int(index) for index in edge)) for edge in face_edges)
    return edges, faces


def _model_3d(points: np.ndarray) -> _BoundaryModel:
    if ConvexHull is None:
        return _FAILED_MODEL
    try:
        hull = ConvexHull(points)
    except Exception:
        return _FAILED_MODEL
    try:
        strict = _grouped_hull_faces(points, hull, merge_parallel=False)
        loose = _grouped_hull_faces(points, hull, merge_parallel=True)
    except Exception:
        return _FAILED_MODEL
    if strict is None or loose is None:
        return _FAILED_MODEL
    strict_boundary = _edges_and_faces(points, strict)
    loose_boundary = _edges_and_faces(points, loose)
    if strict_boundary is None or loose_boundary is None:
        return _FAILED_MODEL
    edges, faces = strict_boundary
    loose_edges, loose_faces = loose_boundary
    corners: set[int] = set()
    for face in faces:
        corners.update(face)
    corners.update(int(index) for index in hull.vertices)
    return _BoundaryModel(
        False,
        3,
        set(range(len(points))) - corners,
        edges,
        faces,
        loose_edges,
        loose_faces,
    )


def _boundary_model(points: np.ndarray) -> _BoundaryModel:
    dimension = _affine_dimension(points)
    if dimension <= 0:
        return _BoundaryModel(False, 0, set(), set(), set())
    if dimension == 1:
        return _model_1d(points)
    if dimension == 2:
        return _model_2d(points)
    return _model_3d(points)


def _classify_subset(
    model: _BoundaryModel,
    subset: set[int],
    *,
    loose: bool = False,
) -> tuple[str, str | None]:
    if model.failed or not subset:
        return ("ambiguous", "degenerate_shell")
    edges = model.edges
    faces = model.faces
    if loose and model.loose_edges is not None and model.loose_faces is not None:
        edges = model.loose_edges
        faces = model.loose_faces
    if model.dimension <= 0:
        return ("corner", None)
    if any(index in model.non_vertices for index in subset):
        return ("ambiguous", "non_vertex")
    if len(subset) == 1:
        return ("corner", None)
    if len(subset) == 2:
        edge = tuple(sorted(subset))
        if edge in edges:
            return ("edge", None)
        return ("ambiguous", "non_edge")
    if frozenset(subset) in faces:
        return ("face", None)
    return ("ambiguous", "not_a_single_facet")


def _tolerance_sensitive(model: _BoundaryModel, subset: set[int]) -> bool:
    if model.loose_faces is None:
        return False
    return _classify_subset(model, subset) != _classify_subset(
        model, subset, loose=True
    )


def _combine_relations(
    left: tuple[str, str | None],
    right: tuple[str, str | None],
) -> tuple[str, str | None]:
    if left == right:
        return left
    _left_relation, left_ambiguity = left
    _right_relation, right_ambiguity = right
    if left_ambiguity is None and right_ambiguity is None:
        return ("ambiguous", "facet_mismatch")
    mismatch = {"non_edge", "not_a_single_facet"}
    if left_ambiguity is None and right_ambiguity in mismatch:
        return ("ambiguous", "facet_mismatch")
    if right_ambiguity is None and left_ambiguity in mismatch:
        return ("ambiguous", "facet_mismatch")
    if left_ambiguity is None:
        return ("ambiguous", right_ambiguity)
    if right_ambiguity is None:
        return ("ambiguous", left_ambiguity)
    if left_ambiguity != right_ambiguity:
        return ("ambiguous", "inconsistent_geometry")
    return ("ambiguous", left_ambiguity)


def _positions_match(
    left_position,
    right_position,
    shift: tuple[int, int, int],
    lattice: np.ndarray,
) -> bool:
    expected = (
        np.asarray(right_position, dtype=float)
        + np.asarray(shift, dtype=float) @ lattice
    )
    return (
        float(np.linalg.norm(np.asarray(left_position, dtype=float) - expected))
        <= PERIODIC_NEIGHBOR_TOLERANCE_A
    )


def _orient_pair(
    left_centre: CentreId,
    left_vertex: ShellVertex,
    right_centre: CentreId,
    right_vertex: ShellVertex,
) -> tuple[CentreId, ShellVertex, CentreId, ShellVertex, tuple[int, int, int]]:
    shift = _sub(left_vertex.image_shift, right_vertex.image_shift)
    if right_centre < left_centre or (
        left_centre == right_centre and shift != _canonical_shift(shift)
    ):
        left_centre, right_centre = right_centre, left_centre
        left_vertex, right_vertex = right_vertex, left_vertex
        shift = _sub(left_vertex.image_shift, right_vertex.image_shift)
    return left_centre, left_vertex, right_centre, right_vertex, shift


def _build_contacts(
    shells: Sequence[CoordinationShell],
    lattice: np.ndarray,
) -> tuple[SharedContact, ...]:
    by_centre = {shell.centre: shell for shell in shells}
    occurrences: dict[tuple[int, int], list[tuple[CentreId, ShellVertex]]] = (
        defaultdict(list)
    )
    for shell in shells:
        for vertex in shell.vertices:
            occurrences[(vertex.molecule_index, vertex.realization_index)].append(
                (shell.centre, vertex)
            )
    grouped: dict[
        tuple[CentreId, CentreId, tuple[int, int, int]],
        list[SharedVertex],
    ] = defaultdict(list)
    for hits in occurrences.values():
        for (left_centre, left_vertex), (right_centre, right_vertex) in combinations(
            hits, 2
        ):
            if (
                left_centre == right_centre
                and left_vertex.image_shift == right_vertex.image_shift
            ):
                continue
            (
                left_centre,
                left_vertex,
                right_centre,
                right_vertex,
                shift,
            ) = _orient_pair(left_centre, left_vertex, right_centre, right_vertex)
            if left_centre == right_centre and shift == (0, 0, 0):
                continue
            identity = left_vertex.identity
            bucket = grouped[(left_centre, right_centre, shift)]
            if identity not in bucket:
                bucket.append(identity)

    contacts: list[SharedContact] = []
    models = {
        shell.centre: _boundary_model(
            np.asarray([vertex.position_A for vertex in shell.vertices], dtype=float)
            if shell.vertices
            else np.zeros((0, 3), dtype=float)
        )
        for shell in shells
    }
    for (left_centre, right_centre, shift), shared in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        left_shell = by_centre[left_centre]
        right_shell = by_centre[right_centre]
        relation, ambiguity = _relation_for_share(
            left_shell,
            right_shell,
            models[left_centre],
            models[right_centre],
            shared,
            shift,
            lattice,
        )
        contacts.append(
            SharedContact(
                left=left_centre,
                right=right_centre,
                right_image_shift=shift,
                relation=relation,
                ambiguity=ambiguity,
                shared_vertices=tuple(sorted(shared)),
            )
        )
    return tuple(contacts)


def _index_vertices(shell: CoordinationShell) -> dict[SharedVertex, int]:
    indexed: dict[SharedVertex, int] = {}
    for index, vertex in enumerate(shell.vertices):
        indexed.setdefault(vertex.identity, index)
    return indexed


def _relation_for_share(
    left_shell: CoordinationShell,
    right_shell: CoordinationShell,
    left_model: _BoundaryModel,
    right_model: _BoundaryModel,
    shared: Sequence[SharedVertex],
    shift: tuple[int, int, int],
    lattice: np.ndarray,
) -> tuple[str, str | None]:
    left_index = _index_vertices(left_shell)
    right_index = _index_vertices(right_shell)
    left_subset: set[int] = set()
    right_subset: set[int] = set()
    consistent = True
    for identity in shared:
        left_at = left_index.get(identity)
        right_identity = SharedVertex(
            identity.molecule_index,
            identity.realization_index,
            _sub(identity.image_shift, shift),
        )
        right_at = right_index.get(right_identity)
        if left_at is None or right_at is None:
            consistent = False
            break
        if not _positions_match(
            left_shell.vertices[left_at].position_A,
            right_shell.vertices[right_at].position_A,
            shift,
            lattice,
        ):
            consistent = False
            break
        left_subset.add(left_at)
        right_subset.add(right_at)
    if not consistent:
        return ("ambiguous", "inconsistent_geometry")
    if _tolerance_sensitive(left_model, left_subset) or _tolerance_sensitive(
        right_model, right_subset
    ):
        return ("ambiguous", "tolerance_sensitive")
    return _combine_relations(
        _classify_subset(left_model, left_subset),
        _classify_subset(right_model, right_subset),
    )


def _component_cycles(
    centres: Sequence[CentreId],
    contacts: Sequence[SharedContact],
) -> tuple[tuple[int, int, int], ...]:
    adjacency: dict[CentreId, list[tuple[CentreId, tuple[int, int, int]]]] = {
        centre: [] for centre in centres
    }
    for contact in contacts:
        adjacency[contact.left].append((contact.right, contact.right_image_shift))
        if contact.left != contact.right:
            adjacency[contact.right].append(
                (contact.left, _neg(contact.right_image_shift))
            )
    for centre in adjacency:
        adjacency[centre].sort(
            key=lambda item: (
                item[0].molecule_index,
                item[0].realization_index,
                item[1],
            )
        )
    potentials: dict[CentreId, tuple[int, int, int]] = {}
    cycles: set[tuple[int, int, int]] = set()
    for root in sorted(centres):
        if root in potentials:
            continue
        potentials[root] = (0, 0, 0)
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor, shift in adjacency[node]:
                proposed = _add(potentials[node], shift)
                if neighbor not in potentials:
                    potentials[neighbor] = proposed
                    stack.append(neighbor)
                    continue
                canonical = _canonical_shift(_sub(proposed, potentials[neighbor]))
                if canonical != (0, 0, 0):
                    cycles.add(canonical)
    return tuple(sorted(cycles, key=_cycle_sort_key))


def _components(
    shells: Sequence[CoordinationShell],
    contacts: Sequence[SharedContact],
    *,
    include_cycle_translations: bool,
) -> tuple[PolyhedronComponent, ...]:
    centres = [shell.centre for shell in shells]
    parent = {centre: centre for centre in centres}

    def find(centre: CentreId) -> CentreId:
        while parent[centre] != centre:
            parent[centre] = parent[parent[centre]]
            centre = parent[centre]
        return centre

    def union(left: CentreId, right: CentreId) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for contact in contacts:
        if contact.left != contact.right:
            union(contact.left, contact.right)
    groups: dict[CentreId, list[CentreId]] = defaultdict(list)
    for centre in centres:
        groups[find(centre)].append(centre)
    components: list[PolyhedronComponent] = []
    for members in groups.values():
        member_set = set(members)
        ordered = tuple(sorted(members))
        incident = tuple(
            contact
            for contact in contacts
            if contact.left in member_set and contact.right in member_set
        )
        cycles = _component_cycles(ordered, incident)
        generators = integer_translation_basis(cycles)
        components.append(
            PolyhedronComponent(
                centres=ordered,
                rank=len(generators),
                translation_generators=generators,
                cycle_translations=cycles if include_cycle_translations else None,
            )
        )
    components.sort(key=lambda component: component.centres)
    return tuple(components)


def _contact_geometry(contact: SharedContact) -> tuple[Any, ...]:
    return (
        contact.relation,
        contact.ambiguity,
        tuple(sorted(contact.shared_vertices)),
    )


def _diff_contacts(
    before: Sequence[SharedContact],
    after: Sequence[SharedContact],
) -> tuple[
    tuple[SharedContact, ...], tuple[SharedContact, ...], tuple[ContactChange, ...]
]:
    old = {contact.key: contact for contact in before}
    new = {contact.key: contact for contact in after}
    added = tuple(new[key] for key in sorted(new) if key not in old)
    removed = tuple(old[key] for key in sorted(old) if key not in new)
    changed: list[ContactChange] = []
    for key in sorted(set(old) & set(new)):
        if _contact_geometry(old[key]) == _contact_geometry(new[key]):
            continue
        left, right, shift = key
        changed.append(
            ContactChange(
                left=left,
                right=right,
                right_image_shift=shift,
                before_relation=old[key].relation,
                after_relation=new[key].relation,
                before_ambiguity=old[key].ambiguity,
                after_ambiguity=new[key].ambiguity,
                before_shared_vertices=old[key].shared_vertices,
                after_shared_vertices=new[key].shared_vertices,
            )
        )
    return added, removed, tuple(changed)


def _vertex_from_candidate(
    shell: CoordinationShell, candidate: ShellCandidate
) -> ShellVertex:
    distance = _candidate_distance(shell, candidate)
    return ShellVertex(
        molecule_index=int(candidate.molecule_index),
        realization_index=int(candidate.realization_index),
        image_shift=_image_shift(candidate.image_shift),
        position_A=_position(candidate.position_A),
        distance_A=distance,
        formula=candidate.formula,
        species_id=candidate.species_id,
        disorder_component_ids=tuple(candidate.disorder_component_ids),
        occupancy_sum=candidate.occupancy_sum,
        competes_for_site=candidate.competes_for_site,
    )


def _drop_farthest(
    shell: CoordinationShell,
) -> tuple[str | None, tuple[ShellVertex, ...] | None]:
    tiers = _distance_tiers(_vertex_distances(shell))
    if len(tiers) < 2:
        return "single_distance_tier", None
    removed = set(tiers[-1])
    kept = tuple(
        vertex for index, vertex in enumerate(shell.vertices) if index not in removed
    )
    return None, kept


def _add_next(
    shell: CoordinationShell,
) -> tuple[str | None, tuple[ShellVertex, ...] | None]:
    if not shell.outside_candidates:
        return "outside_candidates_missing", None
    distances = [
        _candidate_distance(shell, candidate) for candidate in shell.outside_candidates
    ]
    tiers = _distance_tiers(distances)
    if not tiers:
        return "outside_candidates_missing", None
    shell_distances = _vertex_distances(shell)
    if shell_distances:
        farthest_shell = max(shell_distances)
        nearest_outside = min(distances)
        if nearest_outside - farthest_shell <= SHELL_TIER_DISTANCE_TOLERANCE_A:
            return "split_distance_tier", None
    chosen = tiers[0]
    farthest = max(distances[index] for index in chosen)
    cutoff = shell.search_cutoff_A
    if cutoff is None or (float(cutoff) - farthest) < SHELL_TIER_DISTANCE_TOLERANCE_A:
        return "search_boundary", None
    present = {vertex.identity for vertex in shell.vertices}
    added: list[ShellVertex] = []
    for index in sorted(chosen, key=lambda item: (distances[item], item)):
        candidate = shell.outside_candidates[index]
        vertex = _vertex_from_candidate(shell, candidate)
        if vertex.identity in present:
            continue
        present.add(vertex.identity)
        added.append(vertex)
    return None, tuple(shell.vertices) + tuple(added)


def _boundary_shift(
    shells: Sequence[CoordinationShell],
    lattice: np.ndarray,
    kind: str,
) -> BoundaryShift:
    perturb = _drop_farthest if kind == "drop_farthest_tier" else _add_next
    blocked: list[UnperturbedCentre] = []
    updated: list[CoordinationShell] = []
    for shell in shells:
        reason, vertices = perturb(shell)
        if reason is not None or vertices is None:
            blocked.append(
                UnperturbedCentre(
                    int(shell.molecule_index),
                    int(shell.realization_index),
                    reason or "single_distance_tier",
                )
            )
            continue
        updated.append(replace(shell, vertices=vertices))
    blocked_tuple = tuple(sorted(blocked))
    if blocked_tuple:
        reasons = {item.reason for item in blocked_tuple}
        status = "indeterminate" if "split_distance_tier" in reasons else "insufficient"
        return BoundaryShift(
            kind=kind,
            status=status,
            reason=reasons.pop() if len(reasons) == 1 else None,
            unperturbed_centres=blocked_tuple,
            added=None,
            removed=None,
            changed=None,
            coordination=None,
        )
    before = _build_contacts(shells, lattice)
    after = _build_contacts(updated, lattice)
    added, removed, changed = _diff_contacts(before, after)
    coordination = tuple(
        sorted(
            CoordinationSnapshot(
                int(shell.molecule_index),
                int(shell.realization_index),
                len(shell.vertices),
                len(new_shell.vertices),
            )
            for shell, new_shell in zip(shells, updated, strict=True)
        )
    )
    return BoundaryShift(
        kind=kind,
        status="checked",
        reason=None,
        unperturbed_centres=(),
        added=added,
        removed=removed,
        changed=changed,
        coordination=coordination,
    )


def _shell_boundary(
    shells: Sequence[CoordinationShell],
    lattice: np.ndarray,
) -> ShellBoundary:
    return ShellBoundary(
        drop_farthest_tier=_boundary_shift(shells, lattice, "drop_farthest_tier"),
        add_next_tier=_boundary_shift(shells, lattice, "add_next_tier"),
    )


def polyhedron_network_from_shells(
    shells: Sequence[CoordinationShell],
    lattice,
    *,
    pbc: tuple[bool, bool, bool] = _FULL_PBC,
    classify_boundary: bool = True,
    include_cycle_translations: bool = False,
) -> PolyhedronNetwork:
    """Connect supplied shells into one periodic polyhedron network.

    Custom shells must already satisfy
    ``position = molecule_centre + image_shift @ lattice`` for every vertex
    and outside candidate of a given molecule and realization. The tolerance
    is :data:`~molcrys_kit.constants.config.PERIODIC_NEIGHBOR_TOLERANCE_A`.
    ``pbc`` other than ``(True, True, True)`` is rejected. Set
    ``classify_boundary=False`` to leave ``boundary`` as ``None``.
    """
    flags = _require_full_pbc(pbc)
    lattice_array = _as_lattice(lattice)
    prepared = tuple(_normalize_shell(shell) for shell in shells)
    _validate_shells(prepared, lattice_array)
    contacts = _build_contacts(prepared, lattice_array)
    components = _components(
        prepared,
        contacts,
        include_cycle_translations=include_cycle_translations,
    )
    boundary = _shell_boundary(prepared, lattice_array) if classify_boundary else None
    return PolyhedronNetwork(
        shells=prepared,
        contacts=contacts,
        components=components,
        lattice_A=_lattice_rows(lattice_array),
        pbc=flags,
        boundary=boundary,
        center_kind=_unanimous(shell.center_kind for shell in prepared),
        centre_disorder_policy=_unanimous(
            shell.centre_disorder_policy for shell in prepared
        ),
        ligand_disorder_policy=_unanimous(
            shell.ligand_disorder_policy for shell in prepared
        ),
    )


def _masked_formula(molecule, mask: np.ndarray) -> str:
    symbols = _molecule_formula_symbols(molecule, mask)
    return _format_molecule_formula(symbols)


def _masked_graph(molecule, mask: np.ndarray) -> nx.Graph:
    kept = [
        index for index, flag in enumerate(np.asarray(mask, dtype=bool)) if bool(flag)
    ]
    return molecule.graph.subgraph(kept).copy()


def _node_match(left, right) -> bool:
    return left.get("symbol") == right.get("symbol")


def _assign_species(entries: Sequence[tuple[Any, str, nx.Graph]]) -> dict[Any, str]:
    """Assign ``formula_k`` labels by masked-graph isomorphism."""
    groups: dict[str, list[tuple[nx.Graph, tuple, list[Any]]]] = defaultdict(list)
    assignment: dict[Any, str] = {}
    for key, formula, graph in entries:
        invariant = graph_invariant(graph)
        matched = None
        for index, (reference, reference_invariant, _keys) in enumerate(
            groups[formula]
        ):
            if invariant != reference_invariant:
                continue
            if nx.is_isomorphic(reference, graph, node_match=_node_match):
                matched = index
                break
        if matched is None:
            matched = len(groups[formula])
            groups[formula].append((graph, invariant, [key]))
        else:
            groups[formula][matched][2].append(key)
        assignment[key] = f"{formula}_{matched + 1}"
    return assignment


def _view_from_realization(
    molecule,
    molecule_index: int,
    realization_index: int,
    realization,
    center_kind: str,
    policy: str,
) -> dict[str, Any] | None:
    mask = np.asarray(realization.mask, dtype=bool)
    symbols = _molecule_formula_symbols(molecule, mask)
    if not symbols:
        return None
    return {
        "molecule_index": int(molecule_index),
        "realization_index": int(realization_index),
        "mask": mask.copy(),
        "component_ids": tuple(realization.component_ids),
        "centre": np.asarray(
            _molecule_centre(molecule, center_kind, mask),
            dtype=float,
        ),
        "signature": heavy_signature(
            dict(Counter(symbol for symbol in symbols if symbol != "H"))
        ),
        "formula": _format_molecule_formula(symbols),
        "occupancy_sum": float(realization.occupancy_sum),
        "competes": bool(realization.competes_for_site),
        "partial_occupancy": _mask_is_partial(molecule, mask),
        "policy": policy,
    }


def _mask_is_partial(molecule, mask: np.ndarray) -> bool:
    """True when a kept site is occupied below a full site."""
    count = len(molecule)
    kept = np.asarray(mask, dtype=bool)
    if count == 0 or kept.shape[0] != count or not kept.any():
        return False
    arrays = getattr(molecule, "arrays", {})
    if KEY_OCCUPANCY in arrays:
        occupancies = np.asarray(arrays[KEY_OCCUPANCY], dtype=float)
    else:
        occupancies = np.ones(count, dtype=float)
    if occupancies.shape[0] != count:
        return False
    return bool(np.any(occupancies[kept] < 1.0 - PARTIAL_OCCUPANCY_TOLERANCE))


def _view_key(view: dict[str, Any]) -> tuple[int, int]:
    return (int(view["molecule_index"]), int(view["realization_index"]))


def _has_site_alternative(
    view: dict[str, Any], pool: Sequence[dict[str, Any]], molecule
) -> bool:
    """True when another exclusive realization or a competing neighbour exists."""
    if len(disorder_realizations(molecule, "exclusive")) > 1:
        return True
    clash = float(DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"])
    identity = _view_key(view)
    for other in pool:
        if _view_key(other) == identity:
            continue
        if other["signature"] != view["signature"]:
            continue
        if not view["competes"] or not other["competes"]:
            continue
        distance = float(np.linalg.norm(view["centre"] - other["centre"]))
        if distance <= clash:
            return True
    return False


def _orphan_partial_keys(
    molecules, views: Sequence[dict[str, Any]]
) -> set[tuple[int, int]]:
    """Partial sites that enter with no alternative realization to choose."""
    keys: set[tuple[int, int]] = set()
    for view in views:
        if not view.get("partial_occupancy"):
            continue
        molecule = molecules[int(view["molecule_index"])]
        if _has_site_alternative(view, views, molecule):
            continue
        keys.add(_view_key(view))
    return keys


def _network_uses_orphan_partial(networks, orphan_keys: set[tuple[int, int]]) -> bool:
    for network in networks:
        for shell in network.shells:
            if (int(shell.molecule_index), int(shell.realization_index)) in orphan_keys:
                return True
            for vertex in shell.vertices:
                if (
                    int(vertex.molecule_index),
                    int(vertex.realization_index),
                ) in orphan_keys:
                    return True
    return False


def _policy_views(molecules, center_kind: str, policy: str) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for molecule_index, molecule in enumerate(molecules):
        for realization_index, realization in enumerate(
            disorder_realizations(molecule, policy)
        ):
            view = _view_from_realization(
                molecule,
                molecule_index,
                realization_index,
                realization,
                center_kind,
                policy,
            )
            if view is not None:
                views.append(view)
    return views


def _occupancy_tie(molecule) -> bool:
    realizations = disorder_realizations(molecule, "exclusive")
    if len(realizations) < 2:
        return False
    reference = float(realizations[0].occupancy_sum)
    reference_mask = np.asarray(realizations[0].mask, dtype=bool)
    for realization in realizations[1:]:
        if (
            abs(float(realization.occupancy_sum) - reference)
            > PARTIAL_OCCUPANCY_TOLERANCE
        ):
            continue
        if not np.array_equal(np.asarray(realization.mask, dtype=bool), reference_mask):
            return True
    return False


def _equal_occupancy_dropped(
    raw: Sequence[dict[str, Any]],
    kept: Sequence[dict[str, Any]],
    signatures: set[tuple],
    molecules,
) -> bool:
    kept_ids = {(view["molecule_index"], view["realization_index"]) for view in kept}
    clash = float(DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"])
    for view in raw:
        if view["signature"] not in signatures:
            continue
        identity = (view["molecule_index"], view["realization_index"])
        if identity in kept_ids:
            if _occupancy_tie(molecules[view["molecule_index"]]):
                return True
            continue
        for other in kept:
            if other["signature"] != view["signature"]:
                continue
            if not view["competes"] or not other["competes"]:
                continue
            if (
                abs(float(view["occupancy_sum"]) - float(other["occupancy_sum"]))
                > PARTIAL_OCCUPANCY_TOLERANCE
            ):
                continue
            distance = float(np.linalg.norm(view["centre"] - other["centre"]))
            if distance <= clash:
                return True
    return False


def _has_alternatives(molecule) -> bool:
    return len(disorder_realizations(molecule, "exclusive")) > 1


def _competing_views(views: Sequence[dict[str, Any]]) -> bool:
    clash = float(DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"])
    for left, right in combinations(views, 2):
        if left["molecule_index"] == right["molecule_index"]:
            return True
        if not left["competes"] or not right["competes"]:
            continue
        if left["signature"] != right["signature"]:
            continue
        if float(np.linalg.norm(left["centre"] - right["centre"])) <= clash:
            return True
    return False


def _exclusive_replicas(views: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not views:
        return []
    parent = list(range(len(views)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_molecule: dict[int, list[int]] = defaultdict(list)
    for index, view in enumerate(views):
        by_molecule[view["molecule_index"]].append(index)
    for indices in by_molecule.values():
        for other in indices[1:]:
            union(indices[0], other)
    clash = float(DISORDER_CONFIG["DISORDER_CLASH_THRESHOLD"])
    for left, right in combinations(range(len(views)), 2):
        left_view = views[left]
        right_view = views[right]
        if not left_view["competes"] or not right_view["competes"]:
            continue
        if left_view["signature"] != right_view["signature"]:
            continue
        if float(np.linalg.norm(left_view["centre"] - right_view["centre"])) <= clash:
            union(left, right)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, view in enumerate(views):
        grouped[find(index)].append(view)
    slots: list[list[dict[str, Any]]] = []
    for members in grouped.values():
        members.sort(
            key=lambda view: (
                -float(view["occupancy_sum"]),
                int(view["molecule_index"]),
                int(view["realization_index"]),
            )
        )
        slots.append(members)
    slots.sort(
        key=lambda members: (
            int(members[0]["molecule_index"]),
            int(members[0]["realization_index"]),
        )
    )
    base = [slot[0] for slot in slots]
    replicas = [base]
    for slot_index, slot in enumerate(slots):
        for alternative in slot[1:]:
            replica = [
                alternative if index == slot_index else other[0]
                for index, other in enumerate(slots)
            ]
            replicas.append(replica)
    return replicas


def _species_maps(
    molecules,
    major_views: Sequence[dict[str, Any]],
    centre_views: Sequence[dict[str, Any]],
) -> dict[Any, str]:
    major_masks = {view["molecule_index"]: view["mask"] for view in major_views}
    entries: list[tuple[Any, str, nx.Graph]] = []
    for view in sorted(major_views, key=lambda item: int(item["molecule_index"])):
        molecule = molecules[view["molecule_index"]]
        entries.append(
            (
                ("major", int(view["molecule_index"])),
                _masked_formula(molecule, view["mask"]),
                _masked_graph(molecule, view["mask"]),
            )
        )
    for view in sorted(
        centre_views,
        key=lambda item: (int(item["molecule_index"]), int(item["realization_index"])),
    ):
        mask = major_masks.get(view["molecule_index"])
        if mask is not None and np.array_equal(mask, view["mask"]):
            continue
        molecule = molecules[view["molecule_index"]]
        entries.append(
            (
                (
                    "center",
                    int(view["molecule_index"]),
                    int(view["realization_index"]),
                ),
                _masked_formula(molecule, view["mask"]),
                _masked_graph(molecule, view["mask"]),
            )
        )
    return _assign_species(entries)


def _species_of(
    assignment: dict[Any, str], view: dict[str, Any], major_masks: dict[int, np.ndarray]
) -> str:
    mask = major_masks.get(view["molecule_index"])
    if mask is not None and np.array_equal(mask, view["mask"]):
        return assignment[("major", int(view["molecule_index"]))]
    return assignment[
        ("center", int(view["molecule_index"]), int(view["realization_index"]))
    ]


def _resolve_radius(
    cutoff: float | None,
    search_cutoff: float | None,
    hard_cutoff: float | None,
) -> tuple[float, float | None]:
    if cutoff is not None and search_cutoff is not None:
        raise ValueError(
            "Pass only one of cutoff= or search_cutoff=; both denote the "
            "candidate search radius."
        )
    radius = (
        DEFAULT_MOLECULAR_SEARCH_CUTOFF
        if cutoff is None and search_cutoff is None
        else (cutoff if cutoff is not None else search_cutoff)
    )
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError(f"cutoff/search_cutoff must be positive, got {radius!r}.")
    resolved_hard: float | None = None
    if hard_cutoff is not None:
        resolved_hard = float(hard_cutoff)
        if resolved_hard <= 0.0:
            raise ValueError(
                f"hard_cutoff must be positive when given, got {resolved_hard!r}."
            )
        if resolved_hard > radius:
            radius = resolved_hard
    return radius, resolved_hard


def _translation_extent(points: Sequence[np.ndarray]) -> float:
    if len(points) <= 1:
        return 0.0
    stacked = np.vstack([np.asarray(point, dtype=float) for point in points])
    return float(np.linalg.norm(np.ptp(stacked, axis=0)))


def _candidate_entries(
    centre_view: dict[str, Any],
    ligand_views: Sequence[dict[str, Any]],
    translations: np.ndarray,
    image_shifts: np.ndarray,
    radius: float,
    homo: bool,
) -> list[
    tuple[float, dict[str, Any], tuple[int, int, int], tuple[float, float, float]]
]:
    if not ligand_views or len(translations) == 0:
        return []
    ligand_centres = np.vstack(
        [np.asarray(view["centre"], dtype=float) for view in ligand_views]
    )
    points = ligand_centres[:, None, :] + translations[None, :, :]
    centre = np.asarray(centre_view["centre"], dtype=float)
    distances = np.linalg.norm(points - centre, axis=-1)
    mask = (distances <= radius) & (distances > 1e-6)
    if homo:
        zeros = np.where(np.all(image_shifts == 0, axis=1))[0]
        if len(zeros):
            zero_index = int(zeros[0])
            for row, view in enumerate(ligand_views):
                if int(view["molecule_index"]) == int(centre_view["molecule_index"]):
                    mask[row, zero_index] = False
    rows, columns = np.where(mask)
    entries = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        shift = tuple(int(component) for component in image_shifts[column])
        position = tuple(float(component) for component in points[row, column])
        entries.append(
            (float(distances[row, column]), ligand_views[row], shift, position)
        )
    entries.sort(key=lambda item: (item[0], int(item[1]["molecule_index"]), item[2]))
    return entries


def _shell_from_centre(
    centre_view: dict[str, Any],
    ligand_views: Sequence[dict[str, Any]],
    *,
    species_of_centre: str,
    species_of_ligand,
    translations: np.ndarray,
    image_shifts: np.ndarray,
    radius: float,
    hard_cutoff: float | None,
    homo: bool,
    enforce_enclosure: bool,
    centroid_offset_frac: float,
    fallback_max: int | None,
    center_kind: str,
    centre_role: str | None,
    disorder_policy: str,
) -> CoordinationShell:
    entries = _candidate_entries(
        centre_view,
        ligand_views,
        translations,
        image_shifts,
        radius,
        homo,
    )
    distances = [entry[0] for entry in entries]
    coordinates = [entry[3] for entry in entries]
    centre_position = _position(centre_view["centre"])
    cn_info = detect_coordination_number(
        distances,
        fallback_max=fallback_max,
        coords=coordinates if coordinates else None,
        center=centre_position,
        enforce_enclosure=enforce_enclosure,
        centroid_offset_frac=centroid_offset_frac,
        cutoff=hard_cutoff,
    )
    count = int(cn_info["coordination_number"])

    def _vertex(entry, distance: float) -> ShellVertex:
        _distance, view, shift, position = entry
        return ShellVertex(
            molecule_index=int(view["molecule_index"]),
            realization_index=int(view["realization_index"]),
            image_shift=shift,
            position_A=position,
            distance_A=float(distance),
            formula=view["formula"],
            species_id=species_of_ligand(view),
            disorder_component_ids=tuple(
                str(item) for item in view.get("component_ids") or ()
            ),
            occupancy_sum=float(view["occupancy_sum"]),
            competes_for_site=bool(view["competes"]),
        )

    vertices = tuple(_vertex(entry, entry[0]) for entry in entries[:count])
    outside = tuple(
        ShellCandidate(
            molecule_index=int(entry[1]["molecule_index"]),
            realization_index=int(entry[1]["realization_index"]),
            image_shift=entry[2],
            position_A=entry[3],
            distance_A=float(entry[0]),
            formula=entry[1]["formula"],
            species_id=species_of_ligand(entry[1]),
            disorder_component_ids=tuple(
                str(item) for item in entry[1].get("component_ids") or ()
            ),
            occupancy_sum=float(entry[1]["occupancy_sum"]),
            competes_for_site=bool(entry[1]["competes"]),
        )
        for entry in entries[count:]
    )
    return CoordinationShell(
        molecule_index=int(centre_view["molecule_index"]),
        realization_index=int(centre_view["realization_index"]),
        formula=centre_view["formula"],
        species_id=species_of_centre,
        center_position_A=centre_position,
        vertices=vertices,
        criterion=str(cn_info["mode"]),
        primary_gap_cn=int(cn_info["primary_gap_cn"]),
        enclosure_expanded=bool(cn_info["enclosure_expanded"]),
        search_cutoff_A=float(radius),
        centre_role=centre_role,
        center_kind=center_kind,
        centre_disorder_policy=disorder_policy,
        ligand_disorder_policy="major",
        outside_candidates=outside,
        hard_cutoff_A=None if hard_cutoff is None else float(hard_cutoff),
        enclosed=bool(cn_info["enclosed"]),
        rejection_reason=cn_info.get("rejection_reason"),
        disorder_component_ids=tuple(
            str(item) for item in centre_view.get("component_ids") or ()
        ),
        occupancy_sum=float(centre_view["occupancy_sum"]),
        competes_for_site=bool(centre_view["competes"]),
    )


def _index_filter(values: Sequence[int] | None) -> set[int] | None:
    if values is None:
        return None
    return {int(value) for value in values}


def _select_views(
    views: Sequence[dict[str, Any]],
    signature: tuple,
    indices: set[int] | None,
) -> list[dict[str, Any]]:
    return [
        view
        for view in views
        if view["signature"] == signature
        and (indices is None or int(view["molecule_index"]) in indices)
    ]


def _require_known_species(requested: str | None, known: set[str], role: str) -> None:
    if requested is None or requested in known:
        return
    available = ", ".join(sorted(known)) if known else "none"
    raise ValueError(
        f"{role} {requested!r} is not a species in this structure; "
        f"known {role} ids: {available}."
    )


def _warnings_for(
    status: str,
    policy: str,
    *,
    unresolved_components: bool,
    partial_occupancy: bool,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if policy == "exclusive":
        warnings.append(
            "exclusive polyhedron networks are local diagnostics, one alternative "
            "at a time; they are not a joint configuration of the whole crystal."
        )
    if unresolved_components:
        warnings.append(
            "as_refined keeps mutually exclusive components in one polyhedron "
            "network; it is not a simultaneously occupied crystal."
        )
    if partial_occupancy:
        warnings.append(
            "a centre or ligand that enters this network has partial occupancy "
            "and no exclusive alternative; reported contacts and dimensionality "
            "are conditional on that undetermined occupancy."
        )
    if status == "tie_broken":
        warnings.append(
            "an occupancy tie was broken with the existing disorder tie-break."
        )
    return tuple(warnings)


@collapse_disorder_warnings
def molecular_polyhedron_network(
    structure,
    central: str,
    ligand: str,
    *,
    center_kind: str = "centroid",
    centre_role: str | None = None,
    cutoff: float | None = None,
    search_cutoff: float | None = None,
    hard_cutoff: float | None = None,
    enforce_enclosure: bool = True,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
    fallback_max: int | None = None,
    central_indices: Sequence[int] | None = None,
    ligand_indices: Sequence[int] | None = None,
    central_species: str | None = None,
    ligand_species: str | None = None,
    disorder_policy: str = "major",
    classify_boundary: bool = True,
    include_cycle_translations: bool = False,
) -> PolyhedronNetworkReport:
    """Build one polyhedron network for each ordered replica of ``structure``.

    Radial arguments match molecule-level :func:`find_polyhedra`. ``cutoff``
    and ``search_cutoff`` are the candidate radius; ``hard_cutoff`` fills that
    ball instead of using the gap. Ligand vertices always use the major
    disorder component. The centre follows ``disorder_policy``.

    ``center_kind`` is evaluated on one unwrapped image of each molecule.
    Coordinates that cross a cell face are joined along bonds before the
    centroid, centre of mass, or heavy-atom centroid is taken. The first
    active atom stays fixed. A wrapped molecule that cannot be unwrapped
    raises ``ValueError``.

    ``major`` returns one network. Its status is ``resolved`` when the kept
    realization is unique and fully occupied, and ``tie_broken`` when equal
    occupancies were split by the existing tie-break. It is ``unresolved`` when
    a centre or ligand that enters the network still has partial occupancy and
    no exclusive alternative; the reported contacts and dimensionality are then
    conditional on that undetermined occupancy. ``exclusive`` returns one
    ``conditional_local`` network per local alternative, including one flip of
    a competing centre at a time rather than their Cartesian product.
    ``as_refined`` returns one network and marks it ``unresolved`` while a
    selected centre or ligand molecule still has mutually exclusive components
    or the same undetermined partial occupancy.

    ``central_species`` and ``ligand_species`` are replica-local ``formula_k``
    labels. ``ligand_indices`` and ``ligand_species`` remove candidates before
    the shell is chosen. Moiety strings use the same heavy-atom match as
    :func:`find_polyhedra`: hydrogen counts and formal charge are ignored, and
    protonation or bond order is not checked. Only ``pbc=(True, True, True)``
    is accepted.
    """
    policy = normalize_disorder_policy(disorder_policy)
    if center_kind not in _MOLECULE_CENTRE_KINDS:
        raise ValueError(
            f"center_kind must be one of {_MOLECULE_CENTRE_KINDS!r}, got {center_kind!r}."
        )
    molecules = getattr(structure, "molecules", None)
    lattice_value = getattr(structure, "lattice", None)
    if molecules is None or lattice_value is None:
        raise TypeError(
            "molecular_polyhedron_network requires a MolecularCrystal-like object "
            f"with .molecules and .lattice; got {type(structure).__name__}."
        )
    flags = _require_full_pbc(getattr(structure, "pbc", None))
    lattice = _as_lattice(lattice_value)
    molecules = list(molecules)
    central_signature = _parse_moiety_signature(central)
    ligand_signature = _parse_moiety_signature(ligand)
    radius, resolved_hard = _resolve_radius(cutoff, search_cutoff, hard_cutoff)
    central_index_set = _index_filter(central_indices)
    ligand_index_set = _index_filter(ligand_indices)

    raw_major = _policy_views(molecules, center_kind, "major")
    major_views = _collapse_competing_sites(raw_major)
    ligand_views = _select_views(major_views, ligand_signature, ligand_index_set)
    relevant_signatures = {central_signature, ligand_signature}
    tie_broken = _equal_occupancy_dropped(
        raw_major,
        major_views,
        relevant_signatures,
        molecules,
    )

    if policy == "major":
        replicas = [_select_views(major_views, central_signature, central_index_set)]
    elif policy == "exclusive":
        replicas = _exclusive_replicas(
            _select_views(
                _policy_views(molecules, center_kind, "exclusive"),
                central_signature,
                central_index_set,
            )
        )
    else:
        replicas = [
            _select_views(
                _policy_views(molecules, center_kind, "as_refined"),
                central_signature,
                central_index_set,
            )
        ]

    unresolved_components = False
    if policy == "as_refined":
        watched = {int(view["molecule_index"]) for view in ligand_views}
        for replica in replicas:
            watched.update(int(view["molecule_index"]) for view in replica)
        if any(_has_alternatives(molecules[index]) for index in watched):
            unresolved_components = True
        if any(_competing_views(replica) for replica in replicas):
            unresolved_components = True

    candidate_views = list(raw_major)
    seen_views = {_view_key(view) for view in candidate_views}
    for replica in replicas:
        for view in replica:
            identity = _view_key(view)
            if identity not in seen_views:
                candidate_views.append(view)
                seen_views.add(identity)
    orphan_partial = _orphan_partial_keys(molecules, candidate_views)

    major_masks = {int(view["molecule_index"]): view["mask"] for view in major_views}
    homo = central_signature == ligand_signature
    base_assignment = _species_maps(molecules, major_views, [])
    known_ligand_species = {
        base_assignment[("major", int(view["molecule_index"]))] for view in ligand_views
    }
    known_central_species: set[str] = set()
    for replica in replicas:
        replica_assignment = _species_maps(molecules, major_views, replica)
        known_central_species.update(
            _species_of(replica_assignment, view, major_masks) for view in replica
        )
    _require_known_species(ligand_species, known_ligand_species, "ligand_species")
    _require_known_species(central_species, known_central_species, "central_species")
    networks: list[PolyhedronNetwork] = []
    for replica in replicas:
        assignment = _species_maps(molecules, major_views, replica)
        centres = [
            view
            for view in replica
            if central_species is None
            or _species_of(assignment, view, major_masks) == central_species
        ]
        ligands = [
            view
            for view in ligand_views
            if ligand_species is None
            or assignment[("major", int(view["molecule_index"]))] == ligand_species
        ]
        if not centres:
            continue
        centres = sorted(
            centres,
            key=lambda view: (
                int(view["molecule_index"]),
                int(view["realization_index"]),
            ),
        )
        extent_points = [view["centre"] for view in centres] + [
            view["centre"] for view in ligands
        ]
        translations, image_shifts = _lattice_translations(
            lattice,
            radius + _translation_extent(extent_points),
        )

        def ligand_species_of(view: dict[str, Any]) -> str:
            return assignment[("major", int(view["molecule_index"]))]

        shells = tuple(
            _shell_from_centre(
                view,
                ligands,
                species_of_centre=_species_of(assignment, view, major_masks),
                species_of_ligand=ligand_species_of,
                translations=translations,
                image_shifts=image_shifts,
                radius=radius,
                hard_cutoff=resolved_hard,
                homo=homo,
                enforce_enclosure=enforce_enclosure,
                centroid_offset_frac=centroid_offset_frac,
                fallback_max=fallback_max,
                center_kind=center_kind,
                centre_role=centre_role,
                disorder_policy=policy,
            )
            for view in centres
        )
        networks.append(
            polyhedron_network_from_shells(
                shells,
                lattice,
                pbc=flags,
                classify_boundary=classify_boundary,
                include_cycle_translations=include_cycle_translations,
            )
        )
    partial_occupancy = _network_uses_orphan_partial(networks, orphan_partial)
    if policy == "exclusive":
        status = "conditional_local"
    elif unresolved_components or partial_occupancy:
        status = "unresolved"
    elif tie_broken:
        status = "tie_broken"
    else:
        status = "resolved"
    return PolyhedronNetworkReport(
        networks=tuple(networks),
        configuration_status=status,
        centre_disorder=policy,
        ligand_disorder="major",
        warnings=_warnings_for(
            status,
            policy,
            unresolved_components=unresolved_components,
            partial_occupancy=partial_occupancy,
        ),
    )


__all__ = [
    "BoundaryShift",
    "CentreId",
    "ContactChange",
    "CoordinationShell",
    "CoordinationSnapshot",
    "PolyhedronComponent",
    "PolyhedronNetwork",
    "PolyhedronNetworkReport",
    "SharedContact",
    "SharedVertex",
    "ShellBoundary",
    "ShellCandidate",
    "ShellVertex",
    "UnperturbedCentre",
    "molecular_polyhedron_network",
    "polyhedron_network_from_shells",
]
