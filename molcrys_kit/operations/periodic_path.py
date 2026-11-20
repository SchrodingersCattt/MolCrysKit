"""Geometry-only candidate paths for atom-mapped periodic structures.

This implementation is an exploratory geometric construction. It does not
evaluate a physical potential or infer reaction mechanisms. It uses the bonds
already present on the endpoint structures, retaining integer image shifts in
the shared periodic quotient graph. Distinct bounded lattice windings are kept
as separate candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict, deque
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.transform import Rotation

from ..constants.periodic_path import (
    PERIODIC_PATH_BOND_TOLERANCE_A,
    PERIODIC_PATH_BOND_WEIGHT,
    PERIODIC_PATH_CANDIDATES,
    PERIODIC_PATH_COLLISION_SCALE,
    PERIODIC_PATH_COLLISION_REFINEMENTS,
    PERIODIC_PATH_COLLISION_TOLERANCE_A,
    PERIODIC_PATH_COLLISION_WEIGHT,
    PERIODIC_PATH_COLLISION_WEIGHT_MULTIPLIER,
    PERIODIC_PATH_GRADIENT_TOLERANCE,
    PERIODIC_PATH_GRADIENT_ACCEPTANCE_FACTOR,
    PERIODIC_PATH_IMAGES,
    PERIODIC_PATH_INITIAL_SEPARATION_STEPS,
    PERIODIC_PATH_INITIAL_SEPARATION_FACTOR,
    PERIODIC_PATH_FUNCTION_TOLERANCE,
    PERIODIC_PATH_MAX_ITERATIONS,
    PERIODIC_PATH_OVERLAP_TOLERANCE_A,
    PERIODIC_PATH_SEED_WEIGHT,
    PERIODIC_PATH_SMOOTHNESS_WEIGHT,
    PERIODIC_PATH_SPACING_RATIO,
)
from ..periodic import PeriodicQuotientGraph
from ..structures.crystal import MolecularCrystal
from ..structures.records import BondRecord
from ..utils.geometry import kabsch_align
from ._path_core import materialize_atoms_frame
from ._periodic_path_geometry import (
    ImageEdge,
    _integer_inverse,
    canonical_periodic_basis,
    complete_periodic_cell,
    interpolate_cells,
    nearest_image,
    records_to_edges,
    transform_edge,
)


@dataclass(frozen=True)
class PeriodicPathConfig:
    """Numerical choices for the geometry-only path constructor.

    Weights are dimensionless geometric tuning parameters. Thresholds are
    conservative screening settings, not chemical or energetic guarantees.
    """

    n_images: int = PERIODIC_PATH_IMAGES
    max_candidates: int = PERIODIC_PATH_CANDIDATES
    max_iterations: int = PERIODIC_PATH_MAX_ITERATIONS
    gradient_tolerance: float = PERIODIC_PATH_GRADIENT_TOLERANCE
    collision_scale: float = PERIODIC_PATH_COLLISION_SCALE
    collision_tolerance_A: float = PERIODIC_PATH_COLLISION_TOLERANCE_A
    max_collision_refinements: int = PERIODIC_PATH_COLLISION_REFINEMENTS
    bond_tolerance_A: float = PERIODIC_PATH_BOND_TOLERANCE_A
    max_spacing_ratio: float = PERIODIC_PATH_SPACING_RATIO
    bond_weight: float = PERIODIC_PATH_BOND_WEIGHT
    collision_weight: float = PERIODIC_PATH_COLLISION_WEIGHT
    smoothness_weight: float = PERIODIC_PATH_SMOOTHNESS_WEIGHT
    seed_weight: float = PERIODIC_PATH_SEED_WEIGHT
    fixed_atom_indices: tuple[int, ...] = ()

    def __post_init__(self):
        fixed = tuple(self.fixed_atom_indices)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in fixed):
            raise ValueError("fixed atom indices must be non-negative integers")
        if len(set(fixed)) != len(fixed):
            raise ValueError("fixed atom indices must be unique")
        object.__setattr__(self, "fixed_atom_indices", tuple(int(i) for i in fixed))
        for name, minimum in (
            ("n_images", 3),
            ("max_candidates", 1),
            ("max_iterations", 1),
            ("max_collision_refinements", 0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "gradient_tolerance",
            "collision_scale",
            "bond_tolerance_A",
            "max_spacing_ratio",
            "bond_weight",
            "collision_weight",
            "smoothness_weight",
            "seed_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not np.isfinite(self.collision_tolerance_A)
            or self.collision_tolerance_A < 0
        ):
            raise ValueError("collision_tolerance_A must be finite and non-negative")


@dataclass
class PeriodicPathCandidate:
    """One geometrically screened path and its lattice-image choices."""

    images: list[Atoms]
    product_image_shifts: tuple[tuple[int, int, int], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PeriodicPathResult:
    """Accepted paths, endpoint topology snapshots, and rejected diagnostics."""

    candidates: list[PeriodicPathCandidate]
    product_index_by_reactant: tuple[int, ...]
    reactant_graph: PeriodicQuotientGraph
    product_graph: PeriodicQuotientGraph
    reactant_bond_records: tuple[BondRecord, ...]
    product_bond_records: tuple[BondRecord, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PeriodicPathError(ValueError):
    """No path passed the configured geometric checks."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _mapping(start_symbols, end_symbols, mapping: Sequence[int]) -> np.ndarray:
    count = len(start_symbols)
    if len(end_symbols) != count or count == 0:
        raise ValueError("endpoints must contain the same nonzero number of atoms")
    if mapping is None:
        raise ValueError("an explicit atom mapping is required")
    values = np.asarray(mapping)
    if values.shape != (count,) or values.dtype.kind not in "iu":
        raise ValueError("atom mapping must be a full integer permutation")
    values = values.astype(int)
    if set(values.tolist()) != set(range(count)):
        raise ValueError("atom mapping must be a full integer permutation")
    if any(start_symbols[i] != end_symbols[j] for i, j in enumerate(values)):
        raise ValueError("atom mapping must preserve element identity")
    return values


def _edge_key(edge: ImageEdge) -> tuple[int, int, tuple[int, int, int]]:
    return edge.left, edge.right, edge.shift


def _components_and_shifts(start_edges, end_edges, start, end, end_cell, pbc):
    """Keep unambiguous common edges together while selecting endpoint images."""

    by_start = defaultdict(list)
    by_end = defaultdict(list)
    for edge in start_edges:
        by_start[(edge.left, edge.right)].append(edge)
    for edge in end_edges:
        by_end[(edge.left, edge.right)].append(edge)
    adjacency = defaultdict(list)
    ambiguous = []
    for pair in sorted(set(by_start) & set(by_end)):
        if pair[0] == pair[1]:
            continue
        a, b = by_start[pair], by_end[pair]
        if len(a) != 1 or len(b) != 1:
            ambiguous.append(pair)
            continue
        shift0 = np.asarray(a[0].shift, dtype=int)
        shift1 = np.asarray(b[0].shift, dtype=int)
        delta = shift1 - shift0
        adjacency[pair[0]].append((pair[1], delta, shift0))
        adjacency[pair[1]].append((pair[0], -delta, -shift0))

    count = len(start)
    shifts = np.zeros((count, 3), dtype=int)
    start_gauge = np.zeros((count, 3), dtype=int)
    components = []
    unseen = set(range(count))
    while unseen:
        root = min(unseen)
        queue = deque([root])
        unseen.remove(root)
        members = [root]
        assigned = {root}
        consistent = True
        finite = True
        while queue:
            current = queue.popleft()
            for other, increment, start_increment in sorted(
                adjacency[current], key=lambda row: row[0]
            ):
                candidate = shifts[current] + increment
                gauge = start_gauge[current] + start_increment
                if other not in assigned:
                    assigned.add(other)
                    unseen.discard(other)
                    members.append(other)
                    shifts[other] = candidate
                    start_gauge[other] = gauge
                    queue.append(other)
                else:
                    consistent &= bool(np.array_equal(shifts[other], candidate))
                    finite &= bool(np.array_equal(start_gauge[other], gauge))
        # Self-image and unmatched parallel edges may imply a periodic network.
        member_set = set(members)
        for edge in start_edges:
            if edge.left in member_set and edge.right in member_set:
                required = start_gauge[edge.right] - start_gauge[edge.left]
                if not np.array_equal(required, np.asarray(edge.shift, dtype=int)):
                    finite = False
        if not consistent:
            ambiguous.append(tuple(sorted(members)))
        indices = np.asarray(members, dtype=int)
        # A common integer translation minimizes the total squared displacement
        # when applied to the mean displacement of the component.
        mean_delta = np.mean(
            end[indices] + shifts[indices] @ end_cell - start[indices], axis=0
        )
        _, common_shift = nearest_image(mean_delta, end_cell, pbc)
        shifts[indices] += common_shift
        components.append((tuple(sorted(members)), finite and consistent))
    return shifts, start_gauge, components, ambiguous


def _image_choices(
    shifts, components, start, end, end_cell, start_basis, pbc, maximum,
    fixed=None, fixed_shifts=None,
):
    choices = [shifts.copy()]
    for members, _ in components:
        for axis in np.flatnonzero(pbc):
            for sign in (-1, 1):
                candidate = shifts.copy()
                candidate[np.asarray(members, dtype=int)] += sign * start_basis[axis]
                choices.append(candidate)
    unique = {}
    for candidate in choices:
        candidate = candidate.copy()
        if fixed is not None:
            candidate[fixed] = fixed_shifts
        displacement = end + candidate @ end_cell - start
        score = float(np.sum(displacement * displacement))
        unique[tuple(candidate.ravel())] = (score, candidate)
    return [
        item[1]
        for item in sorted(
            unique.values(), key=lambda item: (item[0], tuple(item[1].ravel()))
        )[:maximum]
    ]


def _initial_images(start, end, cells, shifts, gauge, components):
    count = len(cells)
    fractions = np.linspace(0.0, 1.0, count)
    target = end + shifts @ cells[-1]
    frames = np.asarray([(1.0 - f) * start + f * target for f in fractions])
    for members, finite in components:
        if not finite or len(members) < 2:
            continue
        indices = np.asarray(members, dtype=int)
        first = start[indices] + gauge[indices] @ cells[0]
        last = target[indices] + gauge[indices] @ cells[-1]
        center0, center1 = first.mean(axis=0), last.mean(axis=0)
        rotation, _ = kabsch_align(first - center0, last - center1)
        rotation_vector = Rotation.from_matrix(rotation).as_rotvec()
        residual = last - center1 - (first - center0) @ rotation.T
        for k, fraction in enumerate(fractions):
            current = Rotation.from_rotvec(fraction * rotation_vector).as_matrix()
            contiguous = (first - center0) @ current.T
            contiguous += (
                (1.0 - fraction) * center0 + fraction * center1 + fraction * residual
            )
            frames[k, indices] = contiguous - gauge[indices] @ cells[k]
    frames[0] = start
    frames[-1] = target
    return frames


def _contacts(positions, cell, pbc, numbers, scale):
    radii = np.asarray(covalent_radii, dtype=float)[numbers]
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("all atoms must have finite positive covalent radii")
    cutoff = float(2.0 * scale * radii.max())
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=complete_periodic_cell(cell, pbc),
        pbc=pbc,
    )
    first, second, vectors, shifts = neighbor_list(
        "ijDS", atoms, cutoff, self_interaction=False
    )
    keep = first < second
    same = first == second
    if np.any(same):
        shifts_same = shifts[same]
        first_nonzero = np.argmax(shifts_same != 0, axis=1)
        positive = shifts_same[np.arange(len(shifts_same)), first_nonzero] > 0
        keep[np.flatnonzero(same)] = positive
    first, second, vectors, shifts = (
        first[keep],
        second[keep],
        vectors[keep],
        shifts[keep],
    )
    distances = np.linalg.norm(vectors, axis=1)
    limits = scale * (radii[first] + radii[second])
    clash = distances < limits
    return (
        first[clash],
        second[clash],
        vectors[clash],
        shifts[clash],
        distances[clash],
        limits[clash],
    )


def _separate_exact_overlaps(frames, cells, pbc, numbers, start, target, basis, scale, fixed_mask=None):
    if fixed_mask is None:
        fixed_mask = np.zeros(len(start),dtype=bool)
    for k in range(1, len(frames) - 1):
        for _ in range(PERIODIC_PATH_INITIAL_SEPARATION_STEPS):
            first, second, vectors, _, distances, limits = _contacts(
                frames[k], cells[k], pbc, numbers, scale
            )
            bad = np.flatnonzero(distances <= PERIODIC_PATH_OVERLAP_TOLERANCE_A)
            if not len(bad):
                break
            for index in bad:
                left, right = int(first[index]), int(second[index])
                if left == right:
                    continue
                relative = (target[right] - start[right]) - (target[left] - start[left])
                options = [np.cross(relative, row) for row in basis]
                direction = max(options, key=lambda vector: float(vector @ vector))
                if np.linalg.norm(direction) <= PERIODIC_PATH_OVERLAP_TOLERANCE_A:
                    direction = np.asarray(basis[(left + right) % 3], dtype=float)
                direction = direction / np.linalg.norm(direction)
                displacement = (
                    PERIODIC_PATH_INITIAL_SEPARATION_FACTOR
                    * float(limits[index])
                    * direction
                )
                if fixed_mask[left] and fixed_mask[right]:
                    continue
                if fixed_mask[left]:
                    frames[k, right] += 2 * displacement
                elif fixed_mask[right]:
                    frames[k, left] -= 2 * displacement
                else:
                    frames[k, left] -= displacement
                    frames[k, right] += displacement
    return frames


def _targets(start, end, cells, start_edges, end_edges):
    start_by_key = {_edge_key(edge): edge for edge in start_edges}
    end_by_key = {_edge_key(edge): edge for edge in end_edges}
    keys = sorted(set(start_by_key) | set(end_by_key))
    edges = [
        start_by_key[key] if key in start_by_key else end_by_key[key] for key in keys
    ]
    first = []
    last = []
    for edge in edges:
        shift = np.asarray(edge.shift, dtype=int)
        first.append(
            np.linalg.norm(start[edge.right] + shift @ cells[0] - start[edge.left])
        )
        last.append(
            np.linalg.norm(end[edge.right] + shift @ cells[-1] - end[edge.left])
        )
    return (
        edges,
        np.asarray(first),
        np.asarray(last),
        set(start_by_key) & set(end_by_key),
    )


def _objective(interior, seed, cells, pbc, numbers, edges, d0, d1, config):
    frames = seed.copy()
    frames[1:-1] = interior.reshape(frames[1:-1].shape)
    gradient = np.zeros_like(frames)
    value = 0.0
    fractions = np.linspace(0.0, 1.0, len(frames))
    left = np.asarray([edge.left for edge in edges], dtype=int)
    right = np.asarray([edge.right for edge in edges], dtype=int)
    shifts = np.asarray([edge.shift for edge in edges], dtype=int).reshape((-1, 3))
    for k in range(1, len(frames) - 1):
        if len(edges):
            vectors = frames[k, right] + shifts @ cells[k] - frames[k, left]
            distances = np.linalg.norm(vectors, axis=1)
            if np.any(distances <= PERIODIC_PATH_OVERLAP_TOLERANCE_A):
                raise PeriodicPathError(
                    "a bond vector became singular during geometric optimization"
                )
            residual = distances - ((1.0 - fractions[k]) * d0 + fractions[k] * d1)
            value += config.bond_weight * float(residual @ residual)
            factors = (
                2.0
                * config.bond_weight
                * residual[:, None]
                * vectors
                / distances[:, None]
            )
            np.add.at(gradient[k], left, -factors)
            np.add.at(gradient[k], right, factors)
        first, second, vectors, _, distances, limits = _contacts(
            frames[k], cells[k], pbc, numbers, config.collision_scale
        )
        if len(first):
            if np.any(distances <= PERIODIC_PATH_OVERLAP_TOLERANCE_A):
                raise PeriodicPathError(
                    "an atomic overlap became singular during geometric optimization"
                )
            residual = limits - distances
            value += config.collision_weight * float(residual @ residual)
            factors = (
                2.0
                * config.collision_weight
                * residual[:, None]
                * vectors
                / distances[:, None]
            )
            np.add.at(gradient[k], first, factors)
            np.add.at(gradient[k], second, -factors)
    second_difference = frames[:-2] - 2.0 * frames[1:-1] + frames[2:]
    value += config.smoothness_weight * float(np.sum(second_difference**2))
    gradient[:-2] += 2.0 * config.smoothness_weight * second_difference
    gradient[1:-1] -= 4.0 * config.smoothness_weight * second_difference
    gradient[2:] += 2.0 * config.smoothness_weight * second_difference
    delta = frames[1:-1] - seed[1:-1]
    value += config.seed_weight * float(np.sum(delta**2))
    gradient[1:-1] += 2.0 * config.seed_weight * delta
    return value, gradient[1:-1].ravel()


def _free_objective(interior, seed, cells, pbc, numbers, edges, d0, d1, config, free):
    full = seed[1:-1].copy()
    full[:, free, :] = interior.reshape((len(full), len(free), 3))
    value, gradient = _objective(full.ravel(), seed, cells, pbc, numbers, edges, d0, d1, config)
    return value, gradient.reshape(full.shape)[:, free, :].ravel()


def _diagnostics(frames, cells, pbc, numbers, edges, d0, d1, conserved, result, config):
    clash = 0.0
    exact = False
    max_bond = 0.0
    fractions = np.linspace(0.0, 1.0, len(frames))
    for k, frame in enumerate(frames):
        _, _, _, _, distances, limits = _contacts(
            frame, cells[k], pbc, numbers, config.collision_scale
        )
        if len(distances):
            clash = max(clash, float(np.max(limits - distances)))
            exact |= bool(np.any(distances <= PERIODIC_PATH_OVERLAP_TOLERANCE_A))
        for index, edge in enumerate(edges):
            if _edge_key(edge) in conserved:
                vector = (
                    frame[edge.right]
                    + np.asarray(edge.shift) @ cells[k]
                    - frame[edge.left]
                )
                expected = (1.0 - fractions[k]) * d0[index] + fractions[k] * d1[index]
                max_bond = max(max_bond, abs(float(np.linalg.norm(vector)) - expected))
    steps = np.sqrt(np.mean(np.sum(np.diff(frames, axis=0) ** 2, axis=2), axis=1))
    mean_step = float(np.mean(steps))
    spacing = (
        float(np.max(steps) / mean_step)
        if mean_step > PERIODIC_PATH_OVERLAP_TOLERANCE_A
        else 1.0
    )
    grad = float(np.max(np.abs(result.jac))) if len(result.jac) else 0.0
    return {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "maximum_gradient_component": grad,
        "maximum_collision_penetration_A": clash,
        "exact_overlap": exact,
        "maximum_conserved_bond_deviation_A": max_bond,
        "spacing_ratio": spacing,
        "accepted": bool(
            result.success
            and np.isfinite(result.fun)
            and grad
            <= config.gradient_tolerance * PERIODIC_PATH_GRADIENT_ACCEPTANCE_FACTOR
            and not exact
            and clash <= config.collision_tolerance_A
            and max_bond <= config.bond_tolerance_A
            and spacing <= config.max_spacing_ratio
        ),
    }


def interpolate_periodic_paths(
    reactant: MolecularCrystal,
    product: MolecularCrystal,
    *,
    product_index_by_reactant: Sequence[int],
    config: PeriodicPathConfig | None = None,
) -> PeriodicPathResult:
    """Generate bounded geometric path candidates from mapped endpoints.

    The mapping gives the product index of each reactant atom. Equivalent
    periodic cell bases are aligned before interpolation. Candidates differ by
    bounded component windings and contain only paths that pass the configured
    geometric checks. Rejected candidates remain available in diagnostics.

    Endpoint bonds are read from the structure objects; no new connectivity is
    guessed here. When all candidates fail, :class:`PeriodicPathError` carries
    the rejection diagnostics. Geometric acceptance is not a statement about
    a mechanism or a physical energy surface.
    """

    if not isinstance(reactant, MolecularCrystal) or not isinstance(
        product, MolecularCrystal
    ):
        raise TypeError("both endpoints must be MolecularCrystal objects")
    config = config or PeriodicPathConfig()
    start_atoms, end_atoms = reactant.to_ase(), product.to_ase()
    pbc = np.asarray(reactant.pbc, dtype=bool)
    if pbc.shape != (3,) or not np.array_equal(
        pbc, np.asarray(product.pbc, dtype=bool)
    ):
        raise ValueError("endpoint periodicity flags must agree")
    mapping = _mapping(
        start_atoms.get_chemical_symbols(),
        end_atoms.get_chemical_symbols(),
        product_index_by_reactant,
    )
    inverse_map = np.argsort(mapping)
    start = np.asarray(start_atoms.positions, dtype=float)
    raw_product = np.asarray(end_atoms.positions, dtype=float)
    end = raw_product[mapping]
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("endpoint positions must be finite")
    fractions = np.linspace(0.0, 1.0, config.n_images)
    cells, end_transform = interpolate_cells(
        reactant.lattice, product.lattice, pbc, fractions
    )
    fixed = np.asarray(config.fixed_atom_indices, dtype=int)
    if np.any(fixed >= len(start)):
        raise ValueError("fixed atom index out of range")
    fixed_mask = np.zeros(len(start), dtype=bool)
    fixed_mask[fixed] = True
    free = np.flatnonzero(~fixed_mask)
    fixed_shifts = np.zeros((len(fixed), 3), dtype=int)
    for k, i in enumerate(fixed):
        vector, shift = nearest_image(end[i] - start[i], cells[-1], pbc)
        if np.linalg.norm(vector) > PERIODIC_PATH_OVERLAP_TOLERANCE_A:
            raise ValueError("fixed atom endpoints differ in Cartesian position")
        fixed_shifts[k] = shift
    start_records = tuple(reactant.get_bond_records())
    end_records = tuple(product.get_bond_records())
    start_edges, start_graph = records_to_edges(
        reactant,
        start,
        np.arange(len(start)),
        np.asarray(reactant.lattice),
        pbc,
        start_records,
    )
    product_edges, product_graph = records_to_edges(
        product, raw_product, inverse_map, np.asarray(product.lattice), pbc, end_records
    )
    end_basis_inverse = _integer_inverse(end_transform)
    end_edges = [transform_edge(edge, end_basis_inverse) for edge in product_edges]
    shifts, gauge, components, ambiguous = _components_and_shifts(
        start_edges, end_edges, start, end, cells[-1], pbc
    )
    _, start_basis = canonical_periodic_basis(reactant.lattice, pbc)
    choices = _image_choices(
        shifts,
        components,
        start,
        end,
        cells[-1],
        start_basis,
        pbc,
        config.max_candidates,
        fixed,
        fixed_shifts,
    )
    basis_vectors, _ = canonical_periodic_basis(reactant.lattice, pbc)
    basis_vectors = complete_periodic_cell(basis_vectors, pbc)
    numbers = np.asarray(start_atoms.numbers, dtype=int)
    candidates = []
    rejected = []
    seen_choices = set()
    for choice in choices:
        choice = choice.copy()
        choice[fixed] = fixed_shifts
        choice_key = tuple(choice.ravel())
        if choice_key in seen_choices:
            continue
        seen_choices.add(choice_key)
        adjusted_end_edges = [
            transform_edge(edge, np.eye(3, dtype=int), choice) for edge in end_edges
        ]
        target = end + choice @ cells[-1]
        edges, d0, d1, conserved = _targets(
            start, target, cells, start_edges, adjusted_end_edges
        )
        seed = _initial_images(start, end, cells, choice, gauge, components)
        seed[1:-1, fixed] = start[fixed]
        seed = _separate_exact_overlaps(
            seed,
            cells,
            pbc,
            numbers,
            start,
            target,
            basis_vectors,
            config.collision_scale,
            fixed_mask,
        )
        try:
            current = seed[1:-1, free].ravel()
            refinement_history = []
            for refinement in range(config.max_collision_refinements + 1):
                stage = replace(
                    config,
                    collision_weight=config.collision_weight
                    * PERIODIC_PATH_COLLISION_WEIGHT_MULTIPLIER**refinement,
                )
                objective_args = (seed, cells, pbc, numbers, edges, d0, d1, stage, free)
                if len(free):
                    optimum = minimize(
                        _free_objective,
                        current,
                        args=objective_args,
                        method="L-BFGS-B",
                        jac=True,
                        options={
                            "maxiter": config.max_iterations,
                            "gtol": config.gradient_tolerance,
                            "ftol": PERIODIC_PATH_FUNCTION_TOLERANCE,
                        },
                    )
                else:
                    value, gradient = _free_objective(current, *objective_args)
                    optimum = OptimizeResult(x=current, fun=value, jac=gradient,
                                             success=True, nit=0, message="all atoms fixed")
                current = optimum.x
                frames = seed.copy()
                frames[1:-1, free] = current.reshape((len(frames)-2,len(free),3))
                diagnostic = _diagnostics(
                    frames,
                    cells,
                    pbc,
                    numbers,
                    edges,
                    d0,
                    d1,
                    conserved,
                    optimum,
                    stage,
                )
                refinement_history.append(
                    {
                        "collision_weight": stage.collision_weight,
                        "iterations": diagnostic["iterations"],
                        "maximum_collision_penetration_A": diagnostic[
                            "maximum_collision_penetration_A"
                        ],
                        "optimizer_success": diagnostic["optimizer_success"],
                    }
                )
                if diagnostic["accepted"] or not diagnostic["optimizer_success"]:
                    break
                if (
                    diagnostic["maximum_collision_penetration_A"]
                    <= config.collision_tolerance_A
                ):
                    break
            diagnostic["collision_refinement_steps"] = len(refinement_history) - 1
            diagnostic["refinement_history"] = refinement_history
            diagnostic["total_iterations"] = sum(
                item["iterations"] for item in refinement_history
            )
        except PeriodicPathError as exc:
            rejected.append(
                {
                    "accepted": False,
                    "reason": str(exc),
                    "product_image_shifts": choice.tolist(),
                }
            )
            continue
        diagnostic["product_image_shifts"] = choice.tolist()
        if not diagnostic["accepted"]:
            rejected.append(diagnostic)
            continue
        images = []
        for k, frame in enumerate(frames):
            image = materialize_atoms_frame(
                start_atoms, frame, info_updates={"path_fraction": float(fractions[k])}
            )
            image.set_cell(cells[k], scale_atoms=False)
            image.set_pbc(pbc)
            if len(fixed):
                image.set_constraint(FixAtoms(indices=fixed))
            image.info.pop("_molcrys_bond_records", None)
            if "molecule_index" in image.arrays:
                image.set_array(
                    "reactant_molecule_index", image.arrays["molecule_index"].copy()
                )
                del image.arrays["molecule_index"]
            images.append(image)
        candidates.append(
            PeriodicPathCandidate(
                images, tuple(tuple(int(x) for x in row) for row in choice), diagnostic
            )
        )
    diagnostics = {
        "rejected_candidates": rejected,
        "candidate_count_requested": config.max_candidates,
        "fixed_atom_indices": list(config.fixed_atom_indices),
        "candidate_count_evaluated": len(seen_choices),
        "ambiguous_common_edges": [list(item) for item in ambiguous],
        "end_basis_transform": end_transform.tolist(),
        "candidate_scope": "base and single-component one-cell windings",
    }
    if not candidates:
        raise PeriodicPathError(
            "no candidate passed the configured geometric checks", diagnostics
        )
    return PeriodicPathResult(
        candidates=candidates,
        product_index_by_reactant=tuple(int(value) for value in mapping),
        reactant_graph=start_graph,
        product_graph=product_graph,
        reactant_bond_records=start_records,
        product_bond_records=end_records,
        diagnostics=diagnostics,
    )


__all__ = [
    "PeriodicPathCandidate",
    "PeriodicPathConfig",
    "PeriodicPathError",
    "PeriodicPathResult",
    "interpolate_periodic_paths",
]
