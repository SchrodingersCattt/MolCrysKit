"""Independent-image IDPP refinement, using an explicit atom-mapped warm start.

The image-dependent pair potential is from Smidstrup et al.,
J. Chem. Phys. 140, 214106 (2014), DOI 10.1063/1.4878664. This geometry-only
operation does not calculate physical energies or band/spring forces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import complete_cell, find_mic, minkowski_reduce
from scipy.optimize import minimize

from .path_constraints import (
    PathRestraint,
    _prepare_restraints,
    _restraint_diagnostics,
    _restraint_objective,
)

from ..constants.idpp import (
    IDPP_FUNCTION_TOLERANCE,
    IDPP_GRADIENT_TOLERANCE,
    IDPP_MAX_ITERATIONS,
    IDPP_MINIMUM_DISTANCE_A,
)


@dataclass
class IDPPRefinementResult:
    """Refined ASE images with explicit per-image convergence diagnostics."""

    images: list[Atoms]
    diagnostics: tuple[dict, ...]
    restraint_diagnostics: tuple[dict, ...] = ()

    @property
    def converged(self) -> bool:
        """Whether every interior image satisfied its constrained tolerance."""
        return all(item["converged"] for item in self.diagnostics)


def _distances(positions, cell, pbc, first, second):
    vectors = np.asarray(positions)[second] - np.asarray(positions)[first]
    if np.any(pbc):
        reduced, _ = minkowski_reduce(complete_cell(cell), pbc=pbc)
        vectors, distances = find_mic(vectors, reduced, pbc=pbc)
    else:
        distances = np.linalg.norm(vectors, axis=1)
    return np.asarray(vectors), np.asarray(distances)


def _objective(positions, cell, pbc, first, second, targets, minimum_distance):
    vectors, distances = _distances(positions, cell, pbc, first, second)
    if np.any(distances <= minimum_distance):
        raise ValueError("IDPP is singular for overlapping atom pairs")
    difference = distances - targets
    value = float(np.sum(difference**2 / distances**4))
    derivative = 2 * difference / distances**4 - 4 * difference**2 / distances**5
    contributions = derivative[:, None] * vectors / distances[:, None]
    gradient = np.zeros_like(positions)
    np.add.at(gradient, first, -contributions)
    np.add.at(gradient, second, contributions)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("IDPP objective or gradient is non-finite")
    return value, gradient


def _validate_indices(indices, count):
    values = tuple(indices)
    if any(
        isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= count
        for i in values
    ):
        raise ValueError("prescribed atom indices must be valid non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError("prescribed atom indices must be unique")
    return np.asarray(values, dtype=int)


def refine_idpp_path(
    images: Sequence[Atoms],
    *,
    fractions: Sequence[float] | None = None,
    prescribed_atom_indices: Sequence[int] = (),
    max_iterations: int = IDPP_MAX_ITERATIONS,
    gradient_tolerance: float = IDPP_GRADIENT_TOLERANCE,
    minimum_distance: float = IDPP_MINIMUM_DISTANCE_A,
    restraints: Sequence[PathRestraint] = (),
) -> IDPPRefinementResult:
    """Refine explicitly supplied path images, retaining both endpoints.

    Every interior image starts at the supplied geometry. No linear initial
    path is constructed. Pair-distance targets interpolate the endpoint
    minimum-image distances. Each frame uses its own cell and the common
    periodicity mask. The caller must put different lattice bases in a
    consistent representation and provide corresponding atom indices.

    Prescribed atoms stay at their input position in *each* image. Therefore
    an externally supplied moving fragment can stay fixed along its prescribed
    path; this prescription applies only during this operation. Existing ASE
    FixAtoms constraints are also honored; other ASE
    constraints are rejected. If atom_id arrays exist, their values must be
    unique and their ordering identical across all images.

    Optional geometric restraints contribute to each interior objective;
    their targets, weights, units, and any chemical interpretation belong to
    the caller. They do not move endpoints or prescribed atoms.

    Each image is optimized independently, without an inter-image spring.
    Bond preservation, collision avoidance, path spacing and physical
    mechanisms are not guaranteed. Exact overlaps are rejected because the
    objective is singular. Diagnostics report unconverged images. Input frames
    are not mutated.
    """
    frames = tuple(images)
    if len(frames) < 3:
        raise ValueError("at least three explicitly initialized images are required")
    if any(not isinstance(frame, Atoms) for frame in frames):
        raise TypeError("each image must be an ASE Atoms object")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be positive and finite")
    if not np.isfinite(minimum_distance) or minimum_distance <= 0:
        raise ValueError("minimum_distance must be positive and finite")

    count = len(frames[0])
    if count < 2:
        raise ValueError("at least two atoms are required")
    prescribed = _validate_indices(prescribed_atom_indices, count)
    if fractions is None:
        progress = np.linspace(0.0, 1.0, len(frames))
    else:
        progress = np.asarray(fractions, dtype=float)
        if (
            progress.shape != (len(frames),)
            or not np.all(np.isfinite(progress))
            or not np.isclose(progress[0], 0.0)
            or not np.isclose(progress[-1], 1.0)
            or np.any(np.diff(progress) <= 0)
        ):
            raise ValueError("fractions must increase strictly from 0 to 1")

    numbers = frames[0].numbers
    pbc = frames[0].pbc
    ids = frames[0].arrays.get("atom_id")
    if ids is not None and np.asarray(ids).shape != (count,):
        raise ValueError("atom_id must contain one scalar per atom")
    if ids is not None and len(np.unique(ids)) != count:
        raise ValueError("atom_id values must be unique")
    for frame in frames:
        if len(frame) != count or not np.array_equal(frame.numbers, numbers):
            raise ValueError("all frames must have identical atom order and elements")
        if not np.array_equal(frame.pbc, pbc):
            raise ValueError("all frames must have the same periodic directions")
        if not np.all(np.isfinite(frame.positions)) or not np.all(np.isfinite(frame.cell)):
            raise ValueError("positions and cells must be finite")
        if (ids is None) != ("atom_id" not in frame.arrays):
            raise ValueError("atom_id arrays must be present in all frames or none")
        if ids is not None and not np.array_equal(ids, frame.arrays["atom_id"]):
            raise ValueError("atom_id order must be identical in every frame")
        if any(not isinstance(c, FixAtoms) for c in frame.constraints):
            raise ValueError("only ASE FixAtoms constraints are supported")

    prepared_restraints = _prepare_restraints(restraints, len(frames))
    first, second = np.triu_indices(count, 1)
    _, start = _distances(frames[0].positions, frames[0].cell, pbc, first, second)
    _, end = _distances(frames[-1].positions, frames[-1].cell, pbc, first, second)
    if np.any(start <= minimum_distance) or np.any(end <= minimum_distance):
        raise ValueError("IDPP endpoint contains overlapping atom pairs")

    output = [frame.copy() for frame in frames]
    diagnostics = []
    for index in range(1, len(frames) - 1):
        frame = frames[index]
        seed = np.asarray(frame.positions, dtype=float).copy()
        target = (1.0 - progress[index]) * start + progress[index] * end
        fixed = np.zeros(count, dtype=bool)
        fixed[prescribed] = True
        for constraint in frame.constraints:
            fixed[constraint.get_indices()] = True
        free = np.flatnonzero(~fixed)
        initial, _ = _objective(seed, frame.cell, pbc, first, second, target, minimum_distance)
        initial_restraint, _ = _restraint_objective(
            prepared_restraints, index, seed, frame.cell, pbc
        )
        initial += initial_restraint

        def objective(flat):
            positions = seed.copy()
            positions[free] = flat.reshape((-1, 3))
            value, gradient = _objective(
                positions, frame.cell, pbc, first, second, target, minimum_distance
            )
            extra_value, extra_gradient = _restraint_objective(
                prepared_restraints, index, positions, frame.cell, pbc
            )
            return value + extra_value, (gradient + extra_gradient)[free].ravel()

        if len(free):
            result = minimize(
                objective,
                seed[free].ravel(),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": int(max_iterations),
                    "gtol": float(gradient_tolerance),
                    "ftol": IDPP_FUNCTION_TOLERANCE,
                },
            )
            positions = seed.copy()
            positions[free] = result.x.reshape((-1, 3))
            value, gradient = objective(result.x)
            maximum_gradient = float(np.max(np.abs(gradient))) if len(gradient) else 0.0
            converged = bool(result.success and maximum_gradient <= gradient_tolerance)
            iterations = int(result.nit)
            message = str(result.message)
        else:
            positions, value, maximum_gradient = seed, initial, 0.0
            converged, iterations, message = True, 0, "all atoms prescribed"

        output[index].positions[:] = positions
        output[index].info.pop("_molcrys_bond_records", None)
        diagnostics.append(
            {
                "image": index,
                "fraction": float(progress[index]),
                "initial_objective": initial,
                "final_objective": float(value),
                "maximum_free_gradient_component": maximum_gradient,
                "converged": converged,
                "iterations": iterations,
                "message": message,
                "prescribed_atom_count": int(np.count_nonzero(fixed)),
            }
        )
    restraint_info = _restraint_diagnostics(
        prepared_restraints, [frame.positions for frame in output],
        [frame.cell for frame in output], pbc
    )
    return IDPPRefinementResult(output, tuple(diagnostics), tuple(restraint_info))
