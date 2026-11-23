"""Composable geometric coordinates and soft restraints for path refinement.

Coordinates use reactant-order atom indices. Each relative vector either uses
its minimum image or an explicit integer lattice shift. At minimum-image
switching surfaces derivatives are discontinuous; callers requiring a fixed
periodic branch should supply explicit shifts for a consistently unwrapped
path. No charge, spin, bond order, mechanism, or target is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from ..constants.path_constraints import (
    PATH_COORDINATE_MIN_ANGLE_SINE,
    PATH_COORDINATE_MIN_PLANE_AREA_A2,
    PATH_COORDINATE_MIN_VECTOR_A,
)
from ._periodic_path_geometry import nearest_image

ImageShift = tuple[int, int, int] | None


class PathCoordinate(Protocol):
    """A differentiable scalar of one image with its Cartesian gradient.

    Custom implementations must not mutate the arguments. The returned
    gradient has shape ``(n_atoms, 3)`` and is with respect to positions at
    fixed cell. The value, gradient, targets, and restraint weights must be in
    consistent units chosen by the caller.
    """

    def value_and_gradient(
        self, positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray
    ) -> tuple[float, np.ndarray]: ...


def _indices(values: Sequence[int], expected: int) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) != expected or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0
        for v in values
    ):
        raise ValueError(f"expected {expected} non-negative integer atom indices")
    indices = tuple(int(v) for v in values)
    if len(set(indices)) != len(indices):
        raise ValueError("coordinate atom indices must be distinct")
    return indices


def _shift(value: ImageShift) -> ImageShift:
    if value is None:
        return None
    values = tuple(value)
    if len(values) != 3 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in values
    ):
        raise ValueError("image shifts must be integer triples or None")
    return tuple(int(v) for v in values)


def _relative(positions, cell, pbc, center, other, shift):
    if max(center, other) >= len(positions):
        raise ValueError("coordinate atom index is outside the image")
    delta = np.asarray(positions[other] - positions[center], dtype=float)
    if shift is None:
        return nearest_image(delta, cell, pbc)[0]
    flags = np.asarray(pbc, dtype=bool)
    image = np.asarray(shift, dtype=int)
    if flags.shape != (3,) or np.any(image[~flags] != 0):
        raise ValueError("image shifts must be zero on non-periodic axes")
    return delta + image @ np.asarray(cell, dtype=float)


@dataclass(frozen=True)
class DistanceCoordinate:
    """Distance in Å between two atoms, using MIC or an explicit image."""

    atom_i: int
    atom_j: int
    image_shift: ImageShift = None

    def __post_init__(self):
        i, j = _indices((self.atom_i, self.atom_j), 2)
        object.__setattr__(self, "atom_i", i)
        object.__setattr__(self, "atom_j", j)
        object.__setattr__(self, "image_shift", _shift(self.image_shift))

    def value_and_gradient(self, positions, cell, pbc):
        vector = _relative(
            positions, cell, pbc, self.atom_i, self.atom_j, self.image_shift
        )
        length = float(np.linalg.norm(vector))
        if length <= PATH_COORDINATE_MIN_VECTOR_A:
            raise ValueError("distance coordinate is singular at overlapping atoms")
        gradient = np.zeros_like(positions, dtype=float)
        gradient[self.atom_i] -= vector / length
        gradient[self.atom_j] += vector / length
        return length, gradient


@dataclass(frozen=True)
class AngleCoordinate:
    """Angle in radians at ``center``, using separate images for its arms."""

    atom_i: int
    center: int
    atom_k: int
    image_shifts: tuple[ImageShift, ImageShift] = (None, None)

    def __post_init__(self):
        i, center, k = _indices((self.atom_i, self.center, self.atom_k), 3)
        shifts = tuple(self.image_shifts)
        if len(shifts) != 2:
            raise ValueError("angle coordinate requires two image shifts")
        object.__setattr__(self, "atom_i", i)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "atom_k", k)
        object.__setattr__(self, "image_shifts", tuple(_shift(x) for x in shifts))

    def value_and_gradient(self, positions, cell, pbc):
        u = _relative(
            positions, cell, pbc, self.center, self.atom_i, self.image_shifts[0]
        )
        v = _relative(
            positions, cell, pbc, self.center, self.atom_k, self.image_shifts[1]
        )
        nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if min(nu, nv) <= PATH_COORDINATE_MIN_VECTOR_A:
            raise ValueError("angle coordinate is singular at overlapping atoms")
        cosine = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
        sine = float(np.linalg.norm(np.cross(u, v)) / (nu * nv))
        if sine <= PATH_COORDINATE_MIN_ANGLE_SINE:
            raise ValueError("angle coordinate is singular for collinear arms")
        du = -(v / (nu * nv) - cosine * u / (nu * nu)) / sine
        dv = -(u / (nu * nv) - cosine * v / (nv * nv)) / sine
        gradient = np.zeros_like(positions, dtype=float)
        gradient[self.atom_i] += du
        gradient[self.atom_k] += dv
        gradient[self.center] -= du + dv
        return float(np.arctan2(sine, cosine)), gradient


@dataclass(frozen=True)
class SignedVolumeCoordinate:
    """Oriented triple product in Å³ for three atoms around a center.

    Zero means the four sites are coplanar (including degenerate cases).
    Reversing two neighbors reverses the sign. This is a geometric coordinate,
    not an inference of chirality, electronic state, or bonding.
    """

    center: int
    atom_i: int
    atom_j: int
    atom_k: int
    image_shifts: tuple[ImageShift, ImageShift, ImageShift] = (None, None, None)

    def __post_init__(self):
        center, i, j, k = _indices(
            (self.center, self.atom_i, self.atom_j, self.atom_k), 4
        )
        shifts = tuple(self.image_shifts)
        if len(shifts) != 3:
            raise ValueError("signed-volume coordinate requires three image shifts")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "atom_i", i)
        object.__setattr__(self, "atom_j", j)
        object.__setattr__(self, "atom_k", k)
        object.__setattr__(self, "image_shifts", tuple(_shift(x) for x in shifts))

    def value_and_gradient(self, positions, cell, pbc):
        i, j, k = self.atom_i, self.atom_j, self.atom_k
        u = _relative(positions, cell, pbc, self.center, i, self.image_shifts[0])
        v = _relative(positions, cell, pbc, self.center, j, self.image_shifts[1])
        w = _relative(positions, cell, pbc, self.center, k, self.image_shifts[2])
        du, dv, dw = np.cross(v, w), np.cross(w, u), np.cross(u, v)
        gradient = np.zeros_like(positions, dtype=float)
        gradient[i] += du
        gradient[j] += dv
        gradient[k] += dw
        gradient[self.center] -= du + dv + dw
        return float(u @ du), gradient


@dataclass(frozen=True)
class SignedPlaneDistanceCoordinate:
    """Signed distance in Å from a center to a plane through three atoms.

    Reversing two plane atoms changes the sign. The plane atoms are unwrapped
    relative to the center using minimum images or caller-supplied shifts.
    """

    center: int
    atom_i: int
    atom_j: int
    atom_k: int
    image_shifts: tuple[ImageShift, ImageShift, ImageShift] = (None, None, None)

    def __post_init__(self):
        center, i, j, k = _indices(
            (self.center, self.atom_i, self.atom_j, self.atom_k), 4
        )
        shifts = tuple(self.image_shifts)
        if len(shifts) != 3:
            raise ValueError("plane-distance coordinate requires three image shifts")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "atom_i", i)
        object.__setattr__(self, "atom_j", j)
        object.__setattr__(self, "atom_k", k)
        object.__setattr__(self, "image_shifts", tuple(_shift(x) for x in shifts))

    def value_and_gradient(self, positions, cell, pbc):
        i, j, k = self.atom_i, self.atom_j, self.atom_k
        u = _relative(positions, cell, pbc, self.center, i, self.image_shifts[0])
        v = _relative(positions, cell, pbc, self.center, j, self.image_shifts[1])
        w = _relative(positions, cell, pbc, self.center, k, self.image_shifts[2])
        normal = np.cross(v - u, w - u)
        area = float(np.linalg.norm(normal))
        if area <= PATH_COORDINATE_MIN_PLANE_AREA_A2:
            raise ValueError(
                "plane-distance coordinate is singular for collinear plane atoms"
            )
        unit = normal / area
        volume = float(u @ np.cross(v, w))
        volume_grad = (np.cross(v, w), np.cross(w, u), np.cross(u, v))
        area_grad = (
            np.cross(v - w, unit),
            np.cross(w - u, unit),
            np.cross(unit, v - u),
        )
        derivatives = [
            -dvolume / area + volume * darea / (area * area)
            for dvolume, darea in zip(volume_grad, area_grad)
        ]
        gradient = np.zeros_like(positions, dtype=float)
        for index, derivative in zip((i, j, k), derivatives):
            gradient[index] += derivative
        gradient[self.center] -= sum(derivatives)
        return -volume / area, gradient


@dataclass(frozen=True)
class LinearCombinationCoordinate:
    """A caller-scaled linear combination of arbitrary path coordinates."""

    terms: tuple[tuple[float, PathCoordinate], ...]

    def __post_init__(self):
        terms = tuple(
            (float(coefficient), coordinate) for coefficient, coordinate in self.terms
        )
        if not terms or any(
            not np.isfinite(c) or not callable(getattr(x, "value_and_gradient", None))
            for c, x in terms
        ):
            raise ValueError("terms must contain finite coefficients and coordinates")
        object.__setattr__(self, "terms", terms)

    def value_and_gradient(self, positions, cell, pbc):
        value = 0.0
        gradient = np.zeros_like(positions, dtype=float)
        for coefficient, coordinate in self.terms:
            component, derivative = _evaluate_coordinate(
                coordinate, positions, cell, pbc
            )
            value += coefficient * component
            gradient += coefficient * derivative
        return float(value), gradient


@dataclass(frozen=True)
class PathRestraint:
    """Quadratic soft restraint with explicit scalar or per-image schedules.

    ``targets`` and ``weights`` are each either a scalar or a sequence with
    one entry per path image, including endpoints. Zero weights disable the
    restraint on those images. Endpoints are never moved by these optimizers;
    endpoint values are reported as diagnostics when the weight is positive.
    """

    coordinate: PathCoordinate
    targets: float | Sequence[float]
    weights: float | Sequence[float]
    name: str | None = None

    def __post_init__(self):
        if not callable(getattr(self.coordinate, "value_and_gradient", None)):
            raise TypeError("coordinate must provide value_and_gradient")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("restraint name must be a string or None")


def _schedule(value, count, name, *, nonnegative=False):
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite scalar or per-image sequence"
        ) from exc
    if array.ndim == 0:
        array = np.full(count, float(array))
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or one value per image")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array.copy()


def _prepare_restraints(restraints, count):
    prepared = []
    for restraint in restraints:
        if not isinstance(restraint, PathRestraint):
            raise TypeError("restraints must contain PathRestraint objects")
        prepared.append(
            (
                restraint,
                _schedule(restraint.targets, count, "restraint targets"),
                _schedule(
                    restraint.weights, count, "restraint weights", nonnegative=True
                ),
            )
        )
    return tuple(prepared)


def _evaluate_coordinate(coordinate, positions, cell, pbc):
    value, gradient = coordinate.value_and_gradient(positions, cell, pbc)
    value = float(value)
    gradient = np.asarray(gradient, dtype=float)
    if (
        not np.isfinite(value)
        or gradient.shape != positions.shape
        or not np.all(np.isfinite(gradient))
    ):
        raise ValueError("coordinate returned a non-finite value or invalid gradient")
    return value, gradient


def _restraint_objective(prepared, index, positions, cell, pbc):
    total = 0.0
    gradient = np.zeros_like(positions, dtype=float)
    for restraint, targets, weights in prepared:
        weight = float(weights[index])
        if weight == 0.0:
            continue
        value, derivative = _evaluate_coordinate(
            restraint.coordinate, positions, cell, pbc
        )
        residual = value - targets[index]
        total += weight * residual * residual
        gradient += (2.0 * weight * residual) * derivative
    return float(total), gradient


def _restraint_diagnostics(prepared, frames, cells, pbc):
    records = []
    for restraint, targets, weights in prepared:
        values = []
        for k, frame in enumerate(frames):
            if weights[k] == 0:
                values.append(None)
            else:
                value, _ = _evaluate_coordinate(
                    restraint.coordinate, frame, cells[k], pbc
                )
                values.append(value)
        residuals = [
            None if value is None else float(value - target)
            for value, target in zip(values, targets)
        ]
        penalties = [
            None if residual is None else float(weight * residual**2)
            for residual, weight in zip(residuals, weights)
        ]
        records.append(
            {
                "name": restraint.name,
                "targets": targets.tolist(),
                "weights": weights.tolist(),
                "values": values,
                "residuals": residuals,
                "weighted_penalties": penalties,
                "applied_to_objective": [
                    bool(0 < k < len(frames) - 1 and weights[k] > 0)
                    for k in range(len(frames))
                ],
            }
        )
    return records
