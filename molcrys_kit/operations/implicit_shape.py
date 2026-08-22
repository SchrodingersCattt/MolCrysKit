"""Bounded vectorized implicit shapes shared by crystal carving operations.

An :class:`ImplicitShape` is purely geometric: ``field(x, y, z) <= 0``
denotes the interior and ``bounds`` supplies a finite Cartesian bounding box.
Material operations decide whether the interior is retained (nanoclusters) or
removed (voids); the shape itself has no topology or output semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ShapeField = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
DEFAULT_SHAPE_BATCH_SIZE = 100_000


def _positive_values(values: Sequence[float], name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} must contain {length} positive finite values.")
    return array


def _axis_vector(axis: str | Sequence[float]) -> tuple[np.ndarray, str]:
    if isinstance(axis, str):
        axis_name = axis.lower()
        if axis_name not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', 'z', or a finite non-zero 3-vector.")
        vector = np.zeros(3, dtype=float)
        vector[{"x": 0, "y": 1, "z": 2}[axis_name]] = 1.0
        return vector, axis_name

    vector = np.asarray(axis, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("axis must be 'x', 'y', 'z', or a finite non-zero 3-vector.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("axis vector must be non-zero.")
    unit = vector / norm
    label = "_".join(f"{value:.6g}" for value in unit)
    return unit, label


@dataclass(frozen=True)
class ImplicitShape:
    """A bounded vectorized implicit shape.

    Parameters
    ----------
    field
        Callable accepting equally shaped NumPy ``x``, ``y`` and ``z`` arrays
        and returning one finite real value per point.
    bounds
        Cartesian ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` bounds in
        Angstrom relative to the shape center.
    name
        Human-readable identifier stored in output metadata.
    parameters
        Optional serializable preset parameters stored in output metadata.
    """

    field: ShapeField
    bounds: np.ndarray
    name: str = "custom"
    parameters: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.field):
            raise TypeError("field must be callable.")
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (3, 2):
            raise ValueError("bounds must have shape (3, 2).")
        if not np.isfinite(bounds).all():
            raise ValueError("bounds must contain only finite values.")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each bounds lower limit must be smaller than its upper limit.")
        name = str(self.name).strip()
        if not name:
            raise ValueError("name must not be empty.")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")
        stored_bounds = bounds.copy()
        stored_bounds.setflags(write=False)
        object.__setattr__(self, "bounds", stored_bounds)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def sphere(cls, radius: float) -> "ImplicitShape":
        radius_value = float(_positive_values([radius], "radius", 1)[0])
        bounds = np.repeat([[-radius_value, radius_value]], 3, axis=0)

        def field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            return (x * x + y * y + z * z) / (radius_value * radius_value) - 1.0

        return cls(field, bounds, "sphere", {"radius_A": radius_value})

    @classmethod
    def box(cls, size: Sequence[float]) -> "ImplicitShape":
        full_size = _positive_values(size, "size", 3)
        half = full_size / 2.0
        bounds = np.column_stack((-half, half))

        def field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            return np.maximum.reduce(
                (np.abs(x) / half[0], np.abs(y) / half[1], np.abs(z) / half[2])
            ) - 1.0

        return cls(field, bounds, "box", {"size_A": full_size.tolist()})

    @classmethod
    def ellipsoid(cls, semi_axes: Sequence[float]) -> "ImplicitShape":
        axes = _positive_values(semi_axes, "semi_axes", 3)
        bounds = np.column_stack((-axes, axes))

        def field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            return (x / axes[0]) ** 2 + (y / axes[1]) ** 2 + (z / axes[2]) ** 2 - 1.0

        return cls(field, bounds, "ellipsoid", {"semi_axes_A": axes.tolist()})

    @classmethod
    def cylinder(
        cls,
        radius: float,
        height: float,
        axis: str | Sequence[float] = "z",
    ) -> "ImplicitShape":
        """Return a finite cylinder along a Cartesian axis or 3-vector."""
        radius_value = float(_positive_values([radius], "radius", 1)[0])
        height_value = float(_positive_values([height], "height", 1)[0])
        half_height = height_value / 2.0
        unit_axis, axis_label = _axis_vector(axis)
        # Exact axis-aligned bounding-box half extent of a capped cylinder.
        radial_extent = radius_value * np.sqrt(np.maximum(0.0, 1.0 - unit_axis**2))
        half_extent = half_height * np.abs(unit_axis) + radial_extent
        bounds = np.column_stack((-half_extent, half_extent))

        def field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            axial_position = x * unit_axis[0] + y * unit_axis[1] + z * unit_axis[2]
            radius_squared = np.maximum(
                0.0,
                x * x + y * y + z * z - axial_position * axial_position,
            )
            radial = radius_squared / (radius_value * radius_value) - 1.0
            axial = np.abs(axial_position) / half_height - 1.0
            return np.maximum(radial, axial)

        return cls(
            field,
            bounds,
            f"cylinder_{axis_label}",
            {
                "radius_A": radius_value,
                "height_A": height_value,
                "axis_cartesian": unit_axis.tolist(),
            },
        )

    @classmethod
    def through_cylinder(
        cls,
        radius: float,
        lattice: Sequence[Sequence[float]],
        direction_hkl: Sequence[int],
    ) -> "ImplicitShape":
        """Return one lattice-period-long cylinder along ``direction_hkl``.

        Periodic copies of the returned finite cylinder join end-to-end when
        the corresponding lattice directions are periodic.
        """
        lattice_array = np.asarray(lattice, dtype=float)
        if lattice_array.shape != (3, 3) or not np.isfinite(lattice_array).all():
            raise ValueError("lattice must be a finite 3 x 3 matrix.")
        if abs(float(np.linalg.det(lattice_array))) < 1e-12:
            raise ValueError("lattice must be non-singular.")
        raw_direction = np.asarray(direction_hkl)
        if raw_direction.shape != (3,) or not np.issubdtype(raw_direction.dtype, np.number):
            raise ValueError("direction_hkl must contain three integers.")
        direction_float = raw_direction.astype(float)
        if not np.isfinite(direction_float).all() or not np.allclose(
            direction_float, np.rint(direction_float)
        ):
            raise ValueError("direction_hkl must contain three integers.")
        direction = np.rint(direction_float).astype(int)
        if not np.any(direction):
            raise ValueError("direction_hkl must be non-zero.")
        divisor = math.gcd(*(abs(int(value)) for value in direction))
        direction //= divisor
        axis = direction @ lattice_array
        height = float(np.linalg.norm(axis))
        shape = cls.cylinder(radius, height, axis=axis)
        parameters = dict(shape.parameters)
        parameters["direction_hkl"] = direction.tolist()
        return cls(shape.field, shape.bounds, "through_cylinder", parameters)


# Backward-compatible public name introduced by the nanocluster API.
NanoShape = ImplicitShape


def evaluate_shape_field(shape: ImplicitShape, local_positions: np.ndarray) -> np.ndarray:
    """Evaluate and validate one shape-field batch."""
    positions = np.asarray(local_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("local_positions must have shape (N, 3).")
    values = np.asarray(shape.field(positions[:, 0], positions[:, 1], positions[:, 2]))
    expected_shape = (len(positions),)
    if values.shape != expected_shape:
        raise ValueError(
            "shape field must return one value per input point; "
            f"expected {expected_shape}, got {values.shape}."
        )
    if np.issubdtype(values.dtype, np.bool_):
        raise TypeError("shape field must return a real-valued implicit field, not booleans.")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise TypeError("shape field must return real numeric values.")
    try:
        values = values.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("shape field must return real numeric values.") from exc
    if not np.isfinite(values).all():
        raise ValueError("shape field returned non-finite values.")
    return values


def resolve_shape_center(
    lattice: np.ndarray,
    center: Sequence[float] | None,
    center_frac: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve mutually exclusive Cartesian/fractional shape centers."""
    if center is not None and center_frac is not None:
        raise ValueError("center and center_frac are mutually exclusive.")
    if center_frac is not None:
        fractional = np.asarray(center_frac, dtype=float)
        if fractional.shape != (3,) or not np.isfinite(fractional).all():
            raise ValueError("center_frac must contain three finite fractional values.")
        return fractional @ lattice, fractional.copy()
    if center is None:
        fractional = np.full(3, 0.5, dtype=float)
        return fractional @ lattice, fractional
    cartesian = np.asarray(center, dtype=float)
    if cartesian.shape != (3,) or not np.isfinite(cartesian).all():
        raise ValueError("center must contain three finite Cartesian values.")
    return cartesian, cartesian @ np.linalg.inv(lattice)


def merge_stable_topk(
    best_ids: np.ndarray,
    best_scores: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the smallest deterministic ``(score, id)`` pairs."""
    combined_ids = np.concatenate((best_ids, candidate_ids))
    combined_scores = np.concatenate((best_scores, candidate_scores))
    order = np.lexsort((combined_ids, combined_scores))[:count]
    return combined_ids[order], combined_scores[order]


__all__ = [
    "DEFAULT_SHAPE_BATCH_SIZE",
    "ImplicitShape",
    "NanoShape",
]
