"""Bounded vectorized implicit shapes shared by crystal carving operations.

An :class:`ImplicitShape` is purely geometric: ``field(x, y, z) <= 0``
denotes the interior and ``bounds`` supplies a finite Cartesian bounding box.
Material operations decide whether the interior is retained (nanoclusters) or
removed (voids); the shape itself has no topology or output semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.spatial import HalfspaceIntersection, QhullError

from ..constants.config import BFDH_GEOMETRY_TOLERANCE
from ..structures.crystal import MolecularCrystal
from ..structures.symmetry import CrystalSymmetry

from ..constants.config import DEFAULT_SHAPE_BATCH_SIZE

ShapeField = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


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


def _bfdh_lattice(
    crystal_or_lattice_or_structure: MolecularCrystal
    | Lattice
    | Structure
    | np.ndarray
    | Sequence[Sequence[float]],
) -> Lattice:
    """Return the pymatgen lattice used to orient BFDH plane normals."""

    obj = crystal_or_lattice_or_structure
    if isinstance(obj, Structure):
        return obj.lattice
    if isinstance(obj, Lattice):
        return obj
    if isinstance(obj, MolecularCrystal):
        return Lattice(obj.lattice)
    array = np.asarray(obj, dtype=float)
    if array.shape != (3, 3):
        raise TypeError(
            "Expected a MolecularCrystal, pymatgen Lattice/Structure, or a "
            "3x3 lattice matrix."
        )
    return Lattice(array)


def _validated_millers(
    miller_indices: Iterable[Sequence[int]] | None,
) -> tuple[tuple[int, int, int], ...] | None:
    """Materialize and validate optional explicit Miller indices."""

    if miller_indices is None:
        return None
    validated: list[tuple[int, int, int]] = []
    for raw in miller_indices:
        array = np.asarray(raw)
        if array.shape != (3,) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("Each Miller index must contain three integers.")
        values = array.astype(float)
        if not np.isfinite(values).all() or not np.allclose(values, np.rint(values)):
            raise ValueError("Each Miller index must contain three integers.")
        hkl = tuple(int(value) for value in np.rint(values))
        if hkl == (0, 0, 0):
            raise ValueError("Miller indices cannot all be zero.")
        validated.append(hkl)
    return tuple(validated)


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
    def bfdh(
        cls,
        crystal_or_lattice_or_structure: MolecularCrystal
        | Lattice
        | Structure
        | np.ndarray
        | Sequence[Sequence[float]],
        max_dimension: float,
        *,
        max_index: int = 2,
        miller_indices: Iterable[Sequence[int]] | None = None,
        symmetry: CrystalSymmetry | None = None,
        extinction_filter: bool = True,
        symprec: float = 1e-5,
    ) -> "ImplicitShape":
        """Return a bounded pure-BFDH growth morphology.

        Plane-center distances are proportional to the BFDH growth-rate
        proxy ``1 / d_hkl``.  The resulting half-space intersection is scaled
        so its largest Cartesian bounding-box span equals ``max_dimension``.
        Explicit ``symmetry`` should describe the parent crystal when the
        coordinates have been disorder-resolved into a lower-symmetry model.
        """

        # Reuse the canonical facet ranking; this factory only converts the
        # ranked crystallographic planes into bounded implicit geometry.
        from ..analysis.bfdh import enumerate_bfdh_facets

        max_dimension_value = float(
            _positive_values([max_dimension], "max_dimension", 1)[0]
        )
        explicit_millers = _validated_millers(miller_indices)
        if symmetry is not None and not isinstance(symmetry, CrystalSymmetry):
            raise TypeError("symmetry must be a CrystalSymmetry.")

        lattice = _bfdh_lattice(crystal_or_lattice_or_structure)
        facets = enumerate_bfdh_facets(
            lattice,
            max_index=max_index,
            miller_indices=explicit_millers,
            symprec=symprec,
            include_equivalents=True,
            include_negative=False,
            extinction_filter=extinction_filter,
            symmetry=symmetry,
        )
        if not facets:
            raise ValueError("BFDH enumeration produced no allowed facets.")

        reciprocal = lattice.reciprocal_lattice_crystallographic
        plane_map: dict[
            tuple[int, int, int],
            tuple[float, tuple[int, int, int], np.ndarray],
        ] = {}
        tolerance = BFDH_GEOMETRY_TOLERANCE
        for facet in facets:
            family = facet.equivalent_millers or (facet.miller_index,)
            for hkl in family:
                reciprocal_vector = np.asarray(
                    reciprocal.get_cartesian_coords(hkl), dtype=float
                )
                unit_normal = reciprocal_vector / np.linalg.norm(reciprocal_vector)
                for sign in (1, -1):
                    normal = sign * unit_normal
                    signed_hkl = tuple(sign * int(value) for value in hkl)
                    key = tuple(np.rint(normal / tolerance).astype(int))
                    candidate = (
                        float(facet.relative_growth_rate),
                        signed_hkl,
                        normal,
                    )
                    current = plane_map.get(key)
                    if current is None or (
                        candidate[0],
                        sum(abs(value) for value in candidate[1]),
                        candidate[1],
                    ) < (
                        current[0],
                        sum(abs(value) for value in current[1]),
                        current[1],
                    ):
                        plane_map[key] = candidate

        plane_entries = sorted(
            plane_map.values(),
            key=lambda item: (item[1], item[0]),
        )
        unit_distances = np.asarray([item[0] for item in plane_entries], dtype=float)
        normals = np.asarray([item[2] for item in plane_entries], dtype=float)
        if np.linalg.matrix_rank(normals, tol=tolerance) < 3:
            raise ValueError(
                "BFDH facets do not enclose a finite 3-D shape; provide at "
                "least three independent facet families."
            )

        halfspaces = np.column_stack((normals, -unit_distances))
        try:
            raw_vertices = HalfspaceIntersection(
                halfspaces, np.zeros(3, dtype=float)
            ).intersections
        except QhullError as exc:
            raise ValueError(
                "BFDH facets do not enclose a finite 3-D shape."
            ) from exc
        if len(raw_vertices) == 0:  # pragma: no cover - guarded by Qhull
            raise ValueError("BFDH half-space intersection has no vertices.")

        ordered_vertices = raw_vertices[
            np.lexsort((raw_vertices[:, 2], raw_vertices[:, 1], raw_vertices[:, 0]))
        ]
        unique_vertices: list[np.ndarray] = []
        for vertex in ordered_vertices:
            if not unique_vertices or not np.allclose(
                vertex, unique_vertices[-1], atol=tolerance, rtol=0.0
            ):
                unique_vertices.append(vertex)
        unit_vertices = np.asarray(unique_vertices, dtype=float)
        spans = np.ptp(unit_vertices, axis=0)
        largest_span = float(np.max(spans))
        if largest_span <= tolerance:  # pragma: no cover - rank check guards this
            raise ValueError("BFDH half-space intersection is degenerate.")

        scale = max_dimension_value / largest_span
        distances = unit_distances * scale
        vertices = unit_vertices * scale
        bounds = np.column_stack((vertices.min(axis=0), vertices.max(axis=0)))

        def field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            values = np.full(np.asarray(x).shape, -np.inf, dtype=float)
            for normal, distance in zip(normals, distances):
                plane_value = (
                    normal[0] * x + normal[1] * y + normal[2] * z
                ) / distance
                np.maximum(values, plane_value, out=values)
            return values - 1.0

        plane_records = [
            {
                "miller_index": list(entry[1]),
                "normal_cartesian": entry[2].tolist(),
                "distance_A": float(distance),
            }
            for entry, distance in zip(plane_entries, distances)
        ]
        if symmetry is not None:
            symmetry_record: dict[str, Any] = {
                "kind": "explicit_parent",
                "source": symmetry.source,
                "space_group_number": symmetry.space_group_number,
                "space_group_symbol": symmetry.space_group_symbol,
                "hall_symbol": symmetry.hall_symbol,
            }
        else:
            symmetry_record = {"kind": "lattice_metric"}

        return cls(
            field,
            bounds,
            "bfdh",
            {
                "max_dimension_A": max_dimension_value,
                "max_index": int(max_index),
                "miller_indices": (
                    [list(hkl) for hkl in explicit_millers]
                    if explicit_millers is not None
                    else None
                ),
                "extinction_filter": bool(extinction_filter),
                "symprec": float(symprec),
                "symmetry": symmetry_record,
                "facet_families": [facet.as_dict() for facet in facets],
                "planes": plane_records,
                "vertices_A": vertices.tolist(),
            },
        )

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
        first_nonzero = int(direction[np.flatnonzero(direction)[0]])
        if first_nonzero < 0:
            direction *= -1
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
