"""Crystallographic affine operations, basis changes, and finite cosets.

MolCrysKit stores lattice vectors as rows and fractional coordinates as row
vectors.  A crystallographic operation supplied in conventional column form
``f' = W f + w`` therefore acts on arrays as ``f' = f @ W.T + w``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..constants.symmetry_path import (
    AFFINE_DETERMINANT_TOLERANCE,
    AFFINE_EQUIVALENCE_TOLERANCE,
    AFFINE_INTEGER_TOLERANCE,
    AFFINE_METRIC_TOLERANCE,
    AFFINE_ORTHOGONALITY_TOLERANCE,
    BASIS_UNIMODULAR_TOLERANCE,
)
from ..utils.geometry import apply_fractional_affine, fractional_linear_to_cartesian


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class FractionalAffineOperation:
    """A fractional affine crystallographic operation.

    ``rotation`` follows the conventional column-vector representation.  Use
    :meth:`apply` for MolCrysKit row-vector coordinates.
    """

    rotation: np.ndarray
    translation: np.ndarray
    xyz: str | None = None
    index: int | None = None
    source: str | None = None
    space_group_number: int | None = None

    def __post_init__(self) -> None:
        rotation = _array(self.rotation, (3, 3), "rotation")
        translation = _array(self.translation, (3,), "translation")
        rounded = np.rint(rotation)
        if not np.allclose(rotation, rounded, atol=AFFINE_INTEGER_TOLERANCE, rtol=0.0):
            raise ValueError("crystallographic rotation must be integer-valued")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(
            abs(determinant), 1.0, atol=AFFINE_DETERMINANT_TOLERANCE, rtol=0.0
        ):
            raise ValueError("crystallographic rotation determinant must be +1 or -1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, fractional: np.ndarray, *, wrap: bool = False) -> np.ndarray:
        """Apply the operation to one ``(3,)`` or many ``(..., 3)`` rows."""
        coordinates = np.asarray(fractional, dtype=float)
        if coordinates.shape[-1:] != (3,):
            raise ValueError("fractional coordinates must end with dimension 3")
        return apply_fractional_affine(
            coordinates, self.rotation, self.translation, wrap=wrap
        )

    def inverse(self) -> "FractionalAffineOperation":
        inverse_rotation = np.linalg.inv(self.rotation)
        inverse_translation = -inverse_rotation @ self.translation
        return FractionalAffineOperation(
            inverse_rotation,
            inverse_translation,
            source=self.source,
            space_group_number=self.space_group_number,
        )

    def compose(
        self, other: "FractionalAffineOperation"
    ) -> "FractionalAffineOperation":
        """Return ``self ∘ other`` (apply ``other``, then ``self``)."""
        rotation = self.rotation @ other.rotation
        translation = self.rotation @ other.translation + self.translation
        return FractionalAffineOperation(
            rotation,
            translation,
            source=self.source or other.source,
            space_group_number=self.space_group_number,
        )

    def canonical_translation(self) -> np.ndarray:
        result = self.translation - np.floor(
            self.translation + AFFINE_EQUIVALENCE_TOLERANCE
        )
        result[np.isclose(result, 1.0, atol=AFFINE_EQUIVALENCE_TOLERANCE)] = 0.0
        return result

    def equivalent_to(
        self,
        other: "FractionalAffineOperation",
        *,
        tolerance: float = AFFINE_EQUIVALENCE_TOLERANCE,
    ) -> bool:
        if not np.allclose(self.rotation, other.rotation, atol=tolerance, rtol=0.0):
            return False
        delta = self.translation - other.translation
        return bool(np.allclose(delta, np.rint(delta), atol=tolerance, rtol=0.0))

    def cartesian_linear(self, lattice: np.ndarray) -> np.ndarray:
        """Return the Cartesian column-form linear operation for row lattice."""
        lattice_array = _array(lattice, (3, 3), "lattice")
        operation = fractional_linear_to_cartesian(self.rotation, lattice_array)
        if not np.allclose(
            operation.T @ operation,
            np.eye(3),
            atol=AFFINE_ORTHOGONALITY_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("operation does not preserve the lattice metric")
        return operation

    def validate_metric(self, lattice: np.ndarray) -> None:
        lattice_array = _array(lattice, (3, 3), "lattice")
        metric = lattice_array @ lattice_array.T
        transformed = self.rotation.T @ metric @ self.rotation
        if not np.allclose(transformed, metric, atol=AFFINE_METRIC_TOLERANCE, rtol=0.0):
            raise ValueError("operation does not preserve the lattice metric")
        self.cartesian_linear(lattice_array)

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.rotation))

    @property
    def is_proper(self) -> bool:
        return self.determinant > 0.0

    @property
    def is_improper(self) -> bool:
        return self.determinant < 0.0


@dataclass(frozen=True, eq=False)
class LatticeBasisChange:
    """Unimodular row-lattice basis change ``L_new = B @ L_old``."""

    matrix: np.ndarray
    origin_shift: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        matrix = _array(self.matrix, (3, 3), "matrix")
        origin_shift = _array(self.origin_shift, (3,), "origin_shift")
        rounded = np.rint(matrix)
        if not np.allclose(matrix, rounded, atol=AFFINE_INTEGER_TOLERANCE, rtol=0.0):
            raise ValueError("basis-change matrix must be integer-valued")
        determinant = float(np.linalg.det(matrix))
        if not np.isclose(
            abs(determinant), 1.0, atol=BASIS_UNIMODULAR_TOLERANCE, rtol=0.0
        ):
            raise ValueError(
                "only unimodular basis changes are supported; supercells require "
                "explicit coset replication"
            )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "origin_shift", origin_shift)

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.matrix))

    @property
    def changes_handedness(self) -> bool:
        return self.determinant < 0.0

    def transform_lattice(self, lattice: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(lattice, dtype=float)

    def old_to_new_fractional(self, fractional: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(fractional, dtype=float)
        return (coordinates - self.origin_shift) @ np.linalg.inv(self.matrix)

    def new_to_old_fractional(self, fractional: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(fractional, dtype=float)
        return coordinates @ self.matrix + self.origin_shift

    def transform_operation(
        self, operation: FractionalAffineOperation
    ) -> FractionalAffineOperation:
        """Conjugate ``operation`` into this basis and origin convention."""
        # Column coordinates satisfy f_new_col = B^{-T}(f_old_col-o_col).
        inverse_transpose = np.linalg.inv(self.matrix.T)
        rotation = inverse_transpose @ operation.rotation @ self.matrix.T
        translation = inverse_transpose @ (
            operation.rotation @ self.origin_shift
            + operation.translation
            - self.origin_shift
        )
        return FractionalAffineOperation(
            rotation,
            translation,
            xyz=operation.xyz,
            index=operation.index,
            source=operation.source,
            space_group_number=operation.space_group_number,
        )


@dataclass(frozen=True)
class CrystalSymmetry:
    """Canonical symmetry metadata and its finite affine operation set."""

    operations: tuple[FractionalAffineOperation, ...]
    space_group_number: int | None = None
    space_group_symbol: str | None = None
    hall_symbol: str | None = None
    source: str = "explicit"
    expanded_from_declaration: bool = False

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if not operations:
            raise ValueError("CrystalSymmetry requires at least one operation")
        object.__setattr__(self, "operations", operations)


def identity_operation() -> FractionalAffineOperation:
    return FractionalAffineOperation(np.eye(3), np.zeros(3), xyz="x,y,z")


def _find_equivalent(
    operation: FractionalAffineOperation,
    operations: Iterable[FractionalAffineOperation],
) -> FractionalAffineOperation | None:
    return next((item for item in operations if operation.equivalent_to(item)), None)


def validate_subgroup(
    group: Iterable[FractionalAffineOperation],
    subgroup: Iterable[FractionalAffineOperation],
) -> tuple[FractionalAffineOperation, ...]:
    """Validate and return a finite affine subgroup modulo translations."""
    full = tuple(group)
    subset = tuple(subgroup)
    if not full or not subset:
        raise ValueError("group and subgroup must be non-empty")
    if _find_equivalent(identity_operation(), subset) is None:
        raise ValueError("subgroup does not contain the identity")
    for item in subset:
        if _find_equivalent(item, full) is None:
            raise ValueError("subgroup contains an operation outside the group")
        if _find_equivalent(item.inverse(), subset) is None:
            raise ValueError("subgroup is not closed under inverse")
    for left in subset:
        for right in subset:
            if _find_equivalent(left.compose(right), subset) is None:
                raise ValueError("subgroup is not closed under composition")
    return subset


def left_cosets(
    group: Iterable[FractionalAffineOperation],
    subgroup: Iterable[FractionalAffineOperation],
) -> tuple[tuple[FractionalAffineOperation, ...], ...]:
    """Enumerate disjoint left cosets of a finite affine subgroup."""
    full = tuple(group)
    subset = validate_subgroup(full, subgroup)
    remaining = list(full)
    cosets = []
    while remaining:
        representative = remaining[0]
        coset = []
        for member in subset:
            product = representative.compose(member)
            canonical = _find_equivalent(product, full)
            if canonical is None:
                raise ValueError("group is not closed under composition")
            if _find_equivalent(canonical, coset) is None:
                coset.append(canonical)
        if len(coset) != len(subset):
            raise ValueError("coset has unexpected cardinality")
        cosets.append(tuple(coset))
        remaining = [
            item for item in remaining if _find_equivalent(item, coset) is None
        ]
    if sum(len(coset) for coset in cosets) != len(full):
        raise ValueError("cosets do not partition the group")
    return tuple(cosets)


def domain_representatives(
    group: Iterable[FractionalAffineOperation],
    subgroup: Iterable[FractionalAffineOperation],
) -> tuple[FractionalAffineOperation, ...]:
    """Return one deterministic representative from each left coset."""
    return tuple(coset[0] for coset in left_cosets(group, subgroup))
