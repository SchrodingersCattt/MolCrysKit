"""Exact integer lattice helpers for periodic quotient graphs.

Image shifts are row vectors. A shift ``s`` displaces a point by ``s @ L``
when the lattice rows are the cell vectors. These helpers never reduce a
generator such as ``(2, 0, 0)`` to a primitive vector; Hermite reduction of a
whole generating set lives in :func:`molcrys_kit.chemistry.topology.integer_translation_basis`.
"""

from __future__ import annotations

import operator
from typing import Sequence

ImageShift = tuple[int, int, int]
Matrix3 = tuple[ImageShift, ImageShift, ImageShift]


def exact_int(value: object) -> int:
    """Return ``value`` when it is an exact integer and not a Boolean."""
    if isinstance(value, bool):
        raise TypeError("translation components must be exact integers, got bool")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"translation components must be exact integers, got {type(value).__name__}"
        ) from exc


def exact_shift(value: Sequence[object], name: str = "image shift") -> ImageShift:
    """Return a length-3 exact integer vector."""
    try:
        components = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be three exact integers") from exc
    if len(components) != 3:
        raise ValueError(f"{name} must be three exact integers")
    try:
        return (
            exact_int(components[0]),
            exact_int(components[1]),
            exact_int(components[2]),
        )
    except TypeError as exc:
        raise TypeError(f"{name} must be three exact integers") from exc


def add(left: ImageShift, right: ImageShift) -> ImageShift:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub(left: ImageShift, right: ImageShift) -> ImageShift:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def neg(shift: ImageShift) -> ImageShift:
    return (-shift[0], -shift[1], -shift[2])


def canonical_shift(shift: ImageShift) -> ImageShift:
    """Flip ``shift`` so its first nonzero component is positive.

    This chooses an orientation. It does not divide out a common factor.
    """
    first = next((component for component in shift if component), 0)
    if first < 0:
        return neg(shift)
    return shift


def dot(left: ImageShift, right: ImageShift) -> int:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def cross(left: ImageShift, right: ImageShift) -> ImageShift:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def det3(matrix: Matrix3) -> int:
    first, second, third = matrix
    return dot(first, cross(second, third))


def matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def vecmul(vector: ImageShift, matrix: Matrix3) -> ImageShift:
    """Return the row-vector product ``vector @ matrix``."""
    return (
        dot(vector, (matrix[0][0], matrix[1][0], matrix[2][0])),
        dot(vector, (matrix[0][1], matrix[1][1], matrix[2][1])),
        dot(vector, (matrix[0][2], matrix[1][2], matrix[2][2])),
    )


def adjugate(matrix: Matrix3) -> Matrix3:
    """Return the classical adjugate, with ``matrix @ adjugate(matrix) == det I``."""
    a, b, c = matrix
    c00 = b[1] * c[2] - b[2] * c[1]
    c01 = -(b[0] * c[2] - b[2] * c[0])
    c02 = b[0] * c[1] - b[1] * c[0]
    c10 = -(a[1] * c[2] - a[2] * c[1])
    c11 = a[0] * c[2] - a[2] * c[0]
    c12 = -(a[0] * c[1] - a[1] * c[0])
    c20 = a[1] * b[2] - a[2] * b[1]
    c21 = -(a[0] * b[2] - a[2] * b[0])
    c22 = a[0] * b[1] - a[1] * b[0]
    return (
        (c00, c10, c20),
        (c01, c11, c21),
        (c02, c12, c22),
    )


def floor_div(numer: int, denom: int) -> int:
    """Integer floor division, including a negative denominator."""
    if denom == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return numer // denom


def det_square(rows: Sequence[Sequence[int]]) -> int:
    """Exact determinant of a 0×0, 1×1, 2×2, or 3×3 integer matrix."""
    size = len(rows)
    if size == 0:
        return 1
    if any(len(row) != size for row in rows):
        raise ValueError("determinant expects a square integer matrix")
    if size == 1:
        return int(rows[0][0])
    if size == 2:
        return int(rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0])
    if size == 3:
        return det3(
            (
                (int(rows[0][0]), int(rows[0][1]), int(rows[0][2])),
                (int(rows[1][0]), int(rows[1][1]), int(rows[1][2])),
                (int(rows[2][0]), int(rows[2][1]), int(rows[2][2])),
            )
        )
    raise ValueError("determinant is implemented for dimension at most three")


def undirected_shift(
    source: str, target: str, shift: ImageShift
) -> tuple[str, str, ImageShift]:
    """Return endpoints in node-id order and the matching image shift.

    A self-image shift is oriented so its first nonzero component is positive.
    ``(2, 0, 0)`` stays ``(2, 0, 0)``.
    """
    if source == target:
        return source, target, canonical_shift(shift)
    if target < source:
        return target, source, neg(shift)
    return source, target, shift
