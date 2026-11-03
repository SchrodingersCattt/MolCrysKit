"""Exact arithmetic for periodic-net comparison.

Image shifts stay integers. Barycentric coordinates and Gram–Schmidt
coefficients are rational. Nothing in this module reads Cartesian geometry.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from fractions import Fraction

from molcrys_kit.periodic._lattice import ImageShift, Matrix3, adjugate, det3, vecmul

F3 = tuple[Fraction, Fraction, Fraction]


def fraction_vector(values: tuple[int, int, int] | F3) -> F3:
    return (Fraction(values[0]), Fraction(values[1]), Fraction(values[2]))


def add_f(left: F3, right: F3) -> F3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub_f(left: F3, right: F3) -> F3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def scale_f(value: F3, factor: Fraction) -> F3:
    return (value[0] * factor, value[1] * factor, value[2] * factor)


def mod1(value: Fraction) -> Fraction:
    """Return ``value`` in ``[0, 1)``."""
    return value - Fraction(math.floor(value))


def mod1_vector(value: F3) -> F3:
    return (mod1(value[0]), mod1(value[1]), mod1(value[2]))


def nearest_int(value: Fraction) -> int:
    """Nearest integer, with halves rounded away from zero."""
    if value >= 0:
        return int(value + Fraction(1, 2))
    return -int(-value + Fraction(1, 2))


def integer_matrix_inverse(matrix: Matrix3) -> tuple[F3, F3, F3]:
    """Return the inverse of an invertible integer 3×3 matrix."""
    determinant = det3(matrix)
    if determinant == 0:
        raise ValueError("matrix is singular")
    adjunct = adjugate(matrix)
    scale = Fraction(1, determinant)
    return tuple(  # type: ignore[return-value]
        tuple(Fraction(entry) * scale for entry in row) for row in adjunct
    )


def matvec_f(vector: tuple[int, int, int] | F3, matrix: tuple[F3, F3, F3]) -> F3:
    """Return the row-vector product ``vector @ matrix`` over rationals."""
    left = fraction_vector(vector)
    result = [Fraction(0), Fraction(0), Fraction(0)]
    for column in range(3):
        result[column] = (
            left[0] * matrix[0][column]
            + left[1] * matrix[1][column]
            + left[2] * matrix[2][column]
        )
    return (result[0], result[1], result[2])


def as_integer_shift(value: F3, name: str) -> ImageShift:
    if any(component.denominator != 1 for component in value):
        raise ValueError(f"{name} is not an integer vector")
    return (int(value[0]), int(value[1]), int(value[2]))


def solve_pinned_system(
    matrix: list[list[int]],
    rhs: list[list[int]],
    pin: int,
) -> list[list[Fraction]] | str:
    """Solve ``matrix @ x = rhs`` after replacing row ``pin`` with a unit pin.

    The result is a list of rational coordinate triples, ``"inconsistent"``,
    or ``"nonunique"``.
    """
    size = len(matrix)
    width = size + 3
    rows = [
        [Fraction(matrix[row][column]) for column in range(size)]
        + [Fraction(rhs[row][column]) for column in range(3)]
        for row in range(size)
    ]
    rows[pin] = [Fraction(0)] * width
    rows[pin][pin] = Fraction(1)
    pivot_of = [-1] * size
    lead = 0
    for column in range(size):
        pivot = next(
            (row for row in range(lead, size) if rows[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        rows[lead], rows[pivot] = rows[pivot], rows[lead]
        divisor = rows[lead][column]
        for index in range(column, width):
            rows[lead][index] /= divisor
        for row in range(size):
            if row == lead or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            for index in range(column, width):
                rows[row][index] -= factor * rows[lead][index]
        pivot_of[column] = lead
        lead += 1
    for row in rows:
        if all(entry == 0 for entry in row[:size]) and any(row[size:]):
            return "inconsistent"
    if any(pivot < 0 for pivot in pivot_of):
        return "nonunique"
    solution = [[Fraction(0), Fraction(0), Fraction(0)] for _ in range(size)]
    for column, row in enumerate(pivot_of):
        solution[column] = [rows[row][size], rows[row][size + 1], rows[row][size + 2]]
    return solution


def quadratic_form(shifts: tuple[ImageShift, ...]) -> tuple[tuple[int, int, int], ...]:
    """Return ``sum_s s^T s`` as a 3×3 integer Gram matrix."""
    form = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for shift in shifts:
        for left in range(3):
            for right in range(3):
                form[left][right] += shift[left] * shift[right]
    return (
        (form[0][0], form[0][1], form[0][2]),
        (form[1][0], form[1][1], form[1][2]),
        (form[2][0], form[2][1], form[2][2]),
    )


def form_dot(
    left: tuple[Fraction, ...] | list[Fraction] | tuple[int, ...] | list[int],
    right: tuple[Fraction, ...] | list[Fraction] | tuple[int, ...] | list[int],
    form: tuple[tuple[int, int, int], ...],
) -> Fraction:
    total = Fraction(0)
    for row in range(3):
        for column in range(3):
            total += (
                Fraction(left[row])
                * Fraction(form[row][column])
                * Fraction(right[column])
            )
    return total


def lll_unimodular(
    form: tuple[tuple[int, int, int], ...],
    *,
    step_limit: int,
) -> Matrix3 | None:
    """Return one LLL-reduced basis of ``Z^3`` for ``form``, or ``None``.

    The basis rows are the new coordinate vectors in the old basis, with
    determinant ±1. A singular form is allowed: null vectors stay in the
    basis and are not inverted. The search stops after ``step_limit`` swaps
    rather than returning a partial basis as if it were reduced.
    """
    basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    delta = Fraction(3, 4)

    def schmidt() -> tuple[list[list[Fraction]], list[Fraction]]:
        mu = [[Fraction(0) for _ in range(3)] for _ in range(3)]
        dual: list[list[Fraction]] = []
        norms: list[Fraction] = []
        for index in range(3):
            current = [Fraction(entry) for entry in basis[index]]
            for earlier in range(index):
                if norms[earlier] == 0:
                    mu[index][earlier] = Fraction(0)
                    continue
                mu[index][earlier] = (
                    form_dot(basis[index], dual[earlier], form) / norms[earlier]
                )
                current = [
                    current[axis] - mu[index][earlier] * dual[earlier][axis]
                    for axis in range(3)
                ]
            dual.append(current)
            norms.append(form_dot(current, current, form))
        return mu, norms

    index = 1
    steps = 0
    while index < 3:
        steps += 1
        if steps > step_limit:
            return None
        mu, norms = schmidt()
        for earlier in range(index - 1, -1, -1):
            if norms[earlier] == 0:
                continue
            quotient = nearest_int(mu[index][earlier])
            if quotient:
                basis[index] = [
                    basis[index][axis] - quotient * basis[earlier][axis]
                    for axis in range(3)
                ]
        mu, norms = schmidt()
        if norms[index - 1] == 0:
            if norms[index] == 0:
                index += 1
                continue
            basis[index], basis[index - 1] = basis[index - 1], basis[index]
            index = max(index - 1, 1)
            continue
        # A zero Gram–Schmidt norm is a radical vector of a semidefinite
        # form. It is already size-reduced, so it must not swap forever
        # with a positive-norm predecessor.
        if norms[index] == 0:
            index += 1
            continue
        lovasz = norms[index] >= (delta - mu[index][index - 1] ** 2) * norms[index - 1]
        if lovasz:
            index += 1
        else:
            basis[index], basis[index - 1] = basis[index - 1], basis[index]
            index = max(index - 1, 1)
    matrix = (
        (basis[0][0], basis[0][1], basis[0][2]),
        (basis[1][0], basis[1][1], basis[1][2]),
        (basis[2][0], basis[2][1], basis[2][2]),
    )
    if det3(matrix) not in (1, -1):
        return None
    if any(not any(row) for row in matrix):
        return None
    return matrix


@lru_cache(maxsize=4)
def unimodular_matrices(entry_limit: int) -> tuple[Matrix3, ...]:
    """Return every integer 3×3 matrix with entries in range and determinant ±1."""
    if entry_limit < 1:
        return ()
    span = range(-entry_limit, entry_limit + 1)
    found: list[Matrix3] = []
    for raw in itertools.product(span, repeat=9):
        matrix = (
            (raw[0], raw[1], raw[2]),
            (raw[3], raw[4], raw[5]),
            (raw[6], raw[7], raw[8]),
        )
        if det3(matrix) in (1, -1):
            found.append(matrix)
    return tuple(found)


def apply_integer_change(shift: ImageShift, inverse: Matrix3) -> ImageShift:
    """Return ``shift @ inverse`` for a unimodular inverse."""
    return vecmul(shift, inverse)


def _gcdex(left: int, right: int) -> tuple[int, int, int]:
    """Return ``(x, y, g)`` with ``x * left + y * right = g`` and ``g > 0``.

    When ``left`` divides ``right``, ``y`` is 0 and ``x`` is the sign of
    ``left``. Hermite reduction relies on that extra normalization.
    """
    old_r, new_r = left, right
    old_s, new_s = 1, 0
    old_t, new_t = 0, 1
    while new_r:
        quotient = old_r // new_r
        old_r, new_r = new_r, old_r - quotient * new_r
        old_s, new_s = new_s, old_s - quotient * new_s
        old_t, new_t = new_t, old_t - quotient * new_t
    gcd, coeff_left, coeff_right = old_r, old_s, old_t
    if gcd < 0:
        gcd, coeff_left, coeff_right = -gcd, -coeff_left, -coeff_right
    if left != 0 and right % left == 0:
        coeff_right = 0
        coeff_left = -1 if left < 0 else 1
    return coeff_left, coeff_right, gcd


def _combine_columns(
    matrix: list[list[int]],
    left: int,
    right: int,
    coeff_ll: int,
    coeff_lr: int,
    coeff_rl: int,
    coeff_rr: int,
) -> None:
    """Replace two columns by an integer combination with determinant ±1."""
    for row in matrix:
        saved = row[left]
        row[left] = coeff_ll * saved + coeff_lr * row[right]
        row[right] = coeff_rl * saved + coeff_rr * row[right]


def hermite_form_shifts(
    shifts: tuple[ImageShift, ...] | list[ImageShift],
) -> tuple[ImageShift, ...]:
    """Return the row-aligned Hermite form of shift rows under ``GL(3, Z)``.

    Column operations only. Row ``i`` of the result is the image of input row
    ``i``, so each row stays attached to the edge that produced it. Pivot
    columns sit on the right. A zero row stays a zero row. This is the unique
    form used as the canonical voltage representative; it is not recomputed
    after a further unimodular change.
    """
    matrix = [list(shift) for shift in shifts]
    rows = len(matrix)
    if rows == 0:
        return ()
    pivot_column = 3
    for row_index in range(rows - 1, -1, -1):
        if pivot_column == 0:
            break
        pivot_column -= 1
        for column in range(pivot_column - 1, -1, -1):
            if matrix[row_index][column] == 0:
                continue
            coeff_left, coeff_right, gcd = _gcdex(
                matrix[row_index][pivot_column], matrix[row_index][column]
            )
            quotient_left = matrix[row_index][pivot_column] // gcd
            quotient_right = matrix[row_index][column] // gcd
            _combine_columns(
                matrix,
                pivot_column,
                column,
                coeff_left,
                coeff_right,
                -quotient_right,
                quotient_left,
            )
        pivot = matrix[row_index][pivot_column]
        if pivot < 0:
            _combine_columns(matrix, pivot_column, pivot_column, -1, 0, -1, 0)
            pivot = -pivot
        if pivot == 0:
            pivot_column += 1
            continue
        for column in range(pivot_column + 1, 3):
            quotient = matrix[row_index][column] // pivot
            _combine_columns(matrix, column, pivot_column, 1, -quotient, 0, 1)
    return tuple((row[0], row[1], row[2]) for row in matrix)


__all__ = [
    "F3",
    "add_f",
    "apply_integer_change",
    "as_integer_shift",
    "form_dot",
    "fraction_vector",
    "hermite_form_shifts",
    "integer_matrix_inverse",
    "lll_unimodular",
    "matvec_f",
    "mod1_vector",
    "quadratic_form",
    "scale_f",
    "solve_pinned_system",
    "sub_f",
    "unimodular_matrices",
]
