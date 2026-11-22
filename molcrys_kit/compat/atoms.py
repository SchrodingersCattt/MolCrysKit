"""Small ASE Atoms and neighbor-list stand-in for the CIF path."""

from __future__ import annotations

import copy

import numpy as np

_ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
}
_SYMBOLS = {number: symbol for symbol, number in _ATOMIC_NUMBERS.items()}


class Atoms:
    """Private Cartesian carrier used when ASE is not installed.

    This is not an ASE substitute and it is not part of the public API.
    ``CrystalMolecule`` subclasses ``ase.Atoms`` on a desktop install. On a
    core install it subclasses this carrier, which implements only the
    operations the CIF reader, molecule container, and disorder clustering
    call. Selection follows ASE's list, slice, integer-array, and boolean-mask
    rules. A scalar index returns a one-atom carrier rather than an ``Atom``.
    """

    def __init__(
        self,
        symbols=None,
        positions=None,
        scaled_positions=None,
        numbers=None,
        cell=None,
        pbc=False,
        info=None,
    ):
        if _is_atoms(symbols):
            other = symbols
            symbols = list(other.get_chemical_symbols())
            positions = np.array(other.get_positions(), dtype=float, copy=True)
            cell = np.array(other.get_cell(), dtype=float, copy=True)
            pbc = np.array(other.get_pbc(), dtype=bool, copy=True)
            info = copy.deepcopy(getattr(other, "info", {}) or {})
            arrays = {
                key: np.array(value, copy=True)
                for key, value in getattr(other, "arrays", {}).items()
                if key not in {"numbers", "positions"}
            }
        else:
            arrays = {}
            info = {} if info is None else dict(info)
        self.info = info
        if symbols is None and numbers is not None:
            symbols = [_SYMBOLS.get(int(number), "X") for number in numbers]
        symbols = list(symbols or [])
        count = len(symbols)
        self._symbols = list(symbols)
        self._cell = (
            np.zeros((3, 3), dtype=float)
            if cell is None
            else np.asarray(cell, dtype=float).reshape(3, 3).copy()
        )
        if scaled_positions is not None:
            fractional = np.asarray(scaled_positions, dtype=float).reshape(count, 3)
            positions = fractional @ self._cell
        if positions is None:
            positions = np.zeros((count, 3), dtype=float)
        self._positions = np.asarray(positions, dtype=float).reshape(count, 3).copy()
        self._numbers = np.array(
            [_ATOMIC_NUMBERS.get(symbol, 0) for symbol in symbols], dtype=int
        )
        if isinstance(pbc, bool):
            self._pbc = np.array([pbc, pbc, pbc], dtype=bool)
        else:
            values = list(pbc)
            self._pbc = np.array(
                values if len(values) == 3 else [False, False, False], dtype=bool
            )
        self.arrays = {"numbers": self._numbers, "positions": self._positions}
        for key, value in arrays.items():
            self.set_array(key, value)

    def __len__(self) -> int:
        return len(self._symbols)

    def __getitem__(self, index) -> "Atoms":
        chosen = _integer_indices(index, len(self))
        picked = Atoms(
            symbols=[self._symbols[item] for item in chosen],
            positions=self._positions[chosen],
            cell=self._cell,
            pbc=self._pbc,
            info=copy.deepcopy(self.info),
        )
        for key, value in self.arrays.items():
            if key in {"numbers", "positions"}:
                continue
            picked.set_array(key, np.asarray(value)[chosen])
        return picked

    def copy(self) -> "Atoms":
        return Atoms(self)

    def get_chemical_symbols(self) -> list[str]:
        return list(self._symbols)

    def get_masses(self) -> np.ndarray:
        """Per-atom masses in amu; an explicit ``masses`` array takes priority."""
        if "masses" in self.arrays:
            return np.asarray(self.arrays["masses"], dtype=float)
        from ..constants import ATOMIC_MASSES

        return np.array(
            [ATOMIC_MASSES.get(symbol, 0.0) for symbol in self._symbols], dtype=float
        )

    def set_masses(self, masses) -> None:
        self.set_array("masses", np.asarray(masses, dtype=float))

    def get_positions(self) -> np.ndarray:
        return self._positions

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @positions.setter
    def positions(self, value) -> None:
        self._positions[:] = np.asarray(value, dtype=float).reshape(len(self), 3)

    def set_positions(self, value) -> None:
        self.positions = value

    def get_cell(self) -> np.ndarray:
        return self._cell

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    @cell.setter
    def cell(self, value) -> None:
        self._cell = np.asarray(value, dtype=float).reshape(3, 3).copy()

    def set_cell(self, cell) -> None:
        self.cell = cell

    def get_pbc(self) -> np.ndarray:
        return self._pbc

    def get_scaled_positions(self, wrap: bool = True) -> np.ndarray:
        """Return fractional coordinates on the ASE-completed cell.

        Zero cell rows are completed the way ``ase.cell.Cell.complete`` does,
        so an empty cell reports Cartesian coordinates instead of zeros.
        A nonzero singular cell raises ``numpy.linalg.LinAlgError``. Wrapping
        is applied only along periodic axes, twice, matching ASE.
        """
        completed = _complete_cell(self._cell)
        fractional = np.linalg.solve(completed.T, np.transpose(self._positions)).T
        if wrap:
            for axis, periodic in enumerate(self._pbc):
                if periodic:
                    fractional[:, axis] %= 1.0
                    fractional[:, axis] %= 1.0
        return fractional

    def set_array(self, name: str, values) -> None:
        array = np.array(values, copy=True)
        if len(self) and len(array) != len(self):
            raise ValueError(f"{name} must contain one value per atom")
        self.arrays[name] = array

    def get_distance(self, a0, a1, mic: bool = False, vector: bool = False):
        """Distance from atom ``a0`` to atom ``a1``.

        ``mic=True`` uses the minimum image. ``vector=True`` returns the
        Cartesian vector from ``a0`` to ``a1`` instead of its length. This is
        the one pair call ``io/cif.py`` makes for an explicit bond that did
        not carry an image shift.
        """
        return self.get_distances(a0, [a1], mic=mic, vector=vector)[0]

    def get_distances(self, a, indices, mic: bool = False, vector: bool = False):
        """Distances from atom ``a`` to each index in ``indices``."""
        positions = self.get_positions()
        chosen = np.asarray(indices, dtype=int).reshape(-1)
        cell = self.get_cell() if mic else None
        periodic = self.get_pbc() if mic else None
        vectors, lengths = get_distances(
            positions[int(a)], positions[chosen], cell=cell, pbc=periodic
        )
        if vector:
            return np.asarray(vectors, dtype=float).reshape(-1, 3)
        return np.asarray(lengths, dtype=float).reshape(-1)

    def get_chemical_formula(self) -> str:
        counts: dict[str, int] = {}
        for symbol in self._symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        ordered = []
        for symbol in ("C", "H"):
            if symbol in counts:
                ordered.append(symbol)
        ordered.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
        return "".join(
            symbol + ("" if counts[symbol] == 1 else str(counts[symbol]))
            for symbol in ordered
        )


def neighbor_list(
    quantities: str, atoms: Atoms, cutoff, self_interaction: bool = False
):
    """Return the same quantity tuple ASE's ``neighbor_list`` returns."""
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    symbols = atoms.get_chemical_symbols()
    count = len(atoms)
    pair_cutoff = _pair_cutoff(symbols, cutoff)
    max_cutoff = float(np.max(pair_cutoff)) if count else 0.0
    indices_i = []
    indices_j = []
    distances = []
    vectors = []
    shifts = []
    if count and max_cutoff > 0.0:
        # Search every periodic image inside the cutoff, including atoms that
        # already sit several cells away from the origin. The shift is the
        # original-cell image: ``r_j + S @ cell - r_i``.
        for i in range(count):
            for j in range(count):
                limit = float(pair_cutoff[i, j])
                if limit <= 0.0:
                    continue
                delta = positions[j] - positions[i]
                for shift in _translations_within(delta, cell, pbc, limit + 1e-8):
                    if not self_interaction and i == j and not np.any(shift):
                        continue
                    vector = delta + shift @ cell
                    distance = float(np.linalg.norm(vector))
                    if distance < limit and distance > 1e-12:
                        indices_i.append(i)
                        indices_j.append(j)
                        distances.append(distance)
                        vectors.append(vector)
                        shifts.append(shift)
    columns = {
        "i": np.asarray(indices_i, dtype=int),
        "j": np.asarray(indices_j, dtype=int),
        "d": np.asarray(distances, dtype=float),
        "D": np.asarray(vectors, dtype=float).reshape(-1, 3),
        "S": np.asarray(shifts, dtype=int).reshape(-1, 3),
    }
    return tuple(columns[name] for name in quantities)


def get_distances(p1, p2, cell=None, pbc=None):
    """Return ``(vectors, distances)`` with ASE's pair-matrix layout.

    Periodic distances are minimum images on a general cell. A coordinate-wise
    ``rint`` of the fractional difference is not that minimum once the cell is
    skewed, so each Cartesian difference is replaced by the shortest lattice
    image inside a ball that is known to contain it.
    """
    left = np.asarray(p1, dtype=float)
    right = np.asarray(p2, dtype=float)
    if left.ndim == 1:
        left = left.reshape(1, 3)
    if right.ndim == 1:
        right = right.reshape(1, 3)
    delta = right[None, :, :] - left[:, None, :]
    matrix = None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3)
    periodic = _periodic_mask(pbc)
    if matrix is not None and np.any(periodic):
        flat = delta.reshape(-1, 3)
        for index, vector in enumerate(flat):
            flat[index] = _minimum_image_vector(vector, matrix, periodic)
        delta = flat.reshape(delta.shape)
    distances = np.linalg.norm(delta, axis=-1)
    return delta, distances


def _integer_indices(index, count: int) -> np.ndarray:
    """Resolve an ASE-style atom selection to integer positions."""
    if isinstance(index, str):
        raise IndexError("chemical-symbol indexing is not supported")
    if isinstance(index, (bool, np.bool_)):
        raise IndexError("a scalar boolean is not an atom index")
    if isinstance(index, slice):
        return np.arange(count, dtype=int)[index]
    if isinstance(index, (int, np.integer)):
        chosen = int(index)
        if chosen < -count or chosen >= count:
            raise IndexError("Index out of range.")
        return np.asarray([chosen], dtype=int)
    array = np.asarray(index)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, (int, np.integer)) and not isinstance(
            item, (bool, np.bool_)
        ):
            return _integer_indices(int(item), count)
        raise IndexError("atom index must be integers, a slice, or a boolean mask")
    if array.dtype == bool:
        if array.shape != (count,):
            raise IndexError(
                f"Length of mask {array.size} must equal number of atoms {count}"
            )
        return np.flatnonzero(array)
    if array.dtype.kind not in "iu":
        raise IndexError("arrays used as indices must be of integer (or boolean) type")
    chosen = np.asarray(array, dtype=int).reshape(-1)
    if np.any((chosen < -count) | (chosen >= count)):
        raise IndexError("Index out of range.")
    return chosen


def _complete_cell(cell: np.ndarray) -> np.ndarray:
    """Fill zero cell rows the way ``ase.cell.Cell.complete`` does."""
    completed = np.array(cell, dtype=float, copy=True)
    missing = np.nonzero(~completed.any(axis=1))[0]
    if len(missing) == 3:
        completed.flat[::4] = 1.0
    elif len(missing) == 2:
        left, singular, right = np.linalg.svd(completed.T)
        completed = (left @ np.diag([singular[0], 1.0, 1.0]) @ right).T
        if np.sign(np.linalg.det(completed)) < 0.0:
            completed[missing[0]] = -completed[missing[0]]
    elif len(missing) == 1:
        axis = int(missing[0])
        completed[axis] = np.cross(completed[axis - 2], completed[axis - 1])
        completed[axis] /= np.linalg.norm(completed[axis])
    return completed


def _is_atoms(value) -> bool:
    return (
        hasattr(value, "get_positions")
        and hasattr(value, "get_chemical_symbols")
        and not isinstance(value, (str, list, tuple, np.ndarray))
    )


def _pair_cutoff(symbols: list[str], cutoff) -> np.ndarray:
    count = len(symbols)
    result = np.zeros((count, count), dtype=float)
    if isinstance(cutoff, dict):
        for i, left in enumerate(symbols):
            for j, right in enumerate(symbols):
                if (left, right) in cutoff:
                    result[i, j] = float(cutoff[(left, right)])
                elif (right, left) in cutoff:
                    result[i, j] = float(cutoff[(right, left)])
        return result
    result[:] = float(cutoff)
    return result


def _periodic_mask(pbc) -> np.ndarray:
    """Normalize an ASE-style ``pbc`` value to three booleans."""
    if pbc is None:
        return np.zeros(3, dtype=bool)
    if isinstance(pbc, (bool, np.bool_)):
        return np.full(3, bool(pbc), dtype=bool)
    values = np.asarray(pbc, dtype=bool).reshape(-1)
    if values.size == 1:
        return np.full(3, bool(values[0]), dtype=bool)
    if values.size != 3:
        return np.zeros(3, dtype=bool)
    return values.copy()


def _gram_schmidt_rows(basis: np.ndarray):
    """Return ``(mu, orthogonal, norm2)`` for row vectors."""
    count = basis.shape[0]
    mu = np.zeros((count, count), dtype=float)
    orthogonal = np.zeros((count, 3), dtype=float)
    norm2 = np.zeros(count, dtype=float)
    for i in range(count):
        vector = np.array(basis[i], dtype=float, copy=True)
        for j in range(i):
            if norm2[j] <= 1e-24:
                mu[i, j] = 0.0
                continue
            mu[i, j] = float(np.dot(basis[i], orthogonal[j]) / norm2[j])
            vector -= mu[i, j] * orthogonal[j]
        orthogonal[i] = vector
        norm2[i] = float(np.dot(vector, vector))
    return mu, orthogonal, norm2


def _translations_within(delta, cell, pbc, radius: float):
    """Integer shifts ``S`` with ``||delta + S @ cell|| < radius``.

    Only periodic, nonzero lattice rows vary. The search is Fincke–Pohst
    enumeration on the Gram–Schmidt form of those rows: a coefficient outside
    the remaining ball cannot lead to a vector inside it, and every integer
    combination inside the ball is visited. The returned shift is expressed in
    the original cell, not in a reduced basis.
    """
    delta = np.asarray(delta, dtype=float).reshape(3)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    mask = _periodic_mask(pbc)
    if radius <= 0.0 or not np.any(mask):
        if float(np.linalg.norm(delta)) < radius:
            return [np.zeros(3, dtype=int)]
        return []
    active = [
        axis
        for axis in range(3)
        if mask[axis] and float(np.linalg.norm(cell[axis])) > 1e-12
    ]
    if not active:
        if float(np.linalg.norm(delta)) < radius:
            return [np.zeros(3, dtype=int)]
        return []
    basis = cell[np.asarray(active, dtype=int)]
    mu, orthogonal, norm2 = _gram_schmidt_rows(basis)
    sigma = np.zeros(len(active), dtype=float)
    parallel2 = 0.0
    for level, length2 in enumerate(norm2):
        if length2 <= 1e-24:
            continue
        sigma[level] = float(np.dot(delta, orthogonal[level]) / length2)
        parallel2 += sigma[level] ** 2 * length2
    outside2 = max(0.0, float(np.dot(delta, delta)) - parallel2)
    limit2 = float(radius) * float(radius)
    if outside2 >= limit2:
        return []
    found: list[np.ndarray] = []
    coeffs = np.zeros(len(active), dtype=int)

    def walk(level: int, used2: float) -> None:
        if norm2[level] <= 1e-24:
            coeffs[level] = 0
            if level == 0:
                found.append(coeffs.copy())
            else:
                walk(level - 1, used2)
            return
        extra = float(sigma[level])
        for higher in range(level + 1, len(active)):
            extra += float(coeffs[higher]) * float(mu[higher, level])
        budget = limit2 - used2
        if budget < 0.0:
            return
        reach = float(np.sqrt(budget / norm2[level]))
        low = int(np.ceil(-reach - extra - 1e-9))
        high = int(np.floor(reach - extra + 1e-9))
        for coeff in range(low, high + 1):
            coeffs[level] = coeff
            used_next = used2 + (coeff + extra) ** 2 * float(norm2[level])
            if used_next > limit2 + 1e-8:
                continue
            if level == 0:
                found.append(coeffs.copy())
            else:
                walk(level - 1, used_next)

    walk(len(active) - 1, outside2)
    shifts = []
    for coeff in found:
        shift = np.zeros(3, dtype=int)
        for local, axis in enumerate(active):
            shift[axis] = int(coeff[local])
        vector = delta + shift @ cell
        if float(np.dot(vector, vector)) < limit2:
            shifts.append(shift)
    shifts.sort(key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
    return shifts


def _minimum_image_vector(delta, cell, pbc) -> np.ndarray:
    """Shortest Cartesian image of ``delta`` under the periodic lattice.

    The zero shift is one feasible image. Rounding the periodic fractional
    coordinates gives another. The shorter of those two is an upper bound, so
    the exhaustive ball around it contains a true minimum. Equal lengths keep
    the lexicographically smallest original-cell shift.
    """
    delta = np.asarray(delta, dtype=float).reshape(3)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    mask = _periodic_mask(pbc)
    if not np.any(mask):
        return delta.copy()
    best_shift = np.zeros(3, dtype=int)
    best_vector = delta.copy()
    best_length = float(np.dot(best_vector, best_vector))
    if abs(float(np.linalg.det(cell))) > 1e-12:
        fractional = delta @ np.linalg.inv(cell)
        rounded = np.zeros(3, dtype=int)
        rounded[mask] = -np.rint(fractional[mask]).astype(int)
        rounded_vector = delta + rounded @ cell
        rounded_length = float(np.dot(rounded_vector, rounded_vector))
        if rounded_length < best_length - 1e-18:
            best_shift = rounded
            best_vector = rounded_vector
            best_length = rounded_length
    radius = float(np.sqrt(max(best_length, 0.0))) + 1e-8
    for shift in _translations_within(delta, cell, mask, radius):
        vector = delta + shift @ cell
        length = float(np.dot(vector, vector))
        if length < best_length - 1e-18 or (
            abs(length - best_length) <= 1e-18
            and tuple(int(value) for value in shift)
            < tuple(int(value) for value in best_shift)
        ):
            best_shift = shift
            best_vector = vector
            best_length = length
    return best_vector
