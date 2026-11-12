"""Small ASE Atoms and neighbor-list stand-in for the CIF path."""

from __future__ import annotations

import copy
from itertools import product

import numpy as np

_ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41,
    "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81,
    "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89,
    "Th": 90, "Pa": 91, "U": 92,
}
_SYMBOLS = {number: symbol for symbol, number in _ATOMIC_NUMBERS.items()}


class Atoms:
    """The slice of ``ase.Atoms`` used while reading a CIF."""

    def __init__(
        self,
        symbols=None,
        positions=None,
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
        if positions is None:
            positions = np.zeros((count, 3), dtype=float)
        self._positions = np.asarray(positions, dtype=float).reshape(count, 3).copy()
        self._numbers = np.array(
            [_ATOMIC_NUMBERS.get(symbol, 0) for symbol in symbols], dtype=int
        )
        self._symbols = list(symbols)
        self._cell = np.zeros((3, 3), dtype=float) if cell is None else np.asarray(cell, dtype=float).reshape(3, 3).copy()
        if isinstance(pbc, bool):
            self._pbc = np.array([pbc, pbc, pbc], dtype=bool)
        else:
            values = list(pbc)
            self._pbc = np.array(values if len(values) == 3 else [False, False, False], dtype=bool)
        self.arrays = {"numbers": self._numbers, "positions": self._positions}
        for key, value in arrays.items():
            self.set_array(key, value)

    def __len__(self) -> int:
        return len(self._symbols)

    def __getitem__(self, index) -> "Atoms":
        if isinstance(index, slice):
            chosen = list(range(len(self)))[index]
        elif isinstance(index, (int, np.integer)):
            chosen = [int(index)]
        else:
            chosen = [int(item) for item in index]
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

    def get_pbc(self) -> np.ndarray:
        return self._pbc

    def get_scaled_positions(self, wrap: bool = True) -> np.ndarray:
        cell = self._cell
        if abs(float(np.linalg.det(cell))) < 1e-12:
            return np.zeros_like(self._positions)
        fractional = self._positions @ np.linalg.inv(cell)
        if wrap:
            fractional = fractional - np.floor(fractional)
        return fractional

    def set_array(self, name: str, values) -> None:
        array = np.array(values, copy=True)
        if len(self) and len(array) != len(self):
            raise ValueError(f"{name} must contain one value per atom")
        self.arrays[name] = array

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
            symbol + ("" if counts[symbol] == 1 else str(counts[symbol])) for symbol in ordered
        )


def neighbor_list(quantities: str, atoms: Atoms, cutoff, self_interaction: bool = False):
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
        for shift in _image_shifts(cell, pbc, max_cutoff):
            offset = shift @ cell
            for i in range(count):
                for j in range(count):
                    if not self_interaction and i == j and not np.any(shift):
                        continue
                    vector = positions[j] + offset - positions[i]
                    distance = float(np.linalg.norm(vector))
                    limit = pair_cutoff[i, j]
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
    """Return ``(vectors, distances)`` with ASE's pair-matrix layout."""
    left = np.asarray(p1, dtype=float)
    right = np.asarray(p2, dtype=float)
    if left.ndim == 1:
        left = left.reshape(1, 3)
    if right.ndim == 1:
        right = right.reshape(1, 3)
    delta = right[None, :, :] - left[:, None, :]
    matrix = None if cell is None else np.asarray(cell, dtype=float)
    periodic = np.zeros(3, dtype=bool) if pbc is None else np.asarray(pbc, dtype=bool)
    if matrix is not None and abs(float(np.linalg.det(matrix))) > 1e-12 and np.any(periodic):
        fractional = delta @ np.linalg.inv(matrix)
        fractional[..., periodic] -= np.rint(fractional[..., periodic])
        delta = fractional @ matrix
    distances = np.linalg.norm(delta, axis=-1)
    return delta, distances


def _is_atoms(value) -> bool:
    return hasattr(value, "get_positions") and hasattr(value, "get_chemical_symbols") and not isinstance(
        value, (str, list, tuple, np.ndarray)
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


def _image_shifts(cell: np.ndarray, pbc: np.ndarray, cutoff: float):
    if abs(float(np.linalg.det(cell))) < 1e-12:
        return [np.zeros(3, dtype=int)]
    ranges = []
    for axis in range(3):
        if not pbc[axis]:
            ranges.append([0])
            continue
        other = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
        height = abs(float(np.dot(cell[axis], other))) / (float(np.linalg.norm(other)) + 1e-15)
        count = int(np.ceil(cutoff / max(height, 1e-8)))
        ranges.append(range(-count, count + 1))
    return [np.array(shift, dtype=int) for shift in product(*ranges)]
