"""Pymatgen-compatible symmetry and lattice types for the CIF path.

Used only when pymatgen itself is not installed. Operation strings and the
space-group tables live in ``data/``; see ``data/ATTRIBUTION.txt``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data"
_XYZ_COMPONENT = re.compile(r"^[xyz0-9+\-*/().]+$")


class SymmOp:
    """Column-affine operation ``f' = R f + t``."""

    def __init__(self, rotation, translation, xyz: str | None = None):
        self._rotation = np.asarray(rotation, dtype=float).reshape(3, 3).copy()
        self._translation = np.mod(np.asarray(translation, dtype=float).reshape(3), 1.0)
        self._translation[np.isclose(self._translation, 1.0)] = 0.0
        self._xyz = xyz

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self._rotation

    @property
    def translation_vector(self) -> np.ndarray:
        return self._translation

    @classmethod
    def from_xyz_str(cls, xyz_str: str) -> "SymmOp":
        components = [part.strip().replace(" ", "") for part in str(xyz_str).split(",")]
        if len(components) != 3:
            raise ValueError(f"invalid symmetry operation: {xyz_str!r}")
        rotation = np.zeros((3, 3), dtype=float)
        translation = np.zeros(3, dtype=float)
        for index, component in enumerate(components):
            if not component or _XYZ_COMPONENT.fullmatch(component) is None:
                raise ValueError(f"invalid symmetry operation: {xyz_str!r}")
            try:
                translation[index] = _eval_xyz(component, 0.0, 0.0, 0.0)
                rotation[index, 0] = _eval_xyz(component, 1.0, 0.0, 0.0) - translation[index]
                rotation[index, 1] = _eval_xyz(component, 0.0, 1.0, 0.0) - translation[index]
                rotation[index, 2] = _eval_xyz(component, 0.0, 0.0, 1.0) - translation[index]
            except Exception as exc:
                raise ValueError(f"invalid symmetry operation: {xyz_str!r}") from exc
        return cls(rotation, translation, xyz=component_string(components))

    @classmethod
    def from_rotation_and_translation(cls, rotation, translation) -> "SymmOp":
        return cls(rotation, translation)

    def as_xyz_str(self) -> str:
        if self._xyz:
            return self._xyz
        return _matrix_to_xyz(self._rotation, self._translation)

    def operate(self, point) -> np.ndarray:
        coordinate = np.asarray(point, dtype=float)
        return self._rotation @ coordinate + self._translation


class Lattice:
    """Row-vector lattice, matching pymatgen's ``matrix`` convention."""

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float).reshape(3, 3).copy()

    @classmethod
    def from_parameters(cls, a, b, c, alpha, beta, gamma) -> "Lattice":
        angles = np.radians([alpha, beta, gamma])
        cos_alpha, cos_beta, cos_gamma = np.cos(angles)
        sin_gamma = np.sin(angles[2])
        if abs(sin_gamma) < 1e-12:
            raise ValueError("cell gamma is degenerate")
        vector_c_x = c * cos_beta
        vector_c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        height_sq = 1.0 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma) ** 2
        if height_sq < -1e-8:
            raise ValueError("cell angles are inconsistent")
        vector_c_z = c * np.sqrt(max(height_sq, 0.0))
        matrix = np.array(
            [
                [a, 0.0, 0.0],
                [b * cos_gamma, b * sin_gamma, 0.0],
                [vector_c_x, vector_c_y, vector_c_z],
            ],
            dtype=float,
        )
        return cls(matrix)

    @classmethod
    def cubic(cls, a: float) -> "Lattice":
        return cls(np.eye(3) * float(a))


class SpaceGroup:
    """Standard-setting space group loaded from the vendored operation table."""

    def __init__(self, int_symbol: str):
        tables = _tables()
        symbol = re.sub(r"\s+", "", str(int_symbol))
        abbreviated = tables["abbreviated_spacegroup_symbols"]
        if symbol in abbreviated:
            symbol = abbreviated[symbol]
        full_to_key = {
            value["full_symbol"]: key
            for key, value in tables["space_group_encoding"].items()
        }
        if symbol in full_to_key:
            symbol = full_to_key[symbol]
        match = None
        for entry in tables["ops"]:
            names = (
                entry["hermann_mauguin"],
                entry["universal_h_m"],
                entry["hermann_mauguin_u"],
            )
            if symbol in names:
                match = entry
                break
        if match is None:
            raise ValueError(f"unknown space group symbol: {int_symbol}")
        self.symbol = match["hermann_mauguin_u"]
        self.int_number = int(match["number"])
        self.symmetry_ops = [SymmOp.from_xyz_str(item) for item in match["symops"]]

    @classmethod
    def from_int_number(cls, int_number: int, hexagonal: bool = True) -> "SpaceGroup":
        if int_number not in range(1, 231):
            raise ValueError(
                f"International number must be between 1 and 230, got {int_number}"
            )
        symbol = _symbol_from_number(int_number, hexagonal=hexagonal)
        if not hexagonal and int_number in (146, 148, 155, 160, 161, 166, 167):
            symbol += ":R"
        return cls(symbol)


def _eval_xyz(component: str, x: float, y: float, z: float) -> float:
    return float(eval(component, {"__builtins__": {}}, {"x": x, "y": y, "z": z}))


def component_string(components: list[str]) -> str:
    return ",".join(components)


def _format_term(value: float) -> str:
    fraction = value - np.floor(value)
    if np.isclose(fraction, 0.0):
        return ""
    known = {0.5: "1/2", 1 / 3: "1/3", 2 / 3: "2/3", 0.25: "1/4", 0.75: "3/4", 1 / 6: "1/6", 5 / 6: "5/6"}
    for number, text in known.items():
        if np.isclose(fraction, number):
            return "+" + text
    return f"+{fraction:.4g}"


def _matrix_to_xyz(rotation: np.ndarray, translation: np.ndarray) -> str:
    parts = []
    for row in range(3):
        terms = []
        for column, axis in enumerate("xyz"):
            coefficient = int(np.rint(rotation[row, column]))
            if coefficient == 0:
                continue
            if coefficient == 1:
                terms.append(axis)
            elif coefficient == -1:
                terms.append("-" + axis)
            else:
                terms.append(f"{coefficient}*{axis}")
        body = "".join(terms) if terms else "0"
        if body.startswith("0"):
            body = _format_term(float(translation[row])).lstrip("+") or "0"
        else:
            body += _format_term(float(translation[row]))
        parts.append(body)
    return ",".join(parts)


@lru_cache(maxsize=1)
def _tables() -> dict:
    encoding = json.loads((_DATA_DIR / "symm_data.json").read_text(encoding="utf-8"))
    operations = json.loads((_DATA_DIR / "symm_ops.json").read_text(encoding="utf-8"))
    for entry in operations:
        entry["hermann_mauguin"] = re.sub(r" ", "", entry["hermann_mauguin"])
        entry["universal_h_m"] = re.sub(r" ", "", entry["universal_h_m"])
    return {
        "abbreviated_spacegroup_symbols": encoding["abbreviated_spacegroup_symbols"],
        "space_group_encoding": encoding["space_group_encoding"],
        "ops": operations,
    }


def _symbol_from_number(int_number: int, hexagonal: bool) -> str:
    symbols = [
        name
        for name, value in _tables()["space_group_encoding"].items()
        if value["int_number"] == int_number
    ]
    if not symbols:
        raise ValueError(f"Invalid international number: {int_number}")
    if len(symbols) == 2:
        for symbol in symbols:
            if "e" in symbol:
                return symbol
        if hexagonal:
            hexagonal_symbols = [symbol for symbol in symbols if symbol.endswith("H")]
            if hexagonal_symbols:
                symbols = hexagonal_symbols
        else:
            symbols = [symbol for symbol in symbols if not symbol.endswith("H")]
    return symbols.pop()
