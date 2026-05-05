"""
Polyhedral geometry definitions and hull serialization helpers.

This module contains framework-independent geometric data for common
high-coordination polyhedra. Analysis routines that infer shells or score
distortions live in :mod:`molcrys_kit.analysis.packing_shell`.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - optional dependency
    ConvexHull = None


def _array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _normalize_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    arr -= arr.mean(axis=0)
    norms = np.linalg.norm(arr, axis=1)
    nonzero = norms > 1e-8
    if np.any(nonzero):
        arr[nonzero] /= norms[nonzero][:, None]
    return arr


def _ring(n: int, z: float, phase_deg: float = 0.0, radius: float = None) -> List[List[float]]:
    if radius is None:
        radius = math.sqrt(max(1.0 - z * z, 1e-8))
    pts = []
    for idx in range(n):
        ang = math.radians(phase_deg + idx * 360.0 / n)
        pts.append([radius * math.cos(ang), radius * math.sin(ang), z])
    return pts


def _cube() -> np.ndarray:
    return _normalize_points(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )


def _square_antiprism() -> np.ndarray:
    return _normalize_points(_ring(4, 0.48, 0.0) + _ring(4, -0.48, 45.0))


def _dodecahedron8() -> np.ndarray:
    pts = []
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            pts.append([sign_x, sign_y, 0.0])
            pts.append([0.0, sign_x / math.sqrt(2), sign_y * math.sqrt(0.5)])
    return _normalize_points(pts[:8])


def _tricapped_trigonal_prism() -> np.ndarray:
    top = _ring(3, 0.55, 0.0)
    bottom = _ring(3, -0.55, 60.0)
    caps = [
        [1.0, 0.0, 0.0],
        [-0.5, math.sqrt(3) / 2.0, 0.0],
        [-0.5, -math.sqrt(3) / 2.0, 0.0],
    ]
    return _normalize_points(top + bottom + caps)


def _capped_square_antiprism() -> np.ndarray:
    return _normalize_points(
        _ring(4, 0.42, 0.0) + _ring(4, -0.42, 45.0) + [[0.0, 0.0, 1.0]]
    )


def _bicapped_square_antiprism() -> np.ndarray:
    return _normalize_points(
        _ring(4, 0.36, 0.0)
        + _ring(4, -0.36, 45.0)
        + [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )


def _bicapped_dodecahedron() -> np.ndarray:
    base = _ring(5, 0.15, 0.0, radius=0.95) + _ring(5, -0.15, 36.0, radius=0.95)
    return _normalize_points(base)


def _capped_pentagonal_antiprism() -> np.ndarray:
    return _normalize_points(
        _ring(5, 0.36, 0.0) + _ring(5, -0.36, 36.0) + [[0.0, 0.0, 1.0]]
    )


def _capped_pentagonal_prism() -> np.ndarray:
    return _normalize_points(
        _ring(5, 0.42, 0.0) + _ring(5, -0.42, 0.0) + [[0.0, 0.0, 1.0]]
    )


def _edge_bicapped_square_antiprism() -> np.ndarray:
    base = _ring(4, 0.34, 0.0) + _ring(4, -0.34, 45.0)
    caps = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    return _normalize_points(base + caps)


def _tetrahedron() -> np.ndarray:
    """Regular tetrahedron (CN=4)."""
    return _normalize_points([
        [1.0,  1.0,  1.0],
        [1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ])


def _square_planar() -> np.ndarray:
    """Square planar (CN=4)."""
    return _normalize_points([
        [1.0,  0.0, 0.0],
        [-1.0,  0.0, 0.0],
        [0.0,  1.0, 0.0],
        [0.0, -1.0, 0.0],
    ])


def _trigonal_bipyramid() -> np.ndarray:
    """Trigonal bipyramid (CN=5)."""
    equatorial = _ring(3, 0.0, 0.0, radius=1.0)
    axial = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    return _normalize_points(equatorial + axial)


def _square_pyramid() -> np.ndarray:
    """Square pyramid (CN=5)."""
    base = _ring(4, -0.30, 0.0, radius=0.95)
    apex = [[0.0, 0.0, 1.0]]
    return _normalize_points(base + apex)


def _octahedron() -> np.ndarray:
    """Regular octahedron (CN=6)."""
    return _normalize_points([
        [1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0,  0.0, -1.0],
    ])


def _trigonal_prism() -> np.ndarray:
    """Trigonal prism (CN=6)."""
    top = _ring(3, 0.55, 0.0)
    bottom = _ring(3, -0.55, 0.0)
    return _normalize_points(top + bottom)


def _pentagonal_bipyramid() -> np.ndarray:
    """Pentagonal bipyramid (CN=7)."""
    equatorial = _ring(5, 0.0, 0.0, radius=1.0)
    axial = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    return _normalize_points(equatorial + axial)


def _capped_octahedron() -> np.ndarray:
    """Capped octahedron (CN=7): octahedron + one face cap."""
    oct_pts = [
        [1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0,  0.0, -1.0],
    ]
    cap = [[0.577, 0.577, 0.577]]
    return _normalize_points(oct_pts + cap)


def _icosahedron() -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    pts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pts.extend(
                [
                    [0.0, s1, s2 * phi],
                    [s1, s2 * phi, 0.0],
                    [s2 * phi, 0.0, s1],
                ]
            )
    return _normalize_points(pts)


def _cuboctahedron() -> np.ndarray:
    pts = []
    for zero in range(3):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                pt = [0.0, 0.0, 0.0]
                axes = [0, 1, 2]
                axes.remove(zero)
                pt[axes[0]] = s1
                pt[axes[1]] = s2
                pts.append(pt)
    return _normalize_points(pts)


IDEAL_POLYHEDRA: Dict[int, Dict[str, np.ndarray]] = {
    4: {
        "tetrahedron": _tetrahedron(),
        "square_planar": _square_planar(),
    },
    5: {
        "trigonal_bipyramid": _trigonal_bipyramid(),
        "square_pyramid": _square_pyramid(),
    },
    6: {
        "octahedron": _octahedron(),
        "trigonal_prism": _trigonal_prism(),
    },
    7: {
        "pentagonal_bipyramid": _pentagonal_bipyramid(),
        "capped_octahedron": _capped_octahedron(),
    },
    8: {
        "cube": _cube(),
        "square_antiprism": _square_antiprism(),
        "dodecahedron": _dodecahedron8(),
    },
    9: {
        "capped_square_antiprism": _capped_square_antiprism(),
        "tricapped_trigonal_prism": _tricapped_trigonal_prism(),
    },
    10: {
        "bicapped_square_antiprism": _bicapped_square_antiprism(),
        "bicapped_dodecahedron": _bicapped_dodecahedron(),
    },
    11: {
        "capped_pentagonal_antiprism": _capped_pentagonal_antiprism(),
        "capped_pentagonal_prism": _capped_pentagonal_prism(),
        "edge_bicapped_square_antiprism": _edge_bicapped_square_antiprism(),
    },
    12: {
        "icosahedron": _icosahedron(),
        "cuboctahedron": _cuboctahedron(),
    },
}


def ideal_polyhedra_for_cn(cn: int) -> Dict[str, np.ndarray]:
    return IDEAL_POLYHEDRA.get(cn, {})


def all_ideal_polyhedra() -> Dict[int, Dict[str, np.ndarray]]:
    return IDEAL_POLYHEDRA


def convex_hull_payload(shell_coords: Iterable[Iterable[float]]) -> Dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < 4 or ConvexHull is None:
        return {"vertices": coords.tolist(), "simplices": [], "edges": []}
    hull = ConvexHull(coords)
    edges = set()
    for simplex in hull.simplices:
        simplex = list(simplex)
        for i, j in itertools.combinations(simplex, 2):
            edges.add(tuple(sorted((int(i), int(j)))))
    return {
        "vertices": coords.tolist(),
        "simplices": hull.simplices.tolist(),
        "edges": [list(edge) for edge in sorted(edges)],
    }


__all__ = [
    "IDEAL_POLYHEDRA",
    "all_ideal_polyhedra",
    "convex_hull_payload",
    "ideal_polyhedra_for_cn",
]
