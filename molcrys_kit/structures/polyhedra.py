"""
Polyhedral geometry definitions and hull serialization helpers.

This module defines ideal coordination polyhedra for CN=4 through CN=12 using
a data-driven registry. Each polyhedron is described by:
  - name: canonical identifier (e.g., "octahedron")
  - cn: coordination number (vertex count)
  - point_group: Schoenflies symbol (e.g., "Oh")
  - category: structural family (platonic, archimedean, prism, antiprism, bipyramid, capped)
  - vertices: normalized unit-sphere coordinates, shape (cn, 3)

Public API (backward-compatible):
  - IDEAL_POLYHEDRA: Dict[int, Dict[str, np.ndarray]]
  - ideal_polyhedra_for_cn(cn) -> Dict[str, np.ndarray]
  - all_ideal_polyhedra() -> Dict[int, Dict[str, np.ndarray]]
  - convex_hull_payload(shell_coords) -> Dict[str, Any]
  - get_polyhedron(name) -> IdealPolyhedron | None   (new)
  - list_polyhedra() -> List[IdealPolyhedron]        (new)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - optional dependency
    ConvexHull = None


# --- Core utilities ---


def _array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _normalize_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Center points at origin and project onto unit sphere."""
    arr = np.array(list(points), dtype=float)
    arr -= arr.mean(axis=0)
    norms = np.linalg.norm(arr, axis=1)
    nonzero = norms > 1e-8
    if np.any(nonzero):
        arr[nonzero] /= norms[nonzero][:, None]
    return arr


# --- Parametric vertex constructors ---


def _ring(
    n: int,
    z: float,
    phase_deg: float = 0.0,
    radius: Optional[float] = None,
) -> List[List[float]]:
    """Generate n equally-spaced points on a ring at height z."""
    if radius is None:
        radius = math.sqrt(max(1.0 - z * z, 1e-8))
    pts = []
    for idx in range(n):
        ang = math.radians(phase_deg + idx * 360.0 / n)
        pts.append([radius * math.cos(ang), radius * math.sin(ang), z])
    return pts


def _bipyramid(n_equatorial: int) -> List[List[float]]:
    """n-gonal bipyramid: equatorial ring + two axial poles."""
    equatorial = _ring(n_equatorial, 0.0, 0.0, radius=1.0)
    axial = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    return equatorial + axial


def _prism(n: int, half_height: float = 0.55) -> List[List[float]]:
    """n-gonal prism: two aligned n-rings."""
    top = _ring(n, half_height, 0.0)
    bottom = _ring(n, -half_height, 0.0)
    return top + bottom


def _antiprism(n: int, half_height: float = 0.48) -> List[List[float]]:
    """n-gonal antiprism: two offset n-rings."""
    top = _ring(n, half_height, 0.0)
    bottom = _ring(n, -half_height, 180.0 / n)
    return top + bottom


def _dodecahedron8() -> List[List[float]]:
    """Eight-vertex dodecahedral reference retained for legacy CN=8 scoring."""
    return [[s1, s2, 0.0] for s1 in (-1, 1) for s2 in (-1, 1)] + [
        [0.0, s1 / math.sqrt(2.0), s2 * math.sqrt(0.5)]
        for s1 in (-1, 1)
        for s2 in (-1, 1)
    ]


_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _icosahedron() -> List[List[float]]:
    """Icosahedron vertices before normalization."""
    return [
        pt
        for s1 in (-1, 1)
        for s2 in (-1, 1)
        for pt in (
            [0.0, s1, s2 * _PHI],
            [s1, s2 * _PHI, 0.0],
            [s2 * _PHI, 0.0, s1],
        )
    ]


def _cuboctahedron() -> List[List[float]]:
    """Cuboctahedron vertices before normalization."""
    return (
        [[s1, s2, 0.0] for s1 in (-1, 1) for s2 in (-1, 1)]
        + [[s1, 0.0, s2] for s1 in (-1, 1) for s2 in (-1, 1)]
        + [[0.0, s1, s2] for s1 in (-1, 1) for s2 in (-1, 1)]
    )


# --- Data class and registry ---


@dataclass(frozen=True)
class IdealPolyhedron:
    """One ideal polyhedron geometry with metadata."""

    name: str
    cn: int
    point_group: str
    category: str
    vertices: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.vertices.shape != (self.cn, 3):
            raise ValueError(
                f"{self.name}: expected ({self.cn}, 3), got {self.vertices.shape}"
            )


_REGISTRY: List[IdealPolyhedron] = []


def _register(
    name: str,
    cn: int,
    raw_vertices: Iterable[Iterable[float]],
    *,
    point_group: str = "?",
    category: str = "other",
) -> None:
    """Normalize vertices and register as an ideal polyhedron."""
    verts = _normalize_points(raw_vertices)
    _REGISTRY.append(
        IdealPolyhedron(
            name=name,
            cn=cn,
            point_group=point_group,
            category=category,
            vertices=verts,
        )
    )


# --- CN = 4 ---

_register(
    "tetrahedron",
    4,
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    point_group="Td",
    category="platonic",
)

_register(
    "square_planar",
    4,
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
    point_group="D4h",
    category="other",
)


# --- CN = 5 ---

_register(
    "trigonal_bipyramid", 5, _bipyramid(3), point_group="D3h", category="bipyramid"
)

_register(
    "square_pyramid",
    5,
    _ring(4, -0.30, 0.0, radius=0.95) + [[0.0, 0.0, 1.0]],
    point_group="C4v",
    category="capped",
)


# --- CN = 6 ---

_register(
    "octahedron",
    6,
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    point_group="Oh",
    category="platonic",
)

_register("trigonal_prism", 6, _prism(3), point_group="D3h", category="prism")


# --- CN = 7 ---

_register(
    "pentagonal_bipyramid", 7, _bipyramid(5), point_group="D5h", category="bipyramid"
)

_register(
    "capped_octahedron",
    7,
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)],
    ],
    point_group="C3v",
    category="capped",
)


# --- CN = 8 ---

_register(
    "cube",
    8,
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    point_group="Oh",
    category="platonic",
)

_register("square_antiprism", 8, _antiprism(4), point_group="D4d", category="antiprism")

_register("dodecahedron", 8, _dodecahedron8(), point_group="D2d", category="other")


# --- CN = 9 ---

_register(
    "tricapped_trigonal_prism",
    9,
    _prism(3) + _ring(3, 0.0, 60.0, radius=1.0),
    point_group="D3h",
    category="capped",
)

_register(
    "capped_square_antiprism",
    9,
    _ring(4, 0.42, 0.0) + _ring(4, -0.42, 45.0) + [[0.0, 0.0, 1.0]],
    point_group="C4v",
    category="capped",
)


# --- CN = 10 ---

_register(
    "bicapped_square_antiprism",
    10,
    _ring(4, 0.36, 0.0) + _ring(4, -0.36, 45.0) + [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    point_group="D4d",
    category="capped",
)

_register(
    "bicapped_dodecahedron",
    10,
    _dodecahedron8() + [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    point_group="D2d",
    category="capped",
)


# --- CN = 11 ---

_register(
    "capped_pentagonal_antiprism",
    11,
    _ring(5, 0.36, 0.0) + _ring(5, -0.36, 36.0) + [[0.0, 0.0, 1.0]],
    point_group="C5v",
    category="capped",
)

_register(
    "capped_pentagonal_prism",
    11,
    _ring(5, 0.42, 0.0) + _ring(5, -0.42, 0.0) + [[0.0, 0.0, 1.0]],
    point_group="C5v",
    category="capped",
)

_register(
    "edge_bicapped_square_antiprism",
    11,
    _ring(4, 0.34, 0.0)
    + _ring(4, -0.34, 45.0)
    + [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    point_group="C2v",
    category="capped",
)

# Tricapped cube (C3v): 8 cube corners + 3 face caps on the three mutually
# orthogonal faces meeting at the corner (1,1,1). Caps lie on the 3-fold axis
# along the body diagonal, so the polyhedron retains C3v symmetry. This is the
# canonical CN=11 environment for several lanthanide and actinide complexes
# and for hydrogen-bonded inorganic-cation coordination shells (e.g. ClO4^-
# around an A2BX5 A-site cation), which the existing pentagonal-(anti)prism
# and edge-bicapped square antiprism candidates do not represent well.
_register(
    "tricapped_cube",
    11,
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    point_group="C3v",
    category="capped",
)


# --- CN = 12 ---

_register("icosahedron", 12, _icosahedron(), point_group="Ih", category="platonic")

_register(
    "cuboctahedron", 12, _cuboctahedron(), point_group="Oh", category="archimedean"
)


# --- Build the legacy IDEAL_POLYHEDRA dict from registry ---

IDEAL_POLYHEDRA: Dict[int, Dict[str, np.ndarray]] = {}
for _poly in _REGISTRY:
    IDEAL_POLYHEDRA.setdefault(_poly.cn, {})[_poly.name] = _poly.vertices


# --- Public API ---


def ideal_polyhedra_for_cn(cn: int) -> Dict[str, np.ndarray]:
    """Return dict of {name: vertices} for a given coordination number."""
    return IDEAL_POLYHEDRA.get(cn, {})


def all_ideal_polyhedra() -> Dict[int, Dict[str, np.ndarray]]:
    """Return the full CN -> {name: vertices} mapping."""
    return IDEAL_POLYHEDRA


def get_polyhedron(name: str) -> Optional[IdealPolyhedron]:
    """Look up a polyhedron by name. Returns None if not found."""
    for poly in _REGISTRY:
        if poly.name == name:
            return poly
    return None


def list_polyhedra() -> List[IdealPolyhedron]:
    """Return all registered polyhedra."""
    return list(_REGISTRY)


def convex_hull_payload(shell_coords: Iterable[Iterable[float]]) -> Dict[str, Any]:
    """Compute convex hull faces and edges for a set of 3D coordinates."""
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
    "IdealPolyhedron",
    "all_ideal_polyhedra",
    "convex_hull_payload",
    "get_polyhedron",
    "ideal_polyhedra_for_cn",
    "list_polyhedra",
]
