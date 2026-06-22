"""Geometry helpers for interaction detectors."""

from __future__ import annotations

import itertools

import numpy as np


def vector_angle_deg(vec1, vec2) -> float:
    """Return the angle between two vectors in degrees."""
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def image_translation(lattice, image: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert an integer periodic image to a Cartesian translation vector."""
    translation = np.asarray(image, dtype=float) @ np.asarray(lattice, dtype=float)
    return tuple(float(v) for v in translation)


def enumerate_lattice_images(
    lattice,
    pbc: tuple[bool, bool, bool] = (True, True, True),
    search_radius_A: float | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """Enumerate plausible periodic images for an interaction search radius.

    The ranges are conservative and based on lattice-vector lengths.  The origin
    image is always included.
    """
    if search_radius_A is None or search_radius_A <= 0:
        ranges = [range(-1, 2) if flag else range(0, 1) for flag in pbc]
    else:
        lat = np.asarray(lattice, dtype=float)
        ranges = []
        for axis, flag in enumerate(pbc):
            if not flag:
                ranges.append(range(0, 1))
                continue
            length = float(np.linalg.norm(lat[axis]))
            max_n = 1 if length == 0 else int(np.ceil(search_radius_A / length)) + 1
            ranges.append(range(-max_n, max_n + 1))

    images = tuple(tuple(int(v) for v in image) for image in itertools.product(*ranges))
    return images if (0, 0, 0) in images else ((0, 0, 0), *images)


def best_fit_plane(points) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return centroid, unit normal, and plane RMSD for a point cloud."""
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal = normal / norm
    deviations = centered @ normal
    rmsd = float(np.sqrt(np.mean(deviations**2)))
    return (
        tuple(float(v) for v in centroid),
        tuple(float(v) for v in normal),
        rmsd,
    )
