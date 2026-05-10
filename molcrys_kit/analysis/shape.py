"""
Topology-gated Continuous Shape Measure (CShM) classification.

References
----------
Pinsky, M.; Avnir, D. "Continuous Symmetry Measures. 5. The Classical
Polyhedra." *Inorg. Chem.* **1998**, *37*, 5575-5582.
DOI: 10.1021/ic9804925. Defines the CShM convention and unit-sphere
normalization used for ideal coordination polyhedra.

Casanova, D.; Cirera, J.; Llunell, M.; Alemany, P.; Avnir, D.; Alvarez, S.
"Minimal Distortion Pathways in Polyhedral Rearrangements." *J. Am. Chem.
Soc.* **2004**, *126*, 1755-1763. DOI: 10.1021/ja036479n. Motivates using
multiple alignment starts to avoid local minima.

Llunell, M.; Casanova, D.; Cirera, J.; Alemany, P.; Alvarez, S. *SHAPE 2.1*,
University of Barcelona, **2013**. Reference implementation whose CShM
conventions are mirrored here.

Cirera, J.; Ruiz, E.; Alvarez, S. "Continuous Shape Measures as a
Stereochemical Tool in Organometallic Chemistry." *Organometallics* **2005**,
*24*, 1556-1562. DOI: 10.1021/om049150z. Provides practical thresholds for
clean, distorted, and irregular shape labels.

Alvarez, S. "Distortion Pathways of Transition Metal Coordination Polyhedra."
*Coord. Chem. Rev.* **2005**, *249*, 1789-1808.
DOI: 10.1016/j.ccr.2005.04.005. Treats CShM as a continuous distortion
coordinate for naming coordination environments.

Kabsch, W. "A solution for the best rotation to relate two sets of vectors."
*Acta Crystallogr.* **1976**, *A32*, 922-923.
DOI: 10.1107/S0567739476001873.

Kabsch, W. "A discussion of the solution for the best rotation to relate two
sets of vectors." *Acta Crystallogr.* **1978**, *A34*, 827-828.
DOI: 10.1107/S0567739478001680.

Gower, J. C. "Generalized Procrustes Analysis." *Psychometrika* **1975**,
*40*, 33-51. DOI: 10.1007/BF02291478.

Kuhn, H. W. "The Hungarian Method for the Assignment Problem." *Naval Res.
Logist. Quart.* **1955**, *2*, 83-97. DOI: 10.1002/nav.3800020109.

Munkres, J. "Algorithms for the Assignment and Transportation Problems."
*J. Soc. Ind. Appl. Math.* **1957**, *5*, 32-38. DOI: 10.1137/0105003.

Jonker, R.; Volgenant, A. "A Shortest Augmenting Path Algorithm for Dense and
Sparse Linear Assignment Problems." *Computing* **1987**, *38*, 325-340.
DOI: 10.1007/BF02278710. This is the assignment algorithm used by SciPy's
``linear_sum_assignment``.

Besl, P. J.; McKay, N. D. "A Method for Registration of 3-D Shapes."
*IEEE Trans. Pattern Anal. Mach. Intell.* **1992**, *14*, 239-256.
DOI: 10.1109/34.121791. Gives the iterative correspondence/alignment pattern
used here with Hungarian assignment plus Kabsch rotation.

Barber, C. B.; Dobkin, D. P.; Huhdanpaa, H. T. "The Quickhull Algorithm for
Convex Hulls." *ACM Trans. Math. Softw.* **1996**, *22*, 469-483.
DOI: 10.1145/235815.235821.

Hoppe, R. "The Coordination Number -- an 'Inorganic Chameleon'."
*Angew. Chem. Int. Ed.* **1970**, *9*, 25-34. DOI: 10.1002/anie.197000251.

Lima-de-Faria, J.; Hellner, E.; Liebau, F.; Makovicky, E.; Parthe, E.
"Nomenclature of Inorganic Structure Types." *Acta Crystallogr.* **1990**,
*A46*, 1-11. DOI: 10.1107/S0108767389008834.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..structures.polyhedra import (
    _polyhedron_topology_signature,
    ideal_polyhedra_for_cn,
    list_polyhedra,
)


CLEAN_CSHM = 0.5
DISTORTED_CSHM = 3.0
AMBIGUOUS_GAP = 0.3


def _array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _unit_sphere(points: Iterable[Iterable[float]], center: Optional[Iterable[float]] = None) -> np.ndarray:
    arr = _array(points)
    if len(arr) == 0:
        return arr.reshape(0, 3)
    if center is None:
        arr = arr - arr.mean(axis=0)
    else:
        arr = arr - np.asarray(center, dtype=float)
    norms = np.linalg.norm(arr, axis=1)
    nonzero = norms > 1e-12
    out = np.zeros_like(arr, dtype=float)
    out[nonzero] = arr[nonzero] / norms[nonzero][:, None]
    return out


def _kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return rotation matrix Q minimizing ||mobile @ Q - target||."""
    covariance = mobile.T @ target
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(mat)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def topology_signature(
    coords: Iterable[Iterable[float]],
    *,
    face_merge_tol_deg: float = 1.0,
) -> Dict[str, Any]:
    """Return a convex-hull topology fingerprint for a shell.

    The signature consists of face-size counts, edge count, and vertex-degree
    counts after coplanar ConvexHull triangles are merged into polygonal faces.

    References: Barber et al., *ACM Trans. Math. Softw.* **1996**, 22,
    469-483. DOI: 10.1145/235815.235821 (Quickhull); Lima-de-Faria et al.,
    *Acta Crystallogr.* **1990**, A46, 1-11.
    DOI: 10.1107/S0108767389008834 (polyhedral face-graph nomenclature).
    """
    return _polyhedron_topology_signature(
        coords, face_merge_tol_deg=face_merge_tol_deg
    )


def cshm(
    actual_unit: Iterable[Iterable[float]],
    ideal_unit: Iterable[Iterable[float]],
    *,
    n_random_inits: int = 8,
    max_iter: int = 25,
    tol: float = 1e-7,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """Compute a Continuous Shape Measure against one ideal.

    The calculation alternates between a linear assignment step (actual vertex
    to ideal vertex) and a Kabsch rotation step, keeping the best result across
    deterministic identity plus random SO(3) initial rotations.

    References: Pinsky & Avnir, *Inorg. Chem.* **1998**, 37, 5575-5582.
    DOI: 10.1021/ic9804925 (CShM definition); Casanova et al.,
    *J. Am. Chem. Soc.* **2004**, 126, 1755-1763. DOI: 10.1021/ja036479n
    (multi-start alignment); Kabsch, *Acta Crystallogr.* **1976**, A32,
    922-923. DOI: 10.1107/S0567739476001873, and **1978**, A34, 827-828.
    DOI: 10.1107/S0567739478001680 (rotation); Jonker & Volgenant,
    *Computing* **1987**, 38, 325-340. DOI: 10.1007/BF02278710
    (assignment); Besl & McKay, *IEEE TPAMI* **1992**, 14, 239-256.
    DOI: 10.1109/34.121791 (iterative correspondence/alignment); Gower,
    *Psychometrika* **1975**, 40, 33-51. DOI: 10.1007/BF02291478
    (Procrustes framing).
    """
    actual = _unit_sphere(actual_unit)
    ideal = _unit_sphere(ideal_unit)
    if actual.shape != ideal.shape:
        raise ValueError(
            f"CShM requires equal shapes, got actual={actual.shape}, ideal={ideal.shape}"
        )
    n = len(actual)
    if n == 0:
        return {
            "cshm": float("inf"),
            "rotation": np.eye(3).tolist(),
            "permutation": [],
            "iterations": 0,
            "sse": float("inf"),
        }

    denom = float(np.sum(actual * actual))
    if denom < 1e-12:
        denom = float(n)

    rng = np.random.default_rng(random_seed)
    starts = [np.eye(3)]
    starts.extend(_random_rotation(rng) for _ in range(max(0, n_random_inits - 1)))

    best: Dict[str, Any] = {
        "cshm": float("inf"),
        "rotation": np.eye(3),
        "permutation": np.arange(n, dtype=int),
        "iterations": 0,
        "sse": float("inf"),
    }

    for initial_rotation in starts:
        rotation = np.asarray(initial_rotation, dtype=float)
        previous_sse = float("inf")
        permutation = np.arange(n, dtype=int)
        iterations = 0
        for iterations in range(1, max_iter + 1):
            transformed_ideal = ideal @ rotation
            diff = actual[:, None, :] - transformed_ideal[None, :, :]
            cost = np.sum(diff * diff, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            if not np.array_equal(row_ind, np.arange(n)):
                order = np.argsort(row_ind)
                col_ind = col_ind[order]
            permutation = col_ind.astype(int)
            assigned_ideal = ideal[permutation]
            rotation = _kabsch_rotation(assigned_ideal, actual)
            residual = actual - assigned_ideal @ rotation
            sse = float(np.sum(residual * residual))
            if abs(previous_sse - sse) <= tol:
                break
            previous_sse = sse

        cshm_value = 100.0 * sse / denom
        if cshm_value < best["cshm"]:
            best = {
                "cshm": cshm_value,
                "rotation": rotation,
                "permutation": permutation,
                "iterations": iterations,
                "sse": sse,
            }

    return {
        "cshm": float(best["cshm"]),
        "rotation": np.asarray(best["rotation"], dtype=float).tolist(),
        "permutation": np.asarray(best["permutation"], dtype=int).tolist(),
        "iterations": int(best["iterations"]),
        "sse": float(best["sse"]),
    }


def _label_modifier(best_cshm: float, confidence_gap: Optional[float]) -> str:
    if confidence_gap is not None and confidence_gap < AMBIGUOUS_GAP:
        return "ambiguous"
    if best_cshm < CLEAN_CSHM:
        return "clean"
    if best_cshm < DISTORTED_CSHM:
        return "distorted"
    return "irregular"


def classify_shell(
    coords: Iterable[Iterable[float]],
    center: Optional[Iterable[float]] = None,
    *,
    cn: Optional[int] = None,
    n_random_inits: int = 8,
    max_iter: int = 25,
    topology_match: bool = True,
) -> Dict[str, Any]:
    """Classify one coordination shell with topology-gated CShM.

    The hull topology first gates impossible ideals of the same CN. Remaining
    candidates are ranked by CShM, and the label is softened to ``distorted``,
    ``irregular``, or ``ambiguous`` when the best score or top-two gap warrants
    caution.

    References: Cirera et al., *Organometallics* **2005**, 24, 1556-1562.
    DOI: 10.1021/om049150z, and Alvarez, *Coord. Chem. Rev.* **2005**, 249,
    1789-1808. DOI: 10.1016/j.ccr.2005.04.005 (CShM label thresholds);
    Llunell et al., *SHAPE 2.1*, University of Barcelona, **2013** (reference
    implementation conventions).
    """
    coords_arr = _array(coords)
    if cn is None:
        cn = int(len(coords_arr))
    else:
        cn = int(cn)
    actual = _unit_sphere(coords_arr, center=center)
    actual_topology = topology_signature(actual)

    all_polys = [poly for poly in list_polyhedra() if poly.cn == cn]
    if topology_match and actual_topology["face_signature"]:
        candidates = [
            poly
            for poly in all_polys
            if poly.face_signature == actual_topology["face_signature"]
            and poly.edge_count == actual_topology["edge_count"]
            and poly.vertex_degree_signature == actual_topology["vertex_degree_signature"]
        ]
        topology_gate = "matched" if candidates else "bypassed"
    else:
        candidates = []
        topology_gate = "unavailable"

    if not candidates:
        candidates = all_polys

    ranked = []
    ideal_vertices = ideal_polyhedra_for_cn(cn)
    for poly in candidates:
        score = cshm(
            actual,
            ideal_vertices[poly.name],
            n_random_inits=n_random_inits,
            max_iter=max_iter,
        )
        ranked.append(
            {
                "name": poly.name,
                "cshm": score["cshm"],
                "method": "topology_gated_cshm",
                "point_group": poly.point_group,
                "category": poly.category,
                "permutation": score["permutation"],
                "iterations": score["iterations"],
            }
        )
    ranked.sort(key=lambda item: item["cshm"])

    best = ranked[0] if ranked else None
    gap = None
    if len(ranked) > 1:
        gap = float(ranked[1]["cshm"] - ranked[0]["cshm"])
    modifier = _label_modifier(float(best["cshm"]), gap) if best else "irregular"

    return {
        "coordination_number": cn,
        "primary_label": best["name"] if best else None,
        "label_modifier": modifier,
        "confidence_gap": gap,
        "cshm_value": float(best["cshm"]) if best else None,
        "best_match": best,
        "candidates": ranked,
        "topology": {
            **actual_topology,
            "gate": topology_gate,
            "candidate_names": [poly.name for poly in candidates],
        },
    }


__all__ = [
    "AMBIGUOUS_GAP",
    "CLEAN_CSHM",
    "DISTORTED_CSHM",
    "classify_shell",
    "cshm",
    "topology_signature",
]
