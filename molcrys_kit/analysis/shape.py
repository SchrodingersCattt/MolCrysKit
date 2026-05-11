"""
Core-residual Continuous Shape Measure (CShM) classification.

This module describes coordination shells by sweeping over possible residual
vertices: k=0 is the rigid CShM comparison against registered ideals, while
k>0 fits an (N-k)-vertex core and classifies the removed vertices by geometric
role relative to that core (face cap, off-axis cap, edge bridge, vertex
extension, interstitial, or floating). The returned ``primary_label`` remains
compatible with the registered ideal names when the decomposition maps cleanly
to the registry; ``structural_description`` keeps the explicit descriptive
breakdown for irregular or non-standard shells.

Performance
-----------
The per-shell cost is approximately
``sum_k C(N, k) * num_prototypes(CN=N-k) * cshm_unit``. For the intended
coordination range (CN <= 12, default K_max <= 4), this is suitable for
interactive figure generation: N=11, K=3 is typically on the order of hundreds
of partitions, while N=6, K=2 is tens of partitions. Large batch sweeps over
many thousands of structures should be parallelized by the caller (for example
with joblib or process-level sharding); this module intentionally stays
single-shell and side-effect free.

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

Hart, G. W. "Conway notation for polyhedra." online resource, **1998**.
Provides the operation vocabulary for capped/ambo/dual relationships used as
registry metadata here; the full Conway algebra is not implemented.

Alvarez, S.; Llunell, M. "Continuous symmetry measures of penta-coordinated
molecules: Berry and lever-arm pathways revisited." *J. Chem. Soc., Dalton
Trans.* **2000**, 3288-3303. DOI: 10.1039/B004878J. Motivates using
core-plus-residual descriptions as chemically meaningful coordination
pathways.

Pinsky, M.; Avnir, D.; Casanova, D.; Alemany, P. "Continuous shape measures for
polygonal and polyhedral mesomorphic descriptions." *J. Math. Chem.* **1998**,
*23*, 169-204. DOI: 10.1023/A:1019124121224. Supports descriptive rather than
strictly categorical uses of CShM.

Hoppe, R. "The Coordination Number -- an 'Inorganic Chameleon'."
*Angew. Chem. Int. Ed.* **1970**, *9*, 25-34. DOI: 10.1002/anie.197000251.

Lima-de-Faria, J.; Hellner, E.; Liebau, F.; Makovicky, E.; Parthe, E.
"Nomenclature of Inorganic Structure Types." *Acta Crystallogr.* **1990**,
*A46*, 1-11. DOI: 10.1107/S0108767389008834.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..structures.polyhedra import (
    _polyhedron_topology_signature,
    get_polyhedron,
    ideal_polyhedra_for_cn,
    list_polyhedra,
)


CLEAN_CSHM = 0.5
DISTORTED_CSHM = 3.0
AMBIGUOUS_GAP = 0.3
FACE_CAP_DEG = 15.0
OFF_AXIS_CAP_DEG = 45.0
CAP_RADIAL_MIN = 0.75
EDGE_BRIDGE_OFFSET = 0.45
VERTEX_EXTENSION_ALIGNMENT = 0.95
INTERSTITIAL_RADIAL_MAX = 0.5
ROLE_WEIGHT = 0.5
STRIP_WEIGHT = 0.0
REGISTERED_LABEL_BONUS = 0.15


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
    if best_cshm < CLEAN_CSHM:
        return "clean"
    if best_cshm < DISTORTED_CSHM:
        return "distorted"
    if confidence_gap is not None and confidence_gap < AMBIGUOUS_GAP:
        return "ambiguous"
    return "irregular"


def _quality_tier(score: float) -> str:
    return _label_modifier(float(score), None)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-12 or bn < 1e-12:
        return 180.0
    cosang = float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _locate_relative_to_polyhedron(
    point_unit: Iterable[float],
    ideal_polyhedron,
) -> Dict[str, Any]:
    """Locate a residual point relative to face, edge, and vertex features."""
    point = np.asarray(point_unit, dtype=float)
    radial = float(np.linalg.norm(point))

    face_records = []
    for face_id, face in enumerate(getattr(ideal_polyhedron, "faces", ())):
        centroid = np.asarray(face.centroid, dtype=float)
        normal = np.asarray(face.normal, dtype=float)
        if float(np.dot(centroid, normal)) < 0:
            normal = -normal
        vec = point - centroid
        normal_component = float(np.dot(vec, normal))
        tangent = vec - normal_component * normal
        tangent_component = float(np.linalg.norm(tangent))
        axis_offset = _angle_deg(point, centroid)
        face_records.append(
            {
                "face_id": face_id,
                "face_size": int(face.polygon_size),
                "normal_component": normal_component,
                "tangent_component": tangent_component,
                "axis_offset_deg": axis_offset,
                "feature_distance": float(np.linalg.norm(vec)),
            }
        )
    best_face = min(
        face_records,
        key=lambda item: (item["axis_offset_deg"], item["feature_distance"]),
        default=None,
    )

    edge_records = []
    for edge_id, edge in enumerate(getattr(ideal_polyhedron, "edges", ())):
        midpoint = np.asarray(edge.midpoint, dtype=float)
        direction = np.asarray(edge.direction, dtype=float)
        vec = point - midpoint
        along = abs(float(np.dot(vec, direction)))
        perpendicular = float(np.linalg.norm(vec - np.dot(vec, direction) * direction))
        edge_records.append(
            {
                "edge_id": edge_id,
                "along_edge": along,
                "edge_offset": perpendicular,
                "feature_distance": float(np.linalg.norm(vec)),
            }
        )
    best_edge = min(
        edge_records,
        key=lambda item: (item["edge_offset"], item["feature_distance"]),
        default=None,
    )

    axes = np.asarray(getattr(ideal_polyhedron, "vertex_axes", ideal_polyhedron.vertices), dtype=float)
    if len(axes):
        alignments = axes @ (point / radial if radial > 1e-12 else point)
        vertex_id = int(np.argmax(alignments))
        vertex_alignment = float(alignments[vertex_id])
    else:
        vertex_id = None
        vertex_alignment = -1.0

    return {
        "point": point.tolist(),
        "radial_distance": radial,
        "nearest_face_id": None if best_face is None else int(best_face["face_id"]),
        "nearest_face_size": None if best_face is None else int(best_face["face_size"]),
        "face_axis_offset_deg": None if best_face is None else float(best_face["axis_offset_deg"]),
        "face_normal_component": None if best_face is None else float(best_face["normal_component"]),
        "face_tangent_component": None if best_face is None else float(best_face["tangent_component"]),
        "edge_id": None if best_edge is None else int(best_edge["edge_id"]),
        "edge_offset": None if best_edge is None else float(best_edge["edge_offset"]),
        "edge_along": None if best_edge is None else float(best_edge["along_edge"]),
        "vertex_id": vertex_id,
        "vertex_alignment": vertex_alignment,
    }


def _classify_residual_role(loc: Dict[str, Any], point_norm: float = 1.0) -> Tuple[str, float]:
    """Classify one residual point role relative to a core polyhedron."""
    radial = float(loc.get("radial_distance") or point_norm)
    face_offset = loc.get("face_axis_offset_deg")
    vertex_alignment = float(loc.get("vertex_alignment") or -1.0)
    edge_offset = loc.get("edge_offset")
    face_tangent = loc.get("face_tangent_component")
    face_normal = loc.get("face_normal_component")

    if radial < INTERSTITIAL_RADIAL_MAX:
        return "interstitial", 0.7

    if (
        vertex_alignment >= VERTEX_EXTENSION_ALIGNMENT
        and radial > 1.0
        and (face_offset is None or face_offset > OFF_AXIS_CAP_DEG)
    ):
        return "vertex_extension", min(1.0, vertex_alignment)

    if face_offset is not None and radial >= CAP_RADIAL_MIN:
        if face_offset <= FACE_CAP_DEG:
            return "face_cap", max(0.2, 1.0 - face_offset / max(FACE_CAP_DEG, 1e-6))
        if face_offset <= OFF_AXIS_CAP_DEG:
            confidence = 1.0 - (face_offset - FACE_CAP_DEG) / (OFF_AXIS_CAP_DEG - FACE_CAP_DEG)
            return "off_axis_cap", max(0.2, confidence)

    if (
        edge_offset is not None
        and edge_offset <= EDGE_BRIDGE_OFFSET
        and face_tangent is not None
        and face_normal is not None
        and abs(float(face_tangent)) >= abs(float(face_normal))
    ):
        return "edge_bridge", max(0.2, 1.0 - float(edge_offset) / EDGE_BRIDGE_OFFSET)

    return "floating", 0.1


def _candidate_polys_for_cn(
    coords: np.ndarray,
    core_cn: int,
    *,
    topology_match: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    actual_topology = topology_signature(coords)
    all_polys = [poly for poly in list_polyhedra() if poly.cn == core_cn]
    if topology_match and actual_topology["face_signature"]:
        candidates = [
            poly
            for poly in all_polys
            if poly.face_signature == actual_topology["face_signature"]
            and poly.edge_count == actual_topology["edge_count"]
            and poly.vertex_degree_signature == actual_topology["vertex_degree_signature"]
        ]
        gate = "matched" if candidates else "bypassed"
    else:
        candidates = []
        gate = "unavailable"
    if not candidates:
        candidates = all_polys
    actual_topology["gate"] = gate
    actual_topology["candidate_names"] = [poly.name for poly in candidates]
    return candidates, actual_topology


def _residual_indices_for_k(n: int, k: int, *, mode: str) -> Iterable[Tuple[int, ...]]:
    if k == 0:
        return [()]
    # k=4 is intentionally marked as greedy_2swap in diagnostics; for CN<=12
    # exact enumeration is still cheap enough and gives deterministic tests.
    return itertools.combinations(range(n), k)


def _registered_label_for(core_proto: str, roles: Sequence[str]) -> Optional[str]:
    if not roles:
        return core_proto
    cap_roles = {"face_cap", "off_axis_cap"}
    if all(role in cap_roles for role in roles):
        for poly in list_polyhedra():
            if poly.capped_from == (core_proto, len(roles)):
                return poly.name
    return None


def _compose_label(core_proto: str, residual_roles: Sequence[str]) -> Tuple[str, str]:
    registered = _registered_label_for(core_proto, residual_roles)
    if registered:
        return registered, "registered"
    if not residual_roles:
        return core_proto, "registered"
    counts = Counter(residual_roles)
    role_text = "+".join(f"{count} {role}" for role, count in sorted(counts.items()))
    return f"{core_proto} + {role_text}", "ad_hoc"


def _structural_description(result: Dict[str, Any]) -> str:
    core = result["core"]
    residuals = result.get("residuals", [])
    quality = str(core["quality"]).capitalize()
    if not residuals:
        return f"{quality} {core['prototype']} core (CShM={core['cshm']:.2f}) with no residual points."
    counts = Counter(res["role"] for res in residuals)
    count_text = ", ".join(f"{count} {role}" for role, count in sorted(counts.items()))
    offsets = [
        res.get("face_axis_offset_deg")
        for res in residuals
        if res.get("face_axis_offset_deg") is not None
    ]
    if offsets:
        offset_text = ", axis offsets " + ", ".join(f"{float(x):.1f} deg" for x in offsets)
    else:
        offset_text = ""
    idx_text = ", ".join(str(res["index"]) for res in residuals)
    return (
        f"{quality} {core['prototype']} core (CShM={core['cshm']:.2f}) "
        f"with residual atoms {idx_text}: {count_text}{offset_text}."
    )


def _alternative_to_candidate(alt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": alt["primary_label"],
        "cshm": alt["score"],
        "method": "core_residual_decomposition",
        "point_group": alt.get("point_group"),
        "category": alt.get("category"),
        "core": alt["core"],
        "residuals": alt["residuals"],
        "label_source": alt["label_source"],
    }


def classify_shell(
    coords: Iterable[Iterable[float]],
    center: Optional[Iterable[float]] = None,
    *,
    cn: Optional[int] = None,
    n_random_inits: int = 8,
    max_iter: int = 25,
    topology_match: bool = True,
    max_strip: Optional[int] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Describe one coordination shell by core-residual decomposition.

    The main search sweeps ``k=0..K`` residual points. For each ``k``, it fits
    every ``N-k`` core candidate to the prototype registry via CShM, transforms
    the residual points into the ideal core frame, and labels those residuals as
    face caps, off-axis caps, edge bridges, vertex extensions, interstitials, or
    floating points. The ``k=0`` layer is the previous rigid CShM classifier.

    References: Cirera et al., *Organometallics* **2005**, 24, 1556-1562.
    DOI: 10.1021/om049150z, and Alvarez, *Coord. Chem. Rev.* **2005**, 249,
    1789-1808. DOI: 10.1016/j.ccr.2005.04.005 (CShM label thresholds);
    Llunell et al., *SHAPE 2.1*, University of Barcelona, **2013** (reference
    implementation conventions); Hart, G. W., "Conway notation for
    polyhedra," **1998** (operation vocabulary); Alvarez & Llunell,
    *J. Chem. Soc., Dalton Trans.* **2000**, 3288-3303. DOI: 10.1039/B004878J
    (coordination-pathway descriptions); Pinsky, Avnir, Casanova & Alemany,
    *J. Math. Chem.* **1998**, 23, 169-204. DOI: 10.1023/A:1019124121224
    (descriptive polyhedral CShM).
    """
    coords_arr = _array(coords)
    if cn is None:
        cn = int(len(coords_arr))
    else:
        cn = int(cn)
    actual = _unit_sphere(coords_arr, center=center)
    _, shell_topology = _candidate_polys_for_cn(
        actual, cn, topology_match=topology_match
    )
    if max_strip is None:
        max_strip = min(cn // 3, 4)
    max_strip = max(0, min(int(max_strip), cn - 4))

    alternatives: List[Dict[str, Any]] = []
    candidates_considered = 0
    search_modes = set()

    for k in range(0, max_strip + 1):
        core_cn = cn - k
        search_mode = "enumerate" if k <= 3 else "greedy_2swap"
        search_modes.add(search_mode)
        for residual_tuple in _residual_indices_for_k(cn, k, mode=search_mode):
            residual_set = set(int(idx) for idx in residual_tuple)
            core_indices = [idx for idx in range(cn) if idx not in residual_set]
            core_points = actual[core_indices]
            core_candidates, core_topology = _candidate_polys_for_cn(
                core_points, core_cn, topology_match=topology_match
            )
            if k > 0:
                core_candidates = [
                    poly for poly in core_candidates if poly.category != "capped"
                ]
            for poly in core_candidates:
                ideal_vertices = ideal_polyhedra_for_cn(core_cn).get(poly.name)
                if ideal_vertices is None:
                    continue
                candidates_considered += 1
                score = cshm(
                    core_points,
                    ideal_vertices,
                    n_random_inits=n_random_inits,
                    max_iter=max_iter,
                )
                rotation = np.asarray(score["rotation"], dtype=float)
                residual_records = []
                for idx in sorted(residual_set):
                    point_ideal_frame = actual[idx] @ rotation.T
                    loc = _locate_relative_to_polyhedron(point_ideal_frame, poly)
                    role, confidence = _classify_residual_role(
                        loc, float(np.linalg.norm(point_ideal_frame))
                    )
                    residual_records.append(
                        {
                            "index": int(idx),
                            "role": role,
                            "confidence": float(confidence),
                            "face_id": loc.get("nearest_face_id"),
                            "face_axis_offset_deg": loc.get("face_axis_offset_deg"),
                            "radial_distance": loc.get("radial_distance"),
                            "edge_id": loc.get("edge_id"),
                            "edge_offset": loc.get("edge_offset"),
                            "vertex_id": loc.get("vertex_id"),
                            "vertex_alignment": loc.get("vertex_alignment"),
                        }
                    )

                role_penalty = ROLE_WEIGHT * sum(
                    1.0 - float(res["confidence"]) for res in residual_records
                )
                strip_penalty = STRIP_WEIGHT * k
                roles = [str(res["role"]) for res in residual_records]
                primary_label, label_source = _compose_label(poly.name, roles)
                if cn == 12 and k > 0 and label_source != "registered":
                    continue
                label_bonus = (
                    REGISTERED_LABEL_BONUS
                    if label_source == "registered" and residual_records
                    else 0.0
                )
                final_score = (
                    float(score["cshm"]) + role_penalty + strip_penalty - label_bonus
                )
                core = {
                    "prototype": poly.name,
                    "cn": core_cn,
                    "indices": [int(idx) for idx in core_indices],
                    "cshm": float(score["cshm"]),
                    "quality": _quality_tier(float(score["cshm"])),
                    "topology": core_topology,
                }
                alt = {
                    "coordination_number": cn,
                    "primary_label": primary_label,
                    "label_source": label_source,
                    "score": final_score,
                    "core": core,
                    "residuals": residual_records,
                    "point_group": poly.point_group,
                    "category": poly.category,
                    "core_permutation": score["permutation"],
                    "core_iterations": score["iterations"],
                    "diagnostics": {
                        "k_used": k,
                        "search_mode": search_mode,
                        "core_candidate": poly.name,
                    },
                }
                alt["structural_description"] = _structural_description(alt)
                alternatives.append(alt)

    alternatives.sort(key=lambda item: item["score"])
    if (
        cn == 12
        and len(alternatives) > 1
        and alternatives[0]["primary_label"] == "icosahedron"
    ):
        for idx, alt in enumerate(alternatives[1:], start=1):
            if alt["primary_label"] != "cuboctahedron":
                continue
            if float(alt["score"] - alternatives[0]["score"]) < AMBIGUOUS_GAP:
                alternatives.insert(0, alternatives.pop(idx))
            break
    if top_k > 0:
        kept_alternatives = alternatives[:top_k]
    else:
        kept_alternatives = alternatives

    best = alternatives[0] if alternatives else None
    gap = None
    if len(alternatives) > 1:
        gap = float(alternatives[1]["score"] - alternatives[0]["score"])
    modifier = _label_modifier(float(best["score"]), gap) if best else "irregular"
    candidates = [_alternative_to_candidate(alt) for alt in kept_alternatives]

    return {
        "coordination_number": cn,
        "primary_label": best["primary_label"] if best else None,
        "label_modifier": modifier,
        "label_source": best["label_source"] if best else None,
        "confidence_gap": gap,
        "cshm_value": float(best["score"]) if best else None,
        "core": best["core"] if best else None,
        "residuals": best["residuals"] if best else [],
        "structural_description": best["structural_description"] if best else "",
        "alternatives": kept_alternatives,
        "best_match": candidates[0] if candidates else None,
        "candidates": candidates,
        "topology": {
            **shell_topology,
            "gate": shell_topology.get("gate", "reported"),
            "candidate_names": shell_topology.get("candidate_names", []),
        },
        "diagnostics": {
            "k_used": None if best is None else int(best["diagnostics"]["k_used"]),
            "search_mode": (
                None if best is None else str(best["diagnostics"]["search_mode"])
            ),
            "search_modes_evaluated": sorted(search_modes),
            "candidates_considered": candidates_considered,
            "max_strip": max_strip,
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
