"""
Packing-shell geometry analysis.

The routines here infer and score local shells from distance-ranked points and
convex-hull geometry. This is distinct from bond-graph chemical environment
analysis, which lives in :mod:`molcrys_kit.analysis.chemical_env`.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from ..structures.polyhedra import ideal_polyhedra_for_cn

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - optional dependency
    ConvexHull = None


DEFAULT_CENTROID_OFFSET_FRAC = 0.15


def _array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def hull_encloses_center(
    coords: np.ndarray,
    center: np.ndarray,
    *,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
    face_tol: float = 1e-3,
) -> bool:
    """Return true when ``center`` is centred inside the hull of ``coords``.

    A packing-shell polyhedron is meaningful only when the central point sits
    roughly at the middle of the neighbour cage. The convex hull checks
    topological enclosure; the centroid-offset check rejects partial shells
    where the center is pressed against one side of the cage.
    """
    coords = np.asarray(coords, dtype=float)
    center = np.asarray(center, dtype=float)
    if len(coords) < 4 or ConvexHull is None:
        return False
    try:
        hull = ConvexHull(coords)
    except Exception:
        return False
    plane_vals = hull.equations[:, :3] @ center + hull.equations[:, 3]
    if np.any(plane_vals > face_tol):
        return False
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    mean_radius = float(np.mean(radii)) if len(radii) else 0.0
    if mean_radius < 1e-6:
        return True
    offset = float(np.linalg.norm(center - centroid))
    return offset <= centroid_offset_frac * mean_radius


def detect_coordination_number(
    distances: Sequence[float],
    fallback_max: int = None,
    *,
    coords: Sequence[Sequence[float]] = None,
    center: Sequence[float] = None,
    enforce_enclosure: bool = True,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
    cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    """Choose a shell coordination number from ordered neighbour distances.

    Three modes are supported and selected by the parameters:

    * ``mode="cutoff"`` — when ``cutoff`` is given, the shell is exactly the
      neighbours whose distance is ``<= cutoff``. The largest-gap heuristic is
      bypassed entirely. ``enforce_enclosure`` is treated as a *diagnostic*
      only here (the returned ``enclosed`` flag still reports whether the
      cutoff shell wraps the center), because the user-supplied ``cutoff``
      acts as a hard upper bound on CN. ``cutoff`` therefore overrides any
      gap-based selection — useful when you want a clean A--B coordination
      count and do not want the algorithm to silently expand the shell.

    * ``mode="gap+enclosure"`` (default when ``cutoff`` is None) — pick the
      largest gap in the sorted distances as the primary CN; if ``coords``
      and ``center`` are provided and ``enforce_enclosure`` is true, expand
      the shell monotonically until its convex hull wraps the center. This
      is the historical behaviour and is preserved for backward
      compatibility.

    * ``mode="gap"`` — pick the largest gap and stop. Use this by passing
      ``enforce_enclosure=False``: the returned CN reflects only the
      A--B distance distribution and is **not** padded out to make the hull
      enclose the central atom. Set this when you want to study an honest
      A--B shell without the hull post-correction (e.g. square-planar or
      open one-sided coordination cages).

    Parameters
    ----------
    distances
        Distances from the central atom to candidate neighbours. Will be
        sorted internally; ``coords`` (if supplied) must already be in the
        same ascending-distance order.
    fallback_max
        Optional upper bound applied to the final CN.
    coords, center
        Cartesian coordinates of the candidate neighbours and the center,
        used for hull-based diagnostics.
    enforce_enclosure
        When ``cutoff`` is None: expand the gap shell until the hull wraps
        the center. When ``cutoff`` is given: only used as a diagnostic flag
        (no expansion past the cutoff).
    centroid_offset_frac
        Allowed centroid-to-center offset for :func:`hull_encloses_center`.
    cutoff
        Hard radial cutoff in the same units as ``distances``. When set,
        forces ``mode="cutoff"`` and overrides the gap heuristic.

    Returns
    -------
    dict
        Always contains ``coordination_number``, ``mode``, ``primary_gap_cn``,
        ``sorted_distances``, ``gaps``, ``enclosed``, ``enclosure_expanded``,
        ``cutoff``, ``gap_index`` and ``gap_value``. Only ``coordination_number``
        and ``mode`` are guaranteed-meaningful in cutoff mode; the gap-related
        fields remain populated for diagnostics so callers can compare the
        chosen cutoff CN against the gap-derived one.
    """
    sorted_distances = np.sort(np.array(distances, dtype=float))
    n = len(sorted_distances)
    coords_arr = np.asarray(coords, dtype=float) if coords is not None else None
    center_arr = np.asarray(center, dtype=float) if center is not None else None

    def _empty_gap_payload() -> Dict[str, Any]:
        return {"primary_gap_cn": 0, "gap_index": None, "gap_value": None, "gaps": []}

    if n == 0:
        payload = {
            "coordination_number": 0,
            "mode": "cutoff" if cutoff is not None else "gap+enclosure",
            "enclosed": False,
            "enclosure_expanded": False,
            "sorted_distances": [],
            "cutoff": float(cutoff) if cutoff is not None else None,
        }
        payload.update(_empty_gap_payload())
        return payload

    gaps = np.diff(sorted_distances) if n > 1 else np.array([], dtype=float)
    primary_cn = int(np.argmax(gaps) + 1) if gaps.size else n

    # --- Cutoff mode: hard radial cutoff overrides gap selection ---
    if cutoff is not None:
        cutoff_val = float(cutoff)
        cn = int(np.sum(sorted_distances <= cutoff_val))
        if fallback_max is not None:
            cn = min(cn, int(fallback_max))
        cn = max(0, cn)

        enclosed = False
        if (
            cn >= 4
            and coords_arr is not None
            and center_arr is not None
            and len(coords_arr) >= cn
        ):
            enclosed = hull_encloses_center(
                coords_arr[:cn],
                center_arr,
                centroid_offset_frac=centroid_offset_frac,
            )

        gap_index = (cn - 1) if (cn >= 1 and gaps.size) else None
        if gap_index is not None:
            gap_index = min(gap_index, len(gaps) - 1)
        gap_value = float(gaps[gap_index]) if gap_index is not None else None
        return {
            "coordination_number": cn,
            "mode": "cutoff",
            "primary_gap_cn": primary_cn,
            "gap_index": gap_index,
            "gap_value": gap_value,
            "sorted_distances": sorted_distances.tolist(),
            "gaps": gaps.tolist(),
            "enclosed": enclosed,
            "enclosure_expanded": False,
            "cutoff": cutoff_val,
        }

    # --- Gap (+ optional enclosure expansion) mode ---
    if n == 1:
        return {
            "coordination_number": 1,
            "mode": "gap",
            "gap_index": 0,
            "gap_value": 0.0,
            "enclosed": False,
            "enclosure_expanded": False,
            "primary_gap_cn": 1,
            "sorted_distances": sorted_distances.tolist(),
            "gaps": [],
            "cutoff": None,
        }

    cn = primary_cn
    enclosed = False
    expanded = False

    if (
        enforce_enclosure
        and coords_arr is not None
        and center_arr is not None
        and len(coords_arr) >= 4
    ):
        if hull_encloses_center(
            coords_arr[:primary_cn],
            center_arr,
            centroid_offset_frac=centroid_offset_frac,
        ):
            enclosed = True
        else:
            for candidate_cn in range(primary_cn + 1, len(coords_arr) + 1):
                if candidate_cn < 4:
                    continue
                if hull_encloses_center(
                    coords_arr[:candidate_cn],
                    center_arr,
                    centroid_offset_frac=centroid_offset_frac,
                ):
                    cn = candidate_cn
                    enclosed = True
                    expanded = True
                    break

    if fallback_max is not None:
        cn = min(cn, int(fallback_max))
    cn = max(1, cn)
    gap_index = min(cn - 1, len(gaps) - 1) if len(gaps) > 0 else None
    gap_value = float(gaps[gap_index]) if gap_index is not None else None
    return {
        "coordination_number": cn,
        "mode": "gap+enclosure" if enforce_enclosure else "gap",
        "gap_index": gap_index,
        "gap_value": gap_value,
        "sorted_distances": sorted_distances.tolist(),
        "gaps": gaps.tolist(),
        "primary_gap_cn": primary_cn,
        "enclosed": enclosed,
        "enclosure_expanded": expanded,
        "cutoff": None,
    }


def compute_angular_signature(
    shell_coords: Iterable[Iterable[float]],
    center: Iterable[float] = None,
) -> Dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) == 0:
        return {"angles": [], "sorted_angles": [], "count": 0}
    center_vec = np.zeros(3, dtype=float) if center is None else np.array(center, dtype=float)
    vectors = coords - center_vec
    norms = np.linalg.norm(vectors, axis=1)
    angles = []
    for i, j in itertools.combinations(range(len(vectors)), 2):
        if norms[i] < 1e-8 or norms[j] < 1e-8:
            continue
        cosang = np.clip(np.dot(vectors[i], vectors[j]) / (norms[i] * norms[j]), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosang))))
    angles.sort()
    return {"angles": angles, "sorted_angles": angles, "count": len(angles)}


def angular_rmsd_vs_ideals(
    shell_coords: Iterable[Iterable[float]],
    center: Iterable[float] = None,
) -> Dict[str, Any]:
    coords = _array(shell_coords)
    cn = int(len(coords))
    signature = compute_angular_signature(coords, center=center)
    actual = np.array(signature["sorted_angles"], dtype=float)
    results = []
    for name, ideal in ideal_polyhedra_for_cn(cn).items():
        ideal_signature = np.array(compute_angular_signature(ideal)["sorted_angles"], dtype=float)
        size = min(len(actual), len(ideal_signature))
        if size == 0:
            rmsd = float("inf")
        else:
            diff = actual[:size] - ideal_signature[:size]
            rmsd = float(np.sqrt(np.mean(diff * diff)))
        results.append({"name": name, "angular_rmsd": rmsd})
    results.sort(key=lambda item: item["angular_rmsd"])
    return {
        "coordination_number": cn,
        "results": results,
        "best_match": results[0] if results else None,
    }


def planarity_analysis(
    shell_coords: Iterable[Iterable[float]],
    group_size: int = 5,
) -> Dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < group_size:
        return {"best_rms": None, "best_indices": [], "group_size": group_size}
    best_rms = float("inf")
    best_indices = None
    combo_iter = itertools.combinations(range(len(coords)), group_size)
    batch_size = 4096
    while True:
        batch = list(itertools.islice(combo_iter, batch_size))
        if not batch:
            break
        idx = np.array(batch, dtype=int)
        subsets = coords[idx]
        centered = subsets - subsets.mean(axis=1, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normals = vh[:, -1, :]
        distances = np.einsum("bgi,bi->bg", centered, normals)
        rms_values = np.sqrt(np.mean(distances * distances, axis=1))
        batch_pos = int(np.argmin(rms_values))
        rms = float(rms_values[batch_pos])
        if rms < best_rms:
            best_rms = rms
            best_indices = tuple(int(x) for x in idx[batch_pos])
    return {
        "best_rms": best_rms if best_indices is not None else None,
        "best_indices": list(best_indices or []),
        "group_size": group_size,
    }


def detect_prism_vs_antiprism(shell_coords: Iterable[Iterable[float]]) -> Dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < 10:
        return {"classification": None, "twist_deg": None}
    z_sorted = np.argsort(coords[:, 2])
    bottom = coords[z_sorted[:5]]
    top = coords[z_sorted[-5:]]
    top_angles = np.sort(np.degrees(np.arctan2(top[:, 1], top[:, 0])) % 360.0)
    bottom_angles = np.sort(np.degrees(np.arctan2(bottom[:, 1], bottom[:, 0])) % 360.0)
    shifts = []
    for angle_top, angle_bottom in zip(top_angles, bottom_angles):
        delta = (angle_top - angle_bottom + 180.0) % 360.0 - 180.0
        shifts.append(abs(delta))
    twist = float(np.mean(shifts))
    classification = "antiprism" if twist > 18.0 else "prism"
    return {"classification": classification, "twist_deg": twist}


# Default neighbour search radius for find_polyhedra when neither ``cutoff``
# nor ``search_cutoff`` is supplied.  6.0 Å covers typical CN<=12 metal-ligand
# shells (M-X up to ~3.2 Å plus a margin) without explosive neighbour pair
# counts.  Override via the ``search_cutoff`` argument when needed.
DEFAULT_POLYHEDRON_SEARCH_CUTOFF = 6.0


def _structure_to_atoms(structure):
    """Coerce ``MolecularCrystal``-like inputs to an ASE ``Atoms`` object."""
    if hasattr(structure, "to_ase") and callable(structure.to_ase):
        return structure.to_ase()
    return structure


def find_polyhedra(
    structure,
    central: str,
    ligand: str,
    *,
    cutoff: Optional[float] = None,
    search_cutoff: Optional[float] = None,
    enforce_enclosure: bool = True,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
    fallback_max: Optional[int] = None,
    score_shape: bool = False,
    central_indices: Optional[Sequence[int]] = None,
) -> Sequence[Dict[str, Any]]:
    """Enumerate A--B coordination polyhedra in a periodic structure.

    For every atom whose chemical symbol equals ``central``, find the
    surrounding atoms whose symbol equals ``ligand`` (PBC-aware), then run
    :func:`detect_coordination_number` on that ordered shell.

    The function strictly considers **only** A--B distances: atoms of any
    other element (including other A atoms that happen to fall inside the
    candidate hull) are never added to the shell. This is the explicit
    bypass for the gap+enclosure expansion described in the docstring of
    :func:`detect_coordination_number`.

    Parameters
    ----------
    structure
        ``molcrys_kit.structures.crystal.MolecularCrystal`` or any object
        with a ``to_ase()`` method, or an ``ase.Atoms`` directly.
    central, ligand
        Chemical symbols of the central (A) and ligand (B) species. They
        may be equal (homo-A--A polyhedra are supported).
    cutoff
        Hard radial cutoff in Å. When set, the polyhedron is exactly the
        B atoms within ``cutoff`` of A — no gap detection, no hull
        expansion (passed through to :func:`detect_coordination_number`
        as ``cutoff``).
    search_cutoff
        Maximum A–B distance considered when collecting candidate
        neighbours via :func:`ase.neighborlist.neighbor_list`. Defaults
        to ``cutoff`` if given, otherwise to
        :data:`DEFAULT_POLYHEDRON_SEARCH_CUTOFF`.
    enforce_enclosure
        Forwarded to :func:`detect_coordination_number`. Set ``False`` to
        get the gap-only CN without expanding to wrap the central atom.
        Has no effect on the chosen CN in cutoff mode.
    centroid_offset_frac, fallback_max
        Forwarded to :func:`detect_coordination_number`.
    score_shape
        When true, also score each shell against the ideal-polyhedra
        catalogue using :func:`angular_rmsd_vs_ideals`. Adds a
        ``best_match`` field per result.
    central_indices
        Optional iterable of atom indices to restrict the search to a
        subset of A atoms (useful when only a few sites are interesting).

    Returns
    -------
    list of dict
        One entry per A atom. Each dict contains:

        * ``center_index``: index in the source ``Atoms`` object
        * ``center_symbol``: ``central``
        * ``center_position``: Cartesian position (length-3 list)
        * ``shell_indices``: ligand atom indices, ordered by distance
        * ``shell_offsets``: PBC image offsets for each shell atom
        * ``shell_coords``: Cartesian positions with PBC offset applied
        * ``shell_distances``: A--B distances, ascending
        * ``coordination_number`` / ``mode`` / ``primary_gap_cn`` /
          ``enclosed`` / ``enclosure_expanded`` / ``cutoff`` /
          ``gap_index`` / ``gap_value`` (from
          :func:`detect_coordination_number`)
        * ``best_match`` (only when ``score_shape=True``): best ideal
          polyhedron and its angular RMSD, or ``None`` for low CN.
    """
    try:
        from ase.neighborlist import neighbor_list
    except ImportError as exc:  # pragma: no cover - ase is a hard dep
        raise ImportError(
            "find_polyhedra requires ASE; install with 'pip install ase'."
        ) from exc

    atoms = _structure_to_atoms(structure)

    if cutoff is not None and search_cutoff is None:
        search_cutoff = float(cutoff)
    if search_cutoff is None:
        search_cutoff = float(DEFAULT_POLYHEDRON_SEARCH_CUTOFF)
    if search_cutoff <= 0:
        raise ValueError(f"search_cutoff must be positive, got {search_cutoff!r}.")

    symbols = list(atoms.get_chemical_symbols())
    positions = np.asarray(atoms.get_positions(), dtype=float)

    central_index_set = (
        {int(i) for i in central_indices} if central_indices is not None else None
    )

    central_atoms = [
        i for i, sym in enumerate(symbols)
        if sym == central
        and (central_index_set is None or i in central_index_set)
    ]
    if not central_atoms:
        return []

    i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", atoms, cutoff=float(search_cutoff))

    shells: Dict[int, list] = {idx: [] for idx in central_atoms}
    for src, dst, dist, vec in zip(i_arr, j_arr, d_arr, D_arr):
        src_i = int(src)
        if src_i not in shells:
            continue
        if symbols[int(dst)] != ligand:
            continue
        if central == ligand and int(dst) == src_i:
            continue
        shells[src_i].append((float(dist), int(dst), np.asarray(vec, dtype=float)))

    results: list = []
    for center_idx in central_atoms:
        entries = shells.get(center_idx, [])
        entries.sort(key=lambda item: item[0])

        center_pos = positions[center_idx]
        if entries:
            distances = [e[0] for e in entries]
            ligand_indices = [e[1] for e in entries]
            shell_vectors = np.array([e[2] for e in entries], dtype=float)
            shell_coords = center_pos[None, :] + shell_vectors
            offsets = []
            for shell_idx, vec in zip(ligand_indices, shell_vectors):
                base_vec = positions[shell_idx] - center_pos
                offsets.append(np.round(vec - base_vec, 6).tolist())
        else:
            distances = []
            ligand_indices = []
            shell_coords = np.empty((0, 3), dtype=float)
            offsets = []

        cn_info = detect_coordination_number(
            distances,
            fallback_max=fallback_max,
            coords=shell_coords if len(shell_coords) else None,
            center=center_pos,
            enforce_enclosure=enforce_enclosure,
            centroid_offset_frac=centroid_offset_frac,
            cutoff=cutoff,
        )

        cn = int(cn_info["coordination_number"])
        kept_distances = distances[:cn]
        kept_indices = ligand_indices[:cn]
        kept_offsets = offsets[:cn]
        kept_coords = (
            shell_coords[:cn].tolist() if len(shell_coords) else []
        )

        record = {
            "center_index": int(center_idx),
            "center_symbol": central,
            "center_position": center_pos.tolist(),
            "shell_indices": kept_indices,
            "shell_offsets": kept_offsets,
            "shell_coords": kept_coords,
            "shell_distances": kept_distances,
            "coordination_number": cn,
            "mode": cn_info["mode"],
            "primary_gap_cn": cn_info["primary_gap_cn"],
            "gap_index": cn_info["gap_index"],
            "gap_value": cn_info["gap_value"],
            "enclosed": cn_info["enclosed"],
            "enclosure_expanded": cn_info["enclosure_expanded"],
            "cutoff": cn_info["cutoff"],
            "search_cutoff": float(search_cutoff),
        }

        if score_shape:
            if cn >= 4 and len(kept_coords) == cn:
                record["best_match"] = angular_rmsd_vs_ideals(
                    kept_coords, center=center_pos
                )["best_match"]
            else:
                record["best_match"] = None

        results.append(record)

    return results


__all__ = [
    "DEFAULT_CENTROID_OFFSET_FRAC",
    "DEFAULT_POLYHEDRON_SEARCH_CUTOFF",
    "angular_rmsd_vs_ideals",
    "compute_angular_signature",
    "detect_coordination_number",
    "detect_prism_vs_antiprism",
    "find_polyhedra",
    "hull_encloses_center",
    "planarity_analysis",
]
