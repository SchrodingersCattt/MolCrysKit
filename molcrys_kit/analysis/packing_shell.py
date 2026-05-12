"""
Packing-shell geometry analysis.

The routines here infer and score local shells from distance-ranked points and
convex-hull geometry. This is distinct from bond-graph chemical environment
analysis, which lives in :mod:`molcrys_kit.analysis.chemical_env`.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..structures.polyhedra import ideal_polyhedra_for_cn
from .formula_moiety import (
    Fragment,
    heavy_signature,
    parse_moiety_string,
)

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
    """Score a shell by legacy sorted-angle RMSD.

    Deprecated for naming; use :func:`molcrys_kit.analysis.shape.classify_shell`
    instead. This function is retained as a low-level diagnostic.
    """
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


# Default neighbour search radius for find_polyhedra(level="atom") when neither
# ``cutoff`` nor ``search_cutoff`` is supplied.  6.0 Å covers typical CN<=12
# metal-ligand shells (M-X up to ~3.2 Å plus a margin) without explosive
# neighbour pair counts.  Override via the ``search_cutoff`` argument when
# needed.
DEFAULT_POLYHEDRON_SEARCH_CUTOFF = 6.0

# Default neighbour search radius for find_polyhedra(level="molecule").  At the
# centroid--centroid scale, A and B molecules typically sit 6--10 Å apart in
# ABX3 / ABX4 / A2BX5 hybrid perovskites, so 12 Å comfortably captures CN<=12
# first shells without inflating the candidate count by replicating distant
# cells unnecessarily.  Override via the ``search_cutoff`` argument when
# needed.
DEFAULT_MOLECULAR_SEARCH_CUTOFF = 12.0

# Centre choice for level="molecule".  ``centroid`` is the unweighted mean of
# all atom positions (matches MolCrysKit's ``CrystalMolecule.get_centroid``),
# ``com`` is the mass-weighted centre of mass, and ``heavy_centroid`` is the
# unweighted mean over non-hydrogen atoms (matches publications that draw
# packing shells from heavy-atom skeletons of organic cations).
_MOLECULE_CENTRE_KINDS = ("centroid", "com", "heavy_centroid")


def _structure_to_atoms(structure):
    """Coerce ``MolecularCrystal``-like inputs to an ASE ``Atoms`` object."""
    if hasattr(structure, "to_ase") and callable(structure.to_ase):
        return structure.to_ase()
    return structure


def _parse_moiety_signature(text: str) -> Tuple[Tuple[str, int], ...]:
    """Parse a single moiety string (e.g. ``"N H4"``, ``"Cl O4"``, ``"I"``).

    The returned heavy-atom signature is what
    :func:`molcrys_kit.analysis.formula_moiety.heavy_signature` returns for
    the parsed composition, so a molecule and a moiety string match iff the
    molecule's non-H elements are the same multiset.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"level='molecule' requires a non-empty moiety string, got {text!r}."
        )
    fragments = parse_moiety_string(text)
    if not fragments:
        raise ValueError(
            f"Could not parse moiety string {text!r}. Expected e.g. "
            "'N H4', 'Cl O4', 'C2 H10 N2 2+', or a single element symbol."
        )
    if len(fragments) > 1:
        raise ValueError(
            f"Moiety string {text!r} contains {len(fragments)} fragments; "
            "find_polyhedra(level='molecule') takes exactly one fragment per "
            "central/ligand argument."
        )
    return heavy_signature(fragments[0].composition)


def _format_molecule_formula(symbols: Sequence[str]) -> str:
    """Render a Hill-ish (C, H, then alphabetical) formula label."""
    counts = Counter(symbols)
    if not counts:
        return ""
    ordered: List[str] = []
    if "C" in counts:
        ordered.append("C")
    if "H" in counts:
        ordered.append("H")
    ordered.extend(sorted(e for e in counts if e not in {"C", "H"}))
    parts: List[str] = []
    for element in ordered:
        count = counts[element]
        parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def _molecule_centre(molecule, kind: str) -> np.ndarray:
    """Return the Cartesian centre of ``molecule`` according to ``kind``."""
    positions = np.asarray(molecule.get_positions(), dtype=float)
    if positions.size == 0:
        raise ValueError("Cannot compute molecule centre: empty molecule.")
    if kind == "centroid":
        return positions.mean(axis=0)
    if kind == "com":
        masses = np.asarray(molecule.get_masses(), dtype=float)
        total = float(masses.sum())
        if total <= 0:
            return positions.mean(axis=0)
        return np.average(positions, axis=0, weights=masses)
    if kind == "heavy_centroid":
        symbols = list(molecule.get_chemical_symbols())
        mask = np.array([s != "H" for s in symbols], dtype=bool)
        if not mask.any():
            return positions.mean(axis=0)
        return positions[mask].mean(axis=0)
    raise ValueError(
        f"center_kind must be one of {_MOLECULE_CENTRE_KINDS!r}, got {kind!r}."
    )


def _lattice_translations(lattice: np.ndarray, search_radius: float) -> np.ndarray:
    """Enumerate every lattice translation whose Cartesian length is
    ``<= search_radius``.

    Uses each axis's family-of-planes perpendicular spacing
    (``V / |b × c|`` and cyclic permutations) as the per-axis enumeration
    bound. That is the strictest envelope that still captures every
    in-radius translation for an arbitrary triclinic lattice, including
    elongated and oblique cells where ``max(|a|, |b|, |c|)`` would
    silently miss diagonal images (``|t| = sqrt(2)·a`` for a cubic cell,
    etc.). Returns the translations as Cartesian Å vectors.
    """
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice must be a 3x3 matrix, got shape {lattice.shape}.")
    a, b, c = lattice[0], lattice[1], lattice[2]
    volume = abs(float(np.dot(a, np.cross(b, c))))
    if volume <= 0:
        raise ValueError("lattice vectors must be linearly independent.")
    d_a = volume / float(np.linalg.norm(np.cross(b, c)))
    d_b = volume / float(np.linalg.norm(np.cross(c, a)))
    d_c = volume / float(np.linalg.norm(np.cross(a, b)))
    n_a = int(np.ceil(search_radius / d_a))
    n_b = int(np.ceil(search_radius / d_b))
    n_c = int(np.ceil(search_radius / d_c))
    translations: List[np.ndarray] = []
    for ia in range(-n_a, n_a + 1):
        for ib in range(-n_b, n_b + 1):
            for ic in range(-n_c, n_c + 1):
                t = ia * a + ib * b + ic * c
                if np.linalg.norm(t) <= search_radius:
                    translations.append(t)
    return np.asarray(translations, dtype=float)


def _find_polyhedra_atom_level(
    structure,
    central: str,
    ligand: str,
    *,
    cutoff: Optional[float],
    search_cutoff: Optional[float],
    enforce_enclosure: bool,
    centroid_offset_frac: float,
    fallback_max: Optional[int],
    score_shape: bool,
    central_indices: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
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

    results: List[Dict[str, Any]] = []
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
            "level": "atom",
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


def _find_polyhedra_molecule_level(
    structure,
    central: str,
    ligand: str,
    *,
    center_kind: str,
    cutoff: Optional[float],
    search_cutoff: Optional[float],
    hard_cutoff: Optional[float],
    enforce_enclosure: bool,
    centroid_offset_frac: float,
    fallback_max: Optional[int],
    score_shape: bool,
    central_indices: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    if center_kind not in _MOLECULE_CENTRE_KINDS:
        raise ValueError(
            f"center_kind must be one of {_MOLECULE_CENTRE_KINDS!r}, "
            f"got {center_kind!r}."
        )
    molecules = getattr(structure, "molecules", None)
    lattice = getattr(structure, "lattice", None)
    if molecules is None or lattice is None:
        raise TypeError(
            "find_polyhedra(level='molecule') requires a MolecularCrystal-like "
            "object with .molecules and .lattice; got "
            f"{type(structure).__name__}. Build one via "
            "molcrys_kit.io.cif.read_mol_crystal()."
        )
    molecules = list(molecules)
    if not molecules:
        return []

    central_sig = _parse_moiety_signature(central)
    ligand_sig = _parse_moiety_signature(ligand)

    # On the molecule level, ``cutoff`` is the candidate **search radius**
    # (the value fed into the lattice-translation enumerator and the
    # ``dists <= radius`` mask). ``search_cutoff`` is a non-deprecated
    # synonym kept for callers who prefer the more self-documenting name;
    # giving both raises immediately to avoid silently picking one. The
    # "fill the ball" historical behaviour is opted into separately via
    # ``hard_cutoff``, which is the value forwarded to
    # :func:`detect_coordination_number` as ``cutoff=``.
    if cutoff is not None and search_cutoff is not None:
        raise ValueError(
            "Pass only one of cutoff= or search_cutoff= on level='molecule'; "
            "they are synonyms (both denote the candidate search radius)."
        )
    radius = cutoff if cutoff is not None else search_cutoff
    if radius is None:
        radius = float(DEFAULT_MOLECULAR_SEARCH_CUTOFF)
    else:
        radius = float(radius)
    if radius <= 0:
        raise ValueError(
            f"cutoff/search_cutoff must be positive, got {radius!r}."
        )
    if hard_cutoff is not None:
        hard_cutoff = float(hard_cutoff)
        if hard_cutoff <= 0:
            raise ValueError(
                f"hard_cutoff must be positive when given, got {hard_cutoff!r}."
            )

    centres = np.array(
        [_molecule_centre(mol, center_kind) for mol in molecules], dtype=float
    )
    sigs: List[Tuple[Tuple[str, int], ...]] = []
    formulas: List[str] = []
    for mol in molecules:
        symbols = list(mol.get_chemical_symbols())
        sigs.append(heavy_signature(dict(Counter(s for s in symbols if s != "H"))))
        formulas.append(_format_molecule_formula(symbols))

    central_index_set = (
        {int(i) for i in central_indices} if central_indices is not None else None
    )

    central_mols = [
        i for i, sig in enumerate(sigs)
        if sig == central_sig
        and (central_index_set is None or i in central_index_set)
    ]
    ligand_mols = [i for i, sig in enumerate(sigs) if sig == ligand_sig]
    if not central_mols:
        return []

    # The translation set must cover every t such that
    # |c_ligand + t - c_central| <= radius for any (central, ligand)
    # pair. The triangle inequality bounds |t| <= radius +
    # |c_ligand - c_central|, so we enumerate to the worst-case pair
    # distance (diagonal of the axis-aligned bounding box of all centres).
    # Without this, centroids near opposite cell corners or in
    # unwrapped molecule reconstructions would silently miss
    # diagonal-image neighbours.
    if len(centres) > 1:
        centroid_extent = float(np.linalg.norm(np.ptp(centres, axis=0)))
    else:
        centroid_extent = 0.0
    translations = _lattice_translations(
        np.asarray(lattice, dtype=float),
        radius + centroid_extent,
    )

    results: List[Dict[str, Any]] = []
    homo_pair = central_sig == ligand_sig
    for center_mol_idx in central_mols:
        center_pos = centres[center_mol_idx]

        # Generate every image of every ligand molecule, vectorised
        if ligand_mols:
            lig_centres = centres[ligand_mols]                       # (L, 3)
            cand_pts = lig_centres[:, None, :] + translations[None, :, :]  # (L, T, 3)
            cand_vecs = cand_pts - center_pos                       # (L, T, 3)
            dists = np.linalg.norm(cand_vecs, axis=-1)              # (L, T)
            mask = dists <= radius
            if homo_pair:
                # Exclude the self image (centroid of the same molecule at t=0)
                self_mask = np.zeros_like(mask, dtype=bool)
                same = [pos for pos, idx in enumerate(ligand_mols) if idx == center_mol_idx]
                if same:
                    zero_t = int(np.argmin(np.linalg.norm(translations, axis=1)))
                    for pos in same:
                        self_mask[pos, zero_t] = True
                mask &= ~self_mask
            # Always drop numerically coincident centres (e.g. PBC duplicates
            # that survived the lattice prune for a centroid sitting on a face)
            mask &= dists > 1e-6
            rows, cols = np.where(mask)
        else:
            rows = cols = np.empty(0, dtype=int)

        if rows.size:
            order = np.argsort(dists[rows, cols])
            sorted_dists = dists[rows, cols][order]
            sorted_pts = cand_pts[rows, cols][order]
            sorted_lig_mols = [int(ligand_mols[r]) for r in rows[order]]
            sorted_offsets = [translations[c].tolist() for c in cols[order]]
        else:
            sorted_dists = np.empty(0, dtype=float)
            sorted_pts = np.empty((0, 3), dtype=float)
            sorted_lig_mols = []
            sorted_offsets = []

        cn_info = detect_coordination_number(
            sorted_dists.tolist(),
            fallback_max=fallback_max,
            coords=sorted_pts.tolist() if sorted_pts.size else None,
            center=center_pos.tolist(),
            enforce_enclosure=enforce_enclosure,
            centroid_offset_frac=centroid_offset_frac,
            cutoff=hard_cutoff,
        )

        cn = int(cn_info["coordination_number"])
        kept_dists = sorted_dists[:cn].tolist()
        kept_pts = sorted_pts[:cn].tolist()
        kept_lig_mols = sorted_lig_mols[:cn]
        kept_offsets = sorted_offsets[:cn]

        record = {
            "level": "molecule",
            "center_molecule_index": int(center_mol_idx),
            "center_formula": formulas[center_mol_idx],
            "center_position": center_pos.tolist(),
            "center_kind": center_kind,
            "shell_molecule_indices": kept_lig_mols,
            "shell_formula": formulas[ligand_mols[0]] if ligand_mols else "",
            "shell_offsets": kept_offsets,
            "shell_coords": kept_pts,
            "shell_distances": kept_dists,
            "coordination_number": cn,
            "mode": cn_info["mode"],
            "primary_gap_cn": cn_info["primary_gap_cn"],
            "gap_index": cn_info["gap_index"],
            "gap_value": cn_info["gap_value"],
            "enclosed": cn_info["enclosed"],
            "enclosure_expanded": cn_info["enclosure_expanded"],
            # ``record["cutoff"]`` echoes what detect_coordination_number
            # received, kept for BC.  Downstream code that wants to
            # inspect "was a hard cap applied?" should read the
            # explicit ``record["hard_cutoff"]`` field below.
            "cutoff": cn_info["cutoff"],
            "search_cutoff": radius,
            "hard_cutoff": hard_cutoff,
        }

        if score_shape:
            if cn >= 4 and len(kept_pts) == cn:
                record["best_match"] = angular_rmsd_vs_ideals(
                    kept_pts, center=center_pos
                )["best_match"]
            else:
                record["best_match"] = None

        results.append(record)

    return results


def find_polyhedra(
    structure,
    central: str,
    ligand: str,
    *,
    level: str = "atom",
    center_kind: str = "centroid",
    cutoff: Optional[float] = None,
    search_cutoff: Optional[float] = None,
    hard_cutoff: Optional[float] = None,
    enforce_enclosure: bool = True,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
    fallback_max: Optional[int] = None,
    score_shape: bool = False,
    central_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Enumerate A--B coordination polyhedra in a periodic structure.

    Two levels of A/B identity are supported via the ``level`` argument:

    * ``level="atom"`` (default, fully backward compatible) — for every
      atom whose chemical symbol equals ``central``, find the surrounding
      atoms whose symbol equals ``ligand`` (PBC-aware via ASE), then run
      :func:`detect_coordination_number` on that ordered shell. The function
      strictly considers **only** A--B distances: atoms of any other element
      (including other A atoms that happen to fall inside the candidate
      hull) are never added to the shell. This is the explicit bypass for
      the gap+enclosure expansion described in the docstring of
      :func:`detect_coordination_number`.

    * ``level="molecule"`` — for every CrystalMolecule whose heavy-atom
      composition matches the ``central`` moiety string, find the
      surrounding CrystalMolecules whose heavy composition matches the
      ``ligand`` moiety string, using molecule centroids (or centres of
      mass, or heavy-atom centroids) under periodic boundary conditions.
      Internally this is the same gap+enclosure algorithm operating on
      centroid--centroid distances instead of atom--atom distances, which
      is the natural picture for hybrid molecular crystals such as the
      ABX3 / ABX4 / A2BX5 hybrid perovskites where ``A`` is an organic
      cation (e.g. ``"C2 H10 N2"``) and ``B`` is a polyatomic anion
      (e.g. ``"Cl O4"``).

    The three radial knobs (``cutoff``, ``search_cutoff``, ``hard_cutoff``)
    carry **different meanings on the two levels** by design; see the
    "Radial knobs" subsection below for the rationale.

    Parameters
    ----------
    structure
        For ``level="atom"``: ``MolecularCrystal``, any object with a
        ``to_ase()`` method, or an ``ase.Atoms`` directly.
        For ``level="molecule"``: must be a ``MolecularCrystal``-like
        object exposing ``.molecules`` and ``.lattice``. Build one via
        :func:`molcrys_kit.io.cif.read_mol_crystal`.
    central, ligand
        For ``level="atom"``: chemical symbols (e.g. ``"N"``, ``"Cl"``).
        For ``level="molecule"``: single-fragment moiety strings as in
        the CIF ``_chemical_formula_moiety`` syntax (e.g. ``"N H4"``,
        ``"Cl O4"``, ``"C2 H10 N2"``). Charge and multiplier tokens are
        accepted but ignored; only the heavy-atom multiset is used for
        matching, so ``"Cl O4"`` and ``"Cl O4 1-"`` are equivalent.
    level
        Either ``"atom"`` (default) or ``"molecule"``.
    center_kind
        How to compute molecule centres for ``level="molecule"``. One of
        ``"centroid"`` (unweighted mean of all atoms; matches
        ``CrystalMolecule.get_centroid``), ``"com"`` (mass-weighted), or
        ``"heavy_centroid"`` (unweighted mean of non-H atoms). Ignored
        when ``level="atom"``.
    cutoff
        See "Radial knobs" below; meaning depends on ``level``.
    search_cutoff
        See "Radial knobs" below; meaning depends on ``level``.
    hard_cutoff
        See "Radial knobs" below; ``level="molecule"`` only — passing it
        with ``level="atom"`` raises ``ValueError``.
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
        Optional iterable of indices restricting the search to a subset
        of central atoms (``level="atom"``) or central molecules
        (``level="molecule"``).

    Radial knobs
    ------------
    The three radius arguments split responsibility cleanly between
    "where do we look for candidates?" and "which neighbours count as
    the shell?". They are wired differently across the two levels so
    that the most natural reading of ``cutoff=`` matches the level's
    most common use-case:

    * ``level="atom"``: ``cutoff`` is the **historical hard radial
      cutoff** (passed straight to :func:`detect_coordination_number` as
      ``cutoff=``); ``search_cutoff`` is the candidate radius for the
      ASE neighbour list. Atom-level callers writing
      ``find_polyhedra(atoms, "Pb", "I", cutoff=3)`` expect "octahedral
      Pb--I within 3 Å", which is the hard-cap semantics, so this stays
      unchanged for full backward compatibility. Passing ``hard_cutoff``
      on atom level raises ``ValueError`` because it would duplicate
      ``cutoff`` and dilute the asymmetry between the two levels.

    * ``level="molecule"``: ``cutoff`` is the **candidate search
      radius**; the CN is then chosen by the gap+enclosure heuristic in
      :func:`detect_coordination_number`. ``search_cutoff`` is a
      non-deprecated synonym of ``cutoff`` for callers who prefer the
      more self-documenting name (passing both raises ``ValueError``).
      Use ``hard_cutoff`` to opt back into the historical "fill the
      ball" behaviour (forwarded to :func:`detect_coordination_number`
      as ``cutoff=``). The molecule-level default
      (:data:`DEFAULT_MOLECULAR_SEARCH_CUTOFF`) is set generously so the
      first packing-shell is captured without forcing callers to pick a
      radius.

      Worked example on a DAP-4 Pa-3 perovskite cell:

      >>> find_polyhedra(crys, "N H4", "Cl O4", level="molecule",
      ...                cutoff=8.0)              # gap+enclosure
      # -> CN=6 octahedron (the NaCl-like first shell stops here)
      >>> find_polyhedra(crys, "N H4", "Cl O4", level="molecule",
      ...                hard_cutoff=8.0)         # historical fill mode
      # -> CN=12 cuboctahedron (the full perovskite A--X12 cage)

    Returns
    -------
    list of dict
        One entry per A atom (or A molecule). Always contains the
        :func:`detect_coordination_number` outputs
        (``coordination_number``, ``mode``, ``primary_gap_cn``,
        ``enclosed``, ``enclosure_expanded``, ``cutoff``, ``gap_index``,
        ``gap_value``) plus ``search_cutoff`` and ``level``.

        Atom level (``level="atom"``) additionally exposes:

        * ``center_index``: index in the source ``Atoms`` object
        * ``center_symbol``: ``central``
        * ``center_position``: Cartesian position (length-3 list)
        * ``shell_indices``: ligand atom indices, ordered by distance
        * ``shell_offsets``: PBC image offsets for each shell atom
        * ``shell_coords``: Cartesian positions with PBC offset applied
        * ``shell_distances``: A--B distances, ascending
        * ``best_match`` (only when ``score_shape=True``)

        Molecule level (``level="molecule"``) additionally exposes:

        * ``center_molecule_index``: index into ``structure.molecules``
        * ``center_formula``: Hill-ish formula of the central molecule
        * ``center_kind``: the ``center_kind`` actually used
        * ``shell_molecule_indices``: indices into ``structure.molecules``
        * ``shell_formula``: Hill-ish formula of the ligand species
        * ``shell_offsets``: lattice translation vectors in Å, one per
          shell entry (these are *Cartesian translations*, not integer
          ``(na, nb, nc)`` images, because the molecule list does not
          carry per-atom image bookkeeping)
        * ``shell_coords``: translated Cartesian centroids
        * ``shell_distances``: A--B centroid distances, ascending
        * ``hard_cutoff``: ``None`` in gap+enclosure mode, or the float
          value passed when the historical hard-cap mode was opted into.
          Downstream code should prefer this field over ``cutoff`` to
          inspect "was a hard cap applied?", since the molecule-level
          ``record["cutoff"]`` echoes what
          :func:`detect_coordination_number` received (i.e. the value of
          ``hard_cutoff``), which is the opposite of the kwarg
          ``cutoff=`` semantics on this level.
        * ``best_match`` (only when ``score_shape=True``)
    """
    if level == "atom":
        if hard_cutoff is not None:
            raise ValueError(
                "hard_cutoff is only meaningful when level='molecule', where "
                "cutoff= now means the search radius. On level='atom', "
                "cutoff= already is the hard radial cap; pass cutoff= "
                "instead of hard_cutoff."
            )
        return _find_polyhedra_atom_level(
            structure,
            central,
            ligand,
            cutoff=cutoff,
            search_cutoff=search_cutoff,
            enforce_enclosure=enforce_enclosure,
            centroid_offset_frac=centroid_offset_frac,
            fallback_max=fallback_max,
            score_shape=score_shape,
            central_indices=central_indices,
        )
    if level == "molecule":
        return _find_polyhedra_molecule_level(
            structure,
            central,
            ligand,
            center_kind=center_kind,
            cutoff=cutoff,
            search_cutoff=search_cutoff,
            hard_cutoff=hard_cutoff,
            enforce_enclosure=enforce_enclosure,
            centroid_offset_frac=centroid_offset_frac,
            fallback_max=fallback_max,
            score_shape=score_shape,
            central_indices=central_indices,
        )
    raise ValueError(f"level must be 'atom' or 'molecule', got {level!r}.")


__all__ = [
    "DEFAULT_CENTROID_OFFSET_FRAC",
    "DEFAULT_MOLECULAR_SEARCH_CUTOFF",
    "DEFAULT_POLYHEDRON_SEARCH_CUTOFF",
    "angular_rmsd_vs_ideals",
    "compute_angular_signature",
    "detect_coordination_number",
    "detect_prism_vs_antiprism",
    "find_polyhedra",
    "hull_encloses_center",
    "planarity_analysis",
]
