"""
Interpolation paths between molecular crystal replicas.

This module constructs rigid-body paths between two molecular crystal replicas,
with molecule-wise correspondence.  The primary mode is an SE(3) geodesic / screw
motion interpolation, with SO(3)+COM and quaternion SLERP baselines for workflow
compatibility and comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence

import networkx as nx
import numpy as np

from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..utils.geometry import (
    cart_to_frac,
    frac_to_cart,
    kabsch_align,
    lattice_at_lambda,
    lattice_deformation_logm,
    minimum_image_vector,
    quaternion_slerp,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_to_axis_angle,
    se3_exp,
    se3_log,
)


class InterpolationMethod(str, Enum):
    """Supported rigid-body interpolation metrics."""

    SE3_SCREW = "se3_screw"
    COM_SO3 = "com_so3"
    SLERP = "slerp"


@dataclass(frozen=True)
class InterpolationConfig:
    """Configuration for crystal replica interpolation."""

    method: InterpolationMethod | str = InterpolationMethod.SE3_SCREW
    n_images: int = 11
    include_endpoints: bool = True


@dataclass(frozen=True)
class MoleculeMatch:
    """Rigid-body correspondence between one molecule in replica A and B."""

    idx_a: int
    idx_b: int
    atom_mapping: np.ndarray
    com_a: np.ndarray
    com_b: np.ndarray
    com_b_unwrapped: np.ndarray
    com_translation: np.ndarray
    rotation_matrix: np.ndarray
    axis: np.ndarray
    angle_rad: float
    fit_rmsd: float
    pose_rmsd: float

    @property
    def angle_deg(self) -> float:
        """Rotation angle in degrees."""
        return float(np.degrees(self.angle_rad))

    @property
    def com_dist(self) -> float:
        """Minimum-image COM displacement length in Angstrom."""
        return float(np.linalg.norm(self.com_translation))


def _coerce_method(method: InterpolationMethod | str) -> InterpolationMethod:
    if isinstance(method, InterpolationMethod):
        return method
    try:
        return InterpolationMethod(str(method))
    except ValueError:
        normalized = str(method).lower().replace("-", "_")
        aliases = {
            "screw_rotation": InterpolationMethod.SE3_SCREW,
            "screw": InterpolationMethod.SE3_SCREW,
            "se3": InterpolationMethod.SE3_SCREW,
            "se3_geodesic": InterpolationMethod.SE3_SCREW,
            "com_alignment": InterpolationMethod.COM_SO3,
            "com": InterpolationMethod.COM_SO3,
            "so3_com": InterpolationMethod.COM_SO3,
            "quaternion_slerp": InterpolationMethod.SLERP,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unknown interpolation method: {method!r}") from None


def _molecule_node_match(attrs_a: dict, attrs_b: dict) -> bool:
    return attrs_a.get("symbol") == attrs_b.get("symbol")


def _iter_isomorphism_mappings(
    graph_a: nx.Graph, graph_b: nx.Graph
) -> Iterable[dict[int, int]]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph_a,
        graph_b,
        node_match=_molecule_node_match,
    )
    return matcher.isomorphisms_iter()


def best_atom_mapping(
    mol_a: CrystalMolecule,
    mol_b: CrystalMolecule,
    *,
    max_isomorphisms: int = 256,
) -> np.ndarray:
    """Return the atom order in ``mol_b`` that best corresponds to ``mol_a``.

    The mapping is an integer array ``order_b`` such that
    ``mol_b.get_positions()[order_b]`` is atom-by-atom comparable to
    ``mol_a.get_positions()``. Candidate graph isomorphisms are element-aware;
    if a molecule has multiple symmetrically equivalent atom mappings, the one
    with the lowest Kabsch RMSD is selected.
    """
    if len(mol_a) != len(mol_b):
        raise ValueError("Cannot map molecules with different atom counts")
    if mol_a.get_chemical_formula() != mol_b.get_chemical_formula():
        raise ValueError(
            "Cannot map molecules with different formulas: "
            f"{mol_a.get_chemical_formula()} vs {mol_b.get_chemical_formula()}"
        )
    if len(mol_a) == 0:
        return np.array([], dtype=int)

    graph_a = mol_a.get_graph()
    graph_b = mol_b.get_graph()
    best_order = None
    best_rmsd = np.inf
    pos_a = np.asarray(mol_a.get_positions(), dtype=float)
    com_a = np.asarray(mol_a.get_center_of_mass(), dtype=float)
    centered_a = pos_a - com_a
    pos_b_full = np.asarray(mol_b.get_positions(), dtype=float)
    com_b = np.asarray(mol_b.get_center_of_mass(), dtype=float)

    count = 0
    for mapping in _iter_isomorphism_mappings(graph_a, graph_b):
        order_b = np.array([mapping[i] for i in range(len(mol_a))], dtype=int)
        centered_b = pos_b_full[order_b] - com_b
        _, rmsd = kabsch_align(centered_a, centered_b)
        if rmsd < best_rmsd:
            best_order = order_b
            best_rmsd = rmsd
        count += 1
        if count >= max_isomorphisms:
            break

    if best_order is None:
        symbols_a = mol_a.get_chemical_symbols()
        symbols_b = mol_b.get_chemical_symbols()
        if symbols_a == symbols_b:
            return np.arange(len(mol_a), dtype=int)
        raise ValueError(
            "Could not find an element-aware graph isomorphism between molecules"
        )
    return best_order


def _minimum_image_translation(
    point_a: np.ndarray,
    point_b: np.ndarray,
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frac_a = cart_to_frac(point_a, lattice)
    frac_b = cart_to_frac(point_b, lattice)
    frac_delta = frac_b - frac_a
    delta_cart = minimum_image_vector(frac_delta, lattice)
    image_shift_frac = cart_to_frac(delta_cart, lattice) - frac_delta
    return delta_cart, image_shift_frac


def _copy_crystal_with_positions(
    crystal: MolecularCrystal,
    positions_by_index: dict[int, np.ndarray],
) -> MolecularCrystal:
    molecules = []
    for index, molecule in enumerate(crystal.molecules):
        copied = molecule.copy()
        if index in positions_by_index:
            copied.set_positions(np.asarray(positions_by_index[index], dtype=float))
        molecules.append(copied)
    return MolecularCrystal(
        np.array(crystal.lattice, dtype=float).copy(),
        molecules,
        crystal.pbc,
        formula_moiety=crystal.formula_moiety,
        disorder_provenance=crystal.disorder_provenance,
    )


def match_molecules(
    crystal_a: MolecularCrystal,
    crystal_b: MolecularCrystal,
    *,
    max_isomorphisms: int = 256,
) -> List[MoleculeMatch]:
    """Match molecules between two crystal replicas and decompose their poses.

    Molecules are matched by formula and atom count, with closest minimum-image
    COM distance as the final assignment criterion.  Each match stores the
    Kabsch-derived relative rotation and the minimum-image COM translation from
    replica A to B.
    """
    if len(crystal_a.molecules) != len(crystal_b.molecules):
        raise ValueError(
            "Replica interpolation requires the same number of molecules; "
            f"got {len(crystal_a.molecules)} and {len(crystal_b.molecules)}"
        )

    lattice_a = np.asarray(crystal_a.lattice, dtype=float)
    lattice_b = np.asarray(crystal_b.lattice, dtype=float)
    if not np.allclose(lattice_a, lattice_b, atol=1e-6):
        raise ValueError("Replica interpolation currently requires identical lattices")

    used_b: set[int] = set()
    matches: List[MoleculeMatch] = []

    for idx_a, mol_a in enumerate(crystal_a.molecules):
        formula = mol_a.get_chemical_formula()
        candidates = [
            idx_b
            for idx_b, mol_b in enumerate(crystal_b.molecules)
            if idx_b not in used_b
            and len(mol_b) == len(mol_a)
            and mol_b.get_chemical_formula() == formula
        ]
        if not candidates:
            raise ValueError(
                f"No unmatched molecule in replica B matches molecule {idx_a} "
                f"({formula}, {len(mol_a)} atoms)"
            )

        com_a = np.asarray(mol_a.get_center_of_mass(), dtype=float)
        best_idx_b = min(
            candidates,
            key=lambda idx_b: np.linalg.norm(
                _minimum_image_translation(
                    com_a,
                    np.asarray(crystal_b.molecules[idx_b].get_center_of_mass(), dtype=float),
                    lattice_a,
                )[0]
            ),
        )
        used_b.add(best_idx_b)
        mol_b = crystal_b.molecules[best_idx_b]

        com_b = np.asarray(mol_b.get_center_of_mass(), dtype=float)
        com_translation, image_shift_frac = _minimum_image_translation(com_a, com_b, lattice_a)
        image_shift_cart = frac_to_cart(image_shift_frac, lattice_a)
        com_b_unwrapped = com_b + image_shift_cart

        order_b = best_atom_mapping(mol_a, mol_b, max_isomorphisms=max_isomorphisms)
        pos_a = np.asarray(mol_a.get_positions(), dtype=float)
        pos_b = np.asarray(mol_b.get_positions(), dtype=float)[order_b] + image_shift_cart
        centered_a = pos_a - com_a
        centered_b = pos_b - com_b_unwrapped
        rotation, fit_rmsd = kabsch_align(centered_a, centered_b)
        axis, angle = rotation_to_axis_angle(rotation)
        pose_rmsd = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (centered_a @ rotation.T + com_a + com_translation - pos_b) ** 2,
                        axis=1,
                    )
                )
            )
        )

        matches.append(
            MoleculeMatch(
                idx_a=idx_a,
                idx_b=best_idx_b,
                atom_mapping=order_b,
                com_a=com_a,
                com_b=com_b,
                com_b_unwrapped=com_b_unwrapped,
                com_translation=com_translation,
                rotation_matrix=rotation,
                axis=axis,
                angle_rad=float(angle),
                fit_rmsd=float(fit_rmsd),
                pose_rmsd=pose_rmsd,
            )
        )

    return matches


def interpolate_pose(
    mol_a: CrystalMolecule,
    match: MoleculeMatch,
    lam: float,
    method: InterpolationMethod | str = InterpolationMethod.SE3_SCREW,
) -> np.ndarray:
    """Interpolate a matched molecule pose at interpolation parameter ``lam``.

    .. warning::

       The ``COM_SO3`` method uses axis-angle scaling for rotation, which is
       degenerate at exactly 180°.  For near-180° rotations, prefer
       ``SE3_SCREW`` or ``SLERP``.
    """
    lam = float(lam)
    method = _coerce_method(method)
    positions_a = np.asarray(mol_a.get_positions(), dtype=float)
    centered = positions_a - match.com_a

    if method == InterpolationMethod.SE3_SCREW:
        relative_xi = se3_log(match.rotation_matrix, match.com_translation)
        rotation, translation = se3_exp(lam * relative_xi)
        return centered @ rotation.T + match.com_a + translation

    if method == InterpolationMethod.COM_SO3:
        rotation = se3_exp(np.concatenate([match.axis * match.angle_rad * lam, np.zeros(3)]))[0]
        return centered @ rotation.T + match.com_a + lam * match.com_translation

    if method == InterpolationMethod.SLERP:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = rotation_matrix_to_quaternion(match.rotation_matrix)
        rotation = quaternion_to_rotation_matrix(quaternion_slerp(q0, q1, lam))
        return centered @ rotation.T + match.com_a + lam * match.com_translation

    raise ValueError(f"Unhandled interpolation method: {method}")


def _lambda_values(n_images: int, include_endpoints: bool) -> np.ndarray:
    if n_images < 1:
        raise ValueError("n_images must be >= 1")
    if include_endpoints:
        if n_images == 1:
            return np.array([0.0])
        return np.linspace(0.0, 1.0, int(n_images))
    return np.linspace(0.0, 1.0, int(n_images) + 2)[1:-1]


def interpolate_crystal(
    crystal_a: MolecularCrystal,
    crystal_b: MolecularCrystal,
    *,
    method: InterpolationMethod | str = InterpolationMethod.SE3_SCREW,
    n_images: int = 11,
    include_endpoints: bool = True,
    molecule_indices: Optional[Sequence[int]] = None,
    matches: Optional[Sequence[MoleculeMatch]] = None,
) -> List[MolecularCrystal]:
    """Generate interpolated crystal images between two replicas.

    Parameters
    ----------
    crystal_a, crystal_b : MolecularCrystal
        Start and end replicas with matching molecular composition and lattice.
    method : InterpolationMethod or str
        ``"se3_screw"`` for SE(3) geodesic screw motion, ``"com_so3"`` for
        COM-linear plus axis-angle rotation, or ``"slerp"`` for COM-linear plus
        quaternion SLERP.
    n_images : int
        Number of frames returned. Includes endpoints by default.
    include_endpoints : bool
        If False, return only interior frames.
    molecule_indices : sequence of int, optional
        If provided, only these molecules move; all other molecules remain at
        their replica-A positions. This is the rigid-body "partial flip" mode.
    matches : sequence of MoleculeMatch, optional
        Precomputed molecule matches from :func:`match_molecules`.
    """
    method = _coerce_method(method)
    if matches is None:
        matches = match_molecules(crystal_a, crystal_b)
    match_by_idx = {match.idx_a: match for match in matches}
    if molecule_indices is None:
        selected = set(match_by_idx)
    else:
        selected = {int(index) for index in molecule_indices}
        missing = selected - set(match_by_idx)
        if missing:
            raise ValueError(f"Unknown molecule indices requested: {sorted(missing)}")

    frames: List[MolecularCrystal] = []
    for lam in _lambda_values(n_images, include_endpoints):
        positions = {}
        for idx in selected:
            match = match_by_idx[idx]
            positions[idx] = interpolate_pose(
                crystal_a.molecules[idx],
                match,
                float(lam),
                method=method,
            )
        frames.append(_copy_crystal_with_positions(crystal_a, positions))
    return frames


def interpolate_molecule(
    crystal_a: MolecularCrystal,
    crystal_b: MolecularCrystal,
    molecule_index: int,
    *,
    method: InterpolationMethod | str = InterpolationMethod.SE3_SCREW,
    n_images: int = 11,
    include_endpoints: bool = True,
) -> List[MolecularCrystal]:
    """Generate a partial-flip path for one molecule only."""
    return interpolate_crystal(
        crystal_a,
        crystal_b,
        method=method,
        n_images=n_images,
        include_endpoints=include_endpoints,
        molecule_indices=[molecule_index],
    )


def find_flipping_molecules(
    crystal_a: MolecularCrystal,
    crystal_b: MolecularCrystal,
    *,
    rmsd_threshold: float = 0.5,
    angle_threshold: float = 5.0,
    com_threshold: Optional[float] = None,
) -> List[int]:
    """Return molecule indices whose poses differ significantly between replicas."""
    matches = match_molecules(crystal_a, crystal_b)
    selected = []
    for match in matches:
        angle_hit = abs(match.angle_deg) > float(angle_threshold)
        rmsd_hit = match.pose_rmsd > float(rmsd_threshold)
        com_hit = False
        if com_threshold is not None:
            com_hit = match.com_dist > float(com_threshold)
        elif rmsd_threshold is not None:
            com_hit = match.com_dist > float(rmsd_threshold)
        if rmsd_hit or angle_hit or com_hit:
            selected.append(match.idx_a)
    return selected


# ─────────────────────────────────────────────────────────────────────
# Variable-cell interpolation
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VCMoleculeMatch:
    """Molecule match for variable-cell interpolation.

    Stores fractional-coordinate COM positions and the Kabsch rotation
    computed on centered (internal) coordinates.
    """

    idx_a: int
    idx_b: int
    atom_mapping: np.ndarray
    frac_com_a: np.ndarray
    frac_com_b: np.ndarray
    rotation_matrix: np.ndarray
    axis: np.ndarray
    angle_rad: float
    fit_rmsd: float

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(self.angle_rad))


def match_molecules_vc(
    crystal_a: MolecularCrystal,
    crystal_b: MolecularCrystal,
    *,
    max_isomorphisms: int = 256,
) -> List[VCMoleculeMatch]:
    """Match molecules between two replicas with *different* lattices.

    Unlike :func:`match_molecules`, this function does not require identical
    lattices. Molecule matching uses COM distance in Cartesian space (with
    the mean lattice for minimum-image convention). Rotations are decomposed
    from centered (internal) coordinates via Kabsch alignment.

    Parameters
    ----------
    crystal_a, crystal_b : MolecularCrystal
        Start and end crystal structures.  Must have the same number of
        molecules and matching molecular compositions.
    max_isomorphisms : int
        Maximum graph isomorphisms to explore for atom mapping.

    Returns
    -------
    list of VCMoleculeMatch
    """
    if len(crystal_a.molecules) != len(crystal_b.molecules):
        raise ValueError(
            "Variable-cell interpolation requires the same number of molecules; "
            f"got {len(crystal_a.molecules)} and {len(crystal_b.molecules)}"
        )

    lattice_a = np.asarray(crystal_a.lattice, dtype=float)
    lattice_b = np.asarray(crystal_b.lattice, dtype=float)
    # Use mean lattice for COM distance comparison
    lattice_mean = 0.5 * (lattice_a + lattice_b)

    used_b: set[int] = set()
    matches: List[VCMoleculeMatch] = []

    for idx_a, mol_a in enumerate(crystal_a.molecules):
        formula = mol_a.get_chemical_formula()
        candidates = [
            idx_b
            for idx_b, mol_b in enumerate(crystal_b.molecules)
            if idx_b not in used_b
            and len(mol_b) == len(mol_a)
            and mol_b.get_chemical_formula() == formula
        ]
        if not candidates:
            raise ValueError(
                f"No unmatched molecule in replica B matches molecule {idx_a} "
                f"({formula}, {len(mol_a)} atoms)"
            )

        com_a_cart = np.asarray(mol_a.get_center_of_mass(), dtype=float)
        frac_com_a = cart_to_frac(com_a_cart, lattice_a)

        # Find closest match using mean-lattice Cartesian distance
        def _distance(idx_b: int) -> float:
            com_b_cart = np.asarray(
                crystal_b.molecules[idx_b].get_center_of_mass(), dtype=float
            )
            frac_com_b = cart_to_frac(com_b_cart, lattice_b)
            # MIC in mean lattice
            frac_delta = frac_com_b - frac_com_a
            cart_delta = minimum_image_vector(frac_delta, lattice_mean)
            return float(np.linalg.norm(cart_delta))

        best_idx_b = min(candidates, key=_distance)
        used_b.add(best_idx_b)
        mol_b = crystal_b.molecules[best_idx_b]

        com_b_cart = np.asarray(mol_b.get_center_of_mass(), dtype=float)
        frac_com_b = cart_to_frac(com_b_cart, lattice_b)

        # Atom mapping and Kabsch on centered coords
        order_b = best_atom_mapping(mol_a, mol_b, max_isomorphisms=max_isomorphisms)
        pos_a = np.asarray(mol_a.get_positions(), dtype=float)
        pos_b = np.asarray(mol_b.get_positions(), dtype=float)[order_b]
        centered_a = pos_a - com_a_cart
        centered_b = pos_b - com_b_cart
        rotation, fit_rmsd = kabsch_align(centered_a, centered_b)
        axis, angle = rotation_to_axis_angle(rotation)

        matches.append(
            VCMoleculeMatch(
                idx_a=idx_a,
                idx_b=best_idx_b,
                atom_mapping=order_b,
                frac_com_a=frac_com_a,
                frac_com_b=frac_com_b,
                rotation_matrix=rotation,
                axis=axis,
                angle_rad=float(angle),
                fit_rmsd=float(fit_rmsd),
            )
        )

    return matches


def interpolate_crystal_vc(
    crystal_a: MolecularCrystal,
    crystal_b: MolecularCrystal,
    *,
    n_images: int = 11,
    include_endpoints: bool = True,
    molecule_indices: Optional[Sequence[int]] = None,
    matches: Optional[Sequence[VCMoleculeMatch]] = None,
) -> List[MolecularCrystal]:
    """Variable-cell interpolation between two crystal replicas.

    Generates a path of crystal images where both the lattice and molecular
    poses change smoothly from ``crystal_a`` to ``crystal_b``.

    Lattice interpolation uses the GL⁺(3) geodesic (matrix log/exp of the
    deformation gradient).  Molecular COMs are linearly interpolated in
    fractional coordinates.  Molecular orientations are interpolated via
    quaternion SLERP on the Kabsch-derived rotation.

    The output images are suitable as initial guesses for variable-cell NEB
    calculations.

    Parameters
    ----------
    crystal_a, crystal_b : MolecularCrystal
        Start and end crystal structures with matching molecular composition.
    n_images : int
        Number of frames returned (including endpoints by default).
    include_endpoints : bool
        If False, return only interior frames.
    molecule_indices : sequence of int, optional
        If provided, only these molecules are interpolated; others stay at
        their ``crystal_a`` fractional positions (mapped to each image's
        lattice).
    matches : sequence of VCMoleculeMatch, optional
        Precomputed matches from :func:`match_molecules_vc`.

    Returns
    -------
    list of MolecularCrystal
        Interpolated crystal images.
    """
    lattice_a = np.asarray(crystal_a.lattice, dtype=float)
    lattice_b = np.asarray(crystal_b.lattice, dtype=float)
    log_F = lattice_deformation_logm(lattice_a, lattice_b)

    if matches is None:
        matches = match_molecules_vc(crystal_a, crystal_b)
    match_by_idx = {m.idx_a: m for m in matches}

    if molecule_indices is None:
        selected = set(match_by_idx)
    else:
        selected = {int(i) for i in molecule_indices}
        missing = selected - set(match_by_idx)
        if missing:
            raise ValueError(f"Unknown molecule indices: {sorted(missing)}")

    # Precompute fractional COMs for non-selected molecules (stay at A position)
    static_frac_coms: dict[int, np.ndarray] = {}
    static_centered: dict[int, np.ndarray] = {}
    for idx, mol in enumerate(crystal_a.molecules):
        if idx not in selected:
            com_cart = np.asarray(mol.get_center_of_mass(), dtype=float)
            static_frac_coms[idx] = cart_to_frac(com_cart, lattice_a)
            static_centered[idx] = (
                np.asarray(mol.get_positions(), dtype=float) - com_cart
            )

    frames: List[MolecularCrystal] = []
    for lam in _lambda_values(n_images, include_endpoints):
        lat_i = lattice_at_lambda(lattice_a, log_F, lam)

        molecules = []
        for idx, mol in enumerate(crystal_a.molecules):
            copied = mol.copy()

            if idx in selected:
                match = match_by_idx[idx]
                # Fractional COM interpolation with MIC unwrapping
                frac_com_b_unwrapped = match.frac_com_b - np.round(
                    match.frac_com_b - match.frac_com_a
                )
                frac_com_i = (1.0 - lam) * match.frac_com_a + lam * frac_com_b_unwrapped
                cart_com_i = frac_to_cart(frac_com_i, lat_i)
                # Orientation: quaternion SLERP
                q0 = np.array([1.0, 0.0, 0.0, 0.0])
                q1 = rotation_matrix_to_quaternion(match.rotation_matrix)
                R_i = quaternion_to_rotation_matrix(quaternion_slerp(q0, q1, lam))
                # Assemble positions
                pos_a = np.asarray(mol.get_positions(), dtype=float)
                com_a_cart = np.asarray(mol.get_center_of_mass(), dtype=float)
                centered_a = pos_a - com_a_cart
                new_pos = centered_a @ R_i.T + cart_com_i
            else:
                # Static molecule: keep fractional position, map to new lattice
                frac_com = static_frac_coms[idx]
                cart_com_i = frac_to_cart(frac_com, lat_i)
                new_pos = static_centered[idx] + cart_com_i

            copied.set_positions(new_pos)
            molecules.append(copied)

        frames.append(
            MolecularCrystal(
                lat_i.copy(),
                molecules,
                crystal_a.pbc,
                formula_moiety=crystal_a.formula_moiety,
                disorder_provenance=crystal_a.disorder_provenance,
            )
        )

    return frames
