"""Aromatic ring-ring pi-stacking criteria, records, and detector.

The detector classifies ring pairs as face-centered parallel, displaced
parallel, or T-shaped using centroid distance, ring-normal angle, and lateral
offset.  Crystal inputs enable periodic-image searches and identity metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from ...structures.crystal import MolecularCrystal
from ..molecular_identity import ChemicalIdentity, ChemicalIdentityCache
from .base import BaseInteraction, RingRef, build_crystal_atom_offsets
from .geometry import enumerate_lattice_images, image_translation, vector_angle_deg
from .local_geometry import RingGeometry, LocalGeometryCache
from .scoring import ScoringParams, composite_score, scaled_cutoff


PiStackingSubtype = Literal[
    "face_centered_parallel",
    "displaced_parallel",
    "T_shape",
]
"""Supported empirical pi-stacking geometry classes."""


@dataclass(frozen=True)
class PiStackingCriteria:
    """Geometric thresholds for aromatic ring stacking.

    Distances are in Å and angles are in degrees.

    Parallel stacking uses ``max_interplane_distance_A`` (the vertical
    component h of the centroid–centroid vector projected onto the ring
    normal) as the primary distance filter.  This prevents displaced-parallel
    pairs with reasonable h but large lateral offset from being rejected.

    T-shape stacking uses ``max_t_shape_centroid_distance_A`` (the full
    centroid–centroid distance d) with a larger cutoff, since T-shape
    geometry naturally produces larger d values.

    Constructor names ``max_normal_angle_deg`` and ``max_lateral_offset_A``
    are accepted as aliases for the parallel thresholds.
    """

    max_centroid_distance_A: float = 4.5
    max_interplane_distance_A: float = 4.0
    max_t_shape_centroid_distance_A: float = 6.5
    max_parallel_normal_angle_deg: float = 30.0
    min_t_shape_normal_angle_deg: float = 60.0
    max_t_shape_normal_angle_deg: float = 120.0
    max_face_centered_offset_A: float = 1.0
    max_parallel_lateral_offset_A: float = 2.5
    max_t_shape_lateral_offset_A: float = 3.0
    aromatic_only: bool = True
    search_radius_A: float | None = None
    scoring_params: ScoringParams = field(default_factory=ScoringParams)

    def __init__(
        self,
        max_centroid_distance_A: float = 4.5,
        max_interplane_distance_A: float = 4.0,
        max_t_shape_centroid_distance_A: float = 6.5,
        max_parallel_normal_angle_deg: float = 30.0,
        min_t_shape_normal_angle_deg: float = 60.0,
        max_t_shape_normal_angle_deg: float = 120.0,
        max_face_centered_offset_A: float = 1.0,
        max_parallel_lateral_offset_A: float = 2.5,
        max_t_shape_lateral_offset_A: float = 3.0,
        aromatic_only: bool = True,
        search_radius_A: float | None = None,
        scoring_params: ScoringParams | None = None,
        max_normal_angle_deg: float | None = None,
        max_lateral_offset_A: float | None = None,
    ):
        """Initialize pi-stacking thresholds with compatible aliases.

        If ``max_normal_angle_deg`` is supplied, it overrides
        ``max_parallel_normal_angle_deg``.  If ``max_lateral_offset_A`` is
        supplied, it overrides ``max_parallel_lateral_offset_A``.
        """
        if max_normal_angle_deg is not None:
            max_parallel_normal_angle_deg = max_normal_angle_deg
        if max_lateral_offset_A is not None:
            max_parallel_lateral_offset_A = max_lateral_offset_A
        object.__setattr__(
            self,
            "max_centroid_distance_A",
            float(max_centroid_distance_A),
        )
        object.__setattr__(
            self,
            "max_interplane_distance_A",
            float(max_interplane_distance_A),
        )
        object.__setattr__(
            self,
            "max_t_shape_centroid_distance_A",
            float(max_t_shape_centroid_distance_A),
        )
        object.__setattr__(
            self,
            "max_parallel_normal_angle_deg",
            float(max_parallel_normal_angle_deg),
        )
        object.__setattr__(
            self,
            "min_t_shape_normal_angle_deg",
            float(min_t_shape_normal_angle_deg),
        )
        object.__setattr__(
            self,
            "max_t_shape_normal_angle_deg",
            float(max_t_shape_normal_angle_deg),
        )
        object.__setattr__(
            self,
            "max_face_centered_offset_A",
            float(max_face_centered_offset_A),
        )
        object.__setattr__(
            self,
            "max_parallel_lateral_offset_A",
            float(max_parallel_lateral_offset_A),
        )
        object.__setattr__(
            self,
            "max_t_shape_lateral_offset_A",
            float(max_t_shape_lateral_offset_A),
        )
        object.__setattr__(self, "aromatic_only", bool(aromatic_only))
        object.__setattr__(self, "search_radius_A", search_radius_A)
        object.__setattr__(self, "scoring_params", scoring_params or ScoringParams())

    @property
    def max_normal_angle_deg(self) -> float:
        """Backward-compatible alias for the parallel normal-angle cutoff."""
        return self.max_parallel_normal_angle_deg

    @property
    def max_lateral_offset_A(self) -> float:
        """Backward-compatible alias for the displaced-parallel offset cutoff."""
        return self.max_parallel_lateral_offset_A


@dataclass(init=False)
class PiStacking(BaseInteraction):
    """Ring-ring pi-stacking interaction record.

    The record stores both ring references, molecule identities, centroid
    distance, normal/plane angle, lateral offset, assigned stacking subtype,
    periodic image, translation, and criteria metadata.
    """

    ring1: RingRef
    ring2: RingRef
    molecule1_identity: ChemicalIdentity | None
    molecule2_identity: ChemicalIdentity | None
    centroid_distance_A: float
    normal_angle_deg: float
    plane_angle_deg: float
    lateral_offset_A: float
    subtype: PiStackingSubtype

    def __init__(
        self,
        *,
        ring1: RingRef,
        ring2: RingRef,
        centroid_distance_A: float,
        normal_angle_deg: float,
        lateral_offset_A: float,
        subtype: PiStackingSubtype,
        plane_angle_deg: float | None = None,
        molecule1_identity: ChemicalIdentity | None = None,
        molecule2_identity: ChemicalIdentity | None = None,
        image: tuple[int, int, int] = (0, 0, 0),
        translation_A: tuple[float, float, float] | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a pi-stacking interaction from ring references.

        ``plane_angle_deg`` defaults to ``normal_angle_deg`` for compatibility
        with the current ring-normal-based classification.
        """
        plane_angle = normal_angle_deg if plane_angle_deg is None else plane_angle_deg
        BaseInteraction.__init__(
            self,
            kind="pi_stacking",
            participants={"ring1": ring1, "ring2": ring2},
            distance_A=float(centroid_distance_A),
            angle_deg=float(normal_angle_deg),
            score=score,
            image=tuple(int(v) for v in image),
            translation_A=translation_A,
            metadata=metadata or {},
        )
        self.ring1 = ring1
        self.ring2 = ring2
        self.molecule1_identity = molecule1_identity
        self.molecule2_identity = molecule2_identity
        self.centroid_distance_A = float(centroid_distance_A)
        self.normal_angle_deg = float(normal_angle_deg)
        self.plane_angle_deg = float(plane_angle)
        self.lateral_offset_A = float(lateral_offset_A)
        self.subtype = subtype


def find_pi_stacking(
    target: MolecularCrystal | Sequence,
    criteria: PiStackingCriteria | None = None,
) -> list[PiStacking]:
    """Find pi-stacking interactions between molecular rings.

    Rings are obtained from ``LocalGeometry``, optionally restricted to aromatic
    rings.  Each ring pair is evaluated over relevant periodic images for
    crystal inputs, or only in the input coordinates for molecule sequences.
    Accepted pairs are deduplicated by molecule index, ring atom indices, and
    image.
    """
    criteria = criteria or PiStackingCriteria()
    crystal = target if isinstance(target, MolecularCrystal) else None
    molecules = list(crystal.molecules if crystal is not None else target)
    atom_offsets = build_crystal_atom_offsets(molecules)
    local_geometries = LocalGeometryCache(molecules)
    identities = ChemicalIdentityCache(crystal) if crystal is not None else None

    if crystal is not None:
        lattice = np.asarray(crystal.lattice, dtype=float)
        images = enumerate_lattice_images(
            lattice,
            tuple(bool(v) for v in crystal.pbc),
            criteria.search_radius_A
            or scaled_cutoff(criteria.max_t_shape_centroid_distance_A, criteria.scoring_params),
        )
    else:
        lattice = None
        images = ((0, 0, 0),)

    stacks: list[PiStacking] = []
    seen: set[tuple] = set()
    for mol1_idx in range(len(molecules)):
        rings1 = local_geometries[mol1_idx].rings(aromatic_only=criteria.aromatic_only)
        mol2_start = mol1_idx if crystal is not None else mol1_idx + 1
        for mol2_idx in range(mol2_start, len(molecules)):
            rings2 = local_geometries[mol2_idx].rings(aromatic_only=criteria.aromatic_only)
            for ring1 in rings1:
                for ring2 in rings2:
                    for image in images:
                        if mol1_idx == mol2_idx and image == (0, 0, 0):
                            continue
                        translation = _translation(lattice, image)
                        result = _evaluate_ring_pair(
                            molecules,
                            mol1_idx,
                            mol2_idx,
                            ring1,
                            ring2,
                            image,
                            translation,
                            atom_offsets,
                            criteria,
                        )
                        if result is None:
                            continue
                        key = (
                            mol1_idx,
                            ring1.atom_indices,
                            mol2_idx,
                            ring2.atom_indices,
                            tuple(int(v) for v in image),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        ring1_ref, ring2_ref, distance, angle, offset, subtype, score = result
                        stacks.append(
                            PiStacking(
                                ring1=ring1_ref,
                                ring2=ring2_ref,
                                centroid_distance_A=distance,
                                normal_angle_deg=angle,
                                lateral_offset_A=offset,
                                subtype=subtype,
                                molecule1_identity=identities[mol1_idx] if identities else None,
                                molecule2_identity=identities[mol2_idx] if identities else None,
                                image=tuple(int(v) for v in image),
                                translation_A=(
                                    tuple(float(v) for v in translation)
                                    if lattice is not None
                                    else None
                                ),
                                score=score,
                                metadata={
                                    "criteria": _criteria_metadata(criteria),
                                    "score_components": {
                                        "centroid_distance_A": distance,
                                        "interplane_distance_A": float(
                                            abs(np.dot(
                                                np.asarray(ring2.centroid_A) + _translation(lattice, image) - np.asarray(ring1.centroid_A),
                                                np.asarray(ring1.normal) / max(np.linalg.norm(ring1.normal), 1e-10),
                                            ))
                                        ),
                                        "normal_angle_deg": angle,
                                        "lateral_offset_A": offset,
                                        "score_distance_A": float(
                                            distance if subtype == "T_shape" else abs(np.dot(
                                                np.asarray(ring2.centroid_A) + _translation(lattice, image) - np.asarray(ring1.centroid_A),
                                                np.asarray(ring1.normal) / max(np.linalg.norm(ring1.normal), 1e-10),
                                            ))
                                        ),
                                        "lateral_offset_weight": 1.0,
                                    },
                                },
                            )
                        )
    return stacks


def find_pi_stacks(
    target: MolecularCrystal | Sequence,
    criteria: PiStackingCriteria | None = None,
) -> list[PiStacking]:
    """Alias for :func:`find_pi_stacking` with identical behavior."""
    return find_pi_stacking(target, criteria=criteria)


def _evaluate_ring_pair(
    molecules,
    mol1_idx,
    mol2_idx,
    ring1: RingGeometry,
    ring2: RingGeometry,
    image,
    translation,
    atom_offsets,
    criteria,
):
    """Evaluate one ring pair and return construction data if accepted.

    Geometry is decomposed into a right triangle:
      d = centroid-centroid distance (hypotenuse)
      h = interplane distance (projection of d onto ring1 normal)
      l = lateral offset (in-plane displacement)
    Parallel stacking uses h for distance thresholds; T-shape uses d.

    Both subtypes require a **ring-projection overlap check**: the
    approaching ring's centroid, projected onto the target ring's plane,
    must fall within the target ring's circumscribed circle.  This
    prevents false positives where rings are geometrically parallel or
    perpendicular but spatially non-overlapping.
    """
    c1 = np.asarray(ring1.centroid_A, dtype=float)
    c2 = np.asarray(ring2.centroid_A, dtype=float) + translation
    centroid_vec = c2 - c1
    centroid_distance = float(np.linalg.norm(centroid_vec))

    # Coarse prefilter: reject if beyond max possible distance for any subtype
    max_possible = max(
        scaled_cutoff(criteria.max_centroid_distance_A, criteria.scoring_params),
        scaled_cutoff(criteria.max_t_shape_centroid_distance_A, criteria.scoring_params),
    )
    if centroid_distance > max_possible:
        return None

    n1 = np.asarray(ring1.normal, dtype=float)
    n2 = np.asarray(ring2.normal, dtype=float)
    raw_angle = vector_angle_deg(n1, n2)
    normal_angle = min(raw_angle, 180.0 - raw_angle)
    n1_norm = np.linalg.norm(n1)
    if n1_norm == 0:
        return None
    n1_unit = n1 / n1_norm

    # Decompose centroid vector into interplane (h) and lateral (l) components
    interplane_distance = float(abs(np.dot(centroid_vec, n1_unit)))
    lateral_offset = float(np.sqrt(max(centroid_distance**2 - interplane_distance**2, 0.0)))

    # Subtype-specific filtering
    subtype = _classify_pi_stacking(
        normal_angle, lateral_offset, interplane_distance, centroid_distance, criteria,
    )
    if subtype is None:
        return None

    # Ring-projection overlap check: project ring2 centroid onto ring1's
    # plane and verify the projection falls within ring1's circumscribed
    # circle.  For T-shape the approaching ring's edge points at the
    # target; for parallel the offset must keep the rings overlapping.
    # The lateral_offset is exactly this projected distance.
    r1_circumradius = _ring_circumradius(molecules[mol1_idx], ring1)
    if lateral_offset > r1_circumradius:
        # Also check the reverse: project ring1 centroid onto ring2's plane
        n2_norm = np.linalg.norm(n2)
        if n2_norm > 0:
            n2_unit = n2 / n2_norm
            lateral_on_ring2 = float(np.sqrt(max(
                centroid_distance**2 - np.dot(centroid_vec, n2_unit)**2, 0.0,
            )))
            r2_circumradius = _ring_circumradius(molecules[mol2_idx], ring2)
            if lateral_on_ring2 > r2_circumradius:
                return None
        else:
            return None

    # Scoring: use interplane distance for parallel, centroid distance for T-shape
    # with subtype-specific distance center and sigma
    if subtype in ("face_centered_parallel", "displaced_parallel"):
        score_distance = interplane_distance
        dist0 = criteria.scoring_params.pi_centroid_distance0_A
        dist_sigma = criteria.scoring_params.pi_centroid_distance_sigma_A
    else:
        score_distance = centroid_distance
        dist0 = criteria.scoring_params.pi_t_shape_distance0_A
        dist_sigma = criteria.scoring_params.pi_t_shape_distance_sigma_A

    angle0 = (
        criteria.scoring_params.pi_t_shape_angle0_deg
        if subtype == "T_shape"
        else criteria.scoring_params.pi_parallel_angle0_deg
    )
    score = composite_score(
        (
            (
                score_distance,
                dist0,
                dist_sigma,
                "lorentzian",
            ),
            (
                normal_angle,
                angle0,
                criteria.scoring_params.pi_angle_sigma_deg,
                "gaussian",
            ),
        )
    )
    if score < criteria.scoring_params.score_threshold:
        return None
    ring1_ref = RingRef.from_molecule(
        molecules[mol1_idx],
        mol1_idx,
        ring1.atom_indices,
        is_aromatic=ring1.is_aromatic,
        crystal_atom_offset=atom_offsets[mol1_idx],
    )
    ring2_ref = RingRef.from_molecule(
        molecules[mol2_idx],
        mol2_idx,
        ring2.atom_indices,
        is_aromatic=ring2.is_aromatic,
        image=image,
        crystal_atom_offset=atom_offsets[mol2_idx],
    )
    return ring1_ref, ring2_ref, centroid_distance, normal_angle, lateral_offset, subtype, score


def _classify_pi_stacking(
    normal_angle: float,
    lateral_offset: float,
    interplane_distance: float,
    centroid_distance: float,
    criteria: PiStackingCriteria,
) -> PiStackingSubtype | None:
    """Classify ring-pair geometry as a supported pi-stacking subtype.

    Parallel rings are filtered by interplane distance h (not centroid
    distance d), then split into face-centered and displaced by lateral
    offset.  T-shaped rings use centroid distance d with a larger cutoff.
    """
    if normal_angle <= criteria.max_parallel_normal_angle_deg:
        # Parallel: use interplane distance (h) as the primary filter
        if interplane_distance > criteria.max_interplane_distance_A:
            return None
        if lateral_offset <= criteria.max_face_centered_offset_A:
            return "face_centered_parallel"
        if lateral_offset <= criteria.max_parallel_lateral_offset_A:
            return "displaced_parallel"
        return None
    # TODO: the 30°–60° normal-angle gap currently rejects all ring pairs
    # in this range.  Some "tilted stacking" interactions may be physically
    # meaningful but are excluded.  Consider adding a "tilted" subtype or
    # narrowing the gap in a future PR.
    if (
        criteria.min_t_shape_normal_angle_deg
        <= normal_angle
        <= criteria.max_t_shape_normal_angle_deg
        and centroid_distance <= criteria.max_t_shape_centroid_distance_A
        and lateral_offset <= criteria.max_t_shape_lateral_offset_A
    ):
        return "T_shape"
    return None


def _ring_circumradius(molecule, ring: RingGeometry) -> float:
    """Return the circumscribed radius of a ring (max centroid-to-vertex distance).

    This is used for the projection overlap check: if the approaching ring's
    centroid projection onto the target ring's plane falls outside this radius,
    the two rings do not overlap spatially.
    """
    positions = molecule.get_positions()
    centroid = np.asarray(ring.centroid_A, dtype=float)
    ring_positions = positions[list(ring.atom_indices)]
    distances = np.linalg.norm(ring_positions - centroid, axis=1)
    return float(np.max(distances))


def _translation(lattice, image: tuple[int, int, int]) -> np.ndarray:
    """Return the Cartesian translation vector for an image as an array."""
    if lattice is None:
        return np.zeros(3)
    return np.asarray(image_translation(lattice, image), dtype=float)


def _criteria_metadata(criteria: PiStackingCriteria) -> dict[str, Any]:
    """Return a JSON-friendly snapshot of pi-stacking criteria."""
    return {
        "max_centroid_distance_A": criteria.max_centroid_distance_A,
        "max_interplane_distance_A": criteria.max_interplane_distance_A,
        "max_t_shape_centroid_distance_A": criteria.max_t_shape_centroid_distance_A,
        "max_parallel_normal_angle_deg": criteria.max_parallel_normal_angle_deg,
        "min_t_shape_normal_angle_deg": criteria.min_t_shape_normal_angle_deg,
        "max_t_shape_normal_angle_deg": criteria.max_t_shape_normal_angle_deg,
        "max_face_centered_offset_A": criteria.max_face_centered_offset_A,
        "max_parallel_lateral_offset_A": criteria.max_parallel_lateral_offset_A,
        "max_t_shape_lateral_offset_A": criteria.max_t_shape_lateral_offset_A,
        "aromatic_only": criteria.aromatic_only,
        "prefilter_factor": criteria.scoring_params.prefilter_factor,
        "score_threshold": criteria.scoring_params.score_threshold,
    }