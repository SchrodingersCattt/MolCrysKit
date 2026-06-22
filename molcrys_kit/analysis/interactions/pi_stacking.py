"""Aromatic pi-stacking interaction records and detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from ...structures.crystal import MolecularCrystal
from ..molecular_identity import ChemicalIdentity, ChemicalIdentityCache
from .base import BaseInteraction, RingRef, build_crystal_atom_offsets
from .geometry import enumerate_lattice_images, image_translation, vector_angle_deg
from .local_geometry import RingGeometry, LocalGeometryCache


PiStackingSubtype = Literal[
    "face_centered_parallel",
    "displaced_parallel",
    "T_shape",
]


@dataclass(frozen=True)
class PiStackingCriteria:
    """Geometric criteria for aromatic ring stacking."""

    max_centroid_distance_A: float = 4.5
    max_parallel_normal_angle_deg: float = 30.0
    min_t_shape_normal_angle_deg: float = 60.0
    max_t_shape_normal_angle_deg: float = 120.0
    max_face_centered_offset_A: float = 0.75
    max_parallel_lateral_offset_A: float = 2.0
    max_t_shape_lateral_offset_A: float = 2.5
    aromatic_only: bool = True
    search_radius_A: float | None = None

    def __init__(
        self,
        max_centroid_distance_A: float = 4.5,
        max_parallel_normal_angle_deg: float = 30.0,
        min_t_shape_normal_angle_deg: float = 60.0,
        max_t_shape_normal_angle_deg: float = 120.0,
        max_face_centered_offset_A: float = 0.75,
        max_parallel_lateral_offset_A: float = 2.0,
        max_t_shape_lateral_offset_A: float = 2.5,
        aromatic_only: bool = True,
        search_radius_A: float | None = None,
        max_normal_angle_deg: float | None = None,
        max_lateral_offset_A: float | None = None,
    ):
        if max_normal_angle_deg is not None:
            max_parallel_normal_angle_deg = max_normal_angle_deg
        if max_lateral_offset_A is not None:
            max_parallel_lateral_offset_A = max_lateral_offset_A
        object.__setattr__(self, "max_centroid_distance_A", float(max_centroid_distance_A))
        object.__setattr__(self, "max_parallel_normal_angle_deg", float(max_parallel_normal_angle_deg))
        object.__setattr__(self, "min_t_shape_normal_angle_deg", float(min_t_shape_normal_angle_deg))
        object.__setattr__(self, "max_t_shape_normal_angle_deg", float(max_t_shape_normal_angle_deg))
        object.__setattr__(self, "max_face_centered_offset_A", float(max_face_centered_offset_A))
        object.__setattr__(self, "max_parallel_lateral_offset_A", float(max_parallel_lateral_offset_A))
        object.__setattr__(self, "max_t_shape_lateral_offset_A", float(max_t_shape_lateral_offset_A))
        object.__setattr__(self, "aromatic_only", bool(aromatic_only))
        object.__setattr__(self, "search_radius_A", search_radius_A)

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
    """Aromatic ring-ring stacking interaction."""

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
    """Identify aromatic pi-stacking interactions between molecular rings."""
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
            criteria.search_radius_A or criteria.max_centroid_distance_A,
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
                        ring1_ref, ring2_ref, distance, angle, offset, subtype = result
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
                                translation_A=tuple(float(v) for v in translation) if lattice is not None else None,
                                metadata={"criteria": _criteria_metadata(criteria)},
                            )
                        )
    return stacks


def find_pi_stacks(target: MolecularCrystal | Sequence, criteria: PiStackingCriteria | None = None) -> list[PiStacking]:
    """Alias for :func:`find_pi_stacking`."""
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
    c1 = np.asarray(ring1.centroid_A, dtype=float)
    c2 = np.asarray(ring2.centroid_A, dtype=float) + translation
    centroid_vec = c2 - c1
    centroid_distance = float(np.linalg.norm(centroid_vec))
    if centroid_distance > criteria.max_centroid_distance_A:
        return None
    n1 = np.asarray(ring1.normal, dtype=float)
    n2 = np.asarray(ring2.normal, dtype=float)
    raw_angle = vector_angle_deg(n1, n2)
    normal_angle = min(raw_angle, 180.0 - raw_angle)
    n1_norm = np.linalg.norm(n1)
    if n1_norm == 0:
        return None
    n1_unit = n1 / n1_norm
    lateral_vec = centroid_vec - np.dot(centroid_vec, n1_unit) * n1_unit
    lateral_offset = float(np.linalg.norm(lateral_vec))
    subtype = _classify_pi_stacking(normal_angle, lateral_offset, criteria)
    if subtype is None:
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
    return ring1_ref, ring2_ref, centroid_distance, normal_angle, lateral_offset, subtype


def _classify_pi_stacking(
    normal_angle: float,
    lateral_offset: float,
    criteria: PiStackingCriteria,
) -> PiStackingSubtype | None:
    if normal_angle <= criteria.max_parallel_normal_angle_deg:
        if lateral_offset <= criteria.max_face_centered_offset_A:
            return "face_centered_parallel"
        if lateral_offset <= criteria.max_parallel_lateral_offset_A:
            return "displaced_parallel"
        return None
    if (
        criteria.min_t_shape_normal_angle_deg
        <= normal_angle
        <= criteria.max_t_shape_normal_angle_deg
        and lateral_offset <= criteria.max_t_shape_lateral_offset_A
    ):
        return "T_shape"
    return None


def _translation(lattice, image: tuple[int, int, int]) -> np.ndarray:
    if lattice is None:
        return np.zeros(3)
    return np.asarray(image_translation(lattice, image), dtype=float)


def _criteria_metadata(criteria: PiStackingCriteria) -> dict[str, Any]:
    return {
        "max_centroid_distance_A": criteria.max_centroid_distance_A,
        "max_parallel_normal_angle_deg": criteria.max_parallel_normal_angle_deg,
        "min_t_shape_normal_angle_deg": criteria.min_t_shape_normal_angle_deg,
        "max_t_shape_normal_angle_deg": criteria.max_t_shape_normal_angle_deg,
        "max_face_centered_offset_A": criteria.max_face_centered_offset_A,
        "max_parallel_lateral_offset_A": criteria.max_parallel_lateral_offset_A,
        "max_t_shape_lateral_offset_A": criteria.max_t_shape_lateral_offset_A,
        "aromatic_only": criteria.aromatic_only,
    }