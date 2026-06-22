"""C-H···pi criteria, records, and detector.

The detector searches directional C-H donors pointing toward ring centroids.
Crystal inputs enable periodic-image searches and molecule identity metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ...structures.crystal import MolecularCrystal
from ..molecular_identity import ChemicalIdentity, ChemicalIdentityCache
from .base import AtomRef, BaseInteraction, RingRef, build_crystal_atom_offsets
from .geometry import enumerate_lattice_images, image_translation, vector_angle_deg
from .local_geometry import AtomLocalGeometry, LocalGeometryCache


@dataclass(frozen=True)
class CHPiInteractionCriteria:
    """Geometric thresholds for directional C-H···pi interactions.

    A contact is accepted when the H···ring-centroid distance is within the Å
    cutoff and the C-H-centroid angle exceeds the configured minimum in
    degrees.  Donor carbon elements, aromatic-ring filtering, and periodic
    search radius are configurable.
    """

    max_h_centroid_distance_A: float = 3.2
    min_ch_centroid_angle_deg: float = 120.0
    carbon_elements: tuple[str, ...] = ("C",)
    aromatic_only: bool = True
    search_radius_A: float | None = None


@dataclass(init=False)
class CHPiInteraction(BaseInteraction):
    """Directional C-H donor to ring-centroid interaction record.

    The record stores carbon and hydrogen atom references, the acceptor ring
    reference, molecule identities, local carbon geometry, H···centroid
    distance, C-H-centroid angle, periodic image information, and detector
    metadata.
    """

    carbon: AtomRef
    hydrogen: AtomRef
    ring: RingRef
    donor_identity: ChemicalIdentity | None
    acceptor_identity: ChemicalIdentity | None
    carbon_geometry: AtomLocalGeometry | None
    h_centroid_distance_A: float
    ch_centroid_angle_deg: float

    def __init__(
        self,
        *,
        carbon: AtomRef,
        hydrogen: AtomRef,
        ring: RingRef,
        h_centroid_distance_A: float,
        ch_centroid_angle_deg: float,
        donor_identity: ChemicalIdentity | None = None,
        acceptor_identity: ChemicalIdentity | None = None,
        carbon_geometry: AtomLocalGeometry | None = None,
        image: tuple[int, int, int] = (0, 0, 0),
        translation_A: tuple[float, float, float] | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a C-H···pi interaction from references and metrics."""
        BaseInteraction.__init__(
            self,
            kind="ch_pi",
            participants={"carbon": carbon, "hydrogen": hydrogen, "ring": ring},
            distance_A=float(h_centroid_distance_A),
            angle_deg=float(ch_centroid_angle_deg),
            score=score,
            image=tuple(int(v) for v in image),
            translation_A=translation_A,
            metadata=metadata or {},
        )
        self.carbon = carbon
        self.hydrogen = hydrogen
        self.ring = ring
        self.donor_identity = donor_identity
        self.acceptor_identity = acceptor_identity
        self.carbon_geometry = carbon_geometry
        self.h_centroid_distance_A = float(h_centroid_distance_A)
        self.ch_centroid_angle_deg = float(ch_centroid_angle_deg)


def find_ch_pi(
    target: MolecularCrystal | Sequence,
    criteria: CHPiInteractionCriteria | None = None,
) -> list[CHPiInteraction]:
    """Find directional C-H···pi interactions in a target.

    For each candidate carbon atom, bonded hydrogens are obtained from local
    topology.  Each C-H vector is tested against candidate ring centroids using
    the configured H···centroid distance and C-H-centroid angle thresholds.
    Periodic images are considered only for ``MolecularCrystal`` inputs.
    """
    criteria = criteria or CHPiInteractionCriteria()
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
            criteria.search_radius_A or criteria.max_h_centroid_distance_A,
        )
    else:
        lattice = None
        images = ((0, 0, 0),)

    interactions: list[CHPiInteraction] = []
    for donor_idx, ring_mol_idx in _directional_pairs(len(molecules), crystal is not None):
        donor_mol = molecules[donor_idx]
        ring_mol = molecules[ring_mol_idx]
        donor_positions = donor_mol.get_positions()
        donor_symbols = donor_mol.get_chemical_symbols()
        donor_lg = local_geometries[donor_idx]
        ring_lg = local_geometries[ring_mol_idx]
        rings = ring_lg.rings(aromatic_only=criteria.aromatic_only)

        for carbon_idx, symbol in enumerate(donor_symbols):
            if symbol not in criteria.carbon_elements:
                continue
            hydrogens = donor_lg.bonded_hydrogens(carbon_idx)
            if not hydrogens:
                continue
            carbon_pos = donor_positions[carbon_idx]
            for hydrogen_idx in hydrogens:
                h_pos = donor_positions[hydrogen_idx]
                for ring in rings:
                    for image in images:
                        if donor_idx == ring_mol_idx and image == (0, 0, 0):
                            continue
                        translation = _translation(lattice, image)
                        centroid = np.asarray(ring.centroid_A, dtype=float) + translation
                        h_centroid_distance = float(np.linalg.norm(h_pos - centroid))
                        if h_centroid_distance > criteria.max_h_centroid_distance_A:
                            continue
                        angle = vector_angle_deg(carbon_pos - h_pos, centroid - h_pos)
                        if angle < criteria.min_ch_centroid_angle_deg:
                            continue

                        carbon_ref = AtomRef.from_molecule(
                            donor_mol,
                            donor_idx,
                            carbon_idx,
                            crystal_atom_offset=atom_offsets[donor_idx],
                        )
                        hydrogen_ref = AtomRef.from_molecule(
                            donor_mol,
                            donor_idx,
                            hydrogen_idx,
                            crystal_atom_offset=atom_offsets[donor_idx],
                        )
                        ring_ref = RingRef.from_molecule(
                            ring_mol,
                            ring_mol_idx,
                            ring.atom_indices,
                            is_aromatic=ring.is_aromatic,
                            image=image,
                            crystal_atom_offset=atom_offsets[ring_mol_idx],
                        )
                        interactions.append(
                            CHPiInteraction(
                                carbon=carbon_ref,
                                hydrogen=hydrogen_ref,
                                ring=ring_ref,
                                h_centroid_distance_A=h_centroid_distance,
                                ch_centroid_angle_deg=float(angle),
                                donor_identity=identities[donor_idx] if identities else None,
                                acceptor_identity=identities[ring_mol_idx] if identities else None,
                                carbon_geometry=donor_lg.atom(carbon_idx),
                                image=tuple(int(v) for v in image),
                                translation_A=tuple(float(v) for v in translation) if lattice is not None else None,
                                metadata={"criteria": _criteria_metadata(criteria)},
                            )
                        )
    return interactions


def find_ch_pi_interactions(
    target: MolecularCrystal | Sequence,
    criteria: CHPiInteractionCriteria | None = None,
) -> list[CHPiInteraction]:
    """Alias for :func:`find_ch_pi` with identical behavior."""
    return find_ch_pi(target, criteria=criteria)


def _directional_pairs(n_molecules: int, include_periodic_context: bool) -> list[tuple[int, int]]:
    """Return ordered donor/ring molecule pairs for a directional detector.

    With periodic context, all ordered pairs including self-pairs are returned
    so nonzero images can be considered.  Without periodic context, only
    distinct molecule pairs are returned in both directions.
    """
    if include_periodic_context:
        return [(i, j) for i in range(n_molecules) for j in range(n_molecules)]
    return [(i, j) for i in range(n_molecules) for j in range(n_molecules) if i < j] + [
        (j, i) for i in range(n_molecules) for j in range(n_molecules) if i < j
    ]


def _translation(lattice, image: tuple[int, int, int]) -> np.ndarray:
    """Return the Cartesian translation vector for an image as an array."""
    if lattice is None:
        return np.zeros(3)
    return np.asarray(image_translation(lattice, image), dtype=float)


def _criteria_metadata(criteria: CHPiInteractionCriteria) -> dict[str, Any]:
    """Return a JSON-friendly snapshot of C-H···pi criteria."""
    return {
        "max_h_centroid_distance_A": criteria.max_h_centroid_distance_A,
        "min_ch_centroid_angle_deg": criteria.min_ch_centroid_angle_deg,
        "carbon_elements": list(criteria.carbon_elements),
        "aromatic_only": criteria.aromatic_only,
    }