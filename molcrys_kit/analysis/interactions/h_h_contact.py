"""Intermolecular H···H contact criteria, records, and detector.

The detector reports close-but-not-covalent hydrogen-hydrogen contacts between
distinct molecules or periodic images.  Crystal inputs enable PBC image
handling and molecule identity metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ...structures.crystal import MolecularCrystal
from ..molecular_identity import ChemicalIdentity, ChemicalIdentityCache
from .base import AtomRef, BaseInteraction, build_crystal_atom_offsets
from .geometry import enumerate_lattice_images, image_translation
from .scoring import ScoringParams, composite_score, scaled_cutoff, vdw_radius_sum


@dataclass(frozen=True)
class HHContactCriteria:
    """Distance thresholds for intermolecular H···H contacts.

    Contacts are accepted only when the H···H distance lies between the
    configured minimum and maximum cutoffs in Å.  The minimum avoids reporting
    implausibly short overlaps or covalent-like contacts, while the optional
    search radius controls periodic image enumeration for crystal inputs.
    """

    min_h_h_distance_A: float = 1.0
    max_h_h_distance_A: float = 2.4
    search_radius_A: float | None = None
    scoring_params: ScoringParams = field(default_factory=ScoringParams)


@dataclass(init=False)
class HHContact(BaseInteraction):
    """Intermolecular H···H contact record.

    The record stores both hydrogen references, molecule identities, the H···H
    distance, periodic image information, and detector criteria metadata through
    ``BaseInteraction``.
    """

    hydrogen1: AtomRef
    hydrogen2: AtomRef
    molecule1_identity: ChemicalIdentity | None
    molecule2_identity: ChemicalIdentity | None
    h_h_distance_A: float

    def __init__(
        self,
        *,
        hydrogen1: AtomRef,
        hydrogen2: AtomRef,
        h_h_distance_A: float,
        molecule1_identity: ChemicalIdentity | None = None,
        molecule2_identity: ChemicalIdentity | None = None,
        image: tuple[int, int, int] = (0, 0, 0),
        translation_A: tuple[float, float, float] | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize an H···H contact from two hydrogen references."""
        BaseInteraction.__init__(
            self,
            kind="close_contact",
            participants={"hydrogen1": hydrogen1, "hydrogen2": hydrogen2},
            distance_A=float(h_h_distance_A),
            angle_deg=None,
            score=score,
            image=tuple(int(v) for v in image),
            translation_A=translation_A,
            metadata=metadata or {},
        )
        self.hydrogen1 = hydrogen1
        self.hydrogen2 = hydrogen2
        self.molecule1_identity = molecule1_identity
        self.molecule2_identity = molecule2_identity
        self.h_h_distance_A = float(h_h_distance_A)


def find_h_h_contacts(
    target: MolecularCrystal | Sequence,
    criteria: HHContactCriteria | None = None,
) -> list[HHContact]:
    """Find intermolecular H···H contacts in a crystal or sequence.

    For molecule sequences, only distinct molecule pairs are searched.  For
    crystal inputs, same-molecule pairs are also considered in nonzero periodic
    images.  Detected contacts include atom references with flattened crystal
    atom indices, optional molecule identities, image translation, and criteria
    metadata.
    """
    criteria = criteria or HHContactCriteria()
    crystal = target if isinstance(target, MolecularCrystal) else None
    molecules = list(crystal.molecules if crystal is not None else target)
    atom_offsets = build_crystal_atom_offsets(molecules)
    identities = ChemicalIdentityCache(crystal) if crystal is not None else None

    if crystal is not None:
        lattice = np.asarray(crystal.lattice, dtype=float)
        images = enumerate_lattice_images(
            lattice,
            tuple(bool(v) for v in crystal.pbc),
            criteria.search_radius_A
            or scaled_cutoff(criteria.max_h_h_distance_A, criteria.scoring_params),
        )
    else:
        lattice = None
        images = ((0, 0, 0),)

    contacts: list[HHContact] = []
    for mol1_idx in range(len(molecules)):
        mol1 = molecules[mol1_idx]
        pos1 = mol1.get_positions()
        symbols1 = mol1.get_chemical_symbols()
        h1_indices = [idx for idx, symbol in enumerate(symbols1) if symbol == "H"]
        mol2_start = mol1_idx if crystal is not None else mol1_idx + 1
        for mol2_idx in range(mol2_start, len(molecules)):
            mol2 = molecules[mol2_idx]
            pos2 = mol2.get_positions()
            symbols2 = mol2.get_chemical_symbols()
            h2_indices = [idx for idx, symbol in enumerate(symbols2) if symbol == "H"]
            for image in images:
                if mol1_idx == mol2_idx and image == (0, 0, 0):
                    continue
                translation = _translation(lattice, image)
                for h1_idx in h1_indices:
                    h1_pos = pos1[h1_idx]
                    for h2_idx in h2_indices:
                        h2_pos = pos2[h2_idx] + translation
                        distance = float(np.linalg.norm(h1_pos - h2_pos))
                        if not (
                            criteria.min_h_h_distance_A
                            <= distance
                            <= scaled_cutoff(criteria.max_h_h_distance_A, criteria.scoring_params)
                        ):
                            continue
                        score = composite_score(
                            (
                                (
                                    distance,
                                    vdw_radius_sum("H", "H"),
                                    criteria.scoring_params.close_contact_distance_sigma_A,
                                    "lorentzian",
                                ),
                            )
                        )
                        if score < criteria.scoring_params.score_threshold:
                            continue
                        h1_ref = AtomRef.from_molecule(
                            mol1,
                            mol1_idx,
                            h1_idx,
                            crystal_atom_offset=atom_offsets[mol1_idx],
                        )
                        h2_ref = AtomRef.from_molecule(
                            mol2,
                            mol2_idx,
                            h2_idx,
                            image=image,
                            crystal_atom_offset=atom_offsets[mol2_idx],
                        )
                        contacts.append(
                            HHContact(
                                hydrogen1=h1_ref,
                                hydrogen2=h2_ref,
                                h_h_distance_A=distance,
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
                                        "h_h_distance_A": distance,
                                        "h_h_vdw_sum_A": vdw_radius_sum("H", "H"),
                                    },
                                },
                            )
                        )
    return contacts


def _translation(lattice, image: tuple[int, int, int]) -> np.ndarray:
    """Return the Cartesian translation vector for an image as an array."""
    if lattice is None:
        return np.zeros(3)
    return np.asarray(image_translation(lattice, image), dtype=float)


def _criteria_metadata(criteria: HHContactCriteria) -> dict[str, Any]:
    """Return a JSON-friendly snapshot of H···H contact criteria."""
    return {
        "min_h_h_distance_A": criteria.min_h_h_distance_A,
        "max_h_h_distance_A": criteria.max_h_h_distance_A,
        "prefilter_factor": criteria.scoring_params.prefilter_factor,
        "score_threshold": criteria.scoring_params.score_threshold,
    }