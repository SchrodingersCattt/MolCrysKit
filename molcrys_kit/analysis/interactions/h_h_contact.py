"""H···H contact records and detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ...structures.crystal import MolecularCrystal
from ..molecular_identity import ChemicalIdentity, ChemicalIdentityCache
from .base import AtomRef, BaseInteraction, build_crystal_atom_offsets
from .geometry import enumerate_lattice_images, image_translation


@dataclass(frozen=True)
class HHContactCriteria:
    """Distance criteria for intermolecular H···H contacts."""

    min_h_h_distance_A: float = 1.0
    max_h_h_distance_A: float = 2.4
    search_radius_A: float | None = None


@dataclass(init=False)
class HHContact(BaseInteraction):
    """Intermolecular H···H contact."""

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
        BaseInteraction.__init__(
            self,
            kind="h_h_contact",
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
    """Identify intermolecular H···H contacts."""
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
            criteria.search_radius_A or criteria.max_h_h_distance_A,
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
                        if not (criteria.min_h_h_distance_A <= distance <= criteria.max_h_h_distance_A):
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
                                translation_A=tuple(float(v) for v in translation) if lattice is not None else None,
                                metadata={"criteria": _criteria_metadata(criteria)},
                            )
                        )
    return contacts


def _translation(lattice, image: tuple[int, int, int]) -> np.ndarray:
    if lattice is None:
        return np.zeros(3)
    return np.asarray(image_translation(lattice, image), dtype=float)


def _criteria_metadata(criteria: HHContactCriteria) -> dict[str, Any]:
    return {
        "min_h_h_distance_A": criteria.min_h_h_distance_A,
        "max_h_h_distance_A": criteria.max_h_h_distance_A,
    }