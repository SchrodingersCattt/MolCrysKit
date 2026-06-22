"""Halogen-bond interaction records and detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ...structures.crystal import MolecularCrystal
from ..molecular_identity import ChemicalIdentity, ChemicalIdentityCache
from .base import AtomRef, BaseInteraction, build_crystal_atom_offsets
from .geometry import enumerate_lattice_images, image_translation, vector_angle_deg
from .local_geometry import AtomLocalGeometry, LocalGeometryCache


@dataclass(frozen=True)
class HalogenBondCriteria:
    """Geometric and chemical criteria for halogen-bond detection."""

    max_x_acceptor_distance_A: float = 3.8
    min_dxa_angle_deg: float = 150.0
    halogen_elements: tuple[str, ...] = ("Cl", "Br", "I")
    acceptor_elements: tuple[str, ...] = ("N", "O", "S", "P", "F", "Cl", "Br", "I")
    search_radius_A: float | None = None


@dataclass(init=False)
class HalogenBond(BaseInteraction):
    """Directional D-X···A halogen-bond interaction."""

    donor: AtomRef
    halogen: AtomRef
    acceptor: AtomRef
    donor_identity: ChemicalIdentity | None
    acceptor_identity: ChemicalIdentity | None
    halogen_geometry: AtomLocalGeometry | None
    acceptor_geometry: AtomLocalGeometry | None
    x_acceptor_distance_A: float
    donor_acceptor_distance_A: float | None
    dxa_angle_deg: float

    def __init__(
        self,
        *,
        donor: AtomRef,
        halogen: AtomRef,
        acceptor: AtomRef,
        x_acceptor_distance_A: float,
        dxa_angle_deg: float,
        donor_acceptor_distance_A: float | None = None,
        donor_identity: ChemicalIdentity | None = None,
        acceptor_identity: ChemicalIdentity | None = None,
        halogen_geometry: AtomLocalGeometry | None = None,
        acceptor_geometry: AtomLocalGeometry | None = None,
        image: tuple[int, int, int] = (0, 0, 0),
        translation_A: tuple[float, float, float] | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        BaseInteraction.__init__(
            self,
            kind="halogen_bond",
            participants={"donor": donor, "halogen": halogen, "acceptor": acceptor},
            distance_A=float(x_acceptor_distance_A),
            angle_deg=float(dxa_angle_deg),
            score=score,
            image=tuple(int(v) for v in image),
            translation_A=translation_A,
            metadata=metadata or {},
        )
        self.donor = donor
        self.halogen = halogen
        self.acceptor = acceptor
        self.donor_identity = donor_identity
        self.acceptor_identity = acceptor_identity
        self.halogen_geometry = halogen_geometry
        self.acceptor_geometry = acceptor_geometry
        self.x_acceptor_distance_A = float(x_acceptor_distance_A)
        self.donor_acceptor_distance_A = donor_acceptor_distance_A
        self.dxa_angle_deg = float(dxa_angle_deg)


def find_halogen_bonds(
    target: MolecularCrystal | Sequence,
    criteria: HalogenBondCriteria | None = None,
) -> list[HalogenBond]:
    """Identify directional halogen bonds between molecules."""
    criteria = criteria or HalogenBondCriteria()
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
            criteria.search_radius_A or criteria.max_x_acceptor_distance_A,
        )
    else:
        lattice = None
        images = ((0, 0, 0),)

    bonds: list[HalogenBond] = []
    for donor_mol_idx, acceptor_mol_idx in _directional_pairs(len(molecules), crystal is not None):
        donor_mol = molecules[donor_mol_idx]
        acceptor_mol = molecules[acceptor_mol_idx]
        donor_positions = donor_mol.get_positions()
        acceptor_positions = acceptor_mol.get_positions()
        donor_symbols = donor_mol.get_chemical_symbols()
        acceptor_symbols = acceptor_mol.get_chemical_symbols()
        donor_lg = local_geometries[donor_mol_idx]
        acceptor_lg = local_geometries[acceptor_mol_idx]

        for halogen_idx, halogen_symbol in enumerate(donor_symbols):
            if halogen_symbol not in criteria.halogen_elements:
                continue
            donor_neighbors = donor_lg.neighbors(halogen_idx, heavy_only=True)
            if not donor_neighbors:
                continue
            donor_atom_idx = donor_neighbors[0]
            donor_pos = donor_positions[donor_atom_idx]
            halogen_pos = donor_positions[halogen_idx]

            for image in images:
                if donor_mol_idx == acceptor_mol_idx and image == (0, 0, 0):
                    continue
                translation = _translation(lattice, image)
                for acc_idx, acc_symbol in enumerate(acceptor_symbols):
                    if acc_symbol not in criteria.acceptor_elements:
                        continue
                    acc_pos = acceptor_positions[acc_idx] + translation
                    x_acc_distance = float(np.linalg.norm(halogen_pos - acc_pos))
                    if x_acc_distance > criteria.max_x_acceptor_distance_A:
                        continue
                    dxa_angle = vector_angle_deg(donor_pos - halogen_pos, acc_pos - halogen_pos)
                    if dxa_angle < criteria.min_dxa_angle_deg:
                        continue

                    donor_ref = AtomRef.from_molecule(
                        donor_mol,
                        donor_mol_idx,
                        donor_atom_idx,
                        crystal_atom_offset=atom_offsets[donor_mol_idx],
                    )
                    halogen_ref = AtomRef.from_molecule(
                        donor_mol,
                        donor_mol_idx,
                        halogen_idx,
                        crystal_atom_offset=atom_offsets[donor_mol_idx],
                    )
                    acceptor_ref = AtomRef.from_molecule(
                        acceptor_mol,
                        acceptor_mol_idx,
                        acc_idx,
                        image=image,
                        crystal_atom_offset=atom_offsets[acceptor_mol_idx],
                    )
                    bonds.append(
                        HalogenBond(
                            donor=donor_ref,
                            halogen=halogen_ref,
                            acceptor=acceptor_ref,
                            x_acceptor_distance_A=x_acc_distance,
                            donor_acceptor_distance_A=float(np.linalg.norm(donor_pos - acc_pos)),
                            dxa_angle_deg=float(dxa_angle),
                            donor_identity=identities[donor_mol_idx] if identities else None,
                            acceptor_identity=identities[acceptor_mol_idx] if identities else None,
                            halogen_geometry=donor_lg.atom(halogen_idx),
                            acceptor_geometry=acceptor_lg.atom(acc_idx),
                            image=tuple(int(v) for v in image),
                            translation_A=tuple(float(v) for v in translation) if lattice is not None else None,
                            metadata={"criteria": _criteria_metadata(criteria)},
                        )
                    )
    return bonds


def _directional_pairs(n_molecules: int, include_periodic_context: bool) -> list[tuple[int, int]]:
    if include_periodic_context:
        return [(i, j) for i in range(n_molecules) for j in range(n_molecules)]
    return [(i, j) for i in range(n_molecules) for j in range(n_molecules) if i < j] + [
        (j, i) for i in range(n_molecules) for j in range(n_molecules) if i < j
    ]


def _translation(lattice, image: tuple[int, int, int]) -> np.ndarray:
    if lattice is None:
        return np.zeros(3)
    return np.asarray(image_translation(lattice, image), dtype=float)


def _criteria_metadata(criteria: HalogenBondCriteria) -> dict[str, Any]:
    return {
        "max_x_acceptor_distance_A": criteria.max_x_acceptor_distance_A,
        "min_dxa_angle_deg": criteria.min_dxa_angle_deg,
        "halogen_elements": list(criteria.halogen_elements),
        "acceptor_elements": list(criteria.acceptor_elements),
    }