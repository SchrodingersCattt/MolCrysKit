"""Hydrogen-bond interaction records and detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ...constants.config import BONDING_CONFIG, ELECTRONEGATIVE_ELEMENTS
from ...structures.crystal import MolecularCrystal
from .base import AtomRef, BaseInteraction
from .geometry import enumerate_lattice_images, image_translation, vector_angle_deg
from .identity import ChemicalIdentity, ChemicalIdentityCache
from .local_geometry import AtomLocalGeometry, LocalGeometryCache


@dataclass(frozen=True)
class HydrogenBondCriteria:
    """Geometric and chemical criteria for hydrogen-bond detection."""

    max_h_acceptor_distance_A: float = BONDING_CONFIG["MAX_HYDROGEN_BOND_DISTANCE"]
    min_dha_angle_deg: float = BONDING_CONFIG["MIN_HYDROGEN_BOND_ANGLE"]
    donor_elements: tuple[str, ...] = tuple(ELECTRONEGATIVE_ELEMENTS)
    acceptor_elements: tuple[str, ...] = tuple(ELECTRONEGATIVE_ELEMENTS)
    min_donor_h_distance_A: float = BONDING_CONFIG["MIN_COVALENT_DISTANCE"]
    max_donor_h_distance_A: float = BONDING_CONFIG["MAX_COVALENT_DISTANCE"]
    search_radius_A: float | None = None

    @classmethod
    def from_legacy_max_distance(cls, max_distance: float) -> "HydrogenBondCriteria":
        """Build criteria from the legacy ``max_distance`` argument."""
        return cls(max_h_acceptor_distance_A=float(max_distance))


@dataclass(init=False)
class HydrogenBond(BaseInteraction):
    """Representation of a hydrogen bond interaction.

    The constructor accepts the legacy arguments used by MolCrysKit while also
    allowing richer reference/identity/geometry fields.
    """

    donor: Any
    hydrogen: AtomRef | None
    acceptor: Any
    donor_identity: ChemicalIdentity | None
    acceptor_identity: ChemicalIdentity | None
    donor_geometry: AtomLocalGeometry | None
    acceptor_geometry: AtomLocalGeometry | None
    distance: float
    donor_atom_index: int | None
    hydrogen_index: int | None
    acceptor_atom_index: int | None
    h_acceptor_distance_A: float
    donor_acceptor_distance_A: float | None
    dha_angle_deg: float | None

    def __init__(
        self,
        donor,
        acceptor,
        distance: float | None = None,
        donor_atom_index: int | None = None,
        hydrogen_index: int | None = None,
        acceptor_atom_index: int | None = None,
        *,
        hydrogen: AtomRef | None = None,
        donor_identity: ChemicalIdentity | None = None,
        acceptor_identity: ChemicalIdentity | None = None,
        donor_geometry: AtomLocalGeometry | None = None,
        acceptor_geometry: AtomLocalGeometry | None = None,
        h_acceptor_distance_A: float | None = None,
        donor_acceptor_distance_A: float | None = None,
        dha_angle_deg: float | None = None,
        image: tuple[int, int, int] = (0, 0, 0),
        translation_A: tuple[float, float, float] | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        h_acceptor_distance_A = (
            float(distance)
            if h_acceptor_distance_A is None and distance is not None
            else h_acceptor_distance_A
        )
        if h_acceptor_distance_A is None:
            h_acceptor_distance_A = 0.0

        participants = {}
        if isinstance(donor, AtomRef):
            participants["donor"] = donor
        if hydrogen is not None:
            participants["hydrogen"] = hydrogen
        if isinstance(acceptor, AtomRef):
            participants["acceptor"] = acceptor

        BaseInteraction.__init__(
            self,
            kind="hydrogen_bond",
            participants=participants,
            distance_A=float(h_acceptor_distance_A),
            angle_deg=dha_angle_deg,
            score=score,
            image=tuple(int(v) for v in image),
            translation_A=translation_A,
            metadata=metadata or {},
        )
        self.donor = donor
        self.hydrogen = hydrogen
        self.acceptor = acceptor
        self.donor_identity = donor_identity
        self.acceptor_identity = acceptor_identity
        self.donor_geometry = donor_geometry
        self.acceptor_geometry = acceptor_geometry
        self.distance = float(h_acceptor_distance_A)
        self.donor_atom_index = donor_atom_index
        self.hydrogen_index = hydrogen_index
        self.acceptor_atom_index = acceptor_atom_index
        self.h_acceptor_distance_A = float(h_acceptor_distance_A)
        self.donor_acceptor_distance_A = donor_acceptor_distance_A
        self.dha_angle_deg = dha_angle_deg

    def __repr__(self) -> str:
        """String representation of the hydrogen bond."""
        donor_formula = _participant_formula(self.donor, self.donor_identity, "?")
        acceptor_formula = _participant_formula(self.acceptor, self.acceptor_identity, "?")
        return (
            f"HydrogenBond(donor={donor_formula}, "
            f"acceptor={acceptor_formula}, "
            f"distance={self.distance:.3f} Å)"
        )


def find_hydrogen_bonds(
    target: MolecularCrystal | Sequence,
    max_distance: float = 3.5,
    criteria: HydrogenBondCriteria | None = None,
) -> list[HydrogenBond]:
    """
    Identify potential hydrogen bonds between molecules in a molecular crystal.

    Parameters
    ----------
    target : MolecularCrystal or Sequence[CrystalMolecule]
        Crystal or molecule list to inspect.  Passing a crystal enables basic
        PBC image handling.
    max_distance : float, default=3.5
        Legacy maximum H···A distance.  Ignored when ``criteria`` is provided.
    criteria : HydrogenBondCriteria, optional
        Explicit detection criteria.

    Returns
    -------
    list[HydrogenBond]
        Identified hydrogen-bond records.
    """
    if criteria is None:
        criteria = HydrogenBondCriteria.from_legacy_max_distance(max_distance)

    crystal = target if isinstance(target, MolecularCrystal) else None
    molecules = list(crystal.molecules if crystal is not None else target)
    local_geometries = LocalGeometryCache(molecules)
    identities = ChemicalIdentityCache(crystal) if crystal is not None else None

    if crystal is not None:
        lattice = np.asarray(crystal.lattice, dtype=float)
        pbc = tuple(bool(v) for v in crystal.pbc)
        search_radius = criteria.search_radius_A or criteria.max_h_acceptor_distance_A
        images = enumerate_lattice_images(lattice, pbc, search_radius)
    else:
        lattice = None
        images = ((0, 0, 0),)

    hydrogen_bonds: list[HydrogenBond] = []

    if crystal is None:
        molecule_pairs = []
        for i in range(len(molecules)):
            for j in range(i + 1, len(molecules)):
                molecule_pairs.extend([(i, j), (j, i)])
    else:
        molecule_pairs = [
            (i, j)
            for i in range(len(molecules))
            for j in range(len(molecules))
            if i != j
        ]

    for donor_mol_idx, acceptor_mol_idx in molecule_pairs:
        donor_mol = molecules[donor_mol_idx]
        acceptor_mol = molecules[acceptor_mol_idx]
        donor_positions = donor_mol.get_positions()
        acceptor_positions = acceptor_mol.get_positions()
        donor_symbols = donor_mol.get_chemical_symbols()
        acceptor_symbols = acceptor_mol.get_chemical_symbols()
        donor_lg = local_geometries[donor_mol_idx]
        acceptor_lg = local_geometries[acceptor_mol_idx]

        for donor_atom_idx, donor_symbol in enumerate(donor_symbols):
            if donor_symbol not in criteria.donor_elements:
                continue
            hydrogen_indices = _bonded_hydrogens_with_fallback(
                donor_lg,
                donor_positions,
                donor_symbols,
                donor_atom_idx,
                criteria,
            )
            if not hydrogen_indices:
                continue

            for hydrogen_idx in hydrogen_indices:
                donor_atom_pos = donor_positions[donor_atom_idx]
                h_pos = donor_positions[hydrogen_idx]
                for image in images:
                    translation = (
                        np.asarray(image_translation(lattice, image), dtype=float)
                        if lattice is not None
                        else np.zeros(3)
                    )
                    for acc_idx, acc_symbol in enumerate(acceptor_symbols):
                        if acc_symbol not in criteria.acceptor_elements:
                            continue
                        acc_atom_pos = acceptor_positions[acc_idx] + translation
                        h_acceptor_distance = float(np.linalg.norm(h_pos - acc_atom_pos))
                        if h_acceptor_distance > criteria.max_h_acceptor_distance_A:
                            continue

                        dha_angle = vector_angle_deg(donor_atom_pos - h_pos, acc_atom_pos - h_pos)
                        if dha_angle < criteria.min_dha_angle_deg:
                            continue

                        donor_ref = AtomRef.from_molecule(
                            donor_mol, donor_mol_idx, donor_atom_idx
                        )
                        hydrogen_ref = AtomRef.from_molecule(
                            donor_mol, donor_mol_idx, hydrogen_idx
                        )
                        acceptor_ref = AtomRef.from_molecule(
                            acceptor_mol, acceptor_mol_idx, acc_idx, image=image
                        )
                        donor_identity = identities[donor_mol_idx] if identities else None
                        acceptor_identity = identities[acceptor_mol_idx] if identities else None
                        metadata = {
                            "criteria": {
                                "max_h_acceptor_distance_A": criteria.max_h_acceptor_distance_A,
                                "min_dha_angle_deg": criteria.min_dha_angle_deg,
                                "donor_elements": list(criteria.donor_elements),
                                "acceptor_elements": list(criteria.acceptor_elements),
                            }
                        }
                        hydrogen_bonds.append(
                            HydrogenBond(
                                donor=donor_ref,
                                hydrogen=hydrogen_ref,
                                acceptor=acceptor_ref,
                                distance=h_acceptor_distance,
                                donor_atom_index=donor_atom_idx,
                                hydrogen_index=hydrogen_idx,
                                acceptor_atom_index=acc_idx,
                                donor_identity=donor_identity,
                                acceptor_identity=acceptor_identity,
                                donor_geometry=donor_lg.atom(donor_atom_idx),
                                acceptor_geometry=acceptor_lg.atom(acc_idx),
                                donor_acceptor_distance_A=float(
                                    np.linalg.norm(donor_atom_pos - acc_atom_pos)
                                ),
                                dha_angle_deg=float(dha_angle),
                                image=tuple(int(v) for v in image),
                                translation_A=(
                                    tuple(float(v) for v in translation)
                                    if lattice is not None
                                    else None
                                ),
                                metadata=metadata,
                            )
                        )

    return hydrogen_bonds


def _bonded_hydrogens_with_fallback(
    local_geometry,
    donor_positions,
    donor_symbols,
    donor_atom_idx: int,
    criteria: HydrogenBondCriteria,
) -> tuple[int, ...]:
    hydrogens = local_geometry.bonded_hydrogens(donor_atom_idx)
    if hydrogens:
        return hydrogens

    fallback = []
    donor_pos = donor_positions[donor_atom_idx]
    for idx, symbol in enumerate(donor_symbols):
        if symbol != "H":
            continue
        distance = float(np.linalg.norm(donor_positions[idx] - donor_pos))
        if criteria.min_donor_h_distance_A <= distance <= criteria.max_donor_h_distance_A:
            fallback.append(idx)
    return tuple(fallback)


def _participant_formula(participant, identity: ChemicalIdentity | None, default: str) -> str:
    if identity is not None:
        return identity.formula
    if hasattr(participant, "get_chemical_formula"):
        return participant.get_chemical_formula()
    if isinstance(participant, AtomRef):
        return f"mol{participant.molecule_index}:{participant.symbol}{participant.atom_index}"
    return default