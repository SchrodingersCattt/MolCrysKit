# molcrys_kit/constants/config.py
"""
Configuration constants for molecular crystal analysis.
"""

# Default bonding configuration parameters
BONDING_CONFIG = {
    "MAX_HYDROGEN_BOND_DISTANCE": 3.5,  # Maximum H...acceptor distance in Å
    "MIN_COVALENT_DISTANCE": 0.8,        # Minimum covalent bond distance in Å
    "MAX_COVALENT_DISTANCE": 1.2,        # Maximum covalent bond distance in Å
    "MIN_HYDROGEN_BOND_ANGLE": 120.0,    # Minimum donor-H-acceptor angle in degrees
}

# Elements commonly involved in hydrogen bonding
ELECTRONEGATIVE_ELEMENTS = ["N", "O", "F", "Cl", "S"]  # Added Cl and S for broader applicability
"""
Analysis of intermolecular interactions in molecular crystals.

This module provides functionality for analyzing various types of intermolecular
interactions such as hydrogen bonds, halogen bonds, and other non-covalent interactions.
"""

import numpy as np
from typing import List
from ..structures.molecule import CrystalMolecule
from ..constants.config import (
    BONDING_CONFIG,
    ELECTRONEGATIVE_ELEMENTS,
)


class HydrogenBond:
    """
    Representation of a hydrogen bond interaction between two molecules.

    A hydrogen bond is typically formed between a hydrogen atom covalently bonded
    to an electronegative atom (donor) and another electronegative atom (acceptor).
    """

    def __init__(
        self,
        donor: CrystalMolecule,
        acceptor: CrystalMolecule,
        distance: float,
        donor_atom_index: int,
        hydrogen_index: int,
        acceptor_atom_index: int,
    ):
        """
        Initialize a HydrogenBond.

        Parameters
        ----------
        donor : CrystalMolecule
            The molecule acting as the hydrogen bond donor.
        acceptor : CrystalMolecule
            The molecule acting as the hydrogen bond acceptor.
        distance : float
            The distance between the hydrogen and acceptor atoms (in Angstroms).
        donor_atom_index : int
            Index of the donor atom (usually N, O, F, etc.) in the donor molecule.
        hydrogen_index : int
            Index of the hydrogen atom in the donor molecule.
        acceptor_atom_index : int
            Index of the acceptor atom (usually N, O, F, etc.) in the acceptor molecule.
        """
        self.donor = donor
        self.acceptor = acceptor
        self.distance = distance
        self.donor_atom_index = donor_atom_index
        self.hydrogen_index = hydrogen_index
        self.acceptor_atom_index = acceptor_atom_index

    def __repr__(self) -> str:
        """String representation of the hydrogen bond."""
        return (
            f"HydrogenBond(donor={self.donor.get_chemical_formula()}, "
            f"acceptor={self.acceptor.get_chemical_formula()}, "
            f"distance={self.distance:.3f} Å)"
        )


def find_hydrogen_bonds(
    molecules: List[CrystalMolecule], max_distance: float = None
) -> List[HydrogenBond]:
    """
    Identify potential hydrogen bonds between molecules in a molecular crystal.

    This function uses geometric criteria to identify potential hydrogen bonds:
    1. Distance between H and acceptor atom must be ≤ max_distance
    2. Donor-H-acceptor angle should be close to linear (≥ 120°)

    Parameters
    ----------
    molecules : List[CrystalMolecule]
        List of molecules in the crystal.
    max_distance : float, default=None
        Maximum distance (in Angstroms) between H and acceptor atoms for a hydrogen bond.
        If None, uses the default from BONDING_CONFIG.

    Returns
    -------
    List[HydrogenBond]
        List of identified hydrogen bonds.
    """
    if max_distance is None:
        max_distance = BONDING_CONFIG["MAX_HYDROGEN_BOND_DISTANCE"]
        
    hydrogen_bonds = []

    # Check all pairs of molecules
    for i, mol1 in enumerate(molecules):
        for j, mol2 in enumerate(molecules):
            if i >= j:  # Avoid duplicate checks and self-comparison
                continue

            # Get atomic positions and symbols
            pos1 = mol1.get_positions()
            symbols1 = mol1.get_chemical_symbols()
            pos2 = mol2.get_positions()
            symbols2 = mol2.get_chemical_symbols()

            # Check for hydrogen bonds in both directions (mol1->mol2 and mol2->mol1)
            for (
                donor_mol,
                acceptor_mol,
                donor_pos,
                acceptor_pos,
                donor_symbols,
                acceptor_symbols,
            ) in [
                (mol1, mol2, pos1, pos2, symbols1, symbols2),
                (mol2, mol1, pos2, pos1, symbols2, symbols1),
            ]:
                # Find hydrogen atoms in the donor molecule
                h_indices = [idx for idx, symbol in enumerate(donor_symbols) if symbol == "H"]
                
                if not h_indices:
                    continue  # No hydrogen atoms in this molecule

                # Find electronegative atoms in the acceptor molecule
                acceptor_indices = [
                    idx for idx, symbol in enumerate(acceptor_symbols) 
                    if symbol in ELECTRONEGATIVE_ELEMENTS
                ]
                
                if not acceptor_indices:
                    continue  # No acceptor atoms in the acceptor molecule

                # Vectorized distance calculation between H atoms and acceptor atoms
                h_positions = np.array([donor_pos[h_idx] for h_idx in h_indices])
                acceptor_positions = np.array([acceptor_pos[acc_idx] for acc_idx in acceptor_indices])
                
                # Calculate all distances at once using broadcasting
                # Shape: (num_h, num_acceptors, 3)
                diff_vectors = h_positions[:, np.newaxis, :] - acceptor_positions[np.newaxis, :, :]
                # Shape: (num_h, num_acceptors)
                distances = np.linalg.norm(diff_vectors, axis=2)

                # Find pairs that satisfy the distance criterion
                valid_pairs = np.where(distances <= max_distance)
                h_valid_idx, acc_valid_idx = valid_pairs
                
                # Process valid pairs
                for h_idx_local, acc_idx_local in zip(h_valid_idx, acc_valid_idx):
                    h_idx = h_indices[h_idx_local]
                    acc_idx = acceptor_indices[acc_idx_local]
                    distance = distances[h_idx_local, acc_idx_local]

                    # Find the donor atom (the electronegative atom bonded to this H in the donor)
                    donor_atom_idx = None
                    h_pos = donor_pos[h_idx]
                    
                    for atom_idx, symbol in enumerate(donor_symbols):
                        if symbol in ELECTRONEGATIVE_ELEMENTS:
                            d_h_distance = np.linalg.norm(donor_pos[atom_idx] - h_pos)
                            # Typical covalent bond distance range
                            if (BONDING_CONFIG["MIN_COVALENT_DISTANCE"] <= d_h_distance <= 
                                BONDING_CONFIG["MAX_COVALENT_DISTANCE"]):
                                donor_atom_idx = atom_idx
                                break

                    if donor_atom_idx is None:
                        continue  # No suitable donor atom found for this hydrogen

                    # Check the donor-H-acceptor angle
                    donor_atom_pos = donor_pos[donor_atom_idx]
                    acc_atom_pos = acceptor_pos[acc_idx]

                    # Vector from H to donor atom
                    dh_vector = donor_atom_pos - h_pos
                    # Vector from H to acceptor atom
                    ha_vector = acc_atom_pos - h_pos

                    # Calculate angle (in radians)
                    cos_angle = np.dot(dh_vector, ha_vector) / (
                        np.linalg.norm(dh_vector) * np.linalg.norm(ha_vector)
                    )
                    # Clamp to avoid numerical errors
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)

                    # Convert to degrees
                    angle_deg = np.degrees(angle)

                    # Acceptable angles are close to linear (180°)
                    if angle_deg >= BONDING_CONFIG["MIN_HYDROGEN_BOND_ANGLE"]:
                        hb = HydrogenBond(
                            donor=donor_mol,
                            acceptor=acceptor_mol,
                            distance=distance,
                            donor_atom_index=donor_atom_idx,
                            hydrogen_index=h_idx,
                            acceptor_atom_index=acc_idx,
                        )
                        hydrogen_bonds.append(hb)

    return hydrogen_bonds
