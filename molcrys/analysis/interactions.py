"""
Intermolecular interaction detection.

This module detects weak intermolecular interactions such as hydrogen bonds and van der Waals contacts.
"""

import numpy as np
from typing import List, Tuple
from ..structures.molecule import Molecule
from ..structures.crystal import MolecularCrystal


class Interaction:
    """
    Represents an intermolecular interaction.
    
    Attributes
    ----------
    donor : Molecule
        Donor molecule in the interaction.
    acceptor : Molecule
        Acceptor molecule in the interaction.
    distance : float
        Distance between interaction sites.
    angle : float
        Angle of the interaction (e.g., D-H...A for hydrogen bonds).
    type : str
        Type of interaction ('hydrogen_bond', 'vdw', etc.).
    """
    
    def __init__(self, donor: Molecule, acceptor: Molecule, distance: float, 
                 angle: float, type: str = "interaction"):
        """
        Initialize an Interaction.
        
        Parameters
        ----------
        donor : Molecule
            Donor molecule in the interaction.
        acceptor : Molecule
            Acceptor molecule in the interaction.
        distance : float
            Distance between interaction sites.
        angle : float
            Angle of the interaction.
        type : str, default="interaction"
            Type of interaction.
        """
        self.donor = donor
        self.acceptor = acceptor
        self.distance = distance
        self.angle = angle
        self.type = type
    
    def __repr__(self):
        return f"Interaction(type={self.type}, distance={self.distance:.3f}, angle={self.angle:.1f})"


def detect_hydrogen_bonds(crystal: MolecularCrystal, max_distance: float = 3.5, 
                          min_angle: float = 120.0) -> List[Interaction]:
    """
    Detect hydrogen bonds in a molecular crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.
    max_distance : float, default=3.5
        Maximum distance (in Angstroms) for hydrogen bonding.
    min_angle : float, default=120.0
        Minimum angle (in degrees) for hydrogen bonding.
        
    Returns
    -------
    List[Interaction]
        List of detected hydrogen bonds.
    """
    interactions = []
    
    # Simplified implementation - in a real case, we would:
    # 1. Identify hydrogen atoms and potential donor/acceptor atoms
    # 2. Calculate distances and angles according to D-H...A geometry
    # 3. Apply criteria for hydrogen bonding
    
    # For demonstration purposes, we'll generate some dummy interactions
    # if there are multiple molecules
    if len(crystal.molecules) > 1:
        for i, mol1 in enumerate(crystal.molecules[:-1]):
            for mol2 in crystal.molecules[i+1:]:
                # Calculate a dummy distance and angle
                com1 = mol1.center_of_mass
                com2 = mol2.center_of_mass
                distance = np.linalg.norm(com1 - com2)
                
                # Only consider molecules that are close enough
                if distance < max_distance / 5.0:  # Simplified check
                    # Dummy angle
                    angle = 150.0
                    
                    interaction = Interaction(
                        donor=mol1,
                        acceptor=mol2,
                        distance=distance * 5.0,  # Convert to Angstrom-like scale
                        angle=angle,
                        type="hydrogen_bond"
                    )
                    interactions.append(interaction)
    
    return interactions


def detect_vdw_contacts(crystal: MolecularCrystal, threshold_factor: float = 1.2) -> List[Interaction]:
    """
    Detect van der Waals contacts between molecules.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.
    threshold_factor : float, default=1.2
        Factor to multiply sum of vdW radii with to determine contact threshold.
        
    Returns
    -------
    List[Interaction]
        List of detected van der Waals contacts.
    """
    interactions = []
    
    # Simplified implementation
    # A real implementation would use actual van der Waals radii
    
    if len(crystal.molecules) > 1:
        for i, mol1 in enumerate(crystal.molecules[:-1]):
            for mol2 in crystal.molecules[i+1:]:
                # Calculate distance between centers of mass
                com1 = mol1.center_of_mass
                com2 = mol2.center_of_mass
                distance = np.linalg.norm(com1 - com2)
                
                # Dummy vdW contact detection
                if distance < 0.3 * threshold_factor:  # Simplified check
                    interaction = Interaction(
                        donor=mol1,
                        acceptor=mol2,
                        distance=distance * 5.0,  # Convert to Angstrom-like scale
                        angle=180.0,  # Dummy angle
                        type="vdw"
                    )
                    interactions.append(interaction)
    
    return interactions


def analyze_interactions(crystal: MolecularCrystal) -> List[Interaction]:
    """
    Perform comprehensive interaction analysis.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to analyze.
        
    Returns
    -------
    List[Interaction]
        List of all detected interactions.
    """
    hbonds = detect_hydrogen_bonds(crystal)
    vdw_contacts = detect_vdw_contacts(crystal)
    
    return hbonds + vdw_contacts