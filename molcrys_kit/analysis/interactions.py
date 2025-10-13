"""
Intermolecular interaction detection.

This module detects weak intermolecular interactions such as hydrogen bonds and van der Waals contacts.
"""

import numpy as np
from typing import List, Tuple
from ..structures.molecule import Molecule
from ..structures.crystal import MolecularCrystal
from ..constants import get_atomic_radius, has_atomic_radius


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
                
                # Convert to Cartesian for proper distance measurement
                cart_com1 = crystal.fractional_to_cartesian(com1)
                cart_com2 = crystal.fractional_to_cartesian(com2)
                cartesian_distance = np.linalg.norm(cart_com1 - cart_com2)
                
                # Only consider molecules that are close enough
                if cartesian_distance < max_distance:
                    # Dummy angle
                    angle = 150.0
                    
                    interaction = Interaction(
                        donor=mol1,
                        acceptor=mol2,
                        distance=cartesian_distance,
                        angle=angle,
                        type="hydrogen_bond"
                    )
                    interactions.append(interaction)
    
    return interactions


def detect_vdw_contacts(crystal: MolecularCrystal, threshold_factor: float = 1.2) -> List[Interaction]:
    """
    Detect van der Waals contacts between molecules using actual atomic radii.
    
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
    
    if len(crystal.molecules) > 1:
        for i, mol1 in enumerate(crystal.molecules[:-1]):
            com1 = mol1.center_of_mass
            cart_com1 = crystal.fractional_to_cartesian(com1)
            
            # Estimate vdW radius of molecule 1 as average of atomic radii
            mol1_radii = [get_atomic_radius(atom.symbol) for atom in mol1.atoms 
                         if has_atomic_radius(atom.symbol)]
            mol1_vdw_radius = np.mean(mol1_radii) if mol1_radii else 1.0
            
            for mol2 in crystal.molecules[i+1:]:
                com2 = mol2.center_of_mass
                cart_com2 = crystal.fractional_to_cartesian(com2)
                
                # Estimate vdW radius of molecule 2
                mol2_radii = [get_atomic_radius(atom.symbol) for atom in mol2.atoms 
                             if has_atomic_radius(atom.symbol)]
                mol2_vdw_radius = np.mean(mol2_radii) if mol2_radii else 1.0
                
                # Calculate distance between centers of mass
                distance = np.linalg.norm(cart_com1 - cart_com2)
                
                # Check for contact based on vdW radii
                contact_threshold = (mol1_vdw_radius + mol2_vdw_radius) * threshold_factor
                if distance < contact_threshold:
                    interaction = Interaction(
                        donor=mol1,
                        acceptor=mol2,
                        distance=distance,
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