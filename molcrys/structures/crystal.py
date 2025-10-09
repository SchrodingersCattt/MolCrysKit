"""
Molecular crystal representation.

This module defines the MolecularCrystal class which is the main container
for molecular crystals.
"""

import numpy as np
from typing import List, Tuple
from .molecule import Molecule
import itertools


class MolecularCrystal:
    """
    Main container for a molecular crystal.
    
    Attributes
    ----------
    lattice : np.ndarray
        3x3 array representing the lattice vectors as rows.
    molecules : List[Molecule]
        List of molecules in the crystal.
    pbc : Tuple[bool, bool, bool]
        Periodic boundary conditions along each lattice vector.
    """
    
    def __init__(self, lattice: np.ndarray, molecules: List[Molecule], 
                 pbc: Tuple[bool, bool, bool] = (True, True, True)):
        """
        Initialize a MolecularCrystal.
        
        Parameters
        ----------
        lattice : np.ndarray
            3x3 array representing the lattice vectors as rows.
        molecules : List[Molecule]
            List of molecules in the crystal.
        pbc : Tuple[bool, bool, bool], default=(True, True, True)
            Periodic boundary conditions along each lattice vector.
        """
        self.lattice = np.array(lattice)
        self.molecules = molecules
        self.pbc = pbc
    
    def get_supercell(self, n1: int, n2: int, n3: int) -> 'MolecularCrystal':
        """
        Create a supercell of the crystal.
        
        Parameters
        ----------
        n1, n2, n3 : int
            Supercell dimensions along each lattice vector.
            
        Returns
        -------
        MolecularCrystal
            New crystal representing the supercell.
        """
        # Create new lattice vectors
        new_lattice = np.array([
            self.lattice[0] * n1,
            self.lattice[1] * n2,
            self.lattice[2] * n3
        ])
        
        # Generate new molecules by replicating in all directions
        new_molecules = []
        for i, j, k in itertools.product(range(n1), range(n2), range(n3)):
            # Translation vector for this cell
            translation = np.array([
                float(i) / n1,
                float(j) / n2,
                float(k) / n3
            ])
            
            # Copy all molecules and translate them
            for molecule in self.molecules:
                new_molecule = Molecule(
                    atoms=[atom.copy() for atom in molecule.atoms],
                    center_of_mass=molecule.center_of_mass + translation,
                    rotation_matrix=molecule.rotation_matrix.copy() if molecule.rotation_matrix is not None else None
                )
                new_molecule.translate(translation)
                new_molecules.append(new_molecule)
        
        return MolecularCrystal(new_lattice, new_molecules, self.pbc)
    
    def fractional_to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to cartesian coordinates.
        
        Parameters
        ----------
        coords : np.ndarray
            Fractional coordinates.
            
        Returns
        -------
        np.ndarray
            Cartesian coordinates.
        """
        return np.dot(coords, self.lattice)
    
    def cartesian_to_fractional(self, coords: np.ndarray) -> np.ndarray:
        """
        Convert cartesian coordinates to fractional coordinates.
        
        Parameters
        ----------
        coords : np.ndarray
            Cartesian coordinates.
            
        Returns
        -------
        np.ndarray
            Fractional coordinates.
        """
        return np.dot(coords, np.linalg.inv(self.lattice))
    
    def summary(self) -> str:
        """
        Generate a summary of the crystal.
        
        Returns
        -------
        str
            Summary string describing the crystal.
        """
        summary_str = f"MolecularCrystal:\n"
        summary_str += f"  Lattice vectors:\n"
        for i, vec in enumerate(self.lattice):
            summary_str += f"    a{i+1}: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]\n"
        summary_str += f"  Number of molecules: {len(self.molecules)}\n"
        summary_str += f"  PBC: {self.pbc}\n"
        
        total_atoms = sum(len(mol.atoms) for mol in self.molecules)
        summary_str += f"  Total atoms: {total_atoms}\n"
        
        return summary_str