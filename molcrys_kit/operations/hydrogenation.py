"""
Hydrogenation operations for molecular crystals.

This module provides functionality to add hydrogen atoms to molecular crystals
based on geometric rules and chemical constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ase import Atoms
from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..utils.geometry import (
    get_missing_vectors,
    calculate_dihedral_and_adjustment,
    normalize_vector
)


class Hydrogenator:
    """
    Class for adding hydrogen atoms to molecular crystals based on geometric rules.
    """
    
    def __init__(self, crystal: MolecularCrystal):
        """
        Initialize the hydrogenator with a molecular crystal.
        
        Parameters
        ----------
        crystal : MolecularCrystal
            The molecular crystal to hydrogenate.
        """
        self.crystal = crystal
        # Use unwrapped molecules to handle periodic boundary conditions properly
        self.unwrapped_molecules = crystal.get_unwrapped_molecules()
        
        # Default coordination rules for common elements
        self.default_rules = {
            'C': {'geometry': 'tetrahedral', 'target_coordination': 4},
            'N': {'geometry': 'trigonal_pyramidal', 'target_coordination': 3},  # Default for N
            'O': {'geometry': 'bent', 'target_coordination': 2},
            'S': {'geometry': 'tetrahedral', 'target_coordination': 4},
            'P': {'geometry': 'tetrahedral', 'target_coordination': 4},
        }
        
        # Default bond lengths (in Angstroms)
        self.default_bond_lengths = {
            'C-H': 1.09,
            'N-H': 1.01,
            'O-H': 0.96,
            'S-H': 1.34,
            'P-H': 1.42,
        }

    def add_hydrogens(
        self, 
        rules: Optional[Dict] = None, 
        bond_lengths: Optional[Dict] = None
    ) -> MolecularCrystal:
        """
        Add hydrogen atoms to the crystal based on geometric rules.
        
        Parameters
        ----------
        rules : Optional[Dict]
            Override rules for coordination geometry. Format:
            {
                "global_overrides": {
                    "N": {"geometry": "tetrahedral", "target_coordination": 4}
                }
            }
        bond_lengths : Optional[Dict]
            Override bond lengths for specific atom pairs (e.g., "C-H": 1.09).
            
        Returns
        -------
        MolecularCrystal
            New crystal with hydrogen atoms added.
        """
        # Merge default and user-provided rules
        effective_rules = self.default_rules.copy()
        if rules and 'global_overrides' in rules:
            effective_rules.update(rules['global_overrides'])
        
        # Merge default and user-provided bond lengths
        effective_bond_lengths = self.default_bond_lengths.copy()
        if bond_lengths:
            effective_bond_lengths.update(bond_lengths)
        
        # Create new molecules with hydrogen atoms added
        new_molecules = []
        
        for mol_idx, unwrapped_mol in enumerate(self.unwrapped_molecules):
            # Get the original molecule to modify
            original_mol = self.crystal.molecules[mol_idx]
            
            # Get atomic symbols and positions
            symbols = original_mol.get_chemical_symbols()
            positions = original_mol.get_positions()
            
            # Create a new ASE Atoms object to add hydrogens to
            # Start with the same atoms and positions
            new_atoms = original_mol.copy()
            new_positions = positions.copy()
            
            # Get the connectivity graph
            graph = unwrapped_mol.graph
            
            # For each atom in the molecule, check if it needs hydrogens
            for atom_idx in range(len(symbols)):
                symbol = symbols[atom_idx]
                
                # Skip if this atom type doesn't have hydrogenation rules
                if symbol not in effective_rules:
                    continue
                
                # Get the rule for this atom type
                rule = effective_rules[symbol]
                target_coord = rule['target_coordination']
                
                # Find current coordination number (number of bonded atoms)
                neighbors = list(graph.neighbors(atom_idx))
                current_coord = len(neighbors)
                
                # Calculate how many hydrogens to add
                h_to_add = target_coord - current_coord
                if h_to_add <= 0:
                    continue  # Already has enough neighbors
                
                # Get the position of the center atom
                center_pos = positions[atom_idx]
                
                # Get positions of existing neighbors
                neighbor_positions = [positions[n] for n in neighbors]
                
                # Determine the geometry type for missing vectors
                geometry_type = rule['geometry']
                
                # Adjust geometry type to match function naming
                if geometry_type == 'trigonal_pyramidal':
                    geometry_type = 'tetrahedral'  # N with lone pair is like tetrahedral with 3 connections
                
                # Get the bond length for this atom type
                bond_key = f"{symbol}-H"
                bond_len = effective_bond_lengths.get(bond_key, 1.0)
                
                # Calculate positions for new hydrogen atoms
                missing_vectors = get_missing_vectors(
                    center_pos, 
                    neighbor_positions, 
                    geometry_type, 
                    bond_length=bond_len
                )
                
                # Add only as many hydrogens as needed
                for i in range(min(h_to_add, len(missing_vectors))):
                    h_pos = center_pos + missing_vectors[i]
                    
                    # Append the new hydrogen atom
                    new_atoms = new_atoms + Atoms(symbols=['H'], positions=[h_pos])
            
            # Add the modified molecule to the new molecules list
            new_molecules.append(new_atoms)
        
        # Create a new crystal with the modified molecules
        new_crystal = MolecularCrystal(
            lattice=self.crystal.lattice,
            molecules=new_molecules,
            pbc=self.crystal.pbc
        )
        
        # Now perform optimization step for staggered conformations
        new_crystal = self._optimize_conformations(new_crystal, effective_bond_lengths)
        
        return new_crystal
    
    def _optimize_conformations(
        self, 
        crystal: MolecularCrystal, 
        bond_lengths: Dict
    ) -> MolecularCrystal:
        """
        Optimize conformations to achieve staggered arrangements where possible.
        """
        # For now, we'll implement a basic version that adjusts dihedral angles
        # between sp3 centers connected by single bonds
        
        # Get unwrapped molecules for this crystal
        unwrapped_mols = crystal.get_unwrapped_molecules()
        new_molecules = []
        
        for mol_idx, unwrapped_mol in enumerate(unwrapped_mols):
            original_mol = crystal.molecules[mol_idx]
            symbols = original_mol.get_chemical_symbols()
            positions = original_mol.get_positions()
            
            # Create a copy of the original molecule
            new_mol = original_mol.copy()
            new_symbols = new_mol.get_chemical_symbols()
            new_positions = new_mol.get_positions()
            
            # Get the connectivity graph
            graph = unwrapped_mol.graph
            
            # Look for pairs of sp3 atoms connected by a single bond
            # For now, we'll look for C-N, C-C, N-N bonds where both atoms have H
            sp3_elements = ['C', 'N', 'O', 'S', 'P']  # Common sp3 hybridized atoms
            
            for atom1, atom2 in graph.edges():
                elem1 = symbols[atom1]
                elem2 = symbols[atom2]
                
                # Check if both atoms are sp3 hybridized
                if elem1 in sp3_elements and elem2 in sp3_elements:
                    # Get neighbors of both atoms
                    neighbors1 = [n for n in graph.neighbors(atom1)]
                    neighbors2 = [n for n in graph.neighbors(atom2)]
                    
                    # Get positions of neighbors for dihedral calculation
                    pos1 = positions[atom1]
                    pos2 = positions[atom2]
                    
                    # Calculate the adjustment needed for optimal dihedral angle
                    adjustment = calculate_dihedral_and_adjustment(
                        pos1, pos2,
                        [positions[n] for n in neighbors1 if n != atom2],
                        [positions[n] for n in neighbors2 if n != atom1]
                    )
                    
                    # If adjustment is significant, rotate the attached H atoms
                    if abs(adjustment) > 5:  # Only if significant adjustment needed
                        # Rotate H atoms connected to atom1 around the bond axis
                        self._rotate_hydrogens_around_bond(
                            new_mol, atom1, atom2, adjustment
                        )
            
            new_molecules.append(new_mol)
        
        # Create new crystal with optimized conformations
        optimized_crystal = MolecularCrystal(
            lattice=crystal.lattice,
            molecules=new_molecules,
            pbc=crystal.pbc
        )
        
        return optimized_crystal
    
    def _rotate_hydrogens_around_bond(
        self, 
        mol: Atoms, 
        atom1_idx: int, 
        atom2_idx: int, 
        angle_deg: float
    ):
        """
        Rotate hydrogen atoms connected to atom1 around the atom1-atom2 bond.
        """
        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()
        
        # Find hydrogen atoms connected to atom1
        h_indices = []
        for atom_idx, symbol in enumerate(symbols):
            if symbol == 'H':
                # Check if this hydrogen is connected to atom1 by checking distance
                dist = mol.get_distance(atom1_idx, atom_idx, mic=False)
                if dist < 1.5:  # Typical H-X distance is less than 1.5 Å
                    h_indices.append(atom_idx)
        
        if not h_indices:
            return  # No hydrogens to rotate
        
        # Calculate the rotation axis (bond vector from atom1 to atom2)
        bond_vector = positions[atom2_idx] - positions[atom1_idx]
        rotation_axis = normalize_vector(bond_vector)
        
        # Rotate each hydrogen atom around the bond axis
        new_positions = positions.copy()
        for h_idx in h_indices:
            # Vector from atom1 to hydrogen
            h_vector = positions[h_idx] - positions[atom1_idx]
            
            # Rotate the vector around the bond axis
            rotated_h_vector = self._rotate_vector_around_axis(
                h_vector, rotation_axis, angle_deg
            )
            
            # Update the hydrogen position
            new_positions[h_idx] = positions[atom1_idx] + rotated_h_vector
        
        mol.set_positions(new_positions)
    
    def _rotate_vector_around_axis(
        self, 
        vector: np.ndarray, 
        axis: np.ndarray, 
        angle_deg: float
    ) -> np.ndarray:
        """
        Rotate a vector around an axis by an angle in degrees.
        This is a simplified implementation of Rodrigues' rotation formula.
        """
        angle_rad = np.radians(angle_deg)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Rodrigues' rotation formula
        rotated_vector = (
            vector * cos_angle +
            np.cross(axis, vector) * sin_angle +
            axis * np.dot(axis, vector) * (1 - cos_angle)
        )
        
        return rotated_vector