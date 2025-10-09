"""
Structure builders for molecular crystals.

This module provides functionality to build supercells and other complex structures.
"""

import numpy as np
import itertools
from typing import Tuple
from ..structures.atom import Atom
from ..structures.molecule import Molecule
from ..structures.crystal import MolecularCrystal


def build_supercell(crystal: MolecularCrystal, size: Tuple[int, int, int] = (2, 2, 2)) -> MolecularCrystal:
    """
    Build a supercell of the crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to expand.
    size : Tuple[int, int, int], default=(2, 2, 2)
        Supercell dimensions (n1, n2, n3).
        
    Returns
    -------
    MolecularCrystal
        New crystal representing the supercell.
    """
    return crystal.get_supercell(*size)


def build_surface(crystal: MolecularCrystal, miller_indices: Tuple[int, int, int], 
                  layers: int, vacuum: float = 10.0) -> MolecularCrystal:
    """
    Build a surface slab from a crystal.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to slice.
    miller_indices : Tuple[int, int, int]
        Miller indices of the surface plane.
    layers : int
        Number of layers in the slab.
    vacuum : float, default=10.0
        Vacuum spacing to add (in Angstroms).
        
    Returns
    -------
    MolecularCrystal
        Surface slab structure.
    """
    # This is a simplified implementation
    # A real implementation would need to:
    # 1. Determine the cleavage plane based on Miller indices
    # 2. Slice the crystal along that plane
    # 3. Add vacuum spacing
    
    # For now, we just return a copy of the original crystal
    # In a real implementation, this would be much more complex
    
    # Create new lattice with added vacuum along c-axis (simplified)
    new_lattice = crystal.lattice.copy()
    new_lattice[2, 2] += vacuum
    
    # Copy molecules
    new_molecules = []
    for molecule in crystal.molecules:
        new_atoms = [atom.copy() for atom in molecule.atoms]
        new_molecule = Molecule(new_atoms)
        new_molecules.append(new_molecule)
    
    return MolecularCrystal(new_lattice, new_molecules, (True, True, False))


def build_defect_crystal(crystal: MolecularCrystal, defect_sites: list) -> MolecularCrystal:
    """
    Build a crystal with specified defects.
    
    Parameters
    ----------
    crystal : MolecularCrystal
        The perfect crystal.
    defect_sites : list
        List of defect sites to modify.
        
    Returns
    -------
    MolecularCrystal
        Crystal structure with defects.
    """
    # Create a copy of the crystal
    new_molecules = []
    for molecule in crystal.molecules:
        new_atoms = [atom.copy() for atom in molecule.atoms]
        new_molecule = Molecule(new_atoms)
        new_molecules.append(new_molecule)
    
    # In a real implementation, we would modify specific atoms/molecules
    # according to the defect_sites specification
    
    return MolecularCrystal(crystal.lattice.copy(), new_molecules, crystal.pbc)


def create_multilayer_structure(lattice_vectors: np.ndarray, 
                                molecules: list, 
                                layers: int, 
                                interlayer_spacing: float) -> MolecularCrystal:
    """
    Create a multilayer structure by stacking molecules.
    
    Parameters
    ----------
    lattice_vectors : np.ndarray
        3x3 array of lattice vectors.
    molecules : list
        List of molecules to stack.
    layers : int
        Number of layers to create.
    interlayer_spacing : float
        Distance between layers (in fractional coordinates).
        
    Returns
    -------
    MolecularCrystal
        Multilayer structure.
    """
    # Create new lattice with expanded c-vector
    new_lattice = lattice_vectors.copy()
    new_lattice[2] = lattice_vectors[2] * layers + np.array([0, 0, interlayer_spacing * (layers - 1)])
    
    # Create stacked molecules
    new_molecules = []
    for i in range(layers):
        z_shift = i * interlayer_spacing
        for molecule in molecules:
            # Copy molecule
            new_atoms = [atom.copy() for atom in molecule.atoms]
            new_molecule = Molecule(new_atoms)
            
            # Shift in z-direction
            shift_vector = np.array([0, 0, z_shift])
            new_molecule.translate(shift_vector)
            new_molecules.append(new_molecule)
    
    return MolecularCrystal(new_lattice, new_molecules, (True, True, False))