"""
CIF file parsing for molecular crystals.

This module provides functionality to parse CIF files into MolecularCrystal objects.
"""

import numpy as np
from typing import List, Tuple
import warnings
try:
    from pymatgen.io.cif import CifParser
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from ase import Atoms
    from ase.geometry.analysis import Analysis
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = object  # Placeholder

from ..structures.crystal import MolecularCrystal
from ..constants import get_atomic_radius, has_atomic_radius


def parse_cif(filepath: str) -> MolecularCrystal:
    """
    Parse a CIF file into a MolecularCrystal object.
    
    NOTE: This function puts all atoms into a single molecule without molecular identification.
    For molecular identification, use parse_cif_advanced() instead.
    
    Parameters
    ----------
    filepath : str
        Path to the CIF file.
        
    Returns
    -------
    MolecularCrystal
        Parsed crystal structure.
        
    Raises
    ------
    ImportError
        If pymatgen is not available.
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required for CIF parsing. Please install it with 'pip install pymatgen'")
    
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for molecule representation. Please install it with 'pip install ase'")
    
    # Parse the CIF file using pymatgen
    parser = CifParser(filepath)
    structures = parser.get_structures()
    
    # For simplicity, we take the first structure
    structure = structures[0]
    
    # Extract lattice vectors
    lattice = structure.lattice.matrix
    
    # Create ASE Atoms object
    symbols = [site.species_string for site in structure.sites]
    positions = structure.cart_coords
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)
    
    # For now, we put all atoms in a single molecule
    # A more sophisticated implementation would group atoms into molecules
    # NOTE: If you want molecular identification, use parse_cif_advanced() instead
    molecules = [atoms]
    
    # Add a warning to inform users about the limitation
    warnings.warn("parse_cif() puts all atoms in a single molecule. "
                  "Use parse_cif_advanced() for molecular identification.", 
                  UserWarning)
    
    # Assuming periodic boundary conditions in all directions
    pbc = (True, True, True)
    
    return MolecularCrystal(lattice, molecules, pbc)


def parse_cif_advanced(filepath: str) -> MolecularCrystal:
    """
    Parse a CIF file with advanced molecular grouping.
    
    This function attempts to identify discrete molecular units within the crystal.
    
    Parameters
    ----------
    filepath : str
        Path to the CIF file.
        
    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required for CIF parsing. Please install it with 'pip install pymatgen'")
    
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for molecule representation. Please install it with 'pip install ase'")
    
    # Parse the CIF file using pymatgen
    parser = CifParser(filepath)
    structures = parser.get_structures()
    
    # For simplicity, we take the first structure
    structure = structures[0]
    
    # Extract lattice vectors
    lattice = structure.lattice.matrix
    
    # Create ASE Atoms object
    symbols = [site.species_string for site in structure.sites]
    positions = structure.cart_coords
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)
    
    # Identify molecular units using ASE
    molecules = identify_molecules_with_ase(atoms)
    
    # Print information about identified molecules
    print(f"Identified {len(molecules)} molecular units:")
    for i, molecule in enumerate(molecules):
        print(f"  Molecule {i+1}: {len(molecule)} atoms ({', '.join(set(molecule.get_chemical_symbols()))})")
    
    # Assuming periodic boundary conditions in all directions
    pbc = (True, True, True)
    
    return MolecularCrystal(lattice, molecules, pbc)


def identify_molecules_with_ase(atoms: Atoms) -> List[Atoms]:
    """
    Identify discrete molecular units in a crystal using ASE.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing the crystal structure.
        
    Returns
    -------
    List[Atoms]
        List of ASE Atoms objects, each representing a molecular unit.
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for molecule identification. Please install it with 'pip install ase'")
    
    # Simple distance-based clustering
    # This is a simplified implementation - a production version would be more sophisticated
    
    # Get all atom positions
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Calculate distance matrix
    n_atoms = len(atoms)
    distance_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = atoms.get_distance(i, j, mic=True)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    # Simple bonding criteria based on atomic radii
    adjacency_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        radius_i = get_atomic_radius(symbols[i]) if has_atomic_radius(symbols[i]) else 0.5
        for j in range(i + 1, n_atoms):
            radius_j = get_atomic_radius(symbols[j]) if has_atomic_radius(symbols[j]) else 0.5
            # Bonding threshold as sum of covalent radii
            threshold = (radius_i + radius_j) * 1.2
            if distance_matrix[i, j] < threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    # Find connected components (molecular units)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    sparse_matrix = csr_matrix(adjacency_matrix)
    n_components, labels = connected_components(sparse_matrix, directed=False)
    
    # Create separate Atoms objects for each molecular unit
    molecules = []
    for component_idx in range(n_components):
        component_indices = [i for i in range(n_atoms) if labels[i] == component_idx]
        if component_indices:
            molecule_atoms = atoms[component_indices]
            molecules.append(molecule_atoms)
    
    return molecules