"""
CIF file parsing for molecular crystals.

This module provides functionality to parse CIF files into MolecularCrystal objects.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings
import re
import networkx as nx

try:
    from pymatgen.io.cif import CifParser
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = object  # Placeholder

from ..structures.molecule import CrystalMolecule
from ..structures.crystal import MolecularCrystal
from ..constants import get_atomic_radius, has_atomic_radius, is_metal_element, METAL_THRESHOLD_FACTOR, NON_METAL_THRESHOLD_FACTOR


def _clean_species_string(species_string: str) -> str:
    """
    Clean species string by removing occupancy information and other non-element data.
    
    For example:
    - 'O:0.5' -> 'O'
    - 'Fe:?' -> 'Fe'
    - 'Ni:1.0(3)' -> 'Ni'
    
    Parameters
    ----------
    species_string : str
        Raw species string from pymatgen site.
        
    Returns
    -------
    str
        Cleaned atomic symbol.
    """
    # Pre-compile regular expressions for better performance
    _CLEAN_PATTERN = re.compile(r':.*')
    _ELEMENT_PATTERN = re.compile(r'[A-Z][a-z]?')

    cleaned = _CLEAN_PATTERN.split(species_string, 1)[0]
    # Extract only the alphabetic part as the element symbol
    element_match = _ELEMENT_PATTERN.search(cleaned)
    return element_match.group(0) if element_match else cleaned


def read_mol_crystal(filepath: str, bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None) -> MolecularCrystal:
    """
    Parse a CIF file with advanced molecular grouping.
    
    This function attempts to identify discrete molecular units within the crystal.
    
    Parameters
    ----------
    filepath : str
        Path to the CIF file.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
        Keys should be tuples of element symbols (e.g., ('H', 'O')), and values should
        be the distance thresholds for bonding in Angstroms.
        
    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required for CIF parsing. Please install it with 'pip install pymatgen'")
    
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for molecule representation. Please install it with 'pip install ase'")
    
    # Parse the CIF file using pymatgen with special options for handling disordered structures
    # Using occupancy_tolerance to handle disordered structures with '?' or other problematic values
    # Also using more tolerant parameters to handle CIF files with full coordinates
    try:
        parser = CifParser(filepath, occupancy_tolerance=10, site_tolerance=1e-2)
        # Use parse_structures instead of get_structures to avoid deprecation warning
        try:
            structures = parser.parse_structures()
        except AttributeError:
            # Fallback for older pymatgen versions
            structures = parser.get_structures()
    except Exception as e:
        print(f"Warning: CIF parsing failed. Trying with more relaxed parameters...")
        parser = CifParser(filepath, occupancy_tolerance=100, site_tolerance=1e-1, frac_tolerance=1e-1)
        try:
            structures = parser.parse_structures()
        except AttributeError:
            # Fallback for older pymatgen versions
            structures = parser.get_structures()
    
    # For simplicity, we take the first structure
    structure = structures[0]
    
    # Extract lattice vectors
    lattice = structure.lattice.matrix
    
    # Create ASE Atoms object with cleaned symbols
    symbols = [_clean_species_string(site.species_string) for site in structure.sites]
    positions = structure.cart_coords
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)
    
    # Identify molecular units using graph-based approach
    molecules = identify_molecules(atoms, bond_thresholds=bond_thresholds)    
    
    # Assuming periodic boundary conditions in all directions
    pbc = (True, True, True)
    
    return MolecularCrystal(lattice, molecules, pbc)


def parse_cif_advanced(filepath: str, bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None) -> MolecularCrystal:
    """
    Parse a CIF file with advanced molecular grouping.
    
    This function attempts to identify discrete molecular units within the crystal.
    
    Parameters
    ----------
    filepath : str
        Path to the CIF file.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
        Keys should be tuples of element symbols (e.g., ('H', 'O')), and values should
        be the distance thresholds for bonding in Angstroms.
        
    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.
        
    Raises
    ------
    DeprecationWarning
        This function is deprecated and will be removed in a future version.
        Use read_mol_crystal() instead.
    """
    warnings.warn(
        "parse_cif_advanced is deprecated and will be removed in a future version. "
        "Use read_mol_crystal instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return read_mol_crystal(filepath, bond_thresholds)


def identify_molecules(atoms: Atoms, bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None) -> List[CrystalMolecule]:
    """
    Identify discrete molecular units in a crystal using graph-based approach.
    
    This function builds a graph of all atoms in the crystal structure, connecting
    atoms that are likely bonded based on distance criteria. Connected components
    in this graph represent individual molecules.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing the crystal structure.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
        Keys should be tuples of element symbols (e.g., ('H', 'O')), and values should
        be the distance thresholds for bonding in Angstroms.
        
    Returns
    -------
    List[CrystalMolecule]
        List of CrystalMolecule objects, each representing a molecular unit.
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for molecule identification. Please install it with 'pip install ase'")
    
    # Build a global graph for the entire crystal structure
    crystal_graph = nx.Graph()
    
    # Add all atoms as nodes
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    
    for i in range(len(atoms)):
        crystal_graph.add_node(i, symbol=symbols[i], position=positions[i])
    
    # Use ASE neighbor list for efficient bond detection
    i_list, j_list, d_list = neighbor_list('ijd', atoms, cutoff=3.0)  # 3Å should cover most bonds
    
    # Add edges (bonds) based on distance criteria
    for i, j, distance in zip(i_list, j_list, d_list):
        # Check if custom threshold is provided
        pair_key1 = (symbols[i], symbols[j])
        pair_key2 = (symbols[j], symbols[i])
        
        if bond_thresholds and (pair_key1 in bond_thresholds or pair_key2 in bond_thresholds):
            # Use custom threshold if provided
            threshold = bond_thresholds.get(pair_key1, bond_thresholds.get(pair_key2))
        else:
            # Determine threshold based on atomic radii and element types
            radius_i = get_atomic_radius(symbols[i]) if has_atomic_radius(symbols[i]) else 0.5
            radius_j = get_atomic_radius(symbols[j]) if has_atomic_radius(symbols[j]) else 0.5
            is_metal_i = is_metal_element(symbols[i])
            is_metal_j = is_metal_element(symbols[j])
            
            # Determine threshold factor based on element types
            if is_metal_i and is_metal_j:  # Metal-Metal
                factor = METAL_THRESHOLD_FACTOR
            elif not is_metal_i and not is_metal_j:  # Non-metal-Non-metal
                factor = NON_METAL_THRESHOLD_FACTOR
            else:  # Metal-Non-metal
                factor = (METAL_THRESHOLD_FACTOR + NON_METAL_THRESHOLD_FACTOR) / 2
            
            # Bonding threshold as sum of covalent radii multiplied by factor
            threshold = (radius_i + radius_j) * factor
        
        # Add edge if atoms are close enough to be bonded
        if distance < threshold:
            crystal_graph.add_edge(i, j, distance=distance)
    
    # Find connected components (molecular units)
    components = list(nx.connected_components(crystal_graph))
    
    # Create separate CrystalMolecule objects for each molecular unit
    molecules = []
    for component in components:
        # Get indices of atoms in this component
        atom_indices = list(component)
        
        # Extract atoms for this molecule
        molecule_atoms = atoms[atom_indices]
        
        # Create CrystalMolecule object
        molecule = CrystalMolecule(molecule_atoms)
        molecules.append(molecule)
    
    return molecules