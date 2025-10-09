"""
Constants module for MolCrysKit.

This module provides access to atomic properties such as masses and radii.
The data is loaded from JSON files in the same directory.

Atomic masses are in atomic mass units (amu).
Atomic radii are in Angstroms (Å).
"""

import os
import json

# Get the directory where this module is located
_CONSTANTS_DIR = os.path.dirname(__file__)

# Load atomic masses
with open(os.path.join(_CONSTANTS_DIR, 'atomic_masses.json'), 'r') as f:
    ATOMIC_MASSES = json.load(f)

# Load atomic radii
with open(os.path.join(_CONSTANTS_DIR, 'atomic_radii.json'), 'r') as f:
    ATOMIC_RADII = json.load(f)


def get_atomic_mass(symbol: str) -> float:
    """
    Get the atomic mass of an element.
    
    Parameters
    ----------
    symbol : str
        Chemical symbol of the element (e.g., 'H', 'C', 'O').
        
    Returns
    -------
    float
        Atomic mass in atomic mass units (amu).
        
    Raises
    ------
    KeyError
        If the element symbol is not found in the database.
        
    Examples
    --------
    >>> get_atomic_mass('H')
    1.008
    >>> get_atomic_mass('O')
    15.999
    """
    return ATOMIC_MASSES[symbol]


def get_atomic_radius(symbol: str) -> float:
    """
    Get the atomic radius of an element.
    
    Parameters
    ----------
    symbol : str
        Chemical symbol of the element (e.g., 'H', 'C', 'O').
        
    Returns
    -------
    float
        Atomic radius in Angstroms (Å).
        
    Raises
    ------
    KeyError
        If the element symbol is not found in the database.
        
    Examples
    --------
    >>> get_atomic_radius('H')
    0.37
    >>> get_atomic_radius('O')
    0.66
    """
    return ATOMIC_RADII[symbol]


def has_atomic_mass(symbol: str) -> bool:
    """
    Check if atomic mass data is available for an element.
    
    Parameters
    ----------
    symbol : str
        Chemical symbol of the element.
        
    Returns
    -------
    bool
        True if atomic mass data is available, False otherwise.
        
    Examples
    --------
    >>> has_atomic_mass('H')
    True
    >>> has_atomic_mass('Xx')
    False
    """
    return symbol in ATOMIC_MASSES


def has_atomic_radius(symbol: str) -> bool:
    """
    Check if atomic radius data is available for an element.
    
    Parameters
    ----------
    symbol : str
        Chemical symbol of the element.
        
    Returns
    -------
    bool
        True if atomic radius data is available, False otherwise.
        
    Examples
    --------
    >>> has_atomic_radius('H')
    True
    >>> has_atomic_radius('Xx')
    False
    """
    return symbol in ATOMIC_RADII


def list_elements_with_data() -> dict:
    """
    Get lists of elements with available data.
    
    Returns
    -------
    dict
        Dictionary with 'masses' and 'radii' keys containing lists of element symbols.
    """
    return {
        'masses': list(ATOMIC_MASSES.keys()),
        'radii': list(ATOMIC_RADII.keys())
    }