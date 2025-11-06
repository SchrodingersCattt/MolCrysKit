"""
Input/Output module for MolCrysKit.

This module handles reading and writing of molecular crystal data.
"""

from .cif import parse_cif, parse_cif_advanced

__all__ = ['parse_cif', 'parse_cif_advanced']