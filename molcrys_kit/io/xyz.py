"""
XYZ file reader for molecular crystals.

This module provides functionality to read XYZ files into CrystalMolecule objects.
"""

from typing import Optional
import numpy as np

from ase.io import read as ase_read

from ..structures.molecule import CrystalMolecule


def read_xyz(filepath: str, crystal=None) -> CrystalMolecule:
    """
    Read an XYZ file and return a CrystalMolecule object.

    Uses ASE's ``read()`` for robust parsing of standard XYZ format files.

    Parameters
    ----------
    filepath : str
        Path to the XYZ file.
    crystal : MolecularCrystal, optional
        Parent crystal reference to attach to the returned molecule.
        If None, the molecule will have no crystal reference, which means
        fractional-coordinate methods will not be available.

    Returns
    -------
    CrystalMolecule
        The molecule read from the file.  ``crystal`` attribute is set to
        the provided *crystal* argument (or None).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be parsed as a valid XYZ file.

    Examples
    --------
    >>> from molcrys_kit.io.xyz import read_xyz
    >>> mol = read_xyz("guest.xyz")
    >>> print(mol.get_chemical_formula())
    C6H6
    """
    try:
        atoms = ase_read(filepath, format="xyz")
    except FileNotFoundError:
        raise FileNotFoundError(f"XYZ file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Failed to parse XYZ file '{filepath}': {e}")

    # Wrap in CrystalMolecule without PBC checks (XYZ files have no lattice)
    molecule = CrystalMolecule(atoms, crystal=crystal, check_pbc=False)
    return molecule
