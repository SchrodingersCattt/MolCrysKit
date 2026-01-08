"""
Atom representation for molecular crystals.

This module defines the Atom class which represents atomic species and coordinates
in molecular crystals.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MolAtom:
    """
    Represents an atom in a molecular crystal.

    Attributes
    ----------
    symbol : str
        Chemical symbol of the atom (e.g., 'C', 'H', 'O').
    frac_coords : np.ndarray
        Fractional coordinates of the atom within the crystal lattice.
    occupancy : float, default=1.0
        Site occupancy of the atom (0.0 to 1.0).
    """

    symbol: str
    frac_coords: np.ndarray
    occupancy: float = 1.0

    def __post_init__(self):
        """Ensure frac_coords is a numpy array."""
        self.frac_coords = np.array(self.frac_coords)

    def __repr__(self):
        """String representation of the atom."""
        return f"MolAtom(symbol='{self.symbol}', frac_coords={self.frac_coords.tolist()}, occupancy={self.occupancy})"

    def to_cartesian(self, lattice: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to cartesian coordinates.

        Parameters
        ----------
        lattice : np.ndarray
            3x3 array representing the lattice vectors as rows.

        Returns
        -------
        np.ndarray
            Cartesian coordinates of the atom.
        """
        return np.dot(self.frac_coords, lattice)

    def copy(self) -> "MolAtom":
        """
        Create a copy of the atom.

        Returns
        -------
        MolAtom
            A copy of the atom with the same properties.
        """
        return MolAtom(
            symbol=self.symbol,
            frac_coords=self.frac_coords.copy(),
            occupancy=self.occupancy,
        )
