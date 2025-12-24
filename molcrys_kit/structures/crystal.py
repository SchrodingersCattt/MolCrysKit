"""
Molecular crystal representation.

This module defines the MolecularCrystal class which is the main container
for molecular crystals.
"""

import numpy as np
from typing import List, Tuple

try:
    from ase import Atoms

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = object  # Placeholder for type hints

from .molecule import CrystalMolecule
from ..constants import ATOMIC_RADII
import itertools


class MolecularCrystal:
    """
    Main container for a molecular crystal.

    Attributes
    ----------
    lattice : np.ndarray
        3x3 array representing the lattice vectors as rows.
    molecules : List[CrystalMolecule]
        List of molecules in the crystal, each represented as a CrystalMolecule object.
    pbc : Tuple[bool, bool, bool]
        Periodic boundary conditions along each lattice vector.
    """

    def __init__(
        self,
        lattice: np.ndarray,
        molecules: List[Atoms],
        pbc: Tuple[bool, bool, bool] = (True, True, True),
    ):
        """
        Initialize a MolecularCrystal.

        Parameters
        ----------
        lattice : np.ndarray
            3x3 array representing the lattice vectors as rows.
        molecules : List[Atoms]
            List of molecules in the crystal, each represented as an ASE Atoms object.
        pbc : Tuple[bool, bool, bool], default=(True, True, True)
            Periodic boundary conditions along each lattice vector.
        """
        self.lattice = np.array(lattice)
        # Wrap each ASE Atoms object in a CrystalMolecule
        self.molecules = [CrystalMolecule(mol, self) for mol in molecules]
        self.pbc = pbc

    def get_default_atomic_radii(self):
        """
        Get the default atomic radii parameters.

        Returns
        -------
        dict
            Dictionary containing atomic symbols as keys and their corresponding
            covalent radii (in Angstroms) as values.
        """
        return ATOMIC_RADII.copy()

    def get_supercell(self, n1: int, n2: int, n3: int) -> "MolecularCrystal":
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
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE is required for supercell generation. Please install it with 'pip install ase'"
            )

        # Create new lattice vectors
        new_lattice = np.array(
            [self.lattice[0] * n1, self.lattice[1] * n2, self.lattice[2] * n3]
        )

        # Generate new molecules by replicating in all directions
        new_molecules = []
        for i, j, k in itertools.product(range(n1), range(n2), range(n3)):
            # Translation vector for this cell
            translation = np.array([float(i), float(j), float(k)])

            # Copy all molecules and translate them
            for molecule in self.molecules:
                # Create a copy of the ASE Atoms object
                new_atoms = molecule.copy()
                # Apply translation
                new_atoms.positions += np.dot(translation, self.lattice)
                new_molecules.append(new_atoms)

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

    def get_lattice_vectors(self) -> np.ndarray:
        """
        Get the lattice vectors of the crystal.

        Returns
        -------
        np.ndarray
            3x3 array representing the lattice vectors as rows.
        """
        return self.lattice.copy()

    def get_lattice_parameters(self) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate the lattice parameters (a, b, c, alpha, beta, gamma) of the crystal.

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            Lattice parameters (a, b, c, alpha, beta, gamma) where:
            - a, b, c are the lengths of the lattice vectors in Angstroms
            - alpha, beta, gamma are the angles between the lattice vectors in degrees
        """
        # Get lattice vectors
        a_vec, b_vec, c_vec = self.lattice

        # Calculate lengths of lattice vectors
        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)

        # Calculate angles between lattice vectors
        alpha = np.arccos(np.dot(b_vec, c_vec) / (b * c))
        beta = np.arccos(np.dot(a_vec, c_vec) / (a * c))
        gamma = np.arccos(np.dot(a_vec, b_vec) / (a * b))

        # Convert angles from radians to degrees
        alpha_deg = np.degrees(alpha)
        beta_deg = np.degrees(beta)
        gamma_deg = np.degrees(gamma)

        return (a, b, c, alpha_deg, beta_deg, gamma_deg)

    def summary(self) -> str:
        """
        Generate a summary of the crystal.

        Returns
        -------
        str
            Summary string describing the crystal.
        """
        summary_str = "MolecularCrystal:\n"
        summary_str += "  Lattice vectors:\n"
        for i, vec in enumerate(self.lattice):
            summary_str += f"    a{i+1}: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]\n"
        summary_str += f"  Number of molecules: {len(self.molecules)}\n"
        summary_str += f"  PBC: {self.pbc}\n"

        total_atoms = sum(len(atoms) for atoms in self.molecules)
        summary_str += f"  Total atoms: {total_atoms}\n"

        return summary_str

    def get_unwrapped_molecules(self) -> List[CrystalMolecule]:
        """
        Reconstruct whole molecules across periodic boundaries to form continuous molecules.

        This method performs a graph traversal (BFS) to unwrap molecules that span
        periodic boundaries, ensuring covalent bonds remain intact.

        Returns
        -------
        List[CrystalMolecule]
            List of new CrystalMolecule objects with continuous coordinates.
        """
        unwrapped_molecules = []

        # Pre-calculate inverse lattice matrix for efficiency
        inv_lattice = np.linalg.inv(self.lattice)

        # Process each molecule
        for molecule in self.molecules:
            # Create a copy to work with
            mol_copy = molecule.copy()

            # Get positions and graph
            positions = mol_copy.get_positions()
            graph = molecule.graph

            # Track visited atoms to avoid reprocessing
            visited = set()

            # Process each connected component in the molecule
            for node in graph.nodes():
                if node in visited:
                    continue

                # BFS traversal starting from this node
                queue = [node]
                visited.add(node)

                while queue:
                    u = queue.pop(0)

                    # Check all neighbors of u
                    for v in graph.neighbors(u):
                        if v not in visited:
                            # Calculate distance vector
                            d = positions[v] - positions[u]

                            # Apply Minimum Image Convention (MIC)
                            # Convert to fractional coordinates using pre-calculated inv_lattice
                            frac_d = np.dot(d, inv_lattice)

                            # Apply MIC in fractional coordinates
                            frac_d = frac_d - np.round(frac_d)

                            # Convert back to Cartesian coordinates
                            d = np.dot(frac_d, self.lattice)

                            # Update position of v relative to u
                            positions[v] = positions[u] + d

                            # Mark as visited and add to queue
                            visited.add(v)
                            queue.append(v)

            # Update positions in the molecule copy
            mol_copy.set_positions(positions)

            # Create a new CrystalMolecule with unwrapped coordinates
            unwrapped_molecule = CrystalMolecule(mol_copy, self)
            unwrapped_molecules.append(unwrapped_molecule)

        return unwrapped_molecules
