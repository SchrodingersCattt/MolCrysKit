"""
Molecular representation for molecular crystals.

This module defines the CrystalMolecule class which represents molecules
in molecular crystals with additional functionality.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional
from ase import Atoms as AseAtoms
from ase import Atom as AseAtom

try:
    from ..constants import (
        get_atomic_radius,
        has_atomic_radius,
        is_metal_element,
    )
    # Import the new config
    from ..constants.config import BONDING_CONFIG
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


class CrystalMolecule:
    """
    A molecule represented with ASE Atoms functionality but using composition instead of inheritance.

    This class holds an ASE Atoms object as a private attribute and exposes only necessary methods
    by delegation, adding molecular-specific properties and methods including graph 
    representation of internal connectivity.

    Attributes
    ----------
    graph : networkx.Graph
        Graph representation of the molecule's internal connectivity.
    crystal : object, optional
        Reference to the parent crystal structure, used for coordinate conversions.
    """

    def __init__(self, atoms=None, crystal=None, **kwargs):
        """
        Initialize a CrystalMolecule.

        Parameters
        ----------
        atoms : AseAtoms or CrystalMolecule, optional
            ASE Atoms object or CrystalMolecule to initialize the molecule with.
        crystal : object, optional
            Parent crystal structure containing this molecule.
        **kwargs : dict
            Additional arguments passed to ASE Atoms constructor.
        """
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE is required for CrystalMolecule. Please install it with 'pip install ase'"
            )

        if atoms is not None:
            # Check if atoms is already a CrystalMolecule instance
            if isinstance(atoms, CrystalMolecule):
                # Use the backend atoms from the CrystalMolecule
                self._backend_atoms = AseAtoms(atoms._backend_atoms)
            else:
                # Initialize from existing ASE Atoms object
                self._backend_atoms = AseAtoms(atoms)
        else:
            # Initialize with provided kwargs
            self._backend_atoms = AseAtoms(**kwargs)

        self.crystal = crystal
        self._graph = None
        self._adjust_positions_for_pbc()

    def _adjust_positions_for_pbc(self):
        """
        Adjust atomic positions to be contiguous for molecules that span periodic boundaries.
        """
        # Skip adjustment if no cell is defined or only one atom
        if len(self) <= 1 or not self.get_pbc().any():
            return

        # Get positions
        positions = self.get_positions()

        # Use the first atom as reference
        reference_pos = positions[0]

        # Adjust all other atoms to be close to the reference atom considering PBC
        cell = self.get_cell()
        # Check if cell is defined (not all zeros)
        if np.allclose(cell, 0):
            return

        for i in range(1, len(positions)):
            # Calculate the difference vector
            diff = positions[i] - reference_pos

            # Apply minimum image convention
            # Transform to fractional coordinates
            try:
                frac_diff = np.linalg.solve(cell.T, diff)
            except np.linalg.LinAlgError:
                # If cell is singular, skip PBC adjustment
                return

            # Wrap fractional coordinates to [-0.5, 0.5]
            frac_diff = frac_diff - np.round(frac_diff)

            # Transform back to Cartesian coordinates
            diff_wrapped = np.dot(frac_diff, cell)

            # Update position
            positions[i] = reference_pos + diff_wrapped

        # Set the adjusted positions
        self.set_positions(positions)

    @property
    def graph(self) -> nx.Graph:
        """
        Get the graph representation of the molecule's internal connectivity.

        Returns
        -------
        networkx.Graph
            Graph with atoms as nodes and bonds as edges.
        """
        if self._graph is None:
            self._build_graph()
        return self._graph

    def _build_graph(self):
        """
        Build the graph representation of the molecule's internal connectivity.
        """
        self._graph = nx.Graph()

        # Add nodes (atoms)
        for i in range(len(self)):
            self._graph.add_node(i, symbol=self.get_chemical_symbols()[i])

        # Add edges (bonds) based on distance criteria
        if len(self) > 1:
            # Get all atom positions
            positions = self.get_positions()
            symbols = self.get_chemical_symbols()

            # Simple bonding criteria based on atomic radii
            for i in range(len(self)):
                radius_i = (
                    get_atomic_radius(symbols[i])
                    if has_atomic_radius(symbols[i])
                    else BONDING_CONFIG["DEFAULT_ATOMIC_RADIUS"]
                )
                is_metal_i = is_metal_element(symbols[i])
                for j in range(i + 1, len(self)):
                    radius_j = (
                        get_atomic_radius(symbols[j])
                        if has_atomic_radius(symbols[j])
                        else BONDING_CONFIG["DEFAULT_ATOMIC_RADIUS"]
                    )
                    is_metal_j = is_metal_element(symbols[j])

                    # Calculate distance
                    distance = np.linalg.norm(positions[i] - positions[j])

                    # Determine threshold factor based on element types
                    if is_metal_i and is_metal_j:  # Metal-Metal
                        factor = BONDING_CONFIG["METAL_THRESHOLD_FACTOR"]
                    elif not is_metal_i and not is_metal_j:  # Non-metal-Non-metal
                        factor = BONDING_CONFIG["NON_METAL_THRESHOLD_FACTOR"]
                    else:  # Metal-Non-metal
                        factor = BONDING_CONFIG["METAL_NON_METAL_THRESHOLD_FACTOR"]

                    # Bonding threshold as sum of covalent radii multiplied by factor
                    threshold = (radius_i + radius_j) * factor

                    if distance < threshold:
                        self._graph.add_edge(i, j, distance=distance)

    def get_centroid(self) -> np.ndarray:
        """
        Calculate the centroid (geometric center) of the molecule.

        Returns
        -------
        np.ndarray
            Centroid coordinates (x, y, z).
        """
        return np.mean(self.get_positions(), axis=0)

    def get_centroid_frac(self) -> np.ndarray:
        """
        Calculate the centroid (geometric center) of the molecule in fractional coordinates.

        Returns
        -------
        np.ndarray
            Centroid coordinates in fractional coordinates (a, b, c).
        """
        if self.crystal is None:
            raise ValueError(
                "Cannot compute fractional coordinates: 'crystal' reference is not set."
            )
        centroid_cart = self.get_centroid()
        return self.crystal.cartesian_to_fractional(centroid_cart)

    def get_center_of_mass(self) -> np.ndarray:
        """
        Calculate the center of mass of the molecule.

        Returns
        -------
        np.ndarray
            Center of mass coordinates (x, y, z).
        """
        masses = self.get_masses()
        return np.average(self.get_positions(), axis=0, weights=masses)

    def get_ellipsoid_radii(self) -> Tuple[float, float, float]:
        """
        Calculate the radii of the ellipsoid that best fits the molecule.

        This method computes the principal axes based on the distribution of atoms
        in 3D space using singular value decomposition.

        Returns
        -------
        Tuple[float, float, float]
            The three radii (semi-axes) of the fitted ellipsoid, sorted in descending order.
        """
        # Handle single atom case
        if len(self) == 1:
            # For a single atom, use its atomic radius for all three axes
            symbol = self.get_chemical_symbols()[0]
            atomic_radius = (
                get_atomic_radius(symbol) if has_atomic_radius(symbol) else 0.5
            )
            return (atomic_radius, atomic_radius, atomic_radius)

        # Get positions relative to centroid
        centroid = self.get_centroid()
        rel_positions = self.get_positions() - centroid

        # Perform singular value decomposition
        # This gives us the principal axes and their scales
        try:
            _, singular_values, _ = np.linalg.svd(rel_positions, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback in case of SVD failure
            return (0.01, 0.01, 0.01)

        # The singular values represent the scale (spread) along each principal axis
        # We convert them to effective radii by scaling with a factor that accounts
        # for the number of atoms and provides a more realistic size estimate
        n_atoms = len(self)
        # Use sqrt(5) factor to get a more reasonable estimate of the molecular extent
        # This is based on the idea that atoms are roughly distributed in a sphere
        radii = singular_values * np.sqrt(5 / n_atoms)

        # Sort in descending order
        sorted_radii = np.sort(radii)[::-1]

        # Ensure we always return 3 values
        while len(sorted_radii) < 3:
            sorted_radii = np.append(sorted_radii, 0.01)

        return tuple(sorted_radii[:3])

    def get_principal_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the principal axes of the molecule.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The three principal axes as normalized vectors.
        """
        # Handle single atom case
        if len(self) == 1:
            # For a single atom, return arbitrary orthogonal axes
            return (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            )

        # Get positions relative to centroid
        centroid = self.get_centroid()
        rel_positions = self.get_positions() - centroid

        # Perform singular value decomposition to get principal axes
        try:
            u, _, vh = np.linalg.svd(rel_positions, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback in case of SVD failure
            return (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            )

        # The principal axes are the rows of vh (V transpose)
        # They are already normalized by SVD
        axes = vh
        # Ensure we always return 3 axes
        while len(axes) < 3:
            axes = np.append(axes, [np.array([0.0, 0.0, 0.0])], axis=0)

        return tuple(
            axes[i] if np.linalg.norm(axes[i]) > 0 else np.array([1.0, 0.0, 0.0])
            for i in range(3)
        )

    # Delegate necessary ASE Atoms methods to the backend atoms object
    def get_positions(self):
        return self._backend_atoms.get_positions()
    
    def set_positions(self, positions):
        self._backend_atoms.set_positions(positions)
        
    def get_chemical_symbols(self):
        return self._backend_atoms.get_chemical_symbols()
        
    def get_chemical_formula(self):
        return self._backend_atoms.get_chemical_formula()
        
    def get_atomic_numbers(self):
        return self._backend_atoms.get_atomic_numbers()
        
    def get_masses(self):
        return self._backend_atoms.get_masses()
        
    def get_pbc(self):
        return self._backend_atoms.get_pbc()
        
    def get_cell(self):
        return self._backend_atoms.get_cell()
        
    def get_distance(self, a0, a1, mic=False):
        return self._backend_atoms.get_distance(a0, a1, mic=mic)
        
    def __len__(self):
        return len(self._backend_atoms)
        
    def copy(self):
        # Create a new CrystalMolecule with a copy of the backend atoms
        copied_atoms = self._backend_atoms.copy()
        new_molecule = CrystalMolecule(atoms=copied_atoms, crystal=self.crystal)
        return new_molecule