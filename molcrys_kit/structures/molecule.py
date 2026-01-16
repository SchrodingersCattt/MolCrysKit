"""
Molecule representation for molecular crystals.

This module defines the CrystalMolecule class which represents a rigid body of atoms.
"""

import numpy as np
from typing import Tuple
import networkx as nx

from ase import Atoms
from ase.neighborlist import neighbor_list

from ..constants import (
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
    DEFAULT_NEIGHBOR_CUTOFF,
)


class CrystalMolecule(Atoms):
    """
    A molecule represented as an ASE Atoms object with additional functionality.

    This class inherits from ASE Atoms and adds molecular-specific properties
    and methods including graph representation of internal connectivity.

    Attributes
    ----------
    graph : networkx.Graph
        Graph representation of the molecule's internal connectivity.
    crystal : object, optional
        Reference to the parent crystal structure, used for coordinate conversions.
    """

    def copy(self) -> "CrystalMolecule":
        """
        Create a copy of the CrystalMolecule preserving all metadata arrays.

        Returns
        -------
        CrystalMolecule
            A copy of the molecule with the same properties and metadata.
        """
        # Create a copy of the base Atoms object
        atoms_copy = super().copy()
        
        # Create a new CrystalMolecule from the copied atoms
        new_mol = CrystalMolecule(atoms_copy, crystal=self.crystal, check_pbc=False)
        
        # Copy over any additional attributes that might exist in the original
        if hasattr(self, '_graph') and self._graph is not None:
            # Copy the graph if it exists
            new_mol._graph = self._graph.copy()
        
        return new_mol

    def to_ase(self) -> Atoms:
        """
        Convert the CrystalMolecule to an ASE Atoms object.

        This method returns the CrystalMolecule as an ASE Atoms object,
        preserving its positions and any associated cell information.

        Returns
        -------
        Atoms
            An ASE Atoms object representing this molecule.
        """
        # Since CrystalMolecule already inherits from Atoms, we can simply return self
        # But we'll make sure all relevant properties are preserved
        return Atoms(
            symbols=self.get_chemical_symbols(),
            positions=self.get_positions(),
            cell=self.get_cell()
            if hasattr(self, "get_cell") and self.get_cell() is not None
            else np.zeros((3, 3)),
            pbc=self.get_pbc()
            if hasattr(self, "get_pbc") and self.get_pbc() is not None
            else [False, False, False],
        )

    def __init__(
        self, atoms: Atoms = None, crystal=None, check_pbc: bool = True, **kwargs
    ):
        """
        Initialize a CrystalMolecule.

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object to initialize the molecule with.
        crystal : object, optional
            Parent crystal structure containing this molecule.
        check_pbc : bool, default True
            Whether to check and fix PBC wrapping. Set to False if atoms are
            already known to be contiguous (unwrapped).
        **kwargs : dict
            Additional arguments passed to ASE Atoms constructor.
        """
        from ..constants.config import KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL

        if atoms is not None:
            # Extract custom arrays before initializing
            custom_arrays = {}
            n_atoms = len(atoms)
            
            # Save the custom disorder arrays if they exist
            if KEY_OCCUPANCY in atoms.arrays:
                custom_arrays[KEY_OCCUPANCY] = atoms.arrays[KEY_OCCUPANCY].copy()
            if KEY_DISORDER_GROUP in atoms.arrays:
                custom_arrays[KEY_DISORDER_GROUP] = atoms.arrays[KEY_DISORDER_GROUP].copy()
            if KEY_ASSEMBLY in atoms.arrays:
                custom_arrays[KEY_ASSEMBLY] = atoms.arrays[KEY_ASSEMBLY].copy()
            if KEY_LABEL in atoms.arrays:
                custom_arrays[KEY_LABEL] = atoms.arrays[KEY_LABEL].copy()
            
            # Initialize from existing ASE Atoms object
            super().__init__(atoms)
            
            # Restore the custom disorder arrays after initialization
            for key, value in custom_arrays.items():
                # Make sure the array has the right length
                if len(value) == n_atoms:
                    self.set_array(key, value)
                else:
                    # If lengths don't match, use default values
                    if key == KEY_OCCUPANCY:
                        self.set_array(key, np.full(n_atoms, 1.0))
                    elif key == KEY_DISORDER_GROUP:
                        self.set_array(key, np.full(n_atoms, 0, dtype=int))
                    elif key == KEY_ASSEMBLY:
                        self.set_array(key, np.array([''] * n_atoms))
                    elif key == KEY_LABEL:
                        self.set_array(key, np.array(self.get_chemical_symbols()))
        else:
            # Initialize with provided kwargs
            super().__init__(**kwargs)

        self.crystal = crystal
        self._graph = None

        if check_pbc:
            self._adjust_positions_for_pbc()

    def __repr__(self):
        """String representation of the crystal molecule."""
        return f"CrystalMolecule(chemical_formula='{self.get_chemical_formula()}', atoms_count={len(self)})"

    def _adjust_positions_for_pbc(self):
        """
        Adjust atomic positions to be contiguous using graph traversal logic.
        Uses robust bonding thresholds to determine connectivity.
        """
        if len(self) <= 1 or not self.get_pbc().any():
            return

        cell = self.get_cell()
        if np.allclose(cell, 0) or np.abs(np.linalg.det(cell)) < 1e-6:
            return

        # Import locally to avoid circular imports
        from ..analysis.interactions import get_bonding_threshold

        # 1. Build a temporary connectivity graph considering PBC
        i_list, j_list, d_list, D_vectors = neighbor_list(
            "ijdD", self, cutoff=DEFAULT_NEIGHBOR_CUTOFF
        )

        g = nx.Graph()
        g.add_nodes_from(range(len(self)))
        symbols = self.get_chemical_symbols()

        for i, j, distance, vector in zip(i_list, j_list, d_list, D_vectors):
            if i < j:
                # Robust threshold calculation
                rad_i = (
                    get_atomic_radius(symbols[i])
                    if has_atomic_radius(symbols[i])
                    else 0.5
                )
                rad_j = (
                    get_atomic_radius(symbols[j])
                    if has_atomic_radius(symbols[j])
                    else 0.5
                )
                metal_i = is_metal_element(symbols[i])
                metal_j = is_metal_element(symbols[j])

                thresh = get_bonding_threshold(rad_i, rad_j, metal_i, metal_j)

                if distance < thresh:
                    g.add_edge(i, j, distance=distance, vector=vector)

        # 2. Unwrap using BFS from node 0
        visited = {0}
        queue = [0]
        positions = self.get_positions()

        while queue:
            u = queue.pop(0)
            for v in g.neighbors(u):
                if v not in visited:
                    small = min(u, v)
                    large = max(u, v)

                    edge_data = g[small][large]
                    vec_small_to_large = edge_data["vector"]

                    if u == small:
                        shift_vec = vec_small_to_large
                    else:
                        shift_vec = -vec_small_to_large

                    positions[v] = positions[u] + shift_vec

                    visited.add(v)
                    queue.append(v)

        self.set_positions(positions)

    def get_graph(self, neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF) -> nx.Graph:
        if self._graph is None:
            self._build_graph(neighbor_cutoff=neighbor_cutoff)
        return self._graph

    @property
    def graph(self) -> nx.Graph:
        if self._graph is None:
            self._build_graph()
        return self._graph

    def build_graph(self, neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF) -> nx.Graph:
        self._build_graph(neighbor_cutoff)
        return self._graph

    def _build_graph(self, neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF):
        self._graph = nx.Graph()
        symbols = self.get_chemical_symbols()
        for i, symbol in enumerate(symbols):
            self._graph.add_node(i, symbol=symbol)

        if len(self) > 1:
            temp_atoms = Atoms(
                symbols=symbols, positions=self.get_positions(), pbc=False
            )
            i_list, j_list, d_list = neighbor_list(
                "ijd", temp_atoms, cutoff=neighbor_cutoff
            )

            from ..constants import (
                get_atomic_radius,
                has_atomic_radius,
                is_metal_element,
            )
            from ..analysis.interactions import get_bonding_threshold

            radii = [
                get_atomic_radius(s) if has_atomic_radius(s) else 0.5 for s in symbols
            ]
            is_metal_flags = [is_metal_element(s) for s in symbols]

            for i, j, distance in zip(i_list, j_list, d_list):
                if i >= j:
                    continue
                threshold = get_bonding_threshold(
                    radii[i], radii[j], is_metal_flags[i], is_metal_flags[j]
                )
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

