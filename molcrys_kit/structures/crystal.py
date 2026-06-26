"""
Molecular crystal representation.

This module defines the MolecularCrystal class which is the main container
for molecular crystals.
"""

import numpy as np
import networkx as nx

from typing import List, Optional, Tuple

from ase import Atoms
from ase.neighborlist import neighbor_list

from .molecule import CrystalMolecule
from ..constants import (
    ATOMIC_RADII,
    DEFAULT_NEIGHBOR_CUTOFF,
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..utils.geometry import unwrap_positions_along_bonds
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
    formula_moiety : Optional[str]
        Raw CIF _chemical_formula_moiety value when available.
    disorder_provenance : optional
        Source-site audit trail for ordered replicas generated from disorder.
    """

    def __init__(
        self,
        lattice: np.ndarray,
        molecules: List[Atoms],
        pbc: Tuple[bool, bool, bool] = (True, True, True),
        formula_moiety: Optional[str] = None,
        disorder_provenance=None,
        calc_results: Optional[dict] = None,
        metadata: Optional[dict] = None,
        extra_arrays: Optional[dict] = None,
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
        formula_moiety : Optional[str], default=None
            Raw CIF _chemical_formula_moiety value when available.
        disorder_provenance : optional, default=None
            Source-site audit trail for ordered disorder replicas.
        calc_results : Optional[dict], default=None
            Calculator results (energy, forces, stress, etc.) to attach
            when serialising via :meth:`to_ase`.  Populated automatically
            by :meth:`from_ase_atoms` when the source Atoms carries a
            :class:`~ase.calculators.singlepoint.SinglePointCalculator`.
        metadata : Optional[dict], default=None
            Extra per-frame metadata preserved through ExtXYZ ``atoms.info``.
        extra_arrays : Optional[dict], default=None
            Extra per-atom arrays preserved through ExtXYZ ``Properties``
            columns on the flattened ASE Atoms representation.
        """
        from ..constants.config import KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL
        
        self.lattice = np.array(lattice)
        self.pbc = pbc
        self.formula_moiety = formula_moiety
        self.disorder_provenance = disorder_provenance
        self._calc_results: Optional[dict] = calc_results
        self.metadata: dict = dict(metadata or {})
        self.extra_arrays: dict = {
            key: np.asarray(value).copy()
            for key, value in (extra_arrays or {}).items()
        }

        # Wrap each ASE Atoms object in a CrystalMolecule
        self.molecules = []
        for mol in molecules:
            if isinstance(mol, CrystalMolecule):
                # If it's already a CrystalMolecule, just update the reference
                # We assume it's already unwrapped correctly.
                
                # Ensure the atoms object contains the required disorder metadata arrays
                self._ensure_disorder_metadata(mol)
                
                new_mol = (
                    mol.copy()
                )  # Copy ensures we don't mutate the input list objects unexpectedly
                new_mol.crystal = self
                # IMPORTANT: copy() logic in CrystalMolecule needs to respect unwrapped state,
                # but here we manually append to list.
                self.molecules.append(new_mol)
            else:
                # If it's a raw ASE Atoms, wrap it
                # Ensure the atoms object contains the required disorder metadata arrays
                self._ensure_disorder_metadata(mol)
                self.molecules.append(CrystalMolecule(mol, self))

    def _ensure_disorder_metadata(self, atoms_obj):
        """
        Ensures that the atoms object has all required disorder metadata arrays.
        If any are missing, inject default values for the entire structure.
        """
        from ..constants.config import KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL
        
        n_atoms = len(atoms_obj)
        
        # Check if required arrays exist, if not, inject default values
        if KEY_OCCUPANCY not in atoms_obj.arrays:
            atoms_obj.set_array(KEY_OCCUPANCY, np.full(n_atoms, 1.0))
            
        if KEY_DISORDER_GROUP not in atoms_obj.arrays:
            atoms_obj.set_array(KEY_DISORDER_GROUP, np.full(n_atoms, 0, dtype=int))
            
        if KEY_ASSEMBLY not in atoms_obj.arrays:
            atoms_obj.set_array(KEY_ASSEMBLY, np.array([''] * n_atoms))
            
        if KEY_LABEL not in atoms_obj.arrays:
            # Use element symbols as default labels
            atoms_obj.set_array(KEY_LABEL, np.array(atoms_obj.get_chemical_symbols()))

    def __repr__(self):
        """String representation of the molecular crystal."""
        return f"MolecularCrystal(lattice={self.lattice.tolist()}, molecules_count={len(self.molecules)}, pbc={self.pbc})"

    @classmethod
    def from_ase(cls, atoms: Atoms, bond_thresholds=None, max_atoms=None, bond_scale: float = 1.0) -> "MolecularCrystal":
        """
        Create a MolecularCrystal from an ASE Atoms object.

        This method takes an ASE Atoms object and identifies molecular units
        within it using graph-based approach, then creates a MolecularCrystal
        object containing these molecules.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object representing the molecular crystal.
        bond_thresholds : dict, optional
            Custom dictionary with atom pairs as keys and bonding thresholds as values.
            Keys should be tuples of element symbols (e.g., ('H', 'O')), and values should
            be the distance thresholds for bonding in Angstroms.

        Returns
        -------
        MolecularCrystal
            A MolecularCrystal object containing the identified molecular units.
        """
        # Import identify_molecules inside the method to avoid circular import
        from ..io.cif import identify_molecules

        # Extract lattice (cell) from the ASE Atoms object
        lattice = atoms.get_cell()

        # Extract PBC (periodic boundary conditions) from the ASE Atoms object
        pbc = tuple(atoms.get_pbc())

        # Identify molecular units using graph-based approach
        molecules = identify_molecules(atoms, bond_thresholds=bond_thresholds, max_atoms=max_atoms, bond_scale=bond_scale)

        # Create and return a new MolecularCrystal instance
        return cls(lattice, molecules, pbc)

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
            New crystal representing the supercell.  The raw CIF
            `formula_moiety` metadata is not propagated because the repeated
            cell no longer has the same asymmetric-unit formula context.
        """

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
                new_atoms.info.pop("atom_indices", None)
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

    def get_total_nodes(self) -> int:
        """
        Get the total number of atoms (nodes) across all molecules in the crystal.

        Returns
        -------
        int
            Total atom count.
        """
        return sum(len(mol) for mol in self.molecules)

    def get_total_edges(self) -> int:
        """
        Get the total number of bonds (edges) across all molecules in the crystal.
        Triggers graph construction for each molecule if not already built.

        Returns
        -------
        int
            Total edge count.
        """
        # Accessing .graph triggers _build_graph() if self._graph is None
        return sum(mol.graph.number_of_edges() for mol in self.molecules)

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

    def get_unwrapped_molecules(self, max_atoms=None) -> List[CrystalMolecule]:
        """
        Reconstruct whole molecules across periodic boundaries to form continuous molecules.

        Uses robust bonding thresholds instead of hardcoded cutoffs to ensure consistency
        with molecule identification logic.
        """
        from ..analysis.interactions import get_bonding_threshold

        unwrapped_molecules = []

        for molecule in self.molecules:
            # 1. Prepare a temp object with the CRYSTAL's lattice/PBC
            temp_atoms = molecule.to_ase()
            temp_atoms.set_cell(self.lattice)
            temp_atoms.set_pbc(self.pbc)

            symbols = temp_atoms.get_chemical_symbols()

            # 2. Use neighbor_list('D') to get exact vectors
            # Use a slightly larger cutoff to catch all potential bonds
            i_list, j_list, d_list, D_vectors = neighbor_list(
                "ijdD", temp_atoms, cutoff=DEFAULT_NEIGHBOR_CUTOFF
            )

            # 3. Build a temporary graph to traverse
            g = nx.Graph()
            g.add_nodes_from(range(len(temp_atoms)))

            # Add edges based on robust bonding threshold
            for k, (u, v, d_vec, dist) in enumerate(
                zip(i_list, j_list, D_vectors, d_list)
            ):
                if u < v:
                    # Calculate threshold dynamically
                    rad_u = (
                        get_atomic_radius(symbols[u])
                        if has_atomic_radius(symbols[u])
                        else 0.5
                    )
                    rad_v = (
                        get_atomic_radius(symbols[v])
                        if has_atomic_radius(symbols[v])
                        else 0.5
                    )
                    metal_u = is_metal_element(symbols[u])
                    metal_v = is_metal_element(symbols[v])

                    thresh = get_bonding_threshold(rad_u, rad_v, metal_u, metal_v)

                    # Use robust threshold check instead of hardcoded 2.5
                    if dist < thresh:
                        g.add_edge(u, v, vector=d_vec)

            # 4. BFS Traversal to unwrap
            positions, completed = unwrap_positions_along_bonds(
                g,
                range(len(temp_atoms)),
                temp_atoms.get_positions(),
                max_atoms=max_atoms,
            )

            # 5. Create new CrystalMolecule
            new_mol_atoms = temp_atoms.copy()
            new_mol_atoms.set_positions(positions)
            new_mol_atoms.info["unwrap_completed"] = completed
            unwrapped_molecule = CrystalMolecule(new_mol_atoms, self, check_pbc=False)
            unwrapped_molecules.append(unwrapped_molecule)

        return unwrapped_molecules

    def to_ase(self) -> Atoms:
        """
        Convert the MolecularCrystal to an ASE Atoms object.

        This method combines all molecules in the crystal into a single ASE Atoms object,
        preserving their positions and the crystal lattice.  A ``molecule_index``
        per-atom array is stored so that :meth:`from_ase_atoms` can reconstruct
        the original molecule partitioning exactly.

        All standard disorder metadata arrays (``occupancy``, ``disorder_group``,
        ``assembly``, ``label``) are propagated to the flat Atoms.

        Returns
        -------
        Atoms
            An ASE Atoms object representing the entire crystal structure.
        """
        from ..constants.config import (
            KEY_OCCUPANCY, KEY_DISORDER_GROUP,
        )

        n_total = sum(len(molecule) for molecule in self.molecules)
        indices_lists = [
            molecule.info.get("atom_indices")
            for molecule in self.molecules
        ]
        flat_indices = [
            int(index)
            for indices in indices_lists
            if indices is not None
            for index in indices
        ]
        can_restore_order = (
            n_total > 0
            and all(indices is not None for indices in indices_lists)
            and len(flat_indices) == n_total
            and set(flat_indices) == set(range(n_total))
        )

        # --- per-frame / per-atom arrays to propagate ---
        # NOTE: We deliberately skip `assembly` (usually all empty strings) and
        # `label` (usually duplicates `symbols`) because they cause extxyz
        # column-count mismatches and carry no information for non-disordered
        # crystals.  When they do carry information (real disorder), the
        # provenance dict in atoms.info captures the full context.
        base_keys = {"numbers", "positions"}
        skip_string_keys = {"assembly", "label"}
        custom_keys = sorted(
            {
                key
                for molecule in self.molecules
                for key in molecule.arrays.keys()
            }
            - base_keys
            - skip_string_keys
            - {KEY_OCCUPANCY, KEY_DISORDER_GROUP}
        )
        disorder_keys = [KEY_OCCUPANCY, KEY_DISORDER_GROUP] + custom_keys

        if can_restore_order:
            symbols = [None] * n_total
            positions = np.zeros((n_total, 3), dtype=float)
            mol_idx = np.empty(n_total, dtype=int)
            disorder_arrays = {k: [None] * n_total for k in disorder_keys}
            for i_mol, (molecule, indices) in enumerate(zip(self.molecules, indices_lists)):
                molecule_symbols = molecule.get_chemical_symbols()
                molecule_positions = molecule.get_positions()
                for local_index, global_index in enumerate(indices):
                    global_index = int(global_index)
                    symbols[global_index] = molecule_symbols[local_index]
                    positions[global_index] = molecule_positions[local_index]
                    mol_idx[global_index] = i_mol
                for k in disorder_keys:
                    arr = molecule.arrays.get(k)
                    if arr is not None:
                        for local_index, global_index in enumerate(indices):
                            disorder_arrays[k][int(global_index)] = arr[local_index]
        else:
            symbols = []
            positions = []
            mol_idx = np.empty(n_total, dtype=int)
            disorder_arrays = {k: [] for k in disorder_keys}
            offset = 0
            for i_mol, molecule in enumerate(self.molecules):
                symbols.extend(molecule.get_chemical_symbols())
                positions.extend(molecule.get_positions())
                n = len(molecule)
                mol_idx[offset:offset + n] = i_mol
                for k in disorder_keys:
                    arr = molecule.arrays.get(k)
                    if arr is not None:
                        disorder_arrays[k].extend(arr)
                    else:
                        disorder_arrays[k].extend([None] * n)
                offset += n

        atoms = Atoms(
            symbols=symbols, positions=positions, cell=self.lattice, pbc=self.pbc,
        )
        atoms.set_array("molecule_index", mol_idx)

        for key, values in self.extra_arrays.items():
            arr = np.asarray(values)
            if len(arr) != len(atoms):
                raise ValueError(
                    f"Extra array {key!r} has length {len(arr)}; "
                    f"expected {len(atoms)}."
                )
            atoms.set_array(key, arr.copy())

        # --- propagate only arrays with real non-default information ---
        for k in disorder_keys:
            vals = disorder_arrays[k]
            if all(v is not None for v in vals):
                arr = np.array(vals)
                # Only write if there is actual non-default data
                if k == KEY_OCCUPANCY and np.allclose(arr, 1.0):
                    continue
                if k == KEY_DISORDER_GROUP and np.all(arr == 0):
                    continue
                atoms.set_array(k, arr.astype(arr.dtype))

        # --- crystal-level info ---
        atoms.info.update(self.metadata)
        if self.formula_moiety is not None:
            atoms.info["formula_moiety"] = self.formula_moiety
        if self.disorder_provenance is not None:
            import dataclasses
            if hasattr(self.disorder_provenance, "to_dict"):
                atoms.info["disorder_provenance"] = self.disorder_provenance.to_dict()
            elif dataclasses.is_dataclass(self.disorder_provenance):
                atoms.info["disorder_provenance"] = dataclasses.asdict(self.disorder_provenance)
            elif isinstance(self.disorder_provenance, dict):
                atoms.info["disorder_provenance"] = self.disorder_provenance
            else:
                atoms.info["disorder_provenance"] = str(self.disorder_provenance)

        # --- propagate calculator if attached ---
        if self._calc_results is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            atoms.calc = SinglePointCalculator(atoms, **self._calc_results)

        return atoms

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms) -> "MolecularCrystal":
        """
        Reconstruct a MolecularCrystal from a flat ASE Atoms object
        that was produced by :meth:`to_ase`.

        Requires a ``molecule_index`` per-atom array (int).  Falls back to
        :meth:`from_ase` (graph-based molecule identification) if the array
        is missing.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object, typically from :meth:`to_ase` or an extxyz frame.

        Returns
        -------
        MolecularCrystal
        """
        mol_idx = atoms.arrays.get("molecule_index")
        if mol_idx is None:
            return cls.from_ase(atoms)

        from ..constants.config import (
            KEY_OCCUPANCY, KEY_DISORDER_GROUP,
        )

        n_mol = int(mol_idx.max()) + 1
        base_keys = {"numbers", "positions"}
        string_metadata_keys = {"assembly", "label"}
        preserved_array_keys = [
            key
            for key in atoms.arrays.keys()
            if key not in base_keys and key not in string_metadata_keys
        ]
        molecules = []
        for i in range(n_mol):
            mask = mol_idx == i
            indices = np.where(mask)[0]
            sub_atoms = atoms[indices]
            mol = CrystalMolecule(sub_atoms, crystal=None, check_pbc=False)
            mol.info["atom_indices"] = indices.tolist()
            molecules.append(mol)

        info = dict(atoms.info)
        formula_moiety = info.pop("formula_moiety", None)
        disorder_provenance = info.pop("disorder_provenance", None)

        # --- extract calculator results ---
        calc_results = None
        calc = getattr(atoms, "calc", None)
        if calc is not None and hasattr(calc, "results"):
            calc_results = dict(calc.results)

        crystal = cls(
            lattice=atoms.get_cell().array if np.array(atoms.get_cell()).ndim == 2
                     else atoms.get_cell().array,
            molecules=molecules,
            pbc=tuple(atoms.get_pbc()),
            formula_moiety=formula_moiety,
            disorder_provenance=disorder_provenance,
            calc_results=calc_results,
            metadata=info,
            extra_arrays={
                key: np.asarray(atoms.arrays[key]).copy()
                for key in preserved_array_keys
                if key not in {"molecule_index", KEY_OCCUPANCY, KEY_DISORDER_GROUP}
            },
        )
        return crystal
