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
from .records import BondRecord, SiteRecord
from ..constants import (
    ATOMIC_RADII,
    DEFAULT_NEIGHBOR_CUTOFF,
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..utils.geometry import unwrap_positions_along_bonds
import itertools


_BOND_RECORDS_INFO_KEY = "_molcrys_bond_records"


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
        self._chemistry = None

        # Wrap each ASE Atoms object in a CrystalMolecule
        self.molecules = []
        for molecule_index, mol in enumerate(molecules):
            if isinstance(mol, CrystalMolecule):
                # If it's already a CrystalMolecule, just update the reference
                # We assume it's already unwrapped correctly.
                
                # Ensure the atoms object contains the required disorder metadata arrays
                self._ensure_disorder_metadata(mol)
                self._ensure_atom_ids(mol, molecule_index)
                
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
                self._ensure_atom_ids(mol, molecule_index)
                self.molecules.append(CrystalMolecule(mol, self))

        self._deduplicate_atom_ids()

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

    @staticmethod
    def _ensure_atom_ids(atoms_obj, molecule_index: int) -> None:
        """Attach deterministic atom identities when a source has none.

        The array is copied by ASE slicing and by :class:`CrystalMolecule`, so
        identities survive ordinary MolCrysKit copy/operation paths. Existing
        caller-provided identities are retained and validated at crystal level.
        """
        from ..constants.config import KEY_ATOM_ID

        if KEY_ATOM_ID in atoms_obj.arrays and len(atoms_obj.arrays[KEY_ATOM_ID]) == len(atoms_obj):
            return
        atoms_obj.set_array(
            KEY_ATOM_ID,
            np.asarray(
                [f"m{molecule_index}:a{local_index}" for local_index in range(len(atoms_obj))],
                dtype=str,
            ),
        )

    def _deduplicate_atom_ids(self) -> None:
        """Guarantee identity uniqueness after replication or concatenation."""
        from ..constants.config import KEY_ATOM_ID

        seen: set[str] = set()
        for molecule_index, molecule in enumerate(self.molecules):
            values = [str(value) for value in molecule.arrays[KEY_ATOM_ID]]
            changed = False
            for local_index, value in enumerate(values):
                candidate = value
                if not candidate or candidate in seen:
                    candidate = f"{value or 'atom'}~m{molecule_index}:a{local_index}"
                    suffix = 2
                    while candidate in seen:
                        candidate = f"{value or 'atom'}~m{molecule_index}:a{local_index}:{suffix}"
                        suffix += 1
                    values[local_index] = candidate
                    changed = True
                seen.add(candidate)
            if changed:
                del molecule.arrays[KEY_ATOM_ID]
                molecule.set_array(KEY_ATOM_ID, np.asarray(values, dtype=str))

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

    @classmethod
    def from_cif(
        cls,
        path: str,
        bond_thresholds=None,
        max_atoms=None,
        bond_scale: float = 1.0,
        use_asu_first: bool = False,
    ) -> "MolecularCrystal":
        """
        Create a MolecularCrystal from a CIF file.

        Parameters
        ----------
        path : str
            Path to CIF file
        bond_thresholds : dict, optional
            Bond distance thresholds by element pair
        max_atoms : int, optional
            Maximum atoms per molecule
        bond_scale : float, optional
            Bond distance scaling factor
        use_asu_first : bool
            Use ASU-first approach (identify molecules on ASU, then replicate)
            This is more efficient but requires proper symmetry handling.

        Returns
        -------
        MolecularCrystal
            Crystal with identified molecules
            
        Raises
        ------
        NotImplementedError
            If use_asu_first=True (feature not yet implemented)
        """
        if use_asu_first:
            # ASU-first path: identify molecules on ASU, then replicate via symops
            from ..io.cif import _identify_molecules_asu_first, read_mol_crystal
            try:
                return _identify_molecules_asu_first(
                    path,
                    bond_thresholds=bond_thresholds,
                    max_atoms=max_atoms,
                    bond_scale=bond_scale,
                )
            except Exception:
                # Fallback to standard path on any failure
                return read_mol_crystal(
                    path,
                    bond_thresholds=bond_thresholds,
                    max_atoms=max_atoms,
                    bond_scale=bond_scale,
                )
        
        # Standard path: expand to full cell, then identify molecules
        from ..io.cif import read_mol_crystal
        return read_mol_crystal(
            path,
            bond_thresholds=bond_thresholds,
            max_atoms=max_atoms,
            bond_scale=bond_scale,
        )

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
        from .molecule import _strip_stale_frac_arrays
        from ..constants.config import KEY_IMAGE_SHIFT

        inverse_new_lattice = np.linalg.inv(new_lattice)
        periodic = np.asarray(self.pbc, dtype=bool)

        new_molecules = []
        for i, j, k in itertools.product(range(n1), range(n2), range(n3)):
            # Translation vector for this cell
            translation = np.array([float(i), float(j), float(k)])

            # Copy all molecules and translate them
            for molecule in self.molecules:
                # Create a copy of the ASE Atoms object
                new_atoms = molecule.copy()
                new_atoms.info.pop("atom_indices", None)
                new_atoms.info.pop("bond_records", None)
                new_atoms.info.pop("bond_pairs", None)
                # Apply translation
                new_atoms.positions += np.dot(translation, self.lattice)
                supercell_fractional = new_atoms.positions @ inverse_new_lattice
                image_shifts = np.zeros((len(new_atoms), 3), dtype=int)
                image_shifts[:, periodic] = np.floor(
                    supercell_fractional[:, periodic] + 1e-10
                ).astype(int)
                new_atoms.set_array(KEY_IMAGE_SHIFT, image_shifts)
                # Supercell lattice differs from the original; frac coords are stale.
                _strip_stale_frac_arrays(new_atoms)
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

    def copy(self) -> "MolecularCrystal":
        """Return an independent crystal copy with all public metadata.

        Per-atom arrays (including crystallographic provenance and ADPs),
        molecule graphs, lattice data, calculator results, frame metadata, and
        extra ExtXYZ arrays are copied rather than shared.
        """
        import copy

        copied = MolecularCrystal(
            lattice=np.asarray(self.lattice, dtype=float).copy(),
            molecules=[molecule.copy() for molecule in self.molecules],
            pbc=tuple(bool(value) for value in self.pbc),
            formula_moiety=self.formula_moiety,
            disorder_provenance=copy.deepcopy(self.disorder_provenance),
            calc_results=copy.deepcopy(self._calc_results),
            metadata=copy.deepcopy(self.metadata),
            extra_arrays={
                key: np.asarray(value).copy()
                for key, value in self.extra_arrays.items()
            },
        )
        copied._set_chemistry(self._chemistry)
        return copied

    @property
    def chemistry(self):
        """Attached immutable :class:`CrystalChemistry`, when analysed."""
        return self._chemistry

    def _set_chemistry(self, chemistry) -> None:
        """Attach an immutable chemistry snapshot through the owning class."""
        self._chemistry = chemistry

    def _molecule_global_indices(self) -> List[List[int]]:
        """Return each molecule's indices in :meth:`to_ase` ordering."""
        n_total = self.get_total_nodes()
        saved = [molecule.info.get("atom_indices") for molecule in self.molecules]
        flat = [int(index) for indices in saved if indices is not None for index in indices]
        if (
            all(indices is not None for indices in saved)
            and len(flat) == n_total
            and set(flat) == set(range(n_total))
        ):
            return [[int(index) for index in indices] for indices in saved]

        result = []
        offset = 0
        for molecule in self.molecules:
            result.append(list(range(offset, offset + len(molecule))))
            offset += len(molecule)
        return result

    def get_site_records(self) -> List[SiteRecord]:
        """Return immutable, serialisable records for every crystal site.

        The result is sorted by ``global_index`` and is the supported public
        alternative to reading molecule ``arrays`` or ``info`` dictionaries.
        Missing displacement parameters are represented by ``None``.
        """
        from ..constants.config import (
            KEY_ASSEMBLY,
            KEY_ATOM_ID,
            KEY_ASYM_ID,
            KEY_DISORDER_GROUP,
            KEY_FRAC_X,
            KEY_FRAC_Y,
            KEY_FRAC_Z,
            KEY_FORMAL_CHARGE,
            KEY_FORMAL_CHARGE_KNOWN,
            KEY_IMAGE_SHIFT,
            KEY_ISOTOPE,
            KEY_LABEL,
            KEY_OCCUPANCY,
            KEY_SITE_SYMMETRY_ORDER,
            KEY_SYM_OP_INDEX,
            KEY_U_CART,
            KEY_UISO,
        )

        records = []
        global_indices = self._molecule_global_indices()
        inv_lattice = np.linalg.pinv(np.asarray(self.lattice, dtype=float))

        for molecule_index, (molecule, molecule_globals) in enumerate(
            zip(self.molecules, global_indices)
        ):
            symbols = molecule.get_chemical_symbols()
            positions = np.asarray(molecule.get_positions(), dtype=float)
            arrays = molecule.arrays
            for local_index, global_index in enumerate(molecule_globals):
                def _array_value(key, default):
                    array = arrays.get(key)
                    return default if array is None else array[local_index]

                asym_value = int(_array_value(KEY_ASYM_ID, -1))
                sym_op_value = int(_array_value(KEY_SYM_OP_INDEX, -1))
                uiso_value = float(_array_value(KEY_UISO, np.nan))
                uiso = uiso_value if np.isfinite(uiso_value) else None

                raw_u_cart = np.asarray(
                    _array_value(KEY_U_CART, np.full(9, np.nan)), dtype=float
                )
                if raw_u_cart.size == 9 and np.all(np.isfinite(raw_u_cart)):
                    matrix = raw_u_cart.reshape(3, 3)
                    u_cart = tuple(
                        tuple(float(value) for value in row) for row in matrix
                    )
                else:
                    u_cart = None

                position = positions[local_index]
                fractional = position @ inv_lattice
                image_array = arrays.get(KEY_IMAGE_SHIFT)
                if image_array is not None:
                    raw_image = np.asarray(image_array[local_index], dtype=int).reshape(3)
                elif all(key in arrays for key in (KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z)):
                    source_fractional = np.array(
                        [
                            arrays[KEY_FRAC_X][local_index],
                            arrays[KEY_FRAC_Y][local_index],
                            arrays[KEY_FRAC_Z][local_index],
                        ],
                        dtype=float,
                    )
                    raw_image = np.rint(fractional - source_fractional).astype(int)
                else:
                    raw_image = np.array(
                        [
                            int(np.floor(value + 1e-10)) if periodic else 0
                            for value, periodic in zip(fractional, self.pbc)
                        ],
                        dtype=int,
                    )
                label = str(_array_value(KEY_LABEL, symbols[local_index]))
                assembly = str(_array_value(KEY_ASSEMBLY, "")).strip()
                if assembly in {"", ".", "?"}:
                    assembly = None

                records.append(
                    SiteRecord(
                        site_id=str(_array_value(KEY_ATOM_ID, f"g{global_index}")),
                        global_index=int(global_index),
                        molecule_index=molecule_index,
                        local_index=local_index,
                        symbol=str(symbols[local_index]),
                        label=label,
                        isotope=(
                            None
                            if int(_array_value(KEY_ISOTOPE, 0)) <= 0
                            else int(_array_value(KEY_ISOTOPE, 0))
                        ),
                        formal_charge=(
                            None
                            if KEY_FORMAL_CHARGE not in arrays
                            or not bool(_array_value(KEY_FORMAL_CHARGE_KNOWN, True))
                            else int(_array_value(KEY_FORMAL_CHARGE, 0))
                        ),
                        cartesian_position_A=tuple(float(v) for v in position),
                        fractional_position=tuple(float(v) for v in fractional),
                        occupancy=float(_array_value(KEY_OCCUPANCY, 1.0)),
                        disorder_group=int(_array_value(KEY_DISORDER_GROUP, 0)),
                        disorder_assembly=assembly,
                        asym_index=None if asym_value < 0 else asym_value,
                        sym_op_index=None if sym_op_value < 0 else sym_op_value,
                        site_symmetry_order=int(
                            _array_value(KEY_SITE_SYMMETRY_ORDER, 1)
                        ),
                        image_shift=tuple(int(v) for v in raw_image),
                        uiso_A2=uiso,
                        u_cart_A2=u_cart,
                    )
                )

        return sorted(records, key=lambda record: record.global_index)

    def get_bond_records(self) -> List[BondRecord]:
        """Return canonical intramolecular bonds with PBC provenance.

        This method is the supported public replacement for the historical
        ``molecule.info['bond_records']`` payload.  It also reconstructs the
        same contract after ASE/ExtXYZ round-trips where molecule ``info`` is
        intentionally not serialised.
        """
        site_records = {
            (record.molecule_index, record.local_index): record
            for record in self.get_site_records()
        }
        global_indices = self._molecule_global_indices()
        result = []

        for molecule_index, (molecule, molecule_globals) in enumerate(
            zip(self.molecules, global_indices)
        ):
            global_to_local = {
                int(global_index): local_index
                for local_index, global_index in enumerate(molecule_globals)
            }
            legacy_records = []
            legacy_pairs = set()
            seen_legacy = set()
            # Endpoint order defines the shift direction for non-self pairs.
            # Do not fold ``shift`` and ``-shift`` together: they can be
            # distinct contacts to different periodic images.
            for raw in molecule.info.get("bond_records", ()):
                try:
                    raw_left = int(raw["left"])
                    raw_right = int(raw["right"])
                    raw_shift = np.asarray(raw["right_image_shift"], dtype=int)
                    raw_vector = np.asarray(raw["vector"], dtype=float)
                except (KeyError, TypeError, ValueError):
                    continue
                pair = frozenset((raw_left, raw_right))
                if pair.issubset(global_to_local):
                    left_global, right_global = sorted((raw_left, raw_right))
                    if raw_left != left_global:
                        raw_shift = -raw_shift
                        raw_vector = -raw_vector
                    key = (
                        left_global,
                        right_global,
                        tuple(int(value) for value in raw_shift),
                    )
                    if key in seen_legacy:
                        continue
                    seen_legacy.add(key)
                    legacy_pairs.add(pair)
                    legacy_records.append(
                        (left_global, right_global, raw_shift, raw_vector)
                    )

            graph_pairs = {
                frozenset(
                    (
                        int(molecule_globals[int(first_local)]),
                        int(molecule_globals[int(second_local)]),
                    )
                )
                for first_local, second_local in molecule.graph.edges()
            }

            for left_global, right_global, right_shift, vector in sorted(
                legacy_records,
                key=lambda item: (
                    item[0],
                    item[1],
                    tuple(int(value) for value in item[2]),
                ),
            ):
                left_local = global_to_local[left_global]
                right_local = global_to_local[right_global]
                left_site = site_records[(molecule_index, left_local)]
                right_site = site_records[(molecule_index, right_local)]
                result.append(
                    BondRecord(
                        molecule_index=molecule_index,
                        left_local_index=left_local,
                        right_local_index=right_local,
                        left_global_index=left_global,
                        right_global_index=right_global,
                        left_asym_index=left_site.asym_index,
                        right_asym_index=right_site.asym_index,
                        right_image_shift=tuple(int(v) for v in right_shift),
                        vector_A=tuple(float(v) for v in vector),
                        distance_A=float(np.linalg.norm(vector)),
                    )
                )

            # A finite unwrapped embedding cannot represent every edge in a
            # periodic network cycle simultaneously.  Retain the exact
            # neighbor-list edges stored by identify_molecules instead of
            # silently dropping those absent from CrystalMolecule.graph.
            for pair in sorted(
                graph_pairs - legacy_pairs,
                key=lambda item: tuple(sorted(item)),
            ):
                first_global, second_global = sorted(pair)
                first_local = global_to_local[first_global]
                second_local = global_to_local[second_global]
                if first_global <= second_global:
                    left_local, right_local = first_local, second_local
                    left_global, right_global = first_global, second_global
                else:
                    left_local, right_local = second_local, first_local
                    left_global, right_global = second_global, first_global

                left_site = site_records[(molecule_index, left_local)]
                right_site = site_records[(molecule_index, right_local)]
                right_shift = np.asarray(
                    right_site.image_shift, dtype=int
                ) - np.asarray(left_site.image_shift, dtype=int)
                vector = np.asarray(
                    molecule.positions[right_local], dtype=float
                ) - np.asarray(molecule.positions[left_local], dtype=float)

                result.append(
                    BondRecord(
                        molecule_index=molecule_index,
                        left_local_index=left_local,
                        right_local_index=right_local,
                        left_global_index=left_global,
                        right_global_index=right_global,
                        left_asym_index=left_site.asym_index,
                        right_asym_index=right_site.asym_index,
                        right_image_shift=tuple(int(v) for v in right_shift),
                        vector_A=tuple(float(v) for v in vector),
                        distance_A=float(np.linalg.norm(vector)),
                    )
                )

        return sorted(
            result,
            key=lambda record: (
                record.molecule_index,
                record.left_global_index,
                record.right_global_index,
                record.right_image_shift,
            ),
        )

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

        Per-atom metadata arrays (occupancy, disorder_group, assembly, label,
        sym_op_index, asym_id, etc.) are preserved on the returned molecules.
        Stale CIF fractional-coordinate arrays (frac_x/y/z) are stripped
        because the unwrapped Cartesian positions no longer correspond to
        the original CIF fractional coordinates.
        """
        from ..analysis.interactions import get_bonding_threshold
        from .molecule import _strip_stale_frac_arrays
        from ..constants.config import KEY_IMAGE_SHIFT

        unwrapped_molecules = []

        for molecule in self.molecules:
            # 1. Build a lightweight bare Atoms for neighbor_list only.
            #    Using molecule.to_ase() would drop per-atom metadata; instead
            #    we create a minimal object with the crystal's lattice/PBC.
            bare = Atoms(
                symbols=molecule.get_chemical_symbols(),
                positions=molecule.get_positions(),
                cell=self.lattice,
                pbc=self.pbc,
            )

            symbols = bare.get_chemical_symbols()

            # 2. Use neighbor_list('D') to get exact vectors
            # Use a slightly larger cutoff to catch all potential bonds
            i_list, j_list, d_list, D_vectors = neighbor_list(
                "ijdD", bare, cutoff=DEFAULT_NEIGHBOR_CUTOFF
            )

            # 3. Build a temporary graph to traverse
            g = nx.Graph()
            g.add_nodes_from(range(len(bare)))

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
                range(len(bare)),
                bare.get_positions(),
                max_atoms=max_atoms,
            )

            # 5. Create new CrystalMolecule from a COPY of the original
            #    molecule (preserving all per-atom arrays), then overwrite
            #    positions with the unwrapped result.
            new_mol = molecule.copy()
            new_mol.set_positions(positions)
            new_mol.info["unwrap_completed"] = completed
            previous_positions = np.asarray(molecule.get_positions(), dtype=float)
            added_shifts = np.rint(
                (positions - previous_positions)
                @ np.linalg.inv(np.asarray(self.lattice, dtype=float))
            ).astype(int)
            if KEY_IMAGE_SHIFT in new_mol.arrays:
                new_mol.arrays[KEY_IMAGE_SHIFT] = (
                    np.asarray(new_mol.arrays[KEY_IMAGE_SHIFT], dtype=int)
                    + added_shifts
                )
            else:
                new_mol.set_array(KEY_IMAGE_SHIFT, added_shifts)

            # Strip stale CIF fractional coordinates — the unwrapped
            # Cartesian positions are no longer consistent with them.
            _strip_stale_frac_arrays(new_mol)

            unwrapped_molecule = CrystalMolecule(new_mol, self, check_pbc=False)
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
            KEY_ASSEMBLY, KEY_LABEL, KEY_U_CART, KEY_UISO,
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
        # All disorder metadata arrays are propagated to ensure extxyz
        # round-trip preserves the information needed for disorder resolution
        # without re-reading the original CIF file.
        # molecule_index is derived from the current molecule partition.
        # Never propagate a stale index carried by replicated source molecules.
        base_keys = {"numbers", "positions", "molecule_index"}
        string_disorder_keys = {KEY_ASSEMBLY, KEY_LABEL}
        all_custom_keys = sorted(
            {
                key
                for molecule in self.molecules
                for key in molecule.arrays.keys()
            }
            - base_keys
        )

        # Collect ALL per-atom arrays (string and numeric) in one pass.
        def _collect_arrays(key_list):
            """Return {key: values_list} for the given keys."""
            arrays = {k: ([None] * n_total if can_restore_order else [])
                      for k in key_list}
            return arrays

        all_arrays = _collect_arrays(all_custom_keys)

        if can_restore_order:
            symbols = [None] * n_total
            positions = np.zeros((n_total, 3), dtype=float)
            mol_idx = np.empty(n_total, dtype=int)
            for i_mol, (molecule, indices) in enumerate(zip(self.molecules, indices_lists)):
                molecule_symbols = molecule.get_chemical_symbols()
                molecule_positions = molecule.get_positions()
                for local_index, global_index in enumerate(indices):
                    global_index = int(global_index)
                    symbols[global_index] = molecule_symbols[local_index]
                    positions[global_index] = molecule_positions[local_index]
                    mol_idx[global_index] = i_mol
                for k in all_custom_keys:
                    arr = molecule.arrays.get(k)
                    if arr is not None:
                        for local_index, global_index in enumerate(indices):
                            all_arrays[k][int(global_index)] = arr[local_index]
        else:
            symbols = []
            positions = []
            mol_idx = np.empty(n_total, dtype=int)
            offset = 0
            for i_mol, molecule in enumerate(self.molecules):
                symbols.extend(molecule.get_chemical_symbols())
                positions.extend(molecule.get_positions())
                n = len(molecule)
                mol_idx[offset:offset + n] = i_mol
                for k in all_custom_keys:
                    arr = molecule.arrays.get(k)
                    if arr is not None:
                        all_arrays[k].extend(arr)
                    else:
                        all_arrays[k].extend([None] * n)
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

        # --- propagate per-atom arrays ---
        for k in all_custom_keys:
            vals = all_arrays[k]
            if k in {KEY_UISO, KEY_U_CART}:
                sample = next(value for value in vals if value is not None)
                missing = np.full(np.asarray(sample).shape, np.nan, dtype=float)
                vals = [
                    missing.copy() if value is None else np.asarray(value, dtype=float)
                    for value in vals
                ]
            elif not all(v is not None for v in vals):
                continue
            if k in string_disorder_keys:
                # Replace empty strings with "." to prevent ASE extxyz
                # column collapse (whitespace-split format cannot represent
                # empty string tokens).
                sanitised = [v if v else "." for v in vals]
                atoms.set_array(k, np.array(sanitised))
            else:
                arr = np.array(vals)
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

        # Ordinary simple graphs are reconstructed on import. Preserve full
        # provenance only when graph topology cannot express every contact.
        has_non_simple_bonds = False
        for molecule in self.molecules:
            seen_pairs = set()
            for raw in molecule.info.get("bond_records", ()):
                try:
                    left = int(raw["left"])
                    right = int(raw["right"])
                except (KeyError, TypeError, ValueError):
                    continue
                pair = (min(left, right), max(left, right))
                if left == right or pair in seen_pairs:
                    has_non_simple_bonds = True
                    break
                seen_pairs.add(pair)
            if has_non_simple_bonds:
                break
        if has_non_simple_bonds:
            bond_records = self.get_bond_records()
            atoms.info[_BOND_RECORDS_INFO_KEY] = [
                {
                    "molecule_index": record.molecule_index,
                    "left": record.left_global_index,
                    "right": record.right_global_index,
                    "right_image_shift": list(record.right_image_shift),
                    "vector": list(record.vector_A),
                }
                for record in bond_records
            ]

        # --- propagate calculator if attached ---
        if self._calc_results is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            atoms.calc = SinglePointCalculator(atoms, **self._calc_results)

        return atoms

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, bond_scale: float = 1.0) -> "MolecularCrystal":
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
            return cls.from_ase(atoms, bond_scale=bond_scale)

        from ..constants.config import (
            KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY,
        )

        # Desanitise string arrays: "." placeholder → "" (see to_ase)
        asm_arr = atoms.arrays.get(KEY_ASSEMBLY)
        if asm_arr is not None:
            atoms.set_array(
                KEY_ASSEMBLY,
                np.array([("" if v == "." else v) for v in asm_arr]),
            )

        n_mol = int(mol_idx.max()) + 1
        base_keys = {"numbers", "positions"}
        preserved_array_keys = [
            key
            for key in atoms.arrays.keys()
            if key not in base_keys
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
        serialized_bonds = info.pop(_BOND_RECORDS_INFO_KEY, ())
        valid_globals = [
            set(int(index) for index in molecule.info["atom_indices"])
            for molecule in molecules
        ]
        records_by_molecule = [[] for _ in molecules]
        for raw in serialized_bonds:
            try:
                molecule_index = int(raw["molecule_index"])
                left = int(raw["left"])
                right = int(raw["right"])
                shift = [int(value) for value in raw["right_image_shift"]]
                vector = [float(value) for value in raw["vector"]]
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= molecule_index < len(molecules):
                continue
            if not {left, right}.issubset(valid_globals[molecule_index]):
                continue
            records_by_molecule[molecule_index].append(
                {
                    "left": left,
                    "right": right,
                    "right_image_shift": shift,
                    "vector": vector,
                }
            )
        for molecule, records in zip(molecules, records_by_molecule):
            if not records:
                continue
            records.sort(
                key=lambda record: (
                    record["left"],
                    record["right"],
                    tuple(record["right_image_shift"]),
                )
            )
            molecule.info["bond_records"] = records
            molecule.info["bond_pairs"] = sorted(
                {
                    (record["left"], record["right"])
                    for record in records
                    if record["left"] != record["right"]
                }
            )
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
