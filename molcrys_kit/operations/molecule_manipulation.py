"""
Targeted molecule manipulation operations for molecular crystals.

This module provides functionality to select specific molecules inside a
:class:`~molcrys_kit.structures.crystal.MolecularCrystal` and perform
**translation**, **rotation**, and **replacement** operations on them.

All public operations return a *new* ``MolecularCrystal``; the original
crystal is never mutated.
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule
from ..utils.geometry import get_rotation_matrix, min_distance_between_atom_sets


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MoleculeClashError(Exception):
    """Raised when molecule replacement results in atomic clashes that
    cannot be resolved by random rotation attempts.

    Attributes
    ----------
    min_distance : float
        The closest atom–atom distance achieved.
    threshold : float
        The clash threshold that was requested.
    attempts : int
        Number of rotation attempts that were made.
    """

    def __init__(self, min_distance: float, threshold: float, attempts: int):
        self.min_distance = min_distance
        self.threshold = threshold
        self.attempts = attempts
        super().__init__(
            f"Could not resolve clash after {attempts} rotation attempts. "
            f"Minimum distance: {min_distance:.3f} Å (threshold: {threshold:.3f} Å)"
        )


# ---------------------------------------------------------------------------
# Helper: build a new crystal with one molecule swapped
# ---------------------------------------------------------------------------

def _build_crystal_with_replaced_molecule(
    crystal: MolecularCrystal,
    molecule_index: int,
    new_molecule: CrystalMolecule,
) -> MolecularCrystal:
    """Return a new ``MolecularCrystal`` identical to *crystal* except that
    the molecule at *molecule_index* has been replaced by *new_molecule*.
    """
    new_molecules = []
    for i, mol in enumerate(crystal.molecules):
        if i == molecule_index:
            new_molecules.append(new_molecule)
        else:
            new_molecules.append(mol.copy())
    return MolecularCrystal(
        lattice=crystal.lattice.copy(),
        molecules=new_molecules,
        pbc=crystal.pbc,
    )


# ===================================================================
# MoleculeManipulator class
# ===================================================================

class MoleculeManipulator:
    """Provides targeted operations on specific molecules within a
    :class:`~molcrys_kit.structures.crystal.MolecularCrystal`.

    All operations return a **new** ``MolecularCrystal``; the original is
    never modified.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to operate on.

    Examples
    --------
    >>> from molcrys_kit import read_mol_crystal
    >>> from molcrys_kit.operations.molecule_manipulation import MoleculeManipulator
    >>> crystal = read_mol_crystal("examples/MAF-4.cif")
    >>> manip = MoleculeManipulator(crystal)
    >>> new_crystal = manip.translate_molecule(0, vector=[1.0, 0.0, 0.0])
    """

    def __init__(self, crystal: MolecularCrystal):
        self.crystal = crystal

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def select_molecules(
        self,
        indices: Union[int, List[int], None] = None,
        species_id: Optional[str] = None,
    ) -> List[int]:
        """Resolve a selection criterion to a list of molecule indices.

        Exactly one of *indices* or *species_id* must be provided.

        Parameters
        ----------
        indices : int or list of int, optional
            Direct molecule index or list of indices (0-based).
        species_id : str, optional
            A species identifier such as ``"H2O_1"`` as returned by
            :class:`~molcrys_kit.analysis.stoichiometry.StoichiometryAnalyzer`.

        Returns
        -------
        list of int
            Sorted list of molecule indices matching the selection.

        Raises
        ------
        ValueError
            If both or neither selection arguments are given, or if an
            index is out of range.
        """
        if (indices is None) == (species_id is None):
            raise ValueError(
                "Exactly one of 'indices' or 'species_id' must be provided."
            )

        n_mol = len(self.crystal.molecules)

        if indices is not None:
            if isinstance(indices, int):
                indices = [indices]
            for idx in indices:
                if idx < 0 or idx >= n_mol:
                    raise ValueError(
                        f"Molecule index {idx} is out of range "
                        f"(crystal has {n_mol} molecules, valid range 0..{n_mol - 1})."
                    )
            return sorted(set(indices))

        # species_id path
        from ..analysis.stoichiometry import StoichiometryAnalyzer

        analyzer = StoichiometryAnalyzer(self.crystal)
        if species_id not in analyzer.species_map:
            available = list(analyzer.species_map.keys())
            raise ValueError(
                f"Species '{species_id}' not found. "
                f"Available species: {available}"
            )
        return sorted(analyzer.species_map[species_id])

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate_molecule(
        self,
        molecule_index: int,
        vector: np.ndarray,
        fractional: bool = False,
    ) -> MolecularCrystal:
        """Translate a specific molecule by a displacement vector.

        Parameters
        ----------
        molecule_index : int
            Index of the molecule to translate (0-based).
        vector : array_like, shape (3,)
            Displacement vector.  Interpreted as Cartesian (Å) by default,
            or as fractional coordinates if *fractional* is ``True``.
        fractional : bool, default False
            If ``True``, *vector* is in fractional coordinates and will be
            converted to Cartesian before application.

        Returns
        -------
        MolecularCrystal
            New crystal with the translated molecule.
        """
        self._validate_index(molecule_index)
        vector = np.asarray(vector, dtype=float)

        if fractional:
            vector = self.crystal.fractional_to_cartesian(vector)

        # Copy the target molecule and shift its positions
        mol_copy = self.crystal.molecules[molecule_index].copy()
        mol_copy.set_positions(mol_copy.get_positions() + vector)

        return _build_crystal_with_replaced_molecule(
            self.crystal, molecule_index, mol_copy
        )

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate_molecule(
        self,
        molecule_index: int,
        axis: np.ndarray,
        angle: float,
        center: str = "com",
    ) -> MolecularCrystal:
        """Rotate a specific molecule around its centre.

        Parameters
        ----------
        molecule_index : int
            Index of the molecule to rotate (0-based).
        axis : array_like, shape (3,)
            Rotation axis (does not need to be normalised).
        angle : float
            Rotation angle in **degrees**.
        center : str, default ``"com"``
            Pivot point: ``"com"`` for centre of mass, ``"centroid"`` for
            geometric centroid.

        Returns
        -------
        MolecularCrystal
            New crystal with the rotated molecule.
        """
        self._validate_index(molecule_index)
        axis = np.asarray(axis, dtype=float)
        if np.linalg.norm(axis) == 0:
            raise ValueError("Rotation axis must be a non-zero vector.")

        mol_copy = self.crystal.molecules[molecule_index].copy()

        if center == "com":
            pivot = mol_copy.get_center_of_mass()
        elif center == "centroid":
            pivot = mol_copy.get_centroid()
        else:
            raise ValueError(f"Unknown center mode '{center}'. Use 'com' or 'centroid'.")

        angle_rad = np.radians(angle)
        rot_mat = get_rotation_matrix(axis, angle_rad)

        positions = mol_copy.get_positions()
        translated = positions - pivot
        rotated = translated @ rot_mat.T
        mol_copy.set_positions(rotated + pivot)

        return _build_crystal_with_replaced_molecule(
            self.crystal, molecule_index, mol_copy
        )

    # ------------------------------------------------------------------
    # Replacement
    # ------------------------------------------------------------------

    def replace_molecule(
        self,
        molecule_index: int,
        new_molecule: Union[str, CrystalMolecule],
        clash_threshold: float = 1.0,
        max_rotation_attempts: int = 100,
        align_method: str = "com",
    ) -> MolecularCrystal:
        """Replace a molecule with a new one, optionally resolving clashes.

        The new molecule is centred on the old molecule's centre of mass (or
        centroid).  If any atom of the replacement molecule is closer than
        *clash_threshold* Å to any host-framework atom, random rotations are
        attempted to resolve the clash.

        Parameters
        ----------
        molecule_index : int
            Index of the molecule to replace (0-based).
        new_molecule : str or CrystalMolecule
            Either a file path to an XYZ file, or an already-loaded
            :class:`~molcrys_kit.structures.molecule.CrystalMolecule`.
        clash_threshold : float, default 1.0
            Minimum acceptable atom–atom distance (Å) between the
            replacement molecule and the host framework.
        max_rotation_attempts : int, default 100
            Maximum number of random rotations to try when resolving a
            clash.  Set to 0 to skip clash resolution entirely.
        align_method : str, default ``"com"``
            How to align the replacement molecule to the original position:
            ``"com"`` aligns centres of mass, ``"centroid"`` aligns
            geometric centroids.

        Returns
        -------
        MolecularCrystal
            New crystal with the replacement molecule.

        Raises
        ------
        MoleculeClashError
            If the clash cannot be resolved within *max_rotation_attempts*.
        """
        self._validate_index(molecule_index)

        # --- Load if string path ---
        if isinstance(new_molecule, str):
            from ..io.xyz import read_xyz
            new_molecule = read_xyz(new_molecule)

        # --- Compute alignment target ---
        old_mol = self.crystal.molecules[molecule_index]
        if align_method == "com":
            target_point = old_mol.get_center_of_mass()
            new_point = new_molecule.get_center_of_mass()
        elif align_method == "centroid":
            target_point = old_mol.get_centroid()
            new_point = new_molecule.get_centroid()
        else:
            raise ValueError(
                f"Unknown align_method '{align_method}'. Use 'com' or 'centroid'."
            )

        # --- Translate replacement to target position ---
        replacement = new_molecule.copy()
        shift = target_point - new_point
        replacement.set_positions(replacement.get_positions() + shift)

        # --- Collect host positions (all molecules except the target) ---
        host_positions = self._get_host_positions(molecule_index)

        # --- Clash detection & resolution ---
        if host_positions.shape[0] > 0 and clash_threshold > 0:
            replacement, resolved = self._resolve_clash_by_rotation(
                replacement,
                host_positions,
                clash_threshold,
                max_rotation_attempts,
            )
            if not resolved:
                current_min = min_distance_between_atom_sets(
                    replacement.get_positions(), host_positions
                )
                raise MoleculeClashError(
                    min_distance=current_min,
                    threshold=clash_threshold,
                    attempts=max_rotation_attempts,
                )

        return _build_crystal_with_replaced_molecule(
            self.crystal, molecule_index, replacement
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_index(self, molecule_index: int) -> None:
        n_mol = len(self.crystal.molecules)
        if molecule_index < 0 or molecule_index >= n_mol:
            raise ValueError(
                f"Molecule index {molecule_index} out of range "
                f"(crystal has {n_mol} molecules, valid range 0..{n_mol - 1})."
            )

    def _get_host_positions(self, exclude_index: int) -> np.ndarray:
        """Concatenate Cartesian positions of all molecules except the one
        at *exclude_index*.

        Returns
        -------
        np.ndarray, shape (N_host, 3)
            Host-framework atom positions.
        """
        all_positions = []
        for i, mol in enumerate(self.crystal.molecules):
            if i != exclude_index:
                all_positions.append(mol.get_positions())

        if not all_positions:
            return np.empty((0, 3))
        return np.vstack(all_positions)

    @staticmethod
    def _check_clash(
        guest_positions: np.ndarray,
        host_positions: np.ndarray,
        threshold: float,
    ) -> bool:
        """Return ``True`` if any guest–host atom pair is closer than
        *threshold* Å.
        """
        if guest_positions.shape[0] == 0 or host_positions.shape[0] == 0:
            return False
        min_dist = min_distance_between_atom_sets(guest_positions, host_positions)
        return min_dist < threshold

    @staticmethod
    def _resolve_clash_by_rotation(
        molecule: CrystalMolecule,
        host_positions: np.ndarray,
        threshold: float,
        max_attempts: int,
    ) -> Tuple[CrystalMolecule, bool]:
        """Try random rotations around the molecule's COM to resolve clashes.

        Parameters
        ----------
        molecule : CrystalMolecule
            The molecule to rotate (already positioned at the target site).
        host_positions : np.ndarray
            Cartesian positions of all host-framework atoms.
        threshold : float
            Minimum acceptable distance in Å.
        max_attempts : int
            Maximum number of rotation attempts.

        Returns
        -------
        molecule : CrystalMolecule
            The (possibly rotated) molecule.
        success : bool
            ``True`` if a clash-free orientation was found.
        """
        guest_positions = molecule.get_positions()

        # Check if there's already no clash
        if not MoleculeManipulator._check_clash(
            guest_positions, host_positions, threshold
        ):
            return molecule, True

        com = molecule.get_center_of_mass()

        best_molecule = molecule
        best_min_dist = min_distance_between_atom_sets(guest_positions, host_positions)

        for _ in range(max_attempts):
            # Random rotation axis (uniform on sphere)
            axis = np.random.randn(3)
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis /= norm

            # Random angle in [0, 2π)
            angle_rad = np.random.uniform(0, 2 * np.pi)
            rot_mat = get_rotation_matrix(axis, angle_rad)

            # Rotate around COM
            translated = guest_positions - com
            rotated = translated @ rot_mat.T
            new_positions = rotated + com

            current_min = min_distance_between_atom_sets(new_positions, host_positions)

            if current_min >= threshold:
                # Clash resolved — update molecule and return
                result = molecule.copy()
                result.set_positions(new_positions)
                return result, True

            # Track best attempt
            if current_min > best_min_dist:
                best_min_dist = current_min
                best_molecule = molecule.copy()
                best_molecule.set_positions(new_positions)

        # All attempts failed; return the best orientation found
        return best_molecule, False


# ===================================================================
# Module-level convenience functions
# ===================================================================


def translate_molecule(
    crystal: MolecularCrystal,
    molecule_index: int,
    vector: np.ndarray,
    fractional: bool = False,
) -> MolecularCrystal:
    """Translate a specific molecule in a crystal by a displacement vector.

    This is a convenience wrapper around
    :meth:`MoleculeManipulator.translate_molecule`.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal containing the molecule.
    molecule_index : int
        Index of the molecule to translate (0-based).
    vector : array_like, shape (3,)
        Displacement vector (Cartesian Å, or fractional if *fractional*
        is ``True``).
    fractional : bool, default False
        Interpret *vector* as fractional coordinates.

    Returns
    -------
    MolecularCrystal
        New crystal with the translated molecule.
    """
    return MoleculeManipulator(crystal).translate_molecule(
        molecule_index, vector, fractional=fractional
    )


def rotate_molecule(
    crystal: MolecularCrystal,
    molecule_index: int,
    axis: np.ndarray,
    angle: float,
    center: str = "com",
) -> MolecularCrystal:
    """Rotate a specific molecule in a crystal around its centre.

    This is a convenience wrapper around
    :meth:`MoleculeManipulator.rotate_molecule`.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal containing the molecule.
    molecule_index : int
        Index of the molecule to rotate (0-based).
    axis : array_like, shape (3,)
        Rotation axis.
    angle : float
        Rotation angle in **degrees**.
    center : str, default ``"com"``
        Pivot point: ``"com"`` or ``"centroid"``.

    Returns
    -------
    MolecularCrystal
        New crystal with the rotated molecule.
    """
    return MoleculeManipulator(crystal).rotate_molecule(
        molecule_index, axis, angle, center=center
    )


def replace_molecule(
    crystal: MolecularCrystal,
    molecule_index: int,
    new_molecule: Union[str, CrystalMolecule],
    clash_threshold: float = 1.0,
    max_rotation_attempts: int = 100,
    align_method: str = "com",
) -> MolecularCrystal:
    """Replace a molecule in a crystal with a new one, with clash detection.

    The new molecule's centre of mass is aligned to the old molecule's
    centre of mass.  If the replacement is too close to any host-framework
    atom (distance < *clash_threshold*), random rotations are attempted.

    This is a convenience wrapper around
    :meth:`MoleculeManipulator.replace_molecule`.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal containing the molecule to replace.
    molecule_index : int
        Index of the molecule to replace (0-based).
    new_molecule : str or CrystalMolecule
        Path to an XYZ file or an already-loaded ``CrystalMolecule``.
    clash_threshold : float, default 1.0
        Minimum acceptable distance (Å) to host-framework atoms.
    max_rotation_attempts : int, default 100
        Maximum random-rotation attempts for clash resolution.
    align_method : str, default ``"com"``
        ``"com"`` or ``"centroid"``.

    Returns
    -------
    MolecularCrystal
        New crystal with the replacement molecule.

    Raises
    ------
    MoleculeClashError
        If clashes cannot be resolved.
    """
    return MoleculeManipulator(crystal).replace_molecule(
        molecule_index,
        new_molecule,
        clash_threshold=clash_threshold,
        max_rotation_attempts=max_rotation_attempts,
        align_method=align_method,
    )
