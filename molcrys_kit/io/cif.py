"""
CIF file parsing for molecular crystals.

This module provides functionality to parse CIF files into MolecularCrystal objects.
It includes tools for handling disorder information and identifying molecular units.
"""

from typing import List, Tuple, Optional, Dict
import warnings
import re
import numpy as np
import networkx as nx
from dataclasses import dataclass

from pymatgen.io.cif import CifParser
from pymatgen.core.operations import SymmOp
from pymatgen.core.lattice import Lattice

from ase import Atoms
from ase.neighborlist import neighbor_list

from ..structures.molecule import CrystalMolecule
from ..structures.crystal import MolecularCrystal
from ..constants import (
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..utils.geometry import minimum_image_distance
from ..constants import DEFAULT_NEIGHBOR_CUTOFF


def identify_molecules(
    atoms: Atoms, bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None
) -> List[CrystalMolecule]:
    """
    Identify discrete molecular units using robust vector-based unwrapping.

    This implementation solves the "Large Beta Angle" problem by strictly using
    the bond vectors identified by ASE's neighbor list logic, rather than
    guessing nearest neighbors via Minimum Image Convention.
    """
    from ..constants.config import KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL
    
    crystal_graph = nx.Graph()
    symbols = atoms.get_chemical_symbols()

    for i in range(len(atoms)):
        crystal_graph.add_node(i, symbol=symbols[i])

    # -------------------------------------------------------------------------
    # KEY FIX: Request 'D' vector from neighbor_list
    # 'D' is the vector pointing from i to j (taking into account PBC).
    # D_ij = r_j - r_i + shift_vector
    # -------------------------------------------------------------------------
    i_list, j_list, d_list, D_vectors = neighbor_list(
        "ijdD", atoms, cutoff=DEFAULT_NEIGHBOR_CUTOFF
    )

    from ..analysis.interactions import get_bonding_threshold

    for idx, (i, j, distance, D_vec) in enumerate(
        zip(i_list, j_list, d_list, D_vectors)
    ):
        if i >= j:
            continue

        pair_key1, pair_key2 = (symbols[i], symbols[j]), (symbols[j], symbols[i])

        if bond_thresholds and (
            pair_key1 in bond_thresholds or pair_key2 in bond_thresholds
        ):
            threshold = bond_thresholds.get(pair_key1, bond_thresholds.get(pair_key2))
        else:
            radius_i = (
                get_atomic_radius(symbols[i]) if has_atomic_radius(symbols[i]) else 0.5
            )
            radius_j = (
                get_atomic_radius(symbols[j]) if has_atomic_radius(symbols[j]) else 0.5
            )
            is_metal_i, is_metal_j = (
                is_metal_element(symbols[i]),
                is_metal_element(symbols[j]),
            )
            threshold = get_bonding_threshold(
                radius_i, radius_j, is_metal_i, is_metal_j
            )

        if distance < threshold:
            # Store the EXACT vector that connects i to j
            crystal_graph.add_edge(i, j, vector=D_vec)

    components = list(nx.connected_components(crystal_graph))
    molecules = []

    for component in components:
        atom_indices = list(component)
        mol_atoms = atoms[atom_indices]

        # Local (molecule) index -> Global (crystal) index map
        local_to_global = {i: idx for i, idx in enumerate(atom_indices)}

        # Reconstruct molecule topology
        if len(atom_indices) > 1:
            curr_positions = mol_atoms.get_positions()
            visited = {0}
            queue = [0]

            while queue:
                u_local = queue.pop(0)
                u_global = local_to_global[u_local]

                # Check neighbors in global graph
                for v_global in crystal_graph.neighbors(u_global):
                    if v_global in atom_indices:
                        v_local = atom_indices.index(v_global)

                        if v_local not in visited:
                            # Retrieve the stored bond vector
                            edge_data = crystal_graph.get_edge_data(u_global, v_global)
                            d_vec = edge_data["vector"]

                            # Determine direction:
                            # neighbor_list returns D_ij for the pair it found.
                            # We stored D_vec. If graph is undirected, we need to know if
                            # D_vec corresponds to u->v or v->u.
                            #
                            # However, neighbor_list guarantees i < j usually? No.
                            # But in our loop `if i >= j: continue`, we only added edges where i < j.
                            # So the stored `vector` is definitely from smaller_index -> larger_index.

                            idx1 = min(u_global, v_global)
                            idx2 = max(u_global, v_global)

                            # The stored vector is from idx1 -> idx2
                            vector_1_to_2 = d_vec

                            if u_global == idx1:
                                # We are moving u -> v (small -> large)
                                shift = vector_1_to_2
                            else:
                                # We are moving u -> v (large -> small)
                                shift = -vector_1_to_2

                            # Apply exact shift. No guessing, no MIC.
                            curr_positions[v_local] = curr_positions[u_local] + shift

                            visited.add(v_local)
                            queue.append(v_local)

            mol_atoms.set_positions(curr_positions)

        # Preserve disorder-related arrays when creating molecules
        # Copy over disorder metadata for the sliced atoms
        for key in [KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL]:
            if key in atoms.arrays:
                original_array = atoms.arrays[key]
                sliced_array = original_array[atom_indices]
                mol_atoms.set_array(key, sliced_array)
        
        # Create molecule, explicitly disabling internal PBC checks
        # because we have already unwrapped it perfectly.
        molecule = CrystalMolecule(mol_atoms, check_pbc=False)
        molecules.append(molecule)

    return molecules


@dataclass
class DisorderInfo:
    """
    Data class to store raw extracted disorder data from CIF files.

    Fields:
    - labels: Original atom labels (e.g., "C1A", "H2'")
    - symbols: Element symbols
    - frac_coords: nx3 array of fractional coordinates
    - occupancies: Site occupancy (default to 1.0 if missing)
    - disorder_groups: Integer tags (default to 0 if missing or '.' in CIF)
    - assemblies: Assembly ID for each atom (default to empty string if missing)
    - sym_op_indices: Index of the generating symmetry operation for each atom
    """

    labels: List[str]
    symbols: List[str]
    frac_coords: np.ndarray  # shape (n, 3)
    occupancies: List[float]
    disorder_groups: List[int]
    assemblies: List[str] = None  # New field for assembly information
    sym_op_indices: List[int] = None  # New field for symmetry operation indices

    def __post_init__(self):
        if self.assemblies is None:
            self.assemblies = []
        if self.sym_op_indices is None:
            self.sym_op_indices = []

    def summary(self):
        """Print statistics about the disorder information."""
        print("Disorder Summary:")
        print(f"  Total atoms: {len(self.labels)}")
        print(f"  Unique elements: {len(set(self.symbols))}")
        print(
            f"  Atoms with occupancy < 1.0: {sum(1 for occ in self.occupancies if occ < 1.0)}"
        )
        print(f"  Unique disorder groups: {len(set(self.disorder_groups))}")
        print(
            f"  Disorder groups range: {min(self.disorder_groups)} to {max(self.disorder_groups)}"
        )
        if self.sym_op_indices:
            print(f"  Unique sym op indices: {len(set(self.sym_op_indices))}")


def _clean_species_string(species_string: str) -> str:
    """
    Clean up species strings from CIF files.

    This function handles common issues with species strings in CIF files,
    such as charge indicators and isotopes.

    Parameters
    ----------
    species_string : str
        Raw species string from CIF.

    Returns
    -------
    str
        Cleaned species string with only the element symbol.
    """
    # Pre-compile regular expressions for better performance
    _CLEAN_PATTERN = re.compile(r":.*")
    _ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")

    cleaned = _CLEAN_PATTERN.split(species_string, 1)[0]
    # Extract only the alphabetic part as the element symbol
    element_match = _ELEMENT_PATTERN.search(cleaned)
    return element_match.group(0) if element_match else cleaned


def _extract_numeric_value(value_str: str) -> float:
    """
    Extract numeric value from CIF strings like '12.345(6)', '0.5', or '.'.
    Returns 0.0 for invalid/missing values.
    """
    if not isinstance(value_str, str):
        return float(value_str)
    if value_str.strip() in [".", "?", ""]:
        return 0.0
    # Remove parentheses and content inside: '1.23(4)' -> '1.23'
    cleaned = re.sub(r"\(.*?\)", "", value_str)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_symmetry_operations(data_block: dict) -> List[SymmOp]:
    """
    Parse symmetry operations from CIF data block.

    Parameters
    ----------
    data_block : dict
        CIF data block containing the raw CIF data.

    Returns
    -------
    List[SymmOp]
        List of symmetry operations as SymmOp objects.
    """
    # Look for symmetry operations in different possible CIF tags
    equiv_pos_list = data_block.get("_symmetry_equiv_pos_as_xyz", [])
    symop_list = data_block.get("_space_group_symop_operation_xyz", [])

    # Combine the two lists if both exist
    sym_ops_raw = []
    if equiv_pos_list:
        sym_ops_raw.extend(equiv_pos_list)
    elif symop_list:
        sym_ops_raw.extend(symop_list)
    else:
        # If no symmetry operations are found, assume P1 (identity only)
        sym_ops_raw = ["x,y,z"]

    # Convert the raw symmetry operations to SymmOp objects
    sym_ops = []
    for op_str in sym_ops_raw:
        try:
            sym_ops.append(SymmOp.from_xyz_str(op_str))
        except Exception:
            # If we can't parse the operation, skip it
            continue

    # If no operations were successfully parsed, return identity only
    if not sym_ops:
        sym_ops = [SymmOp.from_xyz_str("x,y,z")]

    return sym_ops


def _are_coords_close(
    coord1: np.ndarray, coord2: np.ndarray, lattice: Lattice, tol: float = 0.01
) -> bool:
    """
    Check if two fractional coordinates represent the same position in the unit cell.

    Parameters
    ----------
    coord1, coord2 : np.ndarray
        Fractional coordinates to compare.
    lattice : Lattice
        The crystal lattice for distance calculation.
    tol : float
        Tolerance for considering coordinates as the same (in Angstroms).

    Returns
    -------
    bool
        True if the coordinates are close enough to be considered the same.
    """
    distance = minimum_image_distance(coord1, coord2, lattice.matrix)
    return distance < tol


def scan_cif_disorder(filepath: str) -> DisorderInfo:
    """
    Scan a CIF file and extract raw disorder-related metadata without any logical processing.

    This function extracts the raw data exactly as it appears in the CIF file, preserving
    label suffixes and treating missing/invalid values as defaults.

    This implementation now expands the asymmetric unit to the full unit cell
    by applying symmetry operations, while preserving all disorder metadata.

    Parameters
    ----------
    filepath : str
        Path to the CIF file.

    Returns
    -------
    DisorderInfo
        Object containing raw extracted disorder data for the full unit cell.
    """
    # Parse the CIF file using pymatgen to get the raw data dictionary
    parser = CifParser(filepath, occupancy_tolerance=1, site_tolerance=1e-2)
    cif_data = parser.as_dict()

    # We'll use the first data block for simplicity
    first_key = list(cif_data.keys())[0]
    data_block = cif_data[first_key]

    # Parse symmetry operations from the CIF
    sym_ops = _parse_symmetry_operations(data_block)

    # Parse lattice for distance calculations
    try:
        # Extract lattice parameters using the robust numeric parser
        a = _extract_numeric_value(data_block.get("_cell_length_a", "10.0"))
        b = _extract_numeric_value(data_block.get("_cell_length_b", "10.0"))
        c = _extract_numeric_value(data_block.get("_cell_length_c", "10.0"))
        alpha = _extract_numeric_value(data_block.get("_cell_angle_alpha", "90.0"))
        beta = _extract_numeric_value(data_block.get("_cell_angle_beta", "90.0"))
        gamma = _extract_numeric_value(data_block.get("_cell_angle_gamma", "90.0"))

        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    except (ValueError, TypeError):
        # If lattice parameters are not available, create a default lattice
        lattice = Lattice.cubic(10.0)

    # Extract raw data fields
    labels = data_block.get("_atom_site_label", [])
    symbols = data_block.get("_atom_site_type_symbol", [])

    # If type symbols are missing, try to extract them from labels
    if not symbols or all(s == "" for s in symbols):
        symbols = []
        for label in labels:
            # Extract element symbol from label (e.g., "C1A" -> "C")
            element_match = re.match(r"([A-Za-z]+)", label)
            if element_match:
                symbols.append(element_match.group(1))
            else:
                symbols.append("")

    # Extract fractional coordinates
    frac_x = data_block.get("_atom_site_fract_x", [])
    frac_y = data_block.get("_atom_site_fract_y", [])
    frac_z = data_block.get("_atom_site_fract_z", [])

    # Convert fractional coordinates to numpy array
    n_atoms = len(labels)
    frac_coords = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        try:
            frac_coords[i, 0] = (
                _extract_numeric_value(frac_x[i]) if i < len(frac_x) else 0.0
            )
            frac_coords[i, 1] = (
                _extract_numeric_value(frac_y[i]) if i < len(frac_y) else 0.0
            )
            frac_coords[i, 2] = (
                _extract_numeric_value(frac_z[i]) if i < len(frac_z) else 0.0
            )
        except (ValueError, TypeError, IndexError):
            # If conversion fails, keep as 0.0 but log a warning
            warnings.warn(
                f"Failed to parse coordinates for atom {i}, defaulting to (0,0,0)"
            )
            continue

    # Extract occupancies - default to 1.0 if missing or invalid
    occupancies = []
    raw_occupancies = data_block.get("_atom_site_occupancy", [])

    for i in range(n_atoms):
        if i < len(raw_occupancies) and raw_occupancies[i] not in [".", "?", None]:
            occupancies.append(_extract_numeric_value(raw_occupancies[i]))
        else:
            occupancies.append(1.0)  # Default to 1.0 if missing

    # Extract disorder groups - default to 0 if missing or invalid
    disorder_groups = []
    raw_groups = data_block.get("_atom_site_disorder_group", [])

    for i in range(n_atoms):
        if i < len(raw_groups) and raw_groups[i] not in [".", "?", None]:
            try:
                disorder_groups.append(
                    int(_extract_numeric_value(raw_groups[i]))
                )  # Convert to int after extracting numeric value
            except (ValueError, TypeError):
                disorder_groups.append(0)  # Default to 0 if conversion fails
        else:
            disorder_groups.append(0)  # Default to 0 if missing

    # Extract assembly information - default to empty string if missing or invalid
    assemblies = []
    raw_assemblies = data_block.get("_atom_site_disorder_assembly", [])

    for i in range(n_atoms):
        if i < len(raw_assemblies) and raw_assemblies[i] not in [".", "?", None]:
            try:
                assembly_value = str(raw_assemblies[i]).strip()
                # Normalize: Treat ".", "?", or None as empty string ""
                if assembly_value in [".", "?", ""]:
                    assemblies.append("")
                else:
                    assemblies.append(assembly_value)
            except (ValueError, TypeError):
                assemblies.append("")  # Default to empty string if conversion fails
        else:
            assemblies.append("")  # Default to empty string if missing

    # Ensure all arrays have the same length by padding if necessary
    min_len = n_atoms
    labels = (
        labels[:min_len]
        if len(labels) >= min_len
        else labels + [""] * (min_len - len(labels))
    )
    symbols = (
        symbols[:min_len]
        if len(symbols) >= min_len
        else symbols + [""] * (min_len - len(symbols))
    )
    assemblies = (
        assemblies[:min_len]
        if len(assemblies) >= min_len
        else assemblies + [""] * (min_len - len(assemblies))
    )

    # Expand the asymmetric unit to the full unit cell using symmetry operations
    all_labels = []
    all_symbols = []
    all_frac_coords = []
    all_occupancies = []
    all_disorder_groups = []
    all_assemblies = []  # New list for assemblies
    all_sym_op_indices = []  # New list for symmetry operation indices

    # For each original atom, apply each symmetry operation
    for i in range(len(labels)):
        if not labels[i] or not symbols[i]:  # Skip empty labels or symbols
            continue

        original_coord = frac_coords[i]

        # Apply each symmetry operation with its index
        for op_idx, op in enumerate(sym_ops):
            # Calculate new coordinate by applying the symmetry operation
            new_coord = op.operate(original_coord)

            # Wrap to unit cell (between 0 and 1)
            new_coord = np.mod(new_coord, 1.0)

            # Check if this coordinate is already present for this specific atom type
            is_duplicate = False
            for j, (existing_symbol, existing_coord) in enumerate(
                zip(all_symbols, all_frac_coords)
            ):
                if existing_symbol == symbols[i]:
                    if _are_coords_close(existing_coord, new_coord, lattice):
                        is_duplicate = True
                        break

            if not is_duplicate:
                # Add the new atom with its expanded coordinates and metadata
                all_labels.append(labels[i])
                all_symbols.append(symbols[i])
                all_frac_coords.append(new_coord)
                all_occupancies.append(occupancies[i])
                all_disorder_groups.append(disorder_groups[i])
                all_assemblies.append(
                    assemblies[i]
                )  # Copy the assembly ID to the new atom
                all_sym_op_indices.append(op_idx)  # Store the symmetry operation index

    # Convert lists to appropriate formats
    all_frac_coords = np.array(all_frac_coords)

    return DisorderInfo(
        labels=all_labels,
        symbols=all_symbols,
        frac_coords=all_frac_coords,
        occupancies=all_occupancies,
        disorder_groups=all_disorder_groups,
        assemblies=all_assemblies,  # Include assemblies in the return
        sym_op_indices=all_sym_op_indices,  # Include symmetry operation indices in the return
    )


def read_mol_crystal(
    filepath: str, bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None
) -> MolecularCrystal:
    """
    Parse a CIF file with advanced molecular grouping.

    This function attempts to identify discrete molecular units within the crystal.

    Parameters
    ----------
    filepath : str
        Path to the CIF file.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
        Keys should be tuples of element symbols (e.g., ('H', 'O')), and values should
        be the distance thresholds for bonding in Angstroms.

    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.
    """
    from ..constants.config import KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL
    
    # First, extract disorder info from CIF file
    disorder_info = scan_cif_disorder(filepath)
    
    # Parse the CIF file using pymatgen with special options for handling disordered structures
    # Using occupancy_tolerance to handle disordered structures with '?' or other problematic values
    # Also using more tolerant parameters to handle CIF files with full coordinates
    try:
        parser = CifParser(filepath, occupancy_tolerance=10, site_tolerance=1e-2)
        # Use parse_structures instead of get_structures to avoid deprecation warning
        try:
            structures = parser.parse_structures()
        except AttributeError:
            # Fallback for older pymatgen versions
            structures = parser.get_structures()
    except Exception:
        print("Warning: CIF parsing failed. Trying with more relaxed parameters...")
        parser = CifParser(
            filepath, occupancy_tolerance=100, site_tolerance=1e-1, frac_tolerance=1e-1
        )
        try:
            structures = parser.parse_structures()
        except AttributeError:
            # Fallback for older pymatgen versions
            structures = parser.get_structures()

    # For simplicity, we take the first structure
    structure = structures[0]

    # Extract lattice vectors
    lattice = structure.lattice.matrix

    # Create ASE Atoms object with cleaned symbols
    symbols = [_clean_species_string(site.species_string) for site in structure.sites]
    positions = structure.cart_coords
    
    # Use the disorder_info occupancies which contain the original values from CIF
    # since pymatgen might have expanded them due to symmetry operations
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)
    
    # Set disorder metadata arrays to the atoms object using the raw disorder information
    # Make sure the number of atoms matches the disorder info
    atoms.set_array(KEY_OCCUPANCY, np.array(disorder_info.occupancies[:len(symbols)]))
    atoms.set_array(KEY_DISORDER_GROUP, np.array(disorder_info.disorder_groups[:len(symbols)], dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array(disorder_info.assemblies[:len(symbols)]))
    atoms.set_array(KEY_LABEL, np.array(disorder_info.labels[:len(symbols)]))
    
    # Identify molecular units using graph-based approach
    molecules = identify_molecules(atoms, bond_thresholds=bond_thresholds)

    # Assuming periodic boundary conditions in all directions
    pbc = (True, True, True)

    return MolecularCrystal(lattice, molecules, pbc)


def parse_cif_advanced(
    filepath: str, bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None
) -> MolecularCrystal:
    """
    Parse a CIF file with advanced molecular grouping.

    This function attempts to identify discrete molecular units within the crystal.

    Parameters
    ----------
    filepath : str
        Path to the CIF file.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
        Keys should be tuples of element symbols (e.g., ('H', 'O')), and values should
        be the distance thresholds for bonding in Angstroms.

    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.

    Raises
    ------
    DeprecationWarning
        This function is deprecated and will be removed in a future version.
        Use read_mol_crystal() instead.
    """
    warnings.warn(
        "parse_cif_advanced is deprecated and will be removed in a future version. "
        "Use read_mol_crystal instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_mol_crystal(filepath, bond_thresholds)
