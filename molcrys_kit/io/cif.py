"""
CIF file parsing for molecular crystals.

This module provides functionality to parse CIF files into MolecularCrystal objects.
It includes tools for handling disorder information and identifying molecular units.
"""

from typing import List, Tuple, Optional, Dict
import itertools
import warnings
import re
import numpy as np
import networkx as nx
from dataclasses import dataclass

from pymatgen.io.cif import CifParser
from pymatgen.core.operations import SymmOp
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.groups import SpaceGroup

from ase import Atoms
from ase.neighborlist import neighbor_list

from ..structures.molecule import CrystalMolecule
from ..structures.crystal import MolecularCrystal
from ..constants import (
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..utils.geometry import minimum_image_distance, unwrap_positions_along_bonds
from ..constants import DEFAULT_NEIGHBOR_CUTOFF


class SymmetryAutoExpandedWarning(UserWarning):
    """Warning emitted when CIF identity-only symops are expanded upstream."""


def _first_cif_value(value):
    """Return the first scalar from pymatgen CIF block values."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _clean_space_group_token(value) -> Optional[str]:
    value = _first_cif_value(value)
    if value in (None, "", ".", "?"):
        return None
    return str(value).strip().strip("'\"") or None


def _parse_space_group_number(data_block: dict) -> Optional[int]:
    for tag in ("_space_group_IT_number", "_symmetry_Int_Tables_number"):
        token = _clean_space_group_token(data_block.get(tag))
        if token is None:
            continue
        try:
            return int(float(token))
        except (TypeError, ValueError):
            continue
    return None


def _space_group_name_variants(name: str) -> List[str]:
    compact = re.sub(r"\s+", "", name)
    screw_normalized = re.sub(r"([2346])([123456])", r"\1_\2", compact)
    return [name, compact, screw_normalized]


def _declared_space_group(data_block: dict) -> Optional[SpaceGroup]:
    sg_number = _parse_space_group_number(data_block)
    if sg_number is not None:
        try:
            return SpaceGroup.from_int_number(sg_number)
        except Exception:
            pass

    for tag in ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M"):
        name = _clean_space_group_token(data_block.get(tag))
        if name is None:
            continue
        for variant in _space_group_name_variants(name):
            try:
                return SpaceGroup(variant)
            except Exception:
                continue
    return None


_PYMATGEN_NUMERIC_MISSING_TAGS = {
    "_atom_site_attached_hydrogens",
}


def _sanitize_cif_text_for_pymatgen(text: str) -> Tuple[str, bool]:
    """Return CIF text with pymatgen-hostile ``?`` numeric sentinels fixed.

    SHELX-derived CIFs sometimes use ``?`` for numeric loop fields such as
    ``_atom_site_attached_hydrogens``. The CIF convention allows this as an
    unknown value, but pymatgen's numeric conversion raises before MolCrysKit
    can apply its own tolerant parsers. We only rewrite known numeric tags and
    leave coordinates / disorder labels untouched.
    """
    lines = text.splitlines(keepends=True)
    out = list(lines)
    changed = False
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        j = i + 1
        tags: list[str] = []
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            tags.append(lines[j].strip().split()[0])
            j += 1
        if not tags:
            i = j
            continue
        numeric_cols = {
            col for col, tag in enumerate(tags)
            if tag in _PYMATGEN_NUMERIC_MISSING_TAGS
        }
        if not numeric_cols:
            i = j
            continue
        k = j
        while k < len(lines):
            stripped = lines[k].strip()
            if not stripped:
                k += 1
                continue
            if stripped.lower() == "loop_" or stripped.startswith("_") or stripped.startswith("data_"):
                break
            tokens = stripped.split()
            if len(tokens) == len(tags):
                row_changed = False
                for col in numeric_cols:
                    if tokens[col] == "?":
                        tokens[col] = "0"
                        changed = True
                        row_changed = True
                if row_changed:
                    newline = "\n" if lines[k].endswith("\n") else ""
                    out[k] = " ".join(tokens) + newline
            k += 1
        i = k
    return "".join(out), changed


def _pymatgen_cif_parser(
    filepath: Optional[str] = None,
    *,
    cif_text: Optional[str] = None,
    **kwargs,
) -> CifParser:
    """Create a CifParser, sanitising known numeric ``?`` fields if needed.

    Either *filepath* (path to a CIF file) or *cif_text* (raw CIF string)
    must be provided.  When *cif_text* is given no file I/O occurs.
    """
    if cif_text is not None:
        text = cif_text
    elif filepath is not None:
        with open(filepath, encoding="utf-8") as handle:
            text = handle.read()
    else:
        raise ValueError("Either 'filepath' or 'cif_text' must be provided.")
    sanitized, changed = _sanitize_cif_text_for_pymatgen(text)
    if changed:
        return CifParser.from_str(sanitized, **kwargs)
    return CifParser.from_str(text, **kwargs)


def _build_molecule_graph(
    atoms: Atoms,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    exclude_indices: Optional[set[int]] = None,
    bond_scale: float = 1.0,
) -> nx.Graph:
    """Build the bonded graph used for molecule identification."""
    from ..constants.config import KEY_DISORDER_GROUP, KEY_SYM_OP_INDEX

    crystal_graph = nx.Graph()
    symbols = atoms.get_chemical_symbols()

    excluded = {int(i) for i in (exclude_indices or set())}
    disorder_groups = atoms.arrays.get(KEY_DISORDER_GROUP)
    sym_op_indices = atoms.arrays.get(KEY_SYM_OP_INDEX)

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

    for i, j, distance, D_vec in zip(i_list, j_list, d_list, D_vectors):
        if i >= j:
            continue
        if int(i) in excluded or int(j) in excluded:
            continue
        if disorder_groups is not None:
            group_i = int(disorder_groups[i])
            group_j = int(disorder_groups[j])
            if group_i != 0 and group_j != 0 and group_i != group_j:
                continue
            if (
                group_i != 0
                and group_j != 0
                and sym_op_indices is not None
                and int(sym_op_indices[i]) != int(sym_op_indices[j])
            ):
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

        if distance < threshold * bond_scale:
            # Store the EXACT vector that connects i to j
            crystal_graph.add_edge(i, j, vector=D_vec)

    return crystal_graph


def _component_atom_indices(
    crystal_graph: nx.Graph,
    exclude_indices: Optional[set[int]] = None,
    include_excluded: bool = True,
) -> List[List[int]]:
    """Return sorted atom-index components from a molecule graph."""
    excluded = {int(i) for i in (exclude_indices or set())}
    components = []
    for component in nx.connected_components(crystal_graph):
        atom_indices = sorted(int(i) for i in component)
        if not include_excluded and excluded:
            atom_indices = [i for i in atom_indices if i not in excluded]
        if atom_indices:
            components.append(atom_indices)
    return components


def identify_molecule_indices(
    atoms: Atoms,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    exclude_indices: Optional[set[int]] = None,
    bond_scale: float = 1.0,
) -> List[List[int]]:
    """
    Identify discrete molecular units and return their original atom indices.

    This is a lightweight companion to :func:`identify_molecules` for workflows
    that need molecule membership without constructing CrystalMolecule objects
    or changing the original ASE Atoms ordering. Bond perception is identical
    to ``identify_molecules``. Atoms in ``exclude_indices`` are removed from
    the returned groups.
    """
    crystal_graph = _build_molecule_graph(
        atoms,
        bond_thresholds=bond_thresholds,
        exclude_indices=exclude_indices,
        bond_scale=bond_scale,
    )
    return _component_atom_indices(
        crystal_graph,
        exclude_indices=exclude_indices,
        include_excluded=False,
    )


def identify_molecules(
    atoms: Atoms,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    max_atoms: Optional[int] = None,
    exclude_indices: Optional[set[int]] = None,
    bond_scale: float = 1.0,
) -> List[CrystalMolecule]:
    """
    Identify discrete molecular units using robust vector-based unwrapping.

    This implementation solves the "Large Beta Angle" problem by strictly using
    the bond vectors identified by ASE's neighbor list logic, rather than
    guessing nearest neighbors via Minimum Image Convention.

    When disorder group metadata is present, bonds between two atoms in
    different non-zero PART groups are skipped. This mirrors the disorder
    graph's bonding rule: ordered atoms (group 0) may bond to either
    orientation, but mutually exclusive disorder images must not fuse into one
    molecule. When symmetry-operation provenance is available, atoms in the
    same non-zero PART group must also come from the same generated image
    before they can bond. ``exclude_indices`` remains available for callers
    that need to remove atoms from bond perception entirely.
    """
    from ..constants.config import KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL, KEY_SYM_OP_INDEX

    crystal_graph = _build_molecule_graph(
        atoms,
        bond_thresholds=bond_thresholds,
        exclude_indices=exclude_indices,
        bond_scale=bond_scale,
    )
    components = _component_atom_indices(
        crystal_graph,
        exclude_indices=exclude_indices,
        include_excluded=True,
    )
    molecules = []

    for atom_indices in components:
        mol_atoms = atoms[atom_indices]
        mol_atoms.info["atom_indices"] = list(atom_indices)

        # Reconstruct molecule topology
        if len(atom_indices) > 1:
            curr_positions, completed = unwrap_positions_along_bonds(
                crystal_graph,
                atom_indices,
                atoms.get_positions(),
                max_atoms=max_atoms,
            )
            mol_atoms.set_positions(curr_positions)
            mol_atoms.info["unwrap_completed"] = completed

        # Preserve disorder-related arrays when creating molecules
        # Copy over disorder metadata for the sliced atoms
        for key in [KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL, KEY_SYM_OP_INDEX]:
            if key in atoms.arrays:
                original_array = atoms.arrays[key]
                sliced_array = original_array[atom_indices]
                mol_atoms.set_array(key, sliced_array)
        
        # Create molecule, explicitly disabling internal PBC checks
        # because we have already unwrapped it perfectly.
        molecule = CrystalMolecule(mol_atoms, check_pbc=False)
        molecule.info["atom_indices"] = list(atom_indices)
        molecule.info["bond_pairs"] = [
            (int(u), int(v))
            for u, v in sorted(crystal_graph.subgraph(atom_indices).edges())
        ]
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
    - asym_id: Index of the parent atom in the asymmetric unit (for tracking
      which expanded copies share the same crystallographic site)
    - site_symmetry_order: Site symmetry order for each atom (from CIF field
      _atom_site_site_symmetry_order). Values > 1 indicate special positions.
    """

    labels: List[str]
    symbols: List[str]
    frac_coords: np.ndarray  # shape (n, 3)
    occupancies: List[float]
    disorder_groups: List[int]
    assemblies: List[str] = None  # New field for assembly information
    sym_op_indices: List[int] = None  # New field for symmetry operation indices
    asym_id: List[int] = None  # Index of parent asymmetric-unit atom
    site_symmetry_order: List[int] = None  # Site symmetry order from CIF
    lattice_matrix: np.ndarray = None  # 3x3 lattice matrix (Angstrom)
    formula_moiety: str = None  # _chemical_formula_moiety from CIF

    def __post_init__(self):
        if self.assemblies is None:
            self.assemblies = []
        if self.sym_op_indices is None:
            self.sym_op_indices = []
        if self.asym_id is None:
            self.asym_id = []
        if self.site_symmetry_order is None:
            self.site_symmetry_order = []
        # lattice_matrix stays None when not available (e.g. legacy callers);
        # from_crystal() and scan_cif_disorder() always set it.

    @property
    def has_disorder(self) -> bool:
        """Return *True* if the structure contains any positional disorder.

        Disorder is detected when at least one site has occupancy < 1.0 or
        belongs to a non-zero disorder group.
        """
        if any(occ < 1.0 for occ in self.occupancies):
            return True
        if any(g != 0 for g in self.disorder_groups):
            return True
        return False

    @classmethod
    def from_crystal(cls, crystal) -> "DisorderInfo":
        """Reconstruct a DisorderInfo from a MolecularCrystal's per-atom arrays.

        This allows disorder resolution to work from extxyz-loaded crystals
        without re-reading the original CIF file.

        When the crystal carries stored CIF fractional coordinates
        (per-atom arrays ``frac_x``, ``frac_y``, ``frac_z`` — set by
        :func:`read_mol_crystal`), those exact values are used.  Otherwise
        fractional coordinates are recomputed from Cartesian positions,
        which may introduce floating-point noise.

        Parameters
        ----------
        crystal : MolecularCrystal
            Crystal loaded via ``read_extxyz`` or ``read_mol_crystal``,
            carrying disorder metadata in its per-atom arrays.

        Returns
        -------
        DisorderInfo
        """
        from ..constants.config import (
            KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL,
            KEY_SYM_OP_INDEX, KEY_ASYM_ID, KEY_SITE_SYMMETRY_ORDER,
            KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z,
        )

        atoms = crystal.to_ase()
        n = len(atoms)

        symbols = atoms.get_chemical_symbols()

        labels_arr = atoms.arrays.get(KEY_LABEL)
        labels = list(labels_arr) if labels_arr is not None else list(symbols)

        # Prefer stored CIF fractional coordinates when available.
        # These are set by read_mol_crystal() from the raw CIF parse and
        # avoid the Cartesian→fractional recomputation that introduces noise.
        fx = atoms.arrays.get(KEY_FRAC_X)
        fy = atoms.arrays.get(KEY_FRAC_Y)
        fz = atoms.arrays.get(KEY_FRAC_Z)
        if fx is not None and fy is not None and fz is not None:
            if not (fx.shape == fy.shape == fz.shape):
                raise ValueError(
                    f"frac_x/y/z shape mismatch: {fx.shape}, {fy.shape}, {fz.shape}"
                )
            frac_coords = np.column_stack([fx, fy, fz])
        else:
            # Fallback: recompute from Cartesian (may have precision loss)
            cell = atoms.get_cell()
            frac_coords = cell.scaled_positions(atoms.get_positions())

        occ_arr = atoms.arrays.get(KEY_OCCUPANCY)
        occupancies = list(occ_arr) if occ_arr is not None else [1.0] * n

        dg_arr = atoms.arrays.get(KEY_DISORDER_GROUP)
        disorder_groups = list(int(x) for x in dg_arr) if dg_arr is not None else [0] * n

        asm_arr = atoms.arrays.get(KEY_ASSEMBLY)
        assemblies = list(asm_arr) if asm_arr is not None else [""] * n

        soi_arr = atoms.arrays.get(KEY_SYM_OP_INDEX)
        sym_op_indices = list(int(x) for x in soi_arr) if soi_arr is not None else list(range(n))

        aid_arr = atoms.arrays.get(KEY_ASYM_ID)
        asym_id = list(int(x) for x in aid_arr) if aid_arr is not None else list(range(n))

        sso_arr = atoms.arrays.get(KEY_SITE_SYMMETRY_ORDER)
        site_symmetry_order = list(int(x) for x in sso_arr) if sso_arr is not None else [1] * n

        lattice_matrix = np.array(crystal.lattice, dtype=float)

        return cls(
            labels=labels,
            symbols=symbols,
            frac_coords=frac_coords,
            occupancies=occupancies,
            disorder_groups=disorder_groups,
            assemblies=assemblies,
            sym_op_indices=sym_op_indices,
            asym_id=asym_id,
            site_symmetry_order=site_symmetry_order,
            lattice_matrix=lattice_matrix,
        )

    def summary(self) -> str:
        """Return a multi-line string with disorder statistics."""
        lines: List[str] = []
        lines.append("Disorder Summary:")
        lines.append(f"  Total atoms: {len(self.labels)}")
        lines.append(f"  Unique elements: {len(set(self.symbols))}")
        lines.append(
            f"  Atoms with occupancy < 1.0: {sum(1 for occ in self.occupancies if occ < 1.0)}"
        )
        lines.append(f"  Unique disorder groups: {len(set(self.disorder_groups))}")
        lines.append(
            f"  Disorder groups range: {min(self.disorder_groups)} to {max(self.disorder_groups)}"
        )
        if self.sym_op_indices:
            lines.append(f"  Unique sym op indices: {len(set(self.sym_op_indices))}")
        if self.asym_id:
            lines.append(f"  Unique asym unit parents: {len(set(self.asym_id))}")
        if self.site_symmetry_order:
            special = sum(1 for s in self.site_symmetry_order if s > 1)
            lines.append(f"  Atoms on special positions (site_sym_order>1): {special}")
        return "\n".join(lines)


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


def _extract_formula_moiety(parser: CifParser) -> Optional[str]:
    """Extract the raw _chemical_formula_moiety field from pymatgen's CIF data."""
    try:
        cif_data = getattr(parser, "_cif")
        blocks = list(cif_data.data.values())
        if not blocks:
            return None

        block = blocks[0]
        data = getattr(block, "data", block)
        value = data.get("_chemical_formula_moiety")
        if value is None:
            return None

        value = str(value).strip()
        return value or None
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


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


def _parse_symmetry_operations(
    data_block: dict, *, expand_symmetry: bool = True
) -> List[SymmOp]:
    """Parse symmetry operations from CIF data block.

    When a CIF declares a non-P1 space group but only provides the identity
    operation, optionally derive the full operation set from the declaration.
    """
    equiv_pos_list = data_block.get("_symmetry_equiv_pos_as_xyz", [])
    symop_list = data_block.get("_space_group_symop_operation_xyz", [])

    sym_ops_raw = []
    if equiv_pos_list:
        sym_ops_raw.extend(equiv_pos_list)
    elif symop_list:
        sym_ops_raw.extend(symop_list)
    else:
        sym_ops_raw = ["x,y,z"]

    sym_ops = []
    for op_str in sym_ops_raw:
        try:
            sym_ops.append(SymmOp.from_xyz_str(op_str))
        except Exception:
            continue

    if not sym_ops:
        sym_ops = [SymmOp.from_xyz_str("x,y,z")]

    declared_sg = _declared_space_group(data_block)
    if (
        expand_symmetry
        and declared_sg is not None
        and declared_sg.int_number > 1
        and len(sym_ops) <= 1
        and len(declared_sg.symmetry_ops) > len(sym_ops)
    ):
        expanded = list(declared_sg.symmetry_ops)
        warnings.warn(
            "CIF declares space group "
            f"#{declared_sg.int_number} ({declared_sg.symbol}) but provides "
            f"only {len(sym_ops)} symmetry operation(s); auto-expanded to "
            f"{len(expanded)} operations.",
            SymmetryAutoExpandedWarning,
            stacklevel=2,
        )
        return expanded

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


def scan_cif_disorder(
    filepath: Optional[str] = None,
    *,
    cif_text: Optional[str] = None,
    expand_symmetry: bool = True,
) -> DisorderInfo:
    """
    Scan a CIF and extract raw disorder-related metadata.

    Either *filepath* (path to a CIF file) or *cif_text* (raw CIF string)
    must be provided.  When *cif_text* is given no file I/O occurs.

    Parameters
    ----------
    filepath : str, optional
        Path to the CIF file.
    cif_text : str, optional
        Raw CIF content as a string (mutually exclusive with *filepath*).

    Returns
    -------
    DisorderInfo
        Object containing raw extracted disorder data for the full unit cell.
    """
    # Parse the CIF using pymatgen to get the raw data dictionary
    parser = _pymatgen_cif_parser(
        filepath, cif_text=cif_text,
        occupancy_tolerance=1, site_tolerance=1e-2,
    )
    cif_data = parser.as_dict()
    formula_moiety = _extract_formula_moiety(parser)

    # We'll use the first data block for simplicity
    first_key = list(cif_data.keys())[0]
    data_block = cif_data[first_key]

    # Parse symmetry operations from the CIF
    sym_ops = _parse_symmetry_operations(
        data_block, expand_symmetry=expand_symmetry
    )

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

    # Extract site symmetry order - default to 1 if missing (general position)
    # The field _atom_site_site_symmetry_order stores how many symmetry operations
    # map the atom back to itself (> 1 means it is on a special position).
    # Older CIF files use _atom_site_symmetry_multiplicity with the same meaning.
    site_sym_orders_raw = data_block.get(
        "_atom_site_site_symmetry_order",
        data_block.get("_atom_site_symmetry_multiplicity", []),
    )
    site_sym_orders = []
    for i in range(n_atoms):
        if i < len(site_sym_orders_raw) and site_sym_orders_raw[i] not in [".", "?", None]:
            try:
                site_sym_orders.append(int(_extract_numeric_value(site_sym_orders_raw[i])))
            except (ValueError, TypeError):
                site_sym_orders.append(1)
        else:
            site_sym_orders.append(1)

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
    all_asym_ids = []  # NEW: index of parent asymmetric-unit atom
    all_site_sym_orders = []  # NEW: site symmetry order for each expanded atom

    # Per-element bucket of fractional coordinates already accepted, used
    # to deduplicate symmetry images.  Each bucket is grown as a Python
    # list of (3,) arrays and converted to a numpy array on demand for
    # vectorised minimum-image distance checks.  This replaces an
    # O(N**2) python double loop that previously ran one PBC distance
    # call per existing atom.
    coords_by_symbol: dict[str, list[np.ndarray]] = {}
    lattice_matrix = lattice.matrix
    tol_sq = 0.01 * 0.01  # match _are_coords_close default (Å)

    # Pre-compute the 27 lattice-shift integer offsets used by the
    # minimum-image convention, plus their Cartesian counterparts.  We
    # reuse them on every comparison to avoid rebuilding the shift table.
    _shifts_frac = np.array(
        list(itertools.product([-1, 0, 1], repeat=3)), dtype=float
    )

    # For each original atom, apply each symmetry operation
    for i in range(len(labels)):
        if not labels[i] or not symbols[i]:  # Skip empty labels or symbols
            continue

        original_coord = frac_coords[i]
        sym_i = symbols[i]
        bucket = coords_by_symbol.setdefault(sym_i, [])

        # Apply each symmetry operation with its index
        for op_idx, op in enumerate(sym_ops):
            # Calculate new coordinate by applying the symmetry operation
            new_coord = op.operate(original_coord)

            # Wrap to unit cell (between 0 and 1)
            new_coord = np.mod(new_coord, 1.0)

            # Vectorised dedup against every existing image of this
            # element.  Equivalent to running `_are_coords_close` against
            # each one but ~100x faster on large unit cells (e.g. PAP-4).
            if bucket:
                existing = np.asarray(bucket)  # shape (N, 3)
                deltas = existing - new_coord  # (N, 3)
                deltas -= np.round(deltas)  # bring into [-0.5, 0.5]
                # 27 candidate vectors per existing atom: (N, 27, 3)
                cand_frac = deltas[:, None, :] + _shifts_frac[None, :, :]
                cand_cart = cand_frac @ lattice_matrix  # (N, 27, 3)
                dists_sq = np.einsum("ijk,ijk->ij", cand_cart, cand_cart)
                if np.min(dists_sq) < tol_sq:
                    continue

            # Add the new atom with its expanded coordinates and metadata
            all_labels.append(labels[i])
            all_symbols.append(sym_i)
            all_frac_coords.append(new_coord)
            all_occupancies.append(occupancies[i])
            all_disorder_groups.append(disorder_groups[i])
            all_assemblies.append(
                assemblies[i]
            )  # Copy the assembly ID to the new atom
            all_sym_op_indices.append(op_idx)  # Store the symmetry operation index
            all_asym_ids.append(i)  # NEW: track parent asymmetric-unit atom
            all_site_sym_orders.append(site_sym_orders[i])  # NEW: site symmetry order
            bucket.append(new_coord)

    # Convert lists to appropriate formats
    all_frac_coords = np.array(all_frac_coords)

    return DisorderInfo(
        labels=all_labels,
        symbols=all_symbols,
        frac_coords=all_frac_coords,
        occupancies=all_occupancies,
        disorder_groups=all_disorder_groups,
        assemblies=all_assemblies,
        sym_op_indices=all_sym_op_indices,
        asym_id=all_asym_ids,
        site_symmetry_order=all_site_sym_orders,
        lattice_matrix=lattice_matrix,
        formula_moiety=formula_moiety,
    )


def read_mol_crystal(
    filepath: Optional[str] = None,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    max_atoms: Optional[int] = None,
    bond_scale: float = 1.0,
    resolve_disorder: bool = False,
    *,
    cif_text: Optional[str] = None,
) -> MolecularCrystal:
    """
    Parse a CIF with advanced molecular grouping.

    Either *filepath* (path to a CIF file) or *cif_text* (raw CIF string)
    must be provided.  When *cif_text* is given no file I/O occurs.

    Parameters
    ----------
    filepath : str, optional
        Path to the CIF file.
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
    max_atoms : int, optional
        Optional maximum molecule size passed to molecule identification.
    bond_scale : float
        Scale factor for bonding thresholds.
    resolve_disorder : bool
        Resolve crystallographic disorder before molecule identification.
    cif_text : str, optional
        Raw CIF content as a string (mutually exclusive with *filepath*).

    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.
    """
    from ..constants.config import (
        KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL,
        KEY_SYM_OP_INDEX, KEY_ASYM_ID, KEY_SITE_SYMMETRY_ORDER,
        KEY_FRAC_X, KEY_FRAC_Y, KEY_FRAC_Z,
    )

    # Extract disorder info — this is now the SOLE authority for atomic
    # positions, elements, and all disorder metadata.  We no longer use
    # pymatgen parse_structures() for the Structure, eliminating the
    # misalignment bug where two independent CIF expansion engines
    # produced different atom counts/ordering.
    disorder_info = scan_cif_disorder(filepath, cif_text=cif_text)

    if disorder_info.has_disorder:
        if resolve_disorder:
            from ..analysis.disorder.process import generate_ordered_replicas_from_disordered_sites
            crystals = generate_ordered_replicas_from_disordered_sites(
                filepath, generate_count=1, method="optimal",
            )
            return crystals[0]
        else:
            n_partial = sum(1 for o in disorder_info.occupancies if o < 1.0)
            warnings.warn(
                f"Structure contains disorder ({n_partial} atoms with occupancy < 1.0). "
                "Molecule identification may include disorder fragments. "
                "Use resolve_disorder=True or 'mck operate disorder' to resolve.",
                stacklevel=2,
            )

    # Build Cartesian coordinates from scan_cif_disorder's fractional coords
    # and lattice matrix — both come from the same CIF parse, so they are
    # guaranteed consistent.
    lattice = disorder_info.lattice_matrix
    symbols = disorder_info.symbols
    frac_coords = disorder_info.frac_coords
    positions = frac_coords @ lattice  # fractional → Cartesian

    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)

    # All disorder metadata comes from the same DisorderInfo — no alignment
    # issue since everything is from a single CIF expansion pass.
    n = len(symbols)
    assert len(disorder_info.occupancies) == n, (
        f"DisorderInfo/symbols length mismatch: "
        f"{len(disorder_info.occupancies)} != {n}"
    )
    atoms.set_array(KEY_OCCUPANCY, np.array(disorder_info.occupancies))
    atoms.set_array(KEY_DISORDER_GROUP, np.array(disorder_info.disorder_groups, dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array(disorder_info.assemblies))
    atoms.set_array(KEY_LABEL, np.array(disorder_info.labels))
    if disorder_info.sym_op_indices:
        atoms.set_array(KEY_SYM_OP_INDEX, np.array(disorder_info.sym_op_indices, dtype=int))
    if disorder_info.asym_id:
        atoms.set_array(KEY_ASYM_ID, np.array(disorder_info.asym_id, dtype=int))
    if disorder_info.site_symmetry_order:
        atoms.set_array(KEY_SITE_SYMMETRY_ORDER, np.array(disorder_info.site_symmetry_order, dtype=int))
    # Store CIF fractional coordinates for exact round-trip via from_crystal()
    atoms.set_array(KEY_FRAC_X, frac_coords[:, 0].copy())
    atoms.set_array(KEY_FRAC_Y, frac_coords[:, 1].copy())
    atoms.set_array(KEY_FRAC_Z, frac_coords[:, 2].copy())

    # formula_moiety is extracted inside scan_cif_disorder from the same
    # pymatgen CIF data dict — no second parse needed.
    formula_moiety = disorder_info.formula_moiety

    # Identify molecular units using graph-based approach
    molecules = identify_molecules(atoms, bond_thresholds=bond_thresholds, max_atoms=max_atoms, bond_scale=bond_scale)

    pbc = (True, True, True)
    return MolecularCrystal(lattice, molecules, pbc, formula_moiety=formula_moiety)


def parse_cif_advanced(
    filepath: str,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    max_atoms: Optional[int] = None,
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
        Parsed crystal structure with identified molecular units.  Delegates to
        `read_mol_crystal`, including any `formula_moiety` metadata read from
        `_chemical_formula_moiety`.

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
    return read_mol_crystal(filepath, bond_thresholds, max_atoms=max_atoms, bond_scale=1.0, resolve_disorder=False)
