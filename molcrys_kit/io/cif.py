"""
CIF file parsing for molecular crystals.

This module provides functionality to parse CIF files into MolecularCrystal objects.
It includes tools for handling disorder information and identifying molecular units.
"""

from typing import List, Tuple, Optional, Dict
import copy
import itertools
import warnings
import re
import logging
import numpy as np
import networkx as nx
from dataclasses import dataclass

from pymatgen.io.cif import CifParser
from pymatgen.core.operations import SymmOp
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.groups import SpaceGroup

from ase import Atoms
from ase.neighborlist import neighbor_list

from ..structures.molecule import CrystalMolecule, _refresh_contiguous_bond_geometry
from ..structures.crystal import MolecularCrystal
from ..structures.symmetry import CrystalSymmetry, FractionalAffineOperation
from ..constants import (
    get_atomic_radius,
    has_atomic_radius,
    is_metal_element,
)
from ..utils.geometry import minimum_image_distance, unwrap_positions_along_bonds


logger = logging.getLogger(__name__)

# A CIF occupancy written to three decimal places can differ from an exact
# reciprocal site order by at most 0.0005 (for example, 1/24 written as 0.042).
_ASU_SHARED_H_OCCUPANCY_TOL = 5e-4


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


def _extract_custom_molcrys_provenance_rows(text: str) -> Dict[int, Tuple[int, int, int]]:
    """Parse optional `_molcrys_*` provenance side-table from raw CIF text.

    New-format CIFs write atom-site data using only standard `_atom_site_*`
    fields and place MolCrysKit provenance in a second loop keyed by atom
    index:

    - `_molcrys_atom_index`
    - `_molcrys_sym_op_index`
    - `_molcrys_asym_id`
    - `_molcrys_site_symmetry_order`

    Returns `{atom_index: (sym_op_index, asym_id, site_sym_order)}`.
    Malformed rows are skipped conservatively.
    """
    lines = text.splitlines()
    i = 0
    rows: Dict[int, Tuple[int, int, int]] = {}
    required = {
        "_molcrys_atom_index",
        "_molcrys_sym_op_index",
        "_molcrys_asym_id",
        "_molcrys_site_symmetry_order",
    }
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        j = i + 1
        tags: list[str] = []
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            tags.append(lines[j].strip().split()[0])
            j += 1
        if not tags or not required.issubset(tags):
            i = j
            continue
        tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
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
                try:
                    atom_index = int(tokens[tag_to_idx["_molcrys_atom_index"]])
                    sym_op_index = int(tokens[tag_to_idx["_molcrys_sym_op_index"]])
                    asym_id = int(tokens[tag_to_idx["_molcrys_asym_id"]])
                    site_sym_order = int(tokens[tag_to_idx["_molcrys_site_symmetry_order"]])
                    rows[atom_index] = (sym_op_index, asym_id, site_sym_order)
                except (TypeError, ValueError, IndexError):
                    pass
            k += 1
        i = k
    return rows


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

    if len(atoms) == 0 or bond_scale <= 0:
        return crystal_graph

    from ..analysis.interactions import get_bonding_threshold

    unique_symbols = sorted(set(symbols))
    radii = {
        symbol: get_atomic_radius(symbol) if has_atomic_radius(symbol) else 0.5
        for symbol in unique_symbols
    }
    metal_flags = {symbol: is_metal_element(symbol) for symbol in unique_symbols}
    threshold_by_pair = {}
    for symbol_i in unique_symbols:
        for symbol_j in unique_symbols:
            pair_key1 = (symbol_i, symbol_j)
            pair_key2 = (symbol_j, symbol_i)
            if bond_thresholds and (
                pair_key1 in bond_thresholds or pair_key2 in bond_thresholds
            ):
                threshold = bond_thresholds.get(
                    pair_key1, bond_thresholds.get(pair_key2)
                )
            else:
                threshold = get_bonding_threshold(
                    radii[symbol_i],
                    radii[symbol_j],
                    metal_flags[symbol_i],
                    metal_flags[symbol_j],
                )
            threshold_by_pair[pair_key1] = threshold

    # ASE accepts exact element-pair cutoffs. Use the larger directed value
    # when callers supplied both pair orientations with different thresholds;
    # the final directed acceptance mask below preserves that legacy behavior.
    candidate_cutoffs = {}
    for index, symbol_i in enumerate(unique_symbols):
        for symbol_j in unique_symbols[index:]:
            candidate_cutoffs[(symbol_i, symbol_j)] = (
                max(
                    threshold_by_pair[(symbol_i, symbol_j)],
                    threshold_by_pair[(symbol_j, symbol_i)],
                )
                * bond_scale
            )

    # -------------------------------------------------------------------------
    # Request both the exact PBC vector and its signed lattice-image shift.
    # For an ASE row-vector cell ``M`` the returned values satisfy:
    # ``D_ij = r_j + S_ij @ M - r_i``.  Keeping ``S_ij`` lets callers
    # materialise an already perceived canonical edge on a particular
    # display image without re-running bond perception.
    # -------------------------------------------------------------------------
    i_list, j_list, d_list, D_vectors, shifts = neighbor_list(
        "ijdDS", atoms, cutoff=candidate_cutoffs
    )

    # Periodic self-image contacts are real topology edges for one-site
    # primitive cells, although the simple membership graph cannot represent
    # them as ordinary pair connectivity.
    compatible = (i_list != j_list) | np.any(shifts != 0, axis=1)
    if excluded:
        excluded_indices = np.fromiter(excluded, dtype=int)
        compatible &= ~np.isin(i_list, excluded_indices)
        compatible &= ~np.isin(j_list, excluded_indices)
    if disorder_groups is not None:
        group_i = disorder_groups[i_list].astype(int, copy=False)
        group_j = disorder_groups[j_list].astype(int, copy=False)
        both_disordered = (group_i != 0) & (group_j != 0)
        incompatible = both_disordered & (group_i != group_j)
        if sym_op_indices is not None:
            incompatible |= both_disordered & (
                sym_op_indices[i_list].astype(int, copy=False)
                != sym_op_indices[j_list].astype(int, copy=False)
            )
        compatible &= ~incompatible

    i_list = i_list[compatible]
    j_list = j_list[compatible]
    d_list = d_list[compatible]
    D_vectors = D_vectors[compatible]
    shifts = shifts[compatible]

    thresholds = np.fromiter(
        (
            threshold_by_pair[(symbols[int(i)], symbols[int(j)])]
            for i, j in zip(i_list, j_list)
        ),
        dtype=float,
        count=len(d_list),
    )
    accepted = d_list < thresholds * bond_scale

    canonical_records = {}
    for i, j, D_vec, shift in zip(
        i_list[accepted],
        j_list[accepted],
        D_vectors[accepted],
        shifts[accepted],
    ):
        # ``neighbor_list`` returns both directed orientations.  Keep one
        # deterministic orientation only after normalising the associated
        # signed image relation to ``left < right`` below.
        left, right = int(i), int(j)
        image_shift = np.asarray(shift, dtype=int)
        vector = np.asarray(D_vec, dtype=float)
        if left > right:
            left, right = right, left
            image_shift = -image_shift
            vector = -vector
        elif left == right and tuple(image_shift) > tuple(-image_shift):
            image_shift = -image_shift
            vector = -vector

        key = (left, right, tuple(int(value) for value in image_shift))
        canonical_records.setdefault(
            key,
            {
                "left": left,
                "right": right,
                "right_image_shift": list(key[2]),
                "vector": [float(value) for value in vector],
            },
        )

        # Keep the simple graph for molecule membership and unwrapping. The
        # renderer-ready record list preserves parallel periodic edges.
        if left != right and not crystal_graph.has_edge(left, right):
            crystal_graph.add_edge(
                left,
                right,
                vector=vector,
                image_shift=image_shift,
            )

    crystal_graph.graph["bond_records"] = [
        canonical_records[key] for key in sorted(canonical_records)
    ]

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
    from ..constants.config import (
        KEY_ASSEMBLY,
        KEY_DISORDER_GROUP,
        KEY_LABEL,
        KEY_OCCUPANCY,
        KEY_SYM_OP_INDEX,
    )

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
    component_by_atom = {
        int(atom_index): component_index
        for component_index, atom_indices in enumerate(components)
        for atom_index in atom_indices
    }
    bond_records_by_component = [[] for _ in components]
    for record in crystal_graph.graph.get("bond_records", ()):
        left_component = component_by_atom.get(int(record["left"]))
        right_component = component_by_atom.get(int(record["right"]))
        if left_component is None or left_component != right_component:
            continue
        bond_records_by_component[left_component].append(dict(record))

    for bond_records in bond_records_by_component:
        bond_records.sort(
            key=lambda record: (
                record["left"],
                record["right"],
                tuple(record["right_image_shift"]),
            )
        )
    molecules = []

    for component_index, atom_indices in enumerate(components):
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
        bond_records = bond_records_by_component[component_index]
        molecule.info["bond_pairs"] = [
            (int(min(u, v)), int(max(u, v)))
            for u, v in sorted(crystal_graph.subgraph(atom_indices).edges())
        ]
        # Additive provenance: retain the legacy pair-only contract while
        # exposing the signed PBC relation that was already computed by ASE.
        molecule.info["bond_records"] = bond_records
        global_to_local = {
            global_index: local_index
            for local_index, global_index in enumerate(atom_indices)
        }
        molecule._graph = nx.relabel_nodes(
            crystal_graph.subgraph(atom_indices).copy(),
            global_to_local,
            copy=True,
        )
        local_positions = molecule.get_positions()
        for local_i, local_j, edge_data in molecule._graph.edges(data=True):
            left, right = sorted((int(local_i), int(local_j)))
            vector = local_positions[right] - local_positions[left]
            edge_data["vector"] = np.asarray(vector, dtype=float).copy()
            edge_data["distance"] = float(np.linalg.norm(vector))
            edge_data["image_shift"] = np.zeros(3, dtype=int)
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
    - uiso: Isotropic/equivalent displacement U in Angstrom squared; NaN when
      absent.
    - u_cart: Cartesian anisotropic displacement tensors, flattened to
      shape ``(n, 9)`` for ASE/ExtXYZ compatibility; all-NaN rows denote
      missing tensors.
    - pbc: Periodic boundary conditions as a 3-tuple of bools, e.g.
      ``(True, True, True)`` for fully periodic or ``(True, True, False)``
      for a slab.  Defaults to ``(True, True, True)``.
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
    uiso: List[float] = None  # Isotropic/equivalent U in Angstrom^2
    u_cart: np.ndarray = None  # Cartesian U tensors, flattened shape (n, 9)
    lattice_matrix: np.ndarray = None  # 3x3 lattice matrix (Angstrom)
    formula_moiety: str = None  # _chemical_formula_moiety from CIF
    z_value: int = None  # _cell_formula_units_Z from CIF
    pbc: Tuple[bool, bool, bool] = None  # Periodic boundary conditions

    def __post_init__(self):
        if self.assemblies is None:
            self.assemblies = []
        if self.sym_op_indices is None:
            self.sym_op_indices = []
        if self.asym_id is None:
            self.asym_id = []
        if self.site_symmetry_order is None:
            self.site_symmetry_order = []
        if self.uiso is None:
            self.uiso = [float("nan")] * len(self.labels)
        if self.u_cart is None:
            self.u_cart = np.full((len(self.labels), 9), np.nan, dtype=float)
        else:
            self.u_cart = np.asarray(self.u_cart, dtype=float).reshape(len(self.labels), 9)
        if self.pbc is None:
            self.pbc = (True, True, True)
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

        Fractional coordinates are always recomputed from Cartesian positions
        and the crystal's current lattice.  Any stored CIF-origin
        ``frac_x``/``frac_y``/``frac_z`` arrays are deliberately ignored
        because they become stale after lattice-transforming operations
        (slab cutting, supercell, etc.).

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
            KEY_UISO, KEY_U_CART,
        )

        atoms = crystal.to_ase()
        n = len(atoms)

        symbols = atoms.get_chemical_symbols()

        labels_arr = atoms.arrays.get(KEY_LABEL)
        labels = list(labels_arr) if labels_arr is not None else list(symbols)

        # Always recompute fractional coordinates from Cartesian positions
        # and the current lattice.  Stored frac_x/y/z arrays are CIF-origin
        # values that become invalid after any lattice transformation (slab
        # cutting, supercell, perturbation, etc.).
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

        uiso_arr = atoms.arrays.get(KEY_UISO)
        uiso = list(float(x) for x in uiso_arr) if uiso_arr is not None else [float("nan")] * n

        u_cart_arr = atoms.arrays.get(KEY_U_CART)
        u_cart = (
            np.asarray(u_cart_arr, dtype=float).reshape(n, 9)
            if u_cart_arr is not None
            else np.full((n, 9), np.nan, dtype=float)
        )

        lattice_matrix = np.array(crystal.lattice, dtype=float)

        pbc = tuple(crystal.pbc) if hasattr(crystal, 'pbc') else (True, True, True)

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
            uiso=uiso,
            u_cart=u_cart,
            lattice_matrix=lattice_matrix,
            pbc=pbc,
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


def _as_cif_list(value) -> list:
    """Normalise a scalar or loop column from pymatgen's CIF dictionary."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _optional_cif_number(value) -> float:
    """Parse one optional CIF number, returning NaN for missing values."""
    if value is None or str(value).strip() in {"", ".", "?"}:
        return float("nan")
    cleaned = re.sub(r"\(.*?\)", "", str(value))
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return float("nan")


def _adp_from_cif_block(
    data_block: dict, labels: List[str], lattice: Lattice
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ASU ``(Uiso, Ucart)`` arrays from CIF U or B fields.

    ``Ucart`` is flattened to shape ``(n, 9)`` so ASE ExtXYZ can preserve it
    as a regular multi-column property.  CIF anisotropic components are
    expressed along the crystallographic axes; the returned tensors use an
    orthonormal Cartesian frame.
    """
    n_atoms = len(labels)
    uiso = np.full(n_atoms, np.nan, dtype=float)
    u_cart = np.full((n_atoms, 9), np.nan, dtype=float)
    tags = {str(tag).casefold(): value for tag, value in data_block.items()}

    raw_uiso = _as_cif_list(tags.get("_atom_site_u_iso_or_equiv"))
    raw_biso = _as_cif_list(tags.get("_atom_site_b_iso_or_equiv"))
    for index in range(n_atoms):
        value = _optional_cif_number(raw_uiso[index]) if index < len(raw_uiso) else np.nan
        if not np.isfinite(value) and index < len(raw_biso):
            b_value = _optional_cif_number(raw_biso[index])
            if np.isfinite(b_value):
                value = b_value / (8.0 * np.pi**2)
        uiso[index] = value

    aniso_labels = _as_cif_list(tags.get("_atom_site_aniso_label"))
    if not aniso_labels:
        return uiso, u_cart

    suffixes = ("11", "22", "33", "12", "13", "23")
    u_columns = [
        _as_cif_list(tags.get(f"_atom_site_aniso_u_{suffix}"))
        for suffix in suffixes
    ]
    b_columns = [
        _as_cif_list(tags.get(f"_atom_site_aniso_b_{suffix}"))
        for suffix in suffixes
    ]
    label_to_index = {str(label): index for index, label in enumerate(labels)}
    cart_basis = np.asarray(lattice.matrix, dtype=float).T
    reciprocal_lengths = np.linalg.norm(np.linalg.inv(cart_basis), axis=1)
    cif_to_cart = cart_basis @ np.diag(reciprocal_lengths)

    for row, raw_label in enumerate(aniso_labels):
        atom_index = label_to_index.get(str(raw_label))
        if atom_index is None:
            continue
        values = []
        for u_column, b_column in zip(u_columns, b_columns):
            value = _optional_cif_number(u_column[row]) if row < len(u_column) else np.nan
            if not np.isfinite(value) and row < len(b_column):
                b_value = _optional_cif_number(b_column[row])
                if np.isfinite(b_value):
                    value = b_value / (8.0 * np.pi**2)
            values.append(value)
        if not np.all(np.isfinite(values)):
            continue
        u_cif = np.array(
            [
                [values[0], values[3], values[4]],
                [values[3], values[1], values[5]],
                [values[4], values[5], values[2]],
            ],
            dtype=float,
        )
        matrix = cif_to_cart @ u_cif @ cif_to_cart.T
        matrix = 0.5 * (matrix + matrix.T)
        u_cart[atom_index] = matrix.reshape(9)
        if not np.isfinite(uiso[atom_index]):
            uiso[atom_index] = float(np.trace(matrix) / 3.0)

    return uiso, u_cart


def _transform_u_cart(u_cart_flat: np.ndarray, op: SymmOp, lattice: Lattice) -> np.ndarray:
    """Apply a fractional symmetry rotation to one Cartesian ADP tensor."""
    values = np.asarray(u_cart_flat, dtype=float)
    if values.size != 9 or not np.all(np.isfinite(values)):
        return np.full(9, np.nan, dtype=float)
    cart_basis = np.asarray(lattice.matrix, dtype=float).T
    rotation_cart = (
        cart_basis
        @ np.asarray(op.rotation_matrix, dtype=float)
        @ np.linalg.inv(cart_basis)
    )
    transformed = rotation_cart @ values.reshape(3, 3) @ rotation_cart.T
    return (0.5 * (transformed + transformed.T)).reshape(9)


def _parse_symmetry_operations(
    data_block: dict, *, expand_symmetry: bool = True
) -> List[SymmOp]:
    """Parse symmetry operations from CIF data block.

    When a CIF declares a non-P1 space group but only provides the identity
    operation, optionally derive the full operation set from the declaration.
    """
    parsed = _crystal_symmetry_from_data_block(
        data_block, expand_symmetry=expand_symmetry, strict=False
    )
    return [
        SymmOp.from_rotation_and_translation(
            operation.rotation, operation.translation
        )
        for operation in parsed.operations
    ]


def _crystal_symmetry_from_data_block(
    data_block: dict,
    *,
    expand_symmetry: bool,
    strict: bool,
) -> CrystalSymmetry:
    """Build :class:`CrystalSymmetry` from one pymatgen CIF data block."""
    equiv_pos_list = data_block.get("_symmetry_equiv_pos_as_xyz", [])
    symop_list = data_block.get("_space_group_symop_operation_xyz", [])
    if equiv_pos_list:
        raw_operations = list(equiv_pos_list)
        source = "_symmetry_equiv_pos_as_xyz"
    elif symop_list:
        raw_operations = list(symop_list)
        source = "_space_group_symop_operation_xyz"
    else:
        raw_operations = ["x,y,z"]
        source = "default_identity"

    parsed_pairs = []
    errors = []
    for index, raw in enumerate(raw_operations):
        token = str(raw).strip().strip("'\"")
        try:
            parsed = SymmOp.from_xyz_str(token)
            validated = FractionalAffineOperation(
                parsed.rotation_matrix,
                parsed.translation_vector,
                xyz=token,
                index=index,
                source=source,
            )
            parsed_pairs.append((token, parsed, validated))
        except Exception as error:
            errors.append((index, token, error))
    if strict and errors:
        index, token, error = errors[0]
        raise ValueError(
            f"invalid symmetry operation at index {index}: {token!r}"
        ) from error
    if not parsed_pairs:
        identity = SymmOp.from_xyz_str("x,y,z")
        parsed_pairs = [
            (
                "x,y,z",
                identity,
                FractionalAffineOperation(
                    identity.rotation_matrix,
                    identity.translation_vector,
                    xyz="x,y,z",
                    index=0,
                    source="fallback_identity",
                ),
            )
        ]
        source = "fallback_identity"

    declared_sg = _declared_space_group(data_block)
    expanded = False
    if (
        expand_symmetry
        and declared_sg is not None
        and declared_sg.int_number > 1
        and len(parsed_pairs) <= 1
        and len(declared_sg.symmetry_ops) > len(parsed_pairs)
    ):
        warnings.warn(
            "CIF declares space group "
            f"#{declared_sg.int_number} ({declared_sg.symbol}) but provides "
            f"only {len(parsed_pairs)} symmetry operation(s); auto-expanded to "
            f"{len(declared_sg.symmetry_ops)} operations.",
            SymmetryAutoExpandedWarning,
            stacklevel=3,
        )
        parsed_pairs = [
            (
                operation.as_xyz_str(),
                operation,
                FractionalAffineOperation(
                    operation.rotation_matrix,
                    operation.translation_vector,
                    xyz=operation.as_xyz_str(),
                    index=index,
                    source="space_group_declaration",
                    space_group_number=int(declared_sg.int_number),
                ),
            )
            for index, operation in enumerate(declared_sg.symmetry_ops)
        ]
        source = "space_group_declaration"
        expanded = True

    space_group_number = (
        int(declared_sg.int_number) if declared_sg is not None else None
    )
    space_group_symbol = (
        str(declared_sg.symbol) if declared_sg is not None else None
    )
    hall_symbol = _clean_space_group_token(data_block.get("_space_group_name_Hall"))
    operations = tuple(
        FractionalAffineOperation(
            validated.rotation,
            validated.translation,
            xyz=xyz,
            index=index,
            source=source,
            space_group_number=space_group_number,
        )
        for index, (xyz, _operation, validated) in enumerate(parsed_pairs)
    )
    return CrystalSymmetry(
        operations=operations,
        space_group_number=space_group_number,
        space_group_symbol=space_group_symbol,
        hall_symbol=hall_symbol,
        source=source,
        expanded_from_declaration=expanded,
    )


def read_cif_symmetry(
    filepath: Optional[str] = None,
    *,
    cif_text: Optional[str] = None,
    expand_symmetry: bool = True,
    strict: bool = False,
) -> CrystalSymmetry:
    """Read crystallographic affine operations without expanding atoms.

    Parameters use the same mutually exclusive file/text convention as
    :func:`scan_cif_disorder`.  With ``strict=True``, malformed explicit
    operation strings raise instead of being skipped.
    """
    parser = _pymatgen_cif_parser(filepath, cif_text=cif_text)
    data = parser.as_dict()
    if not data:
        raise ValueError("CIF contains no data blocks")
    first_block = data[next(iter(data))]
    return _crystal_symmetry_from_data_block(
        first_block, expand_symmetry=expand_symmetry, strict=strict
    )


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
    raw_text = cif_text
    if raw_text is None and filepath is not None:
        with open(filepath, encoding="utf-8") as handle:
            raw_text = handle.read()
    cif_data = parser.as_dict()
    formula_moiety = _extract_formula_moiety(parser)

    # We'll use the first data block for simplicity
    first_key = list(cif_data.keys())[0]
    data_block = cif_data[first_key]

    # Parse Z value (_cell_formula_units_Z)
    z_raw = data_block.get("_cell_formula_units_Z")
    z_value = None
    if z_raw and str(z_raw).strip() not in (".", "?", ""):
        try:
            z_value = int(float(_extract_numeric_value(str(z_raw))))
        except (ValueError, TypeError):
            z_value = None

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

    asu_uiso, asu_u_cart = _adp_from_cif_block(data_block, list(labels), lattice)

    # --- Read MolCrysKit custom CIF fields (for round-trip fidelity) ---
    # When a slab/supercell is written to CIF and later re-read, standard
    # CIF fields can't carry sym_op_index or asym_id (P1 slabs have only
    # the identity operation).  These custom _molcrys_* fields preserve the
    # disorder provenance through the CIF round-trip.

    _raw_molcrys_soi = data_block.get("_molcrys_sym_op_index", [])
    _raw_molcrys_aid = data_block.get("_molcrys_asym_id", [])
    _raw_molcrys_sso = data_block.get("_molcrys_site_symmetry_order", [])
    _side_table = _extract_custom_molcrys_provenance_rows(raw_text or "")
    # Only use the custom fields if their length matches n_atoms.
    # Warn when the raw field has more entries than expected — may
    # indicate a hand-edited or corrupted CIF.
    for label, raw, limit in [
        ("_molcrys_sym_op_index", _raw_molcrys_soi, n_atoms),
        ("_molcrys_asym_id", _raw_molcrys_aid, n_atoms),
        ("_molcrys_site_symmetry_order", _raw_molcrys_sso, n_atoms),
    ]:
        if len(raw) > limit:
            logging.warning(
                "CIF field %s has %d entries, expected %d; extra ignored.",
                label, len(raw), limit,
            )
    _have_custom_soi = len(_raw_molcrys_soi) >= n_atoms or bool(_side_table)
    _have_custom_aid = len(_raw_molcrys_aid) >= n_atoms or bool(_side_table)
    _have_custom_sso = len(_raw_molcrys_sso) >= n_atoms or bool(_side_table)

    molcrys_sym_op_indices = []
    molcrys_asym_ids = []
    molcrys_site_sym_orders = []

    for i in range(n_atoms):
        if i in _side_table:
            soi, aid, sso = _side_table[i]
            molcrys_sym_op_indices.append(soi)
            molcrys_asym_ids.append(aid)
            molcrys_site_sym_orders.append(sso)
            continue
        if _have_custom_soi:
            try:
                molcrys_sym_op_indices.append(int(_extract_numeric_value(_raw_molcrys_soi[i])))
            except (ValueError, TypeError, IndexError):
                molcrys_sym_op_indices.append(0)
        if _have_custom_aid:
            try:
                molcrys_asym_ids.append(int(_extract_numeric_value(_raw_molcrys_aid[i])))
            except (ValueError, TypeError, IndexError):
                molcrys_asym_ids.append(-1)
        if _have_custom_sso:
            try:
                molcrys_site_sym_orders.append(int(_extract_numeric_value(_raw_molcrys_sso[i])))
            except (ValueError, TypeError, IndexError):
                molcrys_site_sym_orders.append(1)

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
    all_uiso = []
    all_u_cart = []

    # Per-element buckets canonicalise symmetry-equivalent ASU rows (some CSD
    # exports contain multiple labelled rows from the same orbit).  Distinct
    # labelled sources that are already coincident in the ASU are the important
    # exception: they represent separate provenance/disorder alternatives and
    # must both survive, including in P1.
    coords_by_symbol: dict[str, list[np.ndarray]] = {}
    sources_by_symbol: dict[str, list[int]] = {}
    lattice_matrix = lattice.matrix
    tol_sq = 0.01 * 0.01  # match _are_coords_close default (Å)

    protected_source_mask = np.zeros((len(labels), len(labels)), dtype=bool)
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if symbols[left] != symbols[right] or labels[left] == labels[right]:
                continue
            if _are_coords_close(frac_coords[left], frac_coords[right], lattice):
                protected_source_mask[left, right] = True
                protected_source_mask[right, left] = True

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
        bucket_sources = sources_by_symbol.setdefault(sym_i, [])

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
                existing = np.asarray(bucket)
                deltas = existing - new_coord  # (N, 3)
                deltas -= np.round(deltas)  # bring into [-0.5, 0.5]
                # 27 candidate vectors per existing atom: (N, 27, 3)
                cand_frac = deltas[:, None, :] + _shifts_frac[None, :, :]
                cand_cart = cand_frac @ lattice_matrix  # (N, 27, 3)
                dists_sq = np.einsum("ijk,ijk->ij", cand_cart, cand_cart)
                coincident = np.min(dists_sq, axis=1) < tol_sq
                existing_sources = np.asarray(bucket_sources, dtype=int)
                protected = protected_source_mask[i, existing_sources]
                if np.any(coincident & ~protected):
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
            all_uiso.append(float(asu_uiso[i]))
            all_u_cart.append(_transform_u_cart(asu_u_cart[i], op, lattice))

            # Use custom _molcrys_* provenance when available (CIF round-trip
            # for slabs/supercells where symmetry expansion is P1 identity).
            # Otherwise fall back to symmetry-expansion-based defaults.
            # (Bounds checks are safe because the lists are built for n_atoms;
            # this is defense-in-depth against edge-case CIF corruption.)
            if _have_custom_soi:
                all_sym_op_indices.append(molcrys_sym_op_indices[i])
            else:
                all_sym_op_indices.append(op_idx)

            if _have_custom_aid:
                all_asym_ids.append(molcrys_asym_ids[i])
            else:
                all_asym_ids.append(i)

            if _have_custom_sso:
                all_site_sym_orders.append(molcrys_site_sym_orders[i])
            else:
                all_site_sym_orders.append(site_sym_orders[i])

            bucket.append(new_coord)
            bucket_sources.append(i)

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
        uiso=all_uiso,
        u_cart=np.asarray(all_u_cart, dtype=float).reshape(-1, 9),
        lattice_matrix=lattice_matrix,
        formula_moiety=formula_moiety,
        z_value=z_value,
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
        KEY_UISO, KEY_U_CART,
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
    atoms.set_array(KEY_UISO, np.asarray(disorder_info.uiso, dtype=float))
    atoms.set_array(KEY_U_CART, np.asarray(disorder_info.u_cart, dtype=float).reshape(n, 9))
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


def _parse_cif_asu(
    filepath: Optional[str] = None,
    *,
    cif_text: Optional[str] = None,
) -> Tuple[Atoms, List[SymmOp], Lattice]:
    """
    Parse a CIF file and return only the atoms in the asymmetric unit plus symmetry operations.

    Unlike scan_cif_disorder, this function does NOT expand symmetry operations,
    returning only the atoms explicitly specified in the CIF file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CIF file.
    cif_text : str, optional
        Raw CIF content as a string (mutually exclusive with *filepath*).

    Returns
    -------
    Tuple[Atoms, List[SymmOp], Lattice]
        - ase.Atoms object containing only atoms in the asymmetric unit
        - List of symmetry operations
        - pymatgen Lattice object
    """
    from ..constants.config import (
        KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL,
        KEY_ASYM_ID, KEY_SITE_SYMMETRY_ORDER,
        KEY_UISO, KEY_U_CART,
    )

    # Parse the CIF using pymatgen to get the raw data dictionary
    parser = _pymatgen_cif_parser(
        filepath, cif_text=cif_text,
        occupancy_tolerance=1, site_tolerance=1e-2,
    )
    cif_data = parser.as_dict()

    # We'll use the first data block for simplicity
    first_key = list(cif_data.keys())[0]
    data_block = cif_data[first_key]

    # Parse symmetry operations — use expand_symmetry=True so that the
    # symmetry-source policy is identical to scan_cif_disorder().
    sym_ops = _parse_symmetry_operations(data_block, expand_symmetry=True)

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
            occupancies.append(1.0)

    # Extract disorder groups - default to 0 if missing or invalid
    disorder_groups = []
    raw_groups = data_block.get("_atom_site_disorder_group", [])

    for i in range(n_atoms):
        if i < len(raw_groups) and raw_groups[i] not in [".", "?", None]:
            try:
                disorder_groups.append(int(_extract_numeric_value(raw_groups[i])))
            except (ValueError, TypeError):
                disorder_groups.append(0)
        else:
            disorder_groups.append(0)

    # Extract assembly information - default to empty string if missing
    assemblies = []
    raw_assemblies = data_block.get("_atom_site_disorder_assembly", [])

    for i in range(n_atoms):
        if i < len(raw_assemblies):
            assemblies.append(str(raw_assemblies[i]) if raw_assemblies[i] not in [".", "?", None] else "")
        else:
            assemblies.append("")

    # Extract site symmetry order — same logic as scan_cif_disorder()
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

    asu_uiso, asu_u_cart = _adp_from_cif_block(data_block, list(labels), lattice)

    # Build Cartesian coordinates from fractional coords
    positions = frac_coords @ lattice.matrix  # fractional → Cartesian

    # Build ASE Atoms object (only the asymmetric unit)
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice.matrix, pbc=True)

    # All disorder metadata comes from the same CIF parse
    n = len(symbols)
    atoms.set_array(KEY_OCCUPANCY, np.array(occupancies))
    atoms.set_array(KEY_DISORDER_GROUP, np.array(disorder_groups, dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array(assemblies))
    atoms.set_array(KEY_LABEL, np.array(labels))
    atoms.set_array(KEY_ASYM_ID, np.arange(n, dtype=int))
    atoms.set_array(KEY_SITE_SYMMETRY_ORDER, np.array(site_sym_orders, dtype=int))
    atoms.set_array(KEY_UISO, asu_uiso)
    atoms.set_array(KEY_U_CART, asu_u_cart)

    return atoms, sym_ops, lattice


def _identify_molecules_asu_first(
    filepath: Optional[str] = None,
    *,
    cif_text: Optional[str] = None,
    bond_thresholds: Optional[Dict[Tuple[str, str], float]] = None,
    max_atoms: Optional[int] = None,
    bond_scale: float = 1.0,
    special_position_tol: float = 0.5,
) -> MolecularCrystal:
    """
    ASU-first molecule identification: identify molecules on the asymmetric unit,
    then generate all molecular instances using symmetry operations.

    This is more efficient than the standard path (expand first, then identify),
    especially for high-symmetry crystals, as it avoids redundant computation on
    symmetry-equivalent atoms.

    Parameters
    ----------
    filepath : str, optional
        Path to the CIF file.
    cif_text : str, optional
        Raw CIF content as a string (mutually exclusive with *filepath*).
    bond_thresholds : dict, optional
        Custom dictionary with atom pairs as keys and bonding thresholds as values.
    max_atoms : int, optional
        Optional maximum molecule size passed to molecule identification.
    bond_scale : float
        Scale factor for bonding thresholds.
    special_position_tol : float
        Tolerance for deduplicating molecules at special positions (in Angstroms).

    Returns
    -------
    MolecularCrystal
        Parsed crystal structure with identified molecular units.
    """
    from ..constants.config import (
        KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_SYM_OP_INDEX, KEY_ASYM_ID,
        KEY_SITE_SYMMETRY_ORDER,
        KEY_U_CART,
    )

    # Parse CIF to get only the asymmetric unit atoms
    asu_atoms, sym_ops, lattice = _parse_cif_asu(filepath, cif_text=cif_text)

    # Identify molecules on the asymmetric unit
    asu_molecules = identify_molecules(
        asu_atoms,
        bond_thresholds=bond_thresholds,
        max_atoms=max_atoms,
        bond_scale=bond_scale,
    )

    def _complete_shared_h_center(mol: CrystalMolecule) -> Optional[int]:
        """Return the centre of a complete special-position H orientation."""
        symbols = mol.get_chemical_symbols()
        occupancies = mol.arrays.get(KEY_OCCUPANCY)
        disorder_groups = mol.arrays.get(KEY_DISORDER_GROUP)
        asym_ids = mol.arrays.get(KEY_ASYM_ID)
        site_orders = mol.arrays.get(KEY_SITE_SYMMETRY_ORDER)
        if any(
            values is None
            for values in (occupancies, disorder_groups, asym_ids, site_orders)
        ):
            return None

        centres = [
            i
            for i, symbol in enumerate(symbols)
            if symbol != "H"
            and int(disorder_groups[i]) == 0
            and int(site_orders[i]) > 1
        ]
        if len(centres) != 1:
            return None

        centre = centres[0]
        hydrogens = [i for i in range(len(mol)) if i != centre]
        hydrogen_groups = {int(disorder_groups[i]) for i in hydrogens}
        hydrogen_asym_ids = {int(asym_ids[i]) for i in hydrogens}
        # A hydrogen with its own site stabilizer contributes
        # occupancy * centre_order / hydrogen_order to each centre image.
        expected_occupancies = (
            site_orders[hydrogens].astype(float) / int(site_orders[centre])
        )
        if (
            not hydrogens
            or any(symbols[i] != "H" for i in hydrogens)
            or not np.isclose(
                occupancies[centre],
                1.0,
                rtol=0.0,
                atol=_ASU_SHARED_H_OCCUPANCY_TOL,
            )
            or len(hydrogen_groups) != 1
            or next(iter(hydrogen_groups)) == 0
            or len(hydrogen_asym_ids) != len(hydrogens)
            or not np.allclose(
                occupancies[hydrogens],
                expected_occupancies,
                rtol=0.0,
                atol=_ASU_SHARED_H_OCCUPANCY_TOL,
            )
        ):
            return None
        return centre

    # These ASU components already contain one complete hydrogen orientation.
    # Their symmetry images are alternatives around the same physical centre,
    # unlike true fragments such as ClO that require post-expansion merging.
    complete_h_centres = {
        asu_idx: centre
        for asu_idx, mol in enumerate(asu_molecules)
        if (centre := _complete_shared_h_center(mol)) is not None
    }

    # --- Giant-molecule fallback ---
    # Only fall back when the ASU is a *single connected component* AND the
    # space group has multiplicity > 1 AND we can compare the expected atom
    # count from |G| * |ASU| against the unit-cell volume heuristic.
    # A single-molecule ASU is perfectly valid for e.g. P1, molecular
    # crystals with Z'=1, or network solids (NaCl, Diamond, SiO2).
    if len(asu_molecules) == 1 and len(asu_molecules[0]) == len(asu_atoms):
        n_ops = len(sym_ops)
        expected_atoms = n_ops * len(asu_atoms)
        vol = abs(np.linalg.det(lattice.matrix))
        # Heuristic: if the expected total exceeds ~50 atoms per Å³ of cell
        # volume, the ASU is likely a network solid that was erroneously
        # merged or cross-disorder bonding happened.  For legitimate
        # structures (NaCl: 8 atoms in 179 ų) this ratio is always tiny.
        atoms_per_vol = expected_atoms / max(vol, 1.0)
        if atoms_per_vol > 0.5:
            raise ValueError(
                f"ASU-first: {expected_atoms} atoms in {vol:.0f} ų "
                f"({atoms_per_vol:.2f} atoms/ų) suggests network/cross-disorder; "
                "falling back to standard path"
            )

    # --- Replicate each ASU molecule through all symmetry operations ---
    all_molecules: list = []
    lattice_matrix = lattice.matrix
    inv_lattice = np.linalg.inv(lattice_matrix)

    # Build a set of wrapped fractional "fingerprints" for fast O(1)
    # duplicate lookup.  Key = (asu_mol_idx, rounded anchor frac coord).
    # After the cheap hash check, an expensive atom-wise PBC comparison
    # confirms duplicates so that hash collisions are harmless.
    _seen_keys: set = set()

    def _frac_fingerprint(frac_coords: np.ndarray, anchor_index: int) -> tuple:
        """Round the wrapped anchor atom to a grid for hashing."""
        anchor = np.mod(frac_coords[anchor_index], 1.0)
        return tuple(np.round(anchor, decimals=3))

    def _is_duplicate(
        new_frac: np.ndarray, asu_mol_idx: int, existing_mols: list
    ) -> bool:
        """Check if the molecule is a special-position duplicate.

        Uses a two-stage test:
        1. Hash on (asu_mol_idx, anchor grid) for O(1) reject.
        2. Atom-wise minimum-image comparison within the hash bucket.
        """
        anchor_index = complete_h_centres.get(asu_mol_idx, 0)
        key = (asu_mol_idx, _frac_fingerprint(new_frac, anchor_index))
        if key not in _seen_keys:
            _seen_keys.add(key)
            return False

        # Expensive confirmation: compare all atoms PBC-wise
        for existing_mol in existing_mols:
            if existing_mol.info.get("asu_molecule_index") != asu_mol_idx:
                continue
            ex_frac = existing_mol.positions @ inv_lattice
            if len(ex_frac) != len(new_frac):
                continue
            # Complete H shells are alternative orientations around the same
            # special-position centre, so compare their centre only.  Other
            # ASU molecules still require the original atom-wise comparison.
            compare = (
                [anchor_index]
                if asu_mol_idx in complete_h_centres
                else slice(None)
            )
            deltas = ex_frac[compare] - new_frac[compare]
            deltas -= np.round(deltas)
            cart_deltas = deltas @ lattice_matrix
            dists = np.linalg.norm(cart_deltas, axis=1)
            if np.max(dists) < special_position_tol:
                return True
        return False

    for asu_mol_idx, asu_mol in enumerate(asu_molecules):
        asu_frac_coords = asu_mol.positions @ inv_lattice

        for op_idx, op in enumerate(sym_ops):
            # Apply symmetry operation per atom (SymmOp.operate expects a
            # single 3-vector; the vectorized form uses the rotation matrix
            # and translation directly).
            rot = op.rotation_matrix
            tau = op.translation_vector
            new_frac_coords = (rot @ asu_frac_coords.T).T + tau

            # Place one anchor in the primary cell without independently
            # wrapping atoms and breaking the molecule's contiguous geometry.
            new_frac_coords -= np.floor(new_frac_coords[0])

            # Duplicate detection (special positions)
            if _is_duplicate(new_frac_coords, asu_mol_idx, all_molecules):
                continue

            # Convert back to Cartesian coordinates
            new_positions = new_frac_coords @ lattice_matrix

            # Create the replicated molecule
            new_mol = CrystalMolecule(Atoms(
                symbols=asu_mol.get_chemical_symbols(),
                positions=new_positions,
                cell=lattice_matrix,
                pbc=True,
            ), check_pbc=False)

            # Preserve generic metadata arrays (skip symop/asym which we set below)
            for key in asu_mol.arrays.keys():
                if key in ("numbers", "positions", KEY_SYM_OP_INDEX, KEY_ASYM_ID):
                    continue
                arr = asu_mol.arrays[key]
                if arr is not None and len(arr) == len(asu_mol):
                    values = arr.copy()
                    if key == KEY_U_CART:
                        values = np.vstack(
                            [_transform_u_cart(value, op, lattice) for value in values]
                        )
                    new_mol.set_array(key, values)

            # Set symmetry operation index (same for all atoms in this replica)
            new_mol.set_array(KEY_SYM_OP_INDEX, np.full(len(asu_mol), op_idx, dtype=int))
            # Preserve the original ASU atom IDs
            if KEY_ASYM_ID in asu_mol.arrays:
                new_mol.set_array(KEY_ASYM_ID, asu_mol.arrays[KEY_ASYM_ID].copy())
            else:
                new_mol.set_array(KEY_ASYM_ID, np.arange(len(asu_mol), dtype=int))
            new_mol.info["sym_op_index"] = op_idx
            new_mol.info["asu_molecule_index"] = asu_mol_idx

            # Preserve the authoritative ASU topology. Recompute only the
            # geometry-dependent edge attributes after the symmetry transform.
            new_mol._graph = copy.deepcopy(asu_mol.get_graph())
            _refresh_contiguous_bond_geometry(new_mol)
            new_mol.info["bond_pairs"] = [
                (int(min(atom_i, atom_j)), int(max(atom_i, atom_j)))
                for atom_i, atom_j in sorted(new_mol._graph.edges())
            ]

            all_molecules.append(new_mol)

    # --- Post-replication merge for special-position fragments ---
    # When an ASU molecule sits on a special position (e.g. Cl on a 4-fold
    # axis of ClO4), the ASU only contains a *fragment* of the physical
    # molecule.  After replication, fragments from different symops that
    # overlap spatially need to be merged.
    #
    # Strategy: collect all replicated atoms into one Atoms object, run
    # identify_molecules to re-detect bonds, then rebuild.  This is only
    # needed when there are ASU molecules with atoms on special positions
    # (site_symmetry_order > 1), otherwise skip the expensive re-merge.
    has_special = False
    for mol in all_molecules:
        sso = mol.arrays.get(KEY_SITE_SYMMETRY_ORDER)
        if sso is not None and np.any(sso > 1):
            has_special = True
            break

    if has_special:
        from ase import Atoms as _Atoms

        # Identify which ASU molecules have atoms on special positions
        special_asu_indices = set()
        normal_mols = []
        special_mols = []
        for mol in all_molecules:
            sso = mol.arrays.get(KEY_SITE_SYMMETRY_ORDER)
            asu_idx = mol.info.get("asu_molecule_index", -1)
            if asu_idx in complete_h_centres:
                normal_mols.append(mol)
            elif sso is not None and np.any(sso > 1):
                special_asu_indices.add(asu_idx)
                special_mols.append(mol)
            else:
                # Check if this molecule's ASU type was already flagged
                if asu_idx in special_asu_indices:
                    special_mols.append(mol)
                else:
                    normal_mols.append(mol)

        # Re-check: some normal_mols might belong to special ASU types
        # (the flag is set lazily as we encounter them)
        final_normal = []
        for mol in normal_mols:
            asu_idx = mol.info.get("asu_molecule_index", -1)
            if asu_idx in special_asu_indices:
                special_mols.append(mol)
            else:
                final_normal.append(mol)

        if special_mols:
            # Collect all atoms from special-position molecules
            sp_symbols = []
            sp_positions = []
            sp_asym_ids = []
            for mol in special_mols:
                sp_symbols.extend(mol.get_chemical_symbols())
                sp_positions.extend(mol.positions.tolist())
                aid = mol.arrays.get(KEY_ASYM_ID)
                if aid is not None:
                    sp_asym_ids.extend(aid.tolist())
                else:
                    sp_asym_ids.extend([-1] * len(mol))

            # Deduplicate only same-asym_id same-element overlapping atoms
            inv_lat = np.linalg.inv(lattice_matrix)
            frac_sp = np.array(sp_positions) @ inv_lat
            n_sp = len(sp_symbols)
            keep_mask = np.ones(n_sp, dtype=bool)

            # Group by (element, asym_id) for efficient dedup
            from collections import defaultdict
            groups: dict = defaultdict(list)
            for idx in range(n_sp):
                key = (sp_symbols[idx], sp_asym_ids[idx])
                if key[1] >= 0:
                    groups[key].append(idx)

            for key, indices in groups.items():
                if len(indices) <= 1:
                    continue
                for ii in range(len(indices)):
                    i = indices[ii]
                    if not keep_mask[i]:
                        continue
                    for jj in range(ii + 1, len(indices)):
                        j = indices[jj]
                        if not keep_mask[j]:
                            continue
                        delta = frac_sp[i] - frac_sp[j]
                        delta -= np.round(delta)
                        cart_d = delta @ lattice_matrix
                        if np.linalg.norm(cart_d) < special_position_tol:
                            keep_mask[j] = False

            keep_idx = np.where(keep_mask)[0]
            dedup_atoms = _Atoms(
                symbols=[sp_symbols[i] for i in keep_idx],
                positions=[sp_positions[i] for i in keep_idx],
                cell=lattice_matrix,
                pbc=True,
            )

            merged_special = identify_molecules(
                dedup_atoms,
                bond_thresholds=bond_thresholds,
                max_atoms=max_atoms,
                bond_scale=bond_scale,
            )
            all_molecules = final_normal + merged_special
        else:
            all_molecules = final_normal

    # Build the final MolecularCrystal
    pbc = (True, True, True)
    return MolecularCrystal(lattice_matrix, all_molecules, pbc)
