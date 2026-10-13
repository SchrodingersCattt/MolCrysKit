"""
Output functionality for molecular crystal structures.

This module provides functions for writing molecular crystal data to various formats.
"""

import base64
import hashlib
import json
import os
import re
import warnings
from fractions import Fraction
from io import StringIO
from typing import Optional, Sequence

import ase.io
import numpy as np
from ase.constraints import FixScaled
from ase.io.vasp import write_vasp

from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule


def write_xyz(molecule: CrystalMolecule, filename: str = None) -> str:
    """
    Write a molecule to XYZ format.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to write.
    filename : str, optional
        The filename to write to. If None, returns the XYZ string.

    Returns
    -------
    str
        XYZ format string if filename is None, otherwise None.
    """
    # Get atomic symbols and positions
    symbols = molecule.get_chemical_symbols()
    positions = molecule.get_positions()

    # Create XYZ format string
    xyz_lines = [str(len(symbols)), ""]  # Number of atoms and empty comment line
    for symbol, pos in zip(symbols, positions):
        xyz_lines.append(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

    xyz_string = "\n".join(xyz_lines) + "\n"

    if filename is None:
        return xyz_string
    else:
        with open(filename, "w") as f:
            f.write(xyz_string)


def write_xyz_with_freeze(
    cluster,
    filename: str,
    comment: Optional[str] = None,
    sidecar_path: Optional[str] = None,
) -> str:
    """Write a carved :class:`CrystalCluster` as XYZ plus a JSON sidecar.

    Two artefacts are emitted:

    * ``<filename>`` -- standard XYZ.  The comment line carries the
      cluster mode, atom count, and the count of frozen / cap atoms.  A
      per-atom flag column (``F`` for frozen, ``C`` for cap, ``-``
      otherwise) is appended after the (x, y, z) triple so that humans
      and minimal parsers can still see freeze/cap roles at a glance.
      The numerical XYZ block remains a valid XYZ stream for tools that
      ignore extra trailing tokens.
    * ``<sidecar_path>`` (default ``<filename>.cluster.json``) -- the
      full :class:`molcrys_kit.structures.cluster.ClusterProvenance`
      payload via ``to_dict``.  This is the canonical machine-readable
      record consumed by downstream Gaussian / ORCA input writers
      (kept out of this package on purpose; see
      :mod:`molcrys_kit.operations.cluster`).

    Parameters
    ----------
    cluster : CrystalCluster
        The carved cluster (must expose ``provenance``,
        ``frozen_local_indices``, and ``cap_local_indices``).
    filename : str
        Path of the XYZ output file.
    comment : str, optional
        Extra comment string appended to the XYZ comment line.
    sidecar_path : str, optional
        Path of the JSON sidecar.  Defaults to ``<filename>.cluster.json``.

    Returns
    -------
    str
        Absolute path of the JSON sidecar that was written.
    """
    # Local import to avoid a cycle (operations -> structures.cluster,
    # all loaded eagerly).
    from ..structures.cluster import CrystalCluster

    if not isinstance(cluster, CrystalCluster):
        raise TypeError(
            "write_xyz_with_freeze expects a CrystalCluster instance "
            f"(got {type(cluster).__name__}).  Use write_xyz for plain "
            "CrystalMolecule output."
        )

    symbols = cluster.get_chemical_symbols()
    positions = cluster.get_positions()
    frozen = set(cluster.frozen_local_indices)
    caps = set(cluster.cap_local_indices)

    provenance = cluster.provenance
    header_bits = [
        f"natoms={len(symbols)}",
        f"mode={provenance.mode}",
        f"frozen={len(frozen)}",
        f"caps={len(caps)}",
    ]
    if provenance.max_atoms is not None:
        header_bits.append(f"max_atoms={provenance.max_atoms}")
    if provenance.rcut_A is not None:
        header_bits.append(f"rcut_A={provenance.rcut_A:g}")
    if comment:
        header_bits.append(comment)

    xyz_lines = [str(len(symbols)), " ".join(header_bits)]
    for local_idx, (symbol, pos) in enumerate(zip(symbols, positions)):
        if local_idx in caps:
            flag = "C"
        elif local_idx in frozen:
            flag = "F"
        else:
            flag = "-"
        xyz_lines.append(
            f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} {flag}"
        )

    xyz_string = "\n".join(xyz_lines) + "\n"
    with open(filename, "w") as fh:
        fh.write(xyz_string)

    if sidecar_path is None:
        sidecar_path = filename + ".cluster.json"
    with open(sidecar_path, "w") as fh:
        json.dump(provenance.to_dict(), fh, indent=2, sort_keys=True)

    return os.path.abspath(sidecar_path)


def write_molecule_summary(molecule: CrystalMolecule) -> str:
    """
    Generate a summary string for a molecule.

    Parameters
    ----------
    molecule : CrystalMolecule
        The molecule to summarize.

    Returns
    -------
    str
        Summary string with molecular information.
    """
    summary = []
    summary.append(f"Molecule: {molecule.get_chemical_formula()}")
    summary.append(f"Number of atoms: {len(molecule)}")

    if hasattr(molecule, "crystal") and molecule.crystal is not None:
        centroid = molecule.get_centroid()
        centroid_frac = molecule.get_centroid_frac()
        summary.append(
            f"Centroid (Cartesian): ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})"
        )
        summary.append(
            f"Centroid (Fractional): ({centroid_frac[0]:.4f}, {centroid_frac[1]:.4f}, {centroid_frac[2]:.4f})"
        )

    com = molecule.get_center_of_mass()
    summary.append(f"Center of mass: ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")

    # Get ellipsoid radii
    radii = molecule.get_ellipsoid_radii()
    summary.append(f"Ellipsoid radii: {radii[0]:.3f} × {radii[1]:.3f} × {radii[2]:.3f}")

    return "\n".join(summary)


def write_cif(
    crystal: MolecularCrystal,
    filename: str = None,
    metadata: dict = None,
    *,
    representation: str = "p1",
    symmetry=None,
    sidecar_path: str | None = None,
) -> str:
    """
    Write a crystal structure to CIF format.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to write.
    filename : str, optional
        The filename to write to. If None, returns the CIF string.
    metadata : dict, optional
        Optional metadata to embed in the CIF header.  Currently supports the
        ``"termination_info"`` key whose value should be a
        :class:`~molcrys_kit.operations.surface.TerminationInfo` instance;
        when present the following custom CIF fields are written:
        ``_molcrys_termination_shift``,
        ``_molcrys_termination_index``,
        ``_molcrys_tasker_type``,
        ``_molcrys_tasker_polar``,
        ``_molcrys_tasker_dipole_per_area``,
        ``_molcrys_charge_source``,
        ``_molcrys_tasker2_corrected``.  A crystal's
        ``metadata["cell_transform"]`` is serialized as a compact, URL-safe
        JSON payload in ``_molcrys_cell_transform_json``.

    Returns
    -------
    str
        CIF format string if filename is None, otherwise None.

    Notes
    -----
    Disorder provenance (`sym_op_index`, `asym_id`, `site_symmetry_order`) is
    written in a separate `_molcrys_*` side-table loop keyed by atom index,
    rather than mixed into the `_atom_site_*` loop. This avoids label / column
    misalignment in some third-party CIF parsers and viewers while preserving
    MolCrysKit round-trip fidelity.
    """
    representation = str(representation).lower().replace("-", "_")
    if representation in {"asu", "true_asu", "non_p1", "space_group"}:
        return write_cif_asu(
            crystal,
            filename=filename,
            metadata=metadata,
            symmetry=symmetry,
        )
    if representation in {"disorder", "average", "statistical_average"}:
        return write_disorder_cif(
            crystal,
            filename=filename,
            sidecar_path=sidecar_path,
        )
    if representation not in {"p1", "expanded_p1", "full_cell"}:
        raise ValueError(
            "representation must be one of 'p1', 'asu', or 'statistical_average'"
        )

    from ..constants.config import (
        KEY_ASSEMBLY,
        KEY_ASYM_ID,
        KEY_ATOM_ID,
        KEY_DISORDER_GROUP,
        KEY_LABEL,
        KEY_OCCUPANCY,
        KEY_SITE_SYMMETRY_ORDER,
        KEY_SYM_OP_INDEX,
        KEY_U_CART,
        KEY_UISO,
    )

    # Transform provenance belongs to the crystal itself, while the optional
    # metadata argument remains available for operation-specific annotations.
    metadata = dict(metadata or {})
    if "cell_transform" not in metadata:
        transform = getattr(crystal, "metadata", {}).get("cell_transform")
        if transform is not None:
            metadata["cell_transform"] = transform

    lines = []
    lines.append("data_crystal")
    lines.append("_audit_creation_date              ?")
    lines.append("_audit_creation_method            'MolCrysKit'")
    lines.append("")

    # Add crystal system info (assuming general triclinic)
    lines.append("_symmetry_space_group_name_H-M    'P 1'")
    lines.append("_symmetry_Int_Tables_number       1")
    lines.append("")
    lines.append("loop_")
    lines.append("  _symmetry_equiv_pos_as_xyz")
    lines.append("  x,y,z")
    lines.append("")

    # An expanded P1 file is intentionally broad-format output.  Preserve the
    # requested source group in a separate provenance loop so readers do not
    # confuse the serialization symmetry with the physical/source symmetry.
    forward_context = (getattr(crystal, "metadata", {}) or {}).get("forward_context")
    if forward_context is not None:
        source_symmetry = getattr(forward_context, "symmetry", forward_context)
        source_symbol = getattr(source_symmetry, "space_group_symbol", None)
        source_number = getattr(source_symmetry, "space_group_number", None)
        lines.append("_molcrys_source_space_group_name_H-M  " + repr(source_symbol or "?"))
        lines.append(f"_molcrys_source_space_group_number    {source_number if source_number is not None else '?'}")
        lines.append("_molcrys_representation              'expanded_full_cell_P1'")
        lines.append("")
        lines.extend([
            "loop_",
            "  _molcrys_source_symop_id",
            "  _molcrys_source_symop_xyz",
        ])
        for index, operation in enumerate(getattr(forward_context, "operations", ())):
            lines.append(f"  {index + 1} '{_operation_xyz(operation)}'")
        lines.append("")

    # Add cell parameters
    a, b, c, alpha, beta, gamma = crystal.get_lattice_parameters()
    lines.append(f"_cell_length_a                    {a:.6f}")
    lines.append(f"_cell_length_b                    {b:.6f}")
    lines.append(f"_cell_length_c                    {c:.6f}")
    lines.append(f"_cell_angle_alpha                 {alpha:.6f}")
    lines.append(f"_cell_angle_beta                  {beta:.6f}")
    lines.append(f"_cell_angle_gamma                 {gamma:.6f}")
    lines.append("")

    # Write optional termination / Tasker metadata
    if metadata is not None:
        ti = metadata.get("termination_info")
        if ti is not None:
            lines.append(f"_molcrys_termination_shift        {ti.shift:.6f}")
            lines.append(f"_molcrys_termination_index        {ti.termination_index}")
            lines.append(f"_molcrys_tasker_type              '{ti.tasker_type}'")
            lines.append(f"_molcrys_tasker_polar             {ti.is_polar}")
            lines.append(
                f"_molcrys_tasker_dipole_per_area   {ti.dipole_per_area:.6e}"
            )
            lines.append(f"_molcrys_charge_source            '{ti.charge_source}'")
            lines.append(f"_molcrys_tasker2_corrected        {ti.tasker2_corrected}")
            lines.append("")

        transform = metadata.get("cell_transform")
        if transform is not None:
            payload = base64.urlsafe_b64encode(
                json.dumps(transform, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).decode("ascii")
            lines.append(f"_molcrys_cell_transform_json    '{payload}'")
            lines.append("")

    # Add atom positions with disorder metadata
    lines.append("loop_")
    lines.append("  _atom_site_label")
    lines.append("  _atom_site_type_symbol")
    lines.append("  _atom_site_fract_x")
    lines.append("  _atom_site_fract_y")
    lines.append("  _atom_site_fract_z")
    lines.append("  _atom_site_occupancy")
    lines.append("  _atom_site_disorder_group")
    lines.append("  _atom_site_disorder_assembly")
    lines.append("  _atom_site_U_iso_or_equiv")
    lines.append("  _atom_site_adp_type")

    # Collect all atoms from all molecules including their metadata
    all_symbols = []
    all_frac_positions = []
    all_occupancies = []
    all_disorder_groups = []
    all_assemblies = []
    all_labels = []
    all_sym_op_indices = []
    all_asym_ids = []
    all_site_sym_orders = []
    all_uiso = []
    all_u_cart = []
    all_atom_ids = []
    all_molecule_ids = []
    molecule_offsets = []
    all_bond_edges = []

    for molecule_index, mol in enumerate(crystal.molecules):
        molecule_offsets.append(len(all_symbols))
        molecule_offset = molecule_offsets[-1]
        # Get metadata arrays if they exist
        if hasattr(mol, 'arrays'):
            if KEY_OCCUPANCY in mol.arrays:
                occupancies = mol.arrays[KEY_OCCUPANCY]
            else:
                occupancies = np.full(len(mol), 1.0)
                
            if KEY_DISORDER_GROUP in mol.arrays:
                disorder_groups = mol.arrays[KEY_DISORDER_GROUP]
            else:
                disorder_groups = np.full(len(mol), 0, dtype=int)
                
            if KEY_ASSEMBLY in mol.arrays:
                assemblies = mol.arrays[KEY_ASSEMBLY]
            else:
                assemblies = np.array([''] * len(mol))
                
            if KEY_LABEL in mol.arrays:
                labels = mol.arrays[KEY_LABEL]
            else:
                labels = np.array(mol.get_chemical_symbols())
            if KEY_ATOM_ID in mol.arrays:
                atom_ids = mol.arrays[KEY_ATOM_ID]
            else:
                atom_ids = np.asarray([f"m{len(all_symbols)}:a{i}" for i in range(len(mol))], dtype=str)

            if KEY_SYM_OP_INDEX in mol.arrays:
                sym_op_indices = mol.arrays[KEY_SYM_OP_INDEX]
            else:
                sym_op_indices = np.full(len(mol), 0, dtype=int)

            if KEY_ASYM_ID in mol.arrays:
                asym_ids = mol.arrays[KEY_ASYM_ID]
            else:
                asym_ids = np.full(len(mol), -1, dtype=int)

            if KEY_SITE_SYMMETRY_ORDER in mol.arrays:
                site_sym_orders = mol.arrays[KEY_SITE_SYMMETRY_ORDER]
            else:
                site_sym_orders = np.full(len(mol), 1, dtype=int)

            if KEY_UISO in mol.arrays:
                uiso_values = np.asarray(mol.arrays[KEY_UISO], dtype=float)
            else:
                uiso_values = np.full(len(mol), np.nan, dtype=float)

            if KEY_U_CART in mol.arrays:
                u_cart_values = np.asarray(mol.arrays[KEY_U_CART], dtype=float).reshape(len(mol), 9)
            else:
                u_cart_values = np.full((len(mol), 9), np.nan, dtype=float)
        else:
            # If mol doesn't have arrays attribute, use defaults
            occupancies = np.full(len(mol), 1.0)
            disorder_groups = np.full(len(mol), 0, dtype=int)
            assemblies = np.array([''] * len(mol))
            labels = np.array(mol.get_chemical_symbols())
            atom_ids = np.asarray([f"m{len(all_symbols)}:a{i}" for i in range(len(mol))], dtype=str)
            sym_op_indices = np.full(len(mol), 0, dtype=int)
            asym_ids = np.full(len(mol), -1, dtype=int)
            site_sym_orders = np.full(len(mol), 1, dtype=int)
            uiso_values = np.full(len(mol), np.nan, dtype=float)
            u_cart_values = np.full((len(mol), 9), np.nan, dtype=float)

        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()

        for i, pos in enumerate(positions):
            frac_pos = crystal.cartesian_to_fractional(pos)
            # Ensure fractional coordinates are in [0,1)
            frac_pos = frac_pos - np.floor(frac_pos)

            all_symbols.append(symbols[i])
            all_frac_positions.append(frac_pos)
            all_occupancies.append(occupancies[i])
            all_disorder_groups.append(int(disorder_groups[i]))  # Ensure integer
            all_assemblies.append(assemblies[i] if assemblies[i] else '.')
            all_labels.append(labels[i] if labels[i] else f"{symbols[i]}{i+1}")
            all_sym_op_indices.append(int(sym_op_indices[i]))
            all_asym_ids.append(int(asym_ids[i]))
            all_site_sym_orders.append(int(site_sym_orders[i]))
            all_uiso.append(float(uiso_values[i]))
            all_u_cart.append(np.asarray(u_cart_values[i], dtype=float).reshape(9))
            all_atom_ids.append(str(atom_ids[i]))
            all_molecule_ids.append(int(molecule_index))

        # Preserve an already-materialized molecular graph.  Do not trigger
        # fresh geometric bond perception for molecules that have no graph:
        # explicit template connectivity is the source of truth here.
        graph = getattr(mol, "_graph", None)
        if graph is not None:
            for left, right, edge_data in graph.edges(data=True):
                left = int(left)
                right = int(right)
                if not (0 <= left < len(mol) and 0 <= right < len(mol)):
                    continue
                try:
                    distance = float(edge_data.get("distance"))
                except (AttributeError, TypeError, ValueError):
                    distance = float(
                        np.linalg.norm(
                            np.asarray(positions[right], dtype=float)
                            - np.asarray(positions[left], dtype=float)
                        )
                    )
                all_bond_edges.append(
                    (molecule_offset + left, molecule_offset + right, distance)
                )

    def _sanitise_label(label, symbol: str, atom_index: int) -> str:
        text = str(label).strip() if label is not None else ""
        if not text or text.startswith("_") or text.startswith("_molcrys_"):
            return f"{symbol}{atom_index + 1}"
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", text):
            return f"{symbol}{atom_index + 1}"
        return text

    # Write atom positions with standard atom-site metadata only.
    written_labels = []
    used_labels = set()
    for i, (symbol, frac_pos, occ, group, assembly, label, soi, aid, sso, uiso, u_cart) in enumerate(
        zip(all_symbols, all_frac_positions, all_occupancies, all_disorder_groups,
            all_assemblies, all_labels, all_sym_op_indices, all_asym_ids,
            all_site_sym_orders, all_uiso, all_u_cart)
    ):
        safe_label = _sanitise_label(label, symbol, i)
        if safe_label in used_labels:
            safe_label = f"{symbol}{i + 1}"
            suffix = 1
            while safe_label in used_labels:
                safe_label = f"{symbol}{i + 1}_{suffix}"
                suffix += 1
        used_labels.add(safe_label)
        written_labels.append(safe_label)
        has_u_cart = np.all(np.isfinite(u_cart))
        uiso_token = f"{uiso:.8f}" if np.isfinite(uiso) else "."
        adp_type = "Uani" if has_u_cart else ("Uiso" if np.isfinite(uiso) else ".")
        lines.append(
            f"  {safe_label:8s} {symbol:4s} {frac_pos[0]:10.6f} {frac_pos[1]:10.6f} {frac_pos[2]:10.6f} "
            f"{occ:8.5f} {group:2d} {assembly if assembly != '.' else '.':4s} {uiso_token} {adp_type}"
        )

    anisotropic_rows = []
    cart_basis = np.asarray(crystal.lattice, dtype=float).T
    reciprocal_lengths = np.linalg.norm(np.linalg.inv(cart_basis), axis=1)
    cart_to_cif = np.linalg.inv(cart_basis @ np.diag(reciprocal_lengths))
    for label, raw_u_cart in zip(written_labels, all_u_cart):
        raw_u_cart = np.asarray(raw_u_cart, dtype=float)
        if raw_u_cart.size != 9 or not np.all(np.isfinite(raw_u_cart)):
            continue
        matrix = raw_u_cart.reshape(3, 3)
        u_cif = cart_to_cif @ matrix @ cart_to_cif.T
        anisotropic_rows.append((label, u_cif))

    if anisotropic_rows:
        lines.append("")
        lines.append("loop_")
        lines.append("  _atom_site_aniso_label")
        lines.append("  _atom_site_aniso_U_11")
        lines.append("  _atom_site_aniso_U_22")
        lines.append("  _atom_site_aniso_U_33")
        lines.append("  _atom_site_aniso_U_12")
        lines.append("  _atom_site_aniso_U_13")
        lines.append("  _atom_site_aniso_U_23")
        for label, matrix in anisotropic_rows:
            lines.append(
                f"  {label:8s} {matrix[0, 0]:.8f} {matrix[1, 1]:.8f} "
                f"{matrix[2, 2]:.8f} {matrix[0, 1]:.8f} "
                f"{matrix[0, 2]:.8f} {matrix[1, 2]:.8f}"
            )

    if all_bond_edges:
        lines.append("")
        lines.append("_molcrys_bond_table_source 'materialized_graph'")
        lines.append("loop_")
        lines.append("  _geom_bond_atom_site_label_1")
        lines.append("  _geom_bond_atom_site_label_2")
        lines.append("  _geom_bond_distance")
        lines.append("  _geom_bond_type")
        for left, right, distance in all_bond_edges:
            lines.append(
                f"  {written_labels[left]} {written_labels[right]} {distance:.6f} single"
            )

    lines.append("")
    lines.append("loop_")
    lines.append("  _molcrys_atom_index")
    lines.append("  _molcrys_sym_op_index")
    lines.append("  _molcrys_asym_id")
    lines.append("  _molcrys_site_symmetry_order")
    lines.append("  _molcrys_atom_id")
    lines.append("  _molcrys_molecule_id")
    for i, (soi, aid, sso, molecule_id) in enumerate(
        zip(
            all_sym_op_indices,
            all_asym_ids,
            all_site_sym_orders,
            all_molecule_ids,
        )
    ):
        atom_id = ""
        if i < len(all_atom_ids):
            atom_id = str(all_atom_ids[i]).replace(chr(39), chr(39) * 2)
        lines.append(f"  {i:4d} {soi:4d} {aid:4d} {sso:2d} '{atom_id}' {molecule_id:d}")

    cif_string = "\n".join(lines) + "\n"

    if filename is not None:
        with open(filename, "w") as f:
            f.write(cif_string)

    return cif_string


def _forward_context_for_io(obj, symmetry=None):
    """Resolve the validated forward context attached to a result."""
    if symmetry is not None:
        if not hasattr(symmetry, "lattice"):
            from types import SimpleNamespace

            symmetry = SimpleNamespace(
                lattice=np.asarray(obj.lattice, dtype=float),
                operations=tuple(getattr(symmetry, "operations", symmetry)),
                symmetry=symmetry,
            )
        return symmetry
    metadata = getattr(obj, "metadata", {}) or {}
    context = metadata.get("forward_context")
    if context is None:
        context = metadata.get("source_symmetry")
    if context is None:
        raise ValueError(
            "a true ASU CIF requires the validated forward context or an explicit symmetry"
        )
    return context


def _context_parts(context):
    if hasattr(context, "lattice"):
        lattice = np.asarray(getattr(context, "lattice"), dtype=float)
    else:
        raise ValueError("symmetry context must provide a lattice")
    if hasattr(context, "operations"):
        operations = tuple(context.operations)
    elif hasattr(context, "symmetry") and hasattr(context.symmetry, "operations"):
        operations = tuple(context.symmetry.operations)
    else:
        operations = tuple(context)
    symmetry_obj = getattr(context, "symmetry", context)
    return lattice, operations, symmetry_obj


def _fraction_token(value: float) -> str:
    value = float(value)
    if abs(value) < 1e-10:
        return "0"
    fraction = Fraction(value).limit_denominator(24)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _operation_xyz(operation) -> str:
    if getattr(operation, "xyz", None):
        return str(operation.xyz)
    symbols = ("x", "y", "z")
    expressions = []
    for row, translation in zip(np.asarray(operation.rotation, dtype=int), operation.translation):
        terms = []
        for coefficient, symbol in zip(row, symbols):
            coefficient = int(coefficient)
            if coefficient == 0:
                continue
            if not terms:
                prefix = "" if coefficient == 1 else "-" if coefficient == -1 else str(coefficient)
            else:
                prefix = "+" if coefficient == 1 else "-" if coefficient == -1 else (f"+{coefficient}" if coefficient > 0 else str(coefficient))
            terms.append(f"{prefix}{symbol}")
        if abs(float(translation)) > 1e-10:
            token = _fraction_token(float(translation))
            terms.append(("+" if float(translation) > 0 and terms else "") + token)
        expressions.append("".join(terms) or "0")
    return ",".join(expressions)


def _forward_atom_rows(crystal: MolecularCrystal, lattice: np.ndarray):
    from ..constants.config import (
        KEY_ASSEMBLY,
        KEY_ASYM_ID,
        KEY_ATOM_ID,
        KEY_DISORDER_GROUP,
        KEY_LABEL,
        KEY_OCCUPANCY,
        KEY_SITE_SYMMETRY_ORDER,
        KEY_SYM_OP_INDEX,
    )

    def _entity_key(molecule, molecule_index: int, atom_index: int, arrays) -> tuple:
        """Return the identity used when reducing full-cell rows to an ASU.

        Coordinates and element symbols identify a crystallographic site, but
        they do not identify a molecular entity.  Forward-generated molecules
        carry stable pose/template metadata; ordinary structures fall back to
        their explicit atom IDs, and finally to the molecule/local-atom index.
        The conservative fallbacks intentionally retain independent entities
        when no provenance proves that they are the same atom orbit.
        """
        info = getattr(molecule, "info", {}) or {}
        template_id = info.get("forward_template_identifier")
        pose_index = info.get("forward_pose_index")
        if template_id is not None and pose_index is not None:
            identities = info.get("forward_atom_identities")
            identity = (
                identities[atom_index]
                if identities is not None and len(identities) > atom_index
                else atom_index
            )
            return ("forward", str(template_id), int(pose_index), str(identity))
        asym_ids = arrays.get(KEY_ASYM_ID)
        if asym_ids is not None and len(asym_ids) == len(molecule):
            asym_id = int(asym_ids[atom_index])
            if asym_id >= 0:
                # CIF readers commonly retain the originating asymmetric-unit
                # site while assigning fresh per-molecule atom IDs during
                # expansion.  The ASU site ID is the stable crystallographic
                # identity in that representation.
                return ("asym_id", asym_id)
        atom_ids = arrays.get(KEY_ATOM_ID)
        if atom_ids is not None and len(atom_ids) == len(molecule):
            return ("atom_id", str(atom_ids[atom_index]))
        return ("molecule", int(molecule_index), int(atom_index))

    rows = []
    for molecule_index, molecule in enumerate(crystal.molecules):
        symbols = molecule.get_chemical_symbols()
        positions = np.asarray(molecule.get_positions(), dtype=float)
        fractional = positions @ np.linalg.inv(lattice)
        arrays = getattr(molecule, "arrays", {})
        occupancies = np.asarray(arrays.get(KEY_OCCUPANCY, np.ones(len(molecule))), dtype=float)
        groups = np.asarray(arrays.get(KEY_DISORDER_GROUP, np.zeros(len(molecule), dtype=int)), dtype=int)
        assemblies = np.asarray(arrays.get(KEY_ASSEMBLY, np.full(len(molecule), ".", dtype="U1")))
        labels = np.asarray(arrays.get(KEY_LABEL, np.asarray(symbols, dtype=str)))
        sym_op_indices = np.asarray(
            arrays.get(KEY_SYM_OP_INDEX, np.zeros(len(molecule), dtype=int)),
            dtype=int,
        )
        asym_ids = np.asarray(
            arrays.get(KEY_ASYM_ID, np.full(len(molecule), -1, dtype=int)),
            dtype=int,
        )
        site_symmetry_orders = np.asarray(
            arrays.get(KEY_SITE_SYMMETRY_ORDER, np.ones(len(molecule), dtype=int)),
            dtype=int,
        )
        forward_ids = (getattr(molecule, "info", {}) or {}).get("forward_atom_identities")
        for atom_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "fractional": fractional[atom_index],
                    "occupancy": float(occupancies[atom_index]),
                    "group": int(groups[atom_index]),
                    "assembly": str(assemblies[atom_index]) or ".",
                    "label": str(labels[atom_index]) or f"{symbol}{len(rows) + 1}",
                    "atom_id": str(
                        forward_ids[atom_index]
                        if forward_ids is not None and len(forward_ids) > atom_index
                        else arrays.get(KEY_ATOM_ID, np.asarray([""] * len(molecule)))[atom_index]
                    ),
                    "entity_key": _entity_key(molecule, molecule_index, atom_index, arrays),
                    "molecule_id": int(molecule_index),
                    "molecule_index": molecule_index,
                    "atom_index": atom_index,
                    "sym_op_index": int(sym_op_indices[atom_index]),
                    "asym_id": int(asym_ids[atom_index]),
                    "site_symmetry_order": int(site_symmetry_orders[atom_index]),
                }
            )
    return rows


def _is_equivalent_site(left, right, operations, tolerance=1e-7) -> bool:
    if (
        left["symbol"] != right["symbol"]
        or left.get("entity_key") != right.get("entity_key")
        or int(left.get("group", 0)) != int(right.get("group", 0))
        or str(left.get("assembly", ".")) != str(right.get("assembly", "."))
        or not np.isclose(
            float(left.get("occupancy", 1.0)),
            float(right.get("occupancy", 1.0)),
            atol=tolerance,
            rtol=0.0,
        )
    ):
        return False
    for operation in operations:
        mapped = operation.apply(left["fractional"], wrap=False)
        if np.allclose(mapped - right["fractional"], np.rint(mapped - right["fractional"]), atol=tolerance, rtol=0.0):
            return True
    return False


def _is_symmetry_related_site(left, right, operations, tolerance=1e-7) -> bool:
    """Return whether two rows occupy one crystallographic site orbit.

    This deliberately ignores entity/disorder identity.  Those fields decide
    whether rows are alternatives at the same site, while this predicate
    decides whether retaining both coordinates would cause a second symmetry
    expansion of the same site orbit.
    """
    for operation in operations:
        mapped = operation.apply(left["fractional"], wrap=False)
        if np.allclose(
            mapped - right["fractional"],
            np.rint(mapped - right["fractional"]),
            atol=tolerance,
            rtol=0.0,
        ):
            return True
    return False


def _reduce_asu_rows(rows, operations, tolerance: float):
    """Reduce materialized rows to one coordinate per crystallographic orbit.

    Forward-generated operation mates with the same site identity are reduced
    by ``_is_equivalent_site``.  Independent identities that happen to be
    related by a non-identity operation are retained as alternatives, but are
    moved to the first representative coordinate so the operation loop does
    not expand them twice.  Their summed occupation is bounded by one per
    crystallographic slot; a full-occupancy collision is rejected instead of
    emitting an invalid CIF.
    """
    selected = []
    for row in rows:
        occupancy = float(row.get("occupancy", 1.0))
        if not np.isfinite(occupancy) or occupancy < -tolerance or occupancy > 1.0 + tolerance:
            raise ValueError(
                "true-ASU export requires each site occupancy to lie in [0, 1] "
                f"(got {occupancy:.8g} for {row.get('atom_id') or row.get('label')})"
            )
    for row in rows:
        if not any(
            _is_equivalent_site(row, existing, operations, tolerance)
            for existing in selected
        ):
            selected.append(row)

    # Build connected components under symmetry equivalence.  A pair may be
    # related through more than one operation, hence the small union-find
    # rather than only comparing every row with the first one.
    parent = list(range(len(selected)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            if _is_symmetry_related_site(
                selected[left], selected[right], operations, tolerance
            ):
                union(left, right)

    components = {}
    for index in range(len(selected)):
        components.setdefault(find(index), []).append(index)
    for indices in components.values():
        representative = selected[indices[0]]
        total_occupancy = sum(
            float(selected[index].get("occupancy", 1.0)) for index in indices
        )
        if total_occupancy > 1.0 + tolerance:
            identities = ", ".join(
                str(
                    selected[index].get("atom_id")
                    or selected[index].get("entity_key")
                )
                for index in indices
            )
            raise ValueError(
                "true-ASU export overoccupies a symmetry-related crystallographic "
                f"site (occupancy={total_occupancy:.8g}; identities={identities})"
            )
        coordinate = np.asarray(representative["fractional"], dtype=float).copy()
        for index in indices[1:]:
            # Retain each independent identity/disorder row, but serialize it
            # at the one representative coordinate of the site orbit.
            selected[index]["fractional"] = coordinate.copy()
    return selected


def write_cif_asu(
    crystal: MolecularCrystal,
    filename: str | None = None,
    metadata: dict | None = None,
    *,
    symmetry=None,
) -> str:
    """Write a true asymmetric-unit CIF with the requested operation set.

    The input crystal is normally the full ordered expansion returned by the
    forward builder.  Atom rows are reduced by the supplied group before the
    non-P1 operation loop is written, so a reader will not expand the same
    special-position atom twice.
    """
    context = _forward_context_for_io(crystal, symmetry=symmetry)
    lattice, operations, symmetry_obj = _context_parts(context)
    crystal_metadata = dict(getattr(crystal, "metadata", {}) or {})
    export_metadata = dict(crystal_metadata)
    export_metadata.update(metadata or {})
    chemistry_metadata = export_metadata.get("cif_chemistry")
    if isinstance(chemistry_metadata, dict):
        representation = chemistry_metadata.get("molcrys_representation")
    else:
        representation = None
    representation = export_metadata.get("molcrys_representation", representation)
    representation = export_metadata.get("representation", representation)

    # A P1/statistical output already contains every symmetry image.  Reusing
    # its source context as the serialization group would expand those rows a
    # second time.  Keep the physical source group in provenance, but emit the
    # materialized rows with identity symmetry.
    materialized = str(representation or "").casefold() in {
        "expanded_full_cell_p1",
        "statistical_average",
        "ordered_realization",
        "materialized_p1",
    }
    source_operations = operations
    if materialized:
        from ..structures.symmetry import identity_operation

        operations = (identity_operation(),)
    rows = _forward_atom_rows(crystal, lattice)
    tolerance = float(getattr(context, "fractional_tolerance", 1e-7))
    for row in rows:
        occupancy = float(row.get("occupancy", 1.0))
        if not np.isfinite(occupancy) or occupancy < -tolerance or occupancy > 1.0 + tolerance:
            raise ValueError(
                "ASU export requires each site occupancy to lie in [0, 1] "
                f"(got {occupancy:.8g} for {row.get('atom_id') or row.get('label')})"
            )
    selected = rows if materialized else _reduce_asu_rows(rows, operations, tolerance)
    # Stable labels are required even when source templates used duplicate
    # element labels.
    used = set()
    for index, row in enumerate(selected):
        label = row["label"]
        if (
            not label
            or label in used
            or not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", label)
        ):
            label = f"{row['symbol']}{index + 1}"
        while label in used:
            label = f"{row['symbol']}{index + 1}_{len(used)}"
        row["label"] = label
        used.add(label)

    a, b, c = [float(np.linalg.norm(vector)) for vector in lattice]
    alpha = float(
        np.degrees(
            np.arccos(np.clip(np.dot(lattice[1], lattice[2]) / (b * c), -1.0, 1.0))
        )
    )
    beta = float(
        np.degrees(
            np.arccos(np.clip(np.dot(lattice[0], lattice[2]) / (a * c), -1.0, 1.0))
        )
    )
    gamma = float(
        np.degrees(
            np.arccos(np.clip(np.dot(lattice[0], lattice[1]) / (a * b), -1.0, 1.0))
        )
    )
    number = getattr(symmetry_obj, "space_group_number", None)
    symbol = getattr(symmetry_obj, "space_group_symbol", None) or "?"
    hall = getattr(symmetry_obj, "hall_symbol", None)
    if materialized:
        number = 1
        symbol = "P 1"
        hall = None
    lines = [
        "data_molcrys_forward_asu",
        "_audit_creation_method 'MolCrysKit forward construction'",
        "_molcrys_representation "
        f"'{('expanded_full_cell_P1' if materialized else 'ordered_asymmetric_unit')}'",
        f"_symmetry_space_group_name_H-M '{symbol}'",
        f"_symmetry_Int_Tables_number {number if number is not None else '?'}",
    ]
    if hall:
        lines.append(f"_space_group_name_Hall '{hall}'")
    if materialized:
        source_symbol = getattr(symmetry_obj, "space_group_symbol", None)
        source_number = getattr(symmetry_obj, "space_group_number", None)
        lines.extend(
            [
                f"_molcrys_source_space_group_name_H-M '{source_symbol or '?'}'",
                "_molcrys_source_space_group_number "
                f"{source_number if source_number is not None else '?'}",
                "",
                "loop_",
                "  _molcrys_source_symop_id",
                "  _molcrys_source_symop_xyz",
            ]
        )
        lines.extend(
            f"  {index + 1} '{_operation_xyz(operation)}'"
            for index, operation in enumerate(source_operations)
        )
    lines.extend(["", "loop_", "  _symmetry_equiv_pos_as_xyz"])
    lines.extend(f"  '{_operation_xyz(operation)}'" for operation in operations)
    lines.extend([
        "",
        f"_cell_length_a {a:.8f}",
        f"_cell_length_b {b:.8f}",
        f"_cell_length_c {c:.8f}",
        f"_cell_angle_alpha {alpha:.8f}",
        f"_cell_angle_beta {beta:.8f}",
        f"_cell_angle_gamma {gamma:.8f}",
        "",
    ])
    transform = export_metadata.get("cell_transform")
    if transform is not None:
        payload = base64.urlsafe_b64encode(
            json.dumps(transform, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        lines.extend([f"_molcrys_cell_transform_json '{payload}'", ""])
    termination = export_metadata.get("termination_info")
    if termination is not None:
        for name, attribute in (
            ("_molcrys_termination_shift", "shift"),
            ("_molcrys_termination_index", "termination_index"),
            ("_molcrys_tasker_type", "tasker_type"),
            ("_molcrys_tasker_polar", "is_polar"),
            ("_molcrys_tasker_dipole_per_area", "dipole_per_area"),
            ("_molcrys_charge_source", "charge_source"),
            ("_molcrys_tasker2_corrected", "tasker2_corrected"),
        ):
            value = (
                termination.get(attribute)
                if isinstance(termination, dict)
                else getattr(termination, attribute, None)
            )
            if value is None:
                continue
            if isinstance(value, str):
                value = f"'{value.replace(chr(39), chr(39) * 2)}'"
            lines.append(f"{name} {value}")
        lines.append("")
    lines.extend([
        "loop_",
        "  _atom_site_label",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
        "  _atom_site_occupancy",
        "  _atom_site_disorder_group",
        "  _atom_site_disorder_assembly",
        "  _atom_site_site_symmetry_order",
    ])
    for row in selected:
        fractional = row["fractional"] - np.floor(row["fractional"])
        assembly = row["assembly"] if row["assembly"] not in {"", "None"} else "."
        lines.append(
            f"  {row['label']} {row['symbol']} {fractional[0]:.8f} "
            f"{fractional[1]:.8f} {fractional[2]:.8f} "
            f"{row['occupancy']:.8f} {row['group']} {assembly} "
            f"{int(row['site_symmetry_order'])}"
        )
    selected_by_source = {
        (int(row["molecule_index"]), int(row["atom_index"])): row
        for row in selected
    }
    rows_by_source = {
        (int(row["molecule_index"]), int(row["atom_index"])): row
        for row in rows
    }

    def _selected_equivalent(row):
        """Map a full-cell bond endpoint to its retained ASU representative."""
        if row is None:
            return None
        for candidate in selected:
            if _is_equivalent_site(row, candidate, operations, tolerance):
                return candidate
        return None

    def _symmetry_site_code(right_row, right_source_row) -> str:
        """Encode the operation taking the retained second endpoint to its image.

        CIF's ``_geom_bond_site_symmetry_2`` code belongs to atom site 2.  A
        true-ASU bond therefore records the operation that maps the retained
        ASU row for the *right* endpoint to the source image used by the full
        cell edge.  Keeping that convention in the writer lets the reader
        reconstruct cross-image bonds without guessing from distances.
        """
        if right_row is None or right_source_row is None:
            return "."
        right_fractional = np.asarray(right_row["fractional"], dtype=float)
        right_source_fractional = np.asarray(
            right_source_row["fractional"], dtype=float
        )
        for operation_index, operation in enumerate(operations):
            mapped = np.asarray(operation.apply(right_fractional, wrap=False), dtype=float)
            shift = np.rint(right_source_fractional - mapped).astype(int)
            if not np.allclose(
                mapped + shift,
                right_source_fractional,
                atol=tolerance,
                rtol=0.0,
            ):
                continue
            if np.any(np.abs(shift) > 4):
                raise ValueError("ASU bond symmetry image is outside CIF code range")
            digits = "".join(str(int(value) + 5) for value in shift)
            return f"{operation_index + 1}_{digits}"
        return "."

    asu_bonds = []
    had_bond_candidates = False
    seen_bonds = set()
    for molecule_index, molecule in enumerate(crystal.molecules):
        info = getattr(molecule, "info", {}) or {}
        raw_pairs = info.get("forward_connectivity") or info.get("bond_pairs")
        if raw_pairs is None:
            graph = getattr(molecule, "_graph", None)
            raw_pairs = list(graph.edges()) if graph is not None else []
        for raw_pair in raw_pairs:
            try:
                left, right = int(raw_pair[0]), int(raw_pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            had_bond_candidates = True
            left_source_row = rows_by_source.get((molecule_index, left))
            right_source_row = rows_by_source.get((molecule_index, right))
            left_row = selected_by_source.get((molecule_index, left))
            right_row = selected_by_source.get((molecule_index, right))
            # Root each written bond at a retained ASU endpoint.  Reversing
            # an edge whose second endpoint is retained handles arbitrary
            # graph orientation while skipping complete symmetry-mate
            # molecules that would otherwise duplicate the same bond orbit.
            if left_row is None and right_row is not None:
                left, right = right, left
                left_source_row, right_source_row = right_source_row, left_source_row
                left_row, right_row = right_row, left_row
            if left_row is None:
                continue
            if right_row is None:
                right_row = _selected_equivalent(right_source_row)
            if right_row is None:
                continue
            symmetry_code = _symmetry_site_code(right_row, right_source_row)
            if (
                left_row["label"] == right_row["label"]
                and symmetry_code == "."
            ):
                raise ValueError(
                    "ASU export cannot represent a collapsed explicit bond without "
                    "a symmetry operation mapping its endpoints"
                )
            key = (tuple(sorted((left_row["label"], right_row["label"]))), symmetry_code)
            if key in seen_bonds:
                continue
            seen_bonds.add(key)
            distance = float(
                np.linalg.norm(
                    np.asarray(molecule.positions[right], dtype=float)
                    - np.asarray(molecule.positions[left], dtype=float)
                )
            )
            asu_bonds.append(
                (left_row["label"], right_row["label"], distance, symmetry_code)
            )
    if asu_bonds:
        def _format_bond_endpoint(left: str, right: str) -> str:
            return f"'{right}'" if left == right else right

        lines.extend(
            [
                "",
                "loop_",
                "  _geom_bond_atom_site_label_1",
                "  _geom_bond_atom_site_label_2",
                "  _geom_bond_distance",
                "  _geom_bond_type",
                "  _geom_bond_site_symmetry_2",
            ]
        )
        lines.extend(
            (
                f"  {left} {_format_bond_endpoint(left, right)} "
                f"{distance:.6f} single {symmetry_code}"
            )
            for left, right, distance, symmetry_code in asu_bonds
        )
    elif had_bond_candidates:
        # Keep a discoverable marker for callers that inspect the standard
        # bond schema if a malformed source edge could not be mapped.
        lines.append(
            "# _geom_bond_atom_site_label_1 omitted: unmappable source edge"
        )
    lines.extend(
        [
            "",
            "loop_",
            "  _molcrys_atom_index",
            "  _molcrys_asu_atom_index",
            "  _molcrys_sym_op_index",
            "  _molcrys_asym_id",
            "  _molcrys_site_symmetry_order",
            "  _molcrys_atom_id",
            "  _molcrys_molecule_id",
        ]
    )
    for index, row in enumerate(selected):
        atom_id = str(row.get("atom_id", "")) or "."
        lines.append(
            f"  {index} {index} {int(row.get('sym_op_index', 0))} "
            f"{int(row.get('asym_id', -1))} {int(row.get('site_symmetry_order', 1))} "
            f"'{atom_id.replace(chr(39), chr(39) * 2)}' {int(row.get('molecule_id', -1))}"
        )
    lines.extend(["", "# The full-cell expansion is represented by the operation loop above.", ""])
    text = "\n".join(lines)
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(text)
    return text


def write_cif_p1(crystal: MolecularCrystal, filename: str | None = None, metadata: dict | None = None) -> str:
    """Explicit alias for the expanded full-cell P1 representation."""
    return write_cif(crystal, filename=filename, metadata=metadata, representation="p1")


def write_disorder_cif(model, filename: str | None = None, sidecar_path: str | None = None) -> str:
    """Write the derived average sites of a statistical disorder model.

    Whole-molecule alternatives and correlation semantics are retained in a
    JSON sidecar whenever a filename is provided.  The CIF contains per-site
    marginal occupancies and disorder-group/assembly identifiers; it never
    materialises mutually exclusive alternatives as simultaneous full atoms.
    """
    lattice, operations, symmetry_obj = _context_parts(model.context)
    a, b, c = [float(np.linalg.norm(vector)) for vector in lattice]
    alpha = float(np.degrees(np.arccos(np.clip(np.dot(lattice[1], lattice[2]) / (b * c), -1.0, 1.0))))
    beta = float(np.degrees(np.arccos(np.clip(np.dot(lattice[0], lattice[2]) / (a * c), -1.0, 1.0))))
    gamma = float(np.degrees(np.arccos(np.clip(np.dot(lattice[0], lattice[1]) / (a * b), -1.0, 1.0))))
    # ``average_sites`` is already a full-cell marginal view: construction
    # iterates over every center orbit before merging sites.  Declaring the
    # source space group here would make independent CIF readers expand those
    # rows a second time and report occupancies 2/4/6/... .  Serialize the
    # materialized average in P1 and retain the validated source operations in
    # a provenance loop for MolCrysKit round-trips.
    lines = [
        "data_molcrys_statistical_average",
        "_audit_creation_method 'MolCrysKit forward construction'",
        "_molcrys_representation 'statistical_average'",
        f"_molcrys_correlation_status '{model.correlation_status}'",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "",
        "loop_",
        "  _symmetry_equiv_pos_as_xyz",
        "  'x,y,z'",
    ]
    source_symbol = getattr(symmetry_obj, "space_group_symbol", None)
    source_number = getattr(symmetry_obj, "space_group_number", None)
    lines.extend([
        "",
        f"_molcrys_source_space_group_name_H-M '{source_symbol or '?'}'",
        f"_molcrys_source_space_group_number {source_number if source_number is not None else '?'}",
        "loop_",
        "  _molcrys_source_symop_id",
        "  _molcrys_source_symop_xyz",
    ])
    lines.extend(
        f"  {index + 1} '{_operation_xyz(operation)}'"
        for index, operation in enumerate(operations)
    )
    lines.extend([
        "",
        f"_cell_length_a {a:.8f}",
        f"_cell_length_b {b:.8f}",
        f"_cell_length_c {c:.8f}",
        f"_cell_angle_alpha {alpha:.8f}",
        f"_cell_angle_beta {beta:.8f}",
        f"_cell_angle_gamma {gamma:.8f}",
        "",
        "loop_",
        "  _atom_site_label",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
        "  _atom_site_occupancy",
        "  _atom_site_disorder_group",
        "  _atom_site_disorder_assembly",
        "  _molcrys_alternative_ids",
    ])
    for index, site in enumerate(model.average_sites):
        group = int(hashlib.sha1(site.orbit_id.encode("utf-8")).hexdigest()[:8], 16) % 100000 + 1
        fractional = site.fractional - np.floor(site.fractional)
        alternative_ids = ",".join(site.alternative_ids) or "."
        lines.append(
            f"  {site.symbol}{index + 1} {site.symbol} {fractional[0]:.8f} {fractional[1]:.8f} {fractional[2]:.8f} "
            f"{site.occupancy:.8f} {group} {site.orbit_id} '{alternative_ids}'"
        )
    lines.extend(["", "# Whole-molecule exclusivity and state weights are in the sidecar.", ""])
    text = "\n".join(lines)
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(text)
        if sidecar_path is None:
            sidecar_path = filename + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(model.to_dict(), handle, indent=2, sort_keys=True, default=str)
    return text


def write_forward_cif(obj, filename: str | None = None, *, representation: str = "p1", symmetry=None, sidecar_path: str | None = None) -> str:
    """Dispatch P1, true-ASU, and statistical-average CIF representations."""
    return write_cif(
        obj,
        filename=filename,
        representation=representation,
        symmetry=symmetry,
        sidecar_path=sidecar_path,
    )


def write_poscar(
    crystal: MolecularCrystal,
    filename: Optional[str] = None,
    *,
    comment: Optional[str] = None,
    direct: bool = True,
    wrap: bool = False,
    sort: bool = False,
    selective_dynamics: Optional[np.ndarray] = None,
) -> str:
    """
    Write a crystal structure to VASP POSCAR/CONTCAR format.

    POSCAR can represent lattice vectors, species, positions, and optional
    selective-dynamics flags, but it cannot represent MolCrysKit metadata such
    as occupancy, disorder groups, assemblies, or atom labels. Non-default
    metadata are therefore dropped with a warning and summarized in the POSCAR
    comment line.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to write.
    filename : str, optional
        The filename to write to. If None, only returns the POSCAR string.
    comment : str, optional
        Comment line to use as POSCAR line 1. A lossy-export note is appended
        when metadata must be dropped.
    direct : bool, default=True
        If True, write Direct coordinates. If False, write Cartesian positions.
    wrap : bool, default=False
        If True, wrap atomic positions into the unit cell before writing. The
        default preserves MolCrysKit's unwrapped, contiguous molecule geometry.
    sort : bool, default=False
        If True, let ASE group atoms alphabetically by element for VASP output.
        The default preserves the MolecularCrystal/ASE atom order, which is
        required by workflows such as NEB interpolation.
    selective_dynamics : np.ndarray, optional
        VASP-style boolean flags with shape ``(n_atoms, 3)``. True writes ``T``
        (coordinate free to move) and False writes ``F`` (coordinate fixed).

    Returns
    -------
    str
        POSCAR format string.
    """
    from ..constants.config import (
        KEY_ASSEMBLY,
        KEY_DISORDER_GROUP,
        KEY_LABEL,
        KEY_OCCUPANCY,
    )

    atoms = crystal.to_ase()
    symbols = atoms.get_chemical_symbols()
    n_total = len(atoms)

    if n_total == 0:
        raise ValueError("Cannot write POSCAR for an empty MolecularCrystal")

    occupancies = np.full(n_total, 1.0)
    disorder_groups = np.zeros(n_total, dtype=int)
    assemblies = np.array([""] * n_total, dtype=object)
    labels = np.array(symbols, dtype=object)

    indices_lists = [
        mol.info.get("atom_indices")
        for mol in crystal.molecules
    ]
    flat_indices = [
        int(index)
        for indices in indices_lists
        if indices is not None
        for index in indices
    ]
    use_global_indices = (
        all(indices is not None for indices in indices_lists)
        and len(flat_indices) == n_total
        and set(flat_indices) == set(range(n_total))
    )

    cursor = 0
    for mol, indices in zip(crystal.molecules, indices_lists):
        n_atoms = len(mol)
        arrays = getattr(mol, "arrays", {})
        if use_global_indices:
            targets = np.array([int(index) for index in indices], dtype=int)
        else:
            targets = np.arange(cursor, cursor + n_atoms, dtype=int)
        cursor += n_atoms

        if KEY_OCCUPANCY in arrays:
            occupancies[targets] = arrays[KEY_OCCUPANCY]
        if KEY_DISORDER_GROUP in arrays:
            disorder_groups[targets] = arrays[KEY_DISORDER_GROUP]
        if KEY_ASSEMBLY in arrays:
            assemblies[targets] = arrays[KEY_ASSEMBLY]
        if KEY_LABEL in arrays:
            labels[targets] = arrays[KEY_LABEL]

    dropped = {}
    occupancy_count = sum(
        not np.isclose(float(occ), 1.0) for occ in occupancies
    )
    if occupancy_count:
        dropped["occupancy"] = occupancy_count

    disorder_count = sum(int(group) != 0 for group in disorder_groups)
    if disorder_count:
        dropped["disorder_group"] = disorder_count

    assembly_count = sum(str(assembly).strip() not in {"", "."} for assembly in assemblies)
    if assembly_count:
        dropped["assembly"] = assembly_count

    label_count = 0
    for atom_index, (symbol, label) in enumerate(zip(symbols, labels), start=1):
        label_text = str(label).strip()
        default_labels = {"", ".", symbol, f"{symbol}{atom_index}"}
        if label_text not in default_labels:
            label_count += 1
    if label_count:
        dropped["label"] = label_count

    lossy_summary = ", ".join(f"{key}:{value}" for key, value in dropped.items())
    if dropped:
        warnings.warn(
            "write_poscar: POSCAR cannot represent MolCrysKit metadata; "
            f"dropping {lossy_summary}",
            UserWarning,
            stacklevel=2,
        )

    if wrap:
        atoms.wrap()

    if selective_dynamics is not None:
        flags = np.asarray(selective_dynamics, dtype=bool)
        expected_shape = (len(atoms), 3)
        if flags.shape != expected_shape:
            raise ValueError(
                "selective_dynamics must have shape "
                f"{expected_shape}, got {flags.shape}"
            )
        atoms.set_constraint(
            [
                FixScaled(i, mask=tuple(~flags[i]))
                for i in range(len(atoms))
            ]
        )

    base_comment = comment or f"MolCrysKit {atoms.get_chemical_formula()}"
    line1 = base_comment
    if lossy_summary:
        line1 = f"{base_comment} | MolCrysKit lossy export: dropped {lossy_summary}"

    buffer = StringIO()
    write_vasp(buffer, atoms, direct=direct, sort=sort, vasp5=True)
    lines = buffer.getvalue().splitlines()
    lines[0] = line1
    poscar_string = "\n".join(lines) + "\n"

    if filename is not None:
        with open(filename, "w") as f:
            f.write(poscar_string)

    return poscar_string


def write_poscar_sequence(
    crystals: Sequence[MolecularCrystal],
    directory: str,
    *,
    padding: int = 2,
    filename: str = "POSCAR",
    comment_prefix: Optional[str] = None,
    **poscar_kwargs,
) -> list[str]:
    """Write interpolated crystal frames as a VASP image directory sequence.

    Frames are written as ``directory/00/POSCAR``, ``directory/01/POSCAR``, ...
    by default. Additional keyword arguments are forwarded to
    :func:`write_poscar`.
    """
    if not crystals:
        raise ValueError("Cannot write an empty POSCAR sequence")
    os.makedirs(directory, exist_ok=True)
    written = []
    for index, crystal in enumerate(crystals):
        image_dir = os.path.join(directory, f"{index:0{padding}d}")
        os.makedirs(image_dir, exist_ok=True)
        path = os.path.join(image_dir, filename)
        kwargs = dict(poscar_kwargs)
        if comment_prefix is not None and "comment" not in kwargs:
            kwargs["comment"] = f"{comment_prefix} image={index}"
        write_poscar(crystal, path, **kwargs)
        written.append(os.path.abspath(path))
    return written


def write_cif_sequence(
    crystals: Sequence[MolecularCrystal],
    directory: str,
    *,
    prefix: str = "frame",
    padding: int = 3,
    metadata: Optional[dict] = None,
) -> list[str]:
    """Write interpolated crystal frames as individual CIF files."""
    if not crystals:
        raise ValueError("Cannot write an empty CIF sequence")
    os.makedirs(directory, exist_ok=True)
    written = []
    for index, crystal in enumerate(crystals):
        path = os.path.join(directory, f"{prefix}_{index:0{padding}d}.cif")
        write_cif(crystal, path, metadata=metadata)
        written.append(os.path.abspath(path))
    return written


def write_trajectory(
    crystals: Sequence[MolecularCrystal],
    filename: str,
    *,
    format: str = "extxyz",
    info: Optional[Sequence[dict] | dict] = None,
    arrays: Optional[Sequence[dict] | dict] = None,
    append: bool = False,
    **write_kwargs,
) -> str:
    """Write molecular-crystal frames as an XYZ-family trajectory.

    Parameters
    ----------
    crystals : Sequence[MolecularCrystal]
        Frames to write.
    filename : str
        Output trajectory path.
    format : {"extxyz", "xyz"}, default="extxyz"
        ``"extxyz"`` preserves lattice, PBC, molecule indices, metadata, and
        calculator results through MolCrysKit's ExtXYZ writer. ``"xyz"`` writes
        a plain multi-frame XYZ stream through ASE and is intentionally lossy.
    info, arrays : dict or sequence of dict, optional
        Per-frame ExtXYZ payloads. Supported only for ``format="extxyz"``.
    append : bool, default=False
        Append frames to an existing trajectory.
    **write_kwargs
        Extra writer options forwarded to the underlying ASE/ExtXYZ writer.

    Returns
    -------
    str
        Absolute path to the written trajectory.
    """
    if not crystals:
        raise ValueError("Cannot write an empty trajectory")

    normalized = format.lower().replace("-", "")
    if normalized in {"extxyz", "extendedxyz"}:
        from .extxyz import write_extxyz

        write_extxyz(
            list(crystals),
            filename,
            append=append,
            info=info,
            arrays=arrays,
            **write_kwargs,
        )
        return os.path.abspath(filename)

    if normalized == "xyz":
        if info is not None or arrays is not None:
            raise ValueError("info/arrays payloads are supported only for extxyz output")
        images = [crystal.to_ase() for crystal in crystals]
        ase.io.write(filename, images, format="xyz", append=append, **write_kwargs)
        return os.path.abspath(filename)

    raise ValueError(
        f"Unsupported trajectory format {format!r}; expected 'extxyz' or 'xyz'."
    )


def write_vesta(crystal: MolecularCrystal, filename: str = None) -> str:
    """
    Generate VESTA format representation of a crystal.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to export.
    filename : str, optional
        If provided, write to this file. Otherwise, return the VESTA string.

    Returns
    -------
    str
        VESTA format string if filename is not provided, otherwise None.
    """
    lines = ["#VESTA_FORMAT_VERSION 3.5.0", ""]  # Header

    # Add crystal structure information
    lines.append("CRYSTAL")
    lines.append("INFO")
    lines.append("LEN a=1.0 b=1.0 c=1.0")
    lines.append("ANG alpha=90.0 beta=90.0 gamma=90.0")
    lines.append("SPGP I1")
    lines.append("SPGN 1")
    lines.append("")

    # Add lattice information
    lines.append("CELLP")
    a, b, c = crystal.lattice[0], crystal.lattice[1], crystal.lattice[2]
    lines.append(f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}")
    lines.append(f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f}")
    lines.append(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    lines.append("  0.000000   0.000000   0.000000")
    lines.append("")

    # Add atom positions (ASE Atoms/Atom: iterate over mol; Atom has .symbol and .position in Cartesian coords)
    lines.append("STRUC")
    atom_index = 1
    for mol in crystal.molecules:
        for atom in mol:
            frac = crystal.cartesian_to_fractional(atom.position)
            lines.append(
                f"  {atom_index:4d} {atom.symbol:4s} {atom_index:4d}  1.0000  {frac[0]:8.5f}  {frac[1]:8.5f}  {frac[2]:8.5f}  1a  1  0  0  0  0"
            )
            atom_index += 1
    lines.append("  0 0 0 0 0 0 0")
    lines.append("")

    vesta_content = "\n".join(lines) + "\n"

    if filename:
        with open(filename, "w") as f:
            f.write(vesta_content)
        return None
    else:
        return vesta_content


def export_for_vesta(crystal: MolecularCrystal, filename: str) -> None:
    """
    Export a crystal structure to a VESTA-compatible file.

    Parameters
    ----------
    crystal : MolecularCrystal
        The crystal to export.
    filename : str
        Output filename.
    """
    write_vesta(crystal, filename)
