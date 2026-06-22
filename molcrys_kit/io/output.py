"""
Output functionality for molecular crystal structures.

This module provides functions for writing molecular crystal data to various formats.
"""

import json
import os
from io import StringIO
from typing import Optional, Sequence
import warnings

import ase.io
import numpy as np
from ase.constraints import FixScaled
from ase.io.vasp import write_vasp

from ..structures.molecule import CrystalMolecule
from ..structures.crystal import MolecularCrystal


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


def write_cif(crystal: MolecularCrystal, filename: str = None, metadata: dict = None) -> str:
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
        ``_molcrys_tasker2_corrected``.

    Returns
    -------
    str
        CIF format string if filename is None, otherwise None.
    """
    from ..constants.config import KEY_OCCUPANCY, KEY_DISORDER_GROUP, KEY_ASSEMBLY, KEY_LABEL

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

    for mol in crystal.molecules:
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
        else:
            # If mol doesn't have arrays attribute, use defaults
            occupancies = np.full(len(mol), 1.0)
            disorder_groups = np.full(len(mol), 0, dtype=int)
            assemblies = np.array([''] * len(mol))
            labels = np.array(mol.get_chemical_symbols())

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

    # Write atom positions with metadata
    for i, (symbol, frac_pos, occ, group, assembly, label) in enumerate(
        zip(all_symbols, all_frac_positions, all_occupancies, all_disorder_groups, all_assemblies, all_labels)
    ):
        lines.append(
            f"  {label:8s} {symbol:4s} {frac_pos[0]:10.6f} {frac_pos[1]:10.6f} {frac_pos[2]:10.6f} "
            f"{occ:8.5f} {group:2d} {assembly if assembly != '.' else '.':4s} .  Uiso"
        )

    cif_string = "\n".join(lines) + "\n"

    if filename is not None:
        with open(filename, "w") as f:
            f.write(cif_string)

    return cif_string


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
