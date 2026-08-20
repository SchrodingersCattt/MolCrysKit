#!/usr/bin/env python
"""Export full CIF files from exact CSD refcodes.

This retrieval-only script preserves occupancy and disorder metadata that can be
lost by generic CCDC CIF writer paths. Resolve disorder later with MolCrysKit's
maintained CLI or public API.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cif_quote(value: object) -> str:
    text = "?" if value is None else str(value)
    return "'" + text.replace("'", "''") + "'"


def _float_text(value: object, precision: int = 4) -> str:
    if value is None:
        return "?"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "?"


def _read_refcodes(values: Iterable[str], refcodes_file: Path | None) -> list[str]:
    requested = list(values)
    if refcodes_file is not None:
        requested.extend(refcodes_file.read_text(encoding="utf-8").splitlines())

    result: list[str] = []
    seen: set[str] = set()
    for raw in requested:
        refcode = raw.strip().upper()
        if not refcode or refcode.startswith("#") or refcode in seen:
            continue
        seen.add(refcode)
        result.append(refcode)
    return result


def _disorder_lookup(crystal: Any) -> tuple[dict[str, tuple[int, str]], list[str]]:
    mapping: dict[str, tuple[int, str]] = {}
    warnings: list[str] = []
    disorder = getattr(crystal, "disorder", None)
    if disorder is None:
        return mapping, warnings

    try:
        for assembly in disorder.assemblies:
            assembly_id = str(getattr(assembly, "id", ""))
            for group in assembly.groups:
                group_id = int(getattr(group, "id", 0))
                for atom in group.atoms:
                    mapping[str(atom.label)] = (group_id, assembly_id)
    except (AttributeError, TypeError, ValueError) as exc:
        warnings.append(f"Could not read complete disorder hierarchy: {exc}")
    return mapping, warnings


def export_full_cif_from_csd(
    reader: Any,
    refcode: str,
    output_path: Path,
) -> dict[str, Any]:
    """Export one CSD refcode to a metadata-preserving CIF."""
    entry = reader.entry(refcode)
    crystal = reader.crystal(refcode)
    molecule = crystal.disordered_molecule or crystal.molecule
    disorder_map, warnings = _disorder_lookup(crystal)

    atom_rows: list[dict[str, Any]] = []
    for atom in molecule.atoms:
        label = str(atom.label) if atom.label else "?"
        fractional = atom.fractional_coordinates
        occupancy = 1.0
        try:
            if atom.occupancy is not None:
                occupancy = float(atom.occupancy)
        except (AttributeError, TypeError, ValueError) as exc:
            warnings.append(f"{label}: occupancy unavailable ({exc})")

        u_iso = None
        adp_type = "?"
        try:
            displacement = atom.displacement_parameters
            if displacement is not None:
                u_iso = float(displacement.isotropic_equivalent)
                adp_type = "Uani" if displacement.type == "Anisotropic" else "Uiso"
        except (AttributeError, TypeError, ValueError) as exc:
            warnings.append(f"{label}: displacement parameters unavailable ({exc})")

        group_id, assembly_id = disorder_map.get(label, (None, None))
        atom_rows.append(
            {
                "label": label,
                "symbol": str(atom.atomic_symbol) if atom.atomic_symbol else "?",
                "x": float(fractional.x) if fractional is not None else None,
                "y": float(fractional.y) if fractional is not None else None,
                "z": float(fractional.z) if fractional is not None else None,
                "occupancy": occupancy,
                "u_iso": u_iso,
                "adp_type": adp_type,
                "group": group_id,
                "assembly": assembly_id,
            }
        )

    bonds: list[tuple[str, str, float, str]] = []
    try:
        for bond in molecule.bonds:
            atom_1, atom_2 = bond.atoms
            bonds.append(
                (
                    str(atom_1.label),
                    str(atom_2.label),
                    float(bond.length),
                    str(bond.bond_type),
                )
            )
    except (AttributeError, TypeError, ValueError) as exc:
        warnings.append(f"Bond export incomplete: {exc}")

    a, b, c = crystal.cell_lengths
    alpha, beta, gamma = crystal.cell_angles
    has_disorder = any(row["group"] is not None for row in atom_rows)

    lines = [
        f"data_{refcode}",
        "_audit_creation_method " + _cif_quote("CCDC Python API; query-csd-structures skill"),
        "_chemical_name_common " + _cif_quote(entry.chemical_name or "?"),
        "_chemical_formula_sum " + _cif_quote(entry.formula or "?"),
        f"_refine_ls_R_factor_all {_float_text(entry.r_factor, 2)}",
        f"_cell_length_a {_float_text(a)}",
        f"_cell_length_b {_float_text(b)}",
        f"_cell_length_c {_float_text(c)}",
        f"_cell_angle_alpha {_float_text(alpha, 2)}",
        f"_cell_angle_beta {_float_text(beta, 2)}",
        f"_cell_angle_gamma {_float_text(gamma, 2)}",
        f"_cell_volume {_float_text(crystal.cell_volume, 2)}",
        f"_cell_formula_units_Z {int(crystal.z_value) if crystal.z_value else '?'}",
        "_symmetry_space_group_name_H-M " + _cif_quote(crystal.spacegroup_symbol or "?"),
        "",
    ]

    symmetry_operators = crystal.symmetry_operators
    if symmetry_operators:
        lines.extend(["loop_", "_symmetry_equiv_pos_as_xyz"])
        lines.extend(f"  {_cif_quote(operator)}" for operator in symmetry_operators)
        lines.append("")

    lines.extend(
        [
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
            "_atom_site_U_iso_or_equiv",
            "_atom_site_thermal_displace_type",
        ]
    )
    if has_disorder:
        lines.extend(["_atom_site_disorder_assembly", "_atom_site_disorder_group"])

    written_atom_count = 0
    for row in atom_rows:
        if row["x"] is None:
            warnings.append(f"{row['label']}: omitted because fractional coordinates are unavailable")
            continue
        fields = [
            row["label"],
            row["symbol"],
            f"{row['x']:.6f}",
            f"{row['y']:.6f}",
            f"{row['z']:.6f}",
            f"{row['occupancy']:.4f}",
            _float_text(row["u_iso"]),
            row["adp_type"],
        ]
        if has_disorder:
            fields.extend(
                [
                    str(row["assembly"]) if row["assembly"] is not None else ".",
                    str(row["group"]) if row["group"] is not None else ".",
                ]
            )
        lines.append("  " + " ".join(fields))
        written_atom_count += 1
    lines.append("")

    if bonds:
        lines.extend(
            [
                "loop_",
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_distance",
                "_ccdc_geom_bond_type",
            ]
        )
        lines.extend(
            f"  {label_1} {label_2} {distance:.4f} {bond_type}"
            for label_1, label_2, distance, bond_type in bonds
        )
        lines.append("")

    lines.append("#END")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "refcode": refcode,
        "status": "ok",
        "output": str(output_path),
        "n_source_atoms": len(atom_rows),
        "n_written_atoms": written_atom_count,
        "n_partial_occupancy": sum(row["occupancy"] < 0.999 for row in atom_rows),
        "n_disorder_sites": sum(row["group"] is not None for row in atom_rows),
        "n_bonds": len(bonds),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refcodes", nargs="*", default=[], help="Exact CSD refcodes")
    parser.add_argument("--refcodes-file", type=Path, help="One CSD refcode per line")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    refcodes = _read_refcodes(args.refcodes, args.refcodes_file)
    if not refcodes:
        parser.error("provide --refcodes and/or --refcodes-file")

    try:
        import ccdc
        from ccdc.io import EntryReader
    except ImportError:
        parser.error(
            "the licensed CCDC Python API is unavailable in this interpreter; "
            "activate the CSD environment first"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = args.output_dir / "cifs"
    records: list[dict[str, Any]] = []

    with EntryReader("CSD") as reader:
        for refcode in refcodes:
            output_path = cif_dir / f"{refcode}.cif"
            try:
                records.append(export_full_cif_from_csd(reader, refcode, output_path))
            # CCDC exposes release-specific exception classes that are not
            # available without a licensed installation.
            except Exception as exc:  # noqa: BLE001
                records.append(
                    {
                        "refcode": refcode,
                        "status": "error",
                        "output": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "ccdc_module": str(Path(ccdc.__file__).resolve()),
        "requested_refcodes": refcodes,
        "n_requested": len(refcodes),
        "n_succeeded": sum(record["status"] == "ok" for record in records),
        "n_failed": sum(record["status"] != "ok" for record in records),
        "records": records,
    }
    manifest_path = args.output_dir / "retrieval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 1 if manifest["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
