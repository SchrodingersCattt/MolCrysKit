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


def _entry_value(entry: Any, name: str, warnings: list[str], default: object = None) -> object:
    """Read an optional CCDC entry field without failing on release variants."""
    if entry is None:
        return default
    try:
        value = getattr(entry, name, default)
    except Exception as exc:  # pragma: no cover - release-specific CCDC handles
        warnings.append(f"Entry field {name} unavailable: {type(exc).__name__}: {exc}")
        return default
    return default if value in (None, "") else value


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

    raw_mapping: dict[str, tuple[int, str]] = {}
    raw_groups: set[tuple[str, int]] = set()
    try:
        # CCDC disorder group identifiers are release/API data and may start
        # at zero.  MolCrysKit reserves zero for ordered atoms, so collect the
        # raw identifiers first and remap each (assembly, raw group) pair to a
        # positive deterministic ID below.
        for assembly in disorder.assemblies:
            assembly_id = str(getattr(assembly, "id", ""))
            for group in assembly.groups:
                raw_group_id = int(getattr(group, "id", 0))
                raw_groups.add((assembly_id, raw_group_id))
                for atom in group.atoms:
                    label = str(atom.label)
                    raw_mapping[label] = (raw_group_id, assembly_id)
                    # CCDC appends ``?`` to labels on some alternate sites
                    # (for example ``C4?``), while the disorder hierarchy can
                    # expose the same atom as ``C4``.  Keep both spellings so
                    # that the group/assembly information survives export.
                    if label.endswith("?"):
                        raw_mapping.setdefault(label[:-1], (raw_group_id, assembly_id))
                    else:
                        raw_mapping.setdefault(f"{label}?", (raw_group_id, assembly_id))
    except Exception as exc:  # CCDC release-specific null handles
        warnings.append(f"Could not read complete disorder hierarchy: {exc}")

    # CCDC permits disorder group 0, while MolCrysKit reserves zero for
    # ordered atoms.  Re-number every observed (assembly, group) pair into a
    # deterministic positive range, preserving distinctions between groups.
    group_numbers = {
        key: index + 1
        for index, key in enumerate(sorted(raw_groups, key=lambda item: (item[0], item[1])))
    }
    mapping = {
        label: (group_numbers[(assembly_id, group_id)], assembly_id)
        for label, (group_id, assembly_id) in raw_mapping.items()
        if (assembly_id, group_id) in group_numbers
    }
    return mapping, warnings


def export_full_cif_from_csd(
    reader: Any,
    refcode: str,
    output_path: Path,
) -> dict[str, Any]:
    """Export one CSD refcode to a metadata-preserving CIF."""
    warnings: list[str] = []
    try:
        # Entry metadata is separate from the crystallographic model in the
        # CCDC API.  Fetch it explicitly so the exported CIF remains auditable.
        entry = reader.entry(refcode)
    except Exception as exc:  # pragma: no cover - release-specific CCDC handles
        entry = None
        warnings.append(f"Entry metadata unavailable: {type(exc).__name__}: {exc}")
    crystal = reader.crystal(refcode)

    # ``disordered_molecule`` is not uniformly implemented across CCDC
    # releases.  It may be ``None`` (ordered entries such as BENZEN02), or a
    # property access may raise when no disorder model exists.  Fall back in a
    # documented order and fail with a useful error only when no molecule is
    # available at all.
    molecule = None
    for attr in ("disordered_molecule", "molecule", "asymmetric_unit_molecule"):
        try:
            candidate = getattr(crystal, attr, None)
        except Exception as exc:  # pragma: no cover - release-specific CCDC
            warnings.append(f"{attr} unavailable: {type(exc).__name__}: {exc}")
            candidate = None
        if candidate is not None:
            molecule = candidate
            break
    if molecule is None:
        raise RuntimeError(f"CSD entry {refcode} has no readable molecular model")

    disorder_map, disorder_warnings = _disorder_lookup(crystal)
    warnings.extend(disorder_warnings)

    atom_rows: list[dict[str, Any]] = []
    for atom in molecule.atoms:
        label = str(atom.label) if atom.label else "?"
        try:
            fractional = atom.fractional_coordinates
            if fractional is None:
                raise ValueError("null fractional coordinates")
            coordinates = tuple(float(getattr(fractional, axis)) for axis in ("x", "y", "z"))
        except Exception as exc:  # CCDC null TSite handles (e.g. BENZEN02 H)
            fractional = None
            coordinates = (None, None, None)
            warnings.append(f"{label}: fractional coordinates unavailable ({type(exc).__name__}: {exc})")
        occupancy = 1.0
        try:
            if atom.occupancy is not None:
                occupancy = float(atom.occupancy)
        except (AttributeError, TypeError, ValueError) as exc:
            warnings.append(f"{label}: occupancy unavailable ({exc})")

        u_iso = None
        adp_type = "?"
        aniso: dict[str, float] | None = None
        try:
            displacement = atom.displacement_parameters
            if displacement is not None:
                u_iso = float(displacement.isotropic_equivalent)
                if displacement.type == "Anisotropic":
                    adp_type = "Uani"
                    values = displacement.values
                    aniso = {
                        "u11": float(values[0][0]),
                        "u22": float(values[1][1]),
                        "u33": float(values[2][2]),
                        "u12": float(values[0][1]),
                        "u13": float(values[0][2]),
                        "u23": float(values[1][2]),
                    }
                else:
                    adp_type = "Uiso"
        except Exception as exc:  # CCDC release-specific null handles
            warnings.append(f"{label}: displacement parameters unavailable ({exc})")

        group_id, assembly_id = disorder_map.get(label, (None, None))
        atom_rows.append(
            {
                "label": label,
                "symbol": str(atom.atomic_symbol) if atom.atomic_symbol else "?",
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
                "occupancy": occupancy,
                "u_iso": u_iso,
                "adp_type": adp_type,
                "aniso": aniso,
                "group": group_id,
                "assembly": assembly_id,
            }
        )

    # Collect bond candidates before filtering coordinate-less atoms.  CCDC
    # can expose a valid bond length even when one endpoint's fractional
    # coordinates are a null TSite; those rows must not leak into the CIF.
    bond_candidates: list[tuple[str, str, float, str]] = []
    try:
        for bond in molecule.bonds:
            try:
                atom_1, atom_2 = bond.atoms
                bond_candidates.append(
                    (
                        str(atom_1.label),
                        str(atom_2.label),
                        float(bond.length),
                        str(bond.bond_type),
                    )
                )
            except Exception as exc:  # null TSite handles on missing H coords
                warnings.append(f"Bond export skipped: {type(exc).__name__}: {exc}")
    except (AttributeError, TypeError, ValueError) as exc:
        warnings.append(f"Bond export incomplete: {exc}")

    a, b, c = crystal.cell_lengths
    alpha, beta, gamma = crystal.cell_angles
    # A valid CSD entry can carry partial occupancy without a populated
    # disorder hierarchy (and vice versa).  Emit both occupancy and disorder
    # columns whenever either signal is present; this is essential for the
    # IQUHUY orientational-disorder fixture and avoids silently dropping
    # occupancy metadata on older CCDC releases.
    has_disorder = bool(disorder_map) or any(row["occupancy"] < 0.999 for row in atom_rows)

    try:
        it_number, setting = crystal.spacegroup_number_and_setting
        it_number = int(it_number)
        setting = int(setting)
    except (AttributeError, TypeError, ValueError):
        it_number, setting = None, None
        warnings.append("Space-group IT number/setting unavailable from CCDC API")

    # CCDC 3.6.x exposes the complete operation list and the IT number/setting,
    # but not the Hall symbol.  Write an explicit missing value rather than
    # inventing one from a symbol (which can be setting-dependent).
    hall_symbol = None
    if hall_symbol is None:
        warnings.append("Hall symbol unavailable from CCDC API; wrote '?' in CIF")

    chemical_name = _entry_value(entry, "chemical_name", warnings, "?")
    formula = _entry_value(entry, "formula", warnings, "?")
    r_factor = _entry_value(entry, "r_factor", warnings, None)
    try:
        r_factor_value = None if r_factor is None else float(r_factor)
    except (TypeError, ValueError):
        r_factor_value = None
    lines = [
        f"data_{refcode}",
        "_audit_creation_method " + _cif_quote("CCDC Python API; query-csd-structures skill"),
        "_database_code_CSD " + _cif_quote(refcode),
        "_chemical_name_common " + _cif_quote(chemical_name),
        "_chemical_formula_sum " + _cif_quote(formula),
        f"_refine_ls_R_factor_all {_float_text(r_factor_value, 2)}",
        f"_cell_length_a {_float_text(a)}",
        f"_cell_length_b {_float_text(b)}",
        f"_cell_length_c {_float_text(c)}",
        f"_cell_angle_alpha {_float_text(alpha, 2)}",
        f"_cell_angle_beta {_float_text(beta, 2)}",
        f"_cell_angle_gamma {_float_text(gamma, 2)}",
        f"_cell_volume {_float_text(crystal.cell_volume, 2)}",
        f"_cell_formula_units_Z {int(crystal.z_value) if crystal.z_value else '?'}",
        "_symmetry_space_group_name_H-M " + _cif_quote(crystal.spacegroup_symbol or "?"),
        f"_space_group_IT_number {it_number if it_number is not None else '?'}",
        f"_space_group_setting {setting if setting is not None else '?'}",
        "_space_group_name_Hall " + _cif_quote(hall_symbol),
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
    written_labels: set[str] = set()

    def _has_coordinates(row: dict[str, Any]) -> bool:
        values = (row.get("x"), row.get("y"), row.get("z"))
        return all(value is not None and _is_finite_number(value) for value in values)

    def _is_finite_number(value: object) -> bool:
        try:
            return bool(float(value) == float(value) and abs(float(value)) != float("inf"))
        except (TypeError, ValueError):
            return False

    for row in atom_rows:
        if not _has_coordinates(row):
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
        written_labels.add(row["label"])
    lines.append("")

    # Anisotropic displacement rows are keyed by atom label and therefore
    # must use the same coordinate-backed atom set as the atom-site loop.
    aniso_atoms = [
        row
        for row in atom_rows
        if row["aniso"] is not None and row["label"] in written_labels
    ]
    if aniso_atoms:
        lines.extend(
            [
                "loop_",
                "_atom_site_aniso_label",
                "_atom_site_aniso_U_11",
                "_atom_site_aniso_U_22",
                "_atom_site_aniso_U_33",
                "_atom_site_aniso_U_12",
                "_atom_site_aniso_U_13",
                "_atom_site_aniso_U_23",
            ]
        )
        for row in aniso_atoms:
            values = row["aniso"]
            assert values is not None
            lines.append(
                "  "
                + " ".join(
                    [
                        row["label"],
                        *(_float_text(values[key], 6) for key in ("u11", "u22", "u33", "u12", "u13", "u23")),
                    ]
                )
            )
        lines.append("")

    bonds: list[tuple[str, str, float, str]] = []
    for label_1, label_2, distance, bond_type in bond_candidates:
        if label_1 not in written_labels or label_2 not in written_labels:
            warnings.append(
                f"Bond export skipped: omitted atom endpoint ({label_1}, {label_2})"
            )
            continue
        bonds.append((label_1, label_2, distance, bond_type))

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
        "spacegroup_symbol": crystal.spacegroup_symbol,
        "spacegroup_it_number": it_number,
        "spacegroup_setting": setting,
        "spacegroup_hall": hall_symbol,
        "source": "CSD",
        "chemical_name": None if chemical_name == "?" else str(chemical_name),
        "formula": None if formula == "?" else str(formula),
        "r_factor": r_factor_value,
        "status": "ok",
        # Keep the retrieval manifest portable; the absolute local path is
        # neither needed for provenance nor safe to copy into a fixture bundle.
        "output": output_path.name,
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
        "ccdc_module": "ccdc",
        "ccdc_api_version": str(getattr(ccdc, "__version__", "unknown")),
        "database": "CSD",
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
