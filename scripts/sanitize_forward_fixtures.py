#!/usr/bin/env python
"""Create minimal, de-identified CIF fixtures for the forward builder.

The CCDC exporter intentionally writes a faithful intermediate CIF.  That file
can contain a CSD refcode, compound name, R factor, audit/citation records and
machine-specific provenance.  Those fields are useful locally but are
redundant for a geometry regression fixture and should not be committed.

This script keeps only the crystallographic contract needed by Issue #172:
cell parameters, space-group declaration and operations, atom coordinates and
occupancies, disorder assembly/group labels, anisotropic displacement data (if
present), and the CCDC bond loop (if present).  It builds a new CIF document
instead of editing text in place, so audit comments and hidden paths cannot
leak into the result.

The input directory is expected to contain CIFs produced by
``demo_csd_disorder_workflow.py``.  It may be used with any equivalent CIFs
that contain the same core loops.

Example (run in the ordinary MolCrysKit environment)::

    python scripts/sanitize_forward_fixtures.py \
        --input-dir C:/temp/csd_export/cifs \
        --output-dir tests/data/cif/forward

The source refcodes are retained only in the manifest as reproducibility
metadata.  They are absent from the sanitized CIF data block and atom fields.
No CCDC API or network access is needed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FIXTURES = ("IBPRAC", "BENZEN02", "HXMTAM10", "IQUHUY")

CELL_TAGS = (
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
    "_cell_volume",
    "_cell_formula_units_Z",
)

ATOM_TAGS = (
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z",
    "_atom_site_occupancy",
    "_atom_site_U_iso_or_equiv",
    "_atom_site_thermal_displace_type",
    "_atom_site_disorder_assembly",
    "_atom_site_disorder_group",
)

BOND_TAGS = (
    "_geom_bond_atom_site_label_1",
    "_geom_bond_atom_site_label_2",
    "_geom_bond_distance",
    "_geom_bond_site_symmetry_2",
    "_geom_bond_publ_flag",
    "_ccdc_geom_bond_type",
    "_geom_bond_type",
)


def _canonical_bond_tags(source_tags: list[str]) -> list[str]:
    """Return generic bond-loop tags while retaining source bond values."""
    tags = [tag for tag in BOND_TAGS if tag in source_tags]
    if "_ccdc_geom_bond_type" in tags:
        index = tags.index("_ccdc_geom_bond_type")
        tags[index] = "_geom_bond_type"
    return tags


def _require_gemmi() -> Any:
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("sanitize_forward_fixtures.py requires the 'gemmi' package") from exc
    return gemmi


def _loop_for(block: Any, tag: str) -> Any | None:
    column = block.find_loop(tag)
    if column is None:
        return None
    try:
        return column.get_loop()
    except (AttributeError, RuntimeError):
        return None


def _value(block: Any, *tags: str) -> str | None:
    for tag in tags:
        try:
            value = block.find_value(tag)
        except (AttributeError, RuntimeError):
            value = ""
        if value:
            return str(value)
    return None


def _float_value(value: str | None) -> float | None:
    if value in (None, "", ".", "?"):
        return None
    match = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)", value)
    return float(match.group(1)) if match else None


def _copy_loop(new_block: Any, source_loop: Any, tags: list[str], target_tags: list[str] | None = None) -> int:
    """Copy selected columns from a Gemmi loop and return row count."""

    source_tags = list(source_loop.tags)
    indices = [source_tags.index(tag) for tag in tags]
    target = new_block.init_loop("", target_tags or tags)
    width = source_loop.width()
    values = list(source_loop.values)
    for row_start in range(0, len(values), width):
        target.add_row([str(values[row_start + index]) for index in indices])
    return source_loop.length()


def _copy_atom_loop(
    new_block: Any,
    source_loop: Any,
    tags: list[str],
) -> tuple[int, list[dict[str, Any]]]:
    """Copy atom rows and normalize CCDC's zero-based disorder groups.

    CCDC disorder groups are release/API data and may be numbered ``0, 1,`` …
    (IQUHUY uses exactly that numbering).  MolCrysKit reserves group ``0`` for
    ordered atoms, so writing CCDC's raw ``0`` would silently erase one
    alternative.  The CIF receives deterministic positive IDs; the raw-to-CIF
    mapping is retained in the manifest for provenance.
    """

    source_tags = list(source_loop.tags)
    indices = [source_tags.index(tag) for tag in tags]
    group_source_index = source_tags.index("_atom_site_disorder_group") if "_atom_site_disorder_group" in source_tags else None
    assembly_source_index = source_tags.index("_atom_site_disorder_assembly") if "_atom_site_disorder_assembly" in source_tags else None
    group_target_index = tags.index("_atom_site_disorder_group") if "_atom_site_disorder_group" in tags else None

    target = new_block.init_loop("", tags)
    width = source_loop.width()
    values = list(source_loop.values)
    mappings: list[dict[str, Any]] = []
    group_map: dict[tuple[str, str], int] = {}
    for row_start in range(0, len(values), width):
        row = [str(values[row_start + index]) for index in indices]
        if group_source_index is not None and group_target_index is not None:
            raw_group = str(values[row_start + group_source_index])
            raw_assembly = (
                str(values[row_start + assembly_source_index])
                if assembly_source_index is not None
                else ""
            )
            if raw_group not in ("", ".", "?"):
                key = (raw_assembly, raw_group)
                if key not in group_map:
                    group_map[key] = len(group_map) + 1
                    mappings.append(
                        {
                            "assembly": None if raw_assembly in ("", ".", "?") else raw_assembly,
                            "source_group": raw_group,
                            "fixture_group": group_map[key],
                        }
                    )
                row[group_target_index] = str(group_map[key])
            else:
                row[group_target_index] = "."
        target.add_row(row)
    return source_loop.length(), mappings


def _supplement_benzen02_hydrogens(block: Any, gemmi: Any) -> int:
    """Add idealized in-plane H sites when CCDC has null BENZEN02 H sites.

    BENZEN02 stores hydrogen atoms as riding-model sites with no coordinates in
    the CCDC API.  The issue requires a complete benzene template, so derive
    one H per carbon along the outward ring normal using the measured C ring,
    a 1.083 Å C–H distance, and the deposited cell.  This is explicitly a
    fixture preparation step; it is recorded in the manifest and is not
    presented as an experimental hydrogen refinement.
    """

    atom_loop = _loop_for(block, "_atom_site_label")
    if atom_loop is None:
        return 0
    tags = list(atom_loop.tags)
    symbol_index = tags.index("_atom_site_type_symbol")
    x_indices = [tags.index(tag) for tag in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")]
    label_index = tags.index("_atom_site_label")
    width = atom_loop.width()
    rows = [list(atom_loop.values[i : i + width]) for i in range(0, len(atom_loop.values), width)]
    if any(str(row[symbol_index]).upper() == "H" for row in rows):
        return 0
    carbon_rows = [row for row in rows if str(row[symbol_index]).upper() == "C"]
    if len(carbon_rows) != 6:
        return 0

    def scalar(tag: str, default: float) -> float:
        value = _float_value(block.find_value(tag))
        return default if value is None else value

    cell = gemmi.UnitCell(
        scalar("_cell_length_a", 1.0),
        scalar("_cell_length_b", 1.0),
        scalar("_cell_length_c", 1.0),
        scalar("_cell_angle_alpha", 90.0),
        scalar("_cell_angle_beta", 90.0),
        scalar("_cell_angle_gamma", 90.0),
    )
    frac = np.asarray([[float(row[index]) for index in x_indices] for row in carbon_rows], dtype=float)
    cart = np.asarray(
        [[p.x, p.y, p.z] for p in (cell.orthogonalize(gemmi.Fractional(*f)) for f in frac)],
        dtype=float,
    )
    center = cart.mean(axis=0)
    existing_labels = {str(row[label_index]) for row in rows}
    h_rows: list[list[str]] = []
    for i, (carbon_row, point) in enumerate(zip(carbon_rows, cart), start=1):
        direction = point - center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            continue
        h_cart = point + direction / norm * 1.083
        h_frac = cell.fractionalize(gemmi.Position(*map(float, h_cart)))
        label = f"H{i}"
        while label in existing_labels:
            label = f"H{i}_{len(h_rows) + 1}"
        existing_labels.add(label)
        h_row = ["?"] * len(tags)
        h_row[label_index] = label
        h_row[symbol_index] = "H"
        for index, value in zip(x_indices, (h_frac.x, h_frac.y, h_frac.z)):
            h_row[index] = f"{float(value):.6f}"
        for tag, value in (
            ("_atom_site_occupancy", "1.0000"),
            ("_atom_site_U_iso_or_equiv", "?"),
            ("_atom_site_thermal_displace_type", "?"),
            ("_atom_site_disorder_assembly", "."),
            ("_atom_site_disorder_group", "."),
        ):
            if tag in tags:
                h_row[tags.index(tag)] = value
        h_rows.append(h_row)

    for row in h_rows:
        atom_loop.add_row(row)

    bond_loop = _loop_for(block, "_geom_bond_atom_site_label_1")
    if bond_loop is None:
        bond_loop = block.init_loop(
            "",
            [
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_distance",
                "_ccdc_geom_bond_type",
            ],
        )
    bond_tags = list(bond_loop.tags)
    required_bond = {"_geom_bond_atom_site_label_1", "_geom_bond_atom_site_label_2", "_geom_bond_distance"}
    if not required_bond.issubset(bond_tags):
        return len(h_rows)
    index_1 = bond_tags.index("_geom_bond_atom_site_label_1")
    index_2 = bond_tags.index("_geom_bond_atom_site_label_2")
    distance_index = bond_tags.index("_geom_bond_distance")
    type_index = bond_tags.index("_ccdc_geom_bond_type") if "_ccdc_geom_bond_type" in bond_tags else None
    for carbon_row, h_row in zip(carbon_rows, h_rows):
        bond = ["?"] * len(bond_tags)
        bond[index_1] = str(carbon_row[label_index])
        bond[index_2] = str(h_row[label_index])
        bond[distance_index] = "1.083"
        if type_index is not None:
            bond[type_index] = "Single"
        bond_loop.add_row(bond)
    return len(h_rows)


def sanitize_cif(source_path: Path, output_path: Path, fixture_id: str) -> dict[str, Any]:
    """Write one minimal CIF and return machine-readable fixture metadata."""

    gemmi = _require_gemmi()
    document = gemmi.cif.read_file(str(source_path))
    source = document.sole_block()
    output = gemmi.cif.Document()
    block = output.add_new_block("forward_fixture")

    # Keep numerical cell data exactly as exported, including any uncertainty
    # notation, while deliberately omitting names/formula/R factors/audit data.
    for tag in CELL_TAGS:
        value = _value(source, tag)
        if value is not None:
            block.set_pair(tag, value)

    hm = _value(source, "_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M")
    if hm is not None:
        block.set_pair("_space_group_name_H-M_alt", hm)
    it_number = _value(source, "_space_group_IT_number", "_symmetry_Int_Tables_number")
    if it_number is not None:
        block.set_pair("_space_group_IT_number", it_number)
    setting = _value(source, "_space_group_setting", "_symmetry_space_group_setting")
    if setting is not None:
        block.set_pair("_space_group_setting", setting)
    hall = _value(source, "_space_group_name_Hall")
    # Preserve a known Hall symbol.  If the CCDC API could not expose it, keep
    # an explicit '?' so consumers do not mistake a missing value for a
    # canonical setting.
    block.set_pair("_space_group_name_Hall", hall or "?")

    symmetry_loop = _loop_for(source, "_space_group_symop_operation_xyz")
    if symmetry_loop is None:
        symmetry_loop = _loop_for(source, "_symmetry_equiv_pos_as_xyz")
    n_symmetry_ops = 0
    if symmetry_loop is not None:
        source_tags = list(symmetry_loop.tags)
        op_tag = (
            "_space_group_symop_operation_xyz"
            if "_space_group_symop_operation_xyz" in source_tags
            else "_symmetry_equiv_pos_as_xyz"
        )
        n_symmetry_ops = _copy_loop(block, symmetry_loop, [op_tag])

    atom_loop = _loop_for(source, "_atom_site_label")
    if atom_loop is None:
        raise ValueError(f"{source_path}: missing _atom_site_label loop")
    atom_source_tags = list(atom_loop.tags)
    atom_tags = [tag for tag in ATOM_TAGS if tag in atom_source_tags]
    required = {
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    }
    if not required.issubset(atom_tags):
        missing = ", ".join(sorted(required.difference(atom_tags)))
        raise ValueError(f"{source_path}: atom loop missing {missing}")
    n_atom_sites, disorder_group_mapping = _copy_atom_loop(block, atom_loop, atom_tags)

    # Preserve full anisotropic U data when a source exporter supplied it.
    aniso_loop = _loop_for(source, "_atom_site_aniso_label")
    n_aniso_sites = 0
    if aniso_loop is not None:
        aniso_tags = [tag for tag in list(aniso_loop.tags) if tag.startswith("_atom_site_aniso_")]
        if "_atom_site_aniso_label" in aniso_tags:
            n_aniso_sites = _copy_loop(block, aniso_loop, aniso_tags)

    bond_loop = _loop_for(source, "_geom_bond_atom_site_label_1")
    n_bonds = 0
    if bond_loop is not None:
        bond_source_tags = list(bond_loop.tags)
        bond_tags = _canonical_bond_tags(bond_source_tags)
        if {"_geom_bond_atom_site_label_1", "_geom_bond_atom_site_label_2"}.issubset(
            {tag if tag != "_geom_bond_type" else "_geom_bond_type" for tag in bond_tags}
        ):
            source_bond_tags = list(bond_tags)
            target_bond_tags = list(bond_tags)
            if "_geom_bond_type" in source_bond_tags and "_geom_bond_type" not in bond_source_tags:
                source_bond_tags[source_bond_tags.index("_geom_bond_type")] = "_ccdc_geom_bond_type"
            n_bonds = _copy_loop(block, bond_loop, source_bond_tags, target_bond_tags)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_file(str(output_path))
    payload = output_path.read_bytes()

    atom_values = list(atom_loop.values)
    width = atom_loop.width()
    occ_index = atom_source_tags.index("_atom_site_occupancy") if "_atom_site_occupancy" in atom_source_tags else None
    group_index = atom_source_tags.index("_atom_site_disorder_group") if "_atom_site_disorder_group" in atom_source_tags else None
    assembly_index = atom_source_tags.index("_atom_site_disorder_assembly") if "_atom_site_disorder_assembly" in atom_source_tags else None
    partial = 0
    disorder_sites = 0
    for row_start in range(0, len(atom_values), width):
        if occ_index is not None and (_float_value(str(atom_values[row_start + occ_index])) or 1.0) < 0.999:
            partial += 1
        if group_index is not None and str(atom_values[row_start + group_index]) not in ("", ".", "?"):
            disorder_sites += 1
        elif assembly_index is not None and str(atom_values[row_start + assembly_index]) not in ("", ".", "?"):
            disorder_sites += 1

    return {
        "fixture_id": fixture_id,
        "source_refcode": fixture_id,
        "file": output_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "space_group": hm.strip("'\" ") if hm is not None else None,
        "space_group_it_number": int(_float_value(it_number)) if _float_value(it_number) is not None else None,
        "space_group_setting": int(_float_value(setting)) if _float_value(setting) is not None else None,
        "space_group_hall": (
            hall.strip("'\" ")
            if hall is not None and hall.strip("'\" ") not in ("?", ".", "")
            else None
        ),
        "n_symmetry_operations": n_symmetry_ops,
        "n_atom_sites": n_atom_sites,
        "n_partial_occupancy_sites": partial,
        "n_disorder_sites": disorder_sites,
        "disorder_group_mapping": disorder_group_mapping,
        "n_anisotropic_sites": n_aniso_sites,
        "n_bonds": n_bonds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="CIF directory from the CCDC exporter")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for sanitized CIF fixtures")
    parser.add_argument("--refcodes", nargs="*", default=list(DEFAULT_FIXTURES))
    parser.add_argument("--release", default="CSD 6.01 Updates (September 2026)")
    parser.add_argument("--api-version", default="3.7.2")
    parser.add_argument("--database-name", default="as601be_CIP")
    parser.add_argument("--retrieved-utc", default=None, help="UTC date/time; defaults to now")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    refcodes = [code.strip().upper() for code in args.refcodes if code.strip()]
    if not refcodes:
        parser.error("provide at least one refcode")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for refcode in refcodes:
        source = args.input_dir / f"{refcode}.cif"
        if not source.exists():
            parser.error(f"missing source CIF: {source}")
        output = args.output_dir / f"{refcode}.cif"
        record = sanitize_cif(source, output, refcode)
        if refcode == "BENZEN02":
            record["notes"] = [
                "The licensed CCDC record exposed six hydrogen sites without fractional coordinates; the sanitized fixture retains the six carbon reference sites.",
                "Forward-construction tests must supply an explicit hydrogen-placement policy rather than infer missing coordinates.",
            ]
        records.append(record)

    manifest = {
        "schema": "molcrys-kit.forward-fixtures.v1",
        "source": {
            "database": "Cambridge Structural Database (CSD)",
            "database_name": args.database_name,
            "release": args.release,
            "api_version": args.api_version,
            "retrieved_utc": args.retrieved_utc or datetime.now(timezone.utc).isoformat(),
            "paths": None,
        },
        "fixtures": records,
        "notes": [
            "CIFs are minimal derived fixtures; CSD audit/name/R-factor fields and machine paths were removed.",
            "Coordinates remain subject to the applicable CCDC licence; retain the source refcodes for provenance.",
        ],
    }
    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} sanitized fixtures and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
