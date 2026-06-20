#!/usr/bin/env python
"""
Demo: Full disorder-aware workflow from CSD to ordered periodic structures.

This script demonstrates a complete pipeline for extracting crystal structures
from the Cambridge Structural Database (CSD) with full disorder/occupancy
information, then using MolCrysKit to identify molecules and enumerate explicit
ordered configurations suitable for DFT calculations.

Pipeline:
    1. CCDC Python API: read crystal → extract all atom sites with occupancy,
       anisotropic ADP, disorder_group, disorder_assembly, and bond connectivity.
    2. Write a complete CIF (the CCDC API's built-in writers omit occupancy!).
    3. MolCrysKit scan_cif_disorder: parse the CIF with full disorder metadata.
    4. MolCrysKit identify_molecules: split unit cell into molecular units.
    5. Enumerate disorder states: generate all ordered configurations of the cell.
    6. Output: one CIF per disorder state, each a complete periodic structure.

Requirements:
    - CCDC Python API (CSD license required)
    - MolCrysKit (pip install molcrys-kit)
    - ASE, pymatgen, numpy

Usage:
    # From the CCDC Python environment (which has both ccdc and molcrys_kit):
    python demo_csd_disorder_workflow.py --refcodes ABACIR ABABUB --output-dir ./demo_output

    # Or with a pre-exported full CIF (no CCDC needed for step 3-6):
    python demo_csd_disorder_workflow.py --cifs ./ABACIR.cif ./ABABUB.cif --output-dir ./demo_output

"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional

import re
import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from pymatgen.core.lattice import Lattice

from molcrys_kit.io.cif import scan_cif_disorder, identify_molecules
from molcrys_kit.constants.config import (
    KEY_OCCUPANCY,
    KEY_DISORDER_GROUP,
    KEY_ASSEMBLY,
    KEY_LABEL,
)


# =============================================================================
# Step 1-2: Export full CIF from CSD (requires CCDC Python API)
# =============================================================================

def export_full_cif_from_csd(refcode: str, output_path: Path) -> dict[str, Any]:
    """Export a CSD structure to CIF with full occupancy/disorder/ADP information.

    The CCDC Python API's built-in CIF writers (EntryWriter, CrystalWriter,
    Entry.to_string, Crystal.to_string) all omit _atom_site_occupancy,
    _atom_site_disorder_group, and _atom_site_disorder_assembly.

    This function extracts the information from the API objects and writes
    a standards-compliant CIF that preserves all disorder metadata.
    """
    from ccdc.io import EntryReader  # type: ignore

    with EntryReader("CSD") as reader:
        entry = reader.entry(refcode)
        crystal = reader.crystal(refcode)

    # Get disordered molecule (includes all sites, major + minor)
    mol = crystal.disordered_molecule or crystal.molecule

    # Build disorder group/assembly lookup from crystal.disorder API
    disorder_map: dict[str, tuple[int, str]] = {}
    disorder = getattr(crystal, "disorder", None)
    if disorder is not None:
        try:
            for assembly in disorder.assemblies:
                asm_id = str(getattr(assembly, "id", ""))
                for group in assembly.groups:
                    g_id = int(getattr(group, "id", 0))
                    for atom in group.atoms:
                        disorder_map[str(atom.label)] = (g_id, asm_id)
        except Exception:
            pass

    # Extract atom data
    atom_data = []
    for atom in mol.atoms:
        label = str(atom.label) if atom.label else "?"
        symbol = str(atom.atomic_symbol) if atom.atomic_symbol else "?"
        frac = atom.fractional_coordinates
        fx = float(frac.x) if frac else None
        fy = float(frac.y) if frac else None
        fz = float(frac.z) if frac else None

        occ = 1.0
        try:
            if atom.occupancy is not None:
                occ = float(atom.occupancy)
        except Exception:
            pass

        u_iso = None
        adp_type = "?"
        aniso = None
        try:
            dp = atom.displacement_parameters
            if dp is not None:
                u_iso = float(dp.isotropic_equivalent)
                if dp.type == "Anisotropic":
                    adp_type = "Uani"
                    vals = dp.values
                    esd = dp.uncertainties
                    aniso = {
                        "u11": vals[0][0], "u22": vals[1][1], "u33": vals[2][2],
                        "u12": vals[0][1], "u13": vals[0][2], "u23": vals[1][2],
                        "e11": esd[0][0], "e22": esd[1][1], "e33": esd[2][2],
                        "e12": esd[0][1], "e13": esd[0][2], "e23": esd[1][2],
                    }
                else:
                    adp_type = "Uiso"
        except Exception:
            pass

        dg, da = disorder_map.get(label, (None, None))
        if dg is None and label.endswith("?"):
            dg, da = disorder_map.get(label.rstrip("?"), (None, None))

        atom_data.append({
            "label": label, "symbol": symbol,
            "fx": fx, "fy": fy, "fz": fz,
            "occ": occ, "u_iso": u_iso, "adp_type": adp_type,
            "dg": dg, "da": da, "aniso": aniso,
        })

    # Extract bonds
    bonds = []
    try:
        for b in mol.bonds:
            a1, a2 = b.atoms
            bonds.append((str(a1.label), str(a2.label), float(b.length), str(b.bond_type)))
    except Exception:
        pass

    # Write CIF
    a, b_len, c = crystal.cell_lengths
    alpha, beta, gamma = crystal.cell_angles
    sg = crystal.spacegroup_symbol or "?"
    z_val = crystal.z_value

    def fv(v, p=4):
        return f"{v:.{p}f}" if v is not None else "?"

    def fv_esd(v, e, p=4):
        s = f"{v:.{p}f}"
        return f"{s}({e})" if e and e > 0 else s

    has_disorder = any(d["dg"] is not None for d in atom_data)
    L = []
    L.append(f"data_{refcode}")
    L.append(f"_audit_creation_method            'CCDC API + MolCrysKit demo'")
    L.append(f"_chemical_name_common             '{(entry.chemical_name or '?').replace(chr(39), chr(39)+chr(39))}'")
    L.append(f"_chemical_formula_sum             '{entry.formula or '?'}'")
    rf = entry.r_factor
    L.append(f"_refine_ls_R_factor_all           {fv(rf, 2) if rf else '?'}")
    L.append(f"_cell_length_a                    {fv(a)}")
    L.append(f"_cell_length_b                    {fv(b_len)}")
    L.append(f"_cell_length_c                    {fv(c)}")
    L.append(f"_cell_angle_alpha                 {fv(alpha, 2)}")
    L.append(f"_cell_angle_beta                  {fv(beta, 2)}")
    L.append(f"_cell_angle_gamma                 {fv(gamma, 2)}")
    L.append(f"_cell_volume                      {fv(crystal.cell_volume, 2)}")
    L.append(f"_cell_formula_units_Z             {int(z_val) if z_val else '?'}")
    L.append(f"_symmetry_space_group_name_H-M    '{sg}'")
    L.append("")

    symops = crystal.symmetry_operators
    if symops:
        L.append("loop_")
        L.append("_symmetry_equiv_pos_as_xyz")
        for op in symops:
            L.append(f"  '{op}'")
        L.append("")

    # Atom sites
    L.append("loop_")
    L.append("_atom_site_label")
    L.append("_atom_site_type_symbol")
    L.append("_atom_site_fract_x")
    L.append("_atom_site_fract_y")
    L.append("_atom_site_fract_z")
    L.append("_atom_site_occupancy")
    L.append("_atom_site_U_iso_or_equiv")
    L.append("_atom_site_thermal_displace_type")
    if has_disorder:
        L.append("_atom_site_disorder_assembly")
        L.append("_atom_site_disorder_group")

    for d in atom_data:
        if d["fx"] is None:
            continue
        parts = [f"{d['label']:<8s}", f"{d['symbol']:<4s}",
                 f"{d['fx']:12.6f}", f"{d['fy']:12.6f}", f"{d['fz']:12.6f}",
                 f"{d['occ']:8.4f}", f"{fv(d['u_iso']):>10s}", f"{d['adp_type']:>5s}"]
        if has_disorder:
            parts.append(f"{str(d['da']) if d['da'] is not None else '.':>4s}")
            parts.append(f"{str(d['dg']) if d['dg'] is not None else '.':>4s}")
        L.append("  " + " ".join(parts))
    L.append("")

    # Aniso ADP
    aniso_atoms = [d for d in atom_data if d["aniso"]]
    if aniso_atoms:
        L.append("loop_")
        L.append("_atom_site_aniso_label")
        L.append("_atom_site_aniso_U_11")
        L.append("_atom_site_aniso_U_22")
        L.append("_atom_site_aniso_U_33")
        L.append("_atom_site_aniso_U_12")
        L.append("_atom_site_aniso_U_13")
        L.append("_atom_site_aniso_U_23")
        for d in aniso_atoms:
            u = d["aniso"]
            L.append(f"  {d['label']:<8s}"
                     f" {fv_esd(u['u11'], u['e11']):>12s}"
                     f" {fv_esd(u['u22'], u['e22']):>12s}"
                     f" {fv_esd(u['u33'], u['e33']):>12s}"
                     f" {fv_esd(u['u12'], u['e12']):>12s}"
                     f" {fv_esd(u['u13'], u['e13']):>12s}"
                     f" {fv_esd(u['u23'], u['e23']):>12s}")
        L.append("")

    # Bonds
    if bonds:
        L.append("loop_")
        L.append("_geom_bond_atom_site_label_1")
        L.append("_geom_bond_atom_site_label_2")
        L.append("_geom_bond_distance")
        L.append("_ccdc_geom_bond_type")
        for a1, a2, dist, btype in bonds:
            L.append(f"  {a1:<8s} {a2:<8s} {dist:8.4f} {btype}")
        L.append("")

    L.append("#END")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")

    return {
        "refcode": refcode,
        "n_atoms": len(atom_data),
        "n_partial_occ": sum(1 for d in atom_data if d["occ"] < 0.999),
        "n_aniso": len(aniso_atoms),
        "n_bonds": len(bonds),
        "output": str(output_path),
    }


# =============================================================================
# Step 3-4: Load CIF with MolCrysKit (no CCDC needed from here)
# =============================================================================

def load_full_cell(cif_path: Path) -> Atoms:
    """Load a full CIF into an ASE Atoms with disorder metadata preserved."""
    info = scan_cif_disorder(str(cif_path))
    n = len(info.labels)

    from pymatgen.io.cif import CifParser
    parser = CifParser(str(cif_path), occupancy_tolerance=100)
    cif_data = parser.as_dict()
    block = list(cif_data.values())[0]

    def _num(key, default):
        v = block.get(key, default)
        if isinstance(v, str):
            v = re.sub(r"\([^)]*\)", "", v)
            return float(v)
        return float(v)

    lat = Lattice.from_parameters(
        _num("_cell_length_a", 10), _num("_cell_length_b", 10), _num("_cell_length_c", 10),
        _num("_cell_angle_alpha", 90), _num("_cell_angle_beta", 90), _num("_cell_angle_gamma", 90),
    )
    cart = lat.get_cartesian_coords(info.frac_coords)

    atoms = Atoms(symbols=info.symbols[:n], positions=cart, cell=lat.matrix, pbc=True)
    atoms.set_array(KEY_OCCUPANCY, np.array(info.occupancies[:n]))
    atoms.set_array(KEY_DISORDER_GROUP, np.array(info.disorder_groups[:n], dtype=int))
    atoms.set_array(KEY_ASSEMBLY, np.array(info.assemblies[:n]))
    atoms.set_array(KEY_LABEL, np.array(info.labels[:n]))
    return atoms


# =============================================================================
# Step 5: Enumerate disorder states at cell level
# =============================================================================

def enumerate_disorder_states(atoms: Atoms, mode: str = "enumerate") -> list[dict]:
    """Enumerate cell-level disorder states.

    Args:
        atoms: Full unit cell with disorder metadata.
        mode: 'major', 'minor', 'enumerate', or 'all'.

    Returns:
        List of dicts with 'state_id', 'description', 'keep_mask' (bool array).
    """
    n = len(atoms)
    occs = atoms.arrays.get(KEY_OCCUPANCY, np.ones(n))
    dg = atoms.arrays.get(KEY_DISORDER_GROUP, np.zeros(n, dtype=int))
    asm = atoms.arrays.get(KEY_ASSEMBLY, np.array([""] * n))

    # Find disordered atoms → group by assembly
    assembly_groups: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for j in range(n):
        a_id = str(asm[j]).strip()
        g_id = int(dg[j])
        is_partial = occs[j] < 0.999
        if a_id and (g_id != 0 or is_partial):
            assembly_groups[a_id][g_id].append(j)
        elif not a_id and (g_id != 0 or is_partial):
            assembly_groups["_default"][g_id].append(j)

    if not assembly_groups:
        return [{"state_id": "ordered", "description": "no disorder", "keep_mask": np.ones(n, dtype=bool)}]

    if mode == "all":
        return [{"state_id": "all", "description": "unresolved", "keep_mask": np.ones(n, dtype=bool)}]

    # Mean occupancy per group
    group_occ: dict[str, dict[int, float]] = {}
    for a_id, groups in assembly_groups.items():
        group_occ[a_id] = {g: float(np.mean([occs[j] for j in idx])) for g, idx in groups.items()}

    asm_ids = sorted(assembly_groups.keys())

    if mode in ("major", "minor"):
        sel_fn = max if mode == "major" else min
        selections = {a: sel_fn(group_occ[a], key=group_occ[a].get) for a in asm_ids}
        mask = np.ones(n, dtype=bool)
        for a_id, groups in assembly_groups.items():
            for g_id, indices in groups.items():
                if g_id != selections[a_id]:
                    mask[indices] = False
        desc = ", ".join(f"{a}:g{selections[a]}(occ={group_occ[a][selections[a]]:.3f})" for a in asm_ids)
        return [{"state_id": mode, "description": f"{mode} site ({desc})", "keep_mask": mask}]

    # enumerate: all combinations
    group_choices = [sorted(assembly_groups[a].keys()) for a in asm_ids]
    states = []
    for combo in product(*group_choices):
        state_id = "_".join(f"{a}{g}" for a, g in zip(asm_ids, combo))
        mask = np.ones(n, dtype=bool)
        for a_id, chosen_g in zip(asm_ids, combo):
            for g_id, indices in assembly_groups[a_id].items():
                if g_id != chosen_g:
                    mask[indices] = False
        desc = ", ".join(f"{a}:g{g}(occ={group_occ[a][g]:.3f})" for a, g in zip(asm_ids, combo))
        states.append({"state_id": state_id, "description": f"({desc})", "keep_mask": mask})
    return states


# =============================================================================
# Step 6: Write ordered cells and verify
# =============================================================================

def write_ordered_cell(atoms: Atoms, state: dict, output_path: Path) -> dict:
    """Apply disorder state and write a complete periodic CIF."""
    mask = state["keep_mask"]
    kept = atoms[np.where(mask)[0]]
    kept.set_cell(atoms.get_cell())
    kept.set_pbc(True)

    ase_write(str(output_path), kept, format="cif")

    # Verify: identify molecules in the output
    mols = identify_molecules(kept)
    formulas = defaultdict(int)
    for m in mols:
        formulas[m.get_chemical_formula()] += 1

    return {
        "state_id": state["state_id"],
        "description": state["description"],
        "n_atoms": int(mask.sum()),
        "molecules": dict(formulas),
        "output": str(output_path),
    }


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--refcodes", nargs="*", default=[],
                        help="CSD refcodes to export (requires CCDC Python API)")
    parser.add_argument("--cifs", nargs="*", default=[],
                        help="Pre-exported full CIF files (no CCDC needed)")
    parser.add_argument("--mode", choices=["major", "minor", "enumerate", "all"], default="enumerate")
    parser.add_argument("--output-dir", type=Path, default=Path("./demo_output"))
    args = parser.parse_args(argv or sys.argv[1:])

    if not args.refcodes and not args.cifs:
        parser.error("Provide --refcodes and/or --cifs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cif_paths: list[Path] = []

    # Step 1-2: Export from CSD if refcodes given
    if args.refcodes:
        print("=" * 70)
        print("Step 1-2: Exporting full CIFs from CSD (CCDC Python API)")
        print("=" * 70)
        for refcode in args.refcodes:
            refcode = refcode.strip().upper()
            out = args.output_dir / "full_cifs" / f"{refcode}.cif"
            print(f"\n  Exporting {refcode} ...")
            try:
                result = export_full_cif_from_csd(refcode, out)
                print(f"    ✓ {result['n_atoms']} atoms, {result['n_partial_occ']} partial-occ, "
                      f"{result['n_aniso']} aniso, {result['n_bonds']} bonds")
                print(f"    → {out}")
                cif_paths.append(out)
            except Exception as e:
                print(f"    ✗ FAILED: {e}")

    # Add pre-existing CIFs
    for p in (args.cifs or []):
        cif_paths.append(Path(p))

    # Step 3-6: Process each CIF
    all_results = []
    for cif_path in cif_paths:
        stem = cif_path.stem
        print(f"\n{'=' * 70}")
        print(f"Processing: {cif_path}")
        print("=" * 70)

        # Step 3: Load with MolCrysKit
        print("\n  Step 3: scan_cif_disorder (full disorder metadata)")
        atoms = load_full_cell(cif_path)
        n = len(atoms)
        occs = atoms.arrays[KEY_OCCUPANCY]
        n_partial = sum(1 for o in occs if o < 0.999)
        print(f"    Cell: {n} atoms, {n_partial} with partial occupancy")

        # Step 4: Identify molecules
        print("\n  Step 4: identify_molecules")
        mols = identify_molecules(atoms)
        formula_counts = defaultdict(int)
        for mol in mols:
            na = len(mol)
            mol_occ = mol.arrays.get("occupancy", np.ones(na))
            mol_dg = mol.arrays.get("disorder_group", np.zeros(na, dtype=int))
            n_p = sum(1 for o in mol_occ if o < 0.999)
            formula_counts[mol.get_chemical_formula()] += 1
        for f, cnt in sorted(formula_counts.items()):
            print(f"    {f} × {cnt}")

        # Step 5: Enumerate
        print(f"\n  Step 5: Enumerate disorder states (mode={args.mode})")
        states = enumerate_disorder_states(atoms, args.mode)
        print(f"    {len(states)} state(s):")
        for st in states:
            n_kept = int(st["keep_mask"].sum())
            print(f"      {st['state_id']}: {st['description']} → {n_kept}/{n} atoms")

        # Step 6: Write
        print(f"\n  Step 6: Write ordered cells")
        out_dir = args.output_dir / "ordered_cells" / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        ref_results = []
        for st in states:
            out_path = out_dir / f"{stem}_{st['state_id']}.cif"
            result = write_ordered_cell(atoms, st, out_path)
            ref_results.append(result)
            mol_str = ", ".join(f"{f}×{c}" for f, c in sorted(result["molecules"].items()))
            print(f"      ✓ {out_path.name}: {result['n_atoms']} atoms | {mol_str}")

        all_results.append({"source": str(cif_path), "states": ref_results})

    # Summary
    summary_path = args.output_dir / "workflow_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'=' * 70}")
    print(f"Done. Summary: {summary_path}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
