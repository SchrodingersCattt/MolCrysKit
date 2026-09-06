"""
Full regression: run disorder resolution on ALL example CIF files.
Report atom counts, bond counts, molecule formulas, and any problems.
"""
import os, sys, glob
import numpy as np
from collections import Counter, defaultdict

from molcrys_kit.io.cif import scan_cif_disorder
from molcrys_kit.analysis.disorder.process import generate_ordered_replicas_from_disordered_sites

# Known expected results from previous working state
KNOWN_GOOD = {
    "NatComm-1": 60,
    "PAP-HM4": 176,
    "DAP-4": 336,
}

cif_files = sorted(glob.glob("examples/*.cif"))

results_summary = []

for path in cif_files:
    name = os.path.splitext(os.path.basename(path))[0]
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    try:
        info = scan_cif_disorder(path)
        n_expanded = len(info.labels)
        species_count = Counter(info.symbols)
        dg_count = Counter(info.disorder_groups)
        
        has_disorder = any(dg != 0 for dg in info.disorder_groups) or \
                       any(0 < occ < 1.0 for occ in info.occupancies)
        
        print(f"  Expanded: {n_expanded} atoms, species={dict(species_count)}")
        print(f"  DG distribution: {dict(dg_count)}")
        print(f"  Has disorder: {has_disorder}")
        
        if not has_disorder:
            print(f"  SKIP (no disorder)")
            results_summary.append((name, n_expanded, n_expanded, "no-disorder", 0))
            continue
        
        results = generate_ordered_replicas_from_disordered_sites(path, generate_count=1, method='optimal')
        crystal = results[0]
        n_atoms = crystal.get_total_nodes()
        n_bonds = crystal.get_total_edges()
        
        expected = KNOWN_GOOD.get(name, None)
        if expected:
            status = "✓" if n_atoms == expected else f"✗ (expected {expected})"
        else:
            status = ""
        
        print(f"  Resolved: {n_atoms} atoms, {n_bonds} bonds {status}")
        
        # Check molecules
        molecules = crystal.molecules
        MAX_COORD = {'H': 1, 'C': 4, 'N': 4, 'O': 3, 'S': 6, 'Cl': 4, 'Cd': 8, 'P': 4, 'Zn': 6}
        
        total_problematic = 0
        mol_formulas = []
        for mol in molecules:
            mol_symbols = mol.get_chemical_symbols()
            mol_species = Counter(mol_symbols)
            formula = "".join(f"{e}{mol_species[e]}" for e in sorted(mol_species))
            mol_formulas.append(formula)
            
            g = mol.graph
            for node in g.nodes:
                elem = mol_symbols[node] if node < len(mol_symbols) else '?'
                degree = g.degree(node)
                max_c = MAX_COORD.get(elem, 8)
                if degree > max_c:
                    total_problematic += 1
                elif degree == 0 and elem not in ('Cl', 'Cd', 'Zn'):
                    total_problematic += 1
        
        formula_counts = Counter(mol_formulas)
        print(f"  Molecules: {len(molecules)}")
        for formula, count in formula_counts.most_common():
            print(f"    {count}× {formula}")
        
        if total_problematic > 0:
            print(f"  ⚠ {total_problematic} problematic atoms")
        else:
            print(f"  ✓ All coordination OK")
        
        results_summary.append((name, n_expanded, n_atoms, "OK" if total_problematic == 0 else f"{total_problematic} problems", total_problematic))
        
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results_summary.append((name, 0, 0, f"ERROR: {e}", -1))

print(f"\n\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"{'Name':<45} {'Expanded':>8} {'Resolved':>8} {'Status':<20}")
print(f"{'-'*85}")
for name, n_exp, n_res, status, _ in results_summary:
    print(f"{name:<45} {n_exp:>8} {n_res:>8} {status:<20}")
