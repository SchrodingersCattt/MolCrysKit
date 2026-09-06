"""
Quick regression: run disorder resolution on key CIF files with timeout.
"""
import os, sys, glob, signal
import numpy as np
from collections import Counter, defaultdict

from molcrys_kit.io.cif import scan_cif_disorder
from molcrys_kit.analysis.disorder.process import generate_ordered_replicas_from_disordered_sites

# Key test files - the three target + previously working ones
CIFS = [
    ("NatComm-1",        "examples/NatComm-1.cif",                          60),
    ("PAP-HM4",          "examples/PAP-HM4.cif",                           176),
    ("DAP-4",            "examples/DAP-4.cif",                             336),
    ("DAC-4",            "examples/DAC-4.cif",                              39),
    ("anhydrousCaffeine","examples/anhydrousCaffeine_CGD_2007_7_1406.cif", 480),
    ("anhydrousCaffeine2","examples/anhydrousCaffeine2_CGD_2007_7_1406.cif",144),
    ("ZIF-4",            "examples/ZIF-4.cif",                             368),
    ("TILPEN",           "examples/TILPEN.cif",                             84),
    ("1-HTP",            "examples/1-HTP.cif",                             102),
    ("MAF-4",            "examples/MAF-4.cif",                             369),
    ("DAN-2",            "examples/DAN-2.cif",                              35),
    ("PAP-H4",           "examples/PAP-H4.cif",                            656),
    ("368K",             "examples/368K.cif",                               88),
    ("DAI-X1",           "examples/DAI-X1.cif",                            112),
    ("ZIF-8",            "examples/ZIF-8.cif",                             316),
    ("PAP-M5",           "examples/PAP-M5.cif",                            296),
    ("DAP-O4",           "examples/DAP-O4.cif",                            344),
    ("PAP-4",            "examples/PAP-4.cif",                             304),
    ("DAI-4",            "examples/DAI-4.cif",                             336),
    ("DAP-7",            "examples/DAP-7.cif",                              88),
]

MAX_COORD = {'H': 1, 'C': 4, 'N': 4, 'O': 3, 'S': 6, 'Cl': 4, 'Cd': 8, 'P': 4, 'Zn': 6}

results_summary = []

for name, path, expected in CIFS:
    if not os.path.exists(path):
        print(f"SKIP {name} (not found)")
        continue
    
    print(f"\n--- {name} ---")
    
    try:
        info = scan_cif_disorder(path)
        n_expanded = len(info.labels)
        has_disorder = any(dg != 0 for dg in info.disorder_groups) or \
                       any(0 < occ < 1.0 for occ in info.occupancies)
        
        if not has_disorder:
            print(f"  No disorder, {n_expanded} atoms")
            results_summary.append((name, n_expanded, n_expanded, "no-disorder"))
            continue
        
        # Set alarm for timeout
        import threading
        result_holder = [None, None]  # [crystal, error]
        
        def solve():
            try:
                results = generate_ordered_replicas_from_disordered_sites(path, generate_count=1, method='optimal')
                result_holder[0] = results[0]
            except Exception as e:
                result_holder[1] = str(e)
        
        t = threading.Thread(target=solve)
        t.start()
        t.join(timeout=60)  # 60 second timeout
        
        if t.is_alive():
            print(f"  TIMEOUT (>60s)")
            results_summary.append((name, n_expanded, 0, "TIMEOUT"))
            continue
        
        if result_holder[1]:
            print(f"  ERROR: {result_holder[1]}")
            results_summary.append((name, n_expanded, 0, f"ERROR"))
            continue
        
        crystal = result_holder[0]
        n_atoms = crystal.get_total_nodes()
        n_bonds = crystal.get_total_edges()
        
        # Check coordination
        total_problematic = 0
        for mol in crystal.molecules:
            mol_symbols = mol.get_chemical_symbols()
            g = mol.graph
            for node in g.nodes:
                elem = mol_symbols[node] if node < len(mol_symbols) else '?'
                degree = g.degree(node)
                max_c = MAX_COORD.get(elem, 8)
                if degree > max_c:
                    total_problematic += 1
                elif degree == 0 and elem not in ('Cl', 'Cd', 'Zn'):
                    total_problematic += 1
        
        status_parts = []
        if expected:
            status_parts.append("✓" if n_atoms == expected else f"✗ expected={expected}")
        if total_problematic > 0:
            status_parts.append(f"⚠{total_problematic} overcord")
        else:
            status_parts.append("coord✓")
        
        status = " ".join(status_parts)
        print(f"  {n_atoms} atoms, {n_bonds} bonds, {len(crystal.molecules)} mols  {status}")
        results_summary.append((name, n_expanded, n_atoms, status))
        
    except Exception as e:
        print(f"  ERROR: {e}")
        results_summary.append((name, 0, 0, f"ERROR: {e}"))

print(f"\n\n{'='*75}")
print(f"  SUMMARY")
print(f"{'='*75}")
print(f"{'Name':<30} {'Expanded':>8} {'Resolved':>8} {'Status':<30}")
print(f"{'-'*75}")
for name, n_exp, n_res, status in results_summary:
    print(f"{name:<30} {n_exp:>8} {n_res:>8} {status:<30}")
