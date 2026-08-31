import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import gc
from ase.visualize.plot import plot_atoms
from ase.build import surface as ase_surface
from ase.io import write as ase_write
from pymatgen.core.structure import Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

# MolCrysKit imports
try:
    from molcrys_kit.io.cif import read_mol_crystal
    from molcrys_kit.structures import MolecularCrystal
    from molcrys_kit.operations import generate_topological_slab
except ImportError:
    raise ImportError("Please install molcrys_kit first or check your python path.")

OUTPUT_DIR = Path("benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = OUTPUT_DIR / "benchmark_data.json"

CIF_PATH = Path("benchmarks/slab_examples/beta-HMX_OCHTET12.cif")
MILLER = (2, 1, 0)
LAYERS = 3     # Target layers
VACUUM = 10.0  # Target vacuum in Angstrom

CUSTOM_COLORS = {
    'C': "#535353",
    'H': '#eeeeee',
    'N': "#6e89e1",
    'O': "#bf0603",
    'Cl': '#26c81b',
}

# Scaling Configurations
SCALES_CUBIC = [1, 2, 3, 4, 5, 6]
SCALES_LINEAR = [1, 2, 3, 4, 5, 6]
PMG_ATOM_LIMIT = 1000  # Strict cutoff to prevent freezing

# Cache for geometric quantities to avoid recomputation
GEOMETRIC_CACHE = {}

def time_function(func):
    """
    Times a function and returns (execution_time, result).
    Returns (None, None) if an exception occurs.
    """
    try:
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        return end - start, result
    except Exception:
        return None, None

def run_ase_slab_gen(atoms, miller, layers, vacuum):
    """
    Wraps ASE surface() with standardized output format.
    ASE's surface() puts the slab in the center with vacuum on both sides.
    This function adjusts the output so:
    - Slab is at the bottom (z ~ 1.0 Å)
    - Vacuum is on top
    - The C-lattice vector is exactly thickness + vacuum and orthogonal to AB plane
    """
    ase_slab = ase_surface(atoms, miller, layers=layers, vacuum=vacuum)
    
    positions = ase_slab.get_positions()
    cell = ase_slab.get_cell()
    
    z_coords = positions[:, 2]
    min_z = np.min(z_coords)
    max_z = np.max(z_coords)
    atomic_thickness = max_z - min_z
    
    new_c_vector = np.array([0, 0, atomic_thickness + vacuum])
    cell[2] = new_c_vector
    
    shift_z = -min_z + 1.0
    positions[:, 2] += shift_z
    
    ase_slab.set_positions(positions)
    ase_slab.set_cell(cell, scale_atoms=False)
    
    return ase_slab

def run_pymatgen_slab_gen(ase_structure, miller, layers, vacuum):
    """
    Wraps Pymatgen SlabGenerator with thickness/vacuum fixes.
    """
    # Create a hashable key for caching geometric quantities
    bulk_structure = AseAtomsAdaptor.get_structure(ase_structure)
    lattice_key = tuple(bulk_structure.lattice.matrix.flatten())
    miller_key = tuple(miller)
    
    cache_key = (lattice_key, miller_key)
    
    if cache_key not in GEOMETRIC_CACHE:
        recp_lat = bulk_structure.lattice.reciprocal_lattice
        d_hkl = 2 * np.pi / np.linalg.norm(recp_lat.get_cartesian_coords(miller))
        GEOMETRIC_CACHE[cache_key] = d_hkl
    
    d_hkl = GEOMETRIC_CACHE[cache_key]
    
    effective_layers = max(0.1, float(layers) - 0.5)
    slab_thickness = effective_layers * d_hkl

    slab_gen = SlabGenerator(
        bulk_structure,
        miller_index=miller,
        min_slab_size=slab_thickness,
        min_vacuum_size=vacuum, 
        in_unit_planes=False,
        center_slab=False,
        primitive=False
    )
    slabs = slab_gen.get_slabs(repair=True, tol=0.1, ftol=0.2, ztol=0.2)
    
    if not slabs:
        raise ValueError("No slabs generated")
    
    slab = slabs[0]

    cart_coords = slab.cart_coords
    z_coords = cart_coords[:, 2]
    real_atomic_thickness = np.ptp(z_coords)
    
    target_c_length = real_atomic_thickness + vacuum
    
    new_lattice_matrix = slab.lattice.matrix.copy()
    current_c_norm = np.linalg.norm(new_lattice_matrix[2])
    scale_factor = target_c_length / current_c_norm
    new_lattice_matrix[2] = new_lattice_matrix[2] * scale_factor
    
    slab.lattice = Lattice(new_lattice_matrix)
    return AseAtomsAdaptor.get_atoms(slab)

def run_benchmark():
    print(f"Loading Crystal from: {CIF_PATH}")
    if not CIF_PATH.exists():
        raise FileNotFoundError(f"File not found: {CIF_PATH}")

    crystal = read_mol_crystal(str(CIF_PATH))
    original_ase = crystal.to_ase()
    print(f"Base cell atoms: {len(original_ase)}")

    print("\n--- Running Visual Comparison (Base Cell) ---")
    fig1, axes = plt.subplots(1, 3, figsize=(12, 5))
    PLOT_REPLEAT = [1,1,1]
    
    # ASE
    try:
        ase_slab = run_ase_slab_gen(original_ase, MILLER, LAYERS, VACUUM)
        ase_slab = ase_slab.repeat(PLOT_REPLEAT)
        ase_output_path = OUTPUT_DIR / f"ase_slab.cif"
        ase_write(ase_output_path, ase_slab)
        print(f"Saved ASE slab to {ase_output_path}")
        colors = [CUSTOM_COLORS.get(symbol, 'blue') for symbol in ase_slab.get_chemical_symbols()]
        plot_atoms(ase_slab, axes[0], rotation="90x,0y,90z", colors=colors)
        axes[0].set_title(f"ASE (Geometric)\nFragmented Molecules", fontsize=12, color='red')
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error: {e}", ha='center')
    axes[0].axis('off')

    # Pymatgen
    try:
        pmg_slab_ase = run_pymatgen_slab_gen(original_ase, MILLER, LAYERS, VACUUM)
        pmg_slab_ase = pmg_slab_ase.repeat(PLOT_REPLEAT)
        pmg_output_path = OUTPUT_DIR / f"pymatgen_slab.cif"
        ase_write(pmg_output_path, pmg_slab_ase)
        print(f"Saved Pymatgen slab to {pmg_output_path}")
        
        colors = [CUSTOM_COLORS.get(symbol, 'blue') for symbol in pmg_slab_ase.get_chemical_symbols()]
        plot_atoms(pmg_slab_ase, axes[1], rotation="90x,0y,90z", colors=colors)
        axes[1].set_title(f"Pymatgen (Geometric+Repair)\nRepaired", fontsize=12, color='orange')
    except Exception as e:
        import traceback
        axes[1].text(0.5, 0.5, f"Error: {e}, \nTraceback: {traceback.format_exc()}", ha='center')
    axes[1].axis('off')

    # MolCrysKit
    try:
        molck_slab_obj = generate_topological_slab(crystal, MILLER, layers=LAYERS, vacuum=VACUUM)
        molck_slab_ase = molck_slab_obj.to_ase()
        molck_slab_ase = molck_slab_ase.repeat(PLOT_REPLEAT)
        mck_output_path = OUTPUT_DIR / f"molcrys_kit_slab.cif"
        ase_write(mck_output_path, molck_slab_ase)
        print(f"Saved MolCrysKit slab to {mck_output_path}")

        colors = [CUSTOM_COLORS.get(symbol, 'blue') for symbol in molck_slab_ase.get_chemical_symbols()]
        plot_atoms(molck_slab_ase, axes[2], rotation="90x,0y,90z", colors=colors)
        axes[2].set_title(f"MolCrysKit (Topological)\nPreserved & Fast", fontsize=12, fontweight='bold', color='green')
    except Exception as e:
        axes[2].text(0.5, 0.5, f"Error: {e}", ha='center')
    axes[2].axis('off')

    fig1.suptitle(f"Surface Generation Quality: {MILLER}", fontsize=14)
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "slab_quality_comparison.png", dpi=300)
    print(f"Saved comparison image to {OUTPUT_DIR / 'slab_quality_comparison.png'}")

    print("\n--- Running Performance Benchmark ---")

    # WARM-UP
    print("Warming up JIT/Caches...")
    try:
        _ = generate_topological_slab(crystal, MILLER, layers=LAYERS, vacuum=VACUUM)
        _ = run_ase_slab_gen(original_ase, MILLER, LAYERS, VACUUM)
    except: pass

    data_store = {
        "cubic": {"atoms": [], "ase": [], "mck": []},
        "linear": {"atoms": [], "pmg": []}
    }

#     NUM_RUNS = 3

#     # A. Run Cubic Scaling (ASE & MolCrysKit)
#     print(f"\n[A] Cubic Scaling (n,n,n) for High-Performance Tools - Running {NUM_RUNS} times per scale")
#     for n in SCALES_CUBIC:
#         super_ase = original_ase * (n, n, n)
#         natoms = len(super_ase)
#         data_store["cubic"]["atoms"].append(natoms)
        
#         print(f"  > Scale {n}x{n}x{n} ({natoms} atoms)...")

#         ase_times = []
#         mck_times = []

#         for run in range(NUM_RUNS):
#             print(f"    Run {run+1}/{NUM_RUNS}...")

#             # ASE - Use a fresh copy of atoms for each run
#             fresh_ase = super_ase.copy()
#             ase_time, ase_result = time_function(lambda: run_ase_slab_gen(fresh_ase, MILLER, LAYERS, VACUUM))
            
#             if ase_result is not None:
#                 ase_times.append(ase_time)
#                 print(f"      ASE time: {ase_time:.4f}s for {len(super_ase)} atoms")
                
#                 # Explicitly delete large objects and force garbage collection
#                 del ase_result
#                 gc.collect()
#             else:
#                 ase_times.append(None)
#                 print(f"      ASE Error: Operation failed")

#             # MolCrysKit - Convert ASE atoms to MolecularCrystal inside the timing loop, 
#             # but do not include the conversion in the timed region
#             fresh_ase_for_mck = super_ase.copy()
#             c_super = MolecularCrystal.from_ase(fresh_ase_for_mck)
#             mck_time, mck_result = time_function(lambda: generate_topological_slab(c_super, MILLER, layers=LAYERS, vacuum=VACUUM))
            
#             if mck_result is not None:
#                 mck_times.append(mck_time)
#                 print(f"      MolCrysKit time: {mck_time:.4f}s for {len(super_ase)} atoms")
                
#                 # Explicitly delete large objects and force garbage collection
#                 del mck_result
#                 gc.collect()
#             else:
#                 print(f"      MCK Error: Operation failed")
#                 mck_times.append(None)

#         data_store["cubic"]["ase"].append(ase_times)
#         data_store["cubic"]["mck"].append(mck_times)

#     # B. Run Linear Scaling (Pymatgen)
#     print(f"\n[B] Linear Scaling (n,1,1) for Pymatgen - Running {NUM_RUNS} times per scale")
#     for n in SCALES_LINEAR:
#         super_ase = original_ase * (n, 1, 1)
#         natoms = len(super_ase)
#         data_store["linear"]["atoms"].append(natoms)
        
#         if natoms > PMG_ATOM_LIMIT:
#             data_store["linear"]["pmg"].append([None] * NUM_RUNS)
#             print(f"  > Scale {n}x1x1 ({natoms} atoms): Skipped (too large for Pymatgen)")
#             continue
            
#         print(f"  > Scale {n}x1x1 ({natoms} atoms)...")
        
#         pmg_times = []
#         for run in range(NUM_RUNS):
#             print(f"    Run {run+1}/{NUM_RUNS}...")
            
#             # Use a fresh copy of atoms for each run
#             fresh_ase = super_ase.copy()
#             pmg_time, pmg_result = time_function(lambda: run_pymatgen_slab_gen(fresh_ase, MILLER, LAYERS, VACUUM))
            
#             if pmg_result is not None:
#                 pmg_times.append(pmg_time)
#                 print(f"      Pymatgen time: {pmg_time:.4f}s for {len(super_ase)} atoms")
                
#                 # Explicitly delete large objects and force garbage collection
#                 del pmg_result
#                 gc.collect()
#             else:
#                 print(f"      PMG Error: Operation failed")
#                 pmg_times.append(None)

#         data_store["linear"]["pmg"].append(pmg_times)

#     with open(DATA_FILE, 'w') as f:
#         json.dump(data_store, f, indent=4)
#     print(f"\nBenchmark data saved to {DATA_FILE}")

# def plot_benchmark_results():
#     if not DATA_FILE.exists():
#         print("No data found. Run benchmark first.")
#         return

#     with open(DATA_FILE, 'r') as f:
#         data = json.load(f)

#     fig, ax = plt.subplots(figsize=(5, 3))

#     import numpy as np

#     # MolCrysKit (Cubic) - with error bars
#     mck_x = data["cubic"]["atoms"]
#     mck_y_list = data["cubic"]["mck"]
    
#     mck_means = []
#     mck_stds = []
#     mck_x_filtered = []
    
#     for i, times_list in enumerate(mck_y_list):
#         valid_times = [t for t in times_list if t is not None]
#         if valid_times:
#             mck_means.append(np.mean(valid_times))
#             mck_stds.append(np.std(valid_times))
#             mck_x_filtered.append(mck_x[i])
#         else:
#             mck_means.append(None)
#             mck_stds.append(None)
#             mck_x_filtered.append(mck_x[i])
    
#     mck_x_plot = [x for x, mean in zip(mck_x_filtered, mck_means) if mean is not None]
#     mck_y_plot = [mean for mean in mck_means if mean is not None]
#     mck_err_plot = [std for std, mean in zip(mck_stds, mck_means) if mean is not None]
    
#     if mck_y_plot:
#         ax.errorbar(mck_x_plot, mck_y_plot, yerr=mck_err_plot, fmt='s-', color='#1f77b4', 
#                    linewidth=2.5, markersize=8, label='MolCrysKit (Topological)', capsize=3, capthick=1)

#     # ASE (Cubic) - with error bars
#     ase_x = data["cubic"]["atoms"]
#     ase_y_list = data["cubic"]["ase"]
    
#     ase_means = []
#     ase_stds = []
#     ase_x_filtered = []
    
#     for i, times_list in enumerate(ase_y_list):
#         valid_times = [t for t in times_list if t is not None]
#         if valid_times:
#             ase_means.append(np.mean(valid_times))
#             ase_stds.append(np.std(valid_times))
#             ase_x_filtered.append(ase_x[i])
#         else:
#             ase_means.append(None)
#             ase_stds.append(None)
#             ase_x_filtered.append(ase_x[i])
    
#     ase_x_plot = [x for x, mean in zip(ase_x_filtered, ase_means) if mean is not None]
#     ase_y_plot = [mean for mean in ase_means if mean is not None]
#     ase_err_plot = [std for std, mean in zip(ase_stds, ase_means) if mean is not None]
    
#     if ase_y_plot:
#         ax.errorbar(ase_x_plot, ase_y_plot, yerr=ase_err_plot, fmt='o--', color='gray', 
#                    alpha=0.5, label='ASE (Geometric)', capsize=3, capthick=1)

#     # Pymatgen (Linear) - with error bars
#     pmg_raw_x = data["linear"]["atoms"]
#     pmg_raw_y_list = data["linear"]["pmg"]
    
#     pmg_means = []
#     pmg_stds = []
#     pmg_x_filtered = []
    
#     for i, times_list in enumerate(pmg_raw_y_list):
#         valid_times = [t for t in times_list if t is not None]
#         if valid_times:
#             pmg_means.append(np.mean(valid_times))
#             pmg_stds.append(np.std(valid_times))
#             pmg_x_filtered.append(pmg_raw_x[i])
#         else:
#             pmg_means.append(None)
#             pmg_stds.append(None)
#             pmg_x_filtered.append(pmg_raw_x[i])
    
#     pmg_x_plot = [x for x, mean in zip(pmg_x_filtered, pmg_means) if mean is not None]
#     pmg_y_plot = [mean for mean in pmg_means if mean is not None]
#     pmg_err_plot = [std for std, mean in zip(pmg_stds, pmg_means) if mean is not None]
    
#     if pmg_y_plot:
#         ax.errorbar(pmg_x_plot, pmg_y_plot, yerr=pmg_err_plot, fmt='^-.', color="#999999", 
#                    linewidth=2.5, markersize=10, label='Pymatgen (Geometric with Post-Repair,)', 
#                    capsize=3, capthick=1)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_ylim(1e-4, 1e5)
#     ax.set_xlabel('Number of Atoms', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
    
#     ax.grid(True, which="major", ls="-", alpha=0.4)
#     ax.grid(True, which="minor", ls=":", alpha=0.2)
#     ax.legend(fontsize=12, ncol=1, frameon=False)

#     plt.tight_layout()
#     fig.savefig(OUTPUT_DIR / "slab_benchmark_final.png", dpi=300)
#     print(f"Saved plot to {OUTPUT_DIR / 'slab_benchmark_final.png'}")
#     print(f"Saved plot to {OUTPUT_DIR / 'slab_benchmark_final.pdf'}")
#     plt.show()

if __name__ == "__main__":
    run_benchmark()
    # plot_benchmark_results()