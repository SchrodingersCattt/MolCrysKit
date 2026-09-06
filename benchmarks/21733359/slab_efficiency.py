import multiprocessing
import time
import os
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import platform
import psutil  # 需要安装: pip install psutil

# 设置单线程环境变量，防止 numpy 自动并行干扰多进程或导致 CPU 争抢
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
    raise ImportError("Please install molcrys_kit first.")

from time import sleep
this_file_path = Path(__file__).resolve().parent
print(f"Current working directory: {this_file_path}")


OUTPUT_DIR = Path(this_file_path)
EXAMPLE_DIR = this_file_path / "slab_examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = OUTPUT_DIR / "benchmark_data.json"

# CIF_PATH = Path("examples/Acetaminophen_HXACAN.cif")
CIF_PATH = Path(EXAMPLE_DIR / "Acetaminophen_HXACAN.cif")
MILLER = (0, 1, 0)
LAYERS = 3
VACUUM = 10.0
# sleep(999)
CUSTOM_COLORS = {
    'C': "#535353", 'H': '#eeeeee', 'N': "#6e89e1", 'O': "#bf0603", 'Cl': '#26c81b',
}

SCALES_CUBIC = [1, 2, 3, 4, 5, 6]
SCALES_LINEAR = [1, 2, 3]
PMG_ATOM_LIMIT = 1000

# ---------------------------------------------------------
# Core Logic Functions (Top-level for multiprocessing pickling)
# ---------------------------------------------------------

def run_ase_slab_gen(atoms, miller, layers, vacuum):
    ase_slab = ase_surface(atoms, miller, layers=layers, vacuum=vacuum)
    positions = ase_slab.get_positions()
    cell = ase_slab.get_cell()
    z_coords = positions[:, 2]
    min_z, max_z = np.min(z_coords), np.max(z_coords)
    atomic_thickness = max_z - min_z
    cell[2] = np.array([0, 0, atomic_thickness + vacuum])
    positions[:, 2] += (-min_z + 1.0)
    ase_slab.set_positions(positions)
    ase_slab.set_cell(cell, scale_atoms=False)
    return ase_slab

def run_pymatgen_slab_gen(ase_structure, miller, layers, vacuum):
    bulk_structure = AseAtomsAdaptor.get_structure(ase_structure)
    # 实时计算 d_hkl，保证 benchmark 包含所有开销
    recp_lat = bulk_structure.lattice.reciprocal_lattice
    d_hkl = 2 * np.pi / np.linalg.norm(recp_lat.get_cartesian_coords(miller))
    
    effective_layers = max(0.1, float(layers) - 0.5)
    slab_thickness = effective_layers * d_hkl

    slab_gen = SlabGenerator(
        bulk_structure, miller_index=miller, min_slab_size=slab_thickness,
        min_vacuum_size=vacuum, in_unit_planes=False, center_slab=False, primitive=False
    )
    slabs = slab_gen.get_slabs(repair=True, tol=0.1, ftol=0.2, ztol=0.2)
    if not slabs: raise ValueError("No slabs generated")
    
    slab = slabs[0]
    cart_coords = slab.cart_coords
    real_atomic_thickness = np.ptp(cart_coords[:, 2])
    target_c = real_atomic_thickness + vacuum
    
    new_lat = slab.lattice.matrix.copy()
    new_lat[2] *= (target_c / np.linalg.norm(new_lat[2]))
    slab.lattice = Lattice(new_lat)
    return AseAtomsAdaptor.get_atoms(slab)

# ---------------------------------------------------------
# Multiprocessing Worker
# ---------------------------------------------------------

def _benchmark_worker(task_type, args_tuple, return_dict):
    """
    Worker process function.
    Runs in a completely isolated process space.
    """
    try:
        # 再次强制单线程，确保子进程环境正确
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # 禁止 GC 在计时期间运行，防止不可控的抖动
        gc.disable()
        
        t0 = time.perf_counter()
        
        if task_type == 'ase':
            # args: (atoms, miller, layers, vacuum)
            run_ase_slab_gen(*args_tuple)
            
        elif task_type == 'pmg':
            # args: (atoms, miller, layers, vacuum)
            run_pymatgen_slab_gen(*args_tuple)
            
        elif task_type == 'mck':
            # args: (atoms, miller, layers, vacuum)
            # 注意：为了公平对比，我们在计时开始后进行 from_ase 转换
            # 或者如果你想排除转换时间，可以在 t0 之前做
            # 这里按照之前的逻辑：转换包含在 MCK 的 total pipeline 耗时中
            # (如果之前逻辑是分开的，可以调整位置)
            atoms, m, l, v = args_tuple
            c_super = MolecularCrystal.from_ase(atoms)
            generate_topological_slab(c_super, m, layers=l, vacuum=v)
            
        t1 = time.perf_counter()
        
        return_dict['time'] = t1 - t0
        return_dict['success'] = True
        
    except Exception as e:
        return_dict['error'] = str(e)
        return_dict['success'] = False
    finally:
        gc.enable()

def run_isolated_benchmark(task_type, *args):
    """
    Spawns a new process to run the benchmark task.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p = multiprocessing.Process(target=_benchmark_worker, args=(task_type, args, return_dict))
    p.start()
    p.join() # 等待进程完全结束、内存完全回收
    
    if return_dict.get('success'):
        return return_dict['time']
    else:
        print(f"      [Error in Subprocess]: {return_dict.get('error')}")
        return None

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def run_benchmark():
    print(f"Loading Crystal from: {CIF_PATH}")
    if not CIF_PATH.exists(): raise FileNotFoundError(f"{CIF_PATH}")

    crystal = read_mol_crystal(str(CIF_PATH))
    original_ase = crystal.to_ase()
    print(f"Base cell atoms: {len(original_ase)}")

    # --- WARM UP (Important for OS cache & JIT) ---
    print("\nWarming up (Running each method once)...")
    try:
        run_isolated_benchmark('mck', original_ase, MILLER, LAYERS, VACUUM)
        run_isolated_benchmark('ase', original_ase, MILLER, LAYERS, VACUUM)
        run_isolated_benchmark('pmg', original_ase, MILLER, LAYERS, VACUUM) # Added PMG warmup
    except: pass

    # 收集CPU和其他系统信息作为元数据
    cpu_info = {
        "processor_count": multiprocessing.cpu_count(),
        "processor_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
        "processor_freq_min": psutil.cpu_freq().min if psutil.cpu_freq() else "Unknown",
        "processor_arch": platform.processor() or platform.machine(),
        "cpu_model": platform.processor() or platform.uname().processor if hasattr(platform.uname(), 'processor') else "Unknown",
        "cpu_percent_per_core": psutil.cpu_percent(percpu=True, interval=1),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
    }

    # 尝试通过不同方式获取CPU型号信息（针对不同操作系统）
    try:
        if platform.system() == "Linux":
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_info["cpu_model_detail"] = line.split(':')[1].strip()
                        break
        elif platform.system() == "Darwin":  # macOS
            import subprocess
            cpu_info["cpu_model_detail"] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        elif platform.system() == "Windows":
            import subprocess
            cpu_info["cpu_model_detail"] = subprocess.check_output(['wmic', 'cpu', 'get', 'name'], universal_newlines=True).split('\n')[1].strip()
    except:
        cpu_info["cpu_model_detail"] = "Could not determine"

    data_store = {
        "metadata": cpu_info,
        "cubic": {"atoms": [], "ase": [], "mck": []},
        "linear": {"atoms": [], "pmg": []}
    }
    
    NUM_RUNS = 3

    # [A] Cubic Scaling
    print(f"\n[A] Cubic Scaling (n,n,n)")
    for n in SCALES_CUBIC:
        super_ase = original_ase * (n, n, n)
        natoms = len(super_ase)
        data_store["cubic"]["atoms"].append(natoms)
        print(f"  > Scale {n}x{n}x{n} ({natoms} atoms)...")

        # ASE
        times = []
        for r in range(NUM_RUNS):
            t = run_isolated_benchmark('ase', super_ase, MILLER, LAYERS, VACUUM)
            times.append(t)
            if t: print(f"    ASE Run {r+1}: {t:.4f}s")
        data_store["cubic"]["ase"].append(times)

        # MolCrysKit
        times = []
        for r in range(NUM_RUNS):
            # 这里的逻辑是：把 ASE 对象传进去，Worker 内部转成 Crystal 并计时
            # 如果你想只测算法不测转换，可以修改 Worker 逻辑
            t = run_isolated_benchmark('mck', super_ase, MILLER, LAYERS, VACUUM)
            times.append(t)
            if t: print(f"    MCK Run {r+1}: {t:.4f}s")
        data_store["cubic"]["mck"].append(times)
        
        # 释放主进程内存
        del super_ase
        gc.collect()

    # [B] Linear Scaling
    print(f"\n[B] Linear Scaling (n,1,1)")
    for n in SCALES_LINEAR:
        super_ase = original_ase * (n, 1, 1)
        natoms = len(super_ase)
        data_store["linear"]["atoms"].append(natoms)
        
        if natoms > PMG_ATOM_LIMIT:
            data_store["linear"]["pmg"].append([None]*NUM_RUNS)
            print(f"  > Scale {n}x1x1 ({natoms} atoms): Skipped (limit)")
            continue
            
        print(f"  > Scale {n}x1x1 ({natoms} atoms)...")
        times = []
        for r in range(NUM_RUNS):
            t = run_isolated_benchmark('pmg', super_ase, MILLER, LAYERS, VACUUM)
            times.append(t)
            if t: print(f"    PMG Run {r+1}: {t:.4f}s")
        data_store["linear"]["pmg"].append(times)
        
        del super_ase
        gc.collect()

    with open(DATA_FILE, 'w') as f:
        json.dump(data_store, f, indent=4)
    print(f"\nSaved to {DATA_FILE}")
    print(f"CPU Info: {cpu_info}")

def plot_benchmark_results():
    if not DATA_FILE.exists(): return
    with open(DATA_FILE, 'r') as f: data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(4, 5)) # Slightly larger for paper
    
    # Helper to clean data
    def get_plot_data(x_raw, y_raw_list):
        means, stds, x_clean = [], [], []
        for x, times in zip(x_raw, y_raw_list):
            valid = [t for t in times if t is not None]
            if valid:
                means.append(np.mean(valid))
                stds.append(np.std(valid))
                x_clean.append(x)
        return x_clean, means, stds

    pmg_color, mck_color, ase_color = "#819c37", "#135788", "#b41f5d"
    # Plot PMG
    x, y, err = get_plot_data(data["linear"]["atoms"], data["linear"]["pmg"])
    ax.errorbar(x, y, yerr=err, fmt='^-.', ecolor=pmg_color, markerfacecolor="white", color=pmg_color, alpha=0.6, lw=1.5, label='Pymatgen (Geometric+Repair)', capsize=3, capthick=1)

    # Plot MCK, edge color is blue and face color is white
    x, y, err = get_plot_data(data["cubic"]["atoms"], data["cubic"]["mck"])
    ax.errorbar(x, y, yerr=err, fmt='s-', ecolor=mck_color, markerfacecolor=mck_color,
                color=mck_color, lw=1.5, label='MolCrysKit (Topological)', capsize=3, capthick=1)

    # Plot ASE
    x, y, err = get_plot_data(data["cubic"]["atoms"], data["cubic"]["ase"])
    ax.errorbar(x, y, yerr=err, fmt='o--', ecolor=ase_color, markerfacecolor="white", color=ase_color,alpha=0.6, lw=1.5, label='ASE (Geometric)', capsize=3, capthick=1)

    # remove upper and right edge lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e5)
    ax.set_xlabel('Number of Atoms', fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontweight='bold')
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "benchmark_final.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "benchmark_final.pdf")
    print("Plots saved.")
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    run_benchmark()
    plot_benchmark_results()
