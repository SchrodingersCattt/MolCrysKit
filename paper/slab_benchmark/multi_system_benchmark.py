"""
Multi-System Slab Generation Benchmark
=======================================
Tests MolCrysKit topological slab generation vs ASE across multiple molecular
crystal systems with different unit cell sizes, demonstrating cubic size scaling.

Systems:
  1. Acetaminophen (HXACAN)  – small organic, monoclinic, ~160 atoms/unit cell
  2. Caffeine (anhydrous)    – medium organic, monoclinic, ~192 atoms/unit cell
  3. Isatin (ISATIN)         – medium organic with N/O, monoclinic, ~44 atoms/unit cell

For each system: time slab generation at 4 supercell scales (n×n×n cubic expansion).
Compare MCK (topological) with ASE (geometry-only) as reference.

Thread isolation strategy (from MolCrysKit/benchmarks/slab_efficiency.py):
  - OMP/MKL/NumExpr thread counts forced to 1 at module import AND inside each worker
  - Each timing call runs in a completely isolated subprocess via multiprocessing.Process
  - GC is disabled inside the worker during the timed section
  - Parent process only joins the subprocess; no shared memory beyond a Manager dict

Output:
  results/multi_system_benchmark.json
  results/multi_system_benchmark_figure.pdf / .png

Usage (local):
  cd R1/experiments/benchmark
  python multi_system_benchmark.py

Usage (Bohrium/remote):
  python multi_system_benchmark.py > eff.log 2>&1
"""

# ── thread control (must be set before any numpy/scipy import) ─────────────────
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
import multiprocessing
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── path bootstrap ─────────────────────────────────────────────────────────────
# Allow running from any working directory; project root is 3 levels up.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MCK_DEV = str(_PROJECT_ROOT / "MolCrysKit")
if _MCK_DEV not in sys.path:
    sys.path.insert(0, _MCK_DEV)

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CIF files live next to this script (copied by copy_cifs.py)
CIF_DIR = SCRIPT_DIR / "slab_examples"

SYSTEMS = [
    {
        "name":   "Acetaminophen (HXACAN)",
        "short":  "HXACAN",
        "cif":    CIF_DIR / "Acetaminophen_HXACAN.cif",
        "miller": (0, 1, 0),
    },
    {
        "name":   "Caffeine (anhydrous)",
        "short":  "Caffeine",
        "cif":    CIF_DIR / "anhydrousCaffeine_CGD_2007_7_1406.cif",
        "miller": (0, 0, 1),
    },
    {
        "name":   "Isatin (ISATIN)",
        "short":  "ISATIN",
        "cif":    CIF_DIR / "ISATIN.cif",
        "miller": (1, 0, 0),
    },
]

SCALES   = [1, 2, 3, 4]   # cubic supercell scales (n×n×n)
N_REPEAT = 3               # timing repeats per data point
LAYERS   = 3
VACUUM   = 10.0

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── logging ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── slab generation functions (top-level for pickle) ──────────────────────────

def _run_ase_slab(atoms, miller, layers, vacuum):
    """ASE geometry-only slab generation (identical to canonical benchmark)."""
    from ase.build import surface as ase_surface
    slab = ase_surface(atoms, miller, layers=layers, vacuum=vacuum)
    positions = slab.get_positions()
    cell      = slab.get_cell()
    z         = positions[:, 2]
    thickness = np.max(z) - np.min(z)
    cell[2]   = np.array([0, 0, thickness + vacuum])
    positions[:, 2] += (-np.min(z) + 1.0)
    slab.set_positions(positions)
    slab.set_cell(cell, scale_atoms=False)
    return slab


def _run_mck_slab(atoms, miller, layers, vacuum):
    """MolCrysKit topological slab generation (from_ase conversion included)."""
    from molcrys_kit.structures import MolecularCrystal
    from molcrys_kit.operations import generate_topological_slab
    crystal = MolecularCrystal.from_ase(atoms)
    return generate_topological_slab(crystal, miller, layers=layers, vacuum=vacuum)


# ── isolated subprocess worker ────────────────────────────────────────────────

def _benchmark_worker(task_type: str, args_tuple: tuple, return_dict: dict) -> None:
    """
    Worker executed in a completely isolated subprocess.

    - Forces OMP_NUM_THREADS=1 again (child environment may differ).
    - Disables GC during the timed section to eliminate GC jitter.
    - Returns wall-clock time via a Manager.dict for safe IPC.
    """
    try:
        import os as _os
        _os.environ["OMP_NUM_THREADS"] = "1"
        _os.environ["MKL_NUM_THREADS"] = "1"

        gc.disable()
        t0 = time.perf_counter()

        if task_type == "ase":
            _run_ase_slab(*args_tuple)
        elif task_type == "mck":
            _run_mck_slab(*args_tuple)
        else:
            raise ValueError(f"Unknown task type: {task_type!r}")

        t1 = time.perf_counter()
        return_dict["time"]    = t1 - t0
        return_dict["success"] = True

    except Exception as exc:
        return_dict["error"]     = str(exc)
        return_dict["traceback"] = traceback.format_exc()
        return_dict["success"]   = False
    finally:
        gc.enable()


def run_isolated_benchmark(task_type: str, *args) -> float | None:
    """
    Spawn a fresh subprocess, run one timing call, wait for it to finish,
    then return the measured wall time (or None on failure).

    Using a fresh process per call guarantees:
      - No memory from previous runs leaks into the timed section.
      - Python's import caches, JIT state, etc. are identical for all runs.
    """
    manager     = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(
        target=_benchmark_worker, args=(task_type, args, return_dict)
    )
    p.start()
    p.join()   # blocks until subprocess exits and memory is fully released

    if return_dict.get("success"):
        return return_dict["time"]
    else:
        log(f"    [subprocess error] {return_dict.get('error', 'unknown')}")
        tb = return_dict.get("traceback", "")
        if tb:
            for line in tb.splitlines():
                log(f"      {line}")
        return None


# ── per-system benchmark ───────────────────────────────────────────────────────

def benchmark_system(sys_info: dict) -> dict | None:
    """Run warmup + timed benchmark for one crystal system at all supercell scales."""
    from molcrys_kit.io.cif import read_mol_crystal

    name     = sys_info["short"]
    cif_path = sys_info["cif"]
    miller   = sys_info["miller"]

    log(f"  Loading {cif_path.name} …")
    if not cif_path.exists():
        log(f"  ERROR: CIF not found: {cif_path}")
        return None

    try:
        crystal   = read_mol_crystal(str(cif_path))
        base_ase  = crystal.to_ase()
        base_natoms = len(base_ase)
        log(f"  Base unit cell: {base_natoms} atoms")
    except Exception as exc:
        log(f"  ERROR loading CIF: {exc}")
        traceback.print_exc()
        return None

    # ── warmup (fills OS disk cache, initialises JIT, etc.) ───────────────────
    log("  Warming up …")
    try:
        run_isolated_benchmark("mck", base_ase, miller, LAYERS, VACUUM)
        run_isolated_benchmark("ase", base_ase, miller, LAYERS, VACUUM)
    except Exception as exc:
        log(f"  Warmup error (non-fatal): {exc}")

    result = {
        "name":        sys_info["name"],
        "miller":      list(miller),
        "base_natoms": base_natoms,
        "scales":      [],
        "natoms":      [],
        "ase_times":   [],
        "mck_times":   [],
    }

    for n in SCALES:
        super_ase = base_ase * (n, n, n)
        natoms    = len(super_ase)
        log(f"\n  Scale {n}×{n}×{n} ({natoms} atoms):")
        result["scales"].append(n)
        result["natoms"].append(natoms)

        # ── ASE ───────────────────────────────────────────────────────────────
        ase_run_times = []
        for r in range(N_REPEAT):
            t = run_isolated_benchmark("ase", super_ase, miller, LAYERS, VACUUM)
            ase_run_times.append(t)
            if t is not None:
                log(f"    ASE  run {r + 1}: {t:.4f} s")
            else:
                log(f"    ASE  run {r + 1}: FAILED")
        result["ase_times"].append(ase_run_times)

        # ── MolCrysKit ────────────────────────────────────────────────────────
        mck_run_times = []
        for r in range(N_REPEAT):
            t = run_isolated_benchmark("mck", super_ase, miller, LAYERS, VACUUM)
            mck_run_times.append(t)
            if t is not None:
                log(f"    MCK  run {r + 1}: {t:.4f} s")
            else:
                log(f"    MCK  run {r + 1}: FAILED")
        result["mck_times"].append(mck_run_times)

        del super_ase
        gc.collect()

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def collect_metadata() -> dict:
    meta = {
        "processor_count":  multiprocessing.cpu_count(),
        "platform_system":  platform.system(),
        "platform_release": platform.release(),
        "python_version":   platform.python_version(),
        "timestamp":        TIMESTAMP,
        "cpu_model":        "Unknown",
    }
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        meta["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            import subprocess
            meta["cpu_model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
        elif platform.system() == "Windows":
            import subprocess
            meta["cpu_model"] = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], universal_newlines=True
            ).split("\n")[1].strip()
    except Exception:
        pass
    return meta


def main():
    log("=" * 62)
    log("Multi-System Slab Generation Benchmark (isolated subprocess)")
    log(f"Systems : {[s['short'] for s in SYSTEMS]}")
    log(f"Scales  : {SCALES} (cubic n×n×n)  |  repeats: {N_REPEAT}")
    log("=" * 62)

    meta    = collect_metadata()
    log(f"System  : {meta['cpu_model']}")

    all_results: dict = {}

    for sys_info in SYSTEMS:
        log(f"\n{'─' * 50}")
        log(f"Benchmarking: {sys_info['short']}  ({sys_info['name']})")
        res = benchmark_system(sys_info)
        if res is not None:
            all_results[sys_info["short"]] = res
        else:
            log(f"  SKIPPED: {sys_info['short']}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {"metadata": meta, "systems": all_results}
    out_path = RESULTS_DIR / "multi_system_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved: {out_path}")

    # ── summary table ─────────────────────────────────────────────────────────
    log("\n" + "=" * 62)
    log("SUMMARY")
    log("=" * 62)
    for short, res in all_results.items():
        log(f"\n{res['name']}  (Miller {res['miller']}):")
        log(f"  {'Scale':>6}  {'Atoms':>7}  {'MCK (s)':>11}  {'ASE (s)':>11}  {'Ratio':>7}")
        log(f"  {'─'*6}  {'─'*7}  {'─'*11}  {'─'*11}  {'─'*7}")
        for i, n in enumerate(res["scales"]):
            natoms = res["natoms"][i]
            mck_t  = [t for t in res["mck_times"][i] if t is not None]
            ase_t  = [t for t in res["ase_times"][i]  if t is not None]
            mck_s  = f"{np.mean(mck_t):.3f}" if mck_t else "N/A"
            ase_s  = f"{np.mean(ase_t):.4f}" if ase_t else "N/A"
            ratio  = (f"{np.mean(mck_t) / np.mean(ase_t):.1f}×"
                      if mck_t and ase_t else "N/A")
            log(f"  {n:>5}×  {natoms:>7}  {mck_s:>11}  {ase_s:>11}  {ratio:>7}")

    return output


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 'spawn' is required on all platforms so child processes do NOT inherit
    # the parent's import state, ensuring a truly clean timing environment.
    multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()   # harmless no-op on non-Windows

    try:
        main()
    except Exception as exc:
        log(f"FATAL: {exc}")
        traceback.print_exc()
        sys.exit(1)
