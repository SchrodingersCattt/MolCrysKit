# Slab Generation Benchmark

## Overview

Benchmarks MolCrysKit's topological slab generation against ASE (geometric)
and Pymatgen (geometric + repair) across three molecular crystal systems with
increasing structural complexity.

This analysis supports **Figure 4a2** in the manuscript.

## Systems

| System | CSD Refcode | Space Group | Atoms/Cell |
|--------|-------------|-------------|------------|
| Acetaminophen | HXACAN | Pcab | 160 |
| β-HMX | OCHTET12 | P2₁/n | 56 |
| DAP-M4 | UHILUV02 | P2₁ | 180 |

## Benchmark Configuration

- **Supercell scales**: 1×1×1 through 6×6×6 (cubic expansion)
- **Timing repeats**: 3 per data point
- **Thread isolation**: OMP/MKL/NumExpr forced to 1 thread; each timing run
  in an isolated subprocess
- **Hardware**: 4-core Intel Xeon Platinum 8163 CPU @ 2.50 GHz (Bohrium Cloud)

## Files

| File | Description |
|------|-------------|
| `multi_system_benchmark.py` | Main benchmark script (MolCrysKit vs ASE) |
| `plot_benchmark.py` | Figure 4a2 generation |
| `results/multi_system_benchmark.json` | Timing data (MolCrysKit, ASE, Pymatgen) |

## Reproduction

```bash
cd paper/slab_benchmark

# Re-run benchmark (may take several hours)
python multi_system_benchmark.py

# Re-generate figure from existing results
python plot_benchmark.py
```

## Dependencies

- Python 3.10+
- MolCrysKit (development install)
- ASE
- Pymatgen (for Pymatgen comparison only)
- NumPy, Matplotlib
