# QM9 Hybridization Validation Experiment

## Overview

Validates MolCrysKit's geometry-based hybridization inference against RDKit's
topology-based reference labels on molecules from the
[QM9 dataset](https://doi.org/10.1038/sdata.2014.22) (Ramakrishnan et al., 2014).

This analysis supports **Figures S2–S3** and **Table S1** in the manuscript.

## Pipeline

```
QM9 DFT xyz  ──→  MolCrysKit ChemicalEnvironment  ──→  predicted hybridization
     (B3LYP/6-31G(2df,p) geometry)                           ↕ compare
QM9 GDB SMILES  ──→  RDKit GetHybridization()  ──→  reference hybridization
     (original topology from GDB-17)
```

## Results (n = 1,000, seed = 42)

| Metric | Value | 95% Wilson CI |
|--------|-------|---------------|
| Overall accuracy | 98.8% (8,588/8,692 atoms) | [98.6%–99.0%] |
| C accuracy | 99.7% (6,268/6,285) | [99.6%–99.8%] |
| N accuracy | 96.4% (993/1,030) | [95.1%–97.4%] |
| O accuracy | 96.3% (1,300/1,350) | [95.2%–97.2%] |
| F accuracy | 100.0% (27/27) | [87.5%–100.0%] |
| Aromatic atoms | 99.4% (963/969) | [98.7%–99.7%] |
| Molecules excluded | 13/1,000 (1.3%) | — |

## Files

| File | Description |
|------|-------------|
| `validate_hybridization_qm9_real.py` | Main validation script (DFT xyz pipeline) |
| `plot_qm9_analysis.py` | Figure generation + Wilson CI summary export |
| `results/qm9_real_raw.json` | Full validation results (per-molecule, per-atom) |
| `results/qm9_statistics.json` | Manuscript-ready statistics with Wilson CIs |

## Data Requirements

The QM9 DFT xyz archive must be downloaded separately:
- **Source**: [figshare](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)
- **File**: `dsgdb9nsd.xyz.tar.bz2` (~82 MB, 133,885 molecules)
- **Place in**: `data/` subdirectory

## Reproduction

```bash
cd paper/qm9_benchmark

# Download QM9 data (if not present)
mkdir -p data
# Download dsgdb9nsd.xyz.tar.bz2 from figshare and place in data/

# Run validation (requires MolCrysKit + RDKit)
python validate_hybridization_qm9_real.py --n_samples 1000 --seed 42

# Generate figures
python plot_qm9_analysis.py
```

## Dependencies

- Python 3.10+
- MolCrysKit (development install)
- RDKit 2024.03+
- NumPy, NetworkX, Matplotlib, Pillow
