"""
QM9 Real-Dataset Hybridization Validation for MolCrysKit ChemicalEnvironment

Validates MolCrysKit's geometry-based hybridization inference against
RDKit topology-based reference labels on randomly selected molecules
from the QM9 dataset.

Pipeline:
  QM9 DFT xyz  ──→  MolCrysKit  ──→  predicted hybridization
                                           ↕ compare
  QM9 SMILES   ──→  RDKit topology  ──→  reference hybridization

The DFT-optimized geometries (B3LYP/6-31G(2df,p)) are read directly
from the original QM9 extended xyz archive, ensuring that MolCrysKit
operates on quantum-chemically accurate structures rather than
force-field-generated conformers.

QM9 source: Ramakrishnan et al., Sci. Data 1, 140022 (2014)
Archive: dsgdb9nsd.xyz.tar.bz2  (133,885 molecules, C/H/O/N/F, ≤9 heavy atoms)

Usage:
    python validate_hybridization_qm9_real.py
    python validate_hybridization_qm9_real.py --n_samples 1000 --seed 42
"""

import sys
import os
import json
import time
import random
import argparse
import logging
import tarfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType

# Add MolCrysKit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MolCrysKit'))
from molcrys_kit.analysis.chemical_env import ChemicalEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# QM9 extended-xyz parsing
# ─────────────────────────────────────────────────────────────────────────────

QM9_XYZ_ARCHIVE = os.path.join(os.path.dirname(__file__), "data",
                                "dsgdb9nsd.xyz.tar.bz2")


def parse_qm9_xyz(content: str) -> Optional[Dict]:
    """
    Parse a single QM9 extended-xyz string.

    QM9 format (Ramakrishnan et al.):
      Line 1:  n_atoms
      Line 2:  gdb <id> <scalar properties...>
      Lines 3..n+2:  Element  x  y  z  partial_charge
      Line n+3:  vibrational frequencies
      Line n+4:  SMILES_gdb  SMILES_relaxed
      Line n+5:  InChI_gdb   InChI_relaxed

    Returns dict with keys: elements, positions, smiles, n_atoms
    """
    lines = content.strip().split('\n')
    if len(lines) < 5:
        return None
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None

    if len(lines) < n_atoms + 4:
        return None

    def _parse_float(s: str) -> float:
        """Handle Mathematica-style scientific notation, e.g. '2.2021*^-6'."""
        return float(s.replace('*^', 'e'))

    elements = []
    positions = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].replace('\t', ' ').split()
        elem = parts[0]
        x, y, z = _parse_float(parts[1]), _parse_float(parts[2]), _parse_float(parts[3])
        elements.append(elem)
        positions.append([x, y, z])

    # SMILES line: two SMILES separated by tab
    #   Column 1: GDB SMILES (original, clean topology — no radical artifacts)
    #   Column 2: Relaxed SMILES (Open Babel from DFT xyz — may have degraded bond orders)
    # Use the GDB SMILES (first column) for reference labels.
    smiles_line = lines[2 + n_atoms + 1]  # skip frequencies line
    smiles_parts = smiles_line.strip().split('\t')
    smiles = smiles_parts[0].strip()  # GDB SMILES, not relaxed

    return {
        'elements': elements,
        'positions': np.array(positions),
        'smiles': smiles,
        'n_atoms': n_atoms,
    }


def load_qm9_entries(archive_path: str = QM9_XYZ_ARCHIVE,
                     indices: Optional[List[int]] = None) -> Tuple[List[Dict], int]:
    """
    Load QM9 xyz entries from the tar.bz2 archive in a SINGLE PASS.

    Args:
        archive_path: Path to dsgdb9nsd.xyz.tar.bz2
        indices: 0-based indices of entries to load. If None, loads all.

    Returns:
        (list of parsed xyz dicts, total_count)
    """
    entries = []
    idx_set = set(indices) if indices is not None else None
    total_count = 0

    with tarfile.open(archive_path, 'r:bz2') as tf:
        for i, member in enumerate(tf):
            total_count = i + 1
            if idx_set is not None and i not in idx_set:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            content = f.read().decode('utf-8')
            parsed = parse_qm9_xyz(content)
            if parsed is not None:
                parsed['index'] = i
                parsed['filename'] = member.name
                entries.append(parsed)
            if idx_set is not None and len(entries) == len(idx_set):
                # Read remaining to get total_count (fast skip)
                for j, _ in enumerate(tf, start=i + 1):
                    total_count = j + 1
                break

    return entries, total_count


# ─────────────────────────────────────────────────────────────────────────────
# Hybridization helpers
# ─────────────────────────────────────────────────────────────────────────────

def rdkit_hybridization_to_str(hyb: HybridizationType) -> Optional[str]:
    """Convert RDKit HybridizationType to sp/sp2/sp3 string."""
    mapping = {
        HybridizationType.SP: "sp",
        HybridizationType.SP2: "sp2",
        HybridizationType.SP3: "sp3",
    }
    return mapping.get(hyb, None)


def molcryskit_hybridization(graph: nx.Graph, positions: np.ndarray,
                             atom_idx: int) -> Optional[str]:
    """
    Use MolCrysKit ChemicalEnvironment to infer hybridization for a single atom.
    Returns 'sp', 'sp2', 'sp3', or None if undetermined.
    """
    try:
        env = ChemicalEnvironment((graph, positions))
        site = env.get_site(atom_idx)
        strategy = site.get_hydrogen_completion_strategy()
        geometry = strategy.get('geometry', '')

        geo_to_hyb = {
            'linear': 'sp',
            'trigonal_planar': 'sp2',
            'planar_aromatic': 'sp2',
            'planar_bisector': 'sp2',
            'tetrahedral': 'sp3',
            'bent': 'sp3',
        }
        return geo_to_hyb.get(geometry, None)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-molecule validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_molecule_dft(entry: Dict) -> Optional[Dict]:
    """
    Validate hybridization using DFT xyz geometry from QM9.

    Args:
        entry: parsed QM9 xyz dict with keys:
               elements, positions, smiles, n_atoms

    Returns:
        dict with per-atom results, or None if molecule fails.
    """
    try:
        smiles = entry['smiles']
        xyz_elements = entry['elements']
        xyz_positions = entry['positions']

        # Parse SMILES → RDKit mol (for bond topology + reference labels)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        n_rdkit = mol.GetNumAtoms()

        # Verify atom count match
        if n_rdkit != len(xyz_elements):
            return None

        # Verify element-by-element match (RDKit and QM9 xyz order)
        rdkit_elements = [mol.GetAtomWithIdx(i).GetSymbol()
                          for i in range(n_rdkit)]
        if rdkit_elements != xyz_elements:
            # Try to match by reordering — but for QM9 this should be
            # consistent if we use the relaxed SMILES from the xyz file.
            # If not, skip this molecule.
            return None

        # Build graph from RDKit bond topology
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
        for bond in mol.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        # Use DFT xyz positions (not RDKit generated!)
        positions = xyz_positions

        # Evaluate each heavy atom
        heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        atom_results = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                continue

            idx = atom.GetIdx()
            element = atom.GetSymbol()

            rdkit_hyb = rdkit_hybridization_to_str(atom.GetHybridization())
            if rdkit_hyb not in ('sp', 'sp2', 'sp3'):
                continue

            mck_hyb = molcryskit_hybridization(graph, positions, idx)

            atom_results.append({
                'atom_index': idx,
                'element': element,
                'rdkit_hyb': rdkit_hyb,
                'mck_hyb': mck_hyb,
                'correct': (mck_hyb == rdkit_hyb) if mck_hyb is not None else False,
                'predicted': mck_hyb is not None,
                'is_aromatic': atom.GetIsAromatic(),
            })

        if not atom_results:
            return None

        return {
            'smiles': smiles,
            'n_heavy_atoms': len(heavy_atoms),
            'atoms': atom_results,
        }

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Run validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(entries: List[Dict], desc: str = "") -> Dict:
    """Run validation on a list of QM9 xyz entries."""
    results = []
    failed = []
    t0 = time.time()

    for i, entry in enumerate(entries):
        t_mol = time.time()
        res = validate_molecule_dft(entry)
        elapsed = time.time() - t_mol

        smi = entry.get('smiles', '?')
        if res is not None:
            res['time_s'] = elapsed
            results.append(res)
            status = "OK"
        else:
            failed.append(smi)
            status = "FAIL"

        if (i + 1) % 50 == 0 or i == len(entries) - 1:
            logging.info(f"[{desc}] {i+1}/{len(entries)} | {status} | {smi[:40]}")

    total_time = time.time() - t0

    all_atoms = [a for r in results for a in r['atoms']]
    total_atoms = len(all_atoms)
    correct_atoms = sum(1 for a in all_atoms if a['correct'])
    predicted_atoms = sum(1 for a in all_atoms if a['predicted'])

    # Aromatic atom statistics
    aromatic_atoms = [a for a in all_atoms if a.get('is_aromatic', False)]
    aromatic_total = len(aromatic_atoms)
    aromatic_correct = sum(1 for a in aromatic_atoms if a['correct'])
    n_aromatic_mols = sum(
        1 for r in results
        if any(a.get('is_aromatic', False) for a in r['atoms'])
    )

    elements = sorted(set(a['element'] for a in all_atoms))
    per_element = {}
    for elem in elements:
        elem_atoms = [a for a in all_atoms if a['element'] == elem]
        elem_correct = sum(1 for a in elem_atoms if a['correct'])
        per_element[elem] = {
            'total': len(elem_atoms),
            'correct': elem_correct,
            'accuracy': elem_correct / len(elem_atoms) if elem_atoms else 0.0,
        }

    hyb_types = ['sp', 'sp2', 'sp3']
    confusion = {true_h: {pred_h: 0 for pred_h in hyb_types + ['None']}
                 for true_h in hyb_types}
    for a in all_atoms:
        true_h = a['rdkit_hyb']
        pred_h = a['mck_hyb'] if a['mck_hyb'] is not None else 'None'
        if true_h in confusion:
            confusion[true_h][pred_h if pred_h in confusion[true_h] else 'None'] += 1

    failure_cases = []
    for r in results:
        wrong = [a for a in r['atoms'] if not a['correct']]
        if wrong:
            failure_cases.append({'smiles': r['smiles'], 'wrong_atoms': wrong})

    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'QM9 DFT xyz (B3LYP/6-31G(2df,p))',
        'geometry_source': 'DFT-optimized (from dsgdb9nsd.xyz.tar.bz2)',
        'reference_labels': 'RDKit topology-based GetHybridization()',
        'n_smiles_input': len(entries),
        'n_molecules_ok': len(results),
        'n_molecules_failed': len(failed),
        'failed_smiles': failed,
        'total_atoms_evaluated': total_atoms,
        'atoms_predicted': predicted_atoms,
        'atoms_correct': correct_atoms,
        'overall_accuracy': correct_atoms / total_atoms if total_atoms > 0 else 0.0,
        'coverage': predicted_atoms / total_atoms if total_atoms > 0 else 0.0,
        'per_element': per_element,
        'aromatic': {
            'n_aromatic_molecules': n_aromatic_mols,
            'n_aromatic_atoms': aromatic_total,
            'n_aromatic_correct': aromatic_correct,
            'aromatic_accuracy': aromatic_correct / aromatic_total if aromatic_total > 0 else 0.0,
        },
        'confusion_matrix': confusion,
        'failure_cases': failure_cases,
        'total_time_s': total_time,
        'avg_time_per_mol_s': total_time / len(results) if results else 0.0,
    }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QM9 Hybridization Validation (DFT geometry)'
    )
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of molecules to randomly select (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str,
                        default='results/qm9_real_raw.json',
                        help='Output JSON file')
    parser.add_argument('--xyz_archive', type=str, default=None,
                        help='Path to QM9 xyz archive (default: data/dsgdb9nsd.xyz.tar.bz2)')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    log_file = f'logs/run_qm9_real_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )

    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MolCrysKit'),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    logging.info("=== QM9 Hybridization Validation (DFT geometry) ===")
    logging.info(f"N samples: {args.n_samples}")
    logging.info(f"Random seed: {args.seed}")
    logging.info(f"MolCrysKit git: {git_hash}")
    logging.info(f"RDKit version: {Chem.rdBase.rdkitVersion}")

    # Determine archive path
    archive = args.xyz_archive or QM9_XYZ_ARCHIVE
    if not os.path.exists(archive):
        logging.error(f"QM9 archive not found: {archive}")
        logging.error("Download from: https://figshare.com/collections/"
                      "Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904")
        sys.exit(1)

    # QM9 has exactly 133,885 entries — sample indices without scanning
    N_QM9 = 133885
    random.seed(args.seed)
    np.random.seed(args.seed)

    n_sample = min(args.n_samples, N_QM9)
    sample_indices = sorted(random.sample(range(N_QM9), n_sample))
    logging.info(f"Sampled {len(sample_indices)} indices (seed={args.seed})")

    # Load xyz entries (single pass through archive)
    logging.info(f"Loading DFT xyz entries from {archive} ...")
    entries, n_total = load_qm9_entries(archive, indices=sample_indices)
    logging.info(f"Loaded {len(entries)} entries (archive has {n_total})")
    if entries:
        logging.info(f"First SMILES: {entries[0]['smiles']}")

    # Run validation
    summary = run_validation(entries, desc="QM9-DFT")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (QM9 — DFT geometry)")
    print("=" * 60)
    print(f"Geometry source:     B3LYP/6-31G(2df,p) DFT-optimized xyz")
    print(f"Reference labels:    RDKit topology-based hybridization")
    print(f"Molecules sampled:   {len(entries)} (random, seed={args.seed})")
    print(f"Molecules processed: {summary['n_molecules_ok']} / {summary['n_smiles_input']}")
    print(f"Molecules failed:    {summary['n_molecules_failed']}")
    print(f"Atoms evaluated:     {summary['total_atoms_evaluated']}")
    print(f"Overall accuracy:    {summary['overall_accuracy']:.1%}")
    print(f"Coverage:            {summary['coverage']:.1%}")
    print(f"Total time:          {summary['total_time_s']:.1f} s")
    print()
    print("Per-element accuracy:")
    for elem, stats in sorted(summary['per_element'].items()):
        print(f"  {elem:2s}: {stats['correct']:4d}/{stats['total']:4d} = {stats['accuracy']:.1%}")
    print()
    arom = summary['aromatic']
    print(f"Aromatic molecules:  {arom['n_aromatic_molecules']}")
    print(f"Aromatic atoms:      {arom['n_aromatic_correct']}/{arom['n_aromatic_atoms']}"
          f" = {arom['aromatic_accuracy']:.1%}")
    print()
    print("Confusion matrix (rows=RDKit truth, cols=MolCrysKit prediction):")
    hyb_types = ['sp', 'sp2', 'sp3']
    header = f"{'':8s}" + "".join(f"{h:8s}" for h in hyb_types + ['None'])
    print(header)
    for true_h in hyb_types:
        row = f"{true_h:8s}"
        for pred_h in hyb_types + ['None']:
            row += f"{summary['confusion_matrix'][true_h].get(pred_h, 0):8d}"
        print(row)
    print()
    if summary['failure_cases']:
        print(f"Failure cases ({len(summary['failure_cases'])} shown):")
        for fc in summary['failure_cases'][:5]:
            print(f"  {fc['smiles']}")
            for wa in fc['wrong_atoms'][:3]:
                print(f"    {wa['element']}: RDKit={wa['rdkit_hyb']}, MCK={wa['mck_hyb']}")

    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
                exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Results saved to: {args.output}")
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
