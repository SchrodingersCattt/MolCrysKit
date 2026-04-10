"""
CSD Statistics Experiment
=========================
Goal: Quantify the prevalence of disorder and missing hydrogens in the CSD,
      demonstrating the necessity of MolCrysKit's preprocessing pipeline.

Sampling strategy:
  - Uniform random sample of N entries from the full CSD (1,413,222 entries)
  - Filter: has_3d_structure=True, organic molecules only (no metals)
  - Sample size: 50000 entries
  - Random seed: 42 (reproducible)

Classification (two non-overlapping categories + union):
  Category A — Crystallographic disorder:
      entry.has_disorder == True
      Corresponds to: MolCrysKit Section 2.3 disorder solver
  Category B — Missing / incomplete hydrogen:
      hydrogen_treatment in {Unknown, Missing, Omitted}
      OR no H atoms present in the structural model
      Corresponds to: MolCrysKit Section 2.5.4 H completion
  Union (A ∪ B) — Any preprocessing needed

Note: partial occupancy is NOT reported as a separate category because
crystallographic disorder is itself expressed through partial occupancy;
reporting both would create a confusing overlap. The entry-level
has_disorder flag is the authoritative indicator.

Output:
  - results/csd_statistics_raw.json      — per-entry data
  - results/csd_statistics_summary.json  — aggregated statistics + overlap matrix
  - results/csd_statistics_figure.pdf/png — publication figure

Author: MolCrysKit R1 revision
Date: 2026-03-21 (revised 2026-03-22)
"""

import json
import random
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"csd_stats_{TIMESTAMP}.log"

# ── config ─────────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 50000         # total entries to sample
RANDOM_SEED = 42
REPORT_INTERVAL = 2000      # print progress every N entries

# Elements considered "organic" (C must be present; these are allowed)
ORGANIC_ELEMENTS = {
    'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
    'B', 'Si', 'Se', 'As', 'Te'
}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def is_organic(entry) -> bool:
    """Return True if entry is organic (contains C, no metals beyond allowed set)."""
    try:
        mol = entry.molecule
        symbols = {a.atomic_symbol for a in mol.atoms}
        if 'C' not in symbols:
            return False
        # reject if any element outside allowed set
        if symbols - ORGANIC_ELEMENTS:
            return False
        return True
    except Exception:
        return False


def has_missing_hydrogens(entry) -> bool:
    """
    Return True if the entry likely has missing/incomplete hydrogens.

    Conservative heuristic criteria:
      1. hydrogen_treatment is reported as Unknown, Missing, or Omitted
         (these codes indicate H positions are absent or unreliable)
      2. OR: no H atoms at all in the structural model
         (conservative flag; rare fully-halogenated structures may be
          false positives, but these are uncommon in organic CSD entries)

    Note: 'Constr' (riding model) and 'Mixed' are NOT flagged here.
    Constrained H positions are inferred from heavy-atom geometry and
    are generally adequate for most simulation purposes, even though
    they are not independently refined. This conservative definition
    avoids over-counting.
    """
    try:
        ht = entry.hydrogen_treatment
        if str(ht) in ('Unknown', 'Missing', 'Omitted'):
            return True
        # secondary check: no H atoms present
        mol = entry.molecule
        symbols = [a.atomic_symbol for a in mol.atoms]
        if 'H' not in symbols:
            return True
        return False
    except Exception:
        return False


def analyze_entry(entry) -> dict:
    """Extract disorder and H-completeness info from one CSD entry."""
    result = {
        'identifier': entry.identifier,
        'has_disorder': False,
        'has_missing_h': False,
        'hydrogen_treatment': 'Unknown',
        'num_atoms': 0,
        'num_h': 0,
        'error': None,
    }
    try:
        result['has_disorder'] = bool(entry.has_disorder)
        result['hydrogen_treatment'] = str(entry.hydrogen_treatment)
        result['has_missing_h'] = has_missing_hydrogens(entry)

        mol = entry.molecule
        atoms = mol.atoms
        result['num_atoms'] = len(atoms)
        result['num_h'] = sum(1 for a in atoms if a.atomic_symbol == 'H')

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    log("=" * 60)
    log("CSD Statistics Experiment")
    log(f"Sample size: {SAMPLE_SIZE}, seed: {RANDOM_SEED}")
    log("=" * 60)

    # ── connect to CSD ─────────────────────────────────────────────────────────
    log("Connecting to CSD...")
    import ccdc.io
    reader = ccdc.io.EntryReader('CSD')
    total = len(reader)
    log(f"Total CSD entries: {total:,}")

    # ── uniform random sampling by integer index ───────────────────────────────
    log("Generating random integer indices for sampling...")
    rng = random.Random(RANDOM_SEED)

    # Shuffle a pool of indices (6x sample size) to find enough organic 3D entries
    pool_size = min(total, SAMPLE_SIZE * 6)
    pool_indices = rng.sample(range(total), pool_size)
    log(f"  Pool size: {pool_size:,} random indices")

    candidates = []
    scanned = 0
    t0 = time.time()

    for idx in pool_indices:
        scanned += 1
        if scanned % 2000 == 0:
            elapsed = time.time() - t0
            log(f"  Scanned {scanned:,} / {pool_size:,} | candidates: {len(candidates):,} | {elapsed:.1f}s")
        try:
            entry = reader[idx]
            if not entry.has_3d_structure:
                continue
            if not is_organic(entry):
                continue
            candidates.append(idx)
            if len(candidates) >= SAMPLE_SIZE:
                break
        except Exception:
            continue

    log(f"Found {len(candidates):,} organic 3D candidates in {time.time()-t0:.1f}s")

    # ── analyze sample ─────────────────────────────────────────────────────────
    sample = candidates[:SAMPLE_SIZE]
    log(f"\nAnalyzing {len(sample):,} entries...")

    records = []
    t1 = time.time()

    for i, idx in enumerate(sample):
        if i % REPORT_INTERVAL == 0:
            elapsed = time.time() - t1
            log(f"  Progress: {i:,}/{len(sample):,} ({100*i/len(sample):.1f}%) | {elapsed:.1f}s")
        try:
            entry = reader[idx]
            rec = analyze_entry(entry)
            records.append(rec)
        except Exception as e:
            records.append({
                'identifier': str(idx),
                'error': str(e),
                'has_disorder': False,
                'has_missing_h': False,
                'hydrogen_treatment': 'Unknown',
                'num_atoms': 0,
                'num_h': 0,
            })

    log(f"Analysis complete in {time.time()-t1:.1f}s")

    # ── compute statistics ─────────────────────────────────────────────────────
    valid = [r for r in records if not r.get('error')]
    n = len(valid)
    log(f"\nValid records: {n:,} / {len(records):,}")

    # Two primary categories (not mutually exclusive)
    n_disorder  = sum(1 for r in valid if r['has_disorder'])
    n_missing_h = sum(1 for r in valid if r['has_missing_h'])

    # Union and intersection
    n_union        = sum(1 for r in valid if r['has_disorder'] or r['has_missing_h'])
    n_intersection = sum(1 for r in valid if r['has_disorder'] and r['has_missing_h'])

    # Overlap matrix (four mutually exclusive cells)
    n_disorder_only  = sum(1 for r in valid if r['has_disorder'] and not r['has_missing_h'])
    n_missing_h_only = sum(1 for r in valid if not r['has_disorder'] and r['has_missing_h'])
    n_both           = n_intersection
    n_neither        = sum(1 for r in valid if not r['has_disorder'] and not r['has_missing_h'])

    # Sanity check
    assert n_disorder_only + n_missing_h_only + n_both + n_neither == n, \
        "Overlap matrix does not sum to total!"

    # hydrogen_treatment breakdown
    ht_counts = {}
    for r in valid:
        ht = r['hydrogen_treatment']
        ht_counts[ht] = ht_counts.get(ht, 0) + 1

    summary = {
        'timestamp': TIMESTAMP,
        'total_csd': total,
        'sample_size': n,
        # Primary categories
        'n_disorder': n_disorder,
        'n_missing_h': n_missing_h,
        # Union / intersection
        'n_needing_any_preprocessing': n_union,
        'n_needing_both': n_intersection,
        # Percentages
        'pct_disorder': round(100.0 * n_disorder / n, 2) if n else 0,
        'pct_missing_h': round(100.0 * n_missing_h / n, 2) if n else 0,
        'pct_needing_any': round(100.0 * n_union / n, 2) if n else 0,
        'pct_needing_both': round(100.0 * n_intersection / n, 2) if n else 0,
        # Overlap matrix (mutually exclusive cells)
        'overlap_matrix': {
            'disorder_only':  {'n': n_disorder_only,  'pct': round(100.0 * n_disorder_only / n, 2)},
            'missing_h_only': {'n': n_missing_h_only, 'pct': round(100.0 * n_missing_h_only / n, 2)},
            'both':           {'n': n_both,           'pct': round(100.0 * n_both / n, 2)},
            'neither':        {'n': n_neither,         'pct': round(100.0 * n_neither / n, 2)},
        },
        # Hydrogen treatment breakdown (informational)
        'hydrogen_treatment_breakdown': ht_counts,
    }

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Sample size (valid):              {n:,}")
    log(f"")
    log(f"  Category A — Crystallographic disorder:")
    log(f"    n = {n_disorder:,}  ({summary['pct_disorder']:.1f}%)")
    log(f"")
    log(f"  Category B — Missing / incomplete hydrogen:")
    log(f"    n = {n_missing_h:,}  ({summary['pct_missing_h']:.1f}%)")
    log(f"")
    log(f"  Union (A ∪ B) — Any preprocessing needed:")
    log(f"    n = {n_union:,}  ({summary['pct_needing_any']:.1f}%)")
    log(f"")
    log(f"  Overlap matrix (mutually exclusive cells):")
    log(f"    Disorder only:   {n_disorder_only:,}  ({summary['overlap_matrix']['disorder_only']['pct']:.1f}%)")
    log(f"    Missing H only:  {n_missing_h_only:,}  ({summary['overlap_matrix']['missing_h_only']['pct']:.1f}%)")
    log(f"    Both:            {n_both:,}  ({summary['overlap_matrix']['both']['pct']:.1f}%)")
    log(f"    Neither:         {n_neither:,}  ({summary['overlap_matrix']['neither']['pct']:.1f}%)")
    log(f"    Sum check:       {n_disorder_only + n_missing_h_only + n_both + n_neither:,} (should be {n:,})")
    log(f"")
    log(f"  Hydrogen treatment breakdown:")
    for ht, cnt in sorted(ht_counts.items(), key=lambda x: -x[1]):
        log(f"    {ht:20s}: {cnt:5,}  ({100*cnt/n:.1f}%)")

    # ── save results ───────────────────────────────────────────────────────────
    raw_path = RESULTS_DIR / "csd_statistics_raw.json"
    summary_path = RESULTS_DIR / "csd_statistics_summary.json"

    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    log(f"\nRaw data saved: {raw_path}")

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved:  {summary_path}")

    return summary


if __name__ == '__main__':
    try:
        summary = main()
        sys.exit(0)
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
