"""Dump every regression CIF through optimal/random/enumerate modes.

For each case in `tests/unit/test_disorder_regression.CASES` this script
generates ordered replicas with all three solver methods and writes the
resulting structures plus a per-mode summary into
`output/disorder_dump/<case>/`.

Run with the local `molcrys_kit` (after `pip install -e .`):

    python scripts/dump_disorder_modes.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# Ensure the local package wins over any site-packages copy.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import molcrys_kit as mck
from molcrys_kit.analysis.disorder.process import (  # noqa: E402
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.io import write_cif  # noqa: E402
from tests.unit.test_disorder_regression import CASES  # noqa: E402


CIF_DATA_DIR = ROOT / "tests" / "data" / "cif"
OUT_DIR = ROOT / "output" / "disorder_dump"

RANDOM_COUNT = 3
RANDOM_SEED = 0
ENUMERATE_COUNT = 4


def _formula(symbols):
    counts = Counter(symbols)
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def _moiety_summary(crystal):
    formulas = Counter()
    for mol in crystal.molecules:
        formulas[_formula(mol.get_chemical_symbols())] += 1
    return crystal.get_total_nodes(), formulas


def _moiety_str(formulas):
    return ", ".join(f"{k}*{v}" for k, v in sorted(formulas.items()))


def _safe_call(fn, *, label):
    start = time.time()
    try:
        result = fn()
        return result, time.time() - start, None
    except Exception as exc:  # noqa: BLE001
        return None, time.time() - start, f"{type(exc).__name__}: {exc}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = []
    header = (
        f"{'case':<28} {'mode':<10} {'idx':>3} {'atoms':>6}  formulas"
        "    (elapsed_s, error)"
    )
    summary_lines.append(header)
    summary_lines.append("-" * 110)

    for case in CASES:
        case_dir = OUT_DIR / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        cif_path = CIF_DATA_DIR / case.cif
        if not cif_path.is_file():
            line = f"{case.name:<28} {'MISSING':<10} {'-':>3} {'-':>6}  fixture not found"
            print(line)
            summary_lines.append(line)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # optimal x1
            res, dt, err = _safe_call(
                lambda: generate_ordered_replicas_from_disordered_sites(
                    str(cif_path), generate_count=1, method="optimal"
                ),
                label="optimal",
            )
            if err:
                line = (
                    f"{case.name:<28} {'optimal':<10} {0:>3} {'-':>6}  "
                    f"ERROR ({dt:.1f}s, {err})"
                )
                print(line)
                summary_lines.append(line)
            else:
                for i, crystal in enumerate(res):
                    n, formulas = _moiety_summary(crystal)
                    out = case_dir / f"{case.name}_optimal_{i}.cif"
                    write_cif(crystal, str(out))
                    line = (
                        f"{case.name:<28} {'optimal':<10} {i:>3} {n:>6}  "
                        f"{_moiety_str(formulas)}    ({dt:.1f}s)"
                    )
                    print(line)
                    summary_lines.append(line)

            # random xN
            res, dt, err = _safe_call(
                lambda: generate_ordered_replicas_from_disordered_sites(
                    str(cif_path),
                    generate_count=RANDOM_COUNT,
                    method="random",
                    random_seed=RANDOM_SEED,
                ),
                label="random",
            )
            if err:
                line = (
                    f"{case.name:<28} {'random':<10} {0:>3} {'-':>6}  "
                    f"ERROR ({dt:.1f}s, {err})"
                )
                print(line)
                summary_lines.append(line)
            else:
                for i, crystal in enumerate(res):
                    n, formulas = _moiety_summary(crystal)
                    out = case_dir / f"{case.name}_random_{i}.cif"
                    write_cif(crystal, str(out))
                    line = (
                        f"{case.name:<28} {'random':<10} {i:>3} {n:>6}  "
                        f"{_moiety_str(formulas)}    ({dt:.1f}s)"
                    )
                    print(line)
                    summary_lines.append(line)

            # enumerate xN
            res, dt, err = _safe_call(
                lambda: generate_ordered_replicas_from_disordered_sites(
                    str(cif_path),
                    generate_count=ENUMERATE_COUNT,
                    method="enumerate",
                ),
                label="enumerate",
            )
            if err:
                line = (
                    f"{case.name:<28} {'enumerate':<10} {0:>3} {'-':>6}  "
                    f"ERROR ({dt:.1f}s, {err})"
                )
                print(line)
                summary_lines.append(line)
            else:
                for i, crystal in enumerate(res):
                    n, formulas = _moiety_summary(crystal)
                    out = case_dir / f"{case.name}_enumerate_{i}.cif"
                    write_cif(crystal, str(out))
                    line = (
                        f"{case.name:<28} {'enumerate':<10} {i:>3} {n:>6}  "
                        f"{_moiety_str(formulas)}    ({dt:.1f}s)"
                    )
                    print(line)
                    summary_lines.append(line)

        summary_lines.append("")

    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nWrote summary to {summary_path}")
    print(f"Wrote replicas to {OUT_DIR}")


if __name__ == "__main__":
    main()
