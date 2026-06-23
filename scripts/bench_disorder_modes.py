#!/usr/bin/env python
"""Benchmark coupled vs decoupled disorder enumeration.

This script is intentionally not part of CI.  It measures the expensive
pieces of the disorder pipeline on representative CIFs and prints a compact
table suitable for choosing solver caps.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import molcrys_kit as mck
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder  # noqa: E402
from molcrys_kit.analysis.disorder.solver import DisorderSolver  # noqa: E402
from molcrys_kit.io.cif import _pymatgen_cif_parser, scan_cif_disorder  # noqa: E402


CIF_DATA_DIR = ROOT / "tests" / "data" / "cif"
DEFAULT_CASES = [
    "1-HTP.cif",
    "DAP-4.cif",
    "DAI-X1.cif",
    "TILPEN.cif",
    "PAP-4.cif",
]


def _load(path: Path):
    info = scan_cif_disorder(str(path))
    structure = _pymatgen_cif_parser(str(path)).parse_structures()[0]
    return info, structure.lattice.matrix


def _bench_case(path: Path, *, coupled: bool, limit: int = 32) -> dict:
    info, lattice = _load(path)

    t0 = time.perf_counter()
    builder = DisorderGraphBuilder(info, lattice, coupled=coupled)
    graph = builder.build()
    graph_s = time.perf_counter() - t0

    solver = DisorderSolver(info, graph, lattice, coupled=coupled)
    solver.atom_groups = []
    t1 = time.perf_counter()
    solver._identify_atom_groups()
    alternatives = solver._build_decision_alternatives()
    alts_s = time.perf_counter() - t1

    counts = [len(alts) for alts in alternatives]
    product = math.prod(counts) if counts else 0
    peak = max(counts, default=0)

    t2 = time.perf_counter()
    limited = solver._solve_enumerate(limit)
    enum_limit_s = time.perf_counter() - t2

    enum_all_s = None
    all_count = None
    if product and product <= solver._MAX_ENUMERATED_STRUCTURES:
        t3 = time.perf_counter()
        all_structures = solver._solve_enumerate(None)
        enum_all_s = time.perf_counter() - t3
        all_count = len(all_structures)

    return {
        "case": path.name,
        "coupled": coupled,
        "atoms": len(info.labels),
        "graph_s": graph_s,
        "alts_s": alts_s,
        "components": len(counts),
        "peak_alts": peak,
        "product": product,
        "enum_limit_s": enum_limit_s,
        "limit_count": len(limited),
        "enum_all_s": enum_all_s,
        "all_count": all_count,
    }


def _fmt_seconds(value):
    return "-" if value is None else f"{value:.2f}"


def main(argv: list[str]) -> int:
    cases = argv or DEFAULT_CASES
    rows = []
    for case in cases:
        path = Path(case)
        if not path.is_file():
            path = CIF_DATA_DIR / case
        if not path.is_file():
            print(f"missing CIF fixture: {case}", file=sys.stderr)
            return 2
        for coupled in (True, False):
            rows.append(_bench_case(path, coupled=coupled))

    header = (
        f"{'case':<14} {'mode':<10} {'atoms':>6} {'graph':>7} {'alts':>7} "
        f"{'comp':>5} {'peak':>5} {'prod':>8} {'enum32':>7} {'n32':>5} "
        f"{'enumAll':>7} {'nAll':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        mode = "coupled" if row["coupled"] else "decoupled"
        print(
            f"{row['case']:<14} {mode:<10} {row['atoms']:>6} "
            f"{row['graph_s']:>7.2f} {row['alts_s']:>7.2f} "
            f"{row['components']:>5} {row['peak_alts']:>5} "
            f"{row['product']:>8} {row['enum_limit_s']:>7.2f} "
            f"{row['limit_count']:>5} {_fmt_seconds(row['enum_all_s']):>7} "
            f"{str(row['all_count'] or '-'):>5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
