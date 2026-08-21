"""Benchmark bounded-memory nanocluster selection on a million-point grid."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402

from molcrys_kit.operations import (  # noqa: E402
    DEFAULT_NANOCLUSTER_BATCH_SIZE,
    NanoClusterCarver,
    NanoShape,
)
from molcrys_kit.structures import MolecularCrystal  # noqa: E402


def _process_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-width", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_NANOCLUSTER_BATCH_SIZE)
    args = parser.parse_args()
    if args.half_width < 1:
        parser.error("--half-width must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    lattice = np.eye(3)
    source = MolecularCrystal(
        lattice,
        [Atoms("He", positions=[[0.5, 0.5, 0.5]], cell=lattice, pbc=True)],
    )
    half_width = float(args.half_width)
    bounds = np.repeat([[-half_width, half_width]], 3, axis=0)
    field_calls = 0
    max_field_batch = 0
    peak_rss = _process_rss_bytes()
    baseline_rss = peak_rss

    def squared_radius(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        nonlocal field_calls, max_field_batch, peak_rss
        field_calls += 1
        max_field_batch = max(max_field_batch, len(x))
        current_rss = _process_rss_bytes()
        if current_rss is not None:
            peak_rss = current_rss if peak_rss is None else max(peak_rss, current_rss)
        return x * x + y * y + z * z

    shape = NanoShape(squared_radius, bounds, name="benchmark_cube")
    tracemalloc.start()
    start = time.perf_counter()
    result = NanoClusterCarver(source, batch_size=args.batch_size).carve(
        shape,
        topology_unit="unit_cell",
        target_units=1,
    )
    elapsed_seconds = time.perf_counter() - start
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metadata = result.metadata["nanocluster"]

    report = {
        "grid_candidates": metadata["grid_candidate_count"],
        "bounded_candidates": metadata["candidate_count"],
        "field_calls": field_calls,
        "max_field_batch": max_field_batch,
        "batch_size": args.batch_size,
        "selected_units": metadata["selected_unit_count"],
        "elapsed_seconds": elapsed_seconds,
        "tracemalloc_peak_mib": traced_peak / (1024 * 1024),
        "baseline_rss_mib": None if baseline_rss is None else baseline_rss / (1024 * 1024),
        "sampled_peak_rss_mib": None if peak_rss is None else peak_rss / (1024 * 1024),
        "sampled_rss_delta_mib": (
            None
            if baseline_rss is None or peak_rss is None
            else (peak_rss - baseline_rss) / (1024 * 1024)
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
