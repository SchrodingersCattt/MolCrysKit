"""Small reproducible benchmark for multi-chain fragment construction."""

from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np

# Make the benchmark runnable directly from a source checkout, without relying
# on whichever molcrys-kit version happens to be installed in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molcrys_kit.operations.periodic_chain import build_periodic_chains
from molcrys_kit.structures.periodic_geometry import (
    BoundaryPort,
    ChainSpec,
    ConnectionRule,
    FragmentTemplate,
)


def run(atom_counts=(1000, 10000, 100000)):
    template = FragmentTemplate(
        "repeat",
        ("C", "C"),
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
        (BoundaryPort("in", (0.0, 0.0, 0.0)), BoundaryPort("out", (1.5, 0.0, 0.0))),
        ((0, 1),),
    )
    rule = ConnectionRule(
        "join",
        "repeat",
        "out",
        "repeat",
        "in",
        allowed_image_shifts=((0, 0, 0), (1, 0, 0)),
        distance_range=(1.0, 2.0),
    )
    rows = []
    for count in atom_counts:
        repeats = max(2, int(count) // 4)
        cell = np.diag([max(100.0, 3.0 * float(repeats)), 100.0, 100.0])
        spec = ChainSpec(
            ("repeat",) * repeats,
            chain_count=2,
            chain_centers=((0.0, 0.0, 0.0), (0.0, 0.5, 0.5)),
            target_winding=(1, 0, 0),
            min_distance=0.5,
        )
        t0 = time.perf_counter()
        bundle = build_periodic_chains(
            {"repeat": template}, (rule,), cell, (True, True, True), spec
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "requested_atoms": count,
                "constructed_atoms": len(bundle.atoms),
                "template_atoms": 2,
                "chain_count": 2,
                "seconds": elapsed,
                "graph_edges": len(bundle.graph.edges),
            }
        )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
