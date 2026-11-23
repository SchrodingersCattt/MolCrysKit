"""Modest CPU timings for all-pairs IDPP on small synthetic paths.

This is a local scaling probe, not a performance promise. It times the
existing independent-image refiner on a few atom counts and frame counts.
"""

from __future__ import annotations

import json
import time

import numpy as np
from ase import Atoms

from molcrys_kit.operations.idpp import refine_idpp_path


def _path(atom_count: int, frame_count: int) -> list[Atoms]:
    cell = np.diag([12.0, 12.0, 12.0])
    start = np.zeros((atom_count, 3))
    start[:, 0] = np.linspace(1.0, 8.0, atom_count)
    start[:, 1] = 2.0
    end = start.copy()
    end[:, 1] = 4.0
    images = []
    for frame in range(frame_count):
        weight = frame / (frame_count - 1)
        images.append(
            Atoms(
                symbols=["C"] * atom_count,
                positions=(1.0 - weight) * start + weight * end,
                cell=cell,
                pbc=True,
            )
        )
    return images


def time_idpp_grid(
    atom_counts=(4, 8),
    frame_counts=(3, 5),
) -> list[dict[str, object]]:
    """Return elapsed seconds for each all-pairs IDPP case."""
    rows = []
    for atom_count in atom_counts:
        for frame_count in frame_counts:
            images = _path(int(atom_count), int(frame_count))
            started = time.perf_counter()
            result = refine_idpp_path(images, max_iterations=5)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "atom_count": int(atom_count),
                    "frame_count": int(frame_count),
                    "pair_count": int(atom_count * (atom_count - 1) / 2),
                    "interior_frames": int(frame_count - 2),
                    "seconds": elapsed,
                    "converged": bool(result.converged),
                }
            )
    return rows


def main() -> None:
    print(json.dumps(time_idpp_grid(), indent=2))


if __name__ == "__main__":
    main()
