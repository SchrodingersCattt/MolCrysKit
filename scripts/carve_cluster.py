#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Carve a finite, H-capped QM cluster from a periodic CIF.

Usage examples
--------------

Chemistry-aware carve seeded on a chosen atom (production):

    python scripts/carve_cluster.py \
        --cif path/to/structure.cif \
        --seed-index 17 \
        --mode bond_shells \
        --freeze-shell 1 \
        --out outputs/cluster

Carve seeded on every atom of an element, auto-grouped into multi-atom
nodes whose centres lie within 3.8 A of one another (e.g. M3 trimers):

    python scripts/carve_cluster.py \
        --cif path/to/structure.cif \
        --seed-element M \
        --mode bond_shells \
        --seed-merge-radius 3.8 \
        --freeze-shell 1 \
        --out outputs/cluster

Diagnostic radial carve (no chemistry, useful for convergence sweeps):

    python scripts/carve_cluster.py \
        --cif path/to/structure.cif \
        --seed-index 0 \
        --mode rcut \
        --rcut 6.5 \
        --freeze-shell 1 \
        --out outputs/rcut

Outputs (per cluster group)
---------------------------

* ``<out>__group<k>.xyz``              XYZ with an extra per-atom flag
                                       column (F=frozen, C=cap H, -=free).
* ``<out>__group<k>.xyz.cluster.json`` Machine-readable provenance
                                       (mode, max_atoms / rcut, seed
                                       indices, kept parent indices,
                                       cut bonds, cap and frozen local
                                       indices, per-cap distances, the
                                       X-H table consulted, optional
                                       convention citation, ...).

The downstream Gaussian / ORCA / Psi4 input writer (out of MCK's scope)
should consume the sidecar JSON to translate ``frozen_local_indices``
into ``%opt=ModRedundant`` lines or ``-1 X Y Z`` constraints.
"""

import sys
from molcrys_kit.cli.operate_cmd import cluster


def main() -> int:
    """Backward-compatible wrapper for ``mck operate cluster``."""
    argv = list(sys.argv[1:])
    if "--cif" in argv:
        idx = argv.index("--cif")
        argv[idx] = "--input"
    if "--out" in argv:
        idx = argv.index("--out")
        argv[idx] = "--output"
    cluster.main(args=argv, prog_name="carve_cluster.py", standalone_mode=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
