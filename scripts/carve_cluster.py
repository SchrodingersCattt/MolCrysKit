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


def _translate_legacy_args(argv: list[str]) -> list[str]:
    """Translate legacy script flags to ``mck operate cluster`` arguments."""
    translated: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--cif":
            if idx + 1 >= len(argv):
                translated.append(arg)
                idx += 1
                continue
            translated.append(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--cif="):
            translated.append(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--out":
            translated.append("--output")
            idx += 1
            continue
        if arg.startswith("--out="):
            translated.append("--output=" + arg.split("=", 1)[1])
            idx += 1
            continue
        translated.append(arg)
        idx += 1
    return translated


def main() -> None:
    """Backward-compatible wrapper for ``mck operate cluster``."""
    argv = _translate_legacy_args(list(sys.argv[1:]))
    cluster.main(args=argv, prog_name="carve_cluster.py", standalone_mode=True)


if __name__ == "__main__":
    main()
