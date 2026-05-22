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
        --shells 1 \
        --freeze-shell 1 \
        --out outputs/cluster

Carve seeded on every atom of an element, auto-grouped into multi-atom
nodes whose centres lie within 3.8 A of one another (e.g. M3 trimers):

    python scripts/carve_cluster.py \
        --cif path/to/structure.cif \
        --seed-element M \
        --mode bond_shells \
        --shells 1 \
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
                                       (mode, n_shells / rcut, seed
                                       indices, kept parent indices,
                                       cut bonds, cap and frozen local
                                       indices, per-cap distances, the
                                       X-H table consulted, optional
                                       convention citation, ...).

The downstream Gaussian / ORCA / Psi4 input writer (out of MCK's scope)
should consume the sidecar JSON to translate ``frozen_local_indices``
into ``%opt=ModRedundant`` lines or ``-1 X Y Z`` constraints.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.io.output import write_xyz_with_freeze
from molcrys_kit.operations import ClusterCarver, carve_cluster


def _parse_seed(args: argparse.Namespace):
    if args.seed_element is not None and args.seed_index is not None:
        raise SystemExit(
            "Specify --seed-element OR --seed-index, not both."
        )
    if args.seed_element is not None:
        return args.seed_element
    if args.seed_index is not None:
        return list(args.seed_index)
    raise SystemExit(
        "Specify a seed via --seed-element ELEMENT or --seed-index I [I ...]."
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Carve a QM cluster model from a periodic CIF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cif",
        required=True,
        help="Path to the input CIF (already disorder-resolved for production).",
    )
    parser.add_argument(
        "--mode",
        choices=("bond_shells", "rcut"),
        default="bond_shells",
        help="Carving mode. bond_shells = chemistry-aware (production); "
        "rcut = radial diagnostic.",
    )
    parser.add_argument(
        "--seed-element",
        type=str,
        default=None,
        help='Seed on every atom of this element, e.g. "Zn".',
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        nargs="+",
        default=None,
        help="Explicit zero-based global atom indices to use as seeds "
        "(refer to crystal.to_ase() ordering).",
    )
    parser.add_argument(
        "--shells",
        type=int,
        default=1,
        help="Number of cut-boundary layers crossed beyond the seed "
        "(bond_shells mode only).",
    )
    parser.add_argument(
        "--rcut",
        type=float,
        default=None,
        help="Radial cutoff in Angstrom (rcut mode only).",
    )
    parser.add_argument(
        "--freeze-shell",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="0 = none, 1 = cap H + keepers of every cut, 2 = also "
        "one heavy-atom layer inward.",
    )
    parser.add_argument(
        "--cap-distance",
        type=float,
        default=None,
        help="Uniform cap distance (Angstrom) for every cut.  If omitted, "
        "per-element X-H bond lengths are looked up from "
        "molcrys_kit.constants.config.BOND_LENGTHS (C-H=1.09, N-H=1.01, "
        "O-H=0.96, ...), the same table used by add_hydrogens.",
    )
    parser.add_argument(
        "--cap-bond-length",
        action="append",
        default=[],
        metavar="ELEM=DIST",
        help="Override one entry of the BOND_LENGTHS table, e.g. "
        "'--cap-bond-length C=1.10 --cap-bond-length N=1.00'.  May be "
        "passed multiple times.",
    )
    parser.add_argument(
        "--seed-merge-radius",
        type=float,
        default=0.0,
        help="Distance threshold (Angstrom) for grouping adjacent seeds "
        "into one cluster.  Default 0.0 means 'no auto-grouping -- each "
        "seed is its own cluster'.  Set to the diameter of a multi-atom "
        "node (paddle-wheel ~3.0 A, trimer ~3.8 A, ...) to get one "
        "cluster per node group.",
    )
    parser.add_argument(
        "--convention-reference",
        type=str,
        default="",
        help="Free-text citation block stamped into the sidecar JSON's "
        "convention_reference field.  Use it to record the DOI(s) that "
        "justify your parameter choices for this system.",
    )
    parser.add_argument(
        "--no-stop-at-non-seed-metals",
        dest="stop_at_non_seed_metals",
        action="store_false",
        default=True,
        help="Disable the implicit metal-boundary rule.  By default the "
        "carver treats bonds reaching a non-seed metal as a cut (necessary "
        "for frameworks where two metal nodes are bridged through "
        "non-C-C paths); disable for diagnostic carves that should walk "
        "through every metal site.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output stem.  Each cluster group writes <out>__group<k>.xyz "
        "plus its sidecar JSON.",
    )

    args = parser.parse_args(argv)
    seed = _parse_seed(args)

    if args.mode == "rcut" and args.rcut is None:
        raise SystemExit("--rcut is required when --mode rcut.")

    cap_overrides: dict = {}
    for entry in args.cap_bond_length:
        if "=" not in entry:
            raise SystemExit(
                f"--cap-bond-length expects ELEM=DIST, got '{entry}'."
            )
        elem, dist_str = entry.split("=", 1)
        cap_overrides[elem.strip()] = float(dist_str)

    crystal = read_mol_crystal(args.cif)
    carver = ClusterCarver(crystal, seed_merge_radius=args.seed_merge_radius)
    if args.mode == "bond_shells":
        clusters = carver.carve_bond_shells(
            seed,
            n_shells=args.shells,
            freeze_shell=args.freeze_shell,
            cap_distance=args.cap_distance,
            cap_bond_lengths=cap_overrides or None,
            parent_label=os.path.abspath(args.cif),
            convention_reference=args.convention_reference,
            stop_at_non_seed_metals=args.stop_at_non_seed_metals,
        )
    else:
        clusters = carver.carve_rcut(
            seed,
            rcut=args.rcut,
            freeze_shell=args.freeze_shell,
            cap_distance=args.cap_distance,
            cap_bond_lengths=cap_overrides or None,
            parent_label=os.path.abspath(args.cif),
            convention_reference=args.convention_reference,
        )

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for idx, cluster in enumerate(clusters):
        xyz_path = f"{args.out}__group{idx}.xyz"
        sidecar = write_xyz_with_freeze(cluster, xyz_path)
        print(
            f"[group {idx}] mode={cluster.provenance.mode} "
            f"natoms={len(cluster)} "
            f"frozen={len(cluster.frozen_local_indices)} "
            f"caps={len(cluster.cap_local_indices)} "
            f"-> {xyz_path}\n            sidecar: {sidecar}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
