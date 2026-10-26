"""Independent graph-constrained, whole-molecule periodic matching.

Only one lattice translation is permitted per molecule. No independent
atom minimum-image wrapping, rigid fitting, rotation or cell scaling occurs.
Geometrical candidates are pruned before graph-isomorphism search.
"""

from __future__ import annotations

from collections import Counter
import itertools
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import config


def lattice_from_parameters(lengths, angles):
    a, b, c = lengths
    alpha, beta, gamma = np.radians(angles)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * np.sin(gamma), 0.0],
            [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)],
        ]
    )


def cell_parameters(lattice):
    lattice = np.asarray(lattice)
    lengths = np.linalg.norm(lattice, axis=1)
    angles = [
        np.degrees(
            np.arccos(
                np.clip(np.dot(lattice[i], lattice[j]) / lengths[i] / lengths[j], -1, 1)
            )
        )
        for i, j in ((1, 2), (0, 2), (0, 1))
    ]
    return lengths, np.asarray(angles)


def graph(molecule, chemistry=False):
    g = nx.Graph()
    for i, atom in enumerate(molecule["atoms"]):
        identity = (atom["symbol"],)
        if chemistry:
            identity += tuple(
                atom.get(k) for k in ("formal_charge", "isotope", "stereo")
            )
        g.add_node(i, identity=identity)
    for edge in molecule.get("bonds", []):
        g.add_edge(
            edge[0], edge[1], kind=edge[2] if chemistry and len(edge) > 2 else None
        )
    return g


def signature(molecule):
    return tuple(sorted(Counter(a["symbol"] for a in molecule["atoms"]).items())), len(
        molecule.get("bonds", [])
    )


def molecule_mapping(
    reference, candidate, lattice, *, chemistry=False, occupancy=True, tolerance=None
):
    """Return (candidate indices, Cartesian residuals), or None."""
    tolerance = config()["position_A"] if tolerance is None else tolerance
    if signature(reference) != signature(candidate):
        return None
    lattice = np.asarray(lattice)
    inv = np.linalg.inv(lattice)
    left = np.asarray([a["xyz"] for a in reference["atoms"]])
    right = np.asarray([a["xyz"] for a in candidate["atoms"]])
    if not len(left):
        return None
    base = np.rint((left.mean(0) - right.mean(0)) @ inv).astype(int)
    # Candidate translations surrounding the centroid are tested in Cartesian
    # distance; this remains valid in a skewed cell without componentwise MIC.
    for offset in itertools.product((-1, 0, 1), repeat=3):
        shifted = right + (base + offset) @ lattice
        # A valid bijection with every residual <= tolerance necessarily has
        # a centroid residual <= tolerance. Prune before the O(n_atoms^2)
        # candidate matrix, without weakening the atom/graph check.
        if np.linalg.norm(left.mean(0) - shifted.mean(0)) > tolerance:
            continue
        distance = np.linalg.norm(left[:, None, :] - shifted[None, :, :], axis=2)
        allowed = distance <= tolerance
        for i, a in enumerate(reference["atoms"]):
            for j, b in enumerate(candidate["atoms"]):
                if a["symbol"] != b["symbol"]:
                    allowed[i, j] = False
                if (
                    occupancy
                    and abs(a.get("occupancy", 1.0) - b.get("occupancy", 1.0))
                    > config()["occupancy"]
                ):
                    allowed[i, j] = False
                if chemistry and any(
                    a.get(k) != b.get(k) for k in ("formal_charge", "isotope", "stereo")
                ):
                    allowed[i, j] = False
        if not allowed.any(1).all() or not allowed.any(0).all():
            continue
        gl, gr = graph(reference, chemistry), graph(candidate, chemistry)
        for i in gl:
            gl.nodes[i]["index"] = i
            gr.nodes[i]["index"] = i
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            gl,
            gr,
            node_match=lambda a, b: (
                a["identity"] == b["identity"] and allowed[a["index"], b["index"]]
            ),
            edge_match=lambda a, b: a["kind"] == b["kind"],
        )
        for mapping in matcher.isomorphisms_iter():
            order = [mapping[i] for i in range(len(left))]
            return order, np.linalg.norm(left - shifted[order], axis=1)
    return None


def compare_molecules(reference, candidate, lattice, *, chemistry=False):
    if len(reference) != len(candidate):
        return {
            "status": "mismatch",
            "code": "molecule_count",
            "n_molecules": len(candidate),
        }
    if not reference:
        return {"status": "indeterminate", "code": "empty_reference"}
    n = len(reference)
    scores = np.full((n, n), 1e9)
    matches = {}
    for i, left in enumerate(reference):
        for j, right in enumerate(candidate):
            if signature(left) != signature(right):
                continue
            match = molecule_mapping(left, right, lattice, chemistry=chemistry)
            if match is not None:
                matches[i, j] = match
                scores[i, j] = float(match[1].max())
    rows, cols = linear_sum_assignment(scores)
    if any((i, j) not in matches for i, j in zip(rows, cols)):
        return {"status": "mismatch", "code": "whole_molecule_geometry_or_graph"}
    residuals = np.concatenate([matches[i, j][1] for i, j in zip(rows, cols)])
    return {
        "status": "pass",
        "code": "whole_molecule_match",
        "max_A": float(residuals.max()),
        "rmsd_A": float(np.sqrt(np.mean(residuals**2))),
        "n_molecules": n,
        "n_atoms": len(residuals),
    }


def parse_operation(xyz):
    """Parse crystallographic affine xyz without eval or the target parser."""
    from fractions import Fraction
    import re

    matrix = np.zeros((3, 3))
    shift = np.zeros(3)
    for row, expression in enumerate(xyz.lower().replace(" ", "").split(",")):
        for token in re.findall(r"[+-]?[^+-]+", expression):
            axis = next((x for x in "xyz" if x in token), None)
            if axis:
                coefficient = token.replace(axis, "").rstrip("*")
                matrix[row, "xyz".index(axis)] += float(
                    Fraction(
                        coefficient
                        if coefficient not in ("", "+", "-")
                        else coefficient + "1"
                    )
                )
            else:
                shift[row] += float(Fraction(token))
    return matrix, shift


def independent_representatives(molecules, operations, lattice):
    representatives = []
    inv = np.linalg.inv(lattice)
    for molecule in molecules:
        duplicate = False
        positions = np.asarray([a["xyz"] for a in molecule["atoms"]])
        for existing in representatives:
            if signature(molecule) != signature(existing):
                continue
            for rotation, shift in operations:
                transformed = {
                    **molecule,
                    "atoms": [
                        dict(a, xyz=p.tolist())
                        for a, p in zip(
                            molecule["atoms"],
                            (positions @ inv @ rotation.T + shift) @ lattice,
                        )
                    ],
                }
                if (
                    molecule_mapping(existing, transformed, lattice, occupancy=False)
                    is not None
                ):
                    duplicate = True
                    break
            if duplicate:
                break
        if not duplicate:
            representatives.append(molecule)
    return representatives
