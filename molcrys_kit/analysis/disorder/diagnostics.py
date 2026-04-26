"""
Post-resolution valence-completeness diagnostics for the disorder solver.

Provides :func:`check_valence_completeness`, a lightweight function that
compares the number of H atoms actually bonded to each heavy atom in the
resolved structure against the *expected* count derived from standard
coordination chemistry.  Any mismatch is returned as a
:class:`ValenceDiagnostic` record and should be surfaced as a warning by
the caller.

Design principles
-----------------
* **Non-invasive**: this module never modifies the structure.  It only reads
  atom positions, symbols, and bond topology.
* **Warn, never fix**: raising an explicit warning is far better than a
  silent geometry error or a post-hoc numerical artifact.
* **Conservative expectations**: when the expected count is ambiguous (e.g.,
  protonation-state-dependent) we report both the observed count and the
  expected range rather than asserting a single value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from molcrys_kit.analysis.disorder.info import DisorderInfo
    from molcrys_kit.structures.crystal import MolecularCrystal

logger = logging.getLogger(__name__)

# Expected number of *bonded H atoms* for isolated (non-framework) heavy
# atoms under standard protonation.  Only elements that frequently appear
# as NH4+/NH3/H2O/etc. in organic/hybrid perovskite CIFs are listed;
# atoms not in this table are silently skipped.
#
# Each value is a tuple (min_expected, max_expected).  If the observed
# count falls outside this range, a ValenceDiagnostic is emitted.
_EXPECTED_H_RANGE: dict[str, tuple[int, int]] = {
    "N": (3, 4),   # NH3 = 3, NH4+ = 4; both acceptable
    "O": (0, 2),   # isolated O ranges from bare O2- (0) to H2O (2)
}

# Cutoff for identifying the "isolated" centre pattern:
# a heavy atom that carries partial-occupancy H should not be bonded
# to any non-H heavy atom within this distance (A).
_HEAVY_BOND_CUTOFF: float = 1.8

# X-H bond detection cutoff (A) for counting bonded H in the resolved
# structure.  Generous to catch both neutron (~1.0 A) and X-ray (~0.85 A)
# positions.
_XH_BOND_CUTOFF: float = 1.3


@dataclass
class ValenceDiagnostic:
    """A single H-count mismatch for one atom in the resolved structure.

    Attributes
    ----------
    atom_label:
        CIF label of the heavy atom (e.g. ``"N1"``).
    atom_idx:
        Zero-based index in the resolved structure's atom array.
    element:
        Chemical symbol (e.g. ``"N"``).
    expected_min:
        Lower bound of the expected H count.
    expected_max:
        Upper bound of the expected H count.
    observed:
        Actual number of bonded H atoms in the resolved structure.
    reason:
        Short human-readable description of why the check triggered.
    """

    atom_label: str
    atom_idx: int
    element: str
    expected_min: int
    expected_max: int
    observed: int
    reason: str = field(default="")

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.element} atom '{self.atom_label}' (idx={self.atom_idx}): "
            f"expected {self.expected_min}-{self.expected_max} bonded H, "
            f"got {self.observed}"
            + (f" [{self.reason}]" if self.reason else "")
        )


def check_valence_completeness(
    crystal: "MolecularCrystal",
    info: "DisorderInfo",
    *,
    heavy_bond_cutoff: float = _HEAVY_BOND_CUTOFF,
    xh_bond_cutoff: float = _XH_BOND_CUTOFF,
) -> List[ValenceDiagnostic]:
    """Check whether resolved H counts match expected valence ranges.

    Scans every heavy atom satisfying the *isolated-centre* criterion
    (no heavy non-H neighbour within *heavy_bond_cutoff*) whose element
    has an entry in :data:`_EXPECTED_H_RANGE`.  For each such atom the
    function counts how many H/D atoms in the resolved structure are within
    *xh_bond_cutoff* and compares against the expected range.

    Parameters
    ----------
    crystal:
        Resolved :class:`~molcrys_kit.structures.crystal.MolecularCrystal`
        returned by :meth:`DisorderSolver.solve`.
    info:
        :class:`~molcrys_kit.analysis.disorder.info.DisorderInfo` for the
        *input* (pre-resolution) structure, used only to map atom indices
        back to CIF labels.
    heavy_bond_cutoff:
        Distance (A) within which another heavy atom disqualifies the
        candidate centre from being "isolated".
    xh_bond_cutoff:
        Distance (A) used to count bonded H atoms.

    Returns
    -------
    list[ValenceDiagnostic]
        Empty list means no issues were found.  The caller should forward
        any non-empty result to ``logger.warning``.
    """
    try:
        ase_atoms = crystal.to_ase()
    except Exception:  # pragma: no cover
        return []

    symbols = np.array(ase_atoms.get_chemical_symbols())
    positions = ase_atoms.get_positions()
    cell = np.array(ase_atoms.cell[:])
    pbc = ase_atoms.get_pbc()

    def _mic_dist_matrix(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """Return (n_a, n_b) pairwise distance matrix under MIC."""
        diff = pos_a[:, None, :] - pos_b[None, :, :]  # (n_a, n_b, 3)
        if pbc.any() and cell is not None and np.linalg.det(cell) > 0:
            inv_cell = np.linalg.inv(cell)
            frac_diff = diff @ inv_cell
            frac_diff -= np.round(frac_diff)
            diff = frac_diff @ cell
        return np.linalg.norm(diff, axis=2)

    heavy_mask = ~np.isin(symbols, ["H", "D"])
    h_mask = np.isin(symbols, ["H", "D"])

    heavy_indices = np.where(heavy_mask)[0]
    h_indices = np.where(h_mask)[0]

    if not len(heavy_indices) or not len(h_indices):
        return []

    heavy_pos = positions[heavy_indices]
    h_pos = positions[h_indices]

    d_hh = _mic_dist_matrix(heavy_pos, heavy_pos)   # (n_heavy, n_heavy)
    d_hH = _mic_dist_matrix(heavy_pos, h_pos)        # (n_heavy, n_H)

    def _label(resolved_idx: int) -> str:
        sym = symbols[resolved_idx]
        for k, lbl in enumerate(info.labels):
            if info.symbols[k] == sym:
                return lbl
        return f"{sym}_{resolved_idx}"

    diagnostics: list[ValenceDiagnostic] = []

    for local_i, global_i in enumerate(heavy_indices):
        elem = symbols[global_i]
        if elem not in _EXPECTED_H_RANGE:
            continue

        exp_min, exp_max = _EXPECTED_H_RANGE[elem]

        # Isolation check: no OTHER heavy atom within heavy_bond_cutoff
        other_heavy_dists = d_hh[local_i].copy()
        other_heavy_dists[local_i] = np.inf  # exclude self
        if np.min(other_heavy_dists) < heavy_bond_cutoff:
            continue  # bonded to another heavy atom -- not an isolated centre

        n_H_bonded = int(np.sum(d_hH[local_i] < xh_bond_cutoff))

        if not (exp_min <= n_H_bonded <= exp_max):
            diagnostics.append(
                ValenceDiagnostic(
                    atom_label=_label(global_i),
                    atom_idx=int(global_i),
                    element=elem,
                    expected_min=exp_min,
                    expected_max=exp_max,
                    observed=n_H_bonded,
                    reason=(
                        "isolated centre has unexpected H count; "
                        "possible disorder-resolution artefact"
                    ),
                )
            )

    return diagnostics
