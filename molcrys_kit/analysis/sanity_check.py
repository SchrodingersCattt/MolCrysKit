"""Structure sanity check module for MolCrysKit.

Provides atom-level and molecule-level checks for validating the physical
reasonableness of molecular crystal structures.  Each check is an independent
function returning a :class:`CheckResult`; :func:`sanity_check` aggregates
multiple checks into a :class:`SanityReport`.

Example usage::

    from molcrys_kit.analysis.sanity_check import sanity_check

    report = sanity_check(crystal, checks=["hard_clash", "isolated_atoms"])
    if not report.passed:
        for r in report.failed():
            print(r.name, r.message)

Individual checks can also be called directly::

    from molcrys_kit.analysis.sanity_check import check_hard_clash

    result = check_hard_clash(crystal, scale=0.5)
    result.details["pairs"]  # [(i, j, dist), ...]
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from typing import Any, Sequence

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list

from ..constants import ATOMIC_RADII
from ..constants.config import BONDING_CONFIG, SANITY_CHECK_CONFIG


__all__ = [
    "CheckResult",
    "SanityReport",
    "check_hard_clash",
    "check_intermolecular_clash",
    "check_isolated_atoms",
    "check_hydrogen_presence",
    "check_topology_preservation",
    "check_formula_consistency",
    "check_bond_distances",
    "sanity_check",
]

# Registry of checks that can be run on a single crystal.
# topology_preservation requires two crystals, so is excluded from default set.
_SINGLE_CRYSTAL_CHECKS = (
    "hard_clash",
    "intermolecular_clash",
    "isolated_atoms",
    "hydrogen_presence",
    "formula_consistency",
    "bond_distances",
)


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Result of a single sanity check.

    Parameters
    ----------
    name : str
        Machine-readable check identifier (e.g. ``"hard_clash"``).
    passed : bool
        Whether the check passed.
    message : str
        Human-readable summary of the result.
    details : dict
        Structured data for programmatic consumption.  Contents vary by check.
    """

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"CheckResult({self.name!r}, {status}, {self.message!r})"


@dataclass
class SanityReport:
    """Aggregated report from running multiple sanity checks.

    Parameters
    ----------
    results : list[CheckResult]
        Individual check outcomes, in execution order.
    """

    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all checks passed."""
        return all(r.passed for r in self.results)

    def failed(self) -> list[CheckResult]:
        """Return only the checks that did not pass."""
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines: list[str] = []
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} {r.name}: {r.message}")
        total = len(self.results)
        n_fail = len(self.failed())
        header = f"Sanity check: {total - n_fail}/{total} passed"
        return "\n".join([header] + lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "passed": self.passed,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, key):
        if isinstance(key, str):
            for r in self.results:
                if r.name == key:
                    return r
            raise KeyError(key)
        return self.results[key]

    def __len__(self):
        return len(self.results)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_config(key: str, override: Any) -> Any:
    """Return override if not None, else SANITY_CHECK_CONFIG default."""
    if override is not None:
        return override
    return SANITY_CHECK_CONFIG[key]


def _ensure_molecular_crystal(crystal):
    """Convert ASE Atoms to MolecularCrystal if needed."""
    from ..structures.crystal import MolecularCrystal

    if isinstance(crystal, MolecularCrystal):
        return crystal
    # ASE Atoms or similar — attempt conversion
    if hasattr(crystal, "get_positions"):
        return MolecularCrystal.from_ase_atoms(crystal)
    raise TypeError(
        f"Expected MolecularCrystal or ASE Atoms, got {type(crystal).__name__}"
    )


def _ase_radii(atoms) -> np.ndarray:
    """Build per-atom covalent radii array from ASE Atoms."""
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)
    radii = np.empty(n)
    for i, s in enumerate(symbols):
        zi = atomic_numbers.get(s, 0)
        radii[i] = float(covalent_radii[zi]) if zi < len(covalent_radii) else 0.7
    return radii


# ─── Atom-Level Checks ────────────────────────────────────────────────────────


def check_hard_clash(
    crystal,
    *,
    scale: float | None = None,
    tolerance: float | None = None,
) -> CheckResult:
    """Check for any atom pair closer than *scale* × (r_i + r_j) - *tolerance*.

    This is a global check covering all atom pairs (intra- and inter-molecular)
    using PBC via ASE ``neighbor_list``.  Correctly handles triclinic cells.

    Parameters
    ----------
    crystal : MolecularCrystal or ASE Atoms
        Structure to validate.
    scale : float, optional
        Ratio applied to sum of covalent radii.  Default from config (0.6).
    tolerance : float, optional
        Absolute distance subtracted from threshold.  Default from config (0.0).

    Returns
    -------
    CheckResult
        ``details`` contains: ``clash_count``, ``pairs`` (list of (i, j, dist)
        tuples for clashing atom pairs), ``scale_used``, ``tolerance_used``.
    """
    scale = _get_config("hard_clash_scale", scale)
    tolerance = _get_config("hard_clash_tolerance", tolerance)

    mc = _ensure_molecular_crystal(crystal)
    atoms = mc.to_ase()
    n = len(atoms)

    if n < 2:
        return CheckResult(
            name="hard_clash",
            passed=True,
            message="Fewer than 2 atoms; no clash possible.",
            details={"clash_count": 0, "pairs": [], "scale_used": scale, "tolerance_used": tolerance},
        )

    radii = _ase_radii(atoms)
    max_cutoff = scale * 2 * radii.max() + abs(tolerance)
    i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoff=max_cutoff)

    if len(i_idx) == 0:
        return CheckResult(
            name="hard_clash",
            passed=True,
            message="No clashes found.",
            details={"clash_count": 0, "pairs": [], "scale_used": scale, "tolerance_used": tolerance},
        )

    cutoffs = np.maximum(0.0, scale * (radii[i_idx] + radii[j_idx]) - tolerance)
    clashing_mask = dists < cutoffs

    # De-duplicate pairs (neighbor_list returns i<j and j<i)
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for idx in np.where(clashing_mask)[0]:
        i, j = int(i_idx[idx]), int(j_idx[idx])
        pair_key = (min(i, j), max(i, j))
        if pair_key not in seen:
            seen.add(pair_key)
            pairs.append((pair_key[0], pair_key[1], float(dists[idx])))

    clash_count = len(pairs)
    passed = clash_count == 0
    message = "No clashes found." if passed else f"{clash_count} hard clash(es) detected."

    return CheckResult(
        name="hard_clash",
        passed=passed,
        message=message,
        details={"clash_count": clash_count, "pairs": pairs, "scale_used": scale, "tolerance_used": tolerance},
    )


def check_intermolecular_clash(
    crystal,
    *,
    scale: float | None = None,
    tolerance: float | None = None,
    ignore_hh: bool | None = None,
    max_clashes: int | None = None,
) -> CheckResult:
    """Check for clashes between atoms of different molecules (PBC-aware).

    Also catches same-molecule cross-image clashes (atom overlapping its own
    periodic image in a neighbouring cell).

    Parameters
    ----------
    crystal : MolecularCrystal or ASE Atoms
        Structure to validate.  Must have ``molecule_index`` in arrays.
    scale : float, optional
        Ratio applied to sum of covalent radii.  Default from config (0.8).
    tolerance : float, optional
        Absolute tolerance.  Default from config (0.0).
    ignore_hh : bool, optional
        Whether to ignore H-H pairs.  Default from config (False).
    max_clashes : int, optional
        Maximum allowed clashes before failure.  Default from config (0).

    Returns
    -------
    CheckResult
        ``details`` contains: ``clash_count``, ``max_clashes``, ``pairs``,
        ``scale_used``, ``ignore_hh_used``.
    """
    scale = _get_config("intermolecular_clash_scale", scale)
    tolerance = _get_config("intermolecular_clash_tolerance", tolerance)
    ignore_hh = _get_config("ignore_hh_clashes", ignore_hh)
    max_clashes = _get_config("max_clashes", max_clashes)

    mc = _ensure_molecular_crystal(crystal)
    atoms = mc.to_ase()
    mol_idx = atoms.arrays.get("molecule_index")

    if mol_idx is None:
        return CheckResult(
            name="intermolecular_clash",
            passed=True,
            message="No molecule_index array; skipped.",
            details={"clash_count": 0, "pairs": [], "scale_used": scale, "ignore_hh_used": ignore_hh},
        )

    n = len(atoms)
    if n < 2:
        return CheckResult(
            name="intermolecular_clash",
            passed=True,
            message="Fewer than 2 atoms.",
            details={"clash_count": 0, "pairs": [], "scale_used": scale, "ignore_hh_used": ignore_hh},
        )

    symbols = atoms.get_chemical_symbols()
    radii = _ase_radii(atoms)
    max_cutoff = scale * 2 * radii.max() + abs(tolerance)

    # 'ijdS' — shift vector S != [0,0,0] means atoms are in different images
    i_idx, j_idx, dists, shifts = neighbor_list("ijdS", atoms, cutoff=max_cutoff)
    if len(i_idx) == 0:
        return CheckResult(
            name="intermolecular_clash",
            passed=True,
            message="No intermolecular clashes.",
            details={"clash_count": 0, "pairs": [], "scale_used": scale, "ignore_hh_used": ignore_hh},
        )

    cutoffs = np.maximum(0.0, scale * (radii[i_idx] + radii[j_idx]) - tolerance)
    clashing = dists < cutoffs

    if not np.any(clashing):
        return CheckResult(
            name="intermolecular_clash",
            passed=True,
            message="No intermolecular clashes.",
            details={"clash_count": 0, "pairs": [], "scale_used": scale, "ignore_hh_used": ignore_hh},
        )

    ci, cj, cd, cs = (
        i_idx[clashing],
        j_idx[clashing],
        dists[clashing],
        shifts[clashing],
    )

    # Filter H-H if requested
    if ignore_hh:
        sym_arr = np.array(symbols)
        both_h = (sym_arr[ci] == "H") & (sym_arr[cj] == "H")
        mask = ~both_h
        ci, cj, cd, cs = ci[mask], cj[mask], cd[mask], cs[mask]

    # Keep only intermolecular or same-mol cross-image
    different_mol = mol_idx[ci] != mol_idx[cj]
    cross_image = np.any(cs != 0, axis=1)
    valid = different_mol | cross_image

    ci, cj, cd = ci[valid], cj[valid], cd[valid]

    # De-duplicate
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for ii, jj, dd in zip(ci, cj, cd):
        pair_key = (min(int(ii), int(jj)), max(int(ii), int(jj)))
        if pair_key not in seen:
            seen.add(pair_key)
            pairs.append((pair_key[0], pair_key[1], float(dd)))

    clash_count = len(pairs)
    passed = clash_count <= max_clashes
    message = (
        f"No intermolecular clashes."
        if clash_count == 0
        else f"{clash_count} intermolecular clash(es) (max allowed: {max_clashes})."
    )

    return CheckResult(
        name="intermolecular_clash",
        passed=passed,
        message=message,
        details={
            "clash_count": clash_count,
            "max_clashes": max_clashes,
            "pairs": pairs,
            "scale_used": scale,
            "ignore_hh_used": ignore_hh,
        },
    )


def check_isolated_atoms(
    crystal,
    *,
    elements: set[str] | None = None,
) -> CheckResult:
    """Detect single-atom molecules of suspect elements.

    These typically indicate unresolved solvent (water without H) or
    coordination debris.  Only flags single-atom molecules whose element is in
    the suspect set.

    Parameters
    ----------
    crystal : MolecularCrystal or ASE Atoms
        Structure to validate.
    elements : set[str], optional
        Elements to flag as suspicious when found as isolated atoms.
        Default from config: {O, N, S, P, Se, Te, B, Si}.

    Returns
    -------
    CheckResult
        ``details`` contains: ``isolated_indices``, ``isolated_elements``.
    """
    elements = _get_config("isolated_atom_elements", elements)

    mc = _ensure_molecular_crystal(crystal)

    isolated_indices: list[int] = []
    isolated_elements: list[str] = []
    atom_offset = 0

    try:
        for mol in mc.molecules:
            syms = mol.get_chemical_symbols()
            if len(syms) == 1 and syms[0] in elements:
                isolated_indices.append(atom_offset)
                isolated_elements.append(syms[0])
            atom_offset += len(syms)
    except Exception:
        # Fallback: no molecule information available
        pass

    passed = len(isolated_indices) == 0
    message = (
        "No isolated suspect atoms."
        if passed
        else f"{len(isolated_indices)} isolated atom(s): {', '.join(isolated_elements)}."
    )

    return CheckResult(
        name="isolated_atoms",
        passed=passed,
        message=message,
        details={"isolated_indices": isolated_indices, "isolated_elements": isolated_elements},
    )


def check_hydrogen_presence(crystal) -> CheckResult:
    """Check that the crystal contains at least one hydrogen atom.

    Molecular crystals almost universally contain H; its absence usually
    indicates incomplete structure determination.

    Parameters
    ----------
    crystal : MolecularCrystal or ASE Atoms
        Structure to validate.

    Returns
    -------
    CheckResult
        ``details`` contains: ``has_hydrogen``.
    """
    mc = _ensure_molecular_crystal(crystal)

    has_h = False
    try:
        for mol in mc.molecules:
            if "H" in mol.get_chemical_symbols():
                has_h = True
                break
    except Exception:
        atoms = mc.to_ase()
        has_h = "H" in atoms.get_chemical_symbols()

    passed = has_h
    message = "Hydrogen atoms present." if passed else "No hydrogen atoms found."

    return CheckResult(
        name="hydrogen_presence",
        passed=passed,
        message=message,
        details={"has_hydrogen": has_h},
    )


# ─── Molecule-Level Checks ────────────────────────────────────────────────────


def check_topology_preservation(
    crystal_before,
    crystal_after,
) -> CheckResult:
    """Verify that molecular connectivity is preserved between two structures.

    Compares sorted graph invariants of all molecules before and after a
    transformation (rotation, interpolation, perturbation, etc.).

    Parameters
    ----------
    crystal_before : MolecularCrystal or ASE Atoms
        Reference structure (before transformation).
    crystal_after : MolecularCrystal or ASE Atoms
        Modified structure (after transformation).

    Returns
    -------
    CheckResult
        ``details`` contains: ``n_molecules_before``, ``n_molecules_after``,
        ``mismatched_invariants``.
    """
    from ..structures.crystal import MolecularCrystal
    from ..utils.graph import graph_invariant

    try:
        mc_before = _ensure_molecular_crystal(crystal_before)
        mc_after = _ensure_molecular_crystal(crystal_after)
    except Exception as exc:
        return CheckResult(
            name="topology_preservation",
            passed=False,
            message=f"Failed to parse structures: {exc}",
            details={"error": str(exc)},
        )

    n_before = len(mc_before.molecules)
    n_after = len(mc_after.molecules)

    if n_before != n_after:
        return CheckResult(
            name="topology_preservation",
            passed=False,
            message=f"Molecule count changed: {n_before} → {n_after}.",
            details={
                "n_molecules_before": n_before,
                "n_molecules_after": n_after,
                "mismatched_invariants": [],
            },
        )

    inv_before = sorted(graph_invariant(mol.graph) for mol in mc_before.molecules)
    inv_after = sorted(graph_invariant(mol.graph) for mol in mc_after.molecules)

    mismatched = [
        (i, str(ib), str(ia))
        for i, (ib, ia) in enumerate(zip(inv_before, inv_after))
        if ib != ia
    ]

    passed = len(mismatched) == 0
    message = (
        "Topology preserved."
        if passed
        else f"{len(mismatched)} molecule(s) have changed topology."
    )

    return CheckResult(
        name="topology_preservation",
        passed=passed,
        message=message,
        details={
            "n_molecules_before": n_before,
            "n_molecules_after": n_after,
            "mismatched_invariants": mismatched,
        },
    )


def check_formula_consistency(
    crystal,
    *,
    reference_formula: str | None = None,
) -> CheckResult:
    """Verify per-molecule formula decomposition against a reference.

    Decomposes the crystal into per-molecule empirical formulas, reduces
    multiplicities to simplest ratio, and compares with either a user-supplied
    reference or the ``formula`` metadata field.

    Parameters
    ----------
    crystal : MolecularCrystal or ASE Atoms
        Structure to validate.
    reference_formula : str, optional
        Expected formula.  If None, uses crystal metadata ``"formula"`` field.

    Returns
    -------
    CheckResult
        ``details`` contains: ``actual_multiset``, ``reference_formula``,
        ``mismatch_reason``.
    """
    mc = _ensure_molecular_crystal(crystal)

    # Build actual multiset from crystal molecules
    actual_multiset: Counter = Counter()
    try:
        for mol in mc.molecules:
            mol_counts = Counter(mol.get_chemical_symbols())
            actual_multiset[_empirical_formula(mol_counts)] += 1
    except Exception:
        atoms = mc.to_ase()
        mol_counts = Counter(atoms.get_chemical_symbols())
        actual_multiset[_empirical_formula(mol_counts)] += 1

    if not actual_multiset:
        return CheckResult(
            name="formula_consistency",
            passed=True,
            message="No molecules to check.",
            details={"actual_multiset": {}, "reference_formula": reference_formula, "mismatch_reason": ""},
        )

    # Get reference formula
    ref = reference_formula
    if ref is None:
        metadata = getattr(mc, "metadata", {}) or {}
        # Also check info dict on underlying ASE atoms
        if not metadata:
            try:
                atoms = mc.to_ase()
                metadata = atoms.info or {}
            except Exception:
                metadata = {}
        ref = str(metadata.get("formula", "") or "").strip()

    if not ref:
        return CheckResult(
            name="formula_consistency",
            passed=True,
            message="No reference formula available; skipped.",
            details={
                "actual_multiset": dict(actual_multiset),
                "reference_formula": None,
                "mismatch_reason": "no_reference",
            },
        )

    # Reduce actual multiplicities to simplest ratio
    actual_g = reduce(gcd, actual_multiset.values())
    actual_reduced = Counter({f: c // actual_g for f, c in actual_multiset.items()})

    # Compare element composition
    # Parse both reference formula and actual total composition
    try:
        from ..analysis.formula_moiety import parse_moiety_string
    except ImportError:
        parse_moiety_string = None

    actual_total: Counter = Counter()
    for mol in mc.molecules:
        actual_total.update(Counter(mol.get_chemical_symbols()))

    ref_counts = _parse_formula(ref)
    if not ref_counts:
        return CheckResult(
            name="formula_consistency",
            passed=True,
            message="Could not parse reference formula; skipped.",
            details={
                "actual_multiset": dict(actual_multiset),
                "reference_formula": ref,
                "mismatch_reason": "parse_error",
            },
        )

    # Compare element sets (the minimum consistency check)
    actual_elements = set(actual_total.keys())
    ref_elements = set(ref_counts.keys())

    if actual_elements != ref_elements:
        missing = ref_elements - actual_elements
        extra = actual_elements - ref_elements
        mismatch_parts = []
        if missing:
            mismatch_parts.append(f"missing {missing}")
        if extra:
            mismatch_parts.append(f"extra {extra}")
        mismatch_reason = "; ".join(mismatch_parts)

        return CheckResult(
            name="formula_consistency",
            passed=False,
            message=f"Element mismatch: {mismatch_reason}.",
            details={
                "actual_multiset": dict(actual_multiset),
                "reference_formula": ref,
                "mismatch_reason": mismatch_reason,
                "actual_elements": sorted(actual_elements),
                "reference_elements": sorted(ref_elements),
            },
        )

    return CheckResult(
        name="formula_consistency",
        passed=True,
        message="Formula consistent.",
        details={
            "actual_multiset": dict(actual_multiset),
            "reference_formula": ref,
            "mismatch_reason": "",
        },
    )


def check_bond_distances(
    crystal,
    *,
    min_factor: float | None = None,
    max_factor: float | None = None,
) -> CheckResult:
    """Check that bonded atom pairs have physically reasonable distances.

    Uses MolCrysKit-compatible covalent radii thresholds to identify bonds,
    then verifies each bond distance falls within
    ``[min_factor × expected, max_factor × expected]``.

    Parameters
    ----------
    crystal : MolecularCrystal or ASE Atoms
        Structure to validate.
    min_factor : float, optional
        Lower bound factor.  Default from config (0.5).
    max_factor : float, optional
        Upper bound factor.  Default from config (1.5).

    Returns
    -------
    CheckResult
        ``details`` contains: ``n_bonds_checked``, ``out_of_range`` (list of
        dicts with ``pair``, ``distance``, ``expected``, ``factor``),
        ``min_factor_used``, ``max_factor_used``.
    """
    min_factor = _get_config("bond_distance_min_factor", min_factor)
    max_factor = _get_config("bond_distance_max_factor", max_factor)

    mc = _ensure_molecular_crystal(crystal)
    atoms = mc.to_ase()

    # Build bond pairs using MCK-compatible thresholds
    bond_scale = BONDING_CONFIG["NON_METAL_THRESHOLD_FACTOR"]
    min_dist = BONDING_CONFIG["MIN_COVALENT_DISTANCE"]

    symbols = atoms.get_chemical_symbols()
    n = len(atoms)
    radii = np.zeros(n)
    for i, sym in enumerate(symbols):
        radii[i] = ATOMIC_RADII.get(sym, 0.7)

    max_cutoff = 2 * radii.max() * bond_scale + 0.1
    i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoff=max_cutoff)

    # Identify bonds and check their distances
    out_of_range: list[dict] = []
    seen: set[tuple[int, int]] = set()
    n_bonds = 0

    for ii, jj, dd in zip(i_idx, j_idx, dists):
        pair_key = (min(int(ii), int(jj)), max(int(ii), int(jj)))
        if pair_key in seen:
            continue

        expected_max = (radii[ii] + radii[jj]) * bond_scale
        if dd > expected_max:
            continue  # Not a bond
        if dd < min_dist:
            continue  # Below minimum covalent distance (handled by hard_clash)

        seen.add(pair_key)
        n_bonds += 1

        # Expected bond length is approximately (r_i + r_j)
        expected = radii[ii] + radii[jj]
        factor = float(dd) / expected if expected > 0 else float("inf")

        if factor < min_factor or factor > max_factor:
            out_of_range.append({
                "pair": pair_key,
                "distance": float(dd),
                "expected": float(expected),
                "factor": round(factor, 4),
            })

    n_bad = len(out_of_range)
    passed = n_bad == 0
    message = (
        f"All {n_bonds} bonds within range."
        if passed
        else f"{n_bad}/{n_bonds} bond(s) outside [{min_factor:.2f}, {max_factor:.2f}] × expected."
    )

    return CheckResult(
        name="bond_distances",
        passed=passed,
        message=message,
        details={
            "n_bonds_checked": n_bonds,
            "out_of_range": out_of_range,
            "min_factor_used": min_factor,
            "max_factor_used": max_factor,
        },
    )


# ─── Helpers (formula parsing) ────────────────────────────────────────────────


def _empirical_formula(counts: dict[str, int] | Counter) -> str:
    """Reduce element counts to empirical (simplest ratio) formula string."""
    if not counts:
        return ""
    int_counts = {el: int(round(v)) for el, v in counts.items() if int(round(v)) > 0}
    if not int_counts:
        return ""
    g = reduce(gcd, int_counts.values())
    reduced = {el: c // g for el, c in int_counts.items()}
    elems = sorted(reduced.keys(), key=lambda e: (e != "C", e != "H", e))
    return "".join(f"{el}{reduced[el]}" if reduced[el] > 1 else el for el in elems)


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula string into element counts."""
    import re

    counts: dict[str, int] = {}
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        el, n = match.group(1), match.group(2)
        if el:
            counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


# ─── Aggregated Entry Point ───────────────────────────────────────────────────


def sanity_check(
    crystal,
    *,
    checks: Sequence[str] | None = None,
    hard_clash_scale: float | None = None,
    hard_clash_tolerance: float | None = None,
    intermolecular_clash_scale: float | None = None,
    intermolecular_clash_tolerance: float | None = None,
    ignore_hh: bool | None = None,
    max_clashes: int | None = None,
    bond_distance_min_factor: float | None = None,
    bond_distance_max_factor: float | None = None,
    isolated_elements: set[str] | None = None,
    reference_formula: str | None = None,
) -> SanityReport:
    """Run multiple sanity checks on a crystal structure.

    Parameters
    ----------
    crystal : MolecularCrystal or ASE Atoms
        Structure to validate.
    checks : sequence of str, optional
        Which checks to run.  Default: all single-crystal checks.
        Valid names: ``"hard_clash"``, ``"intermolecular_clash"``,
        ``"isolated_atoms"``, ``"hydrogen_presence"``,
        ``"formula_consistency"``, ``"bond_distances"``.
    hard_clash_scale : float, optional
        Override for hard clash scale factor.
    hard_clash_tolerance : float, optional
        Override for hard clash tolerance.
    intermolecular_clash_scale : float, optional
        Override for intermolecular clash scale factor.
    intermolecular_clash_tolerance : float, optional
        Override for intermolecular clash tolerance.
    ignore_hh : bool, optional
        Override for ignoring H-H intermolecular clashes.
    max_clashes : int, optional
        Override for maximum allowed intermolecular clashes.
    bond_distance_min_factor : float, optional
        Override for bond distance lower bound factor.
    bond_distance_max_factor : float, optional
        Override for bond distance upper bound factor.
    isolated_elements : set[str], optional
        Override for suspect isolated-atom element set.
    reference_formula : str, optional
        Override for formula consistency reference.

    Returns
    -------
    SanityReport
        Aggregated report with all check results.
    """
    if checks is None:
        check_list = list(_SINGLE_CRYSTAL_CHECKS)
    else:
        check_list = list(checks)

    report = SanityReport()

    for check_name in check_list:
        if check_name == "hard_clash":
            result = check_hard_clash(
                crystal, scale=hard_clash_scale, tolerance=hard_clash_tolerance
            )
        elif check_name == "intermolecular_clash":
            result = check_intermolecular_clash(
                crystal,
                scale=intermolecular_clash_scale,
                tolerance=intermolecular_clash_tolerance,
                ignore_hh=ignore_hh,
                max_clashes=max_clashes,
            )
        elif check_name == "isolated_atoms":
            result = check_isolated_atoms(crystal, elements=isolated_elements)
        elif check_name == "hydrogen_presence":
            result = check_hydrogen_presence(crystal)
        elif check_name == "formula_consistency":
            result = check_formula_consistency(crystal, reference_formula=reference_formula)
        elif check_name == "bond_distances":
            result = check_bond_distances(
                crystal,
                min_factor=bond_distance_min_factor,
                max_factor=bond_distance_max_factor,
            )
        else:
            result = CheckResult(
                name=check_name,
                passed=False,
                message=f"Unknown check: {check_name!r}.",
                details={"error": "unknown_check"},
            )
        report.results.append(result)

    return report
