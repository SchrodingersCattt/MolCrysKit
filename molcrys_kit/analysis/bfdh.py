"""Bravais-Friedel-Donnay-Harker (BFDH) facet candidates.

This module implements a lightweight, model-agnostic facet ranking based on
interplanar spacing.  In the pure BFDH approximation, planes with larger
``d_hkl`` grow more slowly and therefore are expected to be morphologically
more important.  The ranking supports downstream slab generation and bounded
implicit BFDH morphologies; it is not a replacement for surface-energy
calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from functools import reduce
from typing import Iterable, Optional, Sequence, Tuple, Any

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..structures.crystal import MolecularCrystal
from ..structures.symmetry import CrystalSymmetry, FractionalAffineOperation


MillerIndex = Tuple[int, int, int]


class _ExtinctFacet(Exception):
    """Raised internally when all members of a facet family are extinct."""


@dataclass(frozen=True)
class BFDHFacetInfo:
    """Metadata for a single BFDH-ranked Miller facet.

    Parameters
    ----------
    miller_index
        Representative Miller index for the facet family.
    d_hkl
        Interplanar spacing in Angstrom.
    relative_growth_rate
        Normalized BFDH growth-rate proxy, ``min(d_hkl) / d_hkl``.  Smaller
        values indicate slower growth and therefore a more important facet.
    relative_morphological_importance
        Normalized BFDH importance proxy, ``d_hkl / max(d_hkl)``.  Larger
        values indicate more important facets.
    rank
        Zero-based rank after sorting by decreasing BFDH importance.
    multiplicity
        Number of symmetry-equivalent Miller indices found for this facet.
    equivalent_millers
        Symmetry-equivalent Miller indices, canonicalized for deterministic
        output.  Empty when symmetry expansion is disabled or unavailable.
    source
        Candidate-generation backend: ``"pymatgen"``, ``"internal"``, or
        ``"explicit"``.
    """

    miller_index: MillerIndex
    d_hkl: float
    relative_growth_rate: float
    relative_morphological_importance: float
    rank: int
    multiplicity: int = 1
    equivalent_millers: Tuple[MillerIndex, ...] = field(default_factory=tuple)
    source: str = "internal"

    @property
    def hkl(self) -> MillerIndex:
        """Alias for :attr:`miller_index`."""

        return self.miller_index

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "miller_index": list(self.miller_index),
            "d_hkl": self.d_hkl,
            "relative_growth_rate": self.relative_growth_rate,
            "relative_morphological_importance": self.relative_morphological_importance,
            "rank": self.rank,
            "multiplicity": self.multiplicity,
            "equivalent_millers": [list(hkl) for hkl in self.equivalent_millers],
            "source": self.source,
        }


def _as_lattice_and_structure(
    crystal_or_lattice_or_structure: MolecularCrystal | Lattice | Structure | np.ndarray | Sequence[Sequence[float]],
) -> tuple[Lattice, Optional[Structure]]:
    """Convert supported inputs to a pymatgen lattice and optional structure."""

    obj = crystal_or_lattice_or_structure
    if isinstance(obj, Structure):
        return obj.lattice, obj
    if isinstance(obj, Lattice):
        return obj, None
    if isinstance(obj, MolecularCrystal):
        return Lattice(obj.lattice), None
    arr = np.asarray(obj, dtype=float)
    if arr.shape != (3, 3):
        raise TypeError(
            "Expected a MolecularCrystal, pymatgen Lattice/Structure, or a 3x3 lattice matrix."
        )
    return Lattice(arr), None


def _gcd3(hkl: MillerIndex) -> int:
    return reduce(gcd, (abs(x) for x in hkl if x != 0), 0) or 1


def _canonical_sign(hkl: Sequence[int], include_negative: bool = False) -> MillerIndex:
    """Return a deterministic representative for an hkl/opposite-hkl pair."""

    hkl_tuple = tuple(int(round(x)) for x in hkl)
    if include_negative:
        return hkl_tuple  # type: ignore[return-value]
    for value in hkl_tuple:
        if value > 0:
            return hkl_tuple  # type: ignore[return-value]
        if value < 0:
            return tuple(-x for x in hkl_tuple)  # type: ignore[return-value]
    return hkl_tuple  # pragma: no cover; (0, 0, 0) is filtered upstream


def enumerate_low_index_millers(
    max_index: int = 2,
    *,
    include_negative: bool = False,
) -> list[MillerIndex]:
    """Enumerate deterministic low-index Miller candidates.

    This fallback enumerator keeps higher-order planes such as ``(2, 0, 0)``
    distinct from ``(1, 0, 0)`` because they have different interplanar
    spacings in a BFDH ranking.  Opposite signs are canonicalized by default.
    """

    if max_index < 1:
        raise ValueError("max_index must be >= 1")

    candidates: set[MillerIndex] = set()
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == k == l == 0:
                    continue
                candidates.add(_canonical_sign((h, k, l), include_negative=include_negative))

    return sorted(
        candidates,
        key=lambda hkl: (sum(abs(v) for v in hkl), max(abs(v) for v in hkl), hkl),
    )


def _deduplicate_millers(
    miller_indices: Iterable[Sequence[int]],
    *,
    include_negative: bool = False,
) -> list[MillerIndex]:
    candidates = {
        _canonical_sign(hkl, include_negative=include_negative)
        for hkl in miller_indices
        if tuple(int(round(x)) for x in hkl) != (0, 0, 0)
    }
    return sorted(
        candidates,
        key=lambda hkl: (sum(abs(v) for v in hkl), max(abs(v) for v in hkl), hkl),
    )


def _structure_for_symmetry(lattice: Lattice) -> Structure:
    """Build a one-site placeholder structure for lattice symmetry helpers."""

    return Structure(lattice, ["H"], [[0, 0, 0]])


def _generate_candidates(
    lattice: Lattice,
    structure: Optional[Structure],
    max_index: int,
    miller_indices: Optional[Iterable[Sequence[int]]],
    use_pymatgen_symmetry: bool,
    include_negative: bool,
    symmetry: Optional[CrystalSymmetry],
) -> tuple[list[MillerIndex], str]:
    if miller_indices is not None:
        return _deduplicate_millers(miller_indices, include_negative=include_negative), "explicit"

    if symmetry is not None:
        # Explicit parent symmetry must control family reduction.  Start from
        # the complete deterministic candidate set instead of allowing the
        # lattice metric (or an ordered coordinate model) to pre-combine it.
        return enumerate_low_index_millers(
            max_index, include_negative=include_negative
        ), "internal"

    if use_pymatgen_symmetry:
        try:
            symmetry_structure = structure if structure is not None else _structure_for_symmetry(lattice)
            return _deduplicate_millers(
                get_symmetrically_distinct_miller_indices(symmetry_structure, max_index),
                include_negative=include_negative,
            ), "pymatgen"
        except Exception:
            # Fall back to deterministic internal enumeration for low-symmetry or
            # otherwise problematic inputs.  The BFDH score itself remains valid.
            pass

    return enumerate_low_index_millers(max_index, include_negative=include_negative), "internal"


def _operation_arrays(operation: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return fractional rotation and translation arrays for one symmetry op."""

    if isinstance(operation, FractionalAffineOperation):
        return operation.rotation, operation.translation
    return (
        np.asarray(operation.rotation_matrix, dtype=float),
        np.asarray(operation.translation_vector, dtype=float),
    )


def _reciprocal_symmetry_millers(
    structure: Optional[Structure],
    hkl: MillerIndex,
    *,
    symprec: float,
    include_negative: bool,
    operations: Optional[Sequence[Any]] = None,
) -> tuple[MillerIndex, ...]:
    """Return structure-symmetry-equivalent Miller indices.

    The fractional-space operation ``x' = R x + t`` transforms reciprocal
    vectors as ``h' = R.T h``.  Translations are intentionally ignored here;
    they are used separately for systematic-absence filtering.
    """

    if operations is None:
        if structure is None:
            return (_canonical_sign(hkl, include_negative=include_negative),)
        try:
            operations = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations(cartesian=False)
        except Exception:
            return (_canonical_sign(hkl, include_negative=include_negative),)

    equivalents: set[MillerIndex] = set()
    hkl_arr = np.asarray(hkl, dtype=int)
    target_order = _gcd3(hkl)
    for op in operations:
        rotation, _ = _operation_arrays(op)
        rotation = np.rint(rotation).astype(int)
        transformed = tuple(int(x) for x in rotation.T @ hkl_arr)
        if transformed == (0, 0, 0):
            continue
        if _gcd3(transformed) != target_order:
            continue
        equivalents.add(_canonical_sign(transformed, include_negative=include_negative))

    equivalents.add(_canonical_sign(hkl, include_negative=include_negative))
    return tuple(sorted(equivalents))


def _is_systematically_allowed(
    structure: Optional[Structure],
    hkl: MillerIndex,
    *,
    symprec: float,
    atol: float = 1e-8,
    operations: Optional[Sequence[Any]] = None,
) -> bool:
    """Return whether an hkl is allowed by translational extinction rules.

    This Donnay-Harker-style filter evaluates symmetry operations whose
    rotational part preserves the reflection family and sums the phase factors
    from their fractional translations.  A vanishing phase sum marks a
    systematic absence.  The test is composition-independent and uses only the
    space-group operations inferred for ``structure``.
    """

    if operations is None:
        if structure is None:
            return True
        try:
            operations = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations(cartesian=False)
        except Exception:
            return True

    hkl_arr = np.asarray(hkl, dtype=int)
    canonical = _canonical_sign(hkl)
    phases: list[complex] = []
    for op in operations:
        rotation, translation = _operation_arrays(op)
        rotation = np.rint(rotation).astype(int)
        transformed = tuple(int(x) for x in rotation.T @ hkl_arr)
        if _canonical_sign(transformed) == canonical:
            phase = np.exp(2j * np.pi * float(np.dot(hkl_arr, translation)))
            phases.append(complex(phase))

    if not phases:
        return True
    return abs(sum(phases) / len(phases)) > atol


def _filter_extinctions(
    lattice: Lattice,
    structure: Optional[Structure],
    candidates: Iterable[MillerIndex],
    *,
    symprec: float,
    include_negative: bool,
    symmetry: Optional[CrystalSymmetry] = None,
    apply_extinction_filter: bool = True,
) -> dict[MillerIndex, tuple[MillerIndex, ...]]:
    if symmetry is not None:
        operations: Optional[Sequence[Any]] = symmetry.operations
    elif structure is not None:
        try:
            operations = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations(cartesian=False)
        except Exception:
            operations = None
    else:
        operations = None

    if operations is None:
        return {hkl: tuple() for hkl in candidates}

    representative_map: dict[MillerIndex, tuple[MillerIndex, ...]] = {}
    for hkl in candidates:
        equivalents = _reciprocal_symmetry_millers(
            structure,
            hkl,
            symprec=symprec,
            include_negative=include_negative,
            operations=operations,
        )
        allowed_equivalents = [
            eq
            for eq in equivalents
            if not apply_extinction_filter
            or _is_systematically_allowed(
                structure, eq, symprec=symprec, operations=operations
            )
        ]
        if not allowed_equivalents:
            continue
        representative = max(
            allowed_equivalents,
            key=lambda eq: (
                _d_hkl(lattice, eq),
                -sum(abs(v) for v in eq),
                -max(abs(v) for v in eq),
                tuple(-v for v in eq),
            ),
        )
        canonical_hkl = _canonical_sign(hkl, include_negative=include_negative)
        representative_map[canonical_hkl] = tuple(sorted(allowed_equivalents))
        if representative != canonical_hkl:
            representative_map[representative] = tuple(sorted(allowed_equivalents))

    return representative_map


def _allowed_representative(
    hkl: MillerIndex,
    d_hkl: float,
    allowed_map: dict[MillerIndex, tuple[MillerIndex, ...]],
    lattice: Lattice,
) -> tuple[MillerIndex, float, tuple[MillerIndex, ...]]:
    allowed_equivalents = allowed_map.get(hkl)
    if allowed_equivalents is None:
        raise _ExtinctFacet
    if not allowed_equivalents:
        return hkl, d_hkl, tuple()

    representative = max(
        allowed_equivalents,
        key=lambda eq: (
            _d_hkl(lattice, eq),
            -sum(abs(v) for v in eq),
            -max(abs(v) for v in eq),
            tuple(-v for v in eq),
        ),
    )
    return representative, _d_hkl(lattice, representative), allowed_equivalents


def _equivalent_millers(
    lattice: Lattice,
    hkl: MillerIndex,
    *,
    symprec: float,
    include_negative: bool,
) -> tuple[MillerIndex, ...]:
    """Return reciprocal-symmetry-equivalent Miller indices."""

    try:
        operations = lattice.get_recp_symmetry_operation(symprec)
    except Exception:
        return (hkl,)

    equivalents: set[MillerIndex] = set()
    for op in operations:
        transformed = tuple(int(round(x)) for x in op.operate(hkl))
        if transformed == (0, 0, 0):
            continue
        # Do not reduce order: (2, 0, 0) and (1, 0, 0) are distinct BFDH planes.
        if _gcd3(transformed) != _gcd3(hkl):
            continue
        equivalents.add(_canonical_sign(transformed, include_negative=include_negative))
    equivalents.add(_canonical_sign(hkl, include_negative=include_negative))
    return tuple(sorted(equivalents))


def _d_hkl(lattice: Lattice, hkl: MillerIndex) -> float:
    recp = lattice.reciprocal_lattice_crystallographic
    g_cart = np.asarray(recp.get_cartesian_coords(hkl), dtype=float)
    norm = float(np.linalg.norm(g_cart))
    if norm <= 0.0:
        raise ValueError(f"Invalid Miller index {hkl}: reciprocal vector has zero length")
    return 1.0 / norm


def enumerate_bfdh_facets(
    crystal_or_lattice_or_structure: MolecularCrystal | Lattice | Structure | np.ndarray | Sequence[Sequence[float]],
    *,
    max_index: int = 2,
    miller_indices: Optional[Iterable[Sequence[int]]] = None,
    top_n: Optional[int] = None,
    symprec: float = 1e-5,
    use_pymatgen_symmetry: bool = True,
    include_equivalents: bool = False,
    include_negative: bool = False,
    extinction_filter: bool = True,
    symmetry: Optional[CrystalSymmetry] = None,
) -> list[BFDHFacetInfo]:
    """Enumerate BFDH-ranked facet candidates.

    Parameters
    ----------
    crystal_or_lattice_or_structure
        A :class:`~molcrys_kit.structures.crystal.MolecularCrystal`, pymatgen
        ``Lattice``/``Structure``, or a 3x3 row-vector lattice matrix.
    max_index
        Maximum absolute Miller index for generated candidates when
        ``miller_indices`` is not supplied.  Defaults to 2.
    miller_indices
        Optional explicit Miller indices.  When supplied, these replace the
        automatically generated candidates and ``max_index`` is ignored.
        They remain subject to ``extinction_filter`` when symmetry operations
        are available.
    top_n
        Optional number of top-ranked facets to return.
    symprec
        Tolerance passed to pymatgen reciprocal-lattice symmetry operations.
    use_pymatgen_symmetry
        Use pymatgen's symmetry-distinct Miller enumeration when available.
        If it fails, MolCrysKit falls back to internal enumeration.
    include_equivalents
        Include reciprocal-symmetry-equivalent Miller indices in each result.
    include_negative
        Keep opposite-sign Miller indices distinct.  By default ``(h,k,l)``
        and ``(-h,-k,-l)`` are treated as the same facet.
    extinction_filter
        Apply Donnay-Harker-style systematic-absence filtering from inferred
        space-group operations when structure information is available. Enabled
        by default.  Lattice-only inputs do not have space-group translations,
        so no extinction filtering is applied for those inputs.
    symmetry
        Optional explicit parent-crystal symmetry.  When supplied, its
        fractional operations take precedence over symmetry inferred from a
        pymatgen ``Structure`` and over lattice-metric family reduction.  This
        is the appropriate input when coordinates have been disorder-resolved
        but the BFDH morphology should retain the source CIF space group.

    Returns
    -------
    list[BFDHFacetInfo]
        Facets sorted by decreasing BFDH morphological importance.
    """

    if miller_indices is None and max_index < 1:
        raise ValueError("max_index must be >= 1")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be >= 1 when provided")
    if symmetry is not None and not isinstance(symmetry, CrystalSymmetry):
        raise TypeError("symmetry must be a CrystalSymmetry")

    lattice, structure = _as_lattice_and_structure(crystal_or_lattice_or_structure)
    candidates, source = _generate_candidates(
        lattice,
        structure,
        max_index,
        miller_indices,
        use_pymatgen_symmetry,
        include_negative,
        symmetry,
    )
    if (
        extinction_filter
        and structure is not None
        and symmetry is None
        and miller_indices is None
    ):
        # Donnay-Harker filtering needs signed low-index candidates before
        # space-group extinctions are applied.  Pymatgen's symmetry-distinct
        # surface helper can omit sign variants such as monoclinic (1 0 -1)
        # that are not equivalent to (1 0 1) and may be important BFDH faces.
        candidates = enumerate_low_index_millers(max_index, include_negative=include_negative)
        source = "internal"
    if not candidates:
        return []
    allowed_map = (
        _filter_extinctions(
            lattice,
            structure,
            candidates,
            symprec=symprec,
            include_negative=include_negative,
            symmetry=symmetry,
            apply_extinction_filter=extinction_filter,
        )
        if extinction_filter or symmetry is not None
        else {hkl: tuple() for hkl in candidates}
    )

    raw_map: dict[MillerIndex, tuple[float, tuple[MillerIndex, ...]]] = {}
    for hkl in candidates:
        try:
            representative, d_hkl, allowed_equivalents = _allowed_representative(
                hkl,
                _d_hkl(lattice, hkl),
                allowed_map,
                lattice,
            )
        except _ExtinctFacet:
            continue
        raw_map[representative] = (d_hkl, allowed_equivalents)
    if not raw_map:
        return []

    raw = [(hkl, d_hkl, allowed_equivalents) for hkl, (d_hkl, allowed_equivalents) in raw_map.items()]
    max_d = max(d for _, d, _ in raw)
    min_d = min(d for _, d, _ in raw)

    sorted_raw = sorted(
        raw,
        key=lambda item: (
            -item[1],
            sum(abs(v) for v in item[0]),
            max(abs(v) for v in item[0]),
            item[0],
        ),
    )
    if top_n is not None:
        sorted_raw = sorted_raw[:top_n]

    results: list[BFDHFacetInfo] = []
    for rank, (hkl, d_hkl, allowed_equivalents) in enumerate(sorted_raw):
        equivalents = (
            allowed_equivalents
            if allowed_equivalents
            else _reciprocal_symmetry_millers(
                structure,
                hkl,
                symprec=symprec,
                include_negative=include_negative,
                operations=symmetry.operations,
            )
            if include_equivalents and symmetry is not None
            else _equivalent_millers(lattice, hkl, symprec=symprec, include_negative=include_negative)
            if include_equivalents
            else tuple()
        )
        results.append(
            BFDHFacetInfo(
                miller_index=hkl,
                d_hkl=float(d_hkl),
                relative_growth_rate=float(min_d / d_hkl),
                relative_morphological_importance=float(d_hkl / max_d),
                rank=rank,
                multiplicity=len(equivalents) if equivalents else 1,
                equivalent_millers=equivalents,
                source=source,
            )
        )

    return results


__all__ = [
    "BFDHFacetInfo",
    "enumerate_bfdh_facets",
    "enumerate_low_index_millers",
]
