"""
Molecular formal charge determination for Tasker surface analysis.

This module provides hybrid charge assignment for CrystalMolecule objects:
  1. user-supplied mol_charge_map (formula → charge)
  2. pymatgen BVAnalyzer auto-guess
  3. zero fallback with warning
"""

import warnings
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..structures.crystal import MolecularCrystal
from ..structures.molecule import CrystalMolecule


def compute_topo_signature(mol: CrystalMolecule) -> str:
    """
    Compute a topology signature for a molecule.

    The signature combines the Hill-order chemical formula with a short hash
    of the sorted node-degree sequence from the molecular connectivity graph,
    making it sensitive to both composition and bond topology.

    Parameters
    ----------
    mol : CrystalMolecule
        Molecule to fingerprint.

    Returns
    -------
    str
        A string of the form "<formula>|<8-char hex hash>" that uniquely
        identifies the molecule type within the crystal.
    """
    # Check cache first
    if getattr(mol, '_topo_signature', None) is not None:
        return mol._topo_signature
    
    formula = mol.get_chemical_formula(mode="hill", empirical=False)
    graph = mol.get_graph()
    degree_seq = sorted(d for _, d in graph.degree())
    degree_hash = hashlib.md5(str(degree_seq).encode()).hexdigest()[:8]
    sig = f"{formula}|{degree_hash}"
    mol._topo_signature = sig
    return sig


@dataclass
class MolChargeResult:
    """
    Formal charge assignment result for one distinct molecule topology.

    Attributes
    ----------
    topo_signature : str
        Topology signature (formula + degree-sequence hash).
    formula : str
        Hill-order chemical formula.
    formal_charge : float
        Assigned formal charge.
    source : str
        Origin of the charge value: "user_map", "auto_guess", or
        "none" (zero fallback with warning).
    """

    topo_signature: str
    formula: str
    formal_charge: float
    source: str  # "user_map" | "auto_guess" | "none"


def _guess_charge_pymatgen(mol: CrystalMolecule) -> Optional[float]:
    """
    Attempt to guess the formal charge of a molecule using pymatgen BVAnalyzer.

    The molecule is placed in a large cubic supercell (100 Å) to eliminate
    spurious periodic interactions, then bond-valence analysis is applied.

    Parameters
    ----------
    mol : CrystalMolecule
        Molecule whose formal charge is to be guessed.

    Returns
    -------
    float or None
        Summed bond-valence oxidation states, or None on any failure.
    """
    try:
        from pymatgen.core import Structure, Lattice
        from pymatgen.analysis.bond_valence import BVAnalyzer

        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()
        big_lattice = Lattice.cubic(100.0)
        pmg_structure = Structure(
            big_lattice,
            species=symbols,
            coords=positions,
            coords_are_cartesian=True,
        )
        valences = BVAnalyzer().get_valences(pmg_structure)
        return float(sum(valences))
    except Exception:
        return None


def assign_mol_formal_charges(
    crystal: MolecularCrystal,
    mol_charge_map: Optional[Dict[str, int]] = None,
) -> Dict[str, MolChargeResult]:
    """
    Assign formal charges to each distinct molecule topology in a crystal.

    Uses a three-level hybrid strategy:

    1. **user_map** – if mol_charge_map contains an entry for the
       molecule formula, that value is used directly.
    2. **auto_guess** – pymatgen :class: is used to estimate
       bond-valence oxidation states, which are summed to give the molecular
       formal charge.
    3. **none** – if both methods fail, formal charge is set to 0 and a
       :class: is emitted.  Downstream Tasker analysis will
       degrade to topology-only ordering.

    Parameters
    ----------
    crystal : MolecularCrystal
        Crystal whose molecule population will be typed and charged.
    mol_charge_map : dict, optional
        Mapping from Hill-order chemical formula (e.g. "C8H9NO2") to
        integer formal charge.

    Returns
    -------
    dict
        Mapping topo_signature -> MolChargeResult for every distinct
        molecule topology found in *crystal*.  Multiple molecules sharing
        the same topology contribute only one entry.
    """
    if mol_charge_map is None:
        mol_charge_map = {}

    results: Dict[str, MolChargeResult] = {}

    for mol in crystal.molecules:
        sig = compute_topo_signature(mol)
        if sig in results:
            continue

        formula = mol.get_chemical_formula(mode="hill", empirical=False)

        if formula in mol_charge_map:
            charge = float(mol_charge_map[formula])
            source = "user_map"
        else:
            guessed = _guess_charge_pymatgen(mol)
            if guessed is not None:
                charge = guessed
                source = "auto_guess"
            else:
                charge = 0.0
                source = "none"
                warnings.warn(
                    f"Could not determine formal charge for molecule ''{formula}''"
                    f" (topo_signature='{sig}').  Defaulting to 0.  "
                    "Tasker analysis will degrade to topology-only ordering.  "
                    "Provide a mol_charge_map to suppress this warning.",
                    UserWarning,
                    stacklevel=2,
                )

        results[sig] = MolChargeResult(
            topo_signature=sig,
            formula=formula,
            formal_charge=charge,
            source=source,
        )

    return results
