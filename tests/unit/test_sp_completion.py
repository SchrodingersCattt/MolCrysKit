from __future__ import annotations

from collections import Counter
from pathlib import Path

from pymatgen.io.cif import CifParser

from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder
from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.io.cif import scan_cif_disorder


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _formula(symbols: list[str]) -> str:
    counts = Counter(symbols)
    return "".join(f"{element}{counts[element]}" for element in sorted(counts))


def test_368k_records_sp_completion_without_symmetry_clash():
    cif = EXAMPLES_DIR / "368K.cif"
    info = scan_cif_disorder(str(cif))
    lattice = CifParser(str(cif)).parse_structures()[0].lattice.matrix

    graph = DisorderGraphBuilder(info, lattice).build()

    pairs = graph.graph.get("sp_completion_pairs", [])
    assert pairs

    symmetry_clashes = 0
    for atoms_a, atoms_b in pairs:
        for atom_a in atoms_a:
            for atom_b in atoms_b:
                if not graph.has_edge(atom_a, atom_b):
                    continue
                if graph[atom_a][atom_b].get("conflict_type") == "symmetry_clash":
                    symmetry_clashes += 1

    assert symmetry_clashes == 0


def test_368k_sp_completion_resolves_complete_cations():
    cif = EXAMPLES_DIR / "368K.cif"
    [crystal] = generate_ordered_replicas_from_disordered_sites(
        str(cif), generate_count=1, method="optimal"
    )

    formulas = Counter(_formula(mol.get_chemical_symbols()) for mol in crystal.molecules)
    assert crystal.get_total_nodes() == 88
    assert formulas["C8F1H11N1"] == 4
    assert formulas["Br1"] == 4
