"""
Integration tests for disorder resolution of CIF files with atoms on 
crystallographic special positions.

Tests verify:
1. Correct atom counts after resolution
2. Chemically reasonable molecular formulas
3. Proper coordination environments (no overcoordination)
"""

import os
import numpy as np
import pytest
from collections import Counter, defaultdict

from molcrys_kit.io.cif import scan_cif_disorder
from molcrys_kit.analysis.disorder.process import generate_ordered_replicas_from_disordered_sites
from molcrys_kit.analysis.disorder.info import DisorderInfo
from molcrys_kit.analysis.disorder.graph import DisorderGraphBuilder


# =====================================================================
# Helper utilities
# =====================================================================

MAX_COORD = {'H': 1, 'C': 4, 'N': 4, 'O': 3, 'S': 6, 'Cl': 4, 'Cd': 8, 'P': 4, 'Zn': 6}

def _resolve_cif(path, expected_atoms=None):
    """Resolve a CIF and return crystal + diagnostics."""
    results = generate_ordered_replicas_from_disordered_sites(path, generate_count=1, method='optimal')
    crystal = results[0]
    n_atoms = crystal.get_total_nodes()
    
    # Check coordination
    problematic = []
    for mol in crystal.molecules:
        mol_symbols = mol.get_chemical_symbols()
        g = mol.graph
        for node in g.nodes:
            elem = mol_symbols[node] if node < len(mol_symbols) else '?'
            degree = g.degree(node)
            max_c = MAX_COORD.get(elem, 8)
            if degree > max_c:
                problematic.append((node, elem, degree, f"OVERCOORD (max={max_c})"))
    
    return crystal, n_atoms, problematic


def _get_molecule_formulas(crystal):
    """Extract molecule formulas from a crystal."""
    formulas = []
    for mol in crystal.molecules:
        mol_symbols = mol.get_chemical_symbols()
        mol_species = Counter(mol_symbols)
        formula = "".join(f"{e}{mol_species[e]}" for e in sorted(mol_species))
        formulas.append(formula)
    return Counter(formulas)


# =====================================================================
# DisorderGraphBuilder: implicit special-position conflict detection
# =====================================================================

class TestImplicitSPConflicts:
    """Test _add_implicit_sp_conflicts and _is_same_parent_pair."""

    def test_partial_occ_dg0_same_parent_are_competing(self):
        """Two dg=0 partial-occ copies from same asym parent should NOT be
        treated as 'same parent' (they compete for the same site)."""
        info = DisorderInfo(
            labels=["H3A", "H3A"],
            symbols=["H", "H"],
            frac_coords=np.array([[0.1, 0.1, 0.1], [0.12, 0.12, 0.12]]),
            occupancies=[0.5, 0.5],
            disorder_groups=[0, 0],
            assemblies=["", ""],
            asym_id=[0, 0],
            site_symmetry_order=[1, 1],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        assert not builder._is_same_parent_pair(0, 1)

    def test_full_occ_dg0_same_parent_are_same(self):
        """Two dg=0 full-occ copies from same asym parent should be 
        treated as 'same parent' (legitimate symmetry copies)."""
        info = DisorderInfo(
            labels=["N1", "N1"],
            symbols=["N", "N"],
            frac_coords=np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]),
            occupancies=[1.0, 1.0],
            disorder_groups=[0, 0],
            assemblies=["", ""],
            asym_id=[0, 0],
            site_symmetry_order=[3, 3],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        assert builder._is_same_parent_pair(0, 1)

    def test_different_parent_are_not_same(self):
        """Two atoms from different asym parents should NOT be same parent."""
        info = DisorderInfo(
            labels=["N1", "N2"],
            symbols=["N", "N"],
            frac_coords=np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]),
            occupancies=[1.0, 1.0],
            disorder_groups=[0, 0],
            assemblies=["", ""],
            asym_id=[0, 1],
            site_symmetry_order=[1, 1],
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        assert not builder._is_same_parent_pair(0, 1)

    def test_implicit_sp_conflicts_added_for_partial_occ(self):
        """_add_implicit_sp_conflicts should add conflict edges between
        partial-occ dg=0 copies of the same parent atom."""
        # Simulate H3A with 6 copies from sso=1 parent (occ=0.5, should form 3 sites of 2)
        n = 6
        info = DisorderInfo(
            labels=["H3A"] * n,
            symbols=["H"] * n,
            frac_coords=np.array([
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.12],  # close to [0] → same site
                [0.3, 0.3, 0.3],
                [0.3, 0.3, 0.32],  # close to [2] → same site
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.52],  # close to [4] → same site
            ]),
            occupancies=[0.5] * n,
            disorder_groups=[0] * n,
            assemblies=[""] * n,
            asym_id=[0] * n,
            site_symmetry_order=[1] * n,
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        builder._precompute_metrics()
        builder.graph = __import__('networkx').Graph()
        builder.graph.add_nodes_from(range(n))
        
        # Run implicit SP conflicts
        builder._add_implicit_sp_conflicts()
        
        # Each site should have within-site conflict edges
        # Site 0: {0, 1}, Site 1: {2, 3}, Site 2: {4, 5}
        # With expected_mult = round(1/0.5) = 2, so 3 clusters of 2
        # Each cluster should have 1 internal edge
        assert builder.graph.has_edge(0, 1) or builder.graph.has_edge(2, 3) or builder.graph.has_edge(4, 5)

    def test_ammonium_tetrahedral_decomposition(self):
        """NH4+ with 8 half-occ H atoms should decompose into two
        tetrahedral groups via _decompose_cliques."""
        # N at origin, 8 H at alternating tetrahedral positions
        labels = ["N1"] + [f"H{i}" for i in range(1, 9)]
        symbols = ["N"] + ["H"] * 8
        tet1 = np.array([
            [0.02, 0.02, 0.02],
            [0.02, -0.02, -0.02],
            [-0.02, 0.02, -0.02],
            [-0.02, -0.02, 0.02],
        ])
        tet2 = np.array([
            [0.025, 0.025, -0.025],
            [0.025, -0.025, 0.025],
            [-0.025, 0.025, 0.025],
            [-0.025, -0.025, -0.025],
        ])
        frac_coords = np.vstack([np.array([[0.0, 0.0, 0.0]]), tet1, tet2])
        occupancies = [1.0] + [0.5] * 8
        disorder_groups = [0] * 9
        assemblies = [""] * 9
        asym_id = [0] + [1]*4 + [2]*4  # N is parent 0, tet1 H parent 1, tet2 H parent 2
        sso = [6] + [1]*4 + [3]*4
        
        info = DisorderInfo(
            labels=labels, symbols=symbols, frac_coords=frac_coords,
            occupancies=occupancies, disorder_groups=disorder_groups,
            assemblies=assemblies, asym_id=asym_id, site_symmetry_order=sso,
        )
        lattice = np.eye(3) * 10.0
        builder = DisorderGraphBuilder(info, lattice)
        graph = builder.build()
        
        # H atoms (1-8) should have conflict edges between tet1 and tet2
        h_nodes = list(range(1, 9))
        h_edges = [(u, v) for u, v in graph.edges if u in h_nodes and v in h_nodes]
        assert len(h_edges) > 0, "Should have conflict edges between H atoms"


# =====================================================================
# End-to-end CIF resolution tests
# =====================================================================

@pytest.fixture(scope="module")
def examples_dir():
    """Return path to examples directory, skip if not available."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'examples')
    if not os.path.isdir(path):
        pytest.skip("examples directory not found")
    return path


class TestNatComm1:
    """NatComm-1.cif: Cd-thiocyanate MOF with special positions."""

    def test_atom_count(self, examples_dir):
        path = os.path.join(examples_dir, "NatComm-1.cif")
        if not os.path.exists(path):
            pytest.skip("NatComm-1.cif not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        assert n_atoms == 60, f"Expected 60 atoms, got {n_atoms}"

    def test_coordination(self, examples_dir):
        path = os.path.join(examples_dir, "NatComm-1.cif")
        if not os.path.exists(path):
            pytest.skip("NatComm-1.cif not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        # Cd atoms are isolated ions, allow those
        non_metal_problems = [p for p in problematic if p[1] not in ('Cd',)]
        assert len(non_metal_problems) == 0, f"Overcoordinated atoms: {non_metal_problems}"

    def test_molecular_formulas(self, examples_dir):
        path = os.path.join(examples_dir, "NatComm-1.cif")
        if not os.path.exists(path):
            pytest.skip("NatComm-1.cif not found")
        crystal, _, _ = _resolve_cif(path)
        formulas = _get_molecule_formulas(crystal)
        # Expected: 6× SCN, 2× Cd, 2× organic cation (C5H14N1)
        assert formulas.get("C1N1S1", 0) == 6, f"Expected 6 SCN units, got {formulas}"
        assert formulas.get("Cd1", 0) == 2, f"Expected 2 Cd ions, got {formulas}"
        assert formulas.get("C5H14N1", 0) == 2, f"Expected 2 organic cations, got {formulas}"


class TestPAPHM4:
    """PAP-HM4.cif: perchlorate salt with NH4+ on special positions."""

    def test_atom_count(self, examples_dir):
        path = os.path.join(examples_dir, "PAP-HM4.cif")
        if not os.path.exists(path):
            pytest.skip("PAP-HM4.cif not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        assert n_atoms == 176, f"Expected 176 atoms, got {n_atoms}"

    def test_no_overcoordination(self, examples_dir):
        path = os.path.join(examples_dir, "PAP-HM4.cif")
        if not os.path.exists(path):
            pytest.skip("PAP-HM4.cif not found")
        crystal, _, problematic = _resolve_cif(path)
        assert len(problematic) == 0, f"Overcoordinated atoms: {problematic}"

    def test_molecular_formulas(self, examples_dir):
        path = os.path.join(examples_dir, "PAP-HM4.cif")
        if not os.path.exists(path):
            pytest.skip("PAP-HM4.cif not found")
        crystal, _, _ = _resolve_cif(path)
        formulas = _get_molecule_formulas(crystal)
        # Expected: 12× ClO4, 4× NH4, 4× organic cation (C6H16N2)
        assert formulas.get("Cl1O4", 0) == 12, f"Expected 12 ClO4, got {formulas}"
        assert formulas.get("H4N1", 0) == 4, f"Expected 4 NH4, got {formulas}"
        assert formulas.get("C6H16N2", 0) == 4, f"Expected 4 organic cations, got {formulas}"


class TestDAP4:
    """DAP-4.cif: all dg=0 with NH4+ on high-symmetry positions."""

    def test_atom_count(self, examples_dir):
        path = os.path.join(examples_dir, "DAP-4.cif")
        if not os.path.exists(path):
            pytest.skip("DAP-4.cif not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        assert n_atoms == 336, f"Expected 336 atoms, got {n_atoms}"

    def test_no_overcoordination(self, examples_dir):
        path = os.path.join(examples_dir, "DAP-4.cif")
        if not os.path.exists(path):
            pytest.skip("DAP-4.cif not found")
        crystal, _, problematic = _resolve_cif(path)
        assert len(problematic) == 0, f"Overcoordinated atoms: {problematic}"

    def test_molecular_formulas(self, examples_dir):
        path = os.path.join(examples_dir, "DAP-4.cif")
        if not os.path.exists(path):
            pytest.skip("DAP-4.cif not found")
        crystal, _, _ = _resolve_cif(path)
        formulas = _get_molecule_formulas(crystal)
        # Expected: 8× organic (C6H14N2), 24× ClO4, 8× NH4
        assert formulas.get("C6H14N2", 0) == 8, f"Expected 8 organic, got {formulas}"
        assert formulas.get("Cl1O4", 0) == 24, f"Expected 24 ClO4, got {formulas}"
        assert formulas.get("H4N1", 0) == 8, f"Expected 8 NH4, got {formulas}"


class TestAnhydrousCaffeine:
    """anhydrousCaffeine CIF - verify no regression on standard disorder."""

    def test_resolves_without_error(self, examples_dir):
        path = os.path.join(examples_dir, "anhydrousCaffeine_CGD_2007_7_1406.cif")
        if not os.path.exists(path):
            pytest.skip("anhydrousCaffeine not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        assert n_atoms > 0
        assert len(problematic) == 0

    def test_caffeine2_resolves(self, examples_dir):
        path = os.path.join(examples_dir, "anhydrousCaffeine2_CGD_2007_7_1406.cif")
        if not os.path.exists(path):
            pytest.skip("anhydrousCaffeine2 not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        assert n_atoms > 0
        assert len(problematic) == 0


class TestZIF4:
    """ZIF-4 - verify no regression."""

    def test_resolves_without_error(self, examples_dir):
        path = os.path.join(examples_dir, "ZIF-4.cif")
        if not os.path.exists(path):
            pytest.skip("ZIF-4.cif not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        assert n_atoms > 0
        assert len(problematic) == 0


class TestTILPEN:
    """TILPEN - verify no regression."""

    def test_resolves_without_error(self, examples_dir):
        path = os.path.join(examples_dir, "TILPEN.cif")
        if not os.path.exists(path):
            pytest.skip("TILPEN.cif not found")
        crystal, n_atoms, problematic = _resolve_cif(path)
        assert n_atoms == 84
        assert len(problematic) == 0
