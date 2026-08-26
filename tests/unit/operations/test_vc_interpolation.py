"""Tests for variable-cell interpolation (interpolate_crystal_vc)."""

import numpy as np
import pytest

from molcrys_kit.utils.geometry import (
    lattice_deformation_logm,
    lattice_at_lambda,
    frac_to_cart,
    volume_of_cell,
)


class TestLatticeInterpolationUtilities:
    """Tests for the GL+(3) geodesic lattice interpolation helpers."""

    def test_logm_identity_gives_zeros(self):
        lat = np.array([[5.0, 0, 0], [0, 6.0, 0], [0, 0, 7.0]])
        L = lattice_deformation_logm(lat, lat)
        np.testing.assert_allclose(L, np.zeros((3, 3)), atol=1e-12)

    def test_isotropic_stretch(self):
        lat_a = np.eye(3) * 4.0
        lat_b = np.eye(3) * 8.0  # 2x isotropic stretch
        L = lattice_deformation_logm(lat_a, lat_b)
        # L should be ln(2) * I
        np.testing.assert_allclose(L, np.log(2) * np.eye(3), atol=1e-10)

    def test_endpoints_reproduced(self):
        lat_a = np.array([[4.6, 0.0, 0.0], [0.5, 6.1, 0.0], [-1.2, 0.0, 16.3]])
        lat_b = np.array([[7.1, 0.0, 0.0], [0.0, 6.9, 0.0], [0.0, 0.0, 19.7]])
        L = lattice_deformation_logm(lat_a, lat_b)
        # lambda=0 -> lat_a
        np.testing.assert_allclose(lattice_at_lambda(lat_a, L, 0.0), lat_a, atol=1e-10)
        # lambda=1 -> lat_b
        np.testing.assert_allclose(lattice_at_lambda(lat_a, L, 1.0), lat_b, atol=1e-10)

    def test_volume_monotonic_for_expansion(self):
        lat_a = np.eye(3) * 5.0
        lat_b = np.eye(3) * 7.0
        L = lattice_deformation_logm(lat_a, lat_b)
        volumes = [
            volume_of_cell(lattice_at_lambda(lat_a, L, lam))
            for lam in np.linspace(0, 1, 11)
        ]
        # Volume should be strictly increasing for isotropic expansion
        for i in range(len(volumes) - 1):
            assert volumes[i + 1] > volumes[i]

    def test_negative_det_raises(self):
        lat_a = np.eye(3)
        lat_b = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1.0]])  # reflection
        with pytest.raises(ValueError, match="non-positive determinant"):
            lattice_deformation_logm(lat_a, lat_b)


class TestInterpolateCrystalVC:
    """Integration tests for variable-cell crystal interpolation."""

    @pytest.fixture
    def simple_pair(self):
        """Two triclinic crystals: a stretched with one molecule rotated 90°."""
        from ase import Atoms as AseAtoms

        from molcrys_kit.structures.crystal import MolecularCrystal
        from molcrys_kit.structures.molecule import CrystalMolecule

        lat_a = np.array([[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
        lat_b = np.array([[6.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 8.0]])

        # Single molecule: 3 atoms (like H2O)
        pos_a = np.array([[2.5, 3.0, 3.5], [2.5, 3.5, 3.5], [3.0, 3.0, 3.5]])
        ase_a = AseAtoms("OHH", positions=pos_a)
        mol_a = CrystalMolecule(atoms=ase_a, check_pbc=False)
        crystal_a = MolecularCrystal(lat_a, [mol_a], [True, True, True])

        # Molecule B: rotated 90° around z, centered at frac (0.5,0.5,0.5) of lat_b
        com_b = np.array([3.0, 3.5, 4.0])
        R90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        centered = pos_a - np.mean(pos_a, axis=0)
        pos_b = centered @ R90z.T + com_b
        ase_b = AseAtoms("OHH", positions=pos_b)
        mol_b = CrystalMolecule(atoms=ase_b, check_pbc=False)
        crystal_b = MolecularCrystal(lat_b, [mol_b], [True, True, True])

        return crystal_a, crystal_b

    def test_endpoints_match(self, simple_pair):
        from molcrys_kit.operations.interpolation import interpolate_crystal_vc

        crystal_a, crystal_b = simple_pair
        images = interpolate_crystal_vc(
            crystal_a, crystal_b, n_images=5, include_endpoints=True
        )
        # First image lattice = lat_a
        np.testing.assert_allclose(images[0].lattice, crystal_a.lattice, atol=1e-8)
        # Last image lattice = lat_b
        np.testing.assert_allclose(images[-1].lattice, crystal_b.lattice, atol=1e-8)

    def test_image_count(self, simple_pair):
        from molcrys_kit.operations.interpolation import interpolate_crystal_vc

        crystal_a, crystal_b = simple_pair
        images = interpolate_crystal_vc(
            crystal_a, crystal_b, n_images=11, include_endpoints=True
        )
        assert len(images) == 11

    def test_volume_changes_smoothly(self, simple_pair):
        from molcrys_kit.operations.interpolation import interpolate_crystal_vc

        crystal_a, crystal_b = simple_pair
        images = interpolate_crystal_vc(
            crystal_a, crystal_b, n_images=11, include_endpoints=True
        )
        volumes = [volume_of_cell(np.asarray(img.lattice)) for img in images]
        # Should be monotonically increasing (isotropic expansion)
        for i in range(len(volumes) - 1):
            assert volumes[i + 1] > volumes[i]

    def test_no_endpoints(self, simple_pair):
        from molcrys_kit.operations.interpolation import interpolate_crystal_vc

        crystal_a, crystal_b = simple_pair
        images = interpolate_crystal_vc(
            crystal_a, crystal_b, n_images=5, include_endpoints=False
        )
        assert len(images) == 5
        # Interior frames should differ from both endpoints
        lat_0 = np.asarray(images[0].lattice)
        assert not np.allclose(lat_0, crystal_a.lattice, atol=1e-6)
        assert not np.allclose(lat_0, crystal_b.lattice, atol=1e-6)

    def test_pbc_crossing_molecule(self):
        """Molecule COM near PBC boundary interpolates via shortest path."""
        from ase import Atoms as AseAtoms

        from molcrys_kit.operations.interpolation import interpolate_crystal_vc
        from molcrys_kit.structures.crystal import MolecularCrystal
        from molcrys_kit.structures.molecule import CrystalMolecule

        lat_a = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        lat_b = np.array([[11.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 11.0]])

        # Molecule A near top of cell: frac COM ~ (0.95, 0.5, 0.5)
        pos_a = np.array([[9.5, 5.0, 5.0], [9.5, 5.5, 5.0], [10.0, 5.0, 5.0]])
        # Molecule B near bottom of cell: frac COM ~ (0.05, 0.5, 0.5)
        # Shortest path crosses the PBC boundary, NOT through 0.5
        pos_b = np.array([[0.55, 5.5, 5.5], [0.55, 6.0, 5.5], [1.05, 5.5, 5.5]])

        mol_a = CrystalMolecule(atoms=AseAtoms("OHH", positions=pos_a), check_pbc=False)
        crystal_a = MolecularCrystal(lat_a, [mol_a], [True, True, True])

        mol_b = CrystalMolecule(atoms=AseAtoms("OHH", positions=pos_b), check_pbc=False)
        crystal_b = MolecularCrystal(lat_b, [mol_b], [True, True, True])

        images = interpolate_crystal_vc(
            crystal_a, crystal_b, n_images=5, include_endpoints=True
        )
        # Midpoint COM x-coordinate should be near the boundary (~10 or ~0),
        # NOT near 5.0 (which would be the long non-MIC path)
        mid_pos = np.asarray(images[2].molecules[0].get_positions())
        mid_com_x = np.mean(mid_pos[:, 0])
        mid_lat = np.asarray(images[2].lattice)
        mid_frac_x = mid_com_x / mid_lat[0, 0]
        # Should be > 0.9 or < 0.1 (near boundary), not ~0.5
        assert mid_frac_x > 0.9 or mid_frac_x < 0.1, (
            f"Midpoint frac COM x = {mid_frac_x:.3f}; expected near PBC boundary"
        )

    def test_endpoint_molecular_positions(self, simple_pair):
        """Endpoint images reproduce molecular positions of crystal_a and crystal_b."""
        from molcrys_kit.operations.interpolation import interpolate_crystal_vc

        crystal_a, crystal_b = simple_pair
        images = interpolate_crystal_vc(
            crystal_a, crystal_b, n_images=5, include_endpoints=True
        )

        # First image positions == crystal_a positions
        pos_a_original = np.asarray(crystal_a.molecules[0].get_positions())
        pos_a_image = np.asarray(images[0].molecules[0].get_positions())
        np.testing.assert_allclose(pos_a_image, pos_a_original, atol=1e-8)

        # Last image positions == crystal_b positions
        pos_b_original = np.asarray(crystal_b.molecules[0].get_positions())
        pos_b_image = np.asarray(images[-1].molecules[0].get_positions())
        np.testing.assert_allclose(pos_b_image, pos_b_original, atol=1e-6)
