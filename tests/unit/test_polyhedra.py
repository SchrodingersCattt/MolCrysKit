"""
Comprehensive tests for molcrys_kit.structures.polyhedra.

Tests cover:
  - Registry completeness (CN=4 through CN=12)
  - Vertex normalization (all on unit sphere)
  - Correct vertex count per CN
  - Self-RMSD = 0 for all ideal polyhedra
  - Cross-RMSD discrimination (each polyhedron best-matches itself)
  - No duplicate names in registry
  - Metadata correctness (point_group, category)
  - Public API backward compatibility
  - IdealPolyhedron dataclass behavior
"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from molcrys_kit.structures.polyhedra import (
    IDEAL_POLYHEDRA,
    IdealPolyhedron,
    all_ideal_polyhedra,
    convex_hull_payload,
    get_polyhedron,
    ideal_polyhedra_for_cn,
    list_polyhedra,
)
from molcrys_kit.analysis.packing_shell import angular_rmsd_vs_ideals

EXPECTED_POLYHEDRA = {
    4: {"tetrahedron", "square_planar"},
    5: {"trigonal_bipyramid", "square_pyramid"},
    6: {"octahedron", "trigonal_prism"},
    7: {"pentagonal_bipyramid", "capped_octahedron"},
    8: {"cube", "square_antiprism", "dodecahedron"},
    9: {"tricapped_trigonal_prism", "capped_square_antiprism"},
    10: {"bicapped_square_antiprism", "bicapped_dodecahedron"},
    11: {
        "capped_pentagonal_antiprism",
        "capped_pentagonal_prism",
        "edge_bicapped_square_antiprism",
        "tricapped_cube",
    },
    12: {"icosahedron", "cuboctahedron"},
}


# --- Fixtures ---


@pytest.fixture
def all_polys():
    """All registered polyhedra as a flat list."""
    return list_polyhedra()


@pytest.fixture(params=list_polyhedra(), ids=lambda p: f"CN{p.cn}_{p.name}")
def poly(request):
    """Parametric fixture: one polyhedron at a time."""
    return request.param


# --- Registry completeness ---


class TestRegistryCompleteness:
    """Ensure all expected coordination numbers are covered."""

    def test_cn_range_4_to_12(self):
        actual_cns = set(IDEAL_POLYHEDRA.keys())
        assert (
            set(EXPECTED_POLYHEDRA) <= actual_cns
        ), f"Missing CNs: {set(EXPECTED_POLYHEDRA) - actual_cns}"

    def test_expected_catalog_entries_are_registered(self):
        """Each expected CN exposes its known candidate polyhedra."""
        for cn, expected_names in EXPECTED_POLYHEDRA.items():
            actual_names = set(IDEAL_POLYHEDRA[cn])
            assert (
                expected_names <= actual_names
            ), f"CN={cn} missing polyhedra: {expected_names - actual_names}"

    def test_no_duplicate_names(self, all_polys):
        names = [p.name for p in all_polys]
        assert len(names) == len(
            set(names)
        ), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_total_count(self, all_polys):
        """Sanity check: catalog has at least the expected entries."""
        expected_count = sum(len(names) for names in EXPECTED_POLYHEDRA.values())
        assert len(all_polys) >= expected_count


# --- Vertex normalization ---


class TestNormalization:
    """All vertices should lie on the unit sphere (centered at origin)."""

    def test_vertices_on_unit_sphere(self, poly):
        norms = np.linalg.norm(poly.vertices, axis=1)
        np.testing.assert_allclose(
            norms,
            np.ones(poly.cn),
            atol=1e-12,
            err_msg=f"{poly.name}: vertices not on unit sphere",
        )

    def test_centroid_near_origin(self, poly):
        """Centroid should be near origin; capped polyhedra may drift slightly."""
        centroid = poly.vertices.mean(axis=0)
        offset = np.linalg.norm(centroid)
        # Symmetric polyhedra: centroid exactly at origin
        # Capped/asymmetric polyhedra: centroid may drift (up to ~0.2 for capped)
        if poly.category in (
            "platonic",
            "archimedean",
            "prism",
            "antiprism",
            "bipyramid",
        ):
            np.testing.assert_allclose(
                centroid,
                np.zeros(3),
                atol=1e-8,
                err_msg=f"{poly.name}: centroid not at origin",
            )
        else:
            # Capped polyhedra: just ensure offset is bounded
            assert offset < 0.3, f"{poly.name}: centroid offset {offset:.4f} too large"

    def test_vertex_count_matches_cn(self, poly):
        assert poly.vertices.shape == (poly.cn, 3)


# --- Angular RMSD self-match ---


class TestSelfRMSD:
    """Each ideal polyhedron should give RMSD=0 when scored against itself."""

    def test_self_rmsd_is_zero(self, poly):
        result = angular_rmsd_vs_ideals(poly.vertices)
        assert result["coordination_number"] == poly.cn
        best = result["best_match"]
        assert best is not None, f"{poly.name}: no match found"
        assert best["name"] == poly.name, (
            f"{poly.name}: best match is '{best['name']}' "
            f"(RMSD={best['angular_rmsd']:.4f} deg)"
        )
        assert (
            best["angular_rmsd"] < 1e-8
        ), f"{poly.name}: self-RMSD = {best['angular_rmsd']:.2e}, expected ~0"


# --- Cross-RMSD discrimination ---


class TestCrossRMSD:
    """Different polyhedra of the same CN should be distinguishable."""

    @pytest.mark.parametrize("cn", [4, 5, 6, 7, 8, 9, 10, 11, 12])
    def test_different_polyhedra_have_nonzero_rmsd(self, cn):
        polys_for_cn = ideal_polyhedra_for_cn(cn)
        names = list(polys_for_cn.keys())
        if len(names) < 2:
            pytest.skip(f"CN={cn} has only 1 polyhedron")
        # Score first polyhedron
        verts0 = polys_for_cn[names[0]]
        result = angular_rmsd_vs_ideals(verts0)
        # Best match must be itself (RMSD~0)
        assert result["best_match"]["name"] == names[0]
        # Second-best must have RMSD > 0
        if len(result["results"]) > 1:
            second = result["results"][1]
            assert second["angular_rmsd"] > 0.1, (
                f"CN={cn}: {names[0]} vs {second['name']} "
                f"RMSD={second['angular_rmsd']:.4f} deg - too similar"
            )


# --- Specific geometry checks ---


class TestSpecificGeometry:
    """Check well-known angular relationships for selected polyhedra."""

    def test_tetrahedron_angles_are_109_47(self):
        """All inter-vertex angles from center should be arccos(-1/3)."""
        verts = ideal_polyhedra_for_cn(4)["tetrahedron"]
        angles = []
        for i in range(4):
            for j in range(i + 1, 4):
                cosang = np.clip(np.dot(verts[i], verts[j]), -1.0, 1.0)
                angles.append(math.degrees(math.acos(cosang)))
        expected = math.degrees(math.acos(-1.0 / 3.0))  # 109.4712 degrees
        np.testing.assert_allclose(angles, [expected] * 6, atol=0.01)

    def test_octahedron_has_90_and_180_angles(self):
        """Octahedron: 12 angles at 90 degrees and 3 at 180 degrees."""
        verts = ideal_polyhedra_for_cn(6)["octahedron"]
        angles = []
        for i in range(6):
            for j in range(i + 1, 6):
                cosang = np.clip(np.dot(verts[i], verts[j]), -1.0, 1.0)
                angles.append(round(math.degrees(math.acos(cosang)), 1))
        assert angles.count(90.0) == 12
        assert angles.count(180.0) == 3

    def test_square_planar_is_coplanar(self):
        """Square planar CN=4: all vertices in one plane."""
        verts = ideal_polyhedra_for_cn(4)["square_planar"]
        # All z-coords should be 0 (or all in same plane)
        _, s, _ = np.linalg.svd(verts - verts.mean(axis=0))
        # Smallest singular value should be ~0 (planar)
        assert s[-1] < 1e-10

    def test_tricapped_cube_has_c3_axis_along_body_diagonal(self):
        """Tricapped cube CN=11: 120 deg rotation about (1,1,1) is a symmetry."""
        verts = ideal_polyhedra_for_cn(11)["tricapped_cube"]
        # Rotation by 120 deg around the body-diagonal cycles (x,y,z) -> (y,z,x).
        rotated = verts[:, [1, 2, 0]]
        # Match each rotated vertex to a vertex in the original set.
        for v in rotated:
            distances = np.linalg.norm(verts - v, axis=1)
            assert distances.min() < 1e-9, (
                f"tricapped_cube: rotated vertex {v} has no partner; "
                f"min distance {distances.min():.2e}"
            )


# --- Metadata ---


class TestMetadata:
    """Check metadata consistency."""

    def test_point_group_is_set(self, poly):
        assert poly.point_group != "?", f"{poly.name}: point_group not set"

    def test_category_is_valid(self, poly):
        valid_categories = {
            "platonic",
            "archimedean",
            "prism",
            "antiprism",
            "bipyramid",
            "capped",
            "other",
        }
        assert (
            poly.category in valid_categories
        ), f"{poly.name}: invalid category '{poly.category}'"

    def test_platonic_have_oh_td_or_ih(self, all_polys):
        """Platonic solids must have Oh, Td, or Ih symmetry."""
        platonics = [p for p in all_polys if p.category == "platonic"]
        for p in platonics:
            assert p.point_group in (
                "Oh",
                "Td",
                "Ih",
            ), f"Platonic {p.name} has unexpected point_group={p.point_group}"


# --- Public API ---


class TestPublicAPI:
    """Backward-compatible public API."""

    def test_ideal_polyhedra_for_cn_returns_dict(self):
        result = ideal_polyhedra_for_cn(8)
        assert isinstance(result, dict)
        assert "cube" in result
        assert isinstance(result["cube"], np.ndarray)

    def test_ideal_polyhedra_for_cn_empty_for_unknown(self):
        assert ideal_polyhedra_for_cn(99) == {}

    def test_all_ideal_polyhedra_returns_nested_dict(self):
        result = all_ideal_polyhedra()
        assert isinstance(result, dict)
        assert 8 in result
        assert "cube" in result[8]

    def test_IDEAL_POLYHEDRA_module_level_accessible(self):
        assert isinstance(IDEAL_POLYHEDRA, dict)
        assert 4 in IDEAL_POLYHEDRA
        assert 12 in IDEAL_POLYHEDRA

    def test_get_polyhedron_found(self):
        p = get_polyhedron("octahedron")
        assert p is not None
        assert p.cn == 6
        assert p.point_group == "Oh"

    def test_get_polyhedron_not_found(self):
        assert get_polyhedron("nonexistent") is None

    def test_list_polyhedra_returns_list_of_dataclass(self):
        polys = list_polyhedra()
        assert len(polys) > 0
        assert all(isinstance(p, IdealPolyhedron) for p in polys)

    def test_convex_hull_payload_minimal(self):
        # Less than 4 points -> no hull
        coords = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = convex_hull_payload(coords)
        assert result["simplices"] == []
        assert result["edges"] == []
        assert len(result["vertices"]) == 3

    def test_convex_hull_payload_cube(self):
        coords = ideal_polyhedra_for_cn(8)["cube"]
        result = convex_hull_payload(coords)
        assert len(result["vertices"]) == 8
        assert len(result["simplices"]) > 0
        # Triangulated cube: 6 faces x 2 triangles = 12 triangles,
        # 12 face edges + 6 diagonal edges = 18 unique edges
        assert len(result["edges"]) >= 12


# --- IdealPolyhedron dataclass ---


class TestIdealPolyhedronDataclass:
    """Test the IdealPolyhedron dataclass behavior."""

    def test_frozen(self):
        p = get_polyhedron("cube")
        with pytest.raises(FrozenInstanceError):
            p.name = "not_a_cube"

    def test_repr_does_not_include_vertices(self):
        p = get_polyhedron("cube")
        r = repr(p)
        # vertices field has repr=False
        assert "vertices" not in r
        assert "array(" not in r

    def test_invalid_shape_raises_value_error(self):
        with pytest.raises(ValueError):
            IdealPolyhedron(
                name="invalid",
                cn=4,
                point_group="?",
                category="other",
                vertices=np.zeros((3, 3)),
            )
