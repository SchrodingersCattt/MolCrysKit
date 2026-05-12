"""Tests for topology-gated CShM shape classification."""

import os

import numpy as np
import pytest

from molcrys_kit.analysis.shape import classify_shell, cshm, topology_signature
from molcrys_kit.structures.polyhedra import get_polyhedron, list_polyhedra

SY_BX10_UNIT = np.array(
    [
        [-0.010183, 0.585285, 0.810764],
        [-0.009656, 0.704126, -0.710010],
        [-0.007929, -0.874757, -0.484496],
        [0.836505, -0.218096, -0.502686],
        [-0.840957, -0.215366, -0.496396],
        [0.817665, -0.177084, 0.547782],
        [-0.822472, -0.174965, 0.541227],
        [-0.007013, -0.881995, 0.471206],
        [0.772137, 0.635413, -0.007389],
        [-0.777679, 0.628619, -0.007310],
    ],
    dtype=float,
)

EAP4_AX11_UNIT = np.array(
    [
        [0.737505, 0.547393, -0.395534],
        [-0.737505, -0.547393, -0.395534],
        [0.000000, 0.000000, -1.000000],
        [0.516923, 0.503351, 0.692408],
        [-0.516923, -0.503351, 0.692408],
        [0.675385, -0.497826, 0.544081],
        [-0.675385, 0.497826, 0.544081],
        [0.631204, -0.618904, -0.467482],
        [-0.631204, 0.618904, -0.467482],
        [0.000000, -0.997303, 0.073398],
        [0.000000, 0.997303, 0.073398],
    ],
    dtype=float,
)


def _rotation(seed=7):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_self_cshm_is_zero_for_registered_ideals():
    for poly in list_polyhedra():
        result = cshm(poly.vertices, poly.vertices, n_random_inits=1)
        assert result["cshm"] < 1e-6, f"{poly.name}: {result['cshm']}"


def test_classify_shell_is_rotation_invariant_for_cube():
    cube = get_polyhedron("cube")
    rotated = cube.vertices @ _rotation()

    baseline = classify_shell(cube.vertices, n_random_inits=2)
    result = classify_shell(rotated, n_random_inits=2)

    assert result["primary_label"] == baseline["primary_label"] == "cube"
    assert result["label_modifier"] == baseline["label_modifier"] == "clean"
    np.testing.assert_allclose(result["cshm_value"], baseline["cshm_value"], atol=1e-8)


def test_cshm_is_permutation_invariant_for_cuboctahedron():
    cuboctahedron = get_polyhedron("cuboctahedron")
    rng = np.random.default_rng(11)
    permuted = cuboctahedron.vertices[rng.permutation(cuboctahedron.cn)]

    baseline = cshm(cuboctahedron.vertices, cuboctahedron.vertices, n_random_inits=2)
    result = cshm(permuted, cuboctahedron.vertices, n_random_inits=2)

    np.testing.assert_allclose(result["cshm"], baseline["cshm"], atol=1e-8)


def test_cshm_increases_with_radial_noise_for_cube():
    cube = get_polyhedron("cube")
    rng = np.random.default_rng(23)
    noise = rng.normal(size=cube.vertices.shape)
    noise -= np.sum(noise * cube.vertices, axis=1, keepdims=True) * cube.vertices

    values = []
    for sigma in (0.02, 0.05, 0.10):
        distorted = cube.vertices + sigma * noise
        result = classify_shell(distorted, n_random_inits=4, topology_match=False)
        assert result["primary_label"] == "cube"
        values.append(result["cshm_value"])

    assert values[0] < values[1] < values[2]


def test_topology_gate_keeps_octahedron_candidate_specific():
    octahedron = get_polyhedron("octahedron")
    signature = topology_signature(octahedron.vertices)
    result = classify_shell(octahedron.vertices, n_random_inits=2)

    assert signature["face_signature"] == {3: 8}
    assert signature["edge_count"] == 12
    assert result["topology"]["gate"] == "matched"
    assert result["topology"]["candidate_names"] == ["octahedron"]
    assert result["primary_label"] == "octahedron"


@pytest.mark.parametrize(
    "case_name, ideal_name, expected_modifier",
    [
        ("DAP-4 BX6", "octahedron", "clean"),
        ("DAP-4 AX12", "cuboctahedron", "clean"),
        ("EAP-4 AX11", "tricapped_cube", "clean"),
        ("PEP/SY/MPEP/HPEP BX10", "bicapped_square_antiprism", "clean"),
    ],
)
def test_mix_regression_reference_labels(case_name, ideal_name, expected_modifier):
    """Pin the selected MIX labels against ideal reference shells.

    The production CIF-to-fragment extraction happens downstream in MatterVis;
    these cases pin the labels expected once the fragment shell reaches the new
    CShM classifier. Optional example CIFs are checked below when available.
    """
    poly = get_polyhedron(ideal_name)
    assert poly is not None, f"{case_name}: missing registered ideal {ideal_name}"
    result = classify_shell(poly.vertices, n_random_inits=2)
    assert result["primary_label"] == ideal_name
    assert result["label_modifier"] == expected_modifier


def test_sy_bx10_decomposes_to_bicapped_cube():
    result = classify_shell(SY_BX10_UNIT, n_random_inits=4)

    assert result["primary_label"] == "bicapped_cube"
    assert result["label_modifier"] == "distorted"
    assert result["label_source"] == "registered"
    assert result["core"]["prototype"] == "cube"
    assert result["core"]["cshm"] < 1.0
    assert len(result["residuals"]) == 2
    assert {res["role"] for res in result["residuals"]} <= {"face_cap", "off_axis_cap"}


def test_eap4_ax11_decomposes_to_tricapped_cube():
    result = classify_shell(EAP4_AX11_UNIT, n_random_inits=4)

    assert result["primary_label"] == "tricapped_cube"
    assert result["label_modifier"] in {"clean", "distorted"}
    assert result["core"]["prototype"] == "cube"
    assert len(result["residuals"]) == 3
    assert {res["role"] for res in result["residuals"]} <= {"face_cap", "off_axis_cap"}


def test_dap4_ax12_keeps_rigid_cuboctahedron_at_k0():
    cuboctahedron = get_polyhedron("cuboctahedron")
    result = classify_shell(cuboctahedron.vertices, n_random_inits=2)

    assert result["primary_label"] == "cuboctahedron"
    assert result["core"]["prototype"] == "cuboctahedron"
    assert result["diagnostics"]["k_used"] == 0
    assert result["residuals"] == []


def test_ad_hoc_label_when_residual_roles_unusual():
    cube = get_polyhedron("cube")
    shell = np.vstack([cube.vertices, np.array([[0.0, 0.0, 1.0]])])
    result = classify_shell(shell, n_random_inits=2)

    assert result["core"]["prototype"] == "cube"
    assert result["label_source"] == "ad_hoc"
    assert "cube" in result["primary_label"]
    assert result["structural_description"]


def test_search_mode_evaluates_greedy_for_k4():
    octahedron = get_polyhedron("octahedron")
    extra = np.array(
        [[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
        dtype=float,
    )
    shell = np.vstack([octahedron.vertices, extra])
    result = classify_shell(shell, n_random_inits=1, max_strip=4, top_k=0)

    assert "greedy_2swap" in result["diagnostics"]["search_modes_evaluated"]
    assert any(
        alt["diagnostics"]["k_used"] == 4
        and alt["diagnostics"]["search_mode"] == "greedy_2swap"
        for alt in result["alternatives"]
    )


def test_optional_cif_fixtures_are_available_for_manual_regression():
    cif_data_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "cif")
    )
    required = [
        "DAP-4.cif",
        "298k-SY.cif",
        "298k-PEP.cif",
        "298K-MPEP.cif",
        "298k-HPEP.cif",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(cif_data_dir, name))]
    if missing:
        pytest.skip(f"optional MIX CIF fixtures are not installed: {missing}")
    for name in required:
        assert os.path.getsize(os.path.join(cif_data_dir, name)) > 0
