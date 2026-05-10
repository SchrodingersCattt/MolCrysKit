"""Tests for topology-gated CShM shape classification."""

import os

import numpy as np
import pytest

from molcrys_kit.analysis.shape import classify_shell, cshm, topology_signature
from molcrys_kit.structures.polyhedra import get_polyhedron, list_polyhedra


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


def test_optional_example_cifs_are_available_for_manual_regression():
    examples_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples")
    )
    required = [
        "DAP-4.cif",
        "298k-SY.cif",
        "298k-PEP.cif",
        "298K-MPEP.cif",
        "298k-HPEP.cif",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(examples_dir, name))]
    if missing:
        pytest.skip(f"optional MIX example CIFs are not installed: {missing}")
    for name in required:
        assert os.path.getsize(os.path.join(examples_dir, name)) > 0
