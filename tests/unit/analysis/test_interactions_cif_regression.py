"""CIF-based regression tests for the directional interaction detectors.

The fixtures are synthetic P1 crystals that reproduce representative contact
geometries without using coordinates from a licensed structural database.
"""

from collections import Counter
from pathlib import Path

import pytest

from molcrys_kit.analysis.interactions import (
    find_halogen_bonds,
    find_hydrogen_bonds,
    find_pi_stacking,
    interaction_profile,
)
from molcrys_kit.io import read_mol_crystal


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CIF_DIR = Path(__file__).resolve().parents[2] / "data" / "cif" / "interactions"


def _read_fixture(name: str):
    return read_mol_crystal(CIF_DIR / name)


def test_hydrogen_bond_regression_from_cif():
    crystal = _read_fixture("hydrogen_bond.cif")

    bonds = find_hydrogen_bonds(crystal)

    assert len(bonds) == 1
    bond = bonds[0]
    assert (bond.donor.symbol, bond.hydrogen.symbol, bond.acceptor.symbol) == (
        "O",
        "H",
        "O",
    )
    assert bond.h_acceptor_distance_A == pytest.approx(1.84, abs=0.01)
    assert bond.donor_acceptor_distance_A == pytest.approx(2.80, abs=0.01)
    assert bond.dha_angle_deg == pytest.approx(180.0, abs=0.1)
    assert bond.image == (0, 0, 0)
    assert bond.score is not None and bond.score > 0.9


def test_halogen_bond_regression_from_cif():
    crystal = _read_fixture("halogen_bond.cif")

    bonds = find_halogen_bonds(crystal)

    assert len(bonds) == 1
    bond = bonds[0]
    assert (bond.donor.symbol, bond.halogen.symbol, bond.acceptor.symbol) == (
        "C",
        "Cl",
        "O",
    )
    assert bond.x_acceptor_distance_A == pytest.approx(3.00, abs=0.01)
    assert bond.dxa_angle_deg == pytest.approx(180.0, abs=0.1)
    assert bond.image == (0, 0, 0)
    assert bond.score is not None and bond.score > 0.6


@pytest.mark.parametrize(
    ("fixture", "subtype", "distance_A", "normal_angle_deg", "offset_A"),
    [
        ("pi_parallel.cif", "face_centered_parallel", 3.413, 0.0, 0.30),
        ("pi_t_shape.cif", "T_shape", 4.20, 90.0, 0.0),
    ],
)
def test_pi_stacking_regression_from_cif(
    fixture: str,
    subtype: str,
    distance_A: float,
    normal_angle_deg: float,
    offset_A: float,
):
    crystal = _read_fixture(fixture)

    stacks = find_pi_stacking(crystal)

    assert len(stacks) == 1
    stack = stacks[0]
    assert stack.subtype == subtype
    assert stack.centroid_distance_A == pytest.approx(distance_A, abs=0.01)
    assert stack.normal_angle_deg == pytest.approx(normal_angle_deg, abs=0.1)
    assert stack.lateral_offset_A == pytest.approx(offset_A, abs=0.01)
    assert stack.image == (0, 0, 0)
    assert stack.score is not None and stack.score > 0.0


def test_interaction_profile_aggregates_multi_interaction_cif():
    crystal = _read_fixture("multi_interaction.cif")

    profile = interaction_profile(crystal)

    assert {key: summary.count for key, summary in profile.summaries.items()} == {
        "hydrogen_bond": 2,
        "halogen_bond": 1,
        "pi_stacking": 0,
    }
    assert Counter(interaction.kind for interaction in profile.interactions) == {
        "hydrogen_bond": 2,
        "halogen_bond": 1,
    }
    hydrogen = profile.summaries["hydrogen_bond"]
    halogen = profile.summaries["halogen_bond"]
    assert hydrogen.sum == pytest.approx(2 * hydrogen.mean)
    assert hydrogen.max == pytest.approx(hydrogen.mean)
    assert halogen.sum == pytest.approx(halogen.mean)
    assert halogen.max == pytest.approx(halogen.mean)
