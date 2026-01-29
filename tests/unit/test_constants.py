"""
Unit tests for molcrys_kit.constants and config (coverage).
"""

import pytest

from molcrys_kit.constants import (
    get_atomic_mass,
    get_atomic_radius,
    has_atomic_mass,
    has_atomic_radius,
    list_elements_with_data,
    is_metal_element,
    METAL_THRESHOLD_FACTOR,
    NON_METAL_THRESHOLD_FACTOR,
    DEFAULT_NEIGHBOR_CUTOFF,
)
from molcrys_kit.constants.config import (
    KEY_OCCUPANCY,
    KEY_DISORDER_GROUP,
    KEY_ASSEMBLY,
    KEY_LABEL,
    TRANSITION_METALS,
    BONDING_CONFIG,
    BOND_LENGTHS,
    COORDINATION_RULES,
    COMMON_SOLVENTS,
)


class TestConstantsInit:
    """Constants module exports and helpers."""

    def test_atomic_mass(self):
        assert get_atomic_mass("H") > 0
        assert get_atomic_mass("C") > 0
        assert get_atomic_mass("O") > 0

    def test_atomic_radius(self):
        assert get_atomic_radius("H") > 0
        assert get_atomic_radius("C") > 0

    def test_has_atomic_mass(self):
        assert has_atomic_mass("H") is True
        assert has_atomic_mass("Xx") is False

    def test_has_atomic_radius(self):
        assert has_atomic_radius("H") is True
        assert has_atomic_radius("Xx") is False

    def test_list_elements_with_data(self):
        data = list_elements_with_data()
        assert "masses" in data
        assert "radii" in data
        assert "H" in data["masses"]
        assert "H" in data["radii"]

    def test_is_metal_element(self):
        assert is_metal_element("Fe") is True
        assert is_metal_element("C") is False
        assert is_metal_element("H") is False

    def test_threshold_factors(self):
        assert METAL_THRESHOLD_FACTOR == 0.5
        assert NON_METAL_THRESHOLD_FACTOR == 1.25
        assert DEFAULT_NEIGHBOR_CUTOFF == 3.5


class TestConfig:
    """Config keys and maps."""

    def test_key_names(self):
        assert KEY_OCCUPANCY == "occupancy"
        assert KEY_DISORDER_GROUP == "disorder_group"
        assert KEY_ASSEMBLY == "assembly"
        assert KEY_LABEL == "label"

    def test_transition_metals_nonempty(self):
        assert len(TRANSITION_METALS) > 0
        assert "Fe" in TRANSITION_METALS
        assert "Cu" in TRANSITION_METALS

    def test_bonding_config(self):
        assert "METAL_THRESHOLD_FACTOR" in BONDING_CONFIG
        assert "NON_METAL_THRESHOLD_FACTOR" in BONDING_CONFIG
        assert "MAX_HYDROGEN_BOND_DISTANCE" in BONDING_CONFIG

    def test_bond_lengths(self):
        assert "C-H" in BOND_LENGTHS
        assert "O-H" in BOND_LENGTHS
        assert BOND_LENGTHS["O-H"] == 0.96

    def test_coordination_rules(self):
        assert "C" in COORDINATION_RULES
        assert COORDINATION_RULES["C"]["target_coordination"] == 4
        assert "N" in COORDINATION_RULES

    def test_common_solvents(self):
        assert "Water" in COMMON_SOLVENTS
        assert COMMON_SOLVENTS["Water"]["formula"] == "H2O"
        assert "DMF" in COMMON_SOLVENTS
