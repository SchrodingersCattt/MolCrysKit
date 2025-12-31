"""
Verification script for Phase 1: Pure CIF Scanner for disorder handling.

This script tests the scan_cif_disorder function to ensure it extracts
raw data correctly without applying any logic or fixing inconsistencies.
"""

import numpy as np
from molcrys_kit.io.cif import scan_cif_disorder


def test_pap_m5_disorder():
    """Test PAP-M5.cif which has explicit disorder groups."""
    filepath = "examples/PAP-M5.cif"
    disorder_info = scan_cif_disorder(filepath)

    print("Testing PAP-M5.cif:")
    disorder_info.summary()

    # General assertions
    assert len(disorder_info.labels) > 0, "Should have at least one atom"
    assert len(disorder_info.labels) == len(
        disorder_info.symbols
    ), "Labels and symbols should have same length"
    assert len(disorder_info.labels) == len(
        disorder_info.occupancies
    ), "Labels and occupancies should have same length"
    assert len(disorder_info.labels) == len(
        disorder_info.disorder_groups
    ), "Labels and disorder groups should have same length"

    # Check that labels retain suffixes
    labels_with_suffixes = [
        label
        for label in disorder_info.labels
        if any(
            c.isalpha() and c != label[0] or c.isdigit() and label.index(c) > 0
            for c in label[1:]
        )
    ]
    assert len(labels_with_suffixes) > 0, "Should have labels with suffixes preserved"
    print(f"Found {len(labels_with_suffixes)} labels with suffixes")

    # Case-specific: Verify disorder_groups contains non-zero integers
    non_zero_groups = [group for group in disorder_info.disorder_groups if group != 0]
    assert len(non_zero_groups) > 0, "PAP-M5.cif should have non-zero disorder groups"
    print(f"Found {len(non_zero_groups)} atoms with non-zero disorder groups")

    print("PAP-M5.cif test passed!\n")


def test_dap_4_disorder():
    """Test DAP-4.cif which has implicit disorder (suffixes, occupancy < 1.0) but likely all groups=0."""
    filepath = "examples/DAP-4.cif"
    disorder_info = scan_cif_disorder(filepath)

    print("Testing DAP-4.cif:")
    disorder_info.summary()

    # General assertions
    assert len(disorder_info.labels) > 0, "Should have at least one atom"
    assert len(disorder_info.labels) == len(
        disorder_info.symbols
    ), "Labels and symbols should have same length"
    assert len(disorder_info.labels) == len(
        disorder_info.occupancies
    ), "Labels and occupancies should have same length"
    assert len(disorder_info.labels) == len(
        disorder_info.disorder_groups
    ), "Labels and disorder groups should have same length"

    # Check that labels retain suffixes
    labels_with_suffixes = [
        label
        for label in disorder_info.labels
        if any(
            c.isalpha() and c != label[0] or c.isdigit() and label.index(c) > 0
            for c in label[1:]
        )
    ]
    assert len(labels_with_suffixes) > 0, "Should have labels with suffixes preserved"
    print(f"Found {len(labels_with_suffixes)} labels with suffixes")

    # Case-specific: Verify disorder_groups are all 0 (reflecting raw file reality)
    non_zero_groups = [group for group in disorder_info.disorder_groups if group != 0]
    assert (
        len(non_zero_groups) == 0
    ), f"DAP-4.cif should have all disorder groups equal to 0, but found {len(non_zero_groups)} non-zero groups"
    print("All disorder groups are 0 as expected")

    # Case-specific: Verify occupancies are ~0.5
    low_occupancy_atoms = [occ for occ in disorder_info.occupancies if occ < 1.0]
    assert (
        len(low_occupancy_atoms) > 0
    ), "DAP-4.cif should have atoms with occupancy < 1.0"
    print(f"Found {len(low_occupancy_atoms)} atoms with occupancy < 1.0")

    # Check if there are atoms with occupancy around 0.5 (common for disordered structures)
    approx_half_occupancy = [
        occ for occ in disorder_info.occupancies if 0.4 <= occ <= 0.6
    ]
    if len(approx_half_occupancy) > 0:
        print(f"Found {len(approx_half_occupancy)} atoms with ~0.5 occupancy")

    print("DAP-4.cif test passed!\n")


def test_general_functionality():
    """Test general functionality of the disorder scanner."""
    filepath = "examples/PAP-M5.cif"  # Using PAP-M5 as a general test case
    disorder_info = scan_cif_disorder(filepath)

    # Verify that frac_coords is a proper nx3 array
    assert isinstance(
        disorder_info.frac_coords, np.ndarray
    ), "frac_coords should be a numpy array"
    assert (
        disorder_info.frac_coords.shape[1] == 3
    ), "frac_coords should have 3 columns for x, y, z"
    assert len(disorder_info.frac_coords) == len(
        disorder_info.labels
    ), "frac_coords should have same number of rows as labels"

    # Verify occupancies are floats between 0 and 1
    for occ in disorder_info.occupancies:
        assert isinstance(occ, float), "Occupancy should be a float"
        assert 0 <= occ <= 1, f"Occupancy {occ} should be between 0 and 1"

    # Verify disorder groups are integers >= 0
    for group in disorder_info.disorder_groups:
        assert isinstance(group, int), "Disorder group should be an integer"
        assert group >= 0, f"Disorder group {group} should be non-negative"

    print("General functionality test passed!\n")


if __name__ == "__main__":
    print("Starting Phase 1: Pure CIF Scanner verification tests...\n")

    test_general_functionality()
    test_pap_m5_disorder()
    test_dap_4_disorder()

    print("All tests passed! The Pure CIF Scanner is working correctly.")
