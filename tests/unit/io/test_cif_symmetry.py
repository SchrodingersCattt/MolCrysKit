"""Public CIF symmetry parsing across operation types."""

import numpy as np
import pytest

from molcrys_kit.io import read_cif_symmetry, scan_cif_disorder
from molcrys_kit.io.cif import SymmetryAutoExpandedWarning


def cif(symmetry_lines: str, *, number: int = 1, symbol: str = "P 1") -> str:
    return f"""data_test
_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number {number}
_space_group_name_H-M_alt '{symbol}'
loop_
_space_group_symop_operation_xyz
{symmetry_lines}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
"""


def test_public_parser_preserves_explicit_screw_and_inversion():
    parsed = read_cif_symmetry(
        cif_text=cif("'x,y,z'\n'-x,y+1/2,-z'\n'-x,-y,-z'", number=1),
        expand_symmetry=False,
    )
    assert len(parsed.operations) == 3
    assert parsed.operations[1].is_proper
    assert parsed.operations[2].is_improper
    np.testing.assert_allclose(parsed.operations[1].translation, [0, 0.5, 0])
    assert parsed.operations[1].xyz == "-x,y+1/2,-z"
    assert parsed.source == "_space_group_symop_operation_xyz"


def test_explicit_loop_is_kept_when_it_disagrees_with_the_declared_group():
    """A non-identity loop is not replaced by the declared space group.

    Identity-only non-P1 input is expanded from the IT number. A loop that
    already lists several operations is the symmetry, even when it is not
    closed and not that group. Expansion of a special-position site must
    follow the written operations.
    """

    def block(operations: str) -> str:
        return f"""data_test
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 2
_space_group_name_H-M_alt 'P -1'
loop_
_space_group_symop_operation_xyz
{operations}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0 0 0 1
"""

    wrong = block("'x, y, z'\n'-x, -y, z+1/2'")
    right = block("'x, y, z'\n'-x, -y, -z'")
    parsed = read_cif_symmetry(cif_text=wrong, expand_symmetry=True)
    assert parsed.expanded_from_declaration is False
    assert parsed.space_group_number == 2
    assert len(parsed.operations) == 2
    np.testing.assert_allclose(parsed.operations[1].translation, [0.0, 0.0, 0.5])
    assert len(scan_cif_disorder(cif_text=wrong).symbols) == 2
    assert len(scan_cif_disorder(cif_text=right).symbols) == 1


def test_identity_only_non_p1_declaration_auto_expands():
    with pytest.warns(SymmetryAutoExpandedWarning):
        parsed = read_cif_symmetry(cif_text=cif("'x,y,z'", number=2, symbol="P -1"))
    assert parsed.expanded_from_declaration
    assert parsed.space_group_number == 2
    assert len(parsed.operations) == 2
    assert any(operation.is_improper for operation in parsed.operations)


def test_strict_parser_rejects_malformed_operation():
    text = cif("'x,y,z'\n'not-an-operation'", number=1)
    with pytest.raises(ValueError, match="invalid symmetry operation"):
        read_cif_symmetry(cif_text=text, expand_symmetry=False, strict=True)


def test_strict_parser_rejects_incomplete_non_p1_operation_loop():
    text = cif("'x,y,z'", number=14, symbol="P 21/c")
    with pytest.raises(ValueError, match="incomplete non-P1 symmetry operation loop"):
        read_cif_symmetry(cif_text=text, expand_symmetry=False, strict=True)


def test_non_strict_parser_skips_malformed_operation():
    text = cif("'x,y,z'\n'not-an-operation'", number=1)
    parsed = read_cif_symmetry(cif_text=text, expand_symmetry=False, strict=False)
    assert len(parsed.operations) == 1


@pytest.mark.parametrize("number", range(1, 231))
def test_all_space_groups_have_inverses_and_generator_closure(number):
    parsed = read_cif_symmetry(
        cif_text=cif("'x,y,z'", number=number), expand_symmetry=True
    )
    operations = parsed.operations
    identity = operations[0]
    assert any(identity.equivalent_to(item) for item in operations)
    generator = operations[1] if len(operations) > 1 else operations[0]
    for operation in operations:
        assert any(operation.inverse().equivalent_to(item) for item in operations)
        product = generator.compose(operation)
        assert any(product.equivalent_to(item) for item in operations)
