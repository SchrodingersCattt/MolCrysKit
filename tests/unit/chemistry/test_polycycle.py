from __future__ import annotations

import pytest

from molcrys_kit.chemistry import notations_equivalent
from molcrys_kit.chemistry.name_conversion import iupac_to_smiles, smiles_to_iupac


@pytest.mark.parametrize(
    ("smiles", "name"),
    (
        ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),
        ("C1CCC2CCCCC2C1", "bicyclo[4.4.0]decane"),
        # This input has four- and five-member rings sharing one atom.
        ("C1CCC12CCCC2", "spiro[4.3]octane"),
        ("c1ccc2ccccc2c1", "bicyclo[4.4.0]dec-1,3,5,7,9-pentaene"),
        ("c1ccc(-c2ccccc2)cc1", "phenylbenzene"),
    ),
)
def test_polycycle_names_round_trip(smiles: str, name: str) -> None:
    result = smiles_to_iupac(smiles)
    assert result.name == name
    rebuilt = iupac_to_smiles(name)
    assert rebuilt.lossless is True
    assert notations_equivalent(smiles, rebuilt.value) is True


def test_polycycle_names_are_general_iupac() -> None:
    result = smiles_to_iupac("C1CC2CCC1C2")
    assert result.preferred is False
    assert result.kind.value == "general_iupac_name"
