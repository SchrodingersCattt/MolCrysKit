"""Synthetic, project-independent tests of declared-composition validation."""
from pathlib import Path
import logging

import pytest

from molcrys_kit.analysis.disorder import generate_ordered_replicas_from_disordered_sites as order

FIXTURE = Path("tests/data/cif/synthetic_water_orientation.cif")


def test_strict_formula_accepts_a_matching_molecule():
    result = order(str(FIXTURE), strict_formula=True)
    assert result[0].molecules[0].get_chemical_formula() == "H2O"


def test_formula_mismatch_warns_or_fails_closed(tmp_path, caplog):
    p = tmp_path / "mismatch.cif"
    p.write_text(FIXTURE.read_text().replace("'H2 O'", "'H4 O'"))
    with caplog.at_level(logging.WARNING):
        result = order(str(p))
    assert result
    assert "No ordered candidate matches the declared formula" in caplog.text
    with pytest.raises(ValueError, match="No ordered candidate matches"):
        order(str(p), strict_formula=True)


def test_strict_formula_fails_if_metadata_unavailable(tmp_path):
    p = tmp_path / "missing.cif"
    p.write_text(FIXTURE.read_text().replace("_chemical_formula_moiety 'H2 O'", ""))
    with pytest.raises(ValueError, match="requires a parseable formula moiety"):
        order(str(p), strict_formula=True)


def test_equivalent_group_renumbering_across_assemblies_preserves_molecules(tmp_path):
    # Two independent disordered hydrogen sites in one synthetic molecule.
    base = FIXTURE.read_text()
    base = base.replace("0.4800 0.5771 0.5000 0.6 A 1", "0.4800 0.5771 0.5000 0.6 B 1")
    base = base.replace("0.5000 0.5771 0.4800 0.4 A 2", "0.5000 0.5771 0.4800 0.4 B 2")
    renamed = base.replace("0.6 B 1", "0.6 B 3").replace("0.4 B 2", "0.4 B 4")
    outputs = []
    for text in (base, renamed):
        p = tmp_path / f"case{len(outputs)}.cif"
        p.write_text(text)
        replicas = order(str(p), method="random", generate_count=5, random_seed=7, strict_formula=True, independent_assemblies=True)
        outputs.append([sorted(m.get_chemical_formula() for m in c.molecules) for c in replicas])
        enumerated = order(
            str(p), method="enumerate", generate_count=10,
            return_kept_indices=True, strict_formula=True,
            independent_assemblies=True,
        )
        assert {tuple(indices) for _, indices in enumerated} == {
            (0, 1, 2), (0, 1, 4), (0, 2, 3), (0, 3, 4)
        }
    assert outputs[0] == outputs[1] == [["H2O"]] * 5


def test_legacy_mode_warns_for_different_local_group_numbers(tmp_path, caplog):
    text = FIXTURE.read_text().replace("0.4800 0.5771 0.5000 0.6 A 1", "0.4800 0.5771 0.5000 0.6 B 3")
    text = text.replace("0.5000 0.5771 0.4800 0.4 A 2", "0.5000 0.5771 0.4800 0.4 B 4")
    p = tmp_path / "different_ids.cif"
    p.write_text(text)
    with caplog.at_level(logging.WARNING):
        order(str(p))
    assert "Multiple multi-group assemblies detected" in caplog.text


def test_legacy_mode_warns_for_matching_local_group_numbers(caplog):
    p = Path("tests/data/cif/synthetic_two_assemblies.cif")
    with caplog.at_level(logging.WARNING):
        order(str(p))
    assert "Multiple multi-group assemblies detected" in caplog.text


def test_direct_bonds_survive_independent_group_renumbering(tmp_path):
    base = Path("tests/data/cif/synthetic_bonded_assemblies.cif").read_text()
    for number, text in enumerate((base, base.replace(".6 B 1", ".6 B 3").replace(".4 B 2", ".4 B 4"))):
        p = tmp_path / f"bonded{number}.cif"
        p.write_text(text)
        results = order(str(p), method="enumerate", generate_count=10,
                        return_kept_indices=True, strict_formula=True,
                        independent_assemblies=True)
        assert {tuple(indices) for _, indices in results} == {
            (0, 2), (0, 3), (1, 2), (1, 3)
        }
        assert all([m.get_chemical_formula() for m in c.molecules] == ["C2"] for c, _ in results)
