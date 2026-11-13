"""Formula moiety / Z metadata parity between the desktop and lightweight CIF readers.

``scan_cif_disorder`` feeds formula-based stoichiometry filtering in the
disorder solver.  The lightweight ``compat.cifparse.CifParser`` (phone /
minimal install) must expose exactly the same ``_chemical_formula_moiety`` and
``_cell_formula_units_Z`` as pymatgen, otherwise the filter is silently skipped
and the number of ordered candidates differs between platforms.
"""

import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

import molcrys_kit.io.cif as cif_module
from molcrys_kit.analysis.disorder.process import (
    generate_ordered_replicas_from_disordered_sites,
)
from molcrys_kit.compat.cifparse import CifParser as CompatCifParser
from molcrys_kit.io.cif import (
    _CaseInsensitiveBlock,
    _extract_formula_moiety,
    _parse_cell_formula_units_z,
    read_cif_symmetry,
    scan_cif_disorder,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "data" / "cif" / "synthetic_two_assemblies.cif"
MOIETY_LINE = "_chemical_formula_moiety 'H2 O'\n"
Z_LINE = "_cell_formula_units_Z 1\n"

# Reference output of the fixture (identical on desktop and lightweight installs).
RANDOM_KEPT = [(0, 1, 2), (0, 1, 2), (0, 3, 4), (0, 1, 2), (0, 1, 2)]
RANDOM_KEPT_INDEPENDENT = [(0, 1, 2), (0, 1, 2), (0, 1, 4), (0, 2, 3), (0, 2, 3)]
ENUMERATE_KEPT = [(0, 1, 2), (0, 3, 4)]
ENUMERATE_KEPT_INDEPENDENT = [(0, 1, 2), (0, 2, 3), (0, 1, 4), (0, 3, 4)]


@pytest.fixture(params=["pymatgen", "compat"])
def reader(request, monkeypatch):
    """Run the test once with pymatgen and once with the lightweight parser."""
    if request.param == "pymatgen":
        pytest.importorskip("pymatgen")
        pytest.importorskip("spglib")
        pytest.importorskip("scipy")
        from pymatgen.io.cif import CifParser as parser
    else:
        parser = CompatCifParser
    monkeypatch.setattr(cif_module, "CifParser", parser)
    return request.param


@pytest.fixture(autouse=True)
def _quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _variant(tmp_path, name="variant.cif", *, moiety=MOIETY_LINE, z=Z_LINE, text=None):
    source = FIXTURE.read_text(encoding="utf-8") if text is None else text
    assert MOIETY_LINE in source and Z_LINE in source
    path = tmp_path / name
    path.write_text(
        source.replace(MOIETY_LINE, moiety).replace(Z_LINE, z), encoding="utf-8"
    )
    return str(path)


def _run(path, **kwargs):
    results = generate_ordered_replicas_from_disordered_sites(
        path, return_kept_indices=True, **kwargs
    )
    return [tuple(int(i) for i in kept) for _crystal, kept in results]


# --------------------------------------------------------------------------
# Real fixture: metadata and candidate semantics
# --------------------------------------------------------------------------


def test_fixture_metadata_on_both_readers(reader):
    info = scan_cif_disorder(str(FIXTURE))
    assert info.formula_moiety == "H2 O"
    assert info.z_value == 1


def test_fixture_random_candidates_keep_duplicates(reader):
    kept = _run(str(FIXTURE), method="random", generate_count=5, random_seed=42)
    # Random sampling may legitimately repeat a candidate; do not de-duplicate.
    assert kept == RANDOM_KEPT
    assert len(set(kept)) < len(kept)


def test_fixture_independent_assemblies_option(reader):
    kept = _run(
        str(FIXTURE),
        method="random",
        generate_count=5,
        random_seed=42,
        independent_assemblies=True,
    )
    assert kept == RANDOM_KEPT_INDEPENDENT


def test_fixture_enumerate_candidates(reader):
    assert _run(str(FIXTURE), method="enumerate", generate_count=0) == ENUMERATE_KEPT
    assert (
        _run(
            str(FIXTURE),
            method="enumerate",
            generate_count=0,
            independent_assemblies=True,
        )
        == ENUMERATE_KEPT_INDEPENDENT
    )


def test_strict_formula_accepts_matching_metadata(reader):
    kept = _run(
        str(FIXTURE),
        method="random",
        generate_count=5,
        random_seed=42,
        strict_formula=True,
    )
    assert kept == RANDOM_KEPT


def test_strict_formula_rejects_mismatching_formula(reader, tmp_path):
    path = _variant(tmp_path, moiety="_chemical_formula_moiety 'H3 O'\n")
    with pytest.raises(ValueError, match="No ordered candidate matches"):
        _run(
            path, method="random", generate_count=5, random_seed=42, strict_formula=True
        )


def test_strict_formula_requires_metadata(reader, tmp_path):
    path = _variant(tmp_path, moiety="", z="")
    with pytest.raises(ValueError, match="requires a parseable formula moiety"):
        _run(
            path, method="random", generate_count=5, random_seed=42, strict_formula=True
        )


# --------------------------------------------------------------------------
# Value variants: identical result on both readers
# --------------------------------------------------------------------------

VARIANTS = {
    "normal": (MOIETY_LINE, Z_LINE, "H2 O", 1, 5),
    "double_quoted": ('_chemical_formula_moiety "H2 O"\n', Z_LINE, "H2 O", 1, 5),
    "multiline_text": (
        "_chemical_formula_moiety\n;\nH2\nO\n;\n",
        Z_LINE,
        "H2 O",
        1,
        5,
    ),
    "multiline_blank_indented": (
        "_chemical_formula_moiety\n;\n  H2\n\n    O\n;\n",
        Z_LINE,
        "H2 O",
        1,
        5,
    ),
    "case_variants": (
        "_CHEMICAL_Formula_Moiety 'H2 O'\n",
        "_Cell_Formula_Units_z 1\n",
        "H2 O",
        1,
        5,
    ),
    "z_with_su": (MOIETY_LINE, "_cell_formula_units_Z 1(1)\n", "H2 O", 1, 5),
    "z_integral_float": (MOIETY_LINE, "_cell_formula_units_Z 1.0\n", "H2 O", 1, 5),
    "missing": ("", "", None, None, 2),
    "unknown_question": (
        "_chemical_formula_moiety ?\n",
        "_cell_formula_units_Z ?\n",
        None,
        None,
        2,
    ),
    "unknown_dot": (
        "_chemical_formula_moiety .\n",
        "_cell_formula_units_Z .\n",
        None,
        None,
        2,
    ),
    "missing_z_only": (MOIETY_LINE, "", "H2 O", None, 2),
    "invalid_z_text": (MOIETY_LINE, "_cell_formula_units_Z abc\n", "H2 O", None, 2),
    "invalid_z_zero": (MOIETY_LINE, "_cell_formula_units_Z 0\n", "H2 O", None, 2),
    "invalid_z_negative": (MOIETY_LINE, "_cell_formula_units_Z -1\n", "H2 O", None, 2),
    # A refinement artefact such as 0.99984 is not a formula-unit count.
    "invalid_z_fractional": (
        MOIETY_LINE,
        "_cell_formula_units_Z 0.99984\n",
        "H2 O",
        None,
        2,
    ),
    # A valid but wrong Z is honoured: no candidate matches, so the legacy
    # best-effort candidates are returned unvalidated.
    "wrong_z": (MOIETY_LINE, "_cell_formula_units_Z 2\n", "H2 O", 2, 2),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_metadata_variants(reader, tmp_path, name):
    moiety, z, formula, z_value, count = VARIANTS[name]
    path = _variant(tmp_path, moiety=moiety, z=z)
    info = scan_cif_disorder(path)
    assert info.formula_moiety == formula
    assert info.z_value == z_value
    kept = _run(path, method="random", generate_count=5, random_seed=42)
    assert len(kept) == count


def test_desktop_and_compat_agree_on_every_variant(tmp_path, monkeypatch):
    pytest.importorskip("pymatgen")
    pytest.importorskip("spglib")
    pytest.importorskip("scipy")
    from pymatgen.io.cif import CifParser as PymatgenCifParser

    observed = {}
    for parser_name, parser in (
        ("pymatgen", PymatgenCifParser),
        ("compat", CompatCifParser),
    ):
        monkeypatch.setattr(cif_module, "CifParser", parser)
        results = {}
        for name, (moiety, z, *_expected) in VARIANTS.items():
            path = _variant(tmp_path, f"{parser_name}_{name}.cif", moiety=moiety, z=z)
            info = scan_cif_disorder(path)
            results[name] = (
                info.formula_moiety,
                info.z_value,
                _run(path, method="random", generate_count=5, random_seed=42),
                _run(path, method="enumerate", generate_count=0),
            )
        observed[parser_name] = results
    assert observed["pymatgen"] == observed["compat"]


# --------------------------------------------------------------------------
# Related tag-case and text-field discrepancies between the two parsers
# --------------------------------------------------------------------------


def test_mixed_case_tags_resolve_on_both_readers(reader, tmp_path):
    text = FIXTURE.read_text(encoding="utf-8").replace(
        "_space_group_IT_number", "_SPACE_GROUP_it_number"
    )
    path = tmp_path / "case.cif"
    path.write_text(text, encoding="utf-8")
    assert read_cif_symmetry(str(path), expand_symmetry=False).space_group_number == 1


def test_absolute_structure_tags_resolve_on_both_readers(reader, tmp_path):
    text = FIXTURE.read_text(encoding="utf-8").replace(
        Z_LINE, Z_LINE + "_refine_ls_abs_structure_Flack 0.06(3)\n"
    )
    path = tmp_path / "flack.cif"
    path.write_text(text, encoding="utf-8")
    flack = scan_cif_disorder(str(path)).cif_chemistry["absolute_structure"]["flack"]
    assert flack["value"] == pytest.approx(0.06)
    assert flack["standard_uncertainty"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    "tag", ["_note", "_publ_section_title", "_exptl_special_details"]
)
def test_compat_text_field_preserves_arbitrary_multiline_text(tag):
    # Only the formula moiety is normalized (at extraction); other text is data.
    body = "first line\n  indented second line\n\nthird after blank\n"
    parsed = CompatCifParser.from_str(
        f"data_x\n{tag}\n;\n{body};\n_other 1\n"
    ).as_dict()["data_x"]
    assert parsed[tag] == "\n" + body.rstrip("\n")
    assert "\n\n" in parsed[tag]
    assert "\n  indented second line" in parsed[tag]
    assert parsed["_other"] == "1"


def test_compat_text_field_preserves_crlf_and_unclosed_field():
    parsed = CompatCifParser.from_str(
        "data_x\n_note\n;\nline one\r\n\r\n  two\n;\n"
    ).as_dict()["data_x"]
    assert parsed["_note"] == "\nline one\r\n\r\n  two"
    unclosed = CompatCifParser.from_str(
        "data_x\n_note\n;\nline one\n\n  two\n"
    ).as_dict()["data_x"]
    assert unclosed["_note"] == "\nline one\n\n  two\n"


def test_compat_multiline_moiety_is_normalized_only_when_extracted():
    parsed = CompatCifParser.from_str(
        "data_x\n_chemical_formula_moiety\n;\n  H2\n\n   O\n;\n"
    ).as_dict()["data_x"]
    assert parsed["_chemical_formula_moiety"] == "\n  H2\n\n   O"
    assert _extract_formula_moiety(parsed) == "H2 O"


def test_case_insensitive_block_lookup():
    block = _CaseInsensitiveBlock({"_Mixed_Case": "1", "_exact": "a", "_EXACT": "b"})
    assert block["_mixed_case"] == "1"
    assert "_MIXED_CASE" in block
    assert block.get("_missing", "d") == "d"
    # Exact spelling wins over a case-insensitive match.
    assert block["_exact"] == "a"
    assert block["_EXACT"] == "b"
    assert list(block) == ["_Mixed_Case", "_exact", "_EXACT"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1", 1),
        ("12", 12),
        (" 4 ", 4),
        ("4.0", 4),
        ("4(1)", 4),
        ("4.00(2)", 4),
        ("1e0", 1),
        (["8", "9"], 8),
        (None, None),
        ("", None),
        ("?", None),
        (".", None),
        ("abc", None),
        ("0", None),
        ("-2", None),
        ("2.5", None),
        ("0.99984", None),
        ("nan", None),
        ("inf", None),
        ("4(", None),
    ],
)
def test_parse_cell_formula_units_z(raw, expected):
    assert _parse_cell_formula_units_z(raw) == expected


@pytest.mark.parametrize(
    "block, expected",
    [
        ({"_chemical_formula_moiety": "H2 O"}, "H2 O"),
        ({"_chemical_formula_moiety": "  H2 O  "}, "H2 O"),
        ({"_Chemical_Formula_MOIETY": "H2 O"}, "H2 O"),
        ({"_chemical_formula_moiety": ["H2 O"]}, "H2 O"),
        ({"_chemical_formula_moiety": "\n  H2\n\n   O"}, "H2 O"),
        ({"_chemical_formula_moiety": "?"}, None),
        ({"_chemical_formula_moiety": "."}, None),
        ({"_chemical_formula_moiety": ""}, None),
        ({}, None),
    ],
)
def test_extract_formula_moiety(block, expected):
    assert _extract_formula_moiety(block) == expected


# --------------------------------------------------------------------------
# Lightweight profile: every optional desktop dependency blocked
# --------------------------------------------------------------------------

_SCRIPT = textwrap.dedent(
    """
    import json, sys, warnings
    warnings.simplefilter("ignore")

    class _Blocker:
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("scipy", "ase", "pymatgen", "spglib"):
                raise ImportError(name)

    sys.meta_path.insert(0, _Blocker())

    import molcrys_kit.io.cif as cif
    from molcrys_kit.analysis.disorder.process import (
        generate_ordered_replicas_from_disordered_sites as gen,
    )

    path = {path!r}
    info = cif.scan_cif_disorder(path)

    def run(**kw):
        out = gen(path, return_kept_indices=True, **kw)
        return [[int(i) for i in kept] for _c, kept in out]

    payload = {{
        "parser": cif.CifParser.__module__,
        "formula": info.formula_moiety,
        "z": info.z_value,
        "random": run(method="random", generate_count=5, random_seed=42),
        "random_independent": run(
            method="random", generate_count=5, random_seed=42,
            independent_assemblies=True,
        ),
        "enumerate": run(method="enumerate", generate_count=0),
        "strict": run(
            method="random", generate_count=5, random_seed=42, strict_formula=True
        ),
        "loaded": [m for m in ("ase", "scipy", "pymatgen", "spglib") if m in sys.modules],
    }}
    print("RESULT" + json.dumps(payload))
    """
)


def test_blocked_desktop_dependencies_match_reference():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(path=str(FIXTURE))],
        capture_output=True,
        text=True,
        timeout=170,
        cwd=ROOT,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    line = next(x for x in completed.stdout.splitlines() if x.startswith("RESULT"))
    payload = json.loads(line[len("RESULT") :])

    assert payload["parser"] == "molcrys_kit.compat.cifparse"
    assert payload["loaded"] == []
    assert payload["formula"] == "H2 O"
    assert payload["z"] == 1
    assert [tuple(k) for k in payload["random"]] == RANDOM_KEPT
    assert [tuple(k) for k in payload["random_independent"]] == RANDOM_KEPT_INDEPENDENT
    assert [tuple(k) for k in payload["enumerate"]] == ENUMERATE_KEPT
    assert [tuple(k) for k in payload["strict"]] == RANDOM_KEPT
