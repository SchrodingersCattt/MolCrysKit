"""Mercury reader contracts using independently constructed synthetic XML only."""

import gzip
import json
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from ase.cell import Cell
from click.testing import CliRunner

from molcrys_kit.cli import main
from molcrys_kit.io import read_mryx, read_mryx_atoms, read_extxyz, write_extxyz
from molcrys_kit.io import mryx


def _vector(parent, name, values, matrix=False):
    node = ET.SubElement(parent, name)
    if matrix:
        node = ET.SubElement(node, "matrix")
    container = ET.SubElement(node, "vector", type="Container")
    for value in values:
        ET.SubElement(container, "item", type="float").text = str(value)


def _document(n=1):
    """Arbitrary artificial cell/sites, not modified or sampled from a real file."""
    root = ET.Element("All")
    graphics = ET.SubElement(root, "graphics_crystal", type="Container")
    for index in range(n):
        handle = ET.SubElement(graphics, "item")
        value = ET.SubElement(handle, "value", type="GraphicsCrystal")
        memento = ET.SubElement(value, "graphics_crystal_memento")
        obj = ET.SubElement(memento, "graphics_object")
        ET.SubElement(obj, "label").text = f"synthetic-{index}"
        structure = ET.SubElement(value, "value", type="ConcreteCrystalStructure")
        cell = ET.SubElement(ET.SubElement(structure, "cell"), "value")
        parameters = [7 + index, 9, 11, 80, 95, 110]
        matrix = np.asarray(Cell.fromcellpar(parameters))
        for key, number in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"),
            [*parameters[:3], *np.radians(parameters[3:])],
        ):
            ET.SubElement(ET.SubElement(cell, key), "value").text = str(number)
        _vector(cell, "frac_to_orth", matrix.T.flatten(), matrix=True)
        _vector(cell, "orth_to_frac", np.linalg.inv(matrix.T).flatten(), matrix=True)
        spacegroup = ET.SubElement(cell, "spacegroup")
        ET.SubElement(spacegroup, "name").text = "P 1"
        ET.SubElement(spacegroup, "number").text = "1"
        molecule = ET.SubElement(
            ET.SubElement(structure, "molecule"), "value", type="Molecule3d"
        )
        atoms = ET.SubElement(molecule, "atoms", type="Container")
        for serial, (label, code, frac) in enumerate(
            [
                ("C901", "9001", [0.13, 0.27, 0.41]),
                ("O902", "9002", [0.17, 0.31, 0.45]),
            ],
            start=101,
        ):
            atom_handle = ET.SubElement(atoms, "item")
            ET.SubElement(atom_handle, "serial_number").text = str(serial)
            atom = ET.SubElement(atom_handle, "value", type="Atom3d")
            ET.SubElement(atom, "label").text = label
            ET.SubElement(ET.SubElement(atom, "element"), "code").text = code
            site = ET.SubElement(ET.SubElement(atom, "site"), "value")
            _vector(site, "fractional", frac)
            _vector(site, "orthogonal", np.array(frac) @ matrix)
            ET.SubElement(ET.SubElement(site, "uncertain_value"), "value").text = "0.37"
        bonds = ET.SubElement(molecule, "bonds", type="Container")
        bond = ET.SubElement(bonds, "item")
        ET.SubElement(bond, "node1").text = "0"
        ET.SubElement(bond, "node2").text = "1"
        edge = ET.SubElement(ET.SubElement(bond, "edge"), "value")
        ET.SubElement(ET.SubElement(edge, "bond_type"), "enumerator").text = "777"
        assemblies = ET.SubElement(structure, "disorder_assemblies")
        assembly = ET.SubElement(ET.SubElement(assemblies, "item"), "value")
        groups = ET.SubElement(assembly, "groups")
        group = ET.SubElement(ET.SubElement(groups, "item"), "value")
        refs = ET.SubElement(group, "atoms")
        ET.SubElement(ET.SubElement(refs, "item"), "serial_number").text = "101"
    return root


def _save(tmp_path, root, compressed=True):
    data = b'<?xml version="1.0"?><!DOCTYPE CCDC_XML>' + ET.tostring(root)
    path = tmp_path / "synthetic.mryx"
    path.write_bytes(gzip.compress(data, mtime=0) if compressed else data)
    return path


def _read(path, **kwargs):
    with pytest.warns(UserWarning, match="MRYX conservative import"):
        return read_mryx_atoms(path, **kwargs)


def test_stored_spacegroup_metadata_is_passed_through_uninterpreted(tmp_path):
    atoms = _read(_save(tmp_path, _document()))
    meta = atoms.info["mryx"]
    assert meta["stored_spacegroup_name"] == "P 1"
    assert meta["stored_spacegroup_number"] == 1


@pytest.mark.parametrize("compressed", [True, False])
def test_cell_coordinates_and_uninterpreted_metadata(tmp_path, compressed):
    atoms = _read(_save(tmp_path, _document(), compressed))
    np.testing.assert_allclose(atoms.cell.cellpar(), [7, 9, 11, 80, 95, 110])
    np.testing.assert_allclose(
        atoms.get_scaled_positions(), [[0.13, 0.27, 0.41], [0.17, 0.31, 0.45]]
    )
    assert atoms.get_chemical_symbols() == ["C", "O"]
    assert atoms.arrays["label"].tolist() == ["C901", "O902"]
    assert "occupancy" not in atoms.arrays
    metadata = atoms.info["mryx"]
    assert metadata["occupancy_semantics"] == "unspecified"
    assert metadata["site_scope"] == "stored_sites"
    assert metadata["diagnostics"]
    raw = json.dumps(metadata)
    for field in ["777", "0.37", "spacegroup", "disorder_assemblies", "9001"]:
        assert field in raw


def test_multistructure_selection(tmp_path):
    path = _save(tmp_path, _document(n=3))
    frames = _read(path, index=":")
    assert len(frames) == 3
    assert _read(path).info["mryx"]["structure_index"] == 0
    assert _read(path, index=-1).info["mryx"]["structure_index"] == 2
    assert frames[1].info["mryx"]["label"] == "synthetic-1"
    for index in [3, -4, "0", "all", True, 1.2]:
        with pytest.raises(ValueError, match="index"):
            read_mryx_atoms(path, index=index)


def test_fractional_only_without_matrix(tmp_path):
    root = _document()
    for cell in root.findall(".//cell/value"):
        cell.remove(cell.find("frac_to_orth"))
        cell.remove(cell.find("orth_to_frac"))
    for site in root.findall(".//site/value"):
        site.remove(site.find("orthogonal"))
    atoms = _read(_save(tmp_path, root))
    np.testing.assert_allclose(atoms.cell.cellpar(), [7, 9, 11, 80, 95, 110])


def test_orthogonal_only_and_symmetry_match_matrix(tmp_path):
    root = _document()
    structure = root.find(".//value[@type='ConcreteCrystalStructure']")
    cell = structure.find("cell/value")
    match = ET.SubElement(ET.SubElement(structure, "crystal_symmetry_match"), "value")
    for tag in ["frac_to_orth", "orth_to_frac"]:
        node = cell.find(tag)
        cell.remove(node)
        match.append(node)
    for site in root.findall(".//site/value"):
        site.remove(site.find("fractional"))
    atoms = _read(_save(tmp_path, root))
    np.testing.assert_allclose(atoms.get_scaled_positions()[0], [0.13, 0.27, 0.41])


@pytest.mark.parametrize(
    "path,text,message",
    [
        (".//a/value", "0", "cell lengths"),
        (".//alpha/value", "nan", "non-finite"),
        (".//gamma/value", "4", "radian angles"),
        (".//orthogonal/vector/item", "999", "coordinates disagree"),
        (".//frac_to_orth/matrix/vector/item", "999", "matrix disagrees"),
        (".//orth_to_frac/matrix/vector/item", "999", "matrices disagree"),
        (".//fractional/vector/item", "inf", "finite vector"),
        (".//node2", "22", "endpoint"),
        (".//node1", "-1", "endpoint"),
        (".//value[@type='Atom3d']/label", "Qq9", "infer an element"),
        (".//value[@type='Atom3d']/label", "custom", "infer an element"),
        (".//groups/item/value/atoms/item/serial_number", "9999", "unknown atom"),
    ],
)
def test_invalid_fields(tmp_path, path, text, message):
    root = _document()
    node = root.find(path)
    assert node is not None
    node.text = text
    with pytest.raises(ValueError, match=message):
        read_mryx_atoms(_save(tmp_path, root))


@pytest.mark.parametrize(
    "path", [".//site/value/fractional/vector/item", ".//site/value", ".//cell/value"]
)
def test_missing_or_wrong_shape(tmp_path, path):
    root = _document()
    for parent in root.iter():
        child = root.find(path)
        if child in list(parent):
            parent.remove(child)
            break
    with pytest.raises(ValueError):
        read_mryx_atoms(_save(tmp_path, root))


def test_conflicting_internal_element_code(tmp_path):
    root = _document()
    for code in root.findall(".//element/code"):
        code.text = "9001"
    with pytest.raises(ValueError, match="conflicting elements"):
        read_mryx_atoms(_save(tmp_path, root))


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not xml",
        b"<All>",
        b"<Other />",
        b"<All />",
        b"\x1f\x8bbad",
        b'<!DOCTYPE All [<!ENTITY x "explosion">]><All>&x;</All>',
        b'<!DOCTYPE All [<!ENTITY x SYSTEM "file:///nonexistent">]><All>&x;</All>',
    ],
)
def test_malformed_or_unsafe_input(tmp_path, payload):
    path = tmp_path / "bad.mryx"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_mryx_atoms(path)


def test_truncated_gzip(tmp_path):
    path = _save(tmp_path, _document())
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ValueError, match="gzip"):
        read_mryx_atoms(path)


@pytest.mark.parametrize(
    "constant,limit",
    [
        ("MRYX_MAX_COMPRESSED_BYTES", 12),
        ("MRYX_MAX_XML_BYTES", 16),
        ("MRYX_MAX_XML_NODES", 8),
        ("MRYX_MAX_XML_DEPTH", 4),
    ],
)
def test_resource_limits(tmp_path, monkeypatch, constant, limit):
    path = _save(tmp_path, _document())
    monkeypatch.setattr(mryx, constant, limit)
    with pytest.raises(ValueError, match="limit"):
        read_mryx_atoms(path)


def test_label_site_and_structure_caps(tmp_path, monkeypatch):
    path = _save(tmp_path, _document())
    monkeypatch.setattr(mryx, "MRYX_MAX_LABEL_LENGTH", 3)
    with pytest.raises(ValueError, match="label exceeds"):
        read_mryx_atoms(path)
    monkeypatch.setattr(mryx, "MRYX_MAX_LABEL_LENGTH", 128)
    monkeypatch.setattr(mryx, "MRYX_MAX_SITES", 1)
    with pytest.raises(ValueError, match="site limit"):
        read_mryx_atoms(path)
    monkeypatch.setattr(mryx, "MRYX_MAX_SITES", 100_000)
    monkeypatch.setattr(mryx, "MRYX_MAX_STRUCTURES", 2)
    with pytest.raises(ValueError, match="structure limit"):
        read_mryx_atoms(_save(tmp_path, _document(n=3)))


def test_single_index_does_not_copy_unselected_structures(tmp_path, monkeypatch):
    path = _save(tmp_path, _document(n=3))
    calls = {"n": 0}
    original = mryx._raw

    def spy(node):
        calls["n"] += 1
        return original(node)

    monkeypatch.setattr(mryx, "_raw", spy)
    atoms = _read(path, index=0)
    one = calls["n"]
    assert one > 0
    assert atoms.info["mryx"]["structure_index"] == 0
    assert len(atoms.info["mryx"]["atoms"]) == 2
    calls["n"] = 0
    last = _read(path, index=-1)
    assert calls["n"] == one
    assert last.info["mryx"]["structure_index"] == 2
    calls["n"] = 0
    frames = _read(path, index=":")
    assert calls["n"] == one * 3
    assert [frame.info["mryx"]["structure_index"] for frame in frames] == [0, 1, 2]


def test_ignored_markup_is_discarded_and_does_not_change_sites(tmp_path, monkeypatch):
    root = _document(n=2)
    scene = ET.Element("scene")
    for index in range(12):
        style = ET.SubElement(scene, "style")
        ET.SubElement(style, "item").text = f"ignored-{index}"
    root.insert(0, scene)
    camera = ET.SubElement(root, "camera")
    ET.SubElement(camera, "item").text = "ignored-camera"
    path = _save(tmp_path, root)
    cleared = []
    original = mryx._release

    def spy(elem):
        cleared.append(elem.tag)
        return original(elem)

    monkeypatch.setattr(mryx, "_release", spy)
    atoms = _read(path, index=0)
    assert atoms.get_chemical_symbols() == ["C", "O"]
    assert atoms.info["mryx"]["structure_index"] == 0
    assert "scene" in cleared and "camera" in cleared and "style" in cleared
    labels = root.findall(".//value[@type='Atom3d']/label")
    labels[2].text = "custom"
    with pytest.raises(ValueError, match="infer an element"):
        read_mryx_atoms(_save(tmp_path, root), index=0)


def test_unselected_malformed_structure_still_fails(tmp_path):
    root = _document(n=2)
    labels = root.findall(".//value[@type='Atom3d']/label")
    labels[2].text = "custom"
    with pytest.raises(ValueError, match="infer an element"):
        read_mryx_atoms(_save(tmp_path, root), index=0)


def test_standard_model_and_extxyz_output(tmp_path):
    path = _save(tmp_path, _document(n=2))
    with pytest.warns(UserWarning, match="MRYX"):
        crystal = read_mryx(path, index=1)
    assert crystal.metadata["mryx"]["structure_index"] == 1
    assert len(crystal.to_ase()) == 2
    out = tmp_path / "stored.extxyz"
    write_extxyz(crystal, str(out))
    restored = read_extxyz(str(out))
    assert restored.metadata["mryx"]["site_scope"] == "stored_sites"
    np.testing.assert_allclose(restored.lattice, crystal.lattice)


def test_cli_convert_selection_and_existing_dispatch(tmp_path):
    path = _save(tmp_path, _document(n=2))
    out = tmp_path / "converted.extxyz"
    result = CliRunner().invoke(
        main, ["io", "convert", str(path), "-o", str(out), "--structure-index", "1"]
    )
    assert result.exit_code == 0, result.output
    assert read_extxyz(str(out)).metadata["mryx"]["structure_index"] == 1
    result = CliRunner().invoke(main, ["io", "info", str(path)])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(
        main, ["io", "convert", str(path), "-o", str(out), "--resolve-disorder"]
    )
    assert result.exit_code != 0
    assert "not supported" in result.output


def test_core_import_does_not_load_mryx_and_the_reader_names_the_desktop_extra():
    script = textwrap.dedent(
        """
        import sys

        class _Blocker:
            _prefixes = ("pymatgen", "ase", "scipy", "spglib", "defusedxml")

            def find_spec(self, name, path, target=None):
                if name in ("numpy", "networkx") or name.startswith(("numpy.", "networkx.")):
                    return None
                if name in self._prefixes or any(name.startswith(prefix + ".") for prefix in self._prefixes):
                    raise ImportError(name)
                return None

        sys.meta_path.insert(0, _Blocker())
        import molcrys_kit.io as io

        assert "ase" not in sys.modules
        assert "defusedxml" not in sys.modules
        try:
            io.read_mryx_atoms
        except ImportError as exc:
            text = str(exc)
            assert "desktop" in text
            assert "defusedxml" in text or "ase" in text
        else:
            raise SystemExit("read_mryx_atoms imported without the desktop extra")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
