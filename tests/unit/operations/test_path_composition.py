"""Explicit atom-subset path composition and invariant checks."""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLengths

from molcrys_kit.operations import PathLayer, compose_path_images, refine_idpp_path


def _base():
    images = []
    for t in (0.0, 0.5, 1.0):
        a = Atoms(
            "CNHO",
            positions=[[1 + t, 1, 1], [3 + t, 1, 1], [3 + t, 2, 1], [6 + t, 1, 1]],
            cell=np.diag([10.0, 8.0, 9.0]),
            pbc=(True, True, True),
        )
        a.set_array("atom_id", np.array([10, 11, 12, 13]))
        images.append(a)
    return images


def _anchored_layer(base):
    layer = []
    for k, frame in enumerate(base):
        fragment = frame[[1, 2]]
        fragment.positions -= fragment.positions[0]
        if k == 1:
            fragment.positions[1] += [0, 0, 0.4]
        layer.append(fragment)
    return PathLayer((1, 2), layer, anchor_index=1, name="fragment")


def test_anchored_layer_composes_and_idpp_can_prescribe_it():
    base = _base()
    before = [a.positions.copy() for a in base]
    layer = _anchored_layer(base)
    result = compose_path_images(base, [layer])
    for k, image in enumerate(result.images):
        np.testing.assert_array_equal(base[k].positions, before[k])
        np.testing.assert_array_equal(image.positions[[0, 1, 3]], before[k][[0, 1, 3]])
        np.testing.assert_array_equal(image.arrays["atom_id"], [10, 11, 12, 13])
    assert result.images[1].positions[2, 2] == pytest.approx(1.4)
    np.testing.assert_array_equal(result.images[0].positions, before[0])
    np.testing.assert_array_equal(result.images[-1].positions, before[-1])
    assert result.diagnostics["atom_identity"] == "atom_id"
    assert result.diagnostics["layers"][0]["name"] == "fragment"
    refined = refine_idpp_path(result.images, prescribed_atom_indices=(1, 2))
    for expected, actual in zip(result.images, refined.images):
        np.testing.assert_array_equal(
            actual.positions[[1, 2]], expected.positions[[1, 2]]
        )


def test_preserve_and_nearest_image_policies_are_explicit():
    base = _base()
    images = [a[[1]] for a in base]
    for image in images:
        image.positions[0] += [10.0, 0.0, 0.0]
    preserved = compose_path_images(base, [PathLayer((1,), images)])
    nearest = compose_path_images(
        base, [PathLayer((1,), images, image_policy="nearest")]
    )
    assert preserved.images[1].positions[1, 0] == pytest.approx(
        base[1].positions[1, 0] + 10
    )
    assert nearest.images[1].positions[1, 0] == pytest.approx(base[1].positions[1, 0])
    for result in (preserved, nearest):
        np.testing.assert_array_equal(result.images[0].positions, base[0].positions)
        np.testing.assert_array_equal(result.images[-1].positions, base[-1].positions)


def test_rejects_overlap_bad_identity_and_endpoint_changes():
    base = _base()
    layer = _anchored_layer(base)
    with pytest.raises(ValueError, match="overlap"):
        compose_path_images(base, [layer, layer])
    bad_id = [a.copy() for a in layer.images]
    for frame in bad_id:
        frame.arrays["atom_id"][1] = 99
    with pytest.raises(ValueError, match="do not match"):
        compose_path_images(base, [PathLayer((1, 2), bad_id, anchor_index=1)])
    bad_end = [a.copy() for a in layer.images]
    bad_end[-1].positions[1] += [0, 0, 0.1]
    with pytest.raises(ValueError, match="endpoint"):
        compose_path_images(base, [PathLayer((1, 2), bad_end, anchor_index=1)])
    changed = compose_path_images(
        base, [PathLayer((1, 2), bad_end, anchor_index=1)], preserve_endpoints=False
    )
    assert changed.images[-1].positions[2, 2] == pytest.approx(1.1)


def test_fixed_atoms_and_unsupported_constraints_are_not_silently_changed():
    base = _base()
    for frame in base:
        frame.set_constraint(FixAtoms(indices=[2]))
    with pytest.raises(ValueError, match="FixAtoms"):
        compose_path_images(base, [_anchored_layer(base)])
    plain = _base()
    plain[1].set_constraint(FixBondLengths([[0, 1]]))
    with pytest.raises(ValueError, match="only ASE FixAtoms"):
        compose_path_images(plain, [_anchored_layer(plain)])


def test_layer_validation_and_id_requirements():
    base = _base()
    with pytest.raises(ValueError, match="unique"):
        PathLayer((1, 1), [a[[1, 1]] for a in base])
    with pytest.raises(ValueError, match="anchor"):
        PathLayer((1, 2), [a[[1, 2]] for a in base], anchor_index=0)
    with pytest.raises(ValueError, match="one ASE Atoms"):
        compose_path_images(base, [PathLayer((1, 2), [base[0][[1, 2]]])])
    no_ids = [a[[1, 2]] for a in base]
    for a in no_ids:
        del a.arrays["atom_id"]
    with pytest.raises(ValueError, match="both supply atom_id"):
        compose_path_images(base, [PathLayer((1, 2), no_ids)])


def test_changed_images_drop_stale_bond_record_cache_without_mutating_input():
    base = _base()
    for frame in base:
        frame.info["_molcrys_bond_records"] = [{"left": 0, "right": 1}]
    result = compose_path_images(base, [_anchored_layer(base)])
    assert "_molcrys_bond_records" not in result.images[1].info
    assert "_molcrys_bond_records" in result.images[0].info
    assert "_molcrys_bond_records" in result.images[2].info
    assert all("_molcrys_bond_records" in frame.info for frame in base)
    assert result.diagnostics["invalidated_bond_record_images"] == [1]


def test_layer_cell_and_pbc_must_already_match_the_base_image():
    base = _base()
    shifted = _anchored_layer(base)
    shifted.images[1].cell[0, 0] = 12.0
    with pytest.raises(ValueError, match="does not transform cells"):
        compose_path_images(base, [shifted])
    rebounded = _anchored_layer(base)
    rebounded.images[1].set_pbc((True, True, False))
    with pytest.raises(ValueError, match="PBC must match"):
        compose_path_images(base, [rebounded])


def test_changed_images_drop_fractional_coordinate_arrays():
    base = _base()
    for frame in base:
        frame.set_array("frac_x", np.zeros(len(frame)))
        frame.info["_molcrys_bond_records"] = [{"left": 0, "right": 1}]
    result = compose_path_images(base, [_anchored_layer(base)])
    assert "frac_x" not in result.images[1].arrays
    assert "_molcrys_bond_records" not in result.images[1].info
    assert "frac_x" in result.images[0].arrays
    refined = refine_idpp_path(result.images)
    assert "frac_x" not in refined.images[1].arrays
    assert "frac_x" in result.images[0].arrays


def test_atom_ids_must_be_one_dimensional():
    base = _base()
    for frame in base:
        del frame.arrays["atom_id"]
        frame.set_array("atom_id", np.repeat(np.arange(4)[:, None], 2, axis=1))
    with pytest.raises(ValueError, match="one scalar per atom"):
        compose_path_images(base, [])
