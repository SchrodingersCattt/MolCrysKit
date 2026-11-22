"""Malformed ``loop_`` blocks fail fast instead of hanging the CIF reader."""

import subprocess
import sys
import textwrap

import pytest

from molcrys_kit.compat.cifparse import CifParser

# Each case ran forever before the guard: no column tags, so no value is ever
# consumed by the loop body.
MALFORMED = [
    "data_broken\nloop_\nabc\n",
    "data_broken\nloop_\n",
    "data_broken\nloop_\nloop_\n_a\n1\n",
    "data_x\n_cell_length_a 5\nloop_\n1 2 3\n",
]


@pytest.mark.parametrize("text", MALFORMED)
def test_loop_without_columns_raises_descriptive_error(text):
    # Run in a child process with a hard wall-clock limit so a regression
    # fails this test instead of hanging the whole suite.
    script = textwrap.dedent(
        f"""
        from molcrys_kit.compat.cifparse import CifParser
        try:
            CifParser.from_str({text!r}).as_dict()
        except ValueError as exc:
            print("ValueError:", exc)
        """
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("CifParser hung on a loop_ with no column tags")
    assert completed.returncode == 0, completed.stderr
    assert "ValueError" in completed.stdout
    assert "loop_" in completed.stdout and "column" in completed.stdout


def test_error_names_the_data_block():
    with pytest.raises(ValueError, match="data block 'data_broken'.*no column"):
        CifParser.from_str("data_broken\nloop_\nabc\n").as_dict()


def test_well_formed_loop_is_unchanged():
    blocks = CifParser.from_str(
        "data_ok\nloop_\n_atom_site_label\n_atom_site_fract_x\nC1 0.1\nC2 0.2\n"
    ).as_dict()
    assert blocks["data_ok"]["_atom_site_label"] == ["C1", "C2"]
    assert blocks["data_ok"]["_atom_site_fract_x"] == ["0.1", "0.2"]
