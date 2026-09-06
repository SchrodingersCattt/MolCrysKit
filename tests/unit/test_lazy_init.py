"""Regression tests for lazy-loading in ``molcrys_kit.__init__``.

These tests run in a subprocess to guarantee a completely fresh interpreter
with no cached imports.  They verify the ``__getattr__`` / ``__dir__`` /
``__all__`` contract that was introduced as part of the lazy-import speedup.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    """Execute *code* in a fresh Python process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


# ── __all__ symbols resolve without error ────────────────────────────

def test_all_symbols_resolve():
    """Every name in ``__all__`` must be importable from the package."""
    result = _run_snippet("""\
        import molcrys_kit
        missing = []
        for name in molcrys_kit.__all__:
            try:
                getattr(molcrys_kit, name)
            except AttributeError:
                missing.append(name)
        if missing:
            raise AssertionError(f"__all__ names not resolvable: {missing}")
    """)
    assert result.returncode == 0, result.stderr


# ── Backward-compat alias ────────────────────────────────────────────

def test_molecule_alias():
    """``Molecule`` remains a backward-compat alias for ``CrystalMolecule``."""
    result = _run_snippet("""\
        from molcrys_kit import Molecule, CrystalMolecule
        assert Molecule is CrystalMolecule
    """)
    assert result.returncode == 0, result.stderr


# ── Caching: second access must not call __getattr__ again ───────────

def test_lazy_import_caching():
    """Resolved lazy imports are cached in module globals."""
    result = _run_snippet("""\
        import molcrys_kit
        _ = molcrys_kit.MolAtom           # triggers __getattr__ -> caches
        assert 'MolAtom' in vars(molcrys_kit), 'not cached in globals'
        cached = molcrys_kit.MolAtom      # should not hit __getattr__
        assert cached is _
    """)
    assert result.returncode == 0, result.stderr


# ── Subpackage attributes ────────────────────────────────────────────

@pytest.mark.parametrize("subpkg", ["structures", "io", "chemistry"])
def test_subpackage_attribute(subpkg: str):
    """``molcrys_kit.<subpackage>`` must resolve even without explicit import."""
    result = _run_snippet(f"""\
        import molcrys_kit
        mod = molcrys_kit.{subpkg}
        assert hasattr(mod, '__name__')
    """)
    assert result.returncode == 0, result.stderr


# ── __dir__ ──────────────────────────────────────────────────────────

def test_dir_has_no_duplicates():
    """``dir(molcrys_kit)`` must return unique entries."""
    result = _run_snippet("""\
        import molcrys_kit
        d = dir(molcrys_kit)
        dupes = [x for x in set(d) if d.count(x) > 1]
        assert not dupes, f"dir() duplicates: {dupes}"
    """)
    assert result.returncode == 0, result.stderr


def test_dir_contains_all_and_version():
    """``dir()`` must include every ``__all__`` entry plus ``__version__``."""
    result = _run_snippet("""\
        import molcrys_kit
        d = set(dir(molcrys_kit))
        missing = set(molcrys_kit.__all__) - d
        assert not missing, f"Missing from dir(): {missing}"
        assert '__version__' in d
        assert 'Molecule' in d
    """)
    assert result.returncode == 0, result.stderr


# ── Unknown attribute raises AttributeError ──────────────────────────

def test_unknown_attribute_raises():
    result = _run_snippet("""\
        import molcrys_kit
        try:
            molcrys_kit._nonexistent_xyz_
            raise AssertionError("should have raised AttributeError")
        except AttributeError:
            pass
    """)
    assert result.returncode == 0, result.stderr


# ── Heavy modules not loaded by bare import ──────────────────────────

def test_bare_import_does_not_load_heavy_deps():
    """``import molcrys_kit`` alone must NOT drag in networkx/ase/scipy."""
    result = _run_snippet("""\
        import sys
        import molcrys_kit  # noqa: F401  — bare import only
        heavy = {'networkx', 'ase', 'scipy', 'pymatgen'}
        loaded = heavy & set(sys.modules)
        assert not loaded, f"Heavy deps loaded by bare import: {loaded}"
    """)
    assert result.returncode == 0, result.stderr
