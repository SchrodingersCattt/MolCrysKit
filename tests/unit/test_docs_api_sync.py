"""Verify docs/api.md stays in sync with source __init__.py __all__ exports.

When you add/remove/rename a public symbol in any sub-package __init__.py,
this test will fail, reminding you to update docs/api.md.

Run locally:  pytest tests/unit/test_docs_api_sync.py -v
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Set

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_API = REPO_ROOT / "docs" / "api.md"

# Sub-packages whose __all__ is checked against docs/api.md.
# Key: dotted import path; Value: human name used in docs/api.md Module Index heading.
PACKAGES: dict[str, str] = {
    "molcrys_kit.structures": "mck.structures",
    "molcrys_kit.io": "mck.io",
    "molcrys_kit.operations": "mck.operations",
    "molcrys_kit.analysis": "mck.analysis",
    "molcrys_kit.analysis.disorder": "mck.analysis.disorder",
    "molcrys_kit.analysis.interactions": "mck.analysis.interactions",
    "molcrys_kit.constants": "mck.constants",
    "molcrys_kit.utils": "mck.utils",
}

# Symbols expected to be in __all__ (or top-level module exports) but
# intentionally omitted from docs/api.md Module Index.
# Reasons: internal, deprecated, aliased elsewhere, or documented in
# a sub-package section.
EXPECTED_MISSING: dict[str, Set[str]] = {
    "molcrys_kit.analysis": {
        # Interaction types re-exported from analysis.interactions;
        # documented under the `mck.analysis.interactions` Module Index section.
        *[
            "AtomLocalGeometry",
            "AtomRef",
            "BaseInteraction",
            "CHPiInteraction",
            "CHPiInteractionCriteria",
            "ChemicalIdentity",
            "ChemicalIdentityCache",
            "HHContact",
            "HHContactCriteria",
            "HalogenBond",
            "HalogenBondCriteria",
            "HydrogenBond",
            "HydrogenBondCriteria",
            "InteractionProfile",
            "InteractionScoreSummary",
            "LocalGeometry",
            "LocalGeometryCache",
            "PiStacking",
            "PiStackingCriteria",
            "RingGeometry",
            "RingRef",
            "ScoringParams",
            "DEFAULT_SCORING_PARAMS",
            "build_crystal_atom_offsets",
            "composite_score",
            "find_ch_pi",
            "find_ch_pi_interactions",
            "find_h_h_contacts",
            "find_halogen_bonds",
            "find_hydrogen_bonds",
            "find_pi_stacking",
            "find_pi_stacks",
            "gaussian_kernel",
            "get_bonding_threshold",
            "interaction_profile",
            "lorentzian_kernel",
            "normalized_vdw_distance",
            "scaled_cutoff",
            "vdw_radius_sum",
        ],
    },
}


def _extract_module_index_symbols(doc_text: str, label: str) -> Set[str]:
    """Extract backtick-quoted symbols from a Module Index section.

    Headings look like: ``### `mck.io```.
    Compact sections may use bullets instead of tables; every public symbol
    must still appear as a backtick-quoted token in its package section.
    """
    lines = doc_text.splitlines()
    in_section = False
    symbols: Set[str] = set()

    for line in lines:
        # Detect section start: heading containing the label in backticks
        if line.startswith("### ") and f"`{label}`" in line:
            in_section = True
            continue
        # Next heading of any level ends the section
        if in_section and line.startswith("##"):
            break
        if in_section:
            for m in re.finditer(r"`([^`]+)`", line):
                symbol = m.group(1)
                if re.fullmatch(r"[A-Za-z_]\w*", symbol):
                    symbols.add(symbol)

    return symbols


def _get_all_exports(package_path: str) -> Set[str]:
    """Import package and return the effective public symbol set.

    Prefers __all__ when defined; falls back to dir() filtered by
    the __init__.py source when __all__ is absent.
    """
    mod = importlib.import_module(package_path)
    all_ = getattr(mod, "__all__", None)
    if all_ is not None:
        return set(all_)

    # No __all__ — extract public names from __init__.py AST
    init_path = Path(mod.__file__)
    src = init_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    public: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    public.add(target.id)
    return public


def _parse_api_md() -> str:
    """Read docs/api.md and return its text.  Skip if file doesn't exist."""
    if not DOCS_API.exists():
        pytest.skip(f"{DOCS_API} not found")
    return DOCS_API.read_text(encoding="utf-8")


# ---- shared fixture ----
@pytest.fixture(scope="module")
def api_text() -> str:
    return _parse_api_md()


# ---- parameterised tests ----
@pytest.mark.parametrize("package_path, label", PACKAGES.items())
def test_module_index_covers_all(
    package_path: str, label: str, api_text: str
) -> None:
    """Every symbol in __all__ must appear in the Module Index table.

    (Symbols listed in EXPECTED_MISSING are intentionally exempt.)
    """
    exported = _get_all_exports(package_path)
    documented = _extract_module_index_symbols(api_text, label)

    expected_missing = EXPECTED_MISSING.get(package_path, set())
    expected = exported - expected_missing  # remove intentional omissions

    missing_in_docs = expected - documented
    extra_in_docs = documented - expected

    msgs = []
    if missing_in_docs:
        msgs.append(
            f"Symbols in {package_path}.__all__ but MISSING from docs/api.md "
            f"Module Index ({label}):\n"
            + "\n".join(f"  - {s}" for s in sorted(missing_in_docs))
            + "\n→ Add them to docs/api.md under ### `{label}`"
        )
    if extra_in_docs:
        msgs.append(
            f"Symbols in docs/api.md Module Index ({label}) but NOT in "
            f"{package_path}.__all__:\n"
            + "\n".join(f"  - {s}" for s in sorted(extra_in_docs))
            + "\n→ Either add to __all__ or remove from docs/api.md"
        )

    assert not msgs, "\n\n".join(msgs)


def test_capability_map_has_major_entries(api_text: str) -> None:
    """Smoke test: Capability Map includes critical entry points."""
    required = [
        "read_mol_crystal",
        "add_hydrogens",
        "generate_topological_slab",
        "enumerate_bfdh_facets",
        "ClusterCarver",
        "write_cif",
        "find_hydrogen_bonds",
    ]
    missing = [s for s in required if s not in api_text]
    assert not missing, (
        f"Critical symbols missing from Capability Map: {missing}"
    )


# ---------------------------------------------------------------------------
# CLI docs sync
# ---------------------------------------------------------------------------

DOCS_CLI = REPO_ROOT / "docs" / "cli.md"


def _get_click_commands() -> Set[str]:
    """Recursively collect all registered Click leaf-command names."""
    import click
    from molcrys_kit.cli import main as cli_main

    names: Set[str] = set()

    def _walk(group: click.BaseCommand) -> None:
        if isinstance(group, click.Group):
            ctx = click.Context(group)
            for name in group.list_commands(ctx):
                cmd = group.get_command(ctx, name)
                if isinstance(cmd, click.Group):
                    _walk(cmd)
                else:
                    names.add(name)

    _walk(cli_main)
    return names


def test_cli_doc_covers_all_commands() -> None:
    """Every registered CLI subcommand must appear in docs/cli.md."""
    if not DOCS_CLI.exists():
        pytest.skip(f"{DOCS_CLI} not found")
    cli_text = DOCS_CLI.read_text(encoding="utf-8")
    registered = _get_click_commands()
    assert registered, "No CLI commands discovered — check molcrys_kit.cli imports."

    missing = {cmd for cmd in registered if cmd not in cli_text}
    assert not missing, (
        "CLI commands registered but missing from docs/cli.md:\n"
        + "\n".join(f"  - {c}" for c in sorted(missing))
        + "\n→ Add them to docs/cli.md"
    )
