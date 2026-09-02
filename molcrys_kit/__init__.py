"""
MolCrysKit: A Python toolkit for molecular crystal analysis and manipulation.

This toolkit provides functionality for parsing, analyzing, and manipulating
molecular crystal structures, with a particular focus on molecular crystals
where well-defined molecules occupy crystallographic sites.

``annotate_chemistry`` attaches an identity-preserving provisional graph;
``infer_chemistry`` refines that graph with bond, charge, and stereo semantics.
"""

# Version is owned by setuptools_scm and derived from the git tag at build /
# install time (see pyproject.toml [tool.setuptools_scm]). The generated
# molcrys_kit/_version.py is gitignored. Both fallbacks below cover users who
# install from an sdist that did not run setuptools_scm.
try:
    from molcrys_kit._version import __version__  # type: ignore[import-not-found]
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("molcrys-kit")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

# ---------------------------------------------------------------------------
# Lazy public API -- heavy imports are deferred to first access so that
# lightweight entry-points (``mck -h``, tab-completion, etc.) stay fast.
# ---------------------------------------------------------------------------

# Mapping: attribute name -> (module path, object name inside that module)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # structures
    "MolAtom": (".structures.atom", "MolAtom"),
    "CrystalMolecule": (".structures.molecule", "CrystalMolecule"),
    "MolecularCrystal": (".structures.crystal", "MolecularCrystal"),
    "CrystalTrajectory": (".structures.trajectory", "CrystalTrajectory"),
    "BondRecord": (".structures.records", "BondRecord"),
    "SiteRecord": (".structures.records", "SiteRecord"),
    # io
    "read_mol_crystal": (".io.cif", "read_mol_crystal"),
    # chemistry
    "ChemistryIndeterminateError": (".chemistry", "ChemistryIndeterminateError"),
    "CrystalChemistry": (".chemistry", "CrystalChemistry"),
    "CrystalStereoClass": (".chemistry", "CrystalStereoClass"),
    "CrystalStereoIndeterminateError": (".chemistry", "CrystalStereoIndeterminateError"),
    "CrystalStereoReport": (".chemistry", "CrystalStereoReport"),
    "EntityRelationship": (".chemistry", "EntityRelationship"),
    "LineNotation": (".chemistry", "LineNotation"),
    "LineNotationError": (".chemistry", "LineNotationError"),
    "NamingIndeterminateError": (".chemistry", "NamingIndeterminateError"),
    "NamingKind": (".chemistry", "NamingKind"),
    "NamingResult": (".chemistry", "NamingResult"),
    "PeriodicTopology": (".chemistry", "PeriodicTopology"),
    "StereoDescriptor": (".chemistry", "StereoDescriptor"),
    "StereoKind": (".chemistry", "StereoKind"),
    "StereoReport": (".chemistry", "StereoReport"),
    "analyze_periodic_topology": (".chemistry", "analyze_periodic_topology"),
    "analyze_crystal_stereochemistry": (".chemistry", "analyze_crystal_stereochemistry"),
    "assign_stereochemistry": (".chemistry", "assign_stereochemistry"),
    "annotate_chemistry": (".chemistry", "annotate_chemistry"),
    "infer_chemistry": (".chemistry", "infer_chemistry"),
    "from_line_notation": (".chemistry", "from_line_notation"),
    "name_crystal": (".chemistry", "name_crystal"),
    "name_entity": (".chemistry", "name_entity"),
    "classify_entity_relationship": (".chemistry", "classify_entity_relationship"),
    "to_line_notation": (".chemistry", "to_line_notation"),
}


def __getattr__(name: str):
    if name == "Molecule":
        # Backward-compatibility alias
        from .structures.molecule import CrystalMolecule
        return CrystalMolecule

    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr)
        # Cache on the module so __getattr__ is not called again.
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__) + list(_LAZY_IMPORTS) + ["Molecule", "__version__"]


__all__ = [
    "MolAtom",
    "CrystalMolecule",
    "MolecularCrystal",
    "CrystalTrajectory",
    "BondRecord",
    "SiteRecord",
    "read_mol_crystal",
    "ChemistryIndeterminateError",
    "CrystalChemistry",
    "CrystalStereoClass",
    "CrystalStereoIndeterminateError",
    "CrystalStereoReport",
    "EntityRelationship",
    "LineNotation",
    "LineNotationError",
    "NamingIndeterminateError",
    "NamingKind",
    "NamingResult",
    "PeriodicTopology",
    "StereoDescriptor",
    "StereoKind",
    "StereoReport",
    "analyze_periodic_topology",
    "analyze_crystal_stereochemistry",
    "assign_stereochemistry",
    "annotate_chemistry",
    "infer_chemistry",
    "from_line_notation",
    "name_crystal",
    "name_entity",
    "classify_entity_relationship",
    "to_line_notation",
    # Backward compatibility
    "Molecule",
]
