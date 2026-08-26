"""
MolCrysKit: A Python toolkit for molecular crystal analysis and manipulation.

This toolkit provides functionality for parsing, analyzing, and manipulating
molecular crystal structures, with a particular focus on molecular crystals
where well-defined molecules occupy crystallographic sites.
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

from .structures.atom import MolAtom
from .structures.molecule import CrystalMolecule
from .structures.crystal import MolecularCrystal
from .structures.trajectory import CrystalTrajectory
from .structures.records import BondRecord, SiteRecord
from .io.cif import read_mol_crystal
from .chemistry import (
    ChemistryIndeterminateError,
    CrystalChemistry,
    PeriodicTopology,
    StereoDescriptor,
    StereoKind,
    StereoReport,
    analyze_periodic_topology,
    assign_stereochemistry,
    annotate_chemistry,
    infer_chemistry,
)

# For backward compatibility
Molecule = CrystalMolecule

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
    "PeriodicTopology",
    "StereoDescriptor",
    "StereoKind",
    "StereoReport",
    "analyze_periodic_topology",
    "assign_stereochemistry",
    "annotate_chemistry",
    "infer_chemistry",
    # Backward compatibility
    "Molecule",
]
