"""
Configuration module for MolCrysKit.

This module provides access to configuration parameters and constants used throughout the package.
"""

# Standard keys for disorder metadata
KEY_OCCUPANCY = "occupancy"          # Default: 1.0
KEY_DISORDER_GROUP = "disorder_group"  # Default: 0 as integer
KEY_ASSEMBLY = "assembly"             # Default: "" empty string
KEY_LABEL = "label"                   # Default: Atom element symbol

# Transition metals set
TRANSITION_METALS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}

# Global threshold factors for bond detection
BONDING_CONFIG = {
    "METAL_THRESHOLD_FACTOR": 0.5,
    "NON_METAL_THRESHOLD_FACTOR": 1.25,
    "METAL_NON_METAL_THRESHOLD_FACTOR": 0.8,
    "DEFAULT_ATOMIC_RADIUS": 0.5,  # Used when atomic radius is not available
    "MAX_HYDROGEN_BOND_DISTANCE": 3.5,  # Maximum distance for hydrogen bonds in Angstroms
    "MIN_HYDROGEN_BOND_ANGLE": 120,  # Minimum angle for hydrogen bonds in degrees
    "MIN_COVALENT_DISTANCE": 0.8,  # Minimum distance for covalent bond detection
    "MAX_COVALENT_DISTANCE": 1.2,  # Maximum distance for covalent bond detection
    "SIGNIFICANT_ROTATION_ANGLE": 5,  # Minimum angle in degrees to consider for rotation
    "TYPICAL_HYDROGEN_X_DISTANCE": 1.5,  # Typical distance for H-X bonds
}

# Default bond lengths (in Angstroms)
BOND_LENGTHS = {
    "C-H": 1.09,
    "N-H": 1.01,
    "O-H": 0.96,
    "S-H": 1.34,
    "P-H": 1.42,
}

# Default coordination rules for common elements
COORDINATION_RULES = {
    "C": {"geometry": "tetrahedral", "target_coordination": 4},
    "N": {"geometry": "trigonal_pyramidal", "target_coordination": 3},
    "O": {"geometry": "bent", "target_coordination": 2},
    "S": {"geometry": "tetrahedral", "target_coordination": 4},
    "P": {"geometry": "tetrahedral", "target_coordination": 4},
}

# Electronegative elements that commonly participate in hydrogen bonding
ELECTRONEGATIVE_ELEMENTS = ["N", "O", "F"]

# Sp3 hybridized elements
SP3_ELEMENTS = ["C", "N", "O", "S", "P"]

# Configuration for disorder handling
DISORDER_CONFIG = {
    "SYMMETRY_SITE_RADIUS": 3.75,  # Maximum distance (in Angstroms) to consider symmetry-generated
    # atoms as competing for the same physical site
    "VALENCE_PRESCREEN_THRESHOLD": 1.95,  # Tightened to exclude salt interactions
    "HARD_SPHERE_THRESHOLD": 0.85,  # Small threshold. Applied to compatible atoms to allow normal packing
    "DISORDER_CLASH_THRESHOLD": 2.2,  # Large threshold. Applied to COMPETING disorder parts to ensure split sites are exclusive
    "ASSEMBLY_CONFLICT_THRESHOLD": 2.2,  # Large threshold for overlapping Assemblies
    "SKIP_METAL_VALENCE_CHECK": True,  # Skip metal valence checks
}

# Bonding thresholds for disorder analysis
BONDING_THRESHOLDS = {
    # H/D with C, N, O, S, P
    "H_CNO_THRESHOLD_MIN": 0.6,
    "H_CNO_THRESHOLD_MAX": 1.4,
    # H/D with other elements
    "H_OTHER_THRESHOLD_MIN": 0.8,
    "H_OTHER_THRESHOLD_MAX": 1.8,
    # C, N, O with each other
    "CNO_THRESHOLD_MIN": 0.8,
    "CNO_THRESHOLD_MAX": 1.9,
    # C-N, C-O, N-O
    "CNO_PAIR_THRESHOLD_MIN": 0.8,
    "CNO_PAIR_THRESHOLD_MAX": 2.0,
    # General threshold for other element pairs
    "GENERAL_THRESHOLD_MIN": 0.5,
    "GENERAL_THRESHOLD_MAX": 2.2,
    # H-H unlikely to bond
    "HH_BOND_POSSIBLE": False,
    # Metal-Organic bonds threshold
    "METAL_NONMETAL_COVALENT_MAX": 2.05,  # Strict cutoff for Metal-Organic bonds
}

# Maximum coordination numbers for elements
MAX_COORDINATION_NUMBERS = {
    "H": 1,
    "D": 1,
    "C": 4,
    "Si": 4,
    "N": 4,
    "P": 4,  # Usually 3+1 for N/P with lone pair
    "O": 2,
    "S": 2,
    "Se": 2,
    "Te": 2,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    # Common metals
    "Li": 4,
    "Na": 6,
    "K": 8,
    "Mg": 6,
    "Ca": 6,
    "Sr": 6,
    "Ba": 8,
    "Al": 4,
    "Ga": 4,
}

# Default coordination number if element is not in the map
DEFAULT_MAX_COORDINATION = 6

# Common solvents database for desolvation
COMMON_SOLVENTS = {
    "Water": {
        "formula": "H2O",
        "heavy_formula": "O",
        "aliases": ["H2O", "O"]
    },
    "Methanol": {
        "formula": "CH4O",
        "heavy_formula": "CO",
        "aliases": ["CH4O", "MeOH", "Methyl alcohol", "Methyl hydrate", "Wood spirit"]
    },
    "Ethanol": {
        "formula": "C2H6O",
        "heavy_formula": "C2O",
        "aliases": ["C2H6O", "EtOH", "Ethyl alcohol", "Grain alcohol"]
    },
    "Dichloromethane": {
        "formula": "CH2Cl2",
        "heavy_formula": "CCl2",
        "aliases": ["CH2Cl2", "DCM", "Methylene chloride", "Methylene dichloride"]
    },
    "DMF": {
        "formula": "C3H7NO",
        "heavy_formula": "C3NO",
        "aliases": ["C3H7NO", "Dimethylformamide", "Formamide, N,N-dimethyl-"]
    },
    "Chloroform": {
        "formula": "CHCl3",
        "heavy_formula": "CCl3",
        "aliases": ["CHCl3", "Trichloromethane", "Methenyl trichloride"]
    },
    "Acetone": {
        "formula": "C3H6O",
        "heavy_formula": "C3O",
        "aliases": ["C3H6O", "Propanone", "Beta-ketopropane", "Dimethyl ketone"]
    },
    "Toluene": {
        "formula": "C7H8",
        "heavy_formula": "C7",
        "aliases": ["C7H8", "Methylbenzene", "Phenylmethane"]
    },
    "THF": {
        "formula": "C4H8O",
        "heavy_formula": "C4O",
        "aliases": ["C4H8O", "Tetrahydrofuran", "Oxolane", "Furan, tetrahydro-"]
    },
    "Acetonitrile": {
        "formula": "C2H3N",
        "heavy_formula": "C2N",
        "aliases": ["C2H3N", "Ethanonitrile", "Methyl cyanide", "Cyano methane"]
    }
}
