"""Analysis routines.

The package surface stays importable without pymatgen, ASE, or SciPy.
Individual analyses import those libraries only when called.
"""

_EXPORTS = {
    "BFDHFacetInfo": (".bfdh", "BFDHFacetInfo"),
    "enumerate_bfdh_facets": (".bfdh", "enumerate_bfdh_facets"),
    "enumerate_low_index_millers": (".bfdh", "enumerate_low_index_millers"),
    "MolChargeResult": (".charge", "MolChargeResult"),
    "assign_mol_formal_charges": (".charge", "assign_mol_formal_charges"),
    "compute_topo_signature": (".charge", "compute_topo_signature"),
    "CellIntegrityReport": (".cell_integrity", "CellIntegrityReport"),
    "check_cell_integrity": (".cell_integrity", "check_cell_integrity"),
    "ChemicalEnvironment": (".chemical_env", "ChemicalEnvironment"),
    "Fragment": (".formula_moiety", "Fragment"),
    "heavy_signature": (".formula_moiety", "heavy_signature"),
    "match_molecule_to_fragment": (".formula_moiety", "match_molecule_to_fragment"),
    "parse_moiety_string": (".formula_moiety", "parse_moiety_string"),
    "AtomLocalGeometry": (".interactions", "AtomLocalGeometry"),
    "AtomRef": (".interactions", "AtomRef"),
    "BaseInteraction": (".interactions", "BaseInteraction"),
    "CHPiInteraction": (".interactions", "CHPiInteraction"),
    "CHPiInteractionCriteria": (".interactions", "CHPiInteractionCriteria"),
    "ChemicalIdentity": (".interactions", "ChemicalIdentity"),
    "ChemicalIdentityCache": (".interactions", "ChemicalIdentityCache"),
    "HalogenBond": (".interactions", "HalogenBond"),
    "HalogenBondCriteria": (".interactions", "HalogenBondCriteria"),
    "HHContact": (".interactions", "HHContact"),
    "HHContactCriteria": (".interactions", "HHContactCriteria"),
    "HydrogenBond": (".interactions", "HydrogenBond"),
    "HydrogenBondCriteria": (".interactions", "HydrogenBondCriteria"),
    "LocalGeometry": (".interactions", "LocalGeometry"),
    "LocalGeometryCache": (".interactions", "LocalGeometryCache"),
    "PiStacking": (".interactions", "PiStacking"),
    "PiStackingCriteria": (".interactions", "PiStackingCriteria"),
    "RingGeometry": (".interactions", "RingGeometry"),
    "RingRef": (".interactions", "RingRef"),
    "build_crystal_atom_offsets": (".interactions", "build_crystal_atom_offsets"),
    "find_ch_pi": (".interactions", "find_ch_pi"),
    "find_ch_pi_interactions": (".interactions", "find_ch_pi_interactions"),
    "find_h_h_contacts": (".interactions", "find_h_h_contacts"),
    "find_halogen_bonds": (".interactions", "find_halogen_bonds"),
    "find_hydrogen_bonds": (".interactions", "find_hydrogen_bonds"),
    "find_pi_stacking": (".interactions", "find_pi_stacking"),
    "find_pi_stacks": (".interactions", "find_pi_stacks"),
    "get_bonding_threshold": (".interactions.bonding", "get_bonding_threshold"),
    "DEFAULT_CENTROID_OFFSET_FRAC": (".packing_shell", "DEFAULT_CENTROID_OFFSET_FRAC"),
    "DEFAULT_MOLECULAR_SEARCH_CUTOFF": (".packing_shell", "DEFAULT_MOLECULAR_SEARCH_CUTOFF"),
    "DEFAULT_POLYHEDRON_SEARCH_CUTOFF": (".packing_shell", "DEFAULT_POLYHEDRON_SEARCH_CUTOFF"),
    "angular_rmsd_vs_ideals": (".packing_shell", "angular_rmsd_vs_ideals"),
    "compute_angular_signature": (".packing_shell", "compute_angular_signature"),
    "detect_coordination_number": (".packing_shell", "detect_coordination_number"),
    "detect_prism_vs_antiprism": (".packing_shell", "detect_prism_vs_antiprism"),
    "find_polyhedra": (".packing_shell", "find_polyhedra"),
    "hull_encloses_center": (".packing_shell", "hull_encloses_center"),
    "planarity_analysis": (".packing_shell", "planarity_analysis"),
    "BoundaryShift": (".polyhedron_network", "BoundaryShift"),
    "CentreId": (".polyhedron_network", "CentreId"),
    "ContactChange": (".polyhedron_network", "ContactChange"),
    "CoordinationShell": (".polyhedron_network", "CoordinationShell"),
    "CoordinationSnapshot": (".polyhedron_network", "CoordinationSnapshot"),
    "PolyhedronComponent": (".polyhedron_network", "PolyhedronComponent"),
    "PolyhedronNetwork": (".polyhedron_network", "PolyhedronNetwork"),
    "PolyhedronNetworkReport": (".polyhedron_network", "PolyhedronNetworkReport"),
    "SharedContact": (".polyhedron_network", "SharedContact"),
    "SharedVertex": (".polyhedron_network", "SharedVertex"),
    "ShellBoundary": (".polyhedron_network", "ShellBoundary"),
    "ShellCandidate": (".polyhedron_network", "ShellCandidate"),
    "ShellVertex": (".polyhedron_network", "ShellVertex"),
    "UnperturbedCentre": (".polyhedron_network", "UnperturbedCentre"),
    "molecular_polyhedron_network": (".polyhedron_network", "molecular_polyhedron_network"),
    "polyhedron_network_from_shells": (".polyhedron_network", "polyhedron_network_from_shells"),
    "validate_periodic_bundle": (".periodic_validation", "validate_periodic_bundle"),
    "DegenerateRingGeometryError": (".ring_conformation", "DegenerateRingGeometryError"),
    "InvalidRingOrderError": (".ring_conformation", "InvalidRingOrderError"),
    "PuckeringCoordinates": (".ring_conformation", "PuckeringCoordinates"),
    "RingConformationError": (".ring_conformation", "RingConformationError"),
    "RingCycleLimitError": (".ring_conformation", "RingCycleLimitError"),
    "RingSystem": (".ring_conformation", "RingSystem"),
    "find_ring_systems": (".ring_conformation", "find_ring_systems"),
    "puckering_coordinates": (".ring_conformation", "puckering_coordinates"),
    "reconstruct_z_from_modes": (".ring_conformation", "reconstruct_z_from_modes"),
    "CheckResult": (".sanity_check", "CheckResult"),
    "SanityReport": (".sanity_check", "SanityReport"),
    "check_bond_distances": (".sanity_check", "check_bond_distances"),
    "check_formula_consistency": (".sanity_check", "check_formula_consistency"),
    "check_hard_clash": (".sanity_check", "check_hard_clash"),
    "check_hydrogen_presence": (".sanity_check", "check_hydrogen_presence"),
    "check_intermolecular_clash": (".sanity_check", "check_intermolecular_clash"),
    "check_isolated_atoms": (".sanity_check", "check_isolated_atoms"),
    "check_topology_preservation": (".sanity_check", "check_topology_preservation"),
    "sanity_check": (".sanity_check", "sanity_check"),
    "classify_shell": (".shape", "classify_shell"),
    "cshm": (".shape", "cshm"),
    "topology_signature": (".shape", "topology_signature"),
    "assign_atoms_to_molecules": (".species", "assign_atoms_to_molecules"),
    "identify_molecules": (".species", "identify_molecules"),
    "FormulaUnitMember": (".stoichiometry", "FormulaUnitMember"),
    "FormulaUnitSelection": (".stoichiometry", "FormulaUnitSelection"),
    "StoichiometryAnalyzer": (".stoichiometry", "StoichiometryAnalyzer"),
    "summarize_structure": (".structure_summary", "summarize_structure"),
    "calculate_accessible_boundary": (".volume", "calculate_accessible_boundary"),
    "calculate_atomic_volumes": (".volume", "calculate_atomic_volumes"),
    "calculate_total_volume": (".volume", "calculate_total_volume"),
    "min_distance_to_boundary": (".volume", "min_distance_to_boundary"),
    "SiteSymmetryAnalysis": (".site_symmetry", "SiteSymmetryAnalysis"),
    "analyze_molecular_pose": (".site_symmetry", "analyze_molecular_pose"),
    "analyze_site_symmetry": (".site_symmetry", "analyze_site_symmetry"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        module_name, attribute = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
