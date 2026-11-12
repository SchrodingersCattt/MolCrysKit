"""Structural components for MolCrysKit.

Heavy optional dependencies (ASE, SciPy, pymatgen) stay behind the submodules
that need them. Importing ``molecule`` or ``crystal`` does not load them.
"""

_EXPORTS = {
    "MolAtom": (".atom", "MolAtom"),
    "CrystalMolecule": (".molecule", "CrystalMolecule"),
    "MolecularCrystal": (".crystal", "MolecularCrystal"),
    "CrystalCluster": (".cluster", "CrystalCluster"),
    "ClusterProvenance": (".cluster", "ClusterProvenance"),
    "all_ideal_polyhedra": (".polyhedra", "all_ideal_polyhedra"),
    "convex_hull_payload": (".polyhedra", "convex_hull_payload"),
    "ideal_polyhedra_for_cn": (".polyhedra", "ideal_polyhedra_for_cn"),
    "CrystalTrajectory": (".trajectory", "CrystalTrajectory"),
    "BondRecord": (".records", "BondRecord"),
    "BOND_ROLE_COORDINATION": (".records", "BOND_ROLE_COORDINATION"),
    "BOND_ROLE_COVALENT": (".records", "BOND_ROLE_COVALENT"),
    "BOND_ROLES": (".records", "BOND_ROLES"),
    "CONNECTIVITY_REGIME_EXTENDED": (".records", "CONNECTIVITY_REGIME_EXTENDED"),
    "CONNECTIVITY_REGIME_MOLECULAR": (".records", "CONNECTIVITY_REGIME_MOLECULAR"),
    "CONNECTIVITY_REGIMES": (".records", "CONNECTIVITY_REGIMES"),
    "ComponentConnectivity": (".records", "ComponentConnectivity"),
    "ConnectivityReport": (".records", "ConnectivityReport"),
    "SiteRecord": (".records", "SiteRecord"),
    "classify_bond_role": (".connectivity", "classify_bond_role"),
    "BoundaryPort": (".periodic_geometry", "BoundaryPort"),
    "ChainSpec": (".periodic_geometry", "ChainSpec"),
    "ConnectionRule": (".periodic_geometry", "ConnectionRule"),
    "FragmentInstance": (".periodic_geometry", "FragmentInstance"),
    "FragmentTemplate": (".periodic_geometry", "FragmentTemplate"),
    "PeriodicBundle": (".periodic_geometry", "PeriodicBundle"),
    "PeriodicEdge": (".periodic_geometry", "PeriodicEdge"),
    "PeriodicGraph": (".periodic_geometry", "PeriodicGraph"),
    "ScrewSpec": (".periodic_geometry", "ScrewSpec"),
    "BondCandidates": (".bond", "BondCandidates"),
    "BondPairs": (".bond", "BondPairs"),
    "VerletBondTracker": (".bond", "VerletBondTracker"),
    "build_bond_candidates": (".bond", "build_bond_candidates"),
    "candidate_list_needs_rebuild": (".bond", "candidate_list_needs_rebuild"),
    "evaluate_bond_candidates": (".bond", "evaluate_bond_candidates"),
    "infer_bond_pairs": (".bond", "infer_bond_pairs"),
    "CrystalSymmetry": (".symmetry", "CrystalSymmetry"),
    "FractionalAffineOperation": (".symmetry", "FractionalAffineOperation"),
    "LatticeBasisChange": (".symmetry", "LatticeBasisChange"),
    "domain_representatives": (".symmetry", "domain_representatives"),
    "identity_operation": (".symmetry", "identity_operation"),
    "left_cosets": (".symmetry", "left_cosets"),
    "validate_subgroup": (".symmetry", "validate_subgroup"),
}

__all__ = [
    "MolAtom",
    "CrystalMolecule",
    "MolecularCrystal",
    "CrystalCluster",
    "ClusterProvenance",
    "Molecule",
    "all_ideal_polyhedra",
    "convex_hull_payload",
    "ideal_polyhedra_for_cn",
    "CrystalTrajectory",
    "BondRecord",
    "BondCandidates",
    "BondPairs",
    "BOND_ROLE_COORDINATION",
    "BOND_ROLE_COVALENT",
    "BOND_ROLES",
    "classify_bond_role",
    "ComponentConnectivity",
    "ConnectivityReport",
    "CONNECTIVITY_REGIME_EXTENDED",
    "CONNECTIVITY_REGIME_MOLECULAR",
    "CONNECTIVITY_REGIMES",
    "SiteRecord",
    "BoundaryPort",
    "ChainSpec",
    "ConnectionRule",
    "FragmentInstance",
    "FragmentTemplate",
    "PeriodicBundle",
    "PeriodicEdge",
    "PeriodicGraph",
    "ScrewSpec",
    "VerletBondTracker",
    "build_bond_candidates",
    "candidate_list_needs_rebuild",
    "evaluate_bond_candidates",
    "infer_bond_pairs",
    "CrystalSymmetry",
    "FractionalAffineOperation",
    "LatticeBasisChange",
    "domain_representatives",
    "identity_operation",
    "left_cosets",
    "validate_subgroup",
]


def __getattr__(name: str):
    if name == "Molecule":
        from .molecule import CrystalMolecule

        globals()[name] = CrystalMolecule
        return CrystalMolecule
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
