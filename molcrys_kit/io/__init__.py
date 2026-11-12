"""Input and output for MolCrysKit.

Readers that need ASE or pymatgen are imported on first use so the CIF path
can load without them.
"""

_EXPORTS = {
    "DisorderInfo": (".cif", "DisorderInfo"),
    "identify_molecule_indices": (".cif", "identify_molecule_indices"),
    "parse_cif_advanced": (".cif", "parse_cif_advanced"),
    "read_mol_crystal": (".cif", "read_mol_crystal"),
    "read_cif_symmetry": (".cif", "read_cif_symmetry"),
    "scan_cif_disorder": (".cif", "scan_cif_disorder"),
    "read_extxyz": (".extxyz", "read_extxyz"),
    "write_extxyz": (".extxyz", "write_extxyz"),
    "write_cif": (".output", "write_cif"),
    "write_cif_asu": (".output", "write_cif_asu"),
    "write_cif_p1": (".output", "write_cif_p1"),
    "write_disorder_cif": (".output", "write_disorder_cif"),
    "write_disorder_cif_asu": (".output", "write_disorder_cif_asu"),
    "write_forward_cif": (".output", "write_forward_cif"),
    "describe_structure_export": (".output", "describe_structure_export"),
    "write_cif_sequence": (".output", "write_cif_sequence"),
    "write_poscar": (".output", "write_poscar"),
    "write_poscar_sequence": (".output", "write_poscar_sequence"),
    "write_trajectory": (".output", "write_trajectory"),
    "write_xyz": (".output", "write_xyz"),
    "write_xyz_with_freeze": (".output", "write_xyz_with_freeze"),
    "read_xyz": (".xyz", "read_xyz"),
    "read_poscar": (".poscar", "read_poscar"),
    "read_periodic_bundle": (".periodic_bundle", "read_periodic_bundle"),
    "write_periodic_bundle": (".periodic_bundle", "write_periodic_bundle"),
    "from_pymatgen_structure": (".pymatgen_io", "from_pymatgen_structure"),
    "to_pymatgen_structure": (".pymatgen_io", "to_pymatgen_structure"),
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
