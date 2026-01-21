# MolCrysKit API Reference

This document serves as a placeholder for the MolCrysKit API reference. The API reference provides detailed information about the classes, functions, and methods available in the MolCrysKit package.

## Modules

### molcrys_kit.structures
- [CrystalMolecule](molcrys_kit/structures/molecule.py#L21-L416): Molecular representation with graph connectivity
- [MolecularCrystal](molcrys_kit/structures/crystal.py#L13-L221): Crystal structure composed of molecules
- [Atom](molcrys_kit/structures/atom.py#L8-L47): Individual atom representation

### molcrys_kit.operations
- [add_hydrogens](molcrys_kit/operations/hydrogen_completion.py#L23-L26): Function to add hydrogen atoms based on geometric rules
- [HydrogenCompleter](molcrys_kit/operations/hydrogen_completion.py#L29-L137): Class for adding hydrogen atoms to molecular crystals
- [generate_topological_slab](molcrys_kit/operations/surface.py#L347-L366): Function to generate surface slabs preserving molecular topology
- [TopologicalSlabGenerator](molcrys_kit/operations/surface.py#L56-L344): Class for generating surface slabs while preserving molecular topology
- [generate_vacancy](molcrys_kit/operations/defects.py#L181-L297): Function to generate vacancies by removing molecular clusters

### molcrys_kit.analysis
- [DisorderSolver](molcrys_kit/analysis/disorder/solver.py#L33-L394): Class for solving disorder problems using graph algorithms
- [DisorderGraphBuilder](molcrys_kit/analysis/disorder/graph.py#L28-L258): Class for building exclusion graphs from disorder data
- [ChemicalEnvironment](molcrys_kit/analysis/chemical_env.py#L18-L157): Class for analyzing chemical environments in molecular crystals

### molcrys_kit.io
- [read_mol_crystal](molcrys_kit/io/cif.py#L21-L112): Function to read molecular crystals from CIF files
- [write_cif](molcrys_kit/io/output.py#L11-L40): Function to write crystal structures to CIF files
- [identify_molecules](molcrys_kit/io/cif.py#L217-L319): Function to identify individual molecules in a crystal structure

## Detailed API Documentation

For detailed API documentation, please refer to the docstrings in the source code files.