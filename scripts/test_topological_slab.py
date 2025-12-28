import os
import sys
import matplotlib.pyplot as plt

# Use absolute path for the project
project_root = "/aisi-nas/guomingyu/personal/MolCrysKit"
sys.path.insert(0, project_root)

try:
    from molcrys_kit.io.cif import read_mol_crystal
    from molcrys_kit.operations import generate_topological_slab
    from molcrys_kit.io.cif import identify_molecules
    import pymatgen
    from ase import Atoms
    from ase.visualize.plot import plot_atoms

    PYMATGEN_AVAILABLE = True
    ASE_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    PYMATGEN_AVAILABLE = False
    ASE_AVAILABLE = False


def main():
    """Main function to test topological slab generation."""
    if not PYMATGEN_AVAILABLE or not ASE_AVAILABLE:
        print("Error: Required packages (pymatgen and/or ASE) not available")
        return

    cif_path = os.path.join(project_root, "examples", "PUBMUU03.cif")

    # Check if the file exists
    if not os.path.exists(cif_path):
        print("Please provide the file in the examples directory.")
        return

    print(f"Loading crystal from {cif_path}...")
    crystal = read_mol_crystal(cif_path)

    print("Original crystal:")
    print(f"  Number of molecules: {len(crystal.molecules)}")
    print(f"  Total atoms: {sum(len(mol) for mol in crystal.molecules)}")

    # Show molecule sizes
    molecule_sizes = [len(mol) for mol in crystal.molecules]
    unique_sizes = sorted(list(set(molecule_sizes)))
    print(f"  Unique molecule sizes: {unique_sizes}")

    print("\nGenerating topological slab...")
    print("  Layers: 3")
    print("  Vacuum: 10.0 Angstroms")

    try:
        slab = generate_topological_slab(
            crystal=crystal,
            miller_indices=(0, 1, 0),
            layers=3,
            # min_thickness=10,
            vacuum=10.0,
        )

        print("\nGenerated slab:")
        print(f"  Number of molecules: {len(slab.molecules)}")
        print(f"  Total atoms: {sum(len(mol) for mol in slab.molecules)}")
        a, b, c, alpha, beta, gamma = slab.get_lattice_parameters()
        print(
            f"  Lattice parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}, α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}"
        )

        # Verification
        print("\nPerforming verification checks...")

        # Check 1: Atom count should be original_atoms * layers
        original_atoms = sum(len(mol) for mol in crystal.molecules)
        slab_atoms = sum(len(mol) for mol in slab.molecules)
        expected_atoms = original_atoms * 3  # 3 layers

        if slab_atoms == expected_atoms:
            print(
                f"✓ Atom count check passed: {slab_atoms} = {original_atoms} × 3 layers"
            )
        else:
            print(f"✗ Atom count check failed: {slab_atoms} ≠ {expected_atoms}")

        # Check 2: Verify no fragmented molecules using graph analysis
        # Convert slab to ASE Atoms using the new to_ase method
        slab_atoms_obj = slab.to_ase()

        # Identify molecules in the slab
        slab_molecules = identify_molecules(slab_atoms_obj)

        print(f"  Identified molecules in slab: {len(slab_molecules)}")

        # Check that all molecules are complete (same sizes as original)
        slab_molecule_sizes = sorted([len(mol) for mol in slab_molecules])
        original_molecule_sizes = sorted([len(mol) for mol in crystal.molecules])

        # For a 3-layer slab, we expect 3 times the original molecules
        expected_molecule_count = len(crystal.molecules) * 3

        if len(slab_molecules) == expected_molecule_count:
            print(
                f"✓ Molecule count check passed: {len(slab_molecules)} = {len(crystal.molecules)} × 3 layers"
            )
        else:
            print(
                f"✗ Molecule count check failed: {len(slab_molecules)} ≠ {expected_molecule_count}"
            )

        # Check if molecule sizes match expected pattern
        # In a 3-layer slab, we should have 3 copies of each original molecule size
        expected_sizes = sorted(original_molecule_sizes * 3)
        if slab_molecule_sizes == expected_sizes:
            print("✓ Molecule size distribution check passed")
        else:
            print("✗ Molecule size distribution check failed")
            print(f"  Expected: {expected_sizes}")
            print(f"  Actual: {slab_molecule_sizes}")

        # Save the slab to a CIF file
        print("\nSaving slab...")

        # Make sure output directory exists
        output_dir = os.path.join(project_root, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert to ASE Atoms for writing and plotting using to_ase method
        output_atoms = slab.to_ase()

        output_path = os.path.join(output_dir, "SLAB_topo.cif")
        output_atoms.write(output_path)
        print(f"Slab structure saved to {output_path}")

        # Plotting along a and b axes
        print("Generating visualization plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # View along a-axis
        plot_atoms(output_atoms, ax1, rotation="0x,90y,0z")
        ax1.set_title("View along a-axis")
        ax1.set_axis_off()

        # View along b-axis
        plot_atoms(output_atoms, ax2, rotation="90x,0y,0z")
        ax2.set_title("View along b-axis")
        ax2.set_axis_off()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "debug_views.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Visualization saved to {plot_path}")

    except Exception as e:
        print(f"Error generating topological slab: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
