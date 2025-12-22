
from molcrys_kit.io import read_mol_crystal

# Read a CIF file with automatic molecule identification
test_structure = "/aisi-nas/guomingyu/personal/eap8/heating/confs/DAP-4_order.cif"
crystal = read_mol_crystal(test_structure)

# Access molecules and their fractional coordinates
for i, molecule in enumerate(crystal.molecules):
    # Get centroid in Cartesian coordinates
    centroid_cart = molecule.get_centroid()
    formula = molecule.get_chemical_formula()
    centroid_frac = molecule.get_centroid_frac()
    
    print(f"Molecule {i+1}:")
    print(f"  Cartesian centroid: [{centroid_cart[0]:.4f}, {centroid_cart[1]:.4f}, {centroid_cart[2]:.4f}]")
    print(f"  Formula: {formula}")
    print(f"  Fractional centroid: [{centroid_frac[0]:.4f}, {centroid_frac[1]:.4f}, {centroid_frac[2]:.4f}]")