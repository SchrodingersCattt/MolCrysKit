from pathlib import Path
from ase.io import write
from molcrys_kit.io.cif import read_mol_crystal
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.operations.defects import VacancyGenerator

cif_path = Path("output/disorder_resolution/TILPEN01_optimal_0.cif")
if not cif_path.exists():
    raise FileNotFoundError(
        f"{cif_path} does not exist. Contact the author for help.  "
    )

crystal = read_mol_crystal(str(cif_path)).get_supercell(2, 2, 2)

analyzer = StoichiometryAnalyzer(crystal)
analyzer.print_species_summary()

generator = VacancyGenerator(crystal)

# Generate two different vacancy-defect crystals
defect_crystal_1 = generator.generate_vacancy(
    method='spatial_cluster',
    target_spec={
        "C5FeN6O_1": 1, 
        "C5H12N_1": 1, 
        "C2H8N_1": 1
    },
    seed_index=0,
    return_removed_cluster=True
)
defect_crystal_2 = generator.generate_vacancy(
    method='spatial_cluster',
    target_spec={
        "C5FeN6O_1": 1, 
        "C5H12N_1": 2
    },
    seed_index=0,
    return_removed_cluster=True
)
defect_crystal_3 = generator.generate_vacancy(
    method='spatial_cluster',
    target_spec={
        "C5FeN6O_1": 1, 
        "C2H8N_1": 2
    },
    seed_index=0,
    return_removed_cluster=True
)

# Convert to ASE Atoms
ase_defect_1, ase_removed_1 = defect_crystal_1[0].to_ase(), defect_crystal_1[1].to_ase()
ase_defect_2, ase_removed_2 = defect_crystal_2[0].to_ase(), defect_crystal_2[1].to_ase()
ase_defect_3, ase_removed_3 = defect_crystal_3[0].to_ase(), defect_crystal_3[1].to_ase()

print(f"Defect 1 atoms: {len(ase_defect_1)}")
print(f"Defect 2 atoms: {len(ase_defect_2)}")
print(f"Defect 3 atoms: {len(ase_defect_3)}")
write("output/defect_test/perfect.cif", crystal.to_ase())
write("output/defect_test/defect_1.cif", ase_defect_1)
write("output/defect_test/defect_2.cif", ase_defect_2)
write("output/defect_test/defect_3.cif", ase_defect_3)

write("output/defect_test/removed_1.cif", ase_removed_1)
write("output/defect_test/removed_2.cif", ase_removed_2)
write("output/defect_test/removed_3.cif", ase_removed_3)

