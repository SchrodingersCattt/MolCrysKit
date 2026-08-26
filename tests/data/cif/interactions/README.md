# Synthetic interaction CIF fixtures

These P1 structures are synthetic regression fixtures created for MolCrysKit.
They do not contain coordinates derived from the CSD or another structure
database. Each fixture places chemically complete molecules in an oversized
orthogonal cell so one intended intermolecular geometry is isolated from
unrelated periodic contacts.

| Fixture | Intended geometry |
|---------|-------------------|
| `hydrogen_bond.cif` | One linear O–H···O contact: H···O = 1.84 Å, O–H···O = 180° |
| `halogen_bond.cif` | One linear C–Cl···O contact: Cl···O = 3.00 Å, C–Cl···O = 180° |
| `pi_parallel.cif` | Two parallel benzene rings separated by 3.40 Å |
| `pi_t_shape.cif` | Two perpendicular benzene rings with centroid distance 4.20 Å |
| `multi_interaction.cif` | Two O–H···O contacts plus one C–Cl···O contact |
