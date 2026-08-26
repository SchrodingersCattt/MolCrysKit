"""Bridge crystal records into immutable chemistry-domain entities."""

from __future__ import annotations

from .models import (
    BondKind,
    ChemicalAtom,
    ChemicalBond,
    CrystalChemistry,
    Embedding,
    Evidence,
    EvidenceSource,
    FiniteChemicalEntity,
    InferenceStatus,
)


class ChemistryIndeterminateError(ValueError):
    """Raised when strict annotation encounters unresolved chemistry."""


def annotate_chemistry(structure, *, strict: bool = False) -> CrystalChemistry:
    """Attach connectivity-preserving chemistry entities to a crystal.

    This first-stage annotation deliberately carries the source topology into
    the chemistry domain without inventing bond orders or atomic charges.
    Later perception passes may replace provisional fields. ``strict=True``
    rejects this unresolved state instead of attaching a provisional result.

    Parameters
    ----------
    structure : MolecularCrystal
        Crystal whose public site and bond records define the mapping.
    strict : bool, default False
        Reject unresolved bond orders and charges.
    """
    sites = structure.get_site_records()
    bonds = structure.get_bond_records()
    site_by_global = {site.global_index: site for site in sites}
    bonds_by_molecule: dict[int, list] = {}
    for bond in bonds:
        bonds_by_molecule.setdefault(bond.molecule_index, []).append(bond)

    evidence = Evidence(
        source=EvidenceSource.INFERRED,
        method="molcrys_kit_connectivity_graph",
        detail="Connectivity only; bond orders and atomic charges unresolved.",
    )
    warning = "Bond orders and atomic charges are unresolved; chemistry is provisional."
    if strict and (sites or bonds):
        raise ChemistryIndeterminateError(warning)

    components = []
    for molecule_index, molecule in enumerate(structure.molecules):
        molecule_sites = sorted(
            (site for site in sites if site.molecule_index == molecule_index),
            key=lambda site: site.local_index,
        )
        atoms = tuple(
            ChemicalAtom(
                atom_id=site.site_id,
                element=site.symbol,
                label=site.label,
                isotope=site.isotope,
                formal_charge=site.formal_charge,
                evidence=(
                    Evidence(
                        source=EvidenceSource.EXPLICIT_CIF,
                        method="cif_atom_site",
                    ),
                )
                if site.isotope is not None or site.formal_charge is not None
                else (evidence,),
            )
            for site in molecule_sites
        )
        entity_bonds = tuple(
            ChemicalBond(
                atom1_id=site_by_global[bond.left_global_index].site_id,
                atom2_id=site_by_global[bond.right_global_index].site_id,
                order=None,
                kind=BondKind.UNKNOWN,
                atom2_image_shift=bond.right_image_shift,
                evidence=(evidence,),
            )
            for bond in bonds_by_molecule.get(molecule_index, ())
        )
        embedding = Embedding(
            coordinates_A=tuple(
                (site.site_id, site.cartesian_position_A) for site in molecule_sites
            ),
            evidence=(evidence,),
        )
        entity = FiniteChemicalEntity(
            entity_id=f"molecule:{molecule_index}",
            atoms=atoms,
            bonds=entity_bonds,
            embedding=embedding,
            net_charge=(
                sum(atom.formal_charge for atom in atoms if atom.formal_charge is not None)
                if atoms and all(atom.formal_charge is not None for atom in atoms)
                else None
            ),
            status=InferenceStatus.PROVISIONAL,
            evidence=(evidence,),
            warnings=(warning,),
        )
        molecule.chemical_entity = entity
        components.append(entity)

    ordered_atom_ids = tuple(
        site.site_id for site in sorted(sites, key=lambda site: site.global_index)
    )
    result = CrystalChemistry(
        components=tuple(components),
        atom_ids_by_global_index=ordered_atom_ids,
        status=InferenceStatus.PROVISIONAL if components else InferenceStatus.INDETERMINATE,
        evidence=(evidence,),
        warnings=(warning,) if components else ("Crystal contains no chemical components.",),
    )
    structure._chemistry = result
    return result


__all__ = ["ChemistryIndeterminateError", "annotate_chemistry"]
