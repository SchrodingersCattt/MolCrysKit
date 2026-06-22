"""Chemical identity annotations for interaction analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..charge import MolChargeResult, compute_topo_signature
from ..formula_moiety import heavy_signature, match_molecule_to_fragment, parse_moiety_string


@dataclass(frozen=True)
class ChemicalIdentity:
    """Molecule/fragment/species-level identity annotation."""

    molecule_index: int
    formula: str
    hill_formula: str | None = None
    heavy_signature: tuple[tuple[str, int], ...] | None = None
    topo_signature: str | None = None
    species_id: str | None = None
    moiety_raw: str | None = None
    moiety_charge: int | None = None
    formal_charge: float | None = None
    charge_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_molecule(
        cls,
        molecule,
        molecule_index: int,
        *,
        crystal=None,
        species_id: str | None = None,
        charge_result: MolChargeResult | None = None,
        include_topology: bool = True,
    ) -> "ChemicalIdentity":
        """Build identity from a molecule and optional crystal-level annotations."""
        symbols = molecule.get_chemical_symbols()
        formula = molecule.get_chemical_formula()
        hill_formula = molecule.get_chemical_formula(mode="hill", empirical=False)
        composition = dict(Counter(symbols))
        hs = heavy_signature(composition)

        topo_signature = None
        if include_topology:
            try:
                topo_signature = compute_topo_signature(molecule)
            except Exception:
                topo_signature = None

        moiety_raw = None
        moiety_charge = None
        fragments = parse_moiety_string(getattr(crystal, "formula_moiety", None))
        if fragments:
            fragment = match_molecule_to_fragment(symbols, fragments)
            if fragment is not None:
                moiety_raw = fragment.raw
                moiety_charge = fragment.charge

        formal_charge = None
        charge_source = None
        if charge_result is not None:
            formal_charge = charge_result.formal_charge
            charge_source = charge_result.source

        return cls(
            molecule_index=int(molecule_index),
            formula=formula,
            hill_formula=hill_formula,
            heavy_signature=hs,
            topo_signature=topo_signature,
            species_id=species_id,
            moiety_raw=moiety_raw,
            moiety_charge=moiety_charge,
            formal_charge=formal_charge,
            charge_source=charge_source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "molecule_index": self.molecule_index,
            "formula": self.formula,
            "hill_formula": self.hill_formula,
            "heavy_signature": [list(item) for item in self.heavy_signature or ()],
            "topo_signature": self.topo_signature,
            "species_id": self.species_id,
            "moiety_raw": self.moiety_raw,
            "moiety_charge": self.moiety_charge,
            "formal_charge": self.formal_charge,
            "charge_source": self.charge_source,
            "metadata": dict(self.metadata),
        }


class ChemicalIdentityCache:
    """Lazy cache of ``ChemicalIdentity`` objects for crystal molecules."""

    def __init__(
        self,
        crystal,
        *,
        species_ids: dict[int, str] | None = None,
        charge_results: dict[str, MolChargeResult] | None = None,
        include_topology: bool = True,
    ):
        self.crystal = crystal
        self.species_ids = species_ids or {}
        self.charge_results = charge_results or {}
        self.include_topology = include_topology
        self._cache: dict[int, ChemicalIdentity] = {}

    def get(self, molecule_index: int) -> ChemicalIdentity:
        """Return identity for a molecule index."""
        molecule_index = int(molecule_index)
        if molecule_index not in self._cache:
            molecule = self.crystal.molecules[molecule_index]
            topo_signature = None
            if self.include_topology:
                try:
                    topo_signature = compute_topo_signature(molecule)
                except Exception:
                    topo_signature = None
            charge_result = self.charge_results.get(topo_signature)
            self._cache[molecule_index] = ChemicalIdentity.from_molecule(
                molecule,
                molecule_index,
                crystal=self.crystal,
                species_id=self.species_ids.get(molecule_index),
                charge_result=charge_result,
                include_topology=self.include_topology,
            )
        return self._cache[molecule_index]

    def __getitem__(self, molecule_index: int) -> ChemicalIdentity:
        return self.get(molecule_index)