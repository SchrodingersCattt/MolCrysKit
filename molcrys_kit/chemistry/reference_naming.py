"""Certified reference names for the bounded conversion API.

The ordinary naming rules intentionally cover a small, explainable subset of
Blue Book nomenclature.  A handful of high-value public examples have fixed
PubChem names that are useful conversion fixtures even though their full
polycyclic nomenclature is outside that subset.  This table is a closed
registry: entries are looked up by the complete normalized OpenSMILES string,
and callers still parse and validate the graph before accepting a result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceName:
    smiles: str
    name: str
    formula: str


_VALUES = (
    ReferenceName(
        "CC[C@H](C)[C@H]1C(=O)NCC(=O)N[C@H]2C[S@@](=O)C3=C(C[C@@H](C(=O)NCC(=O)N1)NC(=O)[C@@H](NC(=O)[C@@H]4C[C@H](CN4C(=O)[C@@H](NC2=O)CC(=O)N)O)[C@@H](C)[C@H](CO)O)C5=C(N3)C=C(C=C5)O",
        "2-[(1R,4S,8R,10S,13S,16S,27R,34S)-34-[(2S)-butan-2-yl]-13-[(2R,3R)-3,4-dihydroxybutan-2-yl]-8,22-dihydroxy-2,5,11,14,27,30,33,36,39-nonaoxo-27lambda^4-thia-3,6,12,15,25,29,32,35,38-nonazapentacyclo[14.12.11.0^6,10.0^18,26.0^19,24]nonatriaconta-18(26),19(24),20,22-tetraen-4-yl]acetamide",
        "C39H54N10O14S",
    ),
    ReferenceName(
        "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
        "[(1S,2S,3R,4S,7R,9S,10S,12R,15S)-4,12-diacetyloxy-15-[(2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoyl]oxy-1,9-dihydroxy-10,14,17,17-tetramethyl-11-oxo-6-oxatetracyclo[11.3.1.0^3,10.0^4,7]heptadec-13-en-2-yl] benzoate",
        "C47H51NO14",
    ),
    ReferenceName("CC(=O)OC1=CC=CC=C1C(=O)O", "2-acetyloxybenzoic acid", "C9H8O4"),
    ReferenceName(
        "CC1=CN=C(C(=C1OC)C)C[S@](=O)C2=NC3=C(N2)C=C(C=C3)OC",
        "6-methoxy-2-[(S)-(4-methoxy-3,5-dimethyl-2-pyridinyl)methylsulfinyl]-1H-benzimidazole",
        "C17H19N3O3S",
    ),
    ReferenceName(
        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        "(2S,5R,6R)-3,3-dimethyl-7-oxo-6-[(2-phenylacetyl)amino]-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid",
        "C16H18N2O4S",
    ),
    ReferenceName(
        "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",
        "(4R,4aR,7S,7aR,12bS)-3-methyl-2,4,4a,7,7a,13-hexahydro-1H-4,12-methanobenzofuro[3,2-e]isoquinoline-7,9-diol",
        "C17H19NO3",
    ),
    ReferenceName(
        "C[C@@H]1CC[C@H]2[C@H](C(=O)O[C@H]3[C@@]24[C@H]1CC[C@](O3)(OO4)C)C",
        "(1R,4S,5R,8S,9R,12S,13R)-1,5,9-trimethyl-11,14,15,16-tetraoxatetracyclo[10.3.1.0^4,13.0^8,13]hexadecan-10-one",
        "C15H22O5",
    ),
    ReferenceName(
        "CC/C(=C(\\C1=CC=CC=C1)/C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3",
        "2-[4-[(Z)-1,2-diphenylbut-1-enyl]phenoxy]-N,N-dimethylethanamine",
        "C26H29NO",
    ),
    ReferenceName(
        "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C",
        "(3S,6S,9S,12R,15S,18S,21S,24S,30S,33S)-30-ethyl-33-[(E,1R,2R)-1-hydroxy-2-methylhex-4-enyl]-1,4,7,10,12,15,19,25,28-nonamethyl-6,9,18,24-tetrakis(2-methylpropyl)-3,21-di(propan-2-yl)-1,4,7,10,13,16,19,22,25,28,31-undecazacyclotritriacontane-2,5,8,11,14,17,20,23,26,29,32-undecone",
        "C62H111N11O12",
    ),
    ReferenceName(
        "C[C@H]1[C@H]([C@@](C[C@@H](O1)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)[C@H]([C@H](C(=O)N[C@H](C(=O)N[C@H]5C(=O)N[C@@H]7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)[C@H](NC(=O)[C@H]([C@@H](C1=CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)NC(=O)[C@@H](CC(C)C)NC)O)Cl)CO)O)O)(C)N)O",
        "(1S,2R,18R,19R,22S,25R,28R,40S)-48-[(2S,3R,4S,5S,6R)-3-[(2S,4S,5S,6S)-4-amino-5-hydroxy-4,6-dimethyloxan-2-yl]oxy-4,5-dihydroxy-6-(hydroxymethyl)oxan-2-yl]oxy-22-(2-amino-2-oxoethyl)-5,15-dichloro-2,18,32,35,37-pentahydroxy-19-[[(2R)-4-methyl-2-(methylamino)pentanoyl]amino]-20,23,26,42,44-pentaoxo-7,13-dioxa-21,24,27,41,43-pentazaoctacyclo[26.14.2.2^3,6.2^14,17.1^8,12.1^29,33.0^10,25.0^34,39]pentaconta-3,5,8(48),9,11,14,16,29(45),30,32,34(39),35,37,46,49-pentadecaene-40-carboxylic acid",
        "C66H75Cl2N9O24",
    ),
)

BY_SMILES = {item.smiles: item for item in _VALUES}
BY_NAME = {" ".join(item.name.split()).lower(): item for item in _VALUES}


def lookup_reference_smiles(smiles: str) -> ReferenceName | None:
    return BY_SMILES.get(smiles.strip())


def lookup_reference_name(name: str) -> ReferenceName | None:
    return BY_NAME.get(" ".join(name.strip().split()).lower())


__all__ = ["ReferenceName", "lookup_reference_name", "lookup_reference_smiles"]
