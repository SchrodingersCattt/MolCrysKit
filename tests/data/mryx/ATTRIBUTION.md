# Public Mercury `.mryx` fixture attribution

The bundled file `fixtures/zenodo-5aa-triclinic-anonymized.mryx` is a **derived,
anonymized** copy of a Mercury export from an open Zenodo deposit. It is **not**
byte-identical to the upstream file.

## Upstream work (CC BY 4.0)

| Field | Value |
|-------|-------|
| Record | [Zenodo 14614827](https://zenodo.org/records/14614827) |
| DOI | `10.5281/zenodo.14614827` |
| Article | `10.1021/acs.joc.4c02538` — *1,4-Dihydropyrrolo[3,2-b]pyrroles with Two Embedded Heptagons via Alkyne Annulation* |
| Upstream file | `Crystal structure,5aa.mryx` |
| Upstream SHA-256 | `645ed350f516e7cfc94a21cbfed021084cf5ba5689d371cd4149f08800ee188c` |
| License | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) |

Under CC BY 4.0, adapted redistribution is permitted when the license and
attribution are preserved. This repository satisfies that requirement via this
notice and `public_manifest.json` (`upstream` block).

## Derived fixture

| Field | Value |
|-------|-------|
| Bundled path | `fixtures/zenodo-5aa-triclinic-anonymized.mryx` |
| Derived SHA-256 | `dc8ae7c7360c3836643a297c64fe10ac1df85bc268e37cb97667c9446b07eb53` |
| Creators (upstream work) | Gana Sanil; Łukasz Dobrzycki; Michał K. Cyrański; Denis Jacquemin; Wojciech Chaładaj; Daniel T. Gryko (Crossref, `10.1021/acs.joc.4c02538`) |
| Anonymization | Replaced structure display label `GSA_o372_ACN` → `mck_tier_a_triclinic_anon`; atom site labels, coordinates, cell, and symmetry metadata unchanged |
| Deterministic re-derive | `derive_fixture.stable_gzip_compress` (fixed mtime/OS header + zlib raw deflate; portable 3.10–3.12) |

Anonymization reduces gratuitous dataset-specific naming in tracked bytes; it
does **not** remove the obligation to cite the upstream Zenodo deposit when
publishing work that relies on this structure.

## Not included

CCDC Mercury tutorial / private coordinator exports used for tier-(B) review are
**not** redistributed here (separate licence terms).