# Periodic-chain fixtures

These JSON requests exercise `mck build chain`. They are explicit geometry
fixtures, not complete bulk material models: charge, force-field parameters,
and dynamics are intentionally absent. Running a request produces a local
bundle containing a CIF and `structure.json`; generated bundles are ignored by
source control. The material examples are an idealized red-phosphorus P-P
motif, an all-trans polyethylene repeat, and a trigonal (gray) selenium
helix; each material request places two independent chains in the cell. The
separate `periodic_winding_regression.json` request is a
mathematical topology regression and is not a material example. The selenium
helix is deliberately not labelled alpha-Se: alpha-monoclinic selenium is a
Se8-ring structure, while the helical chain is the trigonal allotrope.
