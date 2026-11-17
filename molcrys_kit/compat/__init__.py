"""Private fallbacks used only when the desktop extra is not installed.

These types are not a public ASE or pymatgen API. Desktop code keeps using
ASE, pymatgen, and SciPy. The core install (NumPy, NetworkX, and Click) uses
this package for CIF reading, special-position clustering, and structure export.
"""
