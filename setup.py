"""
Setup script for MolCrysKit package.
"""

from setuptools import setup, find_packages

setup(
    name="molcrys-kit",
    version="0.1.0",
    description="Molecular Crystal Toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SchrodingersCattt",
    url="https://github.com/SchrodingersCattt/MolCrysKit",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
    ],
    extras_require={
        "io": ["pymatgen>=2020.0.0"],
        "vis": ["nglview", "py3Dmol"],
        "test": ["pytest>=6.0.0"],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="molecular crystal, crystallography, chemistry, materials science",
)