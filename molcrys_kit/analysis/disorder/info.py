from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DisorderInfo:
    """Data container for storing raw extracted disorder-related data from CIF files.
    
    This class stores the raw metadata exactly as it appears in the file, without any
    logical processing or cleaning.
    """
    labels: List[str]  # Original atom labels (e.g., "C1A", "H2'")
    symbols: List[str]  # Element symbols
    frac_coords: np.ndarray  # nx3 array of fractional coordinates
    occupancies: List[float]  # Site occupancy, default to 1.0 if missing
    disorder_groups: List[int]  # Integer tags for disorder groups, default to 0 if missing or '.'

    def __post_init__(self):
        """Validate the data after initialization."""
        n_atoms = len(self.labels)
        if len(self.symbols) != n_atoms:
            raise ValueError(f"Length mismatch: got {n_atoms} labels but {len(self.symbols)} symbols")
        if len(self.frac_coords) != n_atoms:
            raise ValueError(f"Length mismatch: got {n_atoms} labels but {len(self.frac_coords)} frac_coords")
        if len(self.occupancies) != n_atoms:
            raise ValueError(f"Length mismatch: got {n_atoms} labels but {len(self.occupancies)} occupancies")
        if len(self.disorder_groups) != n_atoms:
            raise ValueError(f"Length mismatch: got {n_atoms} labels but {len(self.disorder_groups)} disorder_groups")

    def summary(self):
        """Print statistics about the disorder information."""
        n_atoms = len(self.labels)
        n_disordered = sum(1 for group in self.disorder_groups if group != 0)
        
        print(f"Total atoms: {n_atoms}")
        print(f"Atoms in disorder groups: {n_disordered}")
        print(f"Unique disorder groups: {len(set(self.disorder_groups))}")
        print(f"Min occupancy: {min(self.occupancies) if self.occupancies else 'N/A'}")
        print(f"Max occupancy: {max(self.occupancies) if self.occupancies else 'N/A'}")
        print(f"Atoms with occupancy < 1.0: {sum(1 for occ in self.occupancies if occ < 1.0)}")
        
        # Show unique labels with suffixes to verify they're preserved
        labels_with_suffixes = [label for label in self.labels if any(c.isalpha() and c != label[0] or c.isdigit() and label.index(c) > 0 for c in label[1:])]
        print(f"Labels with suffixes: {len(labels_with_suffixes)}")
        if labels_with_suffixes[:5]:  # Show first 5 examples
            print(f"Examples: {labels_with_suffixes[:5]}")