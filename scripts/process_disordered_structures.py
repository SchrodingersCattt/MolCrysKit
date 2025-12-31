#!/usr/bin/env python
"""
Process disordered CIF structures and save resolved structures.

This script processes the three example disordered CIF files:
- examples/PAP-M5.cif
- examples/PAP-H4.cif
- examples/DAP-4.cif

It uses the disorder handling pipeline to generate ordered structures
and saves them to the output directory.
"""

import os
import traceback
from pathlib import Path
from molcrys_kit.analysis.disorder import process_disordered_cif
from molcrys_kit.io import write_cif


def process_disordered_cif_files():
    """Process disordered CIF files and save resolved structures."""
    
    # Define the input files to process
    input_files = [
        "examples/PAP-M5.cif",
        "examples/PAP-H4.cif", 
        "examples/DAP-4.cif",
        "examples/EAP-8.cif"
    ]
    
    # Create output directory if it doesn't exist
    output_dir = Path("output/disorder_resolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cif_file in input_files:
        if not os.path.exists(cif_file):
            print(f"Warning: {cif_file} does not exist, skipping...")
            continue
            
        print(f"Processing {cif_file}...")
        
        # Get base filename without extension for naming outputs
        base_name = Path(cif_file).stem
        
        # Process with optimal method
        try:
            optimal_structures = process_disordered_cif(
                cif_file, 
                generate_count=1, 
                method='optimal'
            )
            
            # Save optimal structure
            optimal_filename = output_dir / f"{base_name}_optimal_0.cif"
            write_cif(optimal_structures[0], str(optimal_filename))
            print(f"  - Saved optimal structure to {optimal_filename}")
            
        except Exception as e:
            traceback.print_exc()
            print(f"  - Error processing {cif_file} with optimal method: {e}")
        
        # Process with random method to generate multiple structures
        try:
            random_structures = process_disordered_cif(
                cif_file, 
                generate_count=5, 
                method='random'
            )
            
            # Save multiple random structures
            for i, structure in enumerate(random_structures):
                random_filename = output_dir / f"{base_name}_random_{i}.cif"
                write_cif(structure, str(random_filename))
                print(f"  - Saved random structure {i} to {random_filename}")
                
        except Exception as e:
            print(f"  - Error processing {cif_file} with random method: {e}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    process_disordered_cif_files()