#!/usr/bin/env python3
"""
Calculate radius of gyration (Rg) for each frame in a trajectory PDB file.
"""

import numpy as np
from collections import defaultdict

# Standard atomic masses
ATOMIC_MASSES = {
    'H': 1.008,
    'C': 12.01,
    'N': 14.01,
    'O': 15.999,
    'P': 30.97,
    'S': 32.06,
    'MG': 24.305,
    'CA': 40.078,
    'FE': 55.845,
    'ZN': 65.38,
}

def parse_trajectory_pdb(filename):
    """Parse a multi-model PDB file and return coordinates for each model."""
    models = []
    current_model = []
    current_model_num = None
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                if current_model:
                    models.append((current_model_num, current_model))
                current_model = []
                current_model_num = int(line.split()[1])
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse atom information
                atom_name = line[12:16].strip()
                element = line[76:78].strip()
                if not element:
                    # Try to infer element from atom name
                    element = atom_name[0]
                
                # Handle coordinates that may not have proper spacing
                coord_string = line[30:54].strip()
                # Split the coordinate string and handle cases where there are no spaces
                coords_parts = []
                current_num = ""
                i = 0
                while i < len(coord_string):
                    char = coord_string[i]
                    if char == ' ':
                        if current_num:
                            coords_parts.append(current_num)
                            current_num = ""
                    elif char == '-' and current_num and current_num[-1] not in ['-', '.', 'e', 'E']:
                        # This is likely the start of a new negative number
                        coords_parts.append(current_num)
                        current_num = char
                    else:
                        current_num += char
                    i += 1
                
                if current_num:
                    coords_parts.append(current_num)
                
                # Clean up and parse coordinates
                clean_coords = []
                for part in coords_parts:
                    part = part.strip()
                    if part:
                        clean_coords.append(part)
                
                if len(clean_coords) >= 3:
                    x = float(clean_coords[0])
                    y = float(clean_coords[1])
                    z = float(clean_coords[2])
                else:
                    continue  # Skip malformed lines
                
                current_model.append({
                    'element': element.upper(),
                    'coords': np.array([x, y, z]),
                    'atom_name': atom_name
                })
            elif line.startswith('ENDMDL'):
                if current_model:
                    models.append((current_model_num, current_model))
                    current_model = []
    
    # Handle last model if file doesn't end with ENDMDL
    if current_model:
        models.append((current_model_num, current_model))
    
    return models

def calculate_rg(atoms):
    """Calculate radius of gyration for a set of atoms."""
    coords = []
    masses = []
    
    for atom in atoms:
        element = atom['element']
        mass = ATOMIC_MASSES.get(element, 12.0)  # Default to carbon mass if unknown
        masses.append(mass)
        coords.append(atom['coords'])
    
    coords = np.array(coords)
    masses = np.array(masses)
    
    # Calculate center of mass
    total_mass = np.sum(masses)
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    
    # Calculate Rg
    distances_squared = np.sum((coords - center_of_mass)**2, axis=1)
    rg_squared = np.sum(distances_squared * masses) / total_mass
    rg = np.sqrt(rg_squared)
    
    return rg

def main():
    pdb_file = "/home/aaron/ATX/software/boltz-dev/boltz/examples/saxs_final_test_k0/boltz_results_rna_rg_guidance_k10/predictions/rna_rg_guidance_k10/rna_rg_guidance_k10_model_0_trajectory.pdb"
    
    print(f"Reading trajectory from: {pdb_file}")
    print("=" * 80)
    
    # Parse the trajectory
    models = parse_trajectory_pdb(pdb_file)
    print(f"Total number of models (frames): {len(models)}")
    print()
    
    # Calculate Rg for each frame
    rg_values = []
    for model_num, atoms in models:
        rg = calculate_rg(atoms)
        rg_values.append((model_num, rg))
    
    # Display results
    print("First 10 frames:")
    print("-" * 40)
    print("Frame    Rg (Å)")
    for i in range(min(10, len(rg_values))):
        frame, rg = rg_values[i]
        print(f"{frame:5d}    {rg:8.3f}")
    
    print("\nMiddle 10 frames:")
    print("-" * 40)
    print("Frame    Rg (Å)")
    mid_start = len(rg_values) // 2 - 5
    mid_end = mid_start + 10
    for i in range(max(0, mid_start), min(len(rg_values), mid_end)):
        frame, rg = rg_values[i]
        print(f"{frame:5d}    {rg:8.3f}")
    
    print("\nLast 10 frames:")
    print("-" * 40)
    print("Frame    Rg (Å)")
    start_idx = max(0, len(rg_values) - 10)
    for i in range(start_idx, len(rg_values)):
        frame, rg = rg_values[i]
        print(f"{frame:5d}    {rg:8.3f}")
    
    # Summary statistics
    rg_array = np.array([rg for _, rg in rg_values])
    print("\nSummary Statistics:")
    print("-" * 40)
    print(f"Initial Rg: {rg_values[0][1]:.3f} Å")
    print(f"Final Rg: {rg_values[-1][1]:.3f} Å")
    print(f"Mean Rg: {np.mean(rg_array):.3f} Å")
    print(f"Std Dev: {np.std(rg_array):.3f} Å")
    print(f"Min Rg: {np.min(rg_array):.3f} Å (frame {rg_values[np.argmin(rg_array)][0]})")
    print(f"Max Rg: {np.max(rg_array):.3f} Å (frame {rg_values[np.argmax(rg_array)][0]})")
    
    # Check for convergence trend
    print("\nConvergence Analysis:")
    print("-" * 40)
    first_quarter = rg_array[:len(rg_array)//4]
    last_quarter = rg_array[3*len(rg_array)//4:]
    print(f"Mean Rg (first 25% of trajectory): {np.mean(first_quarter):.3f} Å")
    print(f"Mean Rg (last 25% of trajectory): {np.mean(last_quarter):.3f} Å")
    print(f"Change: {np.mean(last_quarter) - np.mean(first_quarter):.3f} Å")
    
    # Save results to file
    output_file = "rg_trajectory_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("Frame,Rg(Angstrom)\n")
        for frame, rg in rg_values:
            f.write(f"{frame},{rg:.6f}\n")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()