"""Utilities for processing PDB files and extracting structural properties."""

import numpy as np
from typing import Optional, Union, Tuple
import os

def parse_pdb_coordinates(pdb_file_path: str, chain_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """Parse coordinates from a PDB file.
    
    Args:
        pdb_file_path: Path to PDB file
        chain_id: Specific chain to extract (if None, extracts all chains)
        
    Returns:
        coords: Atom coordinates [N, 3]
        atom_mask: Valid atom mask [N]
        atom_names: List of atom names for debugging
    """
    coords = []
    atom_names = []
    
    if not os.path.exists(pdb_file_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
    
    with open(pdb_file_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # Parse PDB format
                atom_name = line[12:16].strip()
                chain = line[21:22].strip()
                
                # Skip if chain_id is specified and doesn't match
                if chain_id is not None and chain != chain_id:
                    continue
                
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    coords.append([x, y, z])
                    atom_names.append(atom_name)
                except ValueError:
                    # Skip lines with invalid coordinates
                    continue
    
    if not coords:
        raise ValueError(f"No valid coordinates found in PDB file: {pdb_file_path}")
    
    coords = np.array(coords)
    atom_mask = np.ones(len(coords), dtype=bool)
    
    return coords, atom_mask, atom_names


def calculate_radius_of_gyration(
    coords: np.ndarray, 
    atom_mask: Optional[np.ndarray] = None,
    mass_weighted: bool = False,
    atom_masses: Optional[np.ndarray] = None
) -> float:
    """Calculate radius of gyration from coordinates.
    
    Args:
        coords: Atom coordinates [N, 3]
        atom_mask: Valid atom mask [N] (optional)
        mass_weighted: Whether to use mass weighting
        atom_masses: Atom masses [N] (required if mass_weighted=True)
        
    Returns:
        rg: Radius of gyration in Angstroms
    """
    if atom_mask is not None:
        valid_coords = coords[atom_mask]
        if mass_weighted and atom_masses is not None:
            valid_masses = atom_masses[atom_mask]
        else:
            valid_masses = None
    else:
        valid_coords = coords
        valid_masses = atom_masses
    
    if len(valid_coords) == 0:
        raise ValueError("No valid coordinates for Rg calculation")
    
    if mass_weighted and valid_masses is not None:
        # Mass-weighted center of mass
        total_mass = np.sum(valid_masses)
        if total_mass == 0:
            raise ValueError("Total mass is zero for mass-weighted Rg calculation")
        
        center_of_mass = np.sum(valid_coords * valid_masses[:, np.newaxis], axis=0) / total_mass
        
        # Mass-weighted Rg
        distances_squared = np.sum((valid_coords - center_of_mass) ** 2, axis=1)
        rg_squared = np.sum(valid_masses * distances_squared) / total_mass
    else:
        # Geometric center
        center = np.mean(valid_coords, axis=0)
        
        # Standard Rg
        distances_squared = np.sum((valid_coords - center) ** 2, axis=1)
        rg_squared = np.mean(distances_squared)
    
    return np.sqrt(rg_squared)


def get_default_atom_masses(atom_names: list) -> np.ndarray:
    """Get default atomic masses for common atom types.
    
    Args:
        atom_names: List of atom names from PDB
        
    Returns:
        masses: Array of atomic masses
    """
    # Standard atomic masses (amu)
    mass_table = {
        'H': 1.008,
        'C': 12.011,
        'N': 14.007,
        'O': 15.999,
        'P': 30.974,
        'S': 32.066,
        'F': 18.998,
        'CL': 35.453,
        'BR': 79.904,
        'I': 126.904,
        'MG': 24.305,
        'CA': 40.078,
        'FE': 55.845,
        'ZN': 65.38,
        'MN': 54.938,
    }
    
    masses = []
    for atom_name in atom_names:
        # Extract element symbol (first 1-2 characters)
        element = atom_name[0].upper()
        if len(atom_name) > 1 and atom_name[1].isalpha():
            element += atom_name[1].upper()
        
        # Get mass from table, default to carbon if unknown
        mass = mass_table.get(element, 12.011)
        masses.append(mass)
    
    return np.array(masses)


def calculate_target_rg_from_pdb(
    pdb_file_path: str,
    chain_id: Optional[str] = None,
    mass_weighted: bool = True,
    atom_selection: Optional[str] = None
) -> Tuple[float, dict]:
    """Calculate target Rg from a reference PDB structure.
    
    Args:
        pdb_file_path: Path to reference PDB file
        chain_id: Specific chain to analyze (if None, uses all chains)
        mass_weighted: Whether to use mass weighting
        atom_selection: Atom selection criteria ('all', 'backbone', 'heavy'). 
                       Default 'all' is recommended for Rg guidance as it should 
                       include all atoms for proper structure comparison.
        
    Returns:
        target_rg: Calculated radius of gyration
        metadata: Dictionary with calculation details
    """
    # Parse PDB coordinates
    coords, atom_mask, atom_names = parse_pdb_coordinates(pdb_file_path, chain_id)
    
    # Apply atom selection if specified
    if atom_selection == 'backbone':
        # Select backbone atoms - handles both protein and nucleic acid
        # Protein backbone: N, CA, C, O
        # Nucleic acid backbone: P, O5', C5', C4', C3', O3'
        protein_backbone = ['N', 'CA', 'C', 'O']
        nucleic_backbone = ['P', "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'", "C2'"]
        backbone_atoms = protein_backbone + nucleic_backbone
        backbone_mask = np.array([name.strip() in backbone_atoms for name in atom_names])
        atom_mask = atom_mask & backbone_mask
    elif atom_selection == 'heavy':
        # Select heavy atoms (non-hydrogen)
        heavy_mask = np.array([not name.strip().startswith('H') for name in atom_names])
        atom_mask = atom_mask & heavy_mask
    elif atom_selection == 'all' or atom_selection is None:
        # Use all atoms (default)
        pass
    else:
        raise ValueError(f"Unknown atom selection: {atom_selection}")
    
    # Get atom masses if needed
    atom_masses = None
    if mass_weighted:
        atom_masses = get_default_atom_masses(atom_names)
    
    # Calculate Rg
    target_rg = calculate_radius_of_gyration(
        coords, atom_mask, mass_weighted, atom_masses
    )
    
    # Prepare metadata
    metadata = {
        'pdb_file': pdb_file_path,
        'chain_id': chain_id,
        'total_atoms': len(coords),
        'selected_atoms': np.sum(atom_mask),
        'mass_weighted': mass_weighted,
        'atom_selection': atom_selection or 'all',
        'target_rg': float(target_rg),
    }
    
    print(f"PDB Target Rg Calculation:")
    print(f"  File: {os.path.basename(pdb_file_path)}")
    print(f"  Chain: {chain_id or 'all'}")
    print(f"  Selection: {atom_selection or 'all'} ({metadata['selected_atoms']}/{metadata['total_atoms']} atoms)")
    print(f"  Mass-weighted: {mass_weighted}")
    print(f"  Target Rg: {target_rg:.2f} Å")
    
    return target_rg, metadata


def validate_pdb_file(pdb_file_path: str) -> bool:
    """Validate that a PDB file exists and contains coordinate data.
    
    Args:
        pdb_file_path: Path to PDB file
        
    Returns:
        is_valid: True if file is valid, False otherwise
    """
    try:
        coords, _, _ = parse_pdb_coordinates(pdb_file_path)
        return len(coords) > 0
    except (FileNotFoundError, ValueError):
        return False


def list_chains_in_pdb(pdb_file_path: str) -> list:
    """List all chain IDs present in a PDB file.
    
    Args:
        pdb_file_path: Path to PDB file
        
    Returns:
        chain_ids: List of unique chain IDs
    """
    chain_ids = set()
    
    if not os.path.exists(pdb_file_path):
        return []
    
    with open(pdb_file_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                chain = line[21:22].strip()
                if chain:
                    chain_ids.add(chain)
    
    return sorted(list(chain_ids))