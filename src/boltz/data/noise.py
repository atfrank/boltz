"""Noise specification parsing and selection functions for targeted denoising."""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from boltz.data.types import Structure


@dataclass
class NoiseSpec:
    """Specification for adding noise to atoms."""
    selection_type: str  # 'chain', 'resid', 'atom', 'distance'
    chain: str
    residue: Optional[Union[int, Tuple[int, int]]] = None  # Single resid or range
    atom_name: Optional[str] = None
    distance: Optional[float] = None
    noise_amount: float = 1.0


def parse_noise_spec(spec_str: str) -> NoiseSpec:
    """Parse a noise specification string.
    
    Parameters
    ----------
    spec_str : str
        Noise specification string like 'chain:A:1.0' or 'distance:A:1:5.0:1.5'
        
    Returns
    -------
    NoiseSpec
        Parsed noise specification
        
    Raises
    ------
    ValueError
        If specification string is invalid
    """
    parts = spec_str.split(':')
    
    if len(parts) < 3:
        raise ValueError(f"Invalid noise specification: {spec_str}")
    
    selection_type = parts[0].lower()
    
    if selection_type == 'chain':
        if len(parts) != 3:
            raise ValueError(f"Chain specification must be 'chain:CHAIN_NAME:NOISE_AMOUNT': {spec_str}")
        return NoiseSpec(
            selection_type='chain',
            chain=parts[1],
            noise_amount=float(parts[2])
        )
    
    elif selection_type == 'resid':
        if len(parts) not in [4, 5]:
            raise ValueError(f"Residue specification must be 'resid:CHAIN:RESID:NOISE' or 'resid:CHAIN:START-END:NOISE': {spec_str}")
        
        # Check if it's a range
        if '-' in parts[2]:
            start, end = map(int, parts[2].split('-'))
            residue_range = (start, end)
        else:
            residue_range = int(parts[2])
        
        return NoiseSpec(
            selection_type='resid',
            chain=parts[1],
            residue=residue_range,
            noise_amount=float(parts[3])
        )
    
    elif selection_type == 'atom':
        if len(parts) != 5:
            raise ValueError(f"Atom specification must be 'atom:CHAIN:RESID:ATOM_NAME:NOISE_AMOUNT': {spec_str}")
        return NoiseSpec(
            selection_type='atom',
            chain=parts[1],
            residue=int(parts[2]),
            atom_name=parts[3],
            noise_amount=float(parts[4])
        )
    
    elif selection_type == 'distance':
        if len(parts) == 5:
            # distance:CHAIN:RESID:DISTANCE:NOISE
            return NoiseSpec(
                selection_type='distance',
                chain=parts[1],
                residue=int(parts[2]),
                distance=float(parts[3]),
                noise_amount=float(parts[4])
            )
        elif len(parts) == 6:
            # distance:CHAIN:RESID:ATOM_NAME:DISTANCE:NOISE
            return NoiseSpec(
                selection_type='distance',
                chain=parts[1],
                residue=int(parts[2]),
                atom_name=parts[3],
                distance=float(parts[4]),
                noise_amount=float(parts[5])
            )
        else:
            raise ValueError(f"Distance specification must be 'distance:CHAIN:RESID:[ATOM_NAME:]DISTANCE:NOISE': {spec_str}")
    
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")


def parse_noise_specifications(spec_list: List[str]) -> List[NoiseSpec]:
    """Parse a list of noise specification strings.
    
    Parameters
    ----------
    spec_list : List[str]
        List of noise specification strings
        
    Returns
    -------
    List[NoiseSpec]
        List of parsed noise specifications
    """
    return [parse_noise_spec(spec) for spec in spec_list]


def find_chain_atoms(structure: Structure, chain_name: str) -> List[int]:
    """Find all atom indices belonging to a specific chain.
    
    Parameters
    ----------
    structure : Structure
        Structure object containing chain and atom information
    chain_name : str
        Name of the chain
        
    Returns
    -------
    List[int]
        List of atom indices
    """
    atom_indices = []
    
    for chain_idx, chain in enumerate(structure.chains):
        if chain['name'] == chain_name:
            # Get atom range for this chain
            start_atom = chain['atom_idx']
            end_atom = start_atom + chain['atom_num']
            atom_indices.extend(range(start_atom, end_atom))
            break
    
    return atom_indices


def find_residue_atoms(structure: Structure, chain_name: str, residue_spec: Union[int, Tuple[int, int]]) -> List[int]:
    """Find all atom indices belonging to specific residue(s).
    
    Parameters
    ----------
    structure : Structure
        Structure object containing residue and atom information
    chain_name : str
        Name of the chain
    residue_spec : Union[int, Tuple[int, int]]
        Residue number or range
        
    Returns
    -------
    List[int]
        List of atom indices
    """
    atom_indices = []
    
    # Find the chain
    chain_idx = None
    for idx, chain in enumerate(structure.chains):
        if chain['name'] == chain_name:
            chain_idx = idx
            break
    
    if chain_idx is None:
        return atom_indices
    
    chain = structure.chains[chain_idx]
    
    # Get residue range for this chain
    res_start = chain['res_idx']
    res_end = res_start + chain['res_num']
    
    # Handle residue range specification
    if isinstance(residue_spec, tuple):
        res_numbers = range(residue_spec[0], residue_spec[1] + 1)
    else:
        res_numbers = [residue_spec]
    
    # Find residues matching the specification
    for res_idx in range(res_start, res_end):
        residue = structure.residues[res_idx]
        if residue['res_idx'] in res_numbers:
            # Get atom range for this residue
            start_atom = residue['atom_idx']
            end_atom = start_atom + residue['atom_num']
            atom_indices.extend(range(start_atom, end_atom))
    
    return atom_indices


def find_specific_atom(structure: Structure, chain_name: str, residue_number: int, atom_name: str) -> List[int]:
    """Find specific atom indices.
    
    Parameters
    ----------
    structure : Structure
        Structure object containing atom information
    chain_name : str
        Name of the chain
    residue_number : int
        Residue number
    atom_name : str
        Atom name (e.g., 'CA', 'N', 'C')
        
    Returns
    -------
    List[int]
        List of atom indices (usually just one)
    """
    # First find the residue
    residue_atoms = find_residue_atoms(structure, chain_name, residue_number)
    
    if not residue_atoms:
        return []
    
    # Then find the specific atom within the residue
    atom_indices = []
    for atom_idx in residue_atoms:
        if atom_idx < len(structure.atoms) and structure.atoms[atom_idx]['name'] == atom_name:
            atom_indices.append(atom_idx)
    
    return atom_indices


def find_atoms_within_distance(structure: Structure, coords: np.ndarray, 
                             reference_atoms: List[int], distance: float) -> List[int]:
    """Find all atoms within a certain distance of reference atoms.
    
    Parameters
    ----------
    structure : Structure
        Structure object
    coords : np.ndarray
        Coordinate array [num_atoms, 3]
    reference_atoms : List[int]
        Indices of reference atoms
    distance : float
        Distance threshold in Angstroms
        
    Returns
    -------
    List[int]
        List of atom indices within distance
    """
    if not reference_atoms:
        return []
    
    atom_indices = []
    num_atoms = len(coords)
    
    # Get reference coordinates
    ref_coords = coords[reference_atoms]
    
    # Check distance for all atoms
    for atom_idx in range(num_atoms):
        atom_coord = coords[atom_idx]
        
        # Calculate minimum distance to any reference atom
        distances = np.linalg.norm(ref_coords - atom_coord, axis=1)
        min_distance = np.min(distances)
        
        if min_distance <= distance:
            atom_indices.append(atom_idx)
    
    return atom_indices


def apply_noise_to_selection(coords: Tensor, noise_specs: List[NoiseSpec], 
                           structure: Structure, device: torch.device,
                           residue_based: bool = False) -> tuple[Tensor, Tensor]:
    """Apply noise to coordinates based on selection specifications.
    
    Parameters
    ----------
    coords : Tensor
        Coordinate tensor [batch, num_atoms, 3]
    noise_specs : List[NoiseSpec]
        List of noise specifications to apply
    structure : Structure
        Structure information for atom selection
    device : torch.device
        Device for tensor operations
    residue_based : bool, optional
        If True, when using distance selection, select entire residues if any atom
        is within distance. Default is False (select individual atoms).
        
    Returns
    -------
    tuple[Tensor, Tensor]
        Coordinate tensor with noise applied, mask of targeted atoms [batch, num_atoms]
    """
    if not noise_specs:
        # Return unchanged coordinates and empty mask
        return coords, torch.zeros((coords.shape[0], coords.shape[1]), dtype=torch.bool, device=device)
        
    # Work with numpy for easier indexing, convert back to tensor
    coords_np = coords.cpu().numpy()
    batch_size, num_atoms, _ = coords_np.shape
    
    # Create noise mask for each specification
    total_noise = np.zeros_like(coords_np)
    # Track which atoms are targeted for noise
    targeted_atoms = np.zeros((batch_size, num_atoms), dtype=bool)
    
    for spec in noise_specs:
        atom_indices = []
        
        if spec.selection_type == 'chain':
            atom_indices = find_chain_atoms(structure, spec.chain)
            print(f"Chain selection '{spec.chain}': found {len(atom_indices)} atoms")
            
        elif spec.selection_type == 'resid':
            atom_indices = find_residue_atoms(structure, spec.chain, spec.residue)
            if isinstance(spec.residue, tuple):
                print(f"Residue range selection chain '{spec.chain}' residues {spec.residue[0]}-{spec.residue[1]}: found {len(atom_indices)} atoms")
            else:
                print(f"Residue selection chain '{spec.chain}' residue {spec.residue}: found {len(atom_indices)} atoms")
            
        elif spec.selection_type == 'atom':
            atom_indices = find_specific_atom(structure, spec.chain, spec.residue, spec.atom_name)
            print(f"Atom selection chain '{spec.chain}' residue {spec.residue} atom '{spec.atom_name}': found {len(atom_indices)} atoms")
            
        elif spec.selection_type == 'distance':
            # First find reference atoms
            if spec.atom_name:
                ref_atoms = find_specific_atom(structure, spec.chain, spec.residue, spec.atom_name)
                print(f"Distance reference: chain '{spec.chain}' residue {spec.residue} atom '{spec.atom_name}': found {len(ref_atoms)} reference atoms")
            else:
                ref_atoms = find_residue_atoms(structure, spec.chain, spec.residue)
                print(f"Distance reference: chain '{spec.chain}' residue {spec.residue}: found {len(ref_atoms)} reference atoms")
                
            # Debug: Show available residues in this chain if no reference atoms found
            if len(ref_atoms) == 0:
                print(f"Debug: Checking available residues in chain '{spec.chain}':")
                chain_residues = []
                for chain in structure.chains:
                    if chain['name'] == spec.chain:
                        res_start = chain['res_idx']
                        res_end = res_start + chain['res_num']
                        for res_idx in range(res_start, res_end):
                            if res_idx < len(structure.residues):
                                residue = structure.residues[res_idx]
                                chain_residues.append(residue['res_idx'])
                        break
                if chain_residues:
                    print(f"  Available residue numbers: {sorted(set(chain_residues))}")
                else:
                    print(f"  No residues found in chain '{spec.chain}'")
            
            # Then find atoms within distance (use first batch for distance calculation)
            if ref_atoms:
                atom_indices = find_atoms_within_distance(
                    structure, coords_np[0], ref_atoms, spec.distance
                )
                print(f"Atoms within {spec.distance}Å of reference: found {len(atom_indices)} atoms")
                
                # If residue-based selection, expand to include all atoms in selected residues
                if residue_based and atom_indices:
                    initial_count = len(atom_indices)
                    # Find all unique residues that have atoms within distance
                    selected_residues = set()
                    
                    # For each atom within distance, find which residue it belongs to
                    for atom_idx in atom_indices:
                        # Find which residue contains this atom
                        for res_idx, residue in enumerate(structure.residues):
                            res_start = residue['atom_idx']
                            res_end = res_start + residue['atom_num']
                            
                            if res_start <= atom_idx < res_end:
                                # Found the residue containing this atom
                                # Get the chain this residue belongs to
                                for chain in structure.chains:
                                    chain_res_start = chain['res_idx']
                                    chain_res_end = chain_res_start + chain['res_num']
                                    
                                    if chain_res_start <= res_idx < chain_res_end:
                                        selected_residues.add((chain['name'], residue['res_idx']))
                                        break
                                break
                    
                    print(f"Residue-based expansion: {len(selected_residues)} residues selected")
                    
                    # Now get all atoms from these residues
                    expanded_indices = []
                    for chain_name, res_idx in selected_residues:
                        # Find the residue and get all its atoms
                        residue = structure.residues[res_idx]
                        res_start = residue['atom_idx']
                        res_end = res_start + residue['atom_num']
                        expanded_indices.extend(range(res_start, res_end))
                    
                    # Use expanded indices instead
                    atom_indices = list(set(expanded_indices))
                    print(f"After residue-based expansion: {initial_count} → {len(atom_indices)} atoms")
            else:
                atom_indices = []
                print(f"No reference atoms found, skipping distance selection")
        
        # Apply noise to selected atoms
        if atom_indices:
            # Ensure indices are within bounds
            valid_indices = [idx for idx in atom_indices if idx < num_atoms]
            out_of_bounds = len(atom_indices) - len(valid_indices)
            if out_of_bounds > 0:
                print(f"Warning: {out_of_bounds} atom indices were out of bounds (total atoms: {num_atoms})")
            
            if valid_indices:
                # Generate noise for selected atoms
                noise = np.random.normal(0, spec.noise_amount, 
                                       (batch_size, len(valid_indices), 3))
                
                # Apply noise to selected atoms
                for batch_idx in range(batch_size):
                    for i, atom_idx in enumerate(valid_indices):
                        total_noise[batch_idx, atom_idx] += noise[batch_idx, i]
                        targeted_atoms[batch_idx, atom_idx] = True
                
                print(f"✓ Applied noise (intensity={spec.noise_amount}) to {len(valid_indices)} atoms for {spec.selection_type} selection")
            else:
                print(f"✗ No valid atoms found for {spec.selection_type} selection")
        else:
            print(f"✗ No atoms selected for {spec.selection_type} selection")
    
    # Apply total noise to coordinates
    coords_with_noise = coords_np + total_noise
    
    # Convert back to tensor
    coords_tensor = torch.tensor(coords_with_noise, dtype=coords.dtype, device=device)
    targeted_mask = torch.tensor(targeted_atoms, dtype=torch.bool, device=device)
    
    return coords_tensor, targeted_mask