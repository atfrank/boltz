import string
from collections.abc import Iterator
from pathlib import Path
from typing import Optional
import numpy as np


def generate_tags() -> Iterator[str]:
    """Generate chain tags.

    Yields
    ------
    str
        The next chain tag

    """
    for i in range(1, 4):
        for j in range(len(string.ascii_uppercase) ** i):
            tag = ""
            for k in range(i):
                tag += string.ascii_uppercase[
                    j
                    // (len(string.ascii_uppercase) ** k)
                    % len(string.ascii_uppercase)
                ]
            yield tag


def read_coords_from_file(file_path: str, expected_atoms: int) -> Optional[np.ndarray]:
    """Read coordinates from PDB or mmCIF file.
    
    Parameters
    ----------
    file_path : str
        Path to PDB or mmCIF file
    expected_atoms : int
        Expected number of atoms
        
    Returns
    -------
    Optional[np.ndarray]
        Coordinate array [num_atoms, 3] or None if failed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    coords = []
    
    try:
        if file_path.suffix.lower() in ['.pdb']:
            # Parse PDB file
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coords.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
        
        elif file_path.suffix.lower() in ['.cif', '.mmcif']:
            # Basic mmCIF parsing - this is simplified
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            in_atom_site = False
            coord_indices = None
            
            for line in lines:
                if line.startswith('_atom_site.'):
                    in_atom_site = True
                    # Find coordinate column indices
                    if 'Cartn_x' in line:
                        coord_indices = []
                        for i, header_line in enumerate(lines):
                            if header_line.startswith('_atom_site.Cartn_x'):
                                coord_indices.append(0)
                            elif header_line.startswith('_atom_site.Cartn_y'):
                                coord_indices.append(1) 
                            elif header_line.startswith('_atom_site.Cartn_z'):
                                coord_indices.append(2)
                            if len(coord_indices) == 3:
                                break
                
                elif in_atom_site and not line.startswith('_') and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) > 10:  # Ensure we have enough columns
                        try:
                            # Try to extract coordinates (typically around columns 10-12)
                            x = float(parts[10])
                            y = float(parts[11])
                            z = float(parts[12])
                            coords.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
        
        if not coords:
            return None
            
        coords_array = np.array(coords)
        
        # Pad or truncate to expected size
        if len(coords_array) < expected_atoms:
            # Pad with zeros
            padding = np.zeros((expected_atoms - len(coords_array), 3))
            coords_array = np.vstack([coords_array, padding])
        elif len(coords_array) > expected_atoms:
            # Truncate
            coords_array = coords_array[:expected_atoms]
            
        return coords_array
        
    except Exception as e:
        print(f"Error reading coordinates from {file_path}: {e}")
        return None
