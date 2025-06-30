"""Radius of gyration steering potential for structure diffusion."""

import os
import torch
import numpy as np
from typing import Optional, Union
from boltz.model.potentials.potentials import Potential
from boltz.model.potentials.schedules import ParameterSchedule


def extract_rg_from_saxs_data(saxs_file_path: str, q_min: float = 0.01, q_max: float = 0.05) -> float:
    """Extract radius of gyration from experimental SAXS data using Guinier analysis.
    
    Guinier equation: ln(I(q)) = ln(I(0)) - (Rg²/3) * q²
    
    Args:
        saxs_file_path: Path to SAXS data file (q, I, sigma format)
        q_min: Minimum q for Guinier fit (default 0.01 Å⁻¹)
        q_max: Maximum q for Guinier fit (default 0.05 Å⁻¹)
        
    Returns:
        Radius of gyration in Angstroms
    """
    try:
        # Load SAXS data
        data = np.loadtxt(saxs_file_path)
        q = data[:, 0]
        I = data[:, 1]
        sigma = data[:, 2] if data.shape[1] > 2 else np.ones_like(I)
        
        # Select Guinier region
        mask = (q >= q_min) & (q <= q_max) & (I > 0)
        if np.sum(mask) < 5:
            raise ValueError(f"Insufficient data points in Guinier region [{q_min}, {q_max}]")
        
        q_guinier = q[mask]
        I_guinier = I[mask]
        sigma_guinier = sigma[mask]
        
        # Weighted linear fit to ln(I) vs q²
        # ln(I) = ln(I0) - (Rg²/3) * q²
        x = q_guinier**2
        y = np.log(I_guinier)
        weights = 1.0 / (sigma_guinier / I_guinier)  # Propagate error to log space
        
        # Weighted least squares
        w_sum = np.sum(weights)
        w_x = np.sum(weights * x)
        w_y = np.sum(weights * y)
        w_xx = np.sum(weights * x * x)
        w_xy = np.sum(weights * x * y)
        
        # Slope = -Rg²/3
        slope = (w_xy - w_x * w_y / w_sum) / (w_xx - w_x * w_x / w_sum)
        rg_squared = -3.0 * slope
        
        if rg_squared <= 0:
            raise ValueError(f"Invalid Rg² = {rg_squared} from Guinier fit")
        
        rg = np.sqrt(rg_squared)
        
        print(f"SAXS Guinier analysis:")
        print(f"  File: {saxs_file_path}")
        print(f"  q range: {q_min} - {q_max} Å⁻¹")
        print(f"  Data points used: {np.sum(mask)}")
        print(f"  Extracted Rg: {rg:.2f} Å")
        
        return float(rg)
        
    except Exception as e:
        raise ValueError(f"Failed to extract Rg from SAXS data {saxs_file_path}: {e}")


class RadiusOfGyrationPotential:
    """Radius of gyration steering potential for diffusion guidance.
    
    This potential applies a harmonic restraint to guide structures toward
    a target radius of gyration. Much simpler and faster than full SAXS
    calculation while still providing effective size/compactness guidance.
    
    This is a standalone implementation that doesn't inherit from the abstract
    Potential class to avoid framework dependencies.
    """
    
    def __init__(
        self,
        target_rg: Optional[float] = None,
        saxs_file_path: Optional[str] = None,
        force_constant: Union[float, ParameterSchedule] = 1.0,
        q_min: float = 0.01,
        q_max: float = 0.05,
        mass_weighted: bool = True,
        atom_selection: Optional[str] = None,
    ):
        """Initialize radius of gyration potential.
        
        Args:
            target_rg: Target radius of gyration in Angstroms. If None, must provide saxs_file_path
            saxs_file_path: Path to experimental SAXS data for Rg extraction via Guinier analysis
            force_constant: Force constant for harmonic restraint (kcal/mol/Å²)
            q_min: Minimum q for Guinier analysis (Å⁻¹)
            q_max: Maximum q for Guinier analysis (Å⁻¹)  
            mass_weighted: Whether to use mass-weighted Rg calculation
            atom_selection: Optional atom selection (not implemented yet)
        """
        
        # Determine target Rg
        if target_rg is not None:
            self.target_rg = float(target_rg)
            print(f"Using user-specified target Rg: {self.target_rg:.2f} Å")
        elif saxs_file_path is not None:
            if not os.path.exists(saxs_file_path):
                raise FileNotFoundError(f"SAXS file not found: {saxs_file_path}")
            self.target_rg = extract_rg_from_saxs_data(saxs_file_path, q_min, q_max)
        else:
            raise ValueError("Must specify either target_rg or saxs_file_path")
        
        # Store parameters
        self.force_constant = force_constant
        self.mass_weighted = mass_weighted
        self.atom_selection = atom_selection
        self.saxs_file_path = saxs_file_path
        
        # Conversion factor: kcal/mol/Å² to internal units
        self.energy_scale = 1.0  # Adjust as needed for the diffusion model
        
        print(f"Initialized Rg potential with target {self.target_rg:.2f} Å")
        
    def compute_rg(self, coords: torch.Tensor, masses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute radius of gyration.
        
        Args:
            coords: Atomic coordinates [batch, n_atoms, 3]
            masses: Atomic masses [batch, n_atoms] (optional)
            
        Returns:
            Radius of gyration [batch]
        """
        # Get weights (masses or uniform)
        if self.mass_weighted and masses is not None:
            total_mass = masses.sum(dim=-1, keepdim=True)
            weights = masses / total_mass  # [batch, n_atoms]
        else:
            n_atoms = coords.shape[-2]
            weights = torch.ones_like(coords[..., 0]) / n_atoms  # [batch, n_atoms]
        
        # Compute center of mass
        center_of_mass = (coords * weights.unsqueeze(-1)).sum(dim=-2, keepdim=True)  # [batch, 1, 3]
        
        # Compute Rg² = Σ w_i * |r_i - r_com|²
        deviations = coords - center_of_mass  # [batch, n_atoms, 3]
        squared_distances = (deviations ** 2).sum(dim=-1)  # [batch, n_atoms]
        rg_squared = (squared_distances * weights).sum(dim=-1)  # [batch]
        
        # Return Rg (add small epsilon for numerical stability)
        rg = torch.sqrt(rg_squared + 1e-8)
        return rg
        
    def compute_energy(self, coords: torch.Tensor, masses: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Rg restraint energy.
        
        Args:
            coords: Atomic coordinates [batch, n_atoms, 3]
            masses: Atomic masses [batch, n_atoms] (optional)
            
        Returns:
            (energy [batch], current_rg [batch])
        """
        # Compute current Rg
        rg = self.compute_rg(coords, masses)
        
        # Get force constant (could be time-dependent)
        if isinstance(self.force_constant, ParameterSchedule):
            # For now, use a dummy time parameter - this would need to be passed from the diffusion model
            k = self.force_constant(torch.tensor(0.5))
        else:
            k = self.force_constant
        
        # Harmonic restraint: E = 0.5 * k * (Rg - Rg_target)²
        delta_rg = rg - self.target_rg
        energy = 0.5 * k * delta_rg ** 2 * self.energy_scale
        
        return energy, rg
        
    def compute_forces(self, coords: torch.Tensor, masses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute forces from Rg restraint using automatic differentiation.
        
        Args:
            coords: Atomic coordinates [batch, n_atoms, 3] (requires_grad=True)
            masses: Atomic masses [batch, n_atoms] (optional)
            
        Returns:
            Forces [batch, n_atoms, 3]
        """
        coords = coords.requires_grad_(True)
        energy, _ = self.compute_energy(coords, masses)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            energy.sum(),
            coords,
            create_graph=True,  # Allow second derivatives
            retain_graph=True,
        )[0]
        
        # Forces are negative gradients
        forces = -gradients
        return forces
        
    def __call__(self, coords: torch.Tensor, masses: Optional[torch.Tensor] = None) -> dict:
        """Main interface for computing Rg potential.
        
        Args:
            coords: Atomic coordinates [batch, n_atoms, 3]
            masses: Atomic masses [batch, n_atoms] (optional)
            
        Returns:
            Dictionary with energy, forces, and diagnostics
        """
        # Ensure coordinates require gradients
        coords = coords.requires_grad_(True)
        
        # Compute energy and current Rg
        energy, current_rg = self.compute_energy(coords, masses)
        
        # Compute forces
        forces = self.compute_forces(coords, masses)
        
        return {
            "energy": energy,
            "forces": forces,
            "radius_of_gyration": current_rg,
            "target_rg": torch.full_like(current_rg, self.target_rg),
            "rg_deviation": current_rg - self.target_rg,
        }


# Factory function for easy creation from config
def create_rg_potential_from_config(config: dict) -> RadiusOfGyrationPotential:
    """Create Rg potential from configuration dictionary.
    
    Args:
        config: Configuration dictionary with potential parameters
        
    Returns:
        Configured RadiusOfGyrationPotential
        
    Example config:
        rg_guidance:
            # Option 1: Specify target directly
            target_rg: 15.2
            force_constant: 10.0
            
            # Option 2: Extract from SAXS data  
            saxs_file_path: "data/experimental.dat"
            q_min: 0.01
            q_max: 0.05
            force_constant: 10.0
            
            # Optional parameters
            mass_weighted: true
            atom_selection: null
    """
    return RadiusOfGyrationPotential(
        target_rg=config.get("target_rg"),
        saxs_file_path=config.get("saxs_file_path"),
        force_constant=config.get("force_constant", 1.0),
        q_min=config.get("q_min", 0.01),
        q_max=config.get("q_max", 0.05),
        mass_weighted=config.get("mass_weighted", True),
        atom_selection=config.get("atom_selection"),
    )