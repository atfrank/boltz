"""Wrapper to integrate Rg potential with Boltz diffusion framework."""

import torch
import numpy as np
from typing import Optional, Union

from boltz.model.potentials.potentials import Potential
from boltz.model.potentials.schedules import ParameterSchedule
from boltz.model.potentials.rg_potential import RadiusOfGyrationPotential
from boltz.data.types import RgGuidanceConfig


class RgPotentialWrapper(Potential):
    """Wrapper for Rg potential to integrate with Boltz diffusion framework.
    
    This class adapts the standalone RadiusOfGyrationPotential to work with
    the Boltz potential framework by implementing the required abstract methods.
    """
    
    def __init__(
        self,
        rg_config: RgGuidanceConfig,
        parameters: Optional[dict] = None,
    ):
        """Initialize Rg potential wrapper.
        
        Args:
            rg_config: Rg guidance configuration
            parameters: Framework parameters (guidance_weight, etc.)
        """
        super().__init__(parameters)
        
        # Calculate target Rg from PDB if specified
        target_rg = rg_config.target_rg
        if hasattr(rg_config, 'reference_pdb_path') and rg_config.reference_pdb_path is not None:
            from boltz.model.potentials.pdb_utils import calculate_target_rg_from_pdb
            
            try:
                calculated_target, metadata = calculate_target_rg_from_pdb(
                    pdb_file_path=rg_config.reference_pdb_path,
                    chain_id=getattr(rg_config, 'pdb_chain_id', None),
                    mass_weighted=rg_config.mass_weighted,
                    atom_selection=getattr(rg_config, 'pdb_atom_selection', None)
                )
                target_rg = calculated_target
                print(f"Using calculated target Rg from PDB: {target_rg:.2f} Å")
            except Exception as e:
                print(f"Warning: Failed to calculate Rg from PDB {rg_config.reference_pdb_path}: {e}")
                if rg_config.target_rg is None:
                    raise ValueError("Both PDB calculation and manual target_rg failed. Please provide a valid target_rg.")
                target_rg = rg_config.target_rg
                print(f"Falling back to manual target_rg: {target_rg}")
        
        # Create the underlying Rg potential
        self.rg_potential = RadiusOfGyrationPotential(
            target_rg=target_rg,
            saxs_file_path=rg_config.saxs_file_path,
            force_constant=rg_config.force_constant,
            q_min=rg_config.q_min,
            q_max=rg_config.q_max,
            mass_weighted=rg_config.mass_weighted,
            atom_selection=rg_config.atom_selection,
        )
        
        # Store for framework integration
        self._last_energy = 0.0
        self._last_rg = 0.0
        self._final_rg = None  # Store final Rg for confidence JSON
        
    def compute_args(self, feats, parameters):
        """Compute arguments for the potential.
        
        For Rg potential, we need access to all atoms, so we create a dummy index.
        """
        # Get atom mask
        atom_pad_mask = feats["atom_pad_mask"]
        
        # Create a dummy index pair (required by framework)
        # We'll use the first two atoms as a placeholder
        num_atoms = atom_pad_mask.sum()
        index = torch.tensor([[0], [min(1, num_atoms-1)]], device=atom_pad_mask.device, dtype=torch.long)
        
        # Pass parameters including the full atom mask for Rg calculation
        args = (
            parameters.get("rg_scale", 1.0),
            feats,  # Full feature dict including atom_pad_mask
        )
        
        return index, args, None
        
    def compute_variable(self, coords, index, compute_gradient=False):
        """Compute the Rg-based energy with analytical gradients.
        
        Args:
            coords: Atomic coordinates [batch, n_atoms, 3]
            index: Dummy index (required by framework)
            compute_gradient: Whether to compute gradients
            
        Returns:
            Energy values and optionally gradients
        """
        # Extract dummy pair for framework compatibility
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        
        # Calculate Rg energy using ALL coordinates
        rg_energy_global, current_rg = self._compute_rg_energy_with_value(coords)
        
        # Store for logging
        self._last_energy = float(rg_energy_global.item())
        self._last_rg = float(current_rg.item())
        
        # Always update final Rg (this will be the last computed value)
        self._final_rg = self._last_rg
        
        # Update step counter and log progress
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        # Log every step to see what's happening during diffusion
        rg_error = abs(self._last_rg - self.rg_potential.target_rg)
        print(f"Rg step {self._step_count:3d}: "
              f"Rg={self._last_rg:5.2f}Å "
              f"target={self.rg_potential.target_rg:5.2f}Å "
              f"error={rg_error:5.2f}Å "
              f"energy={self._last_energy:8.3f} "
              f"k={self.rg_potential.force_constant:6.1f}")
        
        # If force constant is 0.0, return zero energy and gradients
        if self.rg_potential.force_constant == 0.0:
            print("           force_constant=0.0: returning zero energy and gradients")
            energy_per_pair = torch.zeros_like(r_ij[..., 0])
            if not compute_gradient:
                return energy_per_pair
            # Return zero gradients with shape matching other potentials: [..., 2, n_pairs, 3]
            # Since we have one dummy pair, n_pairs = 1
            batch_shape = coords.shape[:-2]  # Get batch dimensions
            grad = torch.zeros(*batch_shape, 2, 1, 3, device=coords.device, dtype=coords.dtype)
            return energy_per_pair, grad
        
        # Convert to tensor with same shape as r_ij distances  
        energy_per_pair = rg_energy_global.expand_as(r_ij[..., 0])
        
        if not compute_gradient:
            return energy_per_pair
        
        # Compute analytical gradients directly
        grad_coords = self._compute_analytical_rg_gradients(coords, current_rg)
        
        # Extract gradients for the dummy pair (framework expects this format)
        grad_i = grad_coords[..., index[0], :]  # Shape: [..., 1, 3]
        grad_j = grad_coords[..., index[1], :]  # Shape: [..., 1, 3]
        grad = torch.stack((grad_i, grad_j), dim=1)  # Shape: [..., 2, 1, 3]
        
        # Log gradient magnitude
        max_grad = grad_coords.abs().max().item()
        print(f"           gradients: max={max_grad:.6f} "
              f"grad_norm={grad_coords.norm().item():.6f}")
        
        # Adaptive force constant - increase if far from target, but cap it
        if rg_error > 2.0 and self._step_count > 10 and self.rg_potential.force_constant < 500.0:
            old_k = self.rg_potential.force_constant
            self.rg_potential.force_constant *= 1.02  # Smaller increase
            print(f"           adaptive: k {old_k:.1f} → {self.rg_potential.force_constant:.1f}")
        
        # Stability check - reduce force constant if gradients are too large
        if max_grad > 1000.0:
            old_k = self.rg_potential.force_constant
            self.rg_potential.force_constant *= 0.8
            print(f"           stability: k {old_k:.1f} → {self.rg_potential.force_constant:.1f} (large gradients)")
        
        return energy_per_pair, grad
    
    def _compute_rg_energy_with_value(self, coords):
        """Compute Rg energy and current Rg value."""
        # Handle batch dimension - extract actual coordinates
        if coords.ndim == 3:
            coord_batch = coords[0]  # [n_atoms, 3]
        elif coords.ndim == 4:
            coord_batch = coords[0, 0]  # [n_atoms, 3]
        else:
            coord_batch = coords
        
        # Compute center of mass
        center_of_mass = coord_batch.mean(dim=0)
        
        # Compute deviations from center of mass
        deviations = coord_batch - center_of_mass
        
        # Compute Rg² = mean(|r_i - r_com|²)
        rg_squared = torch.mean(torch.sum(deviations**2, dim=1))
        current_rg = torch.sqrt(rg_squared)
        
        # Compute energy: E = 0.5 * k * (Rg - Rg_target)²
        delta_rg = current_rg - self.rg_potential.target_rg
        energy = 0.5 * self.rg_potential.force_constant * delta_rg ** 2
        
        return energy, current_rg
    
    def _compute_analytical_rg_gradients(self, coords, current_rg):
        """Compute analytical gradients of Rg energy with respect to coordinates.
        
        The gradient of Rg² with respect to atom i is:
        ∂(Rg²)/∂r_i = (2/N) * (r_i - r_com)
        
        The gradient of Rg with respect to atom i is:
        ∂Rg/∂r_i = (1/Rg) * (1/N) * (r_i - r_com)
        
        The gradient of energy E = 0.5 * k * (Rg - Rg_target)² is:
        ∂E/∂r_i = k * (Rg - Rg_target) * ∂Rg/∂r_i
        """
        # Handle batch dimension
        if coords.ndim == 3:
            coord_batch = coords[0]  # [n_atoms, 3]
        elif coords.ndim == 4:
            coord_batch = coords[0, 0]  # [n_atoms, 3]
        else:
            coord_batch = coords
            
        n_atoms = coord_batch.shape[0]
        
        # Compute center of mass
        center_of_mass = coord_batch.mean(dim=0)
        
        # Deviations from center of mass
        deviations = coord_batch - center_of_mass  # [n_atoms, 3]
        
        # Gradient of Rg with respect to each atom position
        # ∂Rg/∂r_i = (1/Rg) * (1/N) * (r_i - r_com)
        if current_rg > 1e-8:  # Avoid division by zero
            grad_rg = deviations / (n_atoms * current_rg)  # [n_atoms, 3]
        else:
            grad_rg = torch.zeros_like(deviations)
        
        # Gradient of energy with respect to each atom position
        # ∂E/∂r_i = k * (Rg - Rg_target) * ∂Rg/∂r_i
        delta_rg = current_rg - self.rg_potential.target_rg
        grad_energy = self.rg_potential.force_constant * delta_rg * grad_rg  # [n_atoms, 3]
        
        # Expand to match input coords shape
        if coords.ndim == 3:
            grad_coords = torch.zeros_like(coords)
            grad_coords[0] = grad_energy
        elif coords.ndim == 4:
            grad_coords = torch.zeros_like(coords)
            grad_coords[0, 0] = grad_energy
        else:
            grad_coords = grad_energy
            
        return grad_coords
        
    def compute_function(self, variable, *args, compute_derivative=False):
        """Compute the final potential function.
        
        Args:
            variable: The computed variable (energy)
            args: Additional arguments
            compute_derivative: Whether to compute derivative
            
        Returns:
            Scaled energy and optionally derivative
        """
        rg_scale, feats = args
        
        # Apply scaling
        scaled_energy = variable * rg_scale
        
        if not compute_derivative:
            return scaled_energy
            
        # Derivative is just the scaling factor
        derivative = torch.ones_like(variable) * rg_scale
        
        return scaled_energy, derivative
    
    def get_final_rg(self):
        """Get the final computed Rg value."""
        return self._final_rg
    
    def get_target_rg(self):
        """Get the target Rg value."""
        return self.rg_potential.target_rg