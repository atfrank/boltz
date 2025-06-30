"""Robust Rg Potential Wrapper for Boltz Integration.

This wrapper integrates the robust Rg potential with the Boltz diffusion framework,
providing multiple safeguards against the "extreme atom cheating" problem.
"""

import torch
import numpy as np
from typing import Optional, Union

from boltz.model.potentials.potentials import Potential
from boltz.model.potentials.schedules import ParameterSchedule
from boltz.model.potentials.robust_rg_potential import RobustRgPotential
from boltz.data.types import RgGuidanceConfig


class RobustRgPotentialWrapper(Potential):
    """Robust wrapper for Rg potential with outlier protection and displacement constraints."""
    
    def __init__(
        self,
        rg_config: RgGuidanceConfig,
        parameters: Optional[dict] = None,
        # Robustness parameters
        max_displacement_per_step: float = 2.0,
        outlier_threshold: float = 3.0,
        rg_calculation_method: str = "robust",
        gradient_capping: float = 10.0,
        force_ramping: bool = True,
        min_force_constant: float = 1.0,
        ramping_steps: int = 50,
    ):
        """Initialize robust Rg potential wrapper.
        
        Args:
            rg_config: Rg guidance configuration
            parameters: Framework parameters (guidance_weight, etc.)
            max_displacement_per_step: Maximum allowed atom displacement per step (Å)
            outlier_threshold: Threshold for outlier detection (standard deviations)
            rg_calculation_method: "standard", "robust", or "trimmed"
            gradient_capping: Maximum gradient magnitude per atom (kcal/mol/Å)
            force_ramping: Whether to gradually increase force constant
            min_force_constant: Starting force constant for ramping
            ramping_steps: Steps to reach full force constant
        """
        super().__init__(parameters)
        
        # Calculate target Rg from PDB if specified
        target_rg = rg_config.target_rg
        if rg_config.reference_pdb_path is not None:
            from boltz.model.potentials.pdb_utils import calculate_target_rg_from_pdb
            
            try:
                calculated_target, metadata = calculate_target_rg_from_pdb(
                    pdb_file_path=rg_config.reference_pdb_path,
                    chain_id=rg_config.pdb_chain_id,
                    mass_weighted=rg_config.mass_weighted,
                    atom_selection=rg_config.pdb_atom_selection
                )
                target_rg = calculated_target
                print(f"Using calculated target Rg from PDB: {target_rg:.2f} Å")
            except Exception as e:
                print(f"Warning: Failed to calculate Rg from PDB {rg_config.reference_pdb_path}: {e}")
                if rg_config.target_rg is None:
                    raise ValueError("Both PDB calculation and manual target_rg failed. Please provide a valid target_rg.")
                target_rg = rg_config.target_rg
                print(f"Falling back to manual target_rg: {target_rg}")
        
        # Create the robust Rg potential
        self.rg_potential = RobustRgPotential(
            target_rg=target_rg,
            saxs_file_path=rg_config.saxs_file_path,
            force_constant=rg_config.force_constant,
            q_min=getattr(rg_config, 'q_min', 0.01),
            q_max=getattr(rg_config, 'q_max', 0.05),
            mass_weighted=getattr(rg_config, 'mass_weighted', True),
            atom_selection=getattr(rg_config, 'atom_selection', 'all'),
            # Robustness parameters
            max_displacement_per_step=max_displacement_per_step,
            outlier_threshold=outlier_threshold,
            rg_calculation_method=rg_calculation_method,
            gradient_capping=gradient_capping,
            force_ramping=force_ramping,
            min_force_constant=min_force_constant,
            ramping_steps=ramping_steps,
        )
        
        # Store for framework integration
        self._last_info = {}
        self._final_rg = None
        
        print(f"Initialized RobustRgPotentialWrapper with {rg_calculation_method} Rg calculation")
        print(f"Force constant: {rg_config.force_constant} → ramping: {force_ramping}")
        
    def compute_args(self, feats, parameters):
        """Compute arguments for the potential."""
        # Get atom mask
        atom_pad_mask = feats["atom_pad_mask"]
        
        # Create a dummy index pair (required by framework)
        num_atoms = atom_pad_mask.sum()
        index = torch.tensor([[0], [min(1, num_atoms-1)]], device=atom_pad_mask.device, dtype=torch.long)
        
        # Pass parameters including the full atom mask for Rg calculation
        args = (
            parameters.get("rg_scale", 1.0),
            feats,  # Full feature dict including atom_pad_mask
        )
        
        return index, args, None
        
    def compute_variable(self, coords, index, compute_gradient=False):
        """Compute the robust Rg-based energy with protected gradients."""
        # Extract dummy pair for framework compatibility
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        
        # Check for zero force constant
        if self.rg_potential.force_constant == 0.0:
            print("           force_constant=0.0: returning zero energy and gradients")
            energy_per_pair = torch.zeros_like(r_ij[..., 0])
            if not compute_gradient:
                return energy_per_pair
            # Return zero gradients
            batch_shape = coords.shape[:-2]
            grad = torch.zeros(*batch_shape, 2, 1, 3, device=coords.device, dtype=coords.dtype)
            return energy_per_pair, grad
        
        # Compute robust Rg energy and gradients
        if compute_gradient:
            rg_energy_global, grad_coords, info = self.rg_potential.compute_energy_and_gradients(coords)
        else:
            rg_energy_global, info = self.rg_potential.compute_energy(coords)
            grad_coords = None
        
        # Store info for logging and final reporting
        self._last_info = info
        self._final_rg = info["rg"]
        
        # Log progress with robustness information
        step = info.get("step", 0)
        rg = info["rg"]
        target_rg = info["target_rg"]
        rg_error = info["rg_error"]
        effective_k = info["effective_k"]
        energy = info["energy"]
        
        log_msg = (f"Robust Rg step {step:3d}: "
                  f"Rg={rg:5.2f}Å "
                  f"target={target_rg:5.2f}Å "
                  f"error={rg_error:5.2f}Å "
                  f"energy={energy:8.3f} "
                  f"k_eff={effective_k:6.1f}")
        
        # Add robustness info
        if "outliers_detected" in info:
            log_msg += f" outliers={info['outliers_detected']}"
        if "trimmed_atoms" in info:
            log_msg += f" trimmed={info['trimmed_atoms']}"
        if "displacement_violations" in info and info["displacement_violations"] > 0:
            log_msg += f" disp_viol={info['displacement_violations']}"
        
        print(log_msg)
        
        # Additional detailed logging for gradients
        if compute_gradient and grad_coords is not None:
            max_grad = info.get("max_gradient", 0.0)
            grad_norm = info.get("gradient_norm", 0.0)
            print(f"           gradients: max={max_grad:.6f} "
                  f"norm={grad_norm:.6f} "
                  f"valid_atoms={info.get('valid_atoms', 'N/A')}")
        
        # Convert to tensor with same shape as r_ij distances  
        energy_per_pair = rg_energy_global.expand_as(r_ij[..., 0])
        
        if not compute_gradient:
            return energy_per_pair
        
        # Extract gradients for the dummy pair (framework expects this format)
        if grad_coords is not None:
            grad_i = grad_coords[..., index[0], :]  # Shape: [..., 1, 3]
            grad_j = grad_coords[..., index[1], :]  # Shape: [..., 1, 3]
            grad = torch.stack((grad_i, grad_j), dim=1)  # Shape: [..., 2, 1, 3]
        else:
            # Fallback zero gradients
            batch_shape = coords.shape[:-2]
            grad = torch.zeros(*batch_shape, 2, 1, 3, device=coords.device, dtype=coords.dtype)
        
        return energy_per_pair, grad
        
    def compute_function(self, variable, *args, compute_derivative=False):
        """Compute the final potential function with scaling."""
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
        
    def get_robustness_stats(self):
        """Get statistics about robustness features."""
        return {
            "method": self.rg_potential.rg_calculation_method,
            "total_steps": self.rg_potential.step_count,
            "displacement_violations": self.rg_potential.displacement_violations,
            "max_displacement_per_step": self.rg_potential.max_displacement_per_step,
            "gradient_capping": self.rg_potential.gradient_capping,
            "force_ramping": self.rg_potential.force_ramping,
            "current_force_constant": self.rg_potential.get_effective_force_constant(),
            "original_force_constant": self.rg_potential.original_force_constant,
        }