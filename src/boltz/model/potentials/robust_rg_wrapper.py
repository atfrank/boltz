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
        
        # Simple step tracking for logging control
        self._diffusion_step = 0
        self._last_logged_step = -1
        self._guidance_call_count = 0
        
        print(f"Initialized RobustRgPotentialWrapper with {rg_calculation_method} Rg calculation")
        print(f"Force constant: {rg_config.force_constant} → ramping: {force_ramping}")
        
        # Raw guidance tracking
        self._raw_guidance_enabled = False
    
    def set_step_info(self, diffusion_step: int, guidance_step: int = 0):
        """Set step information for proper logging control."""
        if diffusion_step != self._diffusion_step:
            # New diffusion step - reset guidance call count
            self._diffusion_step = diffusion_step
            self._guidance_call_count = 0
        
        # Track which guidance step we're on
        self._guidance_call_count = guidance_step
    
    def log_raw_rg(self, step_idx: int, coords: torch.Tensor, feats: dict):
        """Log Rg on raw denoised coordinates before gradient descent optimization."""
        try:
            # Compute Rg on raw coordinates
            current_rg, com, rg_info = self.rg_potential.compute_robust_rg(coords, masses=None)
            
            print(f"Raw denoised Rg step {step_idx:3d}: "
                  f"Rg={current_rg.item():5.2f}Å "
                  f"target={self.rg_potential.target_rg:5.2f}Å "
                  f"error={abs(current_rg.item() - self.rg_potential.target_rg):5.2f}Å "
                  f"(before optimization)")
        except Exception as e:
            print(f"Warning: Failed to compute raw Rg for step {step_idx}: {e}")
    
    def apply_raw_guidance(self, coords: torch.Tensor, feats: dict, parameters: dict, sigma: float, step_idx: int) -> torch.Tensor:
        """Apply Rg guidance to raw noisy coordinates before neural network denoising.
        
        Args:
            coords: Raw noisy coordinates [batch, num_atoms, 3]
            feats: Feature dictionary
            parameters: Potential parameters including raw_guidance_weight
            sigma: Current noise level
            step_idx: Current diffusion step
            
        Returns:
            Modified coordinates with Rg guidance applied
        """
        try:
            raw_guidance_weight = parameters.get("raw_guidance_weight", 0.0)
            if raw_guidance_weight <= 0:
                return coords
                
            # Note: Raw coordinate guidance is experimental due to gradient flow challenges
            # in the diffusion framework. Current implementation works for first few steps.
                
            # Enable tracking for logging
            if not self._raw_guidance_enabled:
                self._raw_guidance_enabled = True
                print(f"Raw coordinate Rg guidance activated with weight {raw_guidance_weight}")
            
            # First compute Rg for logging (use robust version)
            current_rg_log, com, rg_info = self.rg_potential.compute_robust_rg(coords, masses=None)
            
            # Compute Rg error for condition check
            target_rg = self.rg_potential.target_rg
            rg_error_log = current_rg_log - target_rg
            
            # Adaptive guidance strength based on noise level
            # Higher noise = weaker guidance to avoid overwhelming the diffusion process
            # Log noise levels for first step only
            if step_idx == 0:
                print(f"Raw guidance step {step_idx}: sigma={sigma:.1f}")
            
            # Adjust max_sigma based on observed values - it appears to be much higher than expected
            max_sigma = 800.0  # Further increased based on observed sigma values up to ~671
            noise_factor = max(0.0, 1.0 - (sigma / max_sigma))  # 0 at max noise, 1 at no noise, clamp to [0,1]
            adaptive_strength = raw_guidance_weight * noise_factor * 0.01  # Very small for stability with finite differences
            
            # Apply guidance if we have reasonable noise factor
            # Don't require small error since initial structures can be very wrong
            if noise_factor > 0.05:  # Apply guidance when noise is not too high
                # Robust gradient computation for raw coordinate guidance
                try:
                    # Disable autocast to prevent mixed precision issues that break gradients
                    with torch.cuda.amp.autocast(enabled=False):
                        # Use a no_grad context first to avoid interfering with existing gradients
                        with torch.no_grad():
                            # Create a fresh copy for gradient computation, ensuring float32 precision
                            coords_for_grad = coords.detach().clone()
                            
                            # Force float32 precision to avoid mixed precision issues
                            if coords_for_grad.dtype != torch.float32:
                                coords_for_grad = coords_for_grad.float()
                        
                        # Now enable gradients on the clean copy
                        coords_for_grad.requires_grad_(True)
                        
                        # Ensure gradient computation context
                        with torch.enable_grad():
                            # Use gradient-friendly Rg computation
                            rg_for_grad = self.rg_potential.compute_gradient_friendly_rg(coords_for_grad, masses=None)
                        
                        # Check if autograd gradients are available, otherwise use finite differences
                        if rg_for_grad.requires_grad and rg_for_grad.grad_fn is not None:
                            # Try autograd first (faster when it works)
                            rg_error_for_grad = rg_for_grad - target_rg
                            try:
                                rg_grad = torch.autograd.grad(
                                    outputs=rg_for_grad,
                                    inputs=coords_for_grad,
                                    create_graph=False,
                                    retain_graph=False,
                                    allow_unused=False
                                )[0]
                                if step_idx == 0:
                                    print(f"Raw guidance step {step_idx}: Using autograd gradients")
                            except RuntimeError:
                                # Autograd failed, will fall back to finite differences
                                rg_grad = None
                        else:
                            rg_grad = None
                        
                        # Fallback to finite differences if autograd failed
                        if rg_grad is None:
                            if step_idx == 0:
                                print(f"Raw guidance step {step_idx}: Using finite difference gradients")
                            
                            # Simplified finite differences approach
                            eps = 1e-3
                            
                            # Compute base Rg value
                            rg_base = self.rg_potential.compute_gradient_friendly_rg(coords_for_grad, masses=None)
                            
                            # Initialize gradient tensor
                            rg_grad = torch.zeros_like(coords_for_grad)
                            
                            # Compute gradients for first sample only (handle batch dimension properly)
                            if coords_for_grad.ndim == 3:
                                sample_coords = coords_for_grad[0]  # [num_atoms, 3]
                                sample_grad = torch.zeros_like(sample_coords)
                                
                                # Compute gradients atom by atom to avoid memory issues
                                for i in range(sample_coords.shape[0]):  # atoms
                                    for j in range(3):  # x, y, z
                                        # Create perturbed coordinates
                                        coords_pert = coords_for_grad.clone()
                                        coords_pert[0, i, j] += eps
                                        
                                        # Compute perturbed Rg
                                        rg_pert = self.rg_potential.compute_gradient_friendly_rg(coords_pert, masses=None)
                                        
                                        # Finite difference
                                        sample_grad[i, j] = (rg_pert - rg_base) / eps
                                
                                rg_grad[0] = sample_grad
                            else:
                                # Handle 2D case
                                for i in range(coords_for_grad.shape[0]):
                                    for j in range(3):
                                        coords_pert = coords_for_grad.clone()
                                        coords_pert[i, j] += eps
                                        rg_pert = self.rg_potential.compute_gradient_friendly_rg(coords_pert, masses=None)
                                        rg_grad[i, j] = (rg_pert - rg_base) / eps
                            
                            rg_error_for_grad = rg_base - target_rg
                        
                        # Verify gradient was computed successfully
                        if rg_grad is None or torch.isnan(rg_grad).any() or torch.isinf(rg_grad).any():
                            if step_idx % 10 == 0:
                                print(f"Raw guidance step {step_idx}: Invalid gradients computed, skipping")
                            return coords
                        
                except Exception as e:
                    if step_idx % 10 == 0:  # Log occasionally to avoid spam
                        print(f"Raw guidance step {step_idx}: Gradient computation failed: {e}")
                    return coords
                
                # Apply guidance force: F = -k * (Rg - Rg_target) * dRg/dr
                guidance_force = -adaptive_strength * rg_error_for_grad * rg_grad
                
                # Ensure guidance force is in the same dtype and device as original coords
                if guidance_force.dtype != coords.dtype:
                    guidance_force = guidance_force.to(coords.dtype)
                if guidance_force.device != coords.device:
                    guidance_force = guidance_force.to(coords.device)
                
                # Clip gradients to prevent instability
                max_force = 0.5  # Reduced maximum force per atom for stability
                force_norm = torch.norm(guidance_force, dim=-1, keepdim=True)
                force_clip_mask = force_norm > max_force
                if force_clip_mask.any():
                    guidance_force[force_clip_mask.squeeze(-1)] *= (max_force / force_norm[force_clip_mask])
                
                # Reshape guidance force to match original coordinates shape
                if coords.ndim == 3:
                    # guidance_force is [num_atoms, 3], coords is [batch_size, num_atoms, 3]
                    batch_size = coords.shape[0]
                    guidance_force_batch = torch.zeros_like(coords)
                    guidance_force_batch[0] = guidance_force  # Apply to first sample only
                    guidance_force = guidance_force_batch
                elif coords.ndim == 4:
                    # guidance_force is [num_atoms, 3], coords is [batch1, batch2, num_atoms, 3]
                    batch1, batch2 = coords.shape[0], coords.shape[1]
                    guidance_force_batch = torch.zeros_like(coords)
                    guidance_force_batch[0, 0] = guidance_force  # Apply to first sample only
                    guidance_force = guidance_force_batch
                
                # Apply the guidance
                coords_guided = coords + guidance_force
                
                # Log raw guidance application
                if step_idx == 0 or step_idx % 5 == 0:  # Log first step and every 5 steps
                    print(f"Raw guidance step {step_idx}: "
                          f"sigma={sigma:.1f} "
                          f"Rg={current_rg_log.item():.1f}Å "
                          f"target={target_rg:.1f}Å "
                          f"error={rg_error_log.item():.1f}Å "
                          f"noise_factor={noise_factor:.3f} "
                          f"strength={adaptive_strength:.6f} "
                          f"max_force={guidance_force.abs().max().item():.6f}")
                
                return coords_guided
            else:
                # Log why we're skipping raw guidance (first few steps only)
                if step_idx <= 2:
                    print(f"Raw guidance step {step_idx}: skipping - noise_factor={noise_factor:.3f} (too high noise)")
                return coords
                
        except Exception as e:
            print(f"Warning: Raw guidance failed at step {step_idx}: {e}")
            return coords
        
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
        
        # Only log progress once per diffusion step at the end of optimization
        # Log on first guidance call of each step to show post-optimization result
        should_log = (self._diffusion_step != self._last_logged_step and 
                     self._guidance_call_count == 0)
        
        if should_log:
            # Mark this step as logged
            self._last_logged_step = self._diffusion_step
            
            # Log progress with robustness information
            rg = info["rg"]
            target_rg = info["target_rg"]
            rg_error = info["rg_error"]
            effective_k = info["effective_k"]
            energy = info["energy"]
            
            log_msg = (f"Optimized Rg step {self._diffusion_step:3d}: "
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
            
            log_msg += " (post-optimization)"
            print(log_msg)
        
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