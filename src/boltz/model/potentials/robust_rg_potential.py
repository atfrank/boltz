"""Robust Radius of Gyration Potential with Outlier Protection.

This implementation addresses the "extreme atom cheating" problem where Rg guidance
can achieve target values by displacing a few atoms to extreme positions rather
than reorganizing the overall structure properly.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple
from boltz.model.potentials.rg_potential import RadiusOfGyrationPotential


class RobustRgPotential(RadiusOfGyrationPotential):
    """Enhanced Rg potential with multiple safeguards against extreme atom displacement."""
    
    def __init__(
        self,
        target_rg: Optional[float] = None,
        saxs_file_path: Optional[str] = None,
        force_constant: float = 10.0,
        q_min: float = 0.01,
        q_max: float = 0.05,
        mass_weighted: bool = True,
        atom_selection: str = "all",
        # New parameters for robustness
        max_displacement_per_step: float = 2.0,  # Å per diffusion step
        outlier_threshold: float = 3.0,  # Standard deviations for outlier detection
        rg_calculation_method: str = "robust",  # "standard", "robust", "trimmed"
        gradient_capping: float = 10.0,  # Maximum gradient magnitude per atom (kcal/mol/Å)
        force_ramping: bool = True,  # Gradually increase force constant
        min_force_constant: float = 1.0,  # Starting force constant for ramping
        ramping_steps: int = 50,  # Steps to reach full force constant
    ):
        """Initialize robust Rg potential.
        
        Args:
            target_rg: Target radius of gyration in Angstroms
            saxs_file_path: Path to SAXS data file for Guinier analysis
            force_constant: Force constant for Rg restraint (kcal/mol/Å²)
            q_min, q_max: Q-range for Guinier analysis
            mass_weighted: Whether to use mass-weighted Rg calculation
            atom_selection: Which atoms to include ("all", "heavy", etc.)
            max_displacement_per_step: Maximum allowed displacement per diffusion step
            outlier_threshold: Threshold for outlier detection (standard deviations)
            rg_calculation_method: Method for Rg calculation ("standard", "robust", "trimmed")
            gradient_capping: Maximum gradient magnitude per atom
            force_ramping: Whether to gradually increase force constant
            min_force_constant: Starting force constant for ramping
            ramping_steps: Number of steps to reach full force constant
        """
        super().__init__(
            target_rg=target_rg,
            saxs_file_path=saxs_file_path,
            force_constant=force_constant,
            q_min=q_min,
            q_max=q_max,
            mass_weighted=mass_weighted,
            atom_selection=atom_selection,
        )
        
        # Robustness parameters
        self.max_displacement_per_step = max_displacement_per_step
        self.outlier_threshold = outlier_threshold
        self.rg_calculation_method = rg_calculation_method
        self.gradient_capping = gradient_capping
        self.force_ramping = force_ramping
        self.min_force_constant = min_force_constant
        self.ramping_steps = ramping_steps
        self.original_force_constant = force_constant
        
        # State tracking
        self.step_count = 0
        self.previous_coords = None
        self.displacement_violations = 0
        
        print(f"Initialized RobustRgPotential:")
        print(f"  - Target Rg: {self.target_rg:.2f} Å")
        print(f"  - Force constant: {self.force_constant:.1f} kcal/mol/Å²")
        print(f"  - Max displacement/step: {self.max_displacement_per_step:.1f} Å")
        print(f"  - Rg calculation: {self.rg_calculation_method}")
        print(f"  - Gradient capping: {self.gradient_capping:.1f} kcal/mol/Å")
        print(f"  - Force ramping: {self.force_ramping}")
    
    def get_effective_force_constant(self) -> float:
        """Get current effective force constant (with ramping if enabled)."""
        if not self.force_ramping:
            return self.force_constant
        
        # Linear ramping from min_force_constant to full force_constant
        if self.step_count >= self.ramping_steps:
            return self.original_force_constant
        
        progress = self.step_count / self.ramping_steps
        current_k = self.min_force_constant + progress * (self.original_force_constant - self.min_force_constant)
        return current_k
    
    def compute_robust_rg(self, coords: torch.Tensor, masses: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute radius of gyration with outlier protection.
        
        Returns:
            rg: Computed radius of gyration
            com: Center of mass
            info: Dictionary with diagnostic information
        """
        # Handle batch dimensions
        if coords.ndim == 3:
            coord_batch = coords[0]
        elif coords.ndim == 4:
            coord_batch = coords[0, 0]
        else:
            coord_batch = coords
        
        n_atoms = coord_batch.shape[0]
        
        # Compute center of mass
        if masses is not None and self.mass_weighted:
            if masses.ndim == coords.ndim:
                mass_batch = masses[0] if masses.ndim == 3 else masses[0, 0]
            else:
                mass_batch = masses
            total_mass = mass_batch.sum()
            com = torch.sum(coord_batch * mass_batch.unsqueeze(-1), dim=0) / total_mass
        else:
            com = coord_batch.mean(dim=0)
        
        # Compute deviations from center of mass
        deviations = coord_batch - com
        distances_sq = torch.sum(deviations**2, dim=1)
        
        info = {"method": self.rg_calculation_method, "n_atoms": n_atoms}
        
        if self.rg_calculation_method == "standard":
            # Standard Rg calculation
            rg_squared = torch.mean(distances_sq)
            valid_atoms = torch.ones(n_atoms, dtype=torch.bool, device=coord_batch.device)
            
        elif self.rg_calculation_method == "robust":
            # Robust Rg calculation using Median Absolute Deviation (MAD)
            distances = torch.sqrt(distances_sq)
            median_dist = torch.median(distances)
            mad = torch.median(torch.abs(distances - median_dist))
            
            # Identify outliers using robust threshold
            threshold = median_dist + self.outlier_threshold * mad * 1.4826  # 1.4826 makes MAD consistent with std
            valid_atoms = distances <= threshold
            
            if valid_atoms.sum() < n_atoms * 0.5:  # If too many atoms marked as outliers
                print(f"Warning: {n_atoms - valid_atoms.sum()} atoms marked as outliers, using trimmed method")
                valid_atoms = distances <= torch.quantile(distances, 0.95)  # Keep 95% of atoms
            
            valid_distances_sq = distances_sq[valid_atoms]
            rg_squared = torch.mean(valid_distances_sq) if len(valid_distances_sq) > 0 else torch.mean(distances_sq)
            
            info["outliers_detected"] = (n_atoms - valid_atoms.sum()).item()
            info["outlier_threshold"] = threshold.item()
            
        elif self.rg_calculation_method == "trimmed":
            # Trimmed mean: exclude top 5% of distances
            n_keep = int(n_atoms * 0.95)
            sorted_distances_sq, _ = torch.sort(distances_sq)
            trimmed_distances_sq = sorted_distances_sq[:n_keep]
            rg_squared = torch.mean(trimmed_distances_sq)
            
            threshold_value = sorted_distances_sq[n_keep-1].item()
            valid_atoms = distances_sq <= threshold_value
            info["trimmed_atoms"] = n_atoms - n_keep
            info["trim_threshold"] = threshold_value
        
        else:
            raise ValueError(f"Unknown rg_calculation_method: {self.rg_calculation_method}")
        
        rg = torch.sqrt(rg_squared)
        info["rg"] = rg.item()
        info["valid_atoms"] = valid_atoms.sum().item()
        
        return rg, com, info
    
    def compute_robust_gradients(
        self, 
        coords: torch.Tensor, 
        current_rg: torch.Tensor,
        com: torch.Tensor,
        valid_atoms: torch.Tensor,
        masses: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute gradients with displacement constraints and capping."""
        # Handle batch dimensions
        if coords.ndim == 3:
            coord_batch = coords[0]
        elif coords.ndim == 4:
            coord_batch = coords[0, 0]
        else:
            coord_batch = coords
        
        n_atoms = coord_batch.shape[0]
        
        # Compute deviations from center of mass
        deviations = coord_batch - com
        
        # Gradient of Rg with respect to each atom position
        # ∂Rg/∂r_i = (1/Rg) * (1/N_valid) * (r_i - r_com) for valid atoms
        if current_rg > 1e-8:
            # Only apply gradients to valid (non-outlier) atoms
            grad_rg = torch.zeros_like(deviations)
            
            if valid_atoms.sum() > 0:
                n_valid = valid_atoms.sum().float()
                grad_rg[valid_atoms] = deviations[valid_atoms] / (n_valid * current_rg)
            else:
                # Fallback if no valid atoms
                grad_rg = deviations / (n_atoms * current_rg)
        else:
            grad_rg = torch.zeros_like(deviations)
        
        # Apply displacement constraints
        if self.previous_coords is not None:
            current_displacement = torch.norm(coord_batch - self.previous_coords, dim=1)
            max_allowed = self.max_displacement_per_step
            
            # Identify atoms that would exceed displacement limit
            violating_atoms = current_displacement > max_allowed
            if violating_atoms.sum() > 0:
                self.displacement_violations += violating_atoms.sum().item()
                # Reduce gradients for violating atoms
                reduction_factor = max_allowed / (current_displacement[violating_atoms] + 1e-8)
                grad_rg[violating_atoms] *= reduction_factor.unsqueeze(-1)
        
        # Apply gradient capping
        grad_magnitudes = torch.norm(grad_rg, dim=1)
        capping_mask = grad_magnitudes > self.gradient_capping
        if capping_mask.sum() > 0:
            capping_factors = self.gradient_capping / (grad_magnitudes[capping_mask] + 1e-8)
            grad_rg[capping_mask] *= capping_factors.unsqueeze(-1)
        
        # Gradient of energy with respect to each atom position
        delta_rg = current_rg - self.target_rg
        effective_k = self.get_effective_force_constant()
        grad_energy = effective_k * delta_rg * grad_rg
        
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
    
    def compute_energy(self, coords: torch.Tensor, masses: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Compute Rg energy with robustness features."""
        self.step_count += 1
        
        # Store previous coordinates for displacement monitoring
        if coords.ndim == 3:
            current_coords = coords[0].clone()
        elif coords.ndim == 4:
            current_coords = coords[0, 0].clone()
        else:
            current_coords = coords.clone()
        
        # Compute robust Rg
        current_rg, com, rg_info = self.compute_robust_rg(coords, masses)
        
        # Compute energy: E = 0.5 * k_eff * (Rg - Rg_target)²
        delta_rg = current_rg - self.target_rg
        effective_k = self.get_effective_force_constant()
        energy = 0.5 * effective_k * delta_rg ** 2
        
        # Prepare info dictionary
        info = {
            "step": self.step_count,
            "rg": current_rg.item(),
            "target_rg": self.target_rg,
            "rg_error": abs(delta_rg.item()),
            "effective_k": effective_k,
            "energy": energy.item(),
            "displacement_violations": self.displacement_violations,
            **rg_info
        }
        
        # Update previous coordinates
        self.previous_coords = current_coords
        
        return energy, info
    
    def compute_energy_and_gradients(
        self, 
        coords: torch.Tensor, 
        masses: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute both energy and gradients with robustness features."""
        # Compute robust Rg and related quantities
        current_rg, com, rg_info = self.compute_robust_rg(coords, masses)
        
        # Determine valid atoms from the Rg calculation
        if coords.ndim == 3:
            coord_batch = coords[0]
        elif coords.ndim == 4:
            coord_batch = coords[0, 0]
        else:
            coord_batch = coords
        
        # Reconstruct valid_atoms mask based on the method used
        deviations = coord_batch - com
        distances_sq = torch.sum(deviations**2, dim=1)
        distances = torch.sqrt(distances_sq)
        
        if self.rg_calculation_method == "robust":
            median_dist = torch.median(distances)
            mad = torch.median(torch.abs(distances - median_dist))
            threshold = median_dist + self.outlier_threshold * mad * 1.4826
            valid_atoms = distances <= threshold
        elif self.rg_calculation_method == "trimmed":
            n_keep = int(len(distances) * 0.95)
            sorted_distances, _ = torch.sort(distances)
            threshold_value = sorted_distances[n_keep-1]
            valid_atoms = distances <= threshold_value
        else:  # standard
            valid_atoms = torch.ones_like(distances, dtype=torch.bool)
        
        # Compute energy
        delta_rg = current_rg - self.target_rg
        effective_k = self.get_effective_force_constant()
        energy = 0.5 * effective_k * delta_rg ** 2
        
        # Compute robust gradients
        grad_coords = self.compute_robust_gradients(coords, current_rg, com, valid_atoms, masses)
        
        # Prepare info dictionary
        info = {
            "step": self.step_count,
            "rg": current_rg.item(),
            "target_rg": self.target_rg,
            "rg_error": abs(delta_rg.item()),
            "effective_k": effective_k,
            "energy": energy.item(),
            "max_gradient": grad_coords.abs().max().item(),
            "gradient_norm": grad_coords.norm().item(),
            "displacement_violations": self.displacement_violations,
            **rg_info
        }
        
        return energy, grad_coords, info