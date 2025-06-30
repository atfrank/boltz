"""SAXS scoring system for final structure evaluation."""

import os
import json
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Import JAX for SAXS calculation
try:
    import jax
    import jax.numpy as jnp
    from boltz.model.potentials.saxs_potential import compute_saxs_intensity_jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class SAXSScorer:
    """SAXS scoring system for evaluating final structures.
    
    This class calculates SAXS profiles for final structures and compares
    them with experimental data to provide confidence scores. It does NOT
    provide gradients for diffusion guidance.
    """
    
    def __init__(
        self,
        experimental_file: str,
        max_q_points: int = 100,
        output_dir: Optional[str] = None,
    ):
        """Initialize SAXS scorer.
        
        Args:
            experimental_file: Path to experimental SAXS data file (q, I, sigma)
            max_q_points: Maximum number of q-points to use
            output_dir: Directory for output files (default: same as experimental file)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for SAXS scoring but not available")
            
        if not os.path.exists(experimental_file):
            raise FileNotFoundError(f"Experimental SAXS file not found: {experimental_file}")
            
        self.experimental_file = experimental_file
        self.max_q_points = max_q_points
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path(experimental_file).parent
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load experimental data
        self.q_exp, self.I_exp, self.sigma_exp = self._load_experimental_data()
        
        print(f"SAXS Scorer initialized:")
        print(f"  Experimental file: {experimental_file}")
        print(f"  Q-points: {len(self.q_exp)} (max {max_q_points})")
        print(f"  Q-range: {self.q_exp[0]:.3f} - {self.q_exp[-1]:.3f} Å⁻¹")
        print(f"  Output directory: {self.output_dir}")
        
    def _load_experimental_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load experimental SAXS data."""
        try:
            data = np.loadtxt(self.experimental_file)
            q = data[:, 0]
            I = data[:, 1] 
            sigma = data[:, 2] if data.shape[1] > 2 else np.ones_like(I) * 0.1 * I
            
            # Limit number of points if requested
            if len(q) > self.max_q_points:
                indices = np.linspace(0, len(q)-1, self.max_q_points, dtype=int)
                q = q[indices]
                I = I[indices]
                sigma = sigma[indices]
                
            # Ensure positive intensities and errors
            valid_mask = (I > 0) & (sigma > 0) & (q > 0)
            q = q[valid_mask]
            I = I[valid_mask]
            sigma = sigma[valid_mask]
            
            return q, I, sigma
            
        except Exception as e:
            raise ValueError(f"Failed to load experimental SAXS data: {e}")
            
    def compute_saxs_profile(self, coords: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
        """Compute theoretical SAXS profile for given structure.
        
        Args:
            coords: Atomic coordinates [n_atoms, 3] in Angstroms
            atom_types: Atomic types [n_atoms] (atomic numbers)
            
        Returns:
            Theoretical SAXS intensities at experimental q-values
        """
        # Convert to JAX arrays
        coords_jax = jnp.array(coords)
        atom_types_jax = jnp.array(atom_types)
        q_values_jax = jnp.array(self.q_exp)
        
        # Compute SAXS intensities
        I_calc = compute_saxs_intensity_jax(coords_jax, q_values_jax, atom_types_jax)
        
        return np.array(I_calc)
        
    def compute_optimal_scaling(self, I_calc: np.ndarray) -> Tuple[float, float]:
        """Compute optimal scale and shift parameters using weighted least squares.
        
        Args:
            I_calc: Calculated SAXS intensities
            
        Returns:
            (scale, shift) parameters
        """
        # Weights from experimental errors
        weights = 1.0 / np.maximum(self.sigma_exp**2, 1e-10)
        
        # Weighted least squares for: I_exp = scale * I_calc + shift
        sum_w = np.sum(weights)
        sum_w_I_calc = np.sum(weights * I_calc)
        sum_w_I_exp = np.sum(weights * self.I_exp)
        sum_w_I_calc2 = np.sum(weights * I_calc**2)
        sum_w_I_exp_I_calc = np.sum(weights * self.I_exp * I_calc)
        
        # Solve 2x2 system using Cramer's rule
        det = sum_w_I_calc2 * sum_w - sum_w_I_calc**2
        
        if abs(det) < 1e-10:
            # Fallback to simple scaling
            scale = np.sum(weights * self.I_exp * I_calc) / np.sum(weights * I_calc**2)
            shift = 0.0
        else:
            scale = (sum_w_I_exp_I_calc * sum_w - sum_w_I_exp * sum_w_I_calc) / det
            shift = (sum_w_I_calc2 * sum_w_I_exp - sum_w_I_exp_I_calc * sum_w_I_calc) / det
            
        return float(scale), float(shift)
        
    def compute_chi2(self, I_calc: np.ndarray, scale: float, shift: float) -> float:
        """Compute chi-squared goodness of fit.
        
        Args:
            I_calc: Calculated SAXS intensities
            scale: Scale factor
            shift: Shift parameter
            
        Returns:
            Chi-squared value
        """
        I_fitted = scale * I_calc + shift
        residuals = (self.I_exp - I_fitted) / self.sigma_exp
        chi2 = np.sum(residuals**2)
        
        return float(chi2)
        
    def compute_rg_from_guinier(self, I_calc: np.ndarray, q_min: float = 0.01, q_max: float = 0.05) -> Optional[float]:
        """Extract radius of gyration from calculated profile using Guinier analysis.
        
        Args:
            I_calc: Calculated SAXS intensities
            q_min: Minimum q for Guinier fit
            q_max: Maximum q for Guinier fit
            
        Returns:
            Radius of gyration in Angstroms (None if fit fails)
        """
        try:
            # Select Guinier region
            mask = (self.q_exp >= q_min) & (self.q_exp <= q_max) & (I_calc > 0)
            
            if np.sum(mask) < 5:
                return None
                
            q_guinier = self.q_exp[mask]
            I_guinier = I_calc[mask]
            
            # Linear fit to ln(I) vs q²
            x = q_guinier**2
            y = np.log(I_guinier)
            
            # Simple least squares
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Rg from slope: slope = -Rg²/3
            rg_squared = -3.0 * slope
            
            if rg_squared > 0:
                return float(np.sqrt(rg_squared))
            else:
                return None
                
        except Exception:
            return None
            
    def score_structure(self, coords: np.ndarray, atom_types: np.ndarray, sample_id: str = "sample") -> Dict:
        """Score a single structure against experimental SAXS data.
        
        Args:
            coords: Atomic coordinates [n_atoms, 3]
            atom_types: Atomic types [n_atoms]
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary with scoring results
        """
        # Compute theoretical SAXS profile
        I_calc = self.compute_saxs_profile(coords, atom_types)
        
        # Compute optimal scaling
        scale, shift = self.compute_optimal_scaling(I_calc)
        
        # Compute chi-squared
        chi2 = self.compute_chi2(I_calc, scale, shift)
        
        # Reduced chi-squared
        n_points = len(self.q_exp)
        n_params = 2  # scale and shift
        chi2_reduced = chi2 / max(1, n_points - n_params)
        
        # Extract Rg from calculated profile
        rg_calc = self.compute_rg_from_guinier(I_calc)
        
        # Create results dictionary
        results = {
            "sample_id": sample_id,
            "chi2": chi2,
            "chi2_reduced": chi2_reduced,
            "scale_factor": scale,
            "shift_parameter": shift,
            "n_data_points": n_points,
            "rg_calculated": rg_calc,
            "confidence_score": self._compute_confidence_score(chi2_reduced),
        }
        
        # Save comparison data
        self._save_comparison_data(sample_id, I_calc, scale, shift, results)
        
        return results
        
    def _compute_confidence_score(self, chi2_reduced: float) -> float:
        """Compute confidence score from reduced chi-squared.
        
        Args:
            chi2_reduced: Reduced chi-squared value
            
        Returns:
            Confidence score between 0 and 1 (higher is better)
        """
        # Convert chi2_reduced to confidence score
        # Good fit: chi2_reduced ≈ 1, score ≈ 1
        # Poor fit: chi2_reduced >> 1, score ≈ 0
        
        if chi2_reduced <= 1.0:
            # Very good fits get high scores
            score = 1.0 - 0.2 * (chi2_reduced - 1.0)**2
        else:
            # Poor fits get exponentially decreasing scores
            score = np.exp(-(chi2_reduced - 1.0) / 2.0)
            
        return max(0.0, min(1.0, score))
        
    def _save_comparison_data(self, sample_id: str, I_calc: np.ndarray, scale: float, shift: float, results: Dict):
        """Save experimental vs predicted data comparison files.
        
        Args:
            sample_id: Sample identifier
            I_calc: Calculated SAXS intensities
            scale: Scale factor
            shift: Shift parameter
            results: Results dictionary
        """
        # Apply scaling to calculated intensities
        I_fitted = scale * I_calc + shift
        
        # Create data comparison file
        comparison_file = self.output_dir / f"saxs_comparison_{sample_id}.dat"
        
        with open(comparison_file, 'w') as f:
            f.write("# SAXS Profile Comparison\n")
            f.write(f"# Sample: {sample_id}\n")
            f.write(f"# Chi2: {results['chi2']:.3f}\n")
            f.write(f"# Chi2_reduced: {results['chi2_reduced']:.3f}\n")
            f.write(f"# Scale: {scale:.6f}\n")
            f.write(f"# Shift: {shift:.6f}\n")
            f.write("# Columns: q(A^-1) I_exp I_sigma I_calc I_fitted residual\n")
            
            for i in range(len(self.q_exp)):
                residual = (self.I_exp[i] - I_fitted[i]) / self.sigma_exp[i]
                f.write(f"{self.q_exp[i]:8.4f} {self.I_exp[i]:12.6e} {self.sigma_exp[i]:12.6e} "
                       f"{I_calc[i]:12.6e} {I_fitted[i]:12.6e} {residual:8.3f}\n")
                       
        # Save results as JSON
        json_file = self.output_dir / f"saxs_scores_{sample_id}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"SAXS scoring for {sample_id}: χ² = {results['chi2']:.1f}, confidence = {results['confidence_score']:.3f}")
        
    def score_multiple_structures(self, structures: List[Tuple[np.ndarray, np.ndarray, str]]) -> List[Dict]:
        """Score multiple structures and return sorted results.
        
        Args:
            structures: List of (coords, atom_types, sample_id) tuples
            
        Returns:
            List of results dictionaries sorted by confidence score (best first)
        """
        results = []
        
        for coords, atom_types, sample_id in structures:
            try:
                result = self.score_structure(coords, atom_types, sample_id)
                results.append(result)
            except Exception as e:
                print(f"Failed to score structure {sample_id}: {e}")
                # Add failed result
                results.append({
                    "sample_id": sample_id,
                    "chi2": float('inf'),
                    "chi2_reduced": float('inf'),
                    "confidence_score": 0.0,
                    "error": str(e)
                })
                
        # Sort by confidence score (highest first)
        results.sort(key=lambda x: x.get('confidence_score', 0.0), reverse=True)
        
        # Save summary
        summary_file = self.output_dir / "saxs_scoring_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nSAXS Scoring Summary ({len(results)} structures):")
        print("-" * 60)
        for i, result in enumerate(results[:10]):  # Show top 10
            print(f"{i+1:2d}. {result['sample_id']:15s} χ²={result.get('chi2', float('inf')):8.1f} "
                  f"conf={result.get('confidence_score', 0.0):.3f}")
                  
        return results


# Factory function for easy creation from config
def create_saxs_scorer_from_config(config: dict) -> SAXSScorer:
    """Create SAXS scorer from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SAXSScorer
        
    Example config:
        saxs_scoring:
            experimental_file: "data/experimental.dat"
            max_q_points: 100
            output_dir: "saxs_output"
    """
    return SAXSScorer(
        experimental_file=config["experimental_file"],
        max_q_points=config.get("max_q_points", 100),
        output_dir=config.get("output_dir"),
    )