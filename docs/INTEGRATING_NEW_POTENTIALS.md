# Guide to Integrating New Guiding Potentials in Boltz

This comprehensive guide explains how to add new guiding potentials to the Boltz molecular structure prediction system and integrate them into the confidence score reporting.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Integration with Confidence Scoring](#integration-with-confidence-scoring)
5. [Testing and Validation](#testing-and-validation)
6. [Best Practices](#best-practices)
7. [Example: Complete Integration](#example-complete-integration)

## Overview

Guiding potentials in Boltz allow users to steer the diffusion process toward structures that satisfy experimental constraints. Currently, Boltz supports SAXS (Small-Angle X-ray Scattering) and Rg (Radius of Gyration) guidance. This guide shows how to add new types of experimental guidance.

### Key Components

1. **Data Types** (`src/boltz/data/types.py`): Configuration dataclasses
2. **YAML Parsing** (`src/boltz/data/parse/schema.py`): User input parsing
3. **JSON Serialization** (`src/boltz/main.py`): Inter-process communication
4. **Data Loading** (`src/boltz/data/module/inferencev2.py`): Runtime loading
5. **Potential Implementation** (`src/boltz/model/potentials/`): Core logic
6. **Integration** (`src/boltz/model/potentials/potentials.py`): System integration
7. **Confidence Scoring** (`src/boltz/model/models/boltz2.py`): Score reporting

## Architecture Overview

```
User YAML Input
    ↓
schema.py (Parse YAML)
    ↓
types.py (Create Config Dataclass)
    ↓
main.py (Serialize to JSON)
    ↓
inferencev2.py (Deserialize from JSON)
    ↓
potentials.py (Create Potential Instance)
    ↓
Your Potential Implementation
    ↓
boltz2.py (Apply Forces & Report Scores)
```

## Step-by-Step Implementation

### Step 1: Define Configuration Dataclass

First, create a configuration dataclass in `src/boltz/data/types.py`:

```python
@dataclass(frozen=True)
class MyGuidanceConfig:
    """Configuration for my new experimental guidance."""
    
    # Required parameters
    target_value: float  # The experimental target value
    experimental_data_path: Optional[str] = None  # Path to experimental data file
    
    # Guidance parameters
    force_constant: float = 10.0  # Force constant in kcal/mol/unit²
    guidance_interval: tuple[int, int] = (0, 1000)  # Steps to apply guidance
    
    # Algorithm-specific parameters
    calculation_method: str = "standard"  # Calculation method
    mass_weighted: bool = True  # Use mass weighting
    atom_selection: Optional[str] = None  # Atom selection string
    
    # Robustness parameters (recommended)
    robust_mode: bool = True  # Enable robust calculation
    max_displacement_per_step: float = 2.0  # Max atom displacement (Å)
    outlier_threshold: float = 3.0  # Outlier detection threshold
    gradient_capping: float = 10.0  # Max gradient magnitude
    
    # Force ramping (recommended for stability)
    force_ramping: bool = True  # Gradually increase force
    min_force_constant: float = 1.0  # Starting force constant
    ramping_steps: int = 50  # Steps to reach full force
```

Update the `GuidanceConfig` class to include your new guidance:

```python
@dataclass(frozen=True)
class GuidanceConfig:
    """Guidance configuration."""
    
    saxs: Optional[SAXSGuidanceConfig] = None
    rg: Optional[RgGuidanceConfig] = None
    my_guidance: Optional[MyGuidanceConfig] = None  # Add your guidance
```

### Step 2: Add YAML Parsing

In `src/boltz/data/parse/schema.py`, add parsing for your guidance in the `parse_inference_config` function:

```python
# Around line 1800, after Rg guidance parsing
if "my_guidance" in guidance_section:
    my_params = guidance_section["my_guidance"]
    
    # Validate required fields
    if "target_value" not in my_params and "experimental_data_path" not in my_params:
        msg = "My guidance requires either 'target_value' or 'experimental_data_path'"
        raise ValueError(msg)
    
    my_config = MyGuidanceConfig(
        target_value=my_params.get("target_value"),
        experimental_data_path=my_params.get("experimental_data_path"),
        force_constant=my_params.get("force_constant", 10.0),
        guidance_interval=tuple(my_params.get("guidance_interval", [0, 1000])),
        calculation_method=my_params.get("calculation_method", "standard"),
        mass_weighted=my_params.get("mass_weighted", True),
        atom_selection=my_params.get("atom_selection"),
        # Robustness parameters
        robust_mode=my_params.get("robust_mode", True),
        max_displacement_per_step=my_params.get("max_displacement_per_step", 2.0),
        outlier_threshold=my_params.get("outlier_threshold", 3.0),
        gradient_capping=my_params.get("gradient_capping", 10.0),
        force_ramping=my_params.get("force_ramping", True),
        min_force_constant=my_params.get("min_force_constant", 1.0),
        ramping_steps=my_params.get("ramping_steps", 50),
    )

# Update GuidanceConfig construction
if saxs_config or rg_config or my_config:
    guidance_config = GuidanceConfig(
        saxs=saxs_config, 
        rg=rg_config,
        my_guidance=my_config
    )
```

### Step 3: Update JSON Serialization

In `src/boltz/main.py`, update the guidance serialization (around line 615):

```python
guidance_data = {
    "saxs": {...} if target.guidance.saxs else None,
    "rg": {...} if target.guidance.rg else None,
    "my_guidance": {
        "target_value": target.guidance.my_guidance.target_value,
        "experimental_data_path": target.guidance.my_guidance.experimental_data_path,
        "force_constant": target.guidance.my_guidance.force_constant,
        "guidance_interval": list(target.guidance.my_guidance.guidance_interval),
        "calculation_method": target.guidance.my_guidance.calculation_method,
        "mass_weighted": target.guidance.my_guidance.mass_weighted,
        "atom_selection": target.guidance.my_guidance.atom_selection,
        # Include all robustness parameters
        "robust_mode": target.guidance.my_guidance.robust_mode,
        "max_displacement_per_step": target.guidance.my_guidance.max_displacement_per_step,
        "outlier_threshold": target.guidance.my_guidance.outlier_threshold,
        "gradient_capping": target.guidance.my_guidance.gradient_capping,
        "force_ramping": target.guidance.my_guidance.force_ramping,
        "min_force_constant": target.guidance.my_guidance.min_force_constant,
        "ramping_steps": target.guidance.my_guidance.ramping_steps,
    } if target.guidance.my_guidance else None
}
```

### Step 4: Update JSON Deserialization

In `src/boltz/data/module/inferencev2.py`, add loading for your guidance:

```python
# After Rg guidance loading (around line 150)
my_config = None
if guidance_data.get("my_guidance"):
    my_data = guidance_data["my_guidance"]
    my_config = MyGuidanceConfig(
        target_value=my_data.get("target_value"),
        experimental_data_path=my_data.get("experimental_data_path"),
        force_constant=my_data.get("force_constant", 10.0),
        guidance_interval=tuple(my_data.get("guidance_interval", [0, 1000])),
        calculation_method=my_data.get("calculation_method", "standard"),
        mass_weighted=my_data.get("mass_weighted", True),
        atom_selection=my_data.get("atom_selection"),
        # Robustness parameters
        robust_mode=my_data.get("robust_mode", True),
        max_displacement_per_step=my_data.get("max_displacement_per_step", 2.0),
        outlier_threshold=my_data.get("outlier_threshold", 3.0),
        gradient_capping=my_data.get("gradient_capping", 10.0),
        force_ramping=my_data.get("force_ramping", True),
        min_force_constant=my_data.get("min_force_constant", 1.0),
        ramping_steps=my_data.get("ramping_steps", 50),
    )

if saxs_config or rg_config or my_config:
    guidance = GuidanceConfig(saxs=saxs_config, rg=rg_config, my_guidance=my_config)
```

### Step 5: Implement the Potential

Create `src/boltz/model/potentials/my_potential.py`:

```python
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple

class MyGuidancePotential:
    """Implementation of my experimental guidance potential."""
    
    def __init__(
        self,
        target_value: float,
        force_constant: float = 10.0,
        mass_weighted: bool = True,
        calculation_method: str = "standard",
        atom_selection: Optional[str] = None,
        robust_mode: bool = True,
        max_displacement_per_step: float = 2.0,
        outlier_threshold: float = 3.0,
        gradient_capping: float = 10.0,
        force_ramping: bool = True,
        min_force_constant: float = 1.0,
        ramping_steps: int = 50,
    ):
        self.target_value = target_value
        self.force_constant = force_constant
        self.mass_weighted = mass_weighted
        self.calculation_method = calculation_method
        self.atom_selection = atom_selection
        self.robust_mode = robust_mode
        self.max_displacement_per_step = max_displacement_per_step
        self.outlier_threshold = outlier_threshold
        self.gradient_capping = gradient_capping
        self.force_ramping = force_ramping
        self.min_force_constant = min_force_constant
        self.ramping_steps = ramping_steps
        
        print(f"Initialized MyGuidancePotential:")
        print(f"  - Target value: {self.target_value}")
        print(f"  - Force constant: {self.force_constant}")
        print(f"  - Calculation method: {self.calculation_method}")
        print(f"  - Robust mode: {self.robust_mode}")
        print(f"  - Force ramping: {self.force_ramping}")
    
    def calculate_value(
        self, 
        coords: jnp.ndarray, 
        atom_mask: jnp.ndarray,
        atom_masses: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Calculate the experimental value from coordinates.
        
        Args:
            coords: Atom coordinates [N, 3]
            atom_mask: Valid atom mask [N]
            atom_masses: Atom masses [N] (optional)
            
        Returns:
            Calculated experimental value
        """
        # Example: Calculate some property from coordinates
        valid_coords = coords[atom_mask]
        
        if self.calculation_method == "standard":
            # Your calculation here
            value = jnp.mean(jnp.linalg.norm(valid_coords, axis=-1))
        elif self.calculation_method == "robust":
            # Robust calculation with outlier detection
            distances = jnp.linalg.norm(valid_coords, axis=-1)
            median_dist = jnp.median(distances)
            mad = jnp.median(jnp.abs(distances - median_dist))
            outlier_mask = jnp.abs(distances - median_dist) > self.outlier_threshold * mad
            clean_distances = jnp.where(outlier_mask, median_dist, distances)
            value = jnp.mean(clean_distances)
        else:
            raise ValueError(f"Unknown calculation method: {self.calculation_method}")
        
        return value
    
    def energy(
        self, 
        coords: jnp.ndarray, 
        atom_mask: jnp.ndarray,
        atom_masses: Optional[jnp.ndarray] = None,
        step: int = 0
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Calculate potential energy and gradients.
        
        Args:
            coords: Atom coordinates [N, 3]
            atom_mask: Valid atom mask [N]
            atom_masses: Atom masses [N] (optional)
            step: Current diffusion step
            
        Returns:
            energy: Potential energy
            aux_data: Auxiliary data for logging
        """
        # Calculate current value
        current_value = self.calculate_value(coords, atom_mask, atom_masses)
        
        # Calculate error
        error = current_value - self.target_value
        
        # Determine effective force constant (with optional ramping)
        if self.force_ramping and step < self.ramping_steps:
            k_eff = self.min_force_constant + (
                (self.force_constant - self.min_force_constant) * 
                (step / self.ramping_steps)
            )
        else:
            k_eff = self.force_constant
        
        # Calculate energy
        energy = 0.5 * k_eff * error ** 2
        
        # Prepare auxiliary data for logging
        aux_data = {
            "current_value": current_value,
            "target_value": self.target_value,
            "error": error,
            "energy": energy,
            "k_eff": k_eff,
            "step": step,
        }
        
        return energy, aux_data
    
    def energy_and_gradient(
        self, 
        coords: jnp.ndarray, 
        atom_mask: jnp.ndarray,
        atom_masses: Optional[jnp.ndarray] = None,
        step: int = 0
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Calculate energy and gradients with respect to coordinates.
        
        Args:
            coords: Atom coordinates [N, 3]
            atom_mask: Valid atom mask [N]
            atom_masses: Atom masses [N] (optional)
            step: Current diffusion step
            
        Returns:
            energy: Potential energy
            gradients: Gradients w.r.t. coordinates [N, 3]
            aux_data: Auxiliary data for logging
        """
        # Define value_and_grad function
        def _energy_fn(coords):
            energy, _ = self.energy(coords, atom_mask, atom_masses, step)
            return energy
        
        # Compute energy and gradients
        energy, gradients = jax.value_and_grad(_energy_fn)(coords)
        
        # Get auxiliary data
        _, aux_data = self.energy(coords, atom_mask, atom_masses, step)
        
        # Apply gradient capping if enabled
        if self.gradient_capping > 0:
            grad_norm = jnp.linalg.norm(gradients, axis=-1, keepdims=True)
            scale_factor = jnp.minimum(1.0, self.gradient_capping / (grad_norm + 1e-8))
            gradients = gradients * scale_factor
            
            # Add gradient info to aux_data
            aux_data["max_gradient"] = jnp.max(grad_norm)
            aux_data["capped_gradients"] = jnp.sum(grad_norm > self.gradient_capping)
        
        # Apply displacement limiting if enabled
        if self.max_displacement_per_step > 0:
            # Estimate displacement from gradient (assuming unit time step)
            displacement = jnp.linalg.norm(gradients, axis=-1, keepdims=True)
            scale_factor = jnp.minimum(
                1.0, 
                self.max_displacement_per_step / (displacement + 1e-8)
            )
            gradients = gradients * scale_factor
            
            aux_data["max_displacement"] = jnp.max(displacement)
        
        return energy, gradients, aux_data


class MyGuidanceWrapper:
    """Wrapper to integrate the potential with Boltz's system."""
    
    def __init__(self, config: 'MyGuidanceConfig'):
        """Initialize from configuration."""
        self.config = config
        
        # Load experimental data if provided
        if config.experimental_data_path:
            self.target_value = self._load_experimental_data(config.experimental_data_path)
        else:
            self.target_value = config.target_value
        
        # Create the potential
        self.potential = MyGuidancePotential(
            target_value=self.target_value,
            force_constant=config.force_constant,
            mass_weighted=config.mass_weighted,
            calculation_method=config.calculation_method,
            atom_selection=config.atom_selection,
            robust_mode=config.robust_mode,
            max_displacement_per_step=config.max_displacement_per_step,
            outlier_threshold=config.outlier_threshold,
            gradient_capping=config.gradient_capping,
            force_ramping=config.force_ramping,
            min_force_constant=config.min_force_constant,
            ramping_steps=config.ramping_steps,
        )
        
        # Store guidance interval
        self.start_step = config.guidance_interval[0]
        self.end_step = config.guidance_interval[1]
    
    def _load_experimental_data(self, path: str) -> float:
        """Load experimental data from file."""
        # Implement your data loading logic
        # This is just an example
        import numpy as np
        data = np.loadtxt(path)
        return float(data[0])  # Or process as needed
    
    def apply(
        self,
        coords: jnp.ndarray,
        res_mask: jnp.ndarray,
        atom_mask: jnp.ndarray,
        atom14_mask: jnp.ndarray,
        stage: int,
        step: int,
        self_cond: bool,
        structure: Any,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply the guidance potential.
        
        This method integrates with Boltz's diffusion system.
        """
        # Check if we should apply guidance at this step
        if step < self.start_step or step > self.end_step:
            # Return zero energy and gradients
            return jnp.zeros_like(coords), {"my_guidance_active": False}
        
        # Get atom masses if mass weighting is enabled
        atom_masses = None
        if self.config.mass_weighted and hasattr(structure, 'atom_masses'):
            atom_masses = structure.atom_masses
        
        # Calculate energy and gradients
        energy, gradients, aux_data = self.potential.energy_and_gradient(
            coords, atom_mask, atom_masses, step
        )
        
        # Apply masks to ensure we only affect valid atoms
        gradients = gradients * atom_mask[..., None]
        
        # Add metadata for logging
        aux_data["my_guidance_active"] = True
        aux_data["guidance_step"] = step
        
        # Log progress
        if step % 100 == 0:  # Log every 100 steps
            print(f"MyGuidance step {step:4d}: "
                  f"value={aux_data['current_value']:.2f} "
                  f"target={aux_data['target_value']:.2f} "
                  f"error={aux_data['error']:.2f} "
                  f"energy={aux_data['energy']:.1f} "
                  f"k_eff={aux_data['k_eff']:.1f}")
        
        return gradients, aux_data
```

### Step 6: Integrate with Potentials System

Update `src/boltz/model/potentials/potentials.py`:

```python
# Add import
from boltz.model.potentials.my_potential import MyGuidanceWrapper

# In the compute_validation_inputs function, add your guidance handling:
# Around line 300, after Rg guidance handling

my_guidance_energies = []
my_guidance_config = getattr(inputs.guidance, 'my_guidance', None) if inputs.guidance else None

if my_guidance_config is not None:
    print(f"My guidance activated: target={my_guidance_config.target_value}")
    
    # Create wrapper instance
    my_wrapper = MyGuidanceWrapper(my_guidance_config)
    
    for i in range(len(coords_batch)):
        # Apply guidance
        gradients, aux_data = my_wrapper.apply(
            coords_batch[i],
            resolved_mask[i],
            atom_mask[i],
            atom14_mask[i],
            stage=0,  # Adjust based on your needs
            step=0,   # Should come from diffusion step
            self_cond=False,
            structure=structure,
        )
        
        # Store energy for confidence scoring
        if "energy" in aux_data:
            my_guidance_energies.append(aux_data["energy"])
        
        # Apply gradients to coordinates (if in training/diffusion)
        # coords_batch[i] = coords_batch[i] - learning_rate * gradients

# Add to validation features
if my_guidance_energies:
    validation_feats["my_guidance_energy"] = jnp.array(my_guidance_energies)
    validation_feats["my_guidance_error"] = jnp.array([
        aux_data.get("error", 0.0) for aux_data in aux_data_list
    ])
```

### Step 7: Integration with Confidence Scoring

Update `src/boltz/model/models/boltz2.py` to include your guidance in confidence scoring:

```python
# In the compute_confidence function (around line 600)
# After existing confidence calculations

# Add your guidance scoring
my_guidance_score = None
if "my_guidance_energy" in validation_feats:
    # Calculate confidence based on how well we match the target
    # Lower energy = better match = higher confidence
    my_guidance_energy = validation_feats["my_guidance_energy"]
    my_guidance_error = validation_feats["my_guidance_error"]
    
    # Convert error to confidence score (0-1 range)
    # You can adjust this formula based on your needs
    error_threshold = 5.0  # Error value for 50% confidence
    my_guidance_score = 1.0 / (1.0 + (my_guidance_error / error_threshold) ** 2)
    
    # Log the score
    print(f"My guidance confidence score: {my_guidance_score:.3f} "
          f"(error: {my_guidance_error:.3f})")

# Combine with overall confidence
if my_guidance_score is not None:
    # Weight your guidance score (adjust weight as needed)
    guidance_weight = 0.2  # 20% weight for guidance
    
    # Update overall confidence
    combined_confidence = (
        (1 - guidance_weight) * base_confidence + 
        guidance_weight * my_guidance_score
    )
    
    # Add to confidence output
    confidence_dict["my_guidance_score"] = float(my_guidance_score)
    confidence_dict["my_guidance_error"] = float(my_guidance_error)
    confidence_dict["combined_confidence"] = float(combined_confidence)
```

## Testing and Validation

### 1. Create Test YAML Configuration

Create `examples/test_my_guidance.yaml`:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF

guidance:
  my_guidance:
    target_value: 25.0  # Target experimental value
    force_constant: 50.0
    calculation_method: "robust"
    mass_weighted: true
    
    # Robustness parameters
    robust_mode: true
    force_ramping: false  # Disable for immediate effect
    max_displacement_per_step: 3.0
    outlier_threshold: 2.5
    gradient_capping: 20.0
    
    # When to apply guidance
    guidance_interval: [0, 1000]

output:
  confidence_detailed: true  # Get detailed confidence scores
```

### 2. Write Unit Tests

Create `tests/test_my_guidance.py`:

```python
import pytest
import jax.numpy as jnp
from boltz.data.types import MyGuidanceConfig
from boltz.model.potentials.my_potential import MyGuidancePotential, MyGuidanceWrapper

def test_my_guidance_potential():
    """Test the basic potential calculation."""
    potential = MyGuidancePotential(
        target_value=25.0,
        force_constant=10.0,
        calculation_method="standard"
    )
    
    # Create test coordinates
    coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    atom_mask = jnp.array([True, True])
    
    # Calculate energy
    energy, aux_data = potential.energy(coords, atom_mask, step=0)
    
    assert "current_value" in aux_data
    assert "error" in aux_data
    assert energy >= 0  # Energy should be non-negative

def test_gradient_capping():
    """Test that gradient capping works correctly."""
    potential = MyGuidancePotential(
        target_value=100.0,  # Large target to create large gradients
        force_constant=1000.0,  # Large force constant
        gradient_capping=1.0  # Small cap
    )
    
    coords = jnp.zeros((10, 3))
    atom_mask = jnp.ones(10, dtype=bool)
    
    energy, gradients, aux_data = potential.energy_and_gradient(
        coords, atom_mask, step=0
    )
    
    # Check that no gradient exceeds the cap
    grad_norms = jnp.linalg.norm(gradients, axis=-1)
    assert jnp.all(grad_norms <= 1.0 + 1e-6)

def test_force_ramping():
    """Test that force ramping works correctly."""
    potential = MyGuidancePotential(
        target_value=25.0,
        force_constant=100.0,
        force_ramping=True,
        min_force_constant=1.0,
        ramping_steps=50
    )
    
    coords = jnp.ones((5, 3))
    atom_mask = jnp.ones(5, dtype=bool)
    
    # Test at different steps
    energies = []
    for step in [0, 25, 50, 100]:
        energy, aux_data = potential.energy(coords, atom_mask, step=step)
        energies.append(aux_data["k_eff"])
    
    # Check ramping progression
    assert energies[0] == 1.0  # Minimum at step 0
    assert energies[1] > energies[0]  # Increasing
    assert energies[2] == 100.0  # Full force at step 50
    assert energies[3] == 100.0  # Stays at full force

def test_yaml_parsing():
    """Test that YAML parsing works correctly."""
    yaml_content = """
    guidance:
      my_guidance:
        target_value: 30.0
        force_constant: 75.0
        force_ramping: false
        max_displacement_per_step: 4.0
    """
    
    # This would test the actual parsing in schema.py
    # Implementation depends on your testing setup

def test_confidence_integration():
    """Test integration with confidence scoring."""
    # Create mock validation features
    validation_feats = {
        "my_guidance_energy": jnp.array([10.0, 50.0, 100.0]),
        "my_guidance_error": jnp.array([1.0, 5.0, 10.0])
    }
    
    # Calculate confidence scores
    error_threshold = 5.0
    scores = 1.0 / (1.0 + (validation_feats["my_guidance_error"] / error_threshold) ** 2)
    
    # Check score properties
    assert jnp.all(scores >= 0) and jnp.all(scores <= 1)
    assert scores[0] > scores[1] > scores[2]  # Lower error = higher score
```

### 3. Integration Test

```python
def test_full_integration():
    """Test the complete integration pipeline."""
    from boltz.data.types import MyGuidanceConfig, GuidanceConfig
    from boltz.model.potentials.my_potential import MyGuidanceWrapper
    
    # Create config
    config = MyGuidanceConfig(
        target_value=25.0,
        force_constant=50.0,
        force_ramping=False
    )
    
    # Create wrapper
    wrapper = MyGuidanceWrapper(config)
    
    # Test with realistic data
    batch_size = 2
    n_atoms = 100
    coords = jnp.random.randn(batch_size, n_atoms, 3) * 10
    atom_mask = jnp.ones((batch_size, n_atoms), dtype=bool)
    
    # Apply guidance
    all_gradients = []
    all_aux_data = []
    
    for i in range(batch_size):
        gradients, aux_data = wrapper.apply(
            coords[i],
            atom_mask[i],
            atom_mask[i],
            atom_mask[i],
            stage=0,
            step=50,
            self_cond=False,
            structure=None
        )
        all_gradients.append(gradients)
        all_aux_data.append(aux_data)
    
    # Verify outputs
    assert len(all_gradients) == batch_size
    assert all_gradients[0].shape == (n_atoms, 3)
    assert "energy" in all_aux_data[0]
    assert "current_value" in all_aux_data[0]
```

## Best Practices

### 1. Robustness and Stability

- **Always implement gradient capping** to prevent explosive forces
- **Use force ramping** for large force constants to ensure stability
- **Implement outlier detection** for robust calculations
- **Limit maximum displacement** per step to prevent structure disruption

### 2. Performance Optimization

- **Use JAX operations** for automatic differentiation and GPU acceleration
- **Vectorize calculations** when possible for batch processing
- **Cache expensive computations** if they're reused
- **Profile your code** to identify bottlenecks

### 3. User Experience

- **Provide clear error messages** for invalid configurations
- **Log progress periodically** to help users monitor convergence
- **Document all parameters** with reasonable defaults
- **Include example configurations** in the documentation

### 4. Confidence Score Integration

- **Design meaningful score metrics** that reflect guidance quality
- **Scale scores to [0, 1] range** for consistency
- **Weight appropriately** when combining with other scores
- **Log intermediate values** for debugging

### 5. Testing

- **Write comprehensive unit tests** for each component
- **Test edge cases** (zero values, large forces, etc.)
- **Validate gradient calculations** using finite differences
- **Test the full pipeline** from YAML to confidence scores

## Example: Complete Integration

Here's a complete example of integrating a "Hydrophobic Moment" guidance:

### 1. Configuration (types.py)

```python
@dataclass(frozen=True)
class HydrophobicMomentConfig:
    """Configuration for hydrophobic moment guidance."""
    target_moment: float
    force_constant: float = 10.0
    window_size: int = 11  # Helix window
    force_ramping: bool = True
    min_force_constant: float = 1.0
    ramping_steps: int = 50
```

### 2. Implementation (hydrophobic_moment_potential.py)

```python
class HydrophobicMomentPotential:
    """Guide protein folding based on hydrophobic moment."""
    
    def calculate_moment(self, coords, sequence, atom_mask):
        """Calculate hydrophobic moment for alpha helices."""
        # Implementation details...
        
    def energy_and_gradient(self, coords, sequence, atom_mask, step):
        """Calculate energy and forces."""
        current_moment = self.calculate_moment(coords, sequence, atom_mask)
        error = current_moment - self.target_moment
        
        # Ramping
        if self.force_ramping and step < self.ramping_steps:
            k_eff = self.min_force_constant + (
                (self.force_constant - self.min_force_constant) * 
                (step / self.ramping_steps)
            )
        else:
            k_eff = self.force_constant
        
        energy = 0.5 * k_eff * error ** 2
        
        # Calculate gradients...
        return energy, gradients, {"moment": current_moment, "error": error}
```

### 3. Usage Example

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF

guidance:
  hydrophobic_moment:
    target_moment: 0.5  # Target hydrophobic moment
    force_constant: 30.0
    window_size: 11
    force_ramping: true
    ramping_steps: 100
```

## Troubleshooting

### Common Issues

1. **Parameters not passing through**: Check all 4 locations (schema.py, main.py serialization, inferencev2.py deserialization, types.py)

2. **Gradients exploding**: Implement gradient capping and force ramping

3. **Confidence scores not appearing**: Ensure validation_feats are properly populated and passed to confidence calculation

4. **YAML parsing errors**: Validate your configuration matches the expected schema

5. **Performance issues**: Profile with JAX's built-in profiler, ensure operations are vectorized

### Debugging Tips

1. Add print statements at each stage of the pipeline
2. Save intermediate values to files for inspection
3. Use small test cases to validate calculations
4. Compare numerical gradients with finite differences
5. Visualize the guided structures to ensure sensible behavior

## Conclusion

This guide provides a complete framework for integrating new guiding potentials into Boltz. The key steps are:

1. Define configuration dataclass
2. Implement YAML parsing
3. Add JSON serialization/deserialization
4. Implement the potential with gradients
5. Integrate with the potentials system
6. Add confidence scoring
7. Test thoroughly

Following this pattern ensures your guidance integrates smoothly with Boltz's architecture and provides users with a consistent, robust experience.