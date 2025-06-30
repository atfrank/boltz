# SAXS-Guided Diffusion in Boltz

This document provides a comprehensive guide for implementing and using SAXS (Small Angle X-ray Scattering) guidance in boltz structure generation, as well as a template for implementing similar experimental data guidance systems.

## Table of Contents

1. [Overview](#overview)
2. [Implementation Architecture](#implementation-architecture)
3. [SAXS-Specific Implementation](#saxs-specific-implementation)
4. [Usage Guide](#usage-guide)
5. [Developer Guide for Similar Implementations](#developer-guide-for-similar-implementations)
6. [Troubleshooting](#troubleshooting)
7. [Performance Considerations](#performance-considerations)

## Overview

SAXS-guided diffusion allows boltz to generate protein/RNA structures that are consistent with experimental Small Angle X-ray Scattering data. The system integrates experimental SAXS profiles into the diffusion process as a guidance potential, steering structure generation toward conformations that match the experimental scattering pattern.

### Key Features

- **Real-time SAXS calculation**: Computes theoretical SAXS profiles during diffusion using DENSS
- **Chi-squared guidance**: Uses chi-squared between experimental and calculated profiles as a potential energy
- **Multi-sample support**: Generates multiple independent trajectories with separate logging
- **Comprehensive logging**: Tracks chi-squared evolution during diffusion for analysis
- **Flexible configuration**: YAML-based configuration with extensive parameter control

## Implementation Architecture

The SAXS guidance system follows a modular architecture that can be adapted for other experimental data types:

### 1. Data Type System (`src/boltz/data/types.py`)

```python
@dataclass(frozen=True)
class SAXSGuidanceConfig:
    """SAXS guidance configuration."""
    experimental_data: str              # Path to experimental data
    guidance_weight: float = 0.1        # Strength of guidance
    guidance_interval: int = 5          # Apply guidance every N steps
    resampling_weight: float = 0.05     # Weight for particle resampling
    voxel_size: float = 2.0            # SAXS calculation voxel size
    oversampling: float = 3.0          # FFT oversampling factor
    gradient_epsilon: float = 0.1       # Finite difference step size

@dataclass(frozen=True)
class GuidanceConfig:
    """Container for all guidance types."""
    saxs: Optional[SAXSGuidanceConfig] = None
    # Future: neutron: Optional[NeutronGuidanceConfig] = None
    # Future: nmr: Optional[NMRGuidanceConfig] = None
```

### 2. YAML Schema Extension (`src/boltz/data/parse/schema.py`)

The parser extracts guidance configuration from YAML input and validates parameters:

```python
# Parse guidance configuration
guidance_config = None
guidance_section = schema.get("guidance", {})
if guidance_section:
    saxs_config = None
    if "saxs" in guidance_section:
        saxs_params = guidance_section["saxs"]
        if "experimental_data" not in saxs_params:
            msg = "SAXS guidance requires 'experimental_data' field"
            raise ValueError(msg)
        
        saxs_config = SAXSGuidanceConfig(
            experimental_data=saxs_params["experimental_data"],
            guidance_weight=saxs_params.get("guidance_weight", 0.1),
            # ... other parameters with defaults
        )
    
    if saxs_config:
        guidance_config = GuidanceConfig(saxs=saxs_config)
```

### 3. Data Flow Pipeline

1. **Preprocessing** (`src/boltz/main.py`): Save guidance config to JSON during input processing
2. **Loading** (`src/boltz/data/module/inferencev2.py`): Load guidance config and add to Input object
3. **Feature Integration**: Pass guidance through feature pipeline to model
4. **Model Integration** (`src/boltz/model/models/boltz2.py`): Extract and pass to diffusion module
5. **Potential Creation** (`src/boltz/model/modules/diffusionv2.py`): Create SAXS potential with guidance config

## SAXS-Specific Implementation

### Potential Class (`src/boltz/model/potentials/saxs_potential.py`)

The `SAXSPotential` class implements the core SAXS guidance functionality:

```python
class SAXSPotential(Potential):
    def __init__(self, experimental_saxs_file, parameters=None, ...):
        # Load experimental SAXS data (3-column format: q, I, error)
        self.experimental_data = np.loadtxt(experimental_saxs_file)
        
        # DENSS calculation parameters
        self.voxel_size = voxel_size
        self.oversampling = oversampling
        self.global_B = global_B
        
        # Logging setup
        self._chi2_logs = {}  # Per-sample logging
    
    def compute_variable(self, coords, index, compute_gradient=False):
        """Main SAXS calculation and comparison."""
        # Convert coordinates to PDB-like structure
        pdb = self._coords_to_pdb(coords)
        
        # Calculate theoretical SAXS using DENSS
        pdb2mrc = denss_core.PDB2MRC(pdb)
        pdb2mrc.run_all()
        
        # Compare with experimental data
        chi2, scale_factor = denss_core.calc_chi2(
            self.experimental_data, 
            pdb2mrc.Iq_calc,
            scale=True, interpolation=True
        )
        
        # Log results
        if self.log_chi2:
            self._chi2_logs[self._sample_counter].append({
                'step': self._step_counter,
                'chi2': chi2,
                'scale_factor': scale_factor
            })
        
        return chi2
```

### Key Integration Points

1. **Potential Registration** (`src/boltz/model/potentials/potentials.py`):
```python
def get_potentials(saxs_guidance_config=None):
    potentials = [/* standard potentials */]
    
    if saxs_guidance_config is not None:
        saxs_potential = SAXSPotential(
            experimental_saxs_file=saxs_guidance_config.experimental_data,
            parameters={
                "guidance_interval": saxs_guidance_config.guidance_interval,
                "guidance_weight": saxs_guidance_config.guidance_weight,
                # ...
            }
        )
        potentials.append(saxs_potential)
    
    return potentials
```

2. **Trajectory Enhancement** (`src/boltz/data/write/writer.py`):
```python
# Modified to save per-sample trajectories
if len(trajectory_coords_all.shape) == 4:
    # Multiple samples: extract trajectory for current sample
    trajectory_coords = trajectory_coords_all[:, model_idx]
    
# Save with sample-specific naming
traj_path = struct_dir / f"{record.id}_model_{model_idx}_trajectory.pdb"
```

## Usage Guide

### 1. Prepare SAXS Data

Experimental SAXS data should be in 3-column ASCII format:
```
# q (Å⁻¹)    I(q)         Error
0.02281      88.59        2.617
0.02341      86.65        2.311
0.02402      86.01        2.574
...
```

### 2. Create YAML Configuration

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: "MYDPROTEINSEQUENCE"
      # or RNA/DNA as needed

guidance:
  saxs:
    experimental_data: ./path/to/saxs_data.dat
    guidance_weight: 0.1          # Strength of SAXS guidance (0.05-0.2 typical)
    guidance_interval: 5          # Apply every 5 diffusion steps
    resampling_weight: 0.05       # Particle resampling strength
    voxel_size: 2.0              # Å, DENSS calculation resolution
    oversampling: 3.0            # FFT oversampling factor
    gradient_epsilon: 0.1         # Finite difference step size
```

### 3. Run SAXS-Guided Generation

```bash
python -m boltz.main predict input_with_saxs.yaml \
  --out_dir ./saxs_results \
  --sampling_steps 50 \
  --diffusion_samples 5 \
  --save_trajectory \
  --output_format pdb \
  --override
```

### 4. Analyze Results

**Generated Files:**
- `structure_model_0.pdb`, `structure_model_1.pdb`, etc. - Final structures
- `structure_model_0_trajectory.pdb`, etc. - Diffusion trajectories for each sample  
- `saxs_chi2_sample_0.json`, etc. - Chi-squared evolution logs
- `structure_guidance.json` - Saved guidance configuration

**Chi-squared Log Format:**
```json
[
  {
    "step": 0,
    "sample": 0,
    "chi2": 15.23,
    "scale_factor": 1.05
  },
  {
    "step": 1,
    "sample": 0,
    "chi2": 12.87,
    "scale_factor": 1.02
  }
]
```

### 5. Parameter Optimization

**guidance_weight**: Controls SAXS influence strength
- Start with 0.1, increase if SAXS fit is poor
- Decrease if structures become unrealistic
- Range: 0.05-0.5

**guidance_interval**: Frequency of SAXS evaluation
- Lower values = more frequent guidance, slower execution
- Higher values = less guidance, faster execution
- Range: 1-10

**voxel_size**: DENSS calculation resolution
- Smaller = higher resolution, slower calculation
- Larger = lower resolution, faster calculation
- Range: 1.5-3.0 Å

## Developer Guide for Similar Implementations

This section provides a template for implementing guidance from other experimental data types (NMR, neutron scattering, cross-linking, etc.).

### 1. Define Configuration Types

```python
@dataclass(frozen=True)
class MyExperimentalGuidanceConfig:
    """Configuration for my experimental guidance."""
    experimental_data: str
    guidance_weight: float = 0.1
    guidance_interval: int = 5
    # Add experiment-specific parameters
    specific_param1: float = 1.0
    specific_param2: str = "default"

# Add to GuidanceConfig
@dataclass(frozen=True)
class GuidanceConfig:
    saxs: Optional[SAXSGuidanceConfig] = None
    my_experiment: Optional[MyExperimentalGuidanceConfig] = None
```

### 2. Extend YAML Parser

In `src/boltz/data/parse/schema.py`:

```python
# In parse_boltz_schema function
if "my_experiment" in guidance_section:
    my_exp_params = guidance_section["my_experiment"]
    if "experimental_data" not in my_exp_params:
        msg = "My experiment guidance requires 'experimental_data' field"
        raise ValueError(msg)
    
    my_exp_config = MyExperimentalGuidanceConfig(
        experimental_data=my_exp_params["experimental_data"],
        guidance_weight=my_exp_params.get("guidance_weight", 0.1),
        specific_param1=my_exp_params.get("specific_param1", 1.0),
    )

if my_exp_config:
    guidance_config = GuidanceConfig(my_experiment=my_exp_config)
```

### 3. Implement Potential Class

```python
class MyExperimentalPotential(Potential):
    def __init__(self, experimental_data_file, parameters=None, **kwargs):
        super().__init__(parameters)
        
        # Load experimental data
        self.experimental_data = self._load_experimental_data(experimental_data_file)
        
        # Setup logging
        self.log_data = kwargs.get('log_data', True)
        self._data_logs = {}
        
    def compute_variable(self, coords, index, compute_gradient=False):
        """Calculate experimental observable from coordinates."""
        # 1. Convert coordinates to appropriate representation
        structure = self._coords_to_structure(coords)
        
        # 2. Calculate theoretical observable
        calculated_observable = self._calculate_observable(structure)
        
        # 3. Compare with experimental data
        chi2 = self._compare_with_experiment(calculated_observable)
        
        # 4. Log results
        if self.log_data:
            self._log_comparison(chi2, calculated_observable)
        
        # 5. Handle gradients if requested
        if compute_gradient:
            grad = self._compute_gradient(coords, chi2)
            return chi2, grad
        
        return chi2
    
    def _load_experimental_data(self, filename):
        """Load and validate experimental data."""
        # Implement data loading specific to your experiment
        pass
    
    def _calculate_observable(self, structure):
        """Calculate theoretical observable from structure."""
        # Implement calculation specific to your experiment
        pass
    
    def _compare_with_experiment(self, calculated):
        """Compare calculated vs experimental data."""
        # Implement comparison metric (chi2, RMSD, etc.)
        pass
```

### 4. Register Potential

In `src/boltz/model/potentials/potentials.py`:

```python
def get_potentials(saxs_guidance_config=None, my_experiment_guidance_config=None):
    potentials = [/* standard potentials */]
    
    # SAXS potential
    if saxs_guidance_config is not None:
        potentials.append(SAXSPotential(...))
    
    # Your experimental potential
    if my_experiment_guidance_config is not None:
        my_exp_potential = MyExperimentalPotential(
            experimental_data_file=my_experiment_guidance_config.experimental_data,
            parameters={
                "guidance_interval": my_experiment_guidance_config.guidance_interval,
                "guidance_weight": my_experiment_guidance_config.guidance_weight,
            },
            specific_param1=my_experiment_guidance_config.specific_param1,
        )
        potentials.append(my_exp_potential)
    
    return potentials
```

### 5. Update Function Signatures

Update the diffusion module to pass your guidance config:

```python
# In diffusionv2.py
my_exp_guidance_config = None
if guidance is not None:
    if isinstance(guidance, list):
        guidance_obj = guidance[0]
    else:
        guidance_obj = guidance
        
    if guidance_obj is not None and hasattr(guidance_obj, 'my_experiment'):
        if guidance_obj.my_experiment is not None:
            my_exp_guidance_config = guidance_obj.my_experiment

potentials = get_potentials(
    saxs_guidance_config=saxs_guidance_config,
    my_experiment_guidance_config=my_exp_guidance_config
)
```

### 6. Implementation Checklist

- [ ] Define configuration dataclass
- [ ] Add to GuidanceConfig union type
- [ ] Extend YAML parser with validation
- [ ] Implement potential class with proper interface
- [ ] Add logging and error handling
- [ ] Register in get_potentials() function
- [ ] Update function signatures in diffusion chain
- [ ] Add data loading/preprocessing functions
- [ ] Implement gradient calculation (analytical or finite difference)
- [ ] Add comprehensive testing
- [ ] Document usage and parameters

## Troubleshooting

### Common Issues

1. **DENSS Import Errors**
   ```
   ImportError: DENSS package not found
   ```
   **Solution**: Ensure DENSS is installed and accessible in Python path

2. **SAXS Data Format Errors**
   ```
   ValueError: SAXS data file must have at least 2 columns
   ```
   **Solution**: Verify experimental data is in 3-column format (q, I, error)

3. **Memory Issues with Large Systems**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce `diffusion_samples`, increase `guidance_interval`, or reduce `voxel_size`

4. **Poor SAXS Fits**
   ```
   SAXS guidance summary: min_chi2=50.23, final_chi2=45.67
   ```
   **Solution**: Increase `guidance_weight`, decrease `guidance_interval`, check experimental data quality

5. **Missing Trajectory Files**
   - Ensure `--save_trajectory` flag is used
   - Check that `diffusion_samples` > 1 for multiple trajectories
   - Verify write permissions in output directory

### Performance Optimization

1. **SAXS Calculation Speed**:
   - Increase `guidance_interval` (apply guidance less frequently)
   - Increase `voxel_size` (lower resolution, faster calculation)
   - Reduce `oversampling` factor

2. **Memory Usage**:
   - Reduce `diffusion_samples`
   - Use smaller `voxel_size` for SAXS calculations
   - Enable gradient checkpointing if available

3. **Convergence**:
   - Start with higher `guidance_weight` and decrease
   - Use more `sampling_steps` for better convergence
   - Monitor chi-squared logs for optimization

## Performance Considerations

### Computational Cost

- **SAXS Calculation**: O(N³) where N is number of voxels
- **Guidance Frequency**: Linear scaling with 1/guidance_interval
- **Multiple Samples**: Linear scaling with diffusion_samples

### Scaling Recommendations

| System Size | Guidance Interval | Voxel Size | Diffusion Samples |
|-------------|------------------|------------|-------------------|
| < 100 atoms | 1-3              | 1.5-2.0 Å | 5-10             |
| 100-500 atoms | 3-5            | 2.0-2.5 Å | 3-5              |
| > 500 atoms | 5-10             | 2.5-3.0 Å | 1-3              |

### Memory Requirements

Approximate GPU memory usage:
- Base model: 4-8 GB
- SAXS calculation: 1-2 GB per sample
- Trajectory storage: 100 MB per sample per 100 steps

## Future Extensions

### Planned Features

1. **Analytical Gradients**: Replace finite differences with analytical SAXS gradients
2. **Multi-scale SAXS**: Hierarchical resolution during diffusion
3. **Ensemble Averaging**: Handle dynamic systems with multiple conformations
4. **Experimental Error Weighting**: Better incorporation of data uncertainties

### Integration Opportunities

- **NMR Restraints**: NOE distance constraints, chemical shift prediction
- **Cross-linking Mass Spectrometry**: Distance constraints from XL-MS
- **Cryo-EM Density**: Real-space density fitting
- **Neutron Scattering**: Deuteration-specific contrast variation

This architecture provides a robust foundation for integrating diverse experimental data types into the boltz diffusion process, enabling more accurate and experimentally-consistent structure generation.