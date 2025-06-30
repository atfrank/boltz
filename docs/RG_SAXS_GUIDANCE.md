# Radius of Gyration Guidance and SAXS Scoring

This document describes the new Rg guidance and SAXS scoring system implemented for Boltz structure diffusion.

## Overview

The system provides two complementary approaches for incorporating experimental SAXS data:

1. **Rg Guidance**: Fast, smooth steering during diffusion using radius of gyration restraints
2. **SAXS Scoring**: Detailed evaluation of final structures against full experimental profiles

## Key Benefits

- **Rg Guidance**: ~1000x faster than full SAXS, stable gradients, smooth optimization
- **SAXS Scoring**: Full experimental validation, confidence scores, detailed comparisons
- **Flexible**: Can use either separately or together

## Usage Examples

### 1. Rg Guidance with Direct Target

```yaml
version: 1
sequences:
  - rna:
      id: A
      sequence: GGCUUAUCAAGAGAGGUGGAGGGACUGGCCCGAUGAAACCCGGCAACCAGAAAUGGUGCCAAUUCCUGCAGCGGAAACGUUGAAAGAUGAGCCG

guidance:
  rg:
    target_rg: 28.5  # Angstroms
    force_constant: 10.0  # kcal/mol/Å²
    mass_weighted: true
```

### 2. Rg Guidance from SAXS Data (Recommended)

```yaml
guidance:
  rg:
    # Extract Rg from experimental SAXS using Guinier analysis
    saxs_file_path: "./data/experimental.dat"
    q_min: 0.01  # Guinier region minimum q
    q_max: 0.05  # Guinier region maximum q
    force_constant: 15.0
    mass_weighted: true

# Optional: Score final structures against full SAXS profile
scoring:
  saxs:
    experimental_file: "./data/experimental.dat"
    max_q_points: 100
    output_dir: "./saxs_results"
```

### 3. SAXS Scoring Only (No Guidance)

```yaml
# No guidance section - free diffusion

scoring:
  saxs:
    experimental_file: "./data/experimental.dat"
    max_q_points: 100
    output_dir: "./saxs_results"
```

## Implementation Details

### Rg Potential (`rg_potential.py`)

The Rg potential applies a harmonic restraint:

```
E = 0.5 * k * (Rg_current - Rg_target)²
```

**Key Features:**
- Mass-weighted or uniform-weighted Rg calculation
- Automatic extraction of target Rg from SAXS data via Guinier analysis
- Smooth, analytical gradients via PyTorch autograd
- ~0.5ms computation time for 100 atoms

**Guinier Analysis:**
```
ln(I(q)) = ln(I₀) - (Rg²/3) * q²
```
- Fits experimental data in low-q region (typically 0.01-0.05 Å⁻¹)
- Weighted least squares using experimental errors
- Robust error handling for poor data

### SAXS Scorer (`saxs_scoring.py`)

Comprehensive scoring system for final structure evaluation:

**Features:**
- JAX-based SAXS calculation using Debye equation
- Optimal scaling and shifting via weighted least squares
- Chi-squared goodness of fit metrics
- Confidence scores (0-1 scale)
- Detailed comparison files (experimental vs calculated)
- JSON results with metadata

**Output Files:**
- `saxs_comparison_{sample_id}.dat`: Point-by-point comparison
- `saxs_scores_{sample_id}.json`: Quantitative results
- `saxs_scoring_summary.json`: Ranked results for all samples

## Performance Comparison

| Method | Speed | Gradient Quality | Information Content | Use Case |
|--------|-------|------------------|---------------------|-----------|
| Rg Guidance | ~0.5ms | Excellent | Low (size only) | Diffusion steering |
| SAXS Scoring | ~100ms | N/A | High (full profile) | Final evaluation |
| Full SAXS Guidance | ~500ms | Good | High | Special cases only |

## Configuration Parameters

### Rg Guidance

```yaml
guidance:
  rg:
    # Target specification (choose one)
    target_rg: 25.0                    # Direct specification (Å)
    saxs_file_path: "data/exp.dat"     # Extract via Guinier analysis
    
    # Guinier analysis parameters
    q_min: 0.01                        # Minimum q for fit (Å⁻¹)
    q_max: 0.05                        # Maximum q for fit (Å⁻¹)
    
    # Potential parameters
    force_constant: 10.0               # Restraint strength (kcal/mol/Å²)
    mass_weighted: true                # Use atomic masses
    atom_selection: null               # Future: atom subset
```

### SAXS Scoring

```yaml
scoring:
  saxs:
    experimental_file: "data/exp.dat"  # Required: experimental data
    max_q_points: 100                  # Limit data points for speed
    output_dir: "saxs_results"         # Output directory
```

## Data File Formats

### Experimental SAXS Data
```
# q(Å⁻¹)    I(q)        sigma
0.010       1000.0      50.0
0.020       950.0       47.5
0.030       850.0       42.5
...
```

### Output Comparison File
```
# SAXS Profile Comparison
# Sample: sample_001
# Chi2: 125.3
# Chi2_reduced: 1.28
# Scale: 0.85432
# Shift: 12.45
# Columns: q(A^-1) I_exp I_sigma I_calc I_fitted residual
  0.0100  1000.000000  50.000000   920.150000   895.483210    2.093
  0.0200   950.000000  47.500000   875.200000   859.546890    1.905
...
```

### JSON Results
```json
{
  "sample_id": "sample_001",
  "chi2": 125.3,
  "chi2_reduced": 1.28,
  "scale_factor": 0.85432,
  "shift_parameter": 12.45,
  "n_data_points": 98,
  "rg_calculated": 24.7,
  "confidence_score": 0.756
}
```

## Best Practices

### Choosing Parameters

1. **Force Constant**: 
   - RNA: 10-20 kcal/mol/Å²
   - Proteins: 5-15 kcal/mol/Å²
   - Start conservative, increase if needed

2. **Guinier Range**:
   - Standard: q_min=0.01, q_max=0.05 Å⁻¹
   - Adjust based on data quality
   - Need ≥5 data points in range

3. **Mass Weighting**:
   - Usually `true` for physical accuracy
   - Set `false` for geometric Rg

### Workflow Recommendations

1. **Development**: Start with Rg guidance only for fast iteration
2. **Validation**: Add SAXS scoring to final pipeline
3. **Production**: Use both for optimal balance of speed and accuracy

### Troubleshooting

**Poor Rg Extraction:**
- Check Guinier region has sufficient data points
- Verify low-q data quality
- Adjust q_min/q_max range

**High Chi-squared:**
- Check experimental data format
- Verify structure is reasonable
- Consider multiple conformations

**Optimization Issues:**
- Reduce force constant
- Check for structural clashes
- Verify target Rg is reasonable

## Technical Notes

### Form Factors
- SAXS scoring uses simplified single-Gaussian form factors
- ~2-7% difference from full Cromer-Mann coefficients
- Sufficient accuracy for scoring and guidance

### Numerical Stability
- Extensive safeguards against NaN/Inf values
- Upper triangular distance matrix avoids diagonal zeros
- Robust Guinier fitting with error handling

### Integration with Boltz
- Rg potential designed as standalone module
- SAXS scorer operates on final structures
- Minimal dependencies on diffusion framework