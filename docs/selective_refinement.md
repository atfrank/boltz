# Selective In-Place Refinement

Boltz-2 supports selective in-place refinement, allowing you to refine specific regions of a molecular structure while keeping other parts relatively fixed. This is particularly useful for:

- **Local optimization**: Refining binding sites, loops, or specific domains
- **Conformational sampling**: Exploring alternative conformations of flexible regions
- **Structure correction**: Fixing problematic regions in existing structures
- **Comparative modeling**: Refining homology models or predicted structures

## Quick Start

The basic selective refinement workflow involves three steps:

1. **Provide initial coordinates** with `--init_coords`
2. **Specify regions to refine** with `--add_noise`
3. **Control refinement strength** with `--start_sigma_scale`

```bash
boltz predict input.yaml \
  --init_coords initial_structure.pdb \
  --add_noise "chain:A:1.5" \
  --start_sigma_scale 0.001 \
  --no_random_augmentation
```

## Command Line Options

### Core Refinement Options

#### `--init_coords PATH`
Path to PDB or mmCIF file containing initial atomic coordinates.
- **Required for**: All selective refinement tasks
- **Supports**: PDB (.pdb) and mmCIF (.cif) formats
- **Example**: `--init_coords structure.pdb`

#### `--add_noise SPECIFICATION`
Specify which atoms/residues/regions to add noise to (i.e., to refine). Can be used multiple times.
- **Format**: `type:target:intensity`
- **Multiple**: Use multiple `--add_noise` flags for different regions
- **Example**: `--add_noise "chain:A:1.0" --add_noise "resid:B:100:0.5"`

#### `--start_sigma_scale FLOAT`
Scale factor for initial noise level (default: 1.0).
- **Lower values** (0.001-0.1): Gentle refinement, stays close to initial structure
- **Higher values** (0.5-1.0): More aggressive refinement, allows larger changes
- **Recommended**: 0.001 for local refinement, 0.01-0.1 for conformational sampling
- **Example**: `--start_sigma_scale 0.001`

### Advanced Options

#### `--no_random_augmentation`
Disable random rotation and translation during refinement.
- **Use when**: You want to preserve the exact coordinate frame
- **Default**: Random augmentation is enabled
- **Example**: `--no_random_augmentation`

#### `--non_target_noise_min FLOAT`
Minimum noise level for non-targeted atoms (default: 0.1).
- **Range**: 0.0-1.0
- **Lower values**: Keep non-targeted regions more fixed
- **Example**: `--non_target_noise_min 0.05`

#### `--non_target_noise_range FLOAT`
Noise range for non-targeted atoms (default: 0.2).
- **Formula**: `noise = min + range * step_progress`
- **Controls**: How much non-targeted atoms move during denoising
- **Example**: `--non_target_noise_range 0.1`

#### `--residue_based_selection`
When using distance-based noise, select entire residues if any atom is within the distance.
- **Default**: Select individual atoms within distance
- **Use when**: You want to refine complete residues near a site
- **Example**: `--residue_based_selection`

#### `--save_trajectory`
Save denoising trajectory showing structural evolution.
- **Output**: Multi-model PDB/mmCIF file with all denoising steps
- **Useful for**: Analyzing refinement process and convergence
- **Example**: `--save_trajectory`

## Noise Specification Formats

### Chain-Based Selection
Target entire chains for refinement.

```bash
--add_noise "chain:A:1.0"        # Refine all of chain A with intensity 1.0
--add_noise "chain:B:0.5"        # Refine all of chain B with intensity 0.5
```

### Residue-Based Selection
Target specific residues within a chain.

```bash
--add_noise "resid:A:10:1.0"     # Refine residue 10 in chain A
--add_noise "resid:B:50:0.8"     # Refine residue 50 in chain B
```

### Atom-Based Selection
Target specific atoms with precise control.

```bash
--add_noise "atom:A:10:CA:1.0"   # Refine CA atom of residue 10 in chain A
--add_noise "atom:B:25:CB:0.5"   # Refine CB atom of residue 25 in chain B
```

### Distance-Based Selection
Target all atoms within a distance from a reference point.

```bash
--add_noise "distance:A:10:5.0:1.0"    # Refine atoms within 5.0Å of residue 10 in chain A
--add_noise "distance:B:25:8.0:0.7"    # Refine atoms within 8.0Å of residue 25 in chain B
```

**Format**: `distance:chain:residue:radius:intensity`
- **chain**: Reference chain ID
- **residue**: Reference residue number
- **radius**: Distance cutoff in Angstroms
- **intensity**: Noise intensity (0.0-2.0)

With `--residue_based_selection`, entire residues are selected if any atom is within the distance.

## Practical Examples

### Example 1: Loop Refinement
Refine a flexible loop region (residues 45-55 in chain A):

```bash
boltz predict protein.yaml \
  --init_coords homology_model.pdb \
  --add_noise "resid:A:45:1.0" \
  --add_noise "resid:A:46:1.0" \
  --add_noise "resid:A:47:1.0" \
  --add_noise "resid:A:48:1.0" \
  --add_noise "resid:A:49:1.0" \
  --add_noise "resid:A:50:1.0" \
  --start_sigma_scale 0.01 \
  --no_random_augmentation
```

### Example 2: Binding Site Optimization
Refine a ligand binding site within 6Å of the ligand:

```bash
boltz predict complex.yaml \
  --init_coords docking_result.pdb \
  --add_noise "chain:L:1.5" \
  --add_noise "distance:A:100:6.0:0.8" \
  --start_sigma_scale 0.005 \
  --residue_based_selection \
  --save_trajectory
```

### Example 3: Conformational Sampling
Sample alternative conformations of a flexible domain:

```bash
boltz predict protein.yaml \
  --init_coords experimental.pdb \
  --add_noise "chain:B:1.2" \
  --start_sigma_scale 0.1 \
  --diffusion_samples 5 \
  --save_trajectory
```

### Example 4: Multi-Region Refinement
Refine multiple disconnected regions simultaneously:

```bash
boltz predict complex.yaml \
  --init_coords initial.pdb \
  --add_noise "resid:A:25:1.0" \
  --add_noise "resid:A:78:1.0" \
  --add_noise "chain:C:0.8" \
  --add_noise "distance:B:150:5.0:0.6" \
  --start_sigma_scale 0.01 \
  --non_target_noise_min 0.05 \
  --residue_based_selection
```

## Understanding Noise Intensity

The noise intensity parameter controls how much refinement is applied:

- **0.0-0.3**: Minimal refinement, small local adjustments
- **0.4-0.7**: Moderate refinement, significant conformational changes possible
- **0.8-1.2**: Strong refinement, large structural changes allowed
- **1.3-2.0**: Very strong refinement, extensive remodeling

## Understanding Sigma Scaling

The `--start_sigma_scale` parameter controls the overall refinement strength:

- **0.001**: Very gentle, preserves structure closely (~2.5Å initial noise)
- **0.01**: Gentle refinement (~25Å initial noise)  
- **0.1**: Moderate refinement (~250Å initial noise)
- **1.0**: Full refinement (~2500Å initial noise, default Boltz behavior)

## Output Files

Selective refinement produces the same output structure as standard prediction, plus optional trajectory files:

### Standard Output
- `{name}_model_0.pdb/cif`: Best refined structure
- `confidence_{name}_model_0.json`: Confidence scores
- `plddt_{name}_model_0.npz`: Per-residue confidence

### Trajectory Output (with `--save_trajectory`)
- `{name}_trajectory.pdb`: Multi-model PDB with all denoising steps
- `{name}_trajectory.cif`: mmCIF format trajectory (single model)

## Best Practices

### 1. Start Conservative
Begin with low sigma scaling and moderate noise intensities:
```bash
--start_sigma_scale 0.001 --add_noise "target:1.0"
```

### 2. Use Multiple Samples
Generate multiple refined structures to assess consistency:
```bash
--diffusion_samples 3
```

### 3. Monitor Trajectories
Use `--save_trajectory` to understand refinement behavior:
```bash
--save_trajectory --output_format pdb
```

### 4. Combine Selection Types
Mix different selection methods for complex refinement tasks:
```bash
--add_noise "chain:A:1.0" --add_noise "distance:B:50:8.0:0.5"
```

### 5. Preserve Coordinate Frame
Use `--no_random_augmentation` when coordinate frame matters:
```bash
--no_random_augmentation
```

## Troubleshooting

### Issue: Structure Changes Too Much
- **Reduce** `--start_sigma_scale` (try 0.001)
- **Lower** noise intensities (0.3-0.7 range)
- **Increase** `--non_target_noise_min` to 0.2-0.3

### Issue: Structure Doesn't Change Enough
- **Increase** `--start_sigma_scale` (try 0.01-0.1)
- **Raise** noise intensities (0.8-1.5 range)
- **Check** that target selection is correct

### Issue: Refinement Takes Too Long
- **Reduce** `--sampling_steps` (try 50-100)
- **Use** smaller selection regions
- **Consider** using fewer `--diffusion_samples`

### Issue: Poor Quality Results
- **Enable** potentials with `--use_potentials`
- **Use** `--save_trajectory` to diagnose issues
- **Try** different noise intensity combinations
- **Ensure** initial structure quality is reasonable

## Technical Details

### Differential Noise Scaling
Non-targeted atoms receive reduced noise following the formula:
```
noise_scale = non_target_noise_min + non_target_noise_range * step_progress
```

This ensures non-targeted regions remain relatively stable while allowing some flexibility.

### Coordinate Loading
Initial coordinates are:
1. Loaded from PDB/mmCIF files
2. Matched to the input sequence by residue number and chain ID
3. Centered and potentially rotated (unless `--no_random_augmentation` is used)
4. Used as starting points for the denoising process

### Selection Resolution
- **Atom-based**: Individual atoms are selected
- **Residue-based**: All atoms in specified residues
- **Chain-based**: All atoms in specified chains  
- **Distance-based**: Atoms within distance cutoff (optionally extended to full residues)

## Integration with Other Features

Selective refinement works seamlessly with other Boltz-2 features:

- **MSA usage**: `--use_msa_server` for automatic MSA generation
- **Potentials**: `--use_potentials` for improved physical quality
- **Multiple samples**: `--diffusion_samples N` for ensemble generation
- **Output formats**: `--output_format pdb/mmcif` for different file types
- **Device selection**: `--devices N` for GPU acceleration

This makes selective refinement a powerful tool for structure prediction workflows.