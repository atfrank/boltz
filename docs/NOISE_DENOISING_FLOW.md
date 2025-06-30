# Boltz Noise and Denoising Flow Explanation

## Overview

Boltz uses a diffusion-based approach for structure prediction where the model learns to gradually denoise random coordinates into meaningful protein/RNA structures. Understanding this process is crucial for interpreting trajectories and guidance behavior.

## The Two Coordinate Systems

During diffusion, Boltz maintains **two distinct coordinate systems**:

### 1. Raw Diffusion Coordinates (`atom_coords`)
- **What they are**: The coordinates being iteratively denoised through the diffusion process
- **Initial state**: Pure noise (very large, random coordinates - thousands of Å)
- **Evolution**: Gradually become more structured through the diffusion schedule
- **Saved in**: Standard trajectory files (e.g., `*_trajectory.pdb`)
- **Characteristics**: 
  - Start with Rg values of thousands of Å
  - Gradually decrease as structure emerges
  - Final values represent the raw diffusion output

### 2. Denoised Coordinates (`atom_coords_denoised`)
- **What they are**: Clean structural predictions from the neural network at each diffusion step
- **Source**: Output of `preconditioned_network_forward()` - the neural network's "best guess" at the final structure
- **Characteristics**:
  - Always reasonable structures (Rg ~20-50Å for typical proteins/RNA)
  - Consistent quality throughout diffusion
  - What steering potentials actually operate on
- **Saved in**: Denoised trajectory files (e.g., `*_trajectory_denoised.pdb`)

## The Diffusion Process Step-by-Step

```
Diffusion Step N:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Start with noisy coordinates (atom_coords)                  │
│    • Very spread out, Rg = thousands of Å                     │
│    • Gets less noisy with each step                           │
│                                                               │
│ 2. Neural network denoises coordinates                        │
│    • Input: noisy coordinates + noise level (sigma)           │
│    • Output: clean structure prediction (atom_coords_denoised)│
│    • Rg ~20-50Å (reasonable structure)                       │
│                                                               │
│ 3. Apply steering potentials (if enabled)                     │
│    • Operate on atom_coords_denoised                         │
│    • Compute gradients to improve structure                   │
│    • Multiple gradient descent steps                          │
│                                                               │
│ 4. Update raw coordinates for next step                       │
│    • Combine denoised prediction with noise schedule          │
│    • Apply guidance updates                                   │
│    • Result becomes input for next diffusion step             │
└─────────────────────────────────────────────────────────────────┘
```

## Why Two Coordinate Systems?

### 1. **Diffusion Mathematics**
- The diffusion process requires maintaining noisy coordinates that gradually improve
- The neural network learns to predict clean structures from noisy inputs
- These serve different mathematical purposes in the diffusion framework

### 2. **Steering Effectiveness**
- Potentials work better on reasonable structures than extremely noisy coordinates
- Denoised coordinates provide meaningful geometry for energy calculations
- Gradients are more informative when applied to structured coordinates

### 3. **Interpretability**
- Raw coordinates show the actual diffusion trajectory
- Denoised coordinates show what the model "thinks" the structure should be
- Both provide valuable insights into the prediction process

## Trajectory Files Generated

With the new functionality, Boltz generates two trajectory files:

### Raw Trajectory (`*_trajectory.pdb`)
```
MODEL 1   # Initial: Very spread out (Rg ~3000Å)
...
MODEL 25  # Middle: Partially structured (Rg ~100Å)  
...
MODEL 50  # Final: Converged structure (Rg ~25Å)
```

### Denoised Trajectory (`*_trajectory_denoised.pdb`)
```
MODEL 1   # Initial: Reasonable structure (Rg ~22Å)
...
MODEL 25  # Middle: Refined structure (Rg ~21Å)
...
MODEL 50  # Final: Final structure (Rg ~21Å)
```

## Guidance and Steering

### Where Potentials Operate
- **Input**: `atom_coords_denoised` (clean neural network predictions)
- **Process**: Multiple gradient descent steps to optimize structure
- **Output**: Improved coordinates fed back into diffusion process

### Why Rg Logging Differs from Raw Trajectory
- **Rg logging**: Measures denoised coordinates (~21Å consistently)
- **Raw trajectory**: Shows actual diffusion evolution (3000Å → 21Å)
- **This is expected behavior**: Different coordinate systems serve different purposes

## Noise Schedule

The noise schedule controls how much noise is added at each diffusion step:

```python
# High noise (early steps)
sigma_max = 160.0  # Very noisy
...
# Low noise (final steps)  
sigma_min = 0.4    # Almost clean
```

- **Early steps**: Heavy noise, large structural changes possible
- **Late steps**: Fine-tuning, small adjustments only
- **Neural network**: Learns to denoise at all noise levels

## Practical Implications

### For Structure Analysis
1. **Use denoised trajectory** to understand what the model predicts at each step
2. **Use raw trajectory** to see the actual diffusion evolution
3. **Compare both** to understand guidance effectiveness

### For Guidance Development
1. **Potentials operate on denoised coordinates** - design accordingly
2. **Reasonable structures throughout** - no need to handle extreme geometries
3. **Gradients affect raw coordinates** - guidance influences the next diffusion step

### For Debugging
1. **Consistent denoised Rg** suggests limited guidance effectiveness
2. **Evolving raw Rg** shows normal diffusion progression
3. **Large differences** between trajectories indicate strong guidance influence

## Key Takeaways

1. **Two coordinate systems** serve different purposes in diffusion
2. **Denoised coordinates** are what you should analyze for structure quality
3. **Raw coordinates** show the mathematical diffusion process
4. **Guidance operates** on clean structures, not noisy ones
5. **Both trajectories** provide complementary information about the prediction process

This dual-trajectory system provides complete insight into both the diffusion mathematics and the structural biology, enabling better understanding and debugging of the prediction process.