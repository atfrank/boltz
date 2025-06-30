# Raw Coordinate Guidance

## Overview

Raw coordinate guidance is an advanced feature that applies shape restraints directly to the noisy coordinates **before** neural network denoising during the diffusion process. This provides an additional layer of control during structure generation, particularly useful in the early, high-noise stages.

## How It Works

### Standard vs Raw Guidance

1. **Standard Rg Guidance** (default):
   - Applied to denoised coordinates after neural network processing
   - Works throughout the entire diffusion process
   - More stable and reliable

2. **Raw Coordinate Guidance** (optional):
   - Applied to raw noisy coordinates before neural network denoising
   - Most effective at early diffusion steps when noise is lower
   - Provides additional "steering" during structure formation
   - Uses adaptive strength based on noise level

### Technical Implementation

The raw guidance implementation:
- Computes gradients using finite differences when autograd fails
- Adapts guidance strength based on noise level (weaker at high noise)
- Gracefully handles mixed precision and gradient computation challenges
- Only applies forces when meaningful (noise_factor > 0.05)

## Configuration

### Enabling Raw Guidance

Add the `raw_guidance_weight` parameter to your Rg guidance configuration:

```yaml
guidance:
  rg:
    target_rg: 24.5
    force_constant: 100.0
    raw_guidance_weight: 50.0  # Enable raw coordinate guidance
```

### Disabling Raw Guidance (Default)

Set `raw_guidance_weight` to 0 or omit it entirely:

```yaml
guidance:
  rg:
    target_rg: 24.5
    force_constant: 100.0
    raw_guidance_weight: 0.0  # Disabled (default)
```

## Parameters

- **raw_guidance_weight**: Controls the strength of raw coordinate guidance
  - Range: 0.0 to 100.0 (typically 10.0 to 50.0)
  - Default: 0.0 (disabled)
  - Higher values = stronger guidance on raw coordinates

## Adaptive Behavior

The actual guidance strength is computed as:
```
effective_strength = raw_guidance_weight * noise_factor * scaling_factor
```

Where:
- `noise_factor = max(0, 1 - sigma/max_sigma)` (0 at high noise, 1 at low noise)
- `scaling_factor = 0.01` (for stability)

This ensures:
- Minimal interference at high noise levels (early diffusion)
- Stronger guidance as structure becomes more defined
- Automatic adaptation to the diffusion schedule

## When to Use Raw Guidance

### Recommended Use Cases:
- Difficult targets with specific shape requirements
- When standard guidance alone is insufficient
- Research into diffusion dynamics
- Exploring alternative guidance strategies

### Not Recommended For:
- Initial experiments (use standard guidance first)
- Systems where standard guidance works well
- Production runs without prior testing

## Performance Considerations

Raw guidance uses finite difference gradient computation when autograd fails:
- Slightly slower than standard guidance
- Computational cost scales with system size
- Most impact at low-noise steps (later in diffusion)

## Example Configurations

### Conservative (Recommended Starting Point)
```yaml
guidance:
  rg:
    target_rg: 25.0
    force_constant: 100.0
    raw_guidance_weight: 10.0  # Light raw guidance
```

### Moderate
```yaml
guidance:
  rg:
    target_rg: 25.0
    force_constant: 100.0
    raw_guidance_weight: 30.0  # Moderate raw guidance
```

### Aggressive
```yaml
guidance:
  rg:
    target_rg: 25.0
    force_constant: 100.0
    raw_guidance_weight: 50.0  # Strong raw guidance
```

## Monitoring

When raw guidance is active, you'll see log messages like:
```
Raw coordinate Rg guidance activated with weight 50.0
Raw guidance step 0: Using finite difference gradients
Raw guidance step 5: sigma=5.9 Rg=92.0Å target=24.5Å error=67.5Å noise_factor=0.993 strength=0.496 max_force=0.256
```

## Troubleshooting

If raw guidance isn't working as expected:

1. **Check noise levels**: Raw guidance only activates when noise is low enough
2. **Verify configuration**: Ensure `raw_guidance_weight > 0`
3. **Monitor logs**: Look for "Raw guidance" messages in output
4. **Start conservative**: Begin with `raw_guidance_weight: 10.0`

## Technical Details

The implementation handles several challenges:
- Mixed precision (bfloat16) compatibility
- Gradient computation in no_grad contexts
- Batch dimension handling
- Numerical stability with finite differences

For implementation details, see:
- `src/boltz/model/potentials/robust_rg_wrapper.py`
- `src/boltz/model/potentials/robust_rg_potential.py`