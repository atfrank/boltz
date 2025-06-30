"""Global optimization flags for CUDA acceleration.

This module provides a centralized way to control optimization behavior
across different potential implementations.
"""

from typing import Optional
from boltz.data.types import InferenceOptions

# Global inference options set by main.py
INFERENCE_OPTIONS: Optional[InferenceOptions] = None

def get_optimization_flags() -> InferenceOptions:
    """Get current optimization flags, with safe defaults."""
    if INFERENCE_OPTIONS is None:
        # Default to optimizations enabled
        from boltz.data.types import InferenceOptions
        return InferenceOptions(
            use_cuda_optimizations=True,
            use_vectorized_saxs=True,
            use_optimized_rg=True,
            use_batch_finite_diff=True,
            enable_mixed_precision=True,
        )
    return INFERENCE_OPTIONS

def use_cuda_optimizations() -> bool:
    """Check if CUDA optimizations are enabled."""
    return get_optimization_flags().use_cuda_optimizations

def use_vectorized_saxs() -> bool:
    """Check if vectorized SAXS computation is enabled."""
    return get_optimization_flags().use_vectorized_saxs

def use_optimized_rg() -> bool:
    """Check if optimized Rg calculations are enabled."""
    return get_optimization_flags().use_optimized_rg

def use_batch_finite_diff() -> bool:
    """Check if batch finite difference gradients are enabled."""
    return get_optimization_flags().use_batch_finite_diff

def enable_mixed_precision() -> bool:
    """Check if mixed precision optimizations are enabled."""
    return get_optimization_flags().enable_mixed_precision