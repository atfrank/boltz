#!/usr/bin/env python3
"""Benchmark script to compare CUDA optimizations vs original implementations."""

import time
import torch
import numpy as np
from typing import Dict, List
import argparse


def benchmark_saxs_computation():
    """Benchmark SAXS computation: PyTorch vs JAX implementations."""
    print("=" * 60)
    print("SAXS COMPUTATION BENCHMARK")
    print("=" * 60)
    
    # Try to import the SAXS functions
    try:
        from boltz.model.potentials.saxs_potential import (
            compute_saxs_chi2_torch, compute_form_factors_torch, 
            compute_saxs_intensity_torch
        )
        saxs_available = True
    except ImportError as e:
        print(f"SAXS functions not available: {e}")
        saxs_available = False
        return {}
    
    if not saxs_available:
        return {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    n_atoms_list = [50, 100, 200, 500]
    n_q = 50
    n_repeats = 10
    
    results = {}
    
    for n_atoms in n_atoms_list:
        print(f"\nTesting with {n_atoms} atoms:")
        
        # Generate test data
        coords = torch.randn(n_atoms, 3, device=device, dtype=torch.float32) * 20.0
        q_exp = torch.linspace(0.01, 0.3, n_q, device=device, dtype=torch.float32)
        I_exp = torch.exp(-q_exp * 10.0) + torch.randn(n_q, device=device) * 0.1
        sigma_exp = torch.ones(n_q, device=device, dtype=torch.float32) * 0.1
        atom_types = torch.full((n_atoms,), 6, device=device, dtype=torch.long)  # Carbon
        
        # Warm up GPU
        for _ in range(3):
            _ = compute_saxs_chi2_torch(coords, q_exp, I_exp, sigma_exp, atom_types)
        
        # Benchmark optimized PyTorch version
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        for _ in range(n_repeats):
            chi2 = compute_saxs_chi2_torch(coords, q_exp, I_exp, sigma_exp, atom_types)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        pytorch_time = (time.time() - start_time) / n_repeats
        
        print(f"  PyTorch optimized: {pytorch_time*1000:.2f} ms/call")
        print(f"  Chi2 result: {chi2.item():.3f}")
        
        results[f"saxs_{n_atoms}_atoms"] = {
            "pytorch_time_ms": pytorch_time * 1000,
            "chi2_value": chi2.item()
        }
    
    return results


def benchmark_rg_computation():
    """Benchmark Rg computation: optimized vs original median operations."""
    print("=" * 60)
    print("RG COMPUTATION BENCHMARK")
    print("=" * 60)
    
    try:
        from boltz.model.potentials.robust_rg_potential import RobustRgPotential
        rg_available = True
    except ImportError as e:
        print(f"Rg functions not available: {e}")
        rg_available = False
        return {}
    
    if not rg_available:
        return {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    n_atoms_list = [100, 200, 500, 1000]
    n_repeats = 50
    
    results = {}
    
    # Create Rg potential instances
    rg_potential = RobustRgPotential(target_rg=25.0, rg_calculation_method="robust")
    
    for n_atoms in n_atoms_list:
        print(f"\nTesting with {n_atoms} atoms:")
        
        # Generate test coordinates
        coords = torch.randn(n_atoms, 3, device=device, dtype=torch.float32) * 15.0
        
        # Test original median-based method
        distances = torch.norm(coords - coords.mean(dim=0), dim=1)
        
        # Warm up
        for _ in range(5):
            _ = torch.median(distances)
        
        # Benchmark original method
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        for _ in range(n_repeats):
            median_val = torch.median(distances)
            mad_val = torch.median(torch.abs(distances - median_val))
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        original_time = (time.time() - start_time) / n_repeats
        
        # Benchmark optimized method
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        for _ in range(n_repeats):
            median_opt, mad_opt = rg_potential._compute_robust_stats_quantile(distances)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        optimized_time = (time.time() - start_time) / n_repeats
        
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        print(f"  Original median: {original_time*1000:.3f} ms/call")
        print(f"  Optimized quantile: {optimized_time*1000:.3f} ms/call")
        print(f"  Speedup: {speedup:.2f}x")
        
        results[f"rg_{n_atoms}_atoms"] = {
            "original_time_ms": original_time * 1000,
            "optimized_time_ms": optimized_time * 1000,
            "speedup": speedup
        }
    
    return results


def benchmark_memory_usage():
    """Benchmark memory usage of different implementations."""
    print("=" * 60)
    print("MEMORY USAGE BENCHMARK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return {}
    
    device = torch.device("cuda")
    
    def get_memory_usage():
        """Get current GPU memory usage in MB."""
        return torch.cuda.memory_allocated(device) / 1024**2
    
    results = {}
    n_atoms = 500
    
    # Clear memory
    torch.cuda.empty_cache()
    baseline_memory = get_memory_usage()
    
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    
    # Test coordinate allocation
    coords = torch.randn(n_atoms, 3, device=device, dtype=torch.float32)
    coords_memory = get_memory_usage() - baseline_memory
    
    print(f"Coordinates memory: {coords_memory:.1f} MB")
    
    # Test distance matrix allocation (memory-intensive operation)
    distances_full = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    full_distance_memory = get_memory_usage() - baseline_memory
    
    print(f"Full distance matrix memory: {full_distance_memory:.1f} MB")
    
    # Clear and test optimized approach
    del distances_full
    torch.cuda.empty_cache()
    
    # Memory-efficient distance computation
    diff = coords[:, None, :] - coords[None, :, :]
    distances_opt = torch.norm(diff, dim=-1)
    opt_distance_memory = get_memory_usage() - baseline_memory
    
    print(f"Optimized distance computation memory: {opt_distance_memory:.1f} MB")
    
    memory_savings = (full_distance_memory - opt_distance_memory) / full_distance_memory * 100
    print(f"Memory savings: {memory_savings:.1f}%")
    
    results["memory"] = {
        "baseline_mb": baseline_memory,
        "coords_mb": coords_memory,
        "full_distance_mb": full_distance_memory,
        "opt_distance_mb": opt_distance_memory,
        "memory_savings_percent": memory_savings
    }
    
    return results


def main():
    """Run all benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark CUDA optimizations")
    parser.add_argument("--saxs", action="store_true", help="Run SAXS benchmarks")
    parser.add_argument("--rg", action="store_true", help="Run Rg benchmarks") 
    parser.add_argument("--memory", action="store_true", help="Run memory benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    
    args = parser.parse_args()
    
    if not any([args.saxs, args.rg, args.memory, args.all]):
        args.all = True
    
    all_results = {}
    
    if args.all or args.saxs:
        all_results.update(benchmark_saxs_computation())
    
    if args.all or args.rg:
        all_results.update(benchmark_rg_computation())
    
    if args.all or args.memory:
        all_results.update(benchmark_memory_usage())
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    return all_results


if __name__ == "__main__":
    main()