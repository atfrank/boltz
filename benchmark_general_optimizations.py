#!/usr/bin/env python3
"""Benchmark script for general codebase optimizations beyond SAXS/Rg."""

import time
import torch
import numpy as np
from typing import Dict, List
import argparse
import gc


def benchmark_attention_optimizations():
    """Benchmark attention mechanism memory optimizations."""
    print("=" * 60)
    print("ATTENTION MECHANISM OPTIMIZATION BENCHMARK")
    print("=" * 60)
    
    try:
        from boltz.model.layers.attentionv2 import AttentionPairBias
        from boltz.model.potentials.optimizations import INFERENCE_OPTIONS
        attention_available = True
    except ImportError as e:
        print(f"Attention modules not available: {e}")
        attention_available = False
        return {}
    
    if not attention_available:
        return {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 2
    seq_lens = [64, 128, 256, 512]
    c_s = 384
    c_z = 128
    num_heads = 12
    n_repeats = 5
    
    results = {}
    
    for seq_len in seq_lens:
        print(f"\nTesting attention with sequence length {seq_len}:")
        
        # Create attention module
        attention = AttentionPairBias(
            c_s=c_s,
            c_z=c_z,
            num_heads=num_heads,
            compute_pair_bias=True
        ).to(device)
        
        # Generate test data
        s = torch.randn(batch_size, seq_len, c_s, device=device, dtype=torch.float16)
        z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device, dtype=torch.float16)
        mask = torch.ones(batch_size, seq_len, device=device)
        
        # Warm up
        for _ in range(2):
            with torch.no_grad():
                _ = attention(s, z, mask)
        
        # Clear memory
        torch.cuda.empty_cache() if device.type == "cuda" else None
        
        # Measure memory before
        if device.type == "cuda":
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated(device) / 1024**2
        
        # Benchmark
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        for _ in range(n_repeats):
            with torch.no_grad():
                output = attention(s, z, mask)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_repeats
        
        # Measure memory after
        if device.type == "cuda":
            mem_after = torch.cuda.memory_allocated(device) / 1024**2
            mem_used = mem_after - mem_before
        else:
            mem_used = 0
        
        print(f"  Sequence length {seq_len}: {avg_time*1000:.2f} ms/call")
        print(f"  Memory usage: {mem_used:.1f} MB")
        print(f"  Output shape: {output.shape}")
        
        results[f"attention_seq_{seq_len}"] = {
            "time_ms": avg_time * 1000,
            "memory_mb": mem_used,
            "output_shape": list(output.shape)
        }
        
        # Clean up
        del output, s, z, mask, attention
        torch.cuda.empty_cache() if device.type == "cuda" else None
    
    return results


def benchmark_data_loading_optimizations():
    """Benchmark data loading optimizations."""
    print("=" * 60)
    print("DATA LOADING OPTIMIZATION BENCHMARK")
    print("=" * 60)
    
    try:
        from boltz.data.module.inferencev2 import collate, collate_optimized
        collate_available = True
    except ImportError as e:
        print(f"Data loading modules not available: {e}")
        collate_available = False
        return {}
    
    if not collate_available:
        return {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    batch_sizes = [1, 2, 4, 8]
    seq_len = 128
    feature_dim = 256
    n_repeats = 10
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting data collation with batch size {batch_size}:")
        
        # Generate test data
        test_data = []
        for i in range(batch_size):
            sample = {
                "token_pad_mask": torch.ones(seq_len, dtype=torch.bool, device=device),
                "features": torch.randn(seq_len, feature_dim, device=device),
                "coords": torch.randn(seq_len, 3, device=device),
                "distances": torch.randn(seq_len, seq_len, device=device),
                "record": f"sample_{i}",  # Non-tensor data
                "structure": {"test": "data"}  # Non-tensor data
            }
            test_data.append(sample)
        
        # Benchmark original collate
        start_time = time.time()
        for _ in range(n_repeats):
            # Temporarily disable optimizations for comparison
            from boltz.model.potentials.optimizations import INFERENCE_OPTIONS
            if INFERENCE_OPTIONS:
                original_cuda_opt = INFERENCE_OPTIONS.use_cuda_optimizations
                INFERENCE_OPTIONS.use_cuda_optimizations = False
            
            result_orig = collate(test_data)
            
            if INFERENCE_OPTIONS:
                INFERENCE_OPTIONS.use_cuda_optimizations = original_cuda_opt
        
        orig_time = (time.time() - start_time) / n_repeats
        
        # Benchmark optimized collate
        start_time = time.time()
        for _ in range(n_repeats):
            result_opt = collate_optimized(test_data)
        
        opt_time = (time.time() - start_time) / n_repeats
        
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        
        print(f"  Original collate: {orig_time*1000:.2f} ms/call")
        print(f"  Optimized collate: {opt_time*1000:.2f} ms/call")
        print(f"  Speedup: {speedup:.2f}x")
        
        results[f"collate_batch_{batch_size}"] = {
            "original_time_ms": orig_time * 1000,
            "optimized_time_ms": opt_time * 1000,
            "speedup": speedup
        }
    
    return results


def benchmark_cpu_gpu_migration():
    """Benchmark CPU to GPU migration optimizations."""
    print("=" * 60)
    print("CPU→GPU MIGRATION OPTIMIZATION BENCHMARK")
    print("=" * 60)
    
    try:
        from boltz.data.feature.featurizerv2 import compute_collinear_mask, compute_collinear_mask_gpu
        migration_available = True
    except ImportError as e:
        print(f"Migration modules not available: {e}")
        migration_available = False
        return {}
    
    if not migration_available:
        return {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("⚠️  CUDA not available - migration demo requires GPU")
        return {}
    
    # Test parameters
    n_points_list = [100, 500, 1000, 5000]
    n_repeats = 20
    
    results = {}
    
    for n_points in n_points_list:
        print(f"\nTesting with {n_points} points:")
        
        # Generate test vectors
        v1 = np.random.randn(n_points, 3).astype(np.float32)
        v2 = np.random.randn(n_points, 3).astype(np.float32)
        
        # Benchmark original CPU version
        start_time = time.time()
        for _ in range(n_repeats):
            # Force CPU computation by disabling optimization flag
            from boltz.model.potentials.optimizations import INFERENCE_OPTIONS
            if INFERENCE_OPTIONS:
                original_cuda_opt = INFERENCE_OPTIONS.use_cuda_optimizations
                INFERENCE_OPTIONS.use_cuda_optimizations = False
            
            result_cpu = compute_collinear_mask(v1, v2)
            
            if INFERENCE_OPTIONS:
                INFERENCE_OPTIONS.use_cuda_optimizations = original_cuda_opt
        
        cpu_time = (time.time() - start_time) / n_repeats
        
        # Benchmark GPU version
        start_time = time.time()
        for _ in range(n_repeats):
            result_gpu = compute_collinear_mask_gpu(v1, v2)
        
        gpu_time = (time.time() - start_time) / n_repeats
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        # Verify results match
        results_match = np.allclose(result_cpu, result_gpu)
        
        print(f"  CPU computation: {cpu_time*1000:.2f} ms/call")
        print(f"  GPU computation: {gpu_time*1000:.2f} ms/call")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Results match: {results_match}")
        
        results[f"migration_{n_points}_points"] = {
            "cpu_time_ms": cpu_time * 1000,
            "gpu_time_ms": gpu_time * 1000,
            "speedup": speedup,
            "results_match": results_match
        }
    
    return results


def benchmark_memory_patterns():
    """Benchmark memory usage patterns."""
    print("=" * 60)
    print("MEMORY PATTERN OPTIMIZATION BENCHMARK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - memory benchmark requires GPU")
        return {}
    
    device = torch.device("cuda")
    
    def get_memory_mb():
        return torch.cuda.memory_allocated(device) / 1024**2
    
    results = {}
    
    # Test tensor pre-allocation vs repeated allocation
    print("\nTesting tensor allocation patterns:")
    
    n_iterations = 100
    tensor_size = (1000, 1000)
    
    # Clear memory
    torch.cuda.empty_cache()
    baseline = get_memory_mb()
    
    # Pattern 1: Repeated allocations
    start_time = time.time()
    start_mem = get_memory_mb()
    
    for _ in range(n_iterations):
        tensor = torch.randn(tensor_size, device=device)
        result = torch.sum(tensor ** 2)
        del tensor
    
    repeated_time = time.time() - start_time
    torch.cuda.empty_cache()
    
    # Pattern 2: Pre-allocated tensor reuse
    start_time = time.time()
    start_mem = get_memory_mb()
    
    # Pre-allocate tensor
    tensor = torch.empty(tensor_size, device=device)
    
    for _ in range(n_iterations):
        tensor.normal_()  # In-place random fill
        result = torch.sum(tensor ** 2)
    
    prealloc_time = time.time() - start_time
    
    speedup = repeated_time / prealloc_time
    
    print(f"  Repeated allocations: {repeated_time*1000:.2f} ms")
    print(f"  Pre-allocated reuse: {prealloc_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    results["memory_patterns"] = {
        "repeated_alloc_ms": repeated_time * 1000,
        "prealloc_reuse_ms": prealloc_time * 1000,
        "speedup": speedup
    }
    
    # Test in-place vs out-of-place operations
    print("\nTesting in-place vs out-of-place operations:")
    
    tensor_size = (2000, 2000)
    n_ops = 50
    
    # Out-of-place operations
    torch.cuda.empty_cache()
    tensor = torch.randn(tensor_size, device=device)
    
    start_time = time.time()
    for _ in range(n_ops):
        tensor = tensor + 1.0
        tensor = tensor * 0.99
        tensor = torch.relu(tensor)
    
    out_of_place_time = time.time() - start_time
    
    # In-place operations
    torch.cuda.empty_cache()
    tensor = torch.randn(tensor_size, device=device)
    
    start_time = time.time()
    for _ in range(n_ops):
        tensor.add_(1.0)
        tensor.mul_(0.99)
        tensor.relu_()
    
    in_place_time = time.time() - start_time
    
    speedup = out_of_place_time / in_place_time
    
    print(f"  Out-of-place operations: {out_of_place_time*1000:.2f} ms")
    print(f"  In-place operations: {in_place_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    results["inplace_operations"] = {
        "out_of_place_ms": out_of_place_time * 1000,
        "in_place_ms": in_place_time * 1000,
        "speedup": speedup
    }
    
    return results


def main():
    """Run all general optimization benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark general codebase optimizations")
    parser.add_argument("--attention", action="store_true", help="Run attention benchmarks")
    parser.add_argument("--data-loading", action="store_true", help="Run data loading benchmarks")
    parser.add_argument("--cpu-gpu", action="store_true", help="Run CPU→GPU migration benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory pattern benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    
    args = parser.parse_args()
    
    if not any([args.attention, args.data_loading, args.cpu_gpu, args.memory, args.all]):
        args.all = True
    
    all_results = {}
    
    if args.all or args.attention:
        all_results.update(benchmark_attention_optimizations())
    
    if args.all or args.data_loading:
        all_results.update(benchmark_data_loading_optimizations())
    
    if args.all or args.cpu_gpu:
        all_results.update(benchmark_cpu_gpu_migration())
    
    if args.all or args.memory:
        all_results.update(benchmark_memory_patterns())
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERAL OPTIMIZATION BENCHMARK SUMMARY")
    print("=" * 60)
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        for key, value in results.items():
            if isinstance(value, float):
                if "speedup" in key:
                    print(f"  {key}: {value:.2f}x")
                else:
                    print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    return all_results


if __name__ == "__main__":
    main()