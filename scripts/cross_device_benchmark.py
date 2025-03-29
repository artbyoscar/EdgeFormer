#!/usr/bin/env python
# cross_device_benchmark.py

import argparse
import os
import json
import time
import torch
import numpy as np
import sys
import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def load_device_profile(profile_path):
    """Load a device profile from a JSON file."""
    with open(profile_path, 'r') as f:
        return json.load(f)

def get_current_device_name(device_profiles_dir):
    """Determine which device profile matches the current device."""
    # Get basic system info to match
    import platform
    import psutil
    
    current_info = {
        "os": platform.system(),
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    # Check all profiles
    for profile_file in os.listdir(device_profiles_dir):
        if profile_file.endswith('.json'):
            profile_path = os.path.join(device_profiles_dir, profile_file)
            profile = load_device_profile(profile_path)
            
            # Check if key system properties match
            if (profile.get("os") == current_info["os"] and
                profile.get("cpu") == current_info["cpu"] and
                profile.get("cpu_cores") == current_info["cpu_cores"] and
                abs(profile.get("ram_total_gb", 0) - current_info["ram_total_gb"]) < 1.0):
                
                return profile_file.split('.')[0]  # Return device name without extension
    
    return "unknown_device"

def run_benchmark(model, seq_length, batch_size=1, num_runs=5):
    """Run inference benchmark on the model."""
    device = next(model.parameters()).device
    
    # Create random input
    input_ids = torch.randint(0, 10000, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Warm-up run
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
    
    # Benchmark runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            _ = model(input_ids, attention_mask=attention_mask)
            
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            else:
                import psutil
                memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024**2)  # MB
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    avg_memory = np.mean(memory_usage)
    
    return {
        "avg_time_seconds": avg_time,
        "tokens_per_second": seq_length / avg_time,
        "avg_memory_mb": avg_memory
    }

def main():
    parser = argparse.ArgumentParser(description="Run cross-device benchmarks for EdgeFormer")
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium", "large"], 
                        help="Size of the model to benchmark")
    parser.add_argument("--device_profiles", type=str, required=True, 
                        help="Directory containing device profiles")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save benchmark results")
    parser.add_argument("--sequence_lengths", type=str, default="128,512,1024,2048,4096", 
                        help="Comma-separated list of sequence lengths to test")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Identify current device
    device_name = get_current_device_name(args.device_profiles)
    print(f"Running benchmarks on device: {device_name}")
    
    # Determine model configuration based on size
    hidden_sizes = {
        "tiny": 128,
        "small": 256,
        "medium": 512,
        "large": 768
    }
    
    num_layers = {
        "tiny": 4,
        "small": 6,
        "medium": 8,
        "large": 12
    }
    
    # Initialize model
    config = EdgeFormerConfig(
        hidden_size=hidden_sizes[args.model_size],
        num_hidden_layers=num_layers[args.model_size],
        num_attention_heads=hidden_sizes[args.model_size] // 64,
        intermediate_size=hidden_sizes[args.model_size] * 4,
        max_position_embeddings=8192
    )
    
    model = EdgeFormer(config)
    
    # Move model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Parse sequence lengths
    seq_lengths = [int(x) for x in args.sequence_lengths.split(',')]
    
    # Run benchmarks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {
        "device": device_name,
        "model_size": args.model_size,
        "timestamp": timestamp,
        "results": {}
    }
    
    for seq_len in seq_lengths:
        print(f"Benchmarking sequence length: {seq_len}")
        try:
            benchmark_result = run_benchmark(model, seq_len)
            results["results"][str(seq_len)] = benchmark_result
            print(f"  Time: {benchmark_result['avg_time_seconds']:.4f}s, "
                  f"Speed: {benchmark_result['tokens_per_second']:.2f} tokens/s, "
                  f"Memory: {benchmark_result['avg_memory_mb']:.2f} MB")
        except Exception as e:
            print(f"Error benchmarking sequence length {seq_len}: {e}")
            results["results"][str(seq_len)] = {"error": str(e)}
    
    # Save results
    result_file = os.path.join(args.output_dir, f"{device_name}_{args.model_size}_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results saved to {result_file}")

if __name__ == "__main__":
    main()