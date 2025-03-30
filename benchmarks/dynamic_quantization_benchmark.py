# benchmarks/dynamic_quantization_benchmark.py
import torch
import time
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import EdgeFormerModel
from src.optimization import DynamicQuantizer, measure_model_size

def benchmark_model(model, input_ids, num_runs=5):
    """Benchmark a model's inference time."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(input_ids)
        times.append(time.time() - start)
    
    return np.mean(times)

def run_benchmarks(model_sizes, num_runs=5):
    """Run benchmarks for different model sizes."""
    results = {}
    
    for hidden_size in model_sizes:
        print(f"\nBenchmarking model with hidden_size={hidden_size}...")
        results[hidden_size] = {}
        
        # Create model
        config = EdgeFormerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4
        )
        
        model = EdgeFormerModel(config)
        
        # Input for inference
        input_ids = torch.randint(0, 1000, (1, 32))
        
        # Measure FP32 model
        fp32_size = measure_model_size(model)
        fp32_time = benchmark_model(model, input_ids, num_runs)
        
        results[hidden_size]['fp32'] = {
            'size_mb': fp32_size,
            'time': fp32_time
        }
        
        print(f"  FP32: {fp32_size:.2f} MB, {fp32_time*1000:.2f} ms")
        
        # INT8 quantization
        try:
            print("  Applying INT8 quantization...")
            int8_model = DynamicQuantizer.quantize_model_int8(model)
            
            int8_size = measure_model_size(int8_model)
            int8_time = benchmark_model(int8_model, input_ids, num_runs)
            
            results[hidden_size]['int8'] = {
                'size_mb': int8_size,
                'time': int8_time,
                'compression': fp32_size / int8_size if int8_size > 0 else 0,
                'speed_ratio': fp32_time / int8_time if int8_time > 0 else 0
            }
            
            print(f"  INT8: {int8_size:.2f} MB ({results[hidden_size]['int8']['compression']:.2f}x), "
                  f"{int8_time*1000:.2f} ms ({results[hidden_size]['int8']['speed_ratio']:.2f}x)")
        except Exception as e:
            print(f"  Error with INT8 quantization: {str(e)}")
            results[hidden_size]['int8'] = None
        
        # INT4 quantization
        try:
            print("  Applying INT4 quantization...")
            int4_model = DynamicQuantizer.quantize_model_int4(model)
            
            int4_size = measure_model_size(int4_model)
            int4_time = benchmark_model(int4_model, input_ids, num_runs)
            
            results[hidden_size]['int4'] = {
                'size_mb': int4_size,
                'time': int4_time,
                'compression': fp32_size / int4_size if int4_size > 0 else 0,
                'speed_ratio': fp32_time / int4_time if int4_time > 0 else 0
            }
            
            print(f"  INT4: {int4_size:.2f} MB ({results[hidden_size]['int4']['compression']:.2f}x), "
                  f"{int4_time*1000:.2f} ms ({results[hidden_size]['int4']['speed_ratio']:.2f}x)")
        except Exception as e:
            print(f"  Error with INT4 quantization: {str(e)}")
            results[hidden_size]['int4'] = None
        
        # Clear memory
        del model
        if 'int8_model' in locals():
            del int8_model
        if 'int4_model' in locals():
            del int4_model
        gc.collect()
    
    return results

def plot_results(results, output_dir="."):
    """Plot benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_sizes = list(results.keys())
    
    # Memory usage plot
    plt.figure(figsize=(10, 6))
    
    fp32_sizes = [results[size]['fp32']['size_mb'] for size in model_sizes]
    
    int8_sizes = []
    for size in model_sizes:
        if results[size]['int8'] is not None:
            int8_sizes.append(results[size]['int8']['size_mb'])
        else:
            int8_sizes.append(None)
    
    int4_sizes = []
    for size in model_sizes:
        if results[size]['int4'] is not None:
            int4_sizes.append(results[size]['int4']['size_mb'])
        else:
            int4_sizes.append(None)
    
    plt.plot(model_sizes, fp32_sizes, 'o-', label='FP32')
    if not all(s is None for s in int8_sizes):
        valid_points = [(size, size_mb) for size, size_mb in zip(model_sizes, int8_sizes) if size_mb is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, 's-', label='INT8')
    
    if not all(s is None for s in int4_sizes):
        valid_points = [(size, size_mb) for size, size_mb in zip(model_sizes, int4_sizes) if size_mb is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, 'd-', label='INT4')
    
    plt.title('Memory Usage vs Model Size')
    plt.xlabel('Hidden Size')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/memory_usage.png")
    
    # Inference time plot
    plt.figure(figsize=(10, 6))
    
    fp32_times = [results[size]['fp32']['time'] * 1000 for size in model_sizes]  # Convert to ms
    
    int8_times = []
    for size in model_sizes:
        if results[size]['int8'] is not None:
            int8_times.append(results[size]['int8']['time'] * 1000)  # Convert to ms
        else:
            int8_times.append(None)
    
    int4_times = []
    for size in model_sizes:
        if results[size]['int4'] is not None:
            int4_times.append(results[size]['int4']['time'] * 1000)  # Convert to ms
        else:
            int4_times.append(None)
    
    plt.plot(model_sizes, fp32_times, 'o-', label='FP32')
    if not all(t is None for t in int8_times):
        valid_points = [(size, time) for size, time in zip(model_sizes, int8_times) if time is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, 's-', label='INT8')
    
    if not all(t is None for t in int4_times):
        valid_points = [(size, time) for size, time in zip(model_sizes, int4_times) if time is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, 'd-', label='INT4')
    
    plt.title('Inference Time vs Model Size')
    plt.xlabel('Hidden Size')
    plt.ylabel('Inference Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/inference_time.png")
    
    # Compression ratio plot
    plt.figure(figsize=(10, 6))
    
    int8_compression = []
    for size in model_sizes:
        if results[size]['int8'] is not None:
            int8_compression.append(results[size]['int8']['compression'])
        else:
            int8_compression.append(None)
    
    int4_compression = []
    for size in model_sizes:
        if results[size]['int4'] is not None:
            int4_compression.append(results[size]['int4']['compression'])
        else:
            int4_compression.append(None)
    
    if not all(c is None for c in int8_compression):
        valid_points = [(size, comp) for size, comp in zip(model_sizes, int8_compression) if comp is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, 's-', label='INT8')
    
    if not all(c is None for c in int4_compression):
        valid_points = [(size, comp) for size, comp in zip(model_sizes, int4_compression) if comp is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, 'd-', label='INT4')
    
    plt.title('Compression Ratio vs Model Size')
    plt.xlabel('Hidden Size')
    plt.ylabel('Compression Ratio (x)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/compression_ratio.png")

def save_results_table(results, output_dir="."):
    """Save benchmark results as a markdown table."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/benchmark_results.md", "w") as f:
        f.write("# EdgeFormer Quantization Benchmark Results\n\n")
        
        f.write("## Memory Usage (MB)\n\n")
        f.write("| Model Size | FP32 | INT8 | INT4 | INT8 Compression | INT4 Compression |\n")
        f.write("|------------|------|------|------|------------------|------------------|\n")
        
        for size in results:
            fp32_size = results[size]['fp32']['size_mb']
            
            int8_size = "N/A"
            int8_compression = "N/A"
            if results[size]['int8'] is not None:
                int8_size = f"{results[size]['int8']['size_mb']:.2f}"
                int8_compression = f"{results[size]['int8']['compression']:.2f}x"
            
            int4_size = "N/A"
            int4_compression = "N/A"
            if results[size]['int4'] is not None:
                int4_size = f"{results[size]['int4']['size_mb']:.2f}"
                int4_compression = f"{results[size]['int4']['compression']:.2f}x"
            
            f.write(f"| {size} | {fp32_size:.2f} | {int8_size} | {int4_size} | {int8_compression} | {int4_compression} |\n")
        
        f.write("\n\n")
        
        f.write("## Inference Time (ms)\n\n")
        f.write("| Model Size | FP32 | INT8 | INT4 | INT8 Speed Ratio | INT4 Speed Ratio |\n")
        f.write("|------------|------|------|------|------------------|------------------|\n")
        
        for size in results:
            fp32_time = results[size]['fp32']['time'] * 1000  # Convert to ms
            
            int8_time = "N/A"
            int8_speed = "N/A"
            if results[size]['int8'] is not None:
                int8_time = f"{results[size]['int8']['time'] * 1000:.2f}"
                int8_speed = f"{results[size]['int8']['speed_ratio']:.2f}x"
            
            int4_time = "N/A"
            int4_speed = "N/A"
            if results[size]['int4'] is not None:
                int4_time = f"{results[size]['int4']['time'] * 1000:.2f}"
                int4_speed = f"{results[size]['int4']['speed_ratio']:.2f}x"
            
            f.write(f"| {size} | {fp32_time:.2f} | {int8_time} | {int4_time} | {int8_speed} | {int4_speed} |\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark EdgeFormer quantization')
    parser.add_argument('--model_sizes', nargs='+', type=int, default=[64, 128, 256, 512])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default="benchmark_results")
    
    args = parser.parse_args()
    
    print(f"Running benchmark with model sizes: {args.model_sizes}")
    results = run_benchmarks(args.model_sizes, args.runs)
    
    print(f"\nGenerating plots in {args.output_dir}")
    plot_results(results, args.output_dir)
    
    print(f"Saving results table in {args.output_dir}")
    save_results_table(results, args.output_dir)
    
    print("\nBenchmark complete!")