# benchmarks/quantization_benchmark.py
import torch
import time
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import EdgeFormerModel
from src.optimization.quantization import quantize_edgeformer

def measure_memory(model):
    """Measure memory usage of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params, total_bytes

def benchmark_quantization(model_sizes, bits_options, num_runs=5):
    """Benchmark model sizes with different quantization levels."""
    results = {}
    
    for hidden_size in model_sizes:
        results[hidden_size] = {}
        
        # Create a model
        config = EdgeFormerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4
        )
        
        model = EdgeFormerModel(config)
        model.eval()
        
        # Baseline measurements (FP32)
        fp32_params, fp32_bytes = measure_memory(model)
        
        # Inference test
        input_ids = torch.randint(0, 1000, (1, 32))
        
        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        
        # Benchmark FP32
        fp32_times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(input_ids)
            fp32_times.append(time.time() - start)
        
        results[hidden_size]['fp32'] = {
            'params': fp32_params,
            'bytes': fp32_bytes,
            'time': np.mean(fp32_times)
        }
        
        # Benchmark different quantization options
        for bits in bits_options:
            # Quantize the model
            q_model = quantize_edgeformer(model, bits=bits)
            q_model.eval()
            
            # Measure memory
            q_params, q_bytes = measure_memory(q_model)
            
            # Benchmark inference
            q_times = []
            for _ in range(num_runs):
                start = time.time()
                with torch.no_grad():
                    _ = q_model(input_ids)
                q_times.append(time.time() - start)
            
            results[hidden_size][f'int{bits}'] = {
                'params': q_params,
                'bytes': q_bytes,
                'time': np.mean(q_times)
            }
    
    return results

def plot_results(results):
    """Plot benchmark results."""
    model_sizes = list(results.keys())
    metrics = ['bytes', 'time']
    bits_options = ['fp32', 'int8', 'int4']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, metric in enumerate(metrics):
        for bits in bits_options:
            sizes = model_sizes
            values = [results[size][bits][metric] for size in sizes]
            
            if metric == 'bytes':
                # Convert to MB
                values = [v / (1024 * 1024) for v in values]
                axes[i].set_ylabel('Memory (MB)')
            else:
                axes[i].set_ylabel('Inference Time (s)')
            
            axes[i].plot(sizes, values, marker='o', label=bits)
        
        axes[i].set_xlabel('Hidden Size')
        axes[i].set_title(f'Model {metric.capitalize()} vs Hidden Size')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('quantization_benchmark.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark quantization options')
    parser.add_argument('--model_sizes', nargs='+', type=int, default=[128, 256, 512, 768])
    parser.add_argument('--bits', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--runs', type=int, default=5)
    
    args = parser.parse_args()
    
    results = benchmark_quantization(args.model_sizes, args.bits, args.runs)
    plot_results(results)
    
    # Print summary
    print("Quantization Benchmark Results:")
    print("==============================")
    
    for size in args.model_sizes:
        print(f"\nModel hidden_size={size}:")
        fp32_bytes = results[size]['fp32']['bytes']
        fp32_time = results[size]['fp32']['time']
        
        for bits in ['int8', 'int4']:
            q_bytes = results[size][bits]['bytes']
            q_time = results[size][bits]['time']
            
            compression = fp32_bytes / q_bytes
            slowdown = q_time / fp32_time
            
            print(f"  {bits}: {compression:.2f}x compression, {slowdown:.2f}x inference slowdown")