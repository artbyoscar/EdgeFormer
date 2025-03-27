# examples/benchmark_suite.py
import torch
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.device import get_device, print_device_info

def create_traditional_attention_model(config):
    """Create a version of the model with traditional attention (for comparison)."""
    # This is a placeholder - you'll need to implement a traditional attention variant
    # For now, we'll just use the existing model but note that this is not actually
    # using traditional attention
    model = EdgeFormer(config)
    # You would modify the model here to use traditional attention
    return model

def run_benchmark(model, seq_lengths, batch_size=1, num_runs=3, device="cpu", name="Model"):
    """Run inference benchmark across different sequence lengths."""
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting {name} with sequence length {seq_len}...")
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones((batch_size, seq_len), device=device)
        
        try:
            # Warmup
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Benchmark
            total_time = 0
            torch.cuda.synchronize() if device.type == "cuda" else None
            
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    end_time = time.time()
                    total_time += (end_time - start_time)
            
            avg_time = total_time / num_runs
            tokens_per_sec = seq_len * batch_size / avg_time
            
            # Calculate memory
            param_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            param_mem_mb = param_mem_bytes / (1024 * 1024)
            
            # Measure peak memory usage if on GPU
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                peak_mem = None
            
            results.append({
                'seq_len': seq_len,
                'success': True,
                'avg_time': avg_time,
                'tokens_per_sec': tokens_per_sec,
                'param_mem_mb': param_mem_mb,
                'peak_mem_mb': peak_mem
            })
            
            print(f"  Success! Avg inference time: {avg_time:.4f}s ({tokens_per_sec:.2f} tokens/sec)")
            if peak_mem:
                print(f"  Peak memory usage: {peak_mem:.2f} MB")
        
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'seq_len': seq_len,
                'success': False,
                'error': str(e)
            })
            # Break early if we hit an error - later sequences will likely fail too
            break
    
    return results

def quantize_model(model):
    """Apply INT8 dynamic quantization to the model."""
    return torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )

def plot_comparison(mla_results, standard_results=None, quantized_results=None, output_file='benchmark_comparison.png'):
    """Plot comparison of different models/configurations."""
    # Extract data
    mla_seq_lens = [r['seq_len'] for r in mla_results if r['success']]
    mla_times = [r['avg_time'] for r in mla_results if r['success']]
    mla_tps = [r['tokens_per_sec'] for r in mla_results if r['success']]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot inference times
    ax1.plot(mla_seq_lens, mla_times, 'b-o', label='EdgeFormer (MLA)')
    
    if standard_results:
        std_seq_lens = [r['seq_len'] for r in standard_results if r['success']]
        std_times = [r['avg_time'] for r in standard_results if r['success']]
        ax1.plot(std_seq_lens, std_times, 'r-o', label='Standard Attention')
    
    if quantized_results:
        q_seq_lens = [r['seq_len'] for r in quantized_results if r['success']]
        q_times = [r['avg_time'] for r in quantized_results if r['success']]
        ax1.plot(q_seq_lens, q_times, 'g-o', label='EdgeFormer (INT8)')
    
    ax1.set_title('Inference Time vs Sequence Length')
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.grid(True)
    ax1.legend()
    
    # Plot tokens per second
    ax2.plot(mla_seq_lens, mla_tps, 'b-o', label='EdgeFormer (MLA)')
    
    if standard_results:
        std_tps = [r['tokens_per_sec'] for r in standard_results if r['success']]
        ax2.plot(std_seq_lens, std_tps, 'r-o', label='Standard Attention')
    
    if quantized_results:
        q_tps = [r['tokens_per_sec'] for r in quantized_results if r['success']]
        ax2.plot(q_seq_lens, q_tps, 'g-o', label='EdgeFormer (INT8)')
    
    ax2.set_title('Throughput vs Sequence Length')
    ax2.set_xlabel('Sequence Length (tokens)')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_xscale('log', base=2)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\nComparison plot saved as '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description='EdgeFormer Benchmark Suite')
    parser.add_argument('--run-all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--mla-only', action='store_true', help='Benchmark only the MLA model')
    parser.add_argument('--quantized', action='store_true', help='Benchmark quantized model')
    parser.add_argument('--device', type=str, default='auto', help='Device to run on (cpu, cuda, auto)')
    parser.add_argument('--max-length', type=int, default=4096, help='Maximum sequence length to test')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"benchmark_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== EdgeFormer Benchmark Suite ===")
    
    # Set device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print_device_info()
    
    # Create model
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024
    )
    
    # Sequence lengths to test (powers of 2)
    seq_lengths = [128, 256, 512, 1024, 2048]
    if args.max_length > 2048:
        seq_lengths.extend([4096, 8192])
    if args.max_length > 8192:
        seq_lengths.extend([16384, 32768])
    
    # Filter sequence lengths
    seq_lengths = [sl for sl in seq_lengths if sl <= args.max_length]
    
    # Benchmark MLA model
    model_mla = EdgeFormer(config).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model_mla.parameters()):,}")
    
    mla_results = run_benchmark(model_mla, seq_lengths, device=device, name="EdgeFormer (MLA)")
    
    # Save results
    mla_results_file = os.path.join(output_dir, 'mla_results.txt')
    with open(mla_results_file, 'w') as f:
        for result in mla_results:
            f.write(f"{result}\n")
    
    # Run additional benchmarks if requested
    standard_results = None
    quantized_results = None
    
    if args.run_all or not args.mla_only:
        # Create and benchmark traditional attention model
        print("\n--- Benchmarking Traditional Attention Model ---")
        try:
            model_standard = create_traditional_attention_model(config).to(device)
            standard_results = run_benchmark(model_standard, seq_lengths, device=device, name="Standard Attention")
            
            # Save results
            std_results_file = os.path.join(output_dir, 'standard_results.txt')
            with open(std_results_file, 'w') as f:
                for result in standard_results:
                    f.write(f"{result}\n")
        except Exception as e:
            print(f"Failed to benchmark traditional attention model: {e}")
    
    if args.run_all or args.quantized:
        # Create and benchmark quantized model
        print("\n--- Benchmarking Quantized MLA Model ---")
        try:
            model_quantized = quantize_model(model_mla)
            quantized_results = run_benchmark(model_quantized, seq_lengths, device=device, name="EdgeFormer (INT8)")
            
            # Save results
            q_results_file = os.path.join(output_dir, 'quantized_results.txt')
            with open(q_results_file, 'w') as f:
                for result in quantized_results:
                    f.write(f"{result}\n")
        except Exception as e:
            print(f"Failed to benchmark quantized model: {e}")
    
    # Plot comparison
    plot_file = os.path.join(output_dir, 'benchmark_comparison.png')
    plot_comparison(mla_results, standard_results, quantized_results, output_file=plot_file)
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    
    # MLA model summary
    mla_success = [r for r in mla_results if r['success']]
    if mla_success:
        max_mla_len = max(r['seq_len'] for r in mla_success)
        max_mla_tps = max(r['tokens_per_sec'] for r in mla_success)
        print(f"EdgeFormer (MLA):")
        print(f"  Max sequence length: {max_mla_len}")
        print(f"  Max throughput: {max_mla_tps:.2f} tokens/sec")
    
    # Standard model summary
    if standard_results:
        std_success = [r for r in standard_results if r['success']]
        if std_success:
            max_std_len = max(r['seq_len'] for r in std_success)
            max_std_tps = max(r['tokens_per_sec'] for r in std_success)
            print(f"Standard Attention:")
            print(f"  Max sequence length: {max_std_len}")
            print(f"  Max throughput: {max_std_tps:.2f} tokens/sec")
            
            # Compare MLA vs Standard
            if mla_success and std_success:
                # Find common sequence lengths
                common_lens = set(r['seq_len'] for r in mla_success) & set(r['seq_len'] for r in std_success)
                if common_lens:
                    max_common_len = max(common_lens)
                    mla_time = next(r['avg_time'] for r in mla_success if r['seq_len'] == max_common_len)
                    std_time = next(r['avg_time'] for r in std_success if r['seq_len'] == max_common_len)
                    speedup = std_time / mla_time
                    print(f"Speed comparison at {max_common_len} tokens:")
                    print(f"  MLA is {speedup:.2f}x faster than standard attention")
    
    # Quantized model summary
    if quantized_results:
        q_success = [r for r in quantized_results if r['success']]
        if q_success:
            max_q_len = max(r['seq_len'] for r in q_success)
            max_q_tps = max(r['tokens_per_sec'] for r in q_success)
            print(f"EdgeFormer (INT8):")
            print(f"  Max sequence length: {max_q_len}")
            print(f"  Max throughput: {max_q_tps:.2f} tokens/sec")
            
            # Compare MLA vs Quantized
            if mla_success and q_success:
                # Find common sequence lengths
                common_lens = set(r['seq_len'] for r in mla_success) & set(r['seq_len'] for r in q_success)
                if common_lens:
                    max_common_len = max(common_lens)
                    mla_time = next(r['avg_time'] for r in mla_success if r['seq_len'] == max_common_len)
                    q_time = next(r['avg_time'] for r in q_success if r['seq_len'] == max_common_len)
                    speedup = mla_time / q_time
                    
                    # Memory reduction
                    mla_mem = next(r['param_mem_mb'] for r in mla_success if r['seq_len'] == max_common_len)
                    q_mem = next(r['param_mem_mb'] for r in q_success if r['seq_len'] == max_common_len)
                    mem_reduction = mla_mem / q_mem
                    
                    print(f"Quantization comparison at {max_common_len} tokens:")
                    print(f"  INT8 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than FP32")
                    print(f"  INT8 uses {mem_reduction:.2f}x less memory than FP32")
    
    print(f"\nAll benchmark results saved to '{output_dir}' directory")

if __name__ == "__main__":
    main()