# examples/test_kv_cache_long_sequences.py
import torch
import time
import json
import logging
import sys
import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from utils.memory_tracking import measure_memory_usage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("kv_cache_test")

def test_kv_cache_offloading(
    seq_lengths=[512, 1024, 2048, 4096],
    batch_sizes=[1, 2, 4, 8],
    use_offloading=True,
    output_dir="benchmark_results"
):
    """
    Test KV cache offloading with different sequence lengths and batch sizes.
    
    Args:
        seq_lengths: List of sequence lengths to test
        batch_sizes: List of batch sizes to test
        use_offloading: Whether to use KV cache offloading
        output_dir: Directory to save results
    """
    results = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a smaller model for testing
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=8192  # Support long sequences
    )
    
    # Loop through configurations
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            # Skip configurations that would be too large
            if batch_size * seq_length > 16384 and not use_offloading:
                logger.info(f"Skipping batch_size={batch_size}, seq_length={seq_length} without offloading (too large)")
                continue
                
            logger.info(f"Testing batch_size={batch_size}, seq_length={seq_length}, offloading={use_offloading}")
            
            try:
                # Initialize model
                model = EdgeFormer(config)
                model.eval()
                
                # Enable KV cache offloading if specified
                if use_offloading:
                    from src.utils.kv_cache_offload import kv_cache_offload
                    model = kv_cache_offload(model)
                
                # Create input tensors
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
                attention_mask = torch.ones(batch_size, seq_length)
                
                # Measure memory before
                mem_before = measure_memory_usage()
                
                # Warmup
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                
                # Measure memory after
                mem_after = measure_memory_usage()
                
                # Calculate time
                elapsed_time = time.time() - start_time
                throughput = (batch_size * seq_length) / elapsed_time
                
                # Try continuation with one more token
                next_token = torch.randint(0, config.vocab_size, (batch_size, 1))
                
                cont_start_time = time.time()
                with torch.no_grad():
                    cont_outputs = model.continue_generation(
                        next_token,
                        outputs["past_key_values"]
                    )
                cont_time = time.time() - cont_start_time
                
                # Get KV cache size if available
                kv_cache_size = 0
                if "past_key_values" in outputs and outputs["past_key_values"] is not None:
                    if not use_offloading:  # Only measure if not offloaded
                        kv_cache_size = sum(
                            sum(x.nelement() * x.element_size() for x in layer_kv if x is not None) 
                            for layer_kv in outputs["past_key_values"] if layer_kv is not None
                        ) / (1024 * 1024)  # MB
                
                # Clean up KV cache files
                if hasattr(model, "cleanup_kv_cache"):
                    model.cleanup_kv_cache()
                
                # Record results
                result = {
                    "batch_size": batch_size,
                    "seq_length": batch_size * seq_length,
                    "offloading": use_offloading,
                    "cpu_memory_before_mb": mem_before["cpu_memory_mb"],
                    "cpu_memory_after_mb": mem_after["cpu_memory_mb"],
                    "cpu_memory_diff_mb": mem_after["cpu_memory_mb"] - mem_before["cpu_memory_mb"],
                    "gpu_memory_before_mb": mem_before["gpu_memory_mb"],
                    "gpu_memory_after_mb": mem_after["gpu_memory_mb"],
                    "gpu_memory_diff_mb": mem_after["gpu_memory_mb"] - mem_before["gpu_memory_mb"],
                    "kv_cache_size_mb": kv_cache_size,
                    "time_sec": elapsed_time,
                    "throughput_tokens_per_sec": throughput,
                    "continuation_time_sec": cont_time
                }
                
                results.append(result)
                
                # Log result
                logger.info(f"Result: {result}")
                
                # Free memory
                del model, outputs, cont_outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"Error with batch_size={batch_size}, seq_length={seq_length}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    offload_suffix = "with_offload" if use_offloading else "no_offload"
    results_file = os.path.join(output_dir, f"kv_cache_results_{offload_suffix}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Create table for display
    table_data = []
    for result in results:
        table_data.append([
            result["batch_size"],
            result["seq_length"],
            f"{result['cpu_memory_diff_mb']:.2f}",
            f"{result['gpu_memory_diff_mb']:.2f}",
            f"{result['kv_cache_size_mb']:.2f}",
            f"{result['time_sec']:.2f}",
            f"{result['throughput_tokens_per_sec']:.2f}",
            f"{result['continuation_time_sec']:.4f}"
        ])
    
    headers = ["Batch", "Seq Len", "CPU Mem (MB)", "GPU Mem (MB)", 
               "KV Cache (MB)", "Time (s)", "Tokens/s", "Cont. Time (s)"]
    
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print("\nResults Summary:")
    print(table)
    
    return results

def plot_results(with_offload_results, no_offload_results=None, output_dir="benchmark_results"):
    """Plot memory usage and throughput comparisons."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    seq_lengths = sorted(list(set([r["seq_length"] for r in with_offload_results])))
    
    # Memory usage plot
    plt.figure(figsize=(12, 8))
    
    # With offloading
    cpu_mem_with_offload = [next((r["cpu_memory_diff_mb"] for r in with_offload_results 
                            if r["seq_length"] == sl and r["batch_size"] == 1), 0) 
                            for sl in seq_lengths]
    
    plt.plot(seq_lengths, cpu_mem_with_offload, 'o-', label="CPU Memory (with offload)")
    
    # Without offloading (if available)
    if no_offload_results:
        # Filter sequence lengths that exist in no_offload_results
        no_offload_seq_lengths = sorted(list(set([r["seq_length"] for r in no_offload_results])))
        cpu_mem_no_offload = [next((r["cpu_memory_diff_mb"] for r in no_offload_results 
                             if r["seq_length"] == sl and r["batch_size"] == 1), 0) 
                             for sl in no_offload_seq_lengths]
        
        plt.plot(no_offload_seq_lengths, cpu_mem_no_offload, 's-', label="CPU Memory (no offload)")
    
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"))
    
    # Throughput plot
    plt.figure(figsize=(12, 8))
    
    # With offloading
    throughput_with_offload = [next((r["throughput_tokens_per_sec"] for r in with_offload_results 
                               if r["seq_length"] == sl and r["batch_size"] == 1), 0) 
                               for sl in seq_lengths]
    
    plt.plot(seq_lengths, throughput_with_offload, 'o-', label="Throughput (with offload)")
    
    # Without offloading (if available)
    if no_offload_results:
        throughput_no_offload = [next((r["throughput_tokens_per_sec"] for r in no_offload_results 
                                if r["seq_length"] == sl and r["batch_size"] == 1), 0) 
                                for sl in no_offload_seq_lengths]
        
        plt.plot(no_offload_seq_lengths, throughput_no_offload, 's-', label="Throughput (no offload)")
    
    plt.xlabel("Sequence Length")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Throughput vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KV cache offloading with long sequences")
    parser.add_argument("--with-offload", action="store_true", help="Test with KV cache offloading")
    parser.add_argument("--without-offload", action="store_true", help="Test without KV cache offloading")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--plot-only", action="store_true", help="Only plot existing results")
    
    args = parser.parse_args()
    
    # Set default if no options specified
    if not (args.with_offload or args.without_offload or args.plot_only):
        args.with_offload = True
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Run tests
    with_offload_results = None
    no_offload_results = None
    
    if not args.plot_only:
        if args.with_offload:
            logger.info("Testing with KV cache offloading")
            with_offload_results = test_kv_cache_offloading(
                use_offloading=True,
                output_dir=output_dir
            )
        
        if args.without_offload:
            logger.info("Testing without KV cache offloading")
            no_offload_results = test_kv_cache_offloading(
                use_offloading=False,
                output_dir=output_dir
            )
    else:
        # Load existing results
        with_offload_file = os.path.join(args.output_dir, "kv_cache_results_with_offload.json")
        no_offload_file = os.path.join(args.output_dir, "kv_cache_results_no_offload.json")
        
        if os.path.exists(with_offload_file):
            with open(with_offload_file, "r") as f:
                with_offload_results = json.load(f)
        
        if os.path.exists(no_offload_file):
            with open(no_offload_file, "r") as f:
                no_offload_results = json.load(f)
    
    # Plot results
    if with_offload_results:
        plot_results(with_offload_results, no_offload_results, output_dir)