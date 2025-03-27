# examples/benchmark_simplified.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
import os
import psutil
import sys
import argparse

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.logging import setup_logging
from src.utils.memory_tracking import measure_memory_usage

# Set up logging
logger = setup_logging()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EdgeFormer Benchmark Script")
    
    parser.add_argument("--seq_lengths", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--attention_types", type=str, default="standard,mla,mla_window",
                        help="Comma-separated list of attention types to test (standard, mla, mla_window)")
    parser.add_argument("--skip_max_seq", action="store_true",
                        help="Skip maximum sequence length testing")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: benchmark_results_TIMESTAMP)")
    parser.add_argument("--num_measurements", type=int, default=3,
                        help="Number of measurements to take for memory usage")
    parser.add_argument("--max_position_embeddings", type=int, default=8192,
                        help="Maximum position embeddings for the model configurations")
    
    return parser.parse_args()

def benchmark_memory_usage(
    sequence_lengths,
    model_config,
    title="Memory Benchmark",
    output_dir="benchmark_results",
    num_measurements=3
):
    """Benchmark memory usage for different sequence lengths with improved methodology"""
    results = []
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    logger.info(f"Initializing {title} model...")
    model = EdgeFormer(model_config)
    model.eval()
    
    for seq_len in sequence_lengths:
        logger.info(f"Testing sequence length: {seq_len}")
        
        # Clear cache and collect garbage
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Wait for memory to settle
        time.sleep(1.0)
        
        # Record baseline memory - average of multiple measurements
        baseline_memories = []
        for _ in range(num_measurements):
            mem_info = measure_memory_usage()
            baseline_memories.append(mem_info["cpu_memory_mb"])
            time.sleep(0.2)  # Short delay between measurements
        baseline_memory = sum(baseline_memories) / len(baseline_memories)
        
        logger.info(f"Baseline memory for seq_len={seq_len}: {baseline_memory:.2f} MB")
        
        # Create input tensors
        input_ids = torch.randint(0, model_config.vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)
        
        # Component memory tracking - optional detailed tracking
        # You can add more detailed component tracking here if needed
        
        try:
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            inference_time = time.time() - start_time
            
            # Wait for memory operations to complete
            time.sleep(0.5)
            
            # Record memory after forward pass - average of multiple measurements
            after_memories = []
            for _ in range(num_measurements):
                mem_info = measure_memory_usage()
                after_memories.append(mem_info["cpu_memory_mb"])
                time.sleep(0.2)
            after_memory = sum(after_memories) / len(after_memories)
            
            logger.info(f"After memory for seq_len={seq_len}: {after_memory:.2f} MB")
            
            # Calculate memory used - ensure non-negative value
            memory_used = max(0, after_memory - baseline_memory)
            
            print(f"Sequence length: {seq_len}, Memory usage: {memory_used:.2f} MB, Inference time: {inference_time:.4f}s")
            logger.info(f"Calculated memory for seq_len={seq_len}: {memory_used:.2f} MB (after: {after_memory:.2f} - baseline: {baseline_memory:.2f})")
            
            results.append((seq_len, memory_used, inference_time, True))
        except Exception as e:
            logger.error(f"Error processing seq_len={seq_len}: {str(e)}")
            print(f"Failed at sequence length: {seq_len}, Error: {str(e)}")
            results.append((seq_len, 0, 0, False))
            break
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_{timestamp}.txt")
    
    with open(result_file, "w") as f:
        f.write(f"# {title}\n")
        f.write("Sequence Length, Memory Usage (MB), Inference Time (s), Success\n")
        for seq_len, mem, inf_time, success in results:
            f.write(f"{seq_len}, {mem:.2f}, {inf_time:.4f}, {success}\n")
    
    return results

def plot_benchmark_results(results_list, labels, output_dir="benchmark_results"):
    """Plot memory and time results from multiple benchmarks"""
    plt.figure(figsize=(15, 10))
    
    # Memory usage plot
    plt.subplot(2, 1, 1)
    for i, results in enumerate(results_list):
        seq_lens = [r[0] for r in results if r[3]]  # Only successful runs
        mem_usage = [r[1] for r in results if r[3]]
        plt.plot(seq_lens, mem_usage, marker='o', linestyle='-', label=labels[i])
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    
    # Inference time plot
    plt.subplot(2, 1, 2)
    for i, results in enumerate(results_list):
        seq_lens = [r[0] for r in results if r[3]]  # Only successful runs
        inf_time = [r[2] for r in results if r[3]]
        plt.plot(seq_lens, inf_time, marker='o', linestyle='-', label=labels[i])
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    
    # Save plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"benchmark_comparison_{timestamp}.png"))
    plt.show()

def find_max_sequence_length(config, start_len=1024, max_len=32768, step_factor=2):
    """Find the maximum sequence length that can be processed"""
    
    model = EdgeFormer(config)
    model.eval()
    
    current_len = start_len
    max_supported_len = 0
    
    while current_len <= max_len:
        try:
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info(f"Testing sequence length: {current_len}")
            
            # Create input tensors
            input_ids = torch.randint(0, config.vocab_size, (1, current_len))
            attention_mask = torch.ones(1, current_len)
            
            # Run inference
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            print(f"Successfully processed sequence length: {current_len}")
            logger.info(f"Successfully processed sequence length: {current_len}")
            max_supported_len = current_len
            current_len = int(current_len * step_factor)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed at sequence length: {current_len}. Error: {error_msg}")
            print(f"Failed at sequence length: {current_len}")
            print(f"Error: {error_msg}")
            
            if "out of memory" in error_msg.lower():
                # Memory error handling
                if step_factor > 1.1:
                    step_factor = 1.1
                    current_len = int(max_supported_len * step_factor)
                else:
                    # Binary search for the exact maximum
                    binary_search_start = max_supported_len
                    binary_search_end = current_len
                    
                    # Binary search for the exact maximum
                    while binary_search_end - binary_search_start > 128:
                        mid = (binary_search_start + binary_search_end) // 2
                        try:
                            gc.collect()
                            input_ids = torch.randint(0, config.vocab_size, (1, mid))
                            attention_mask = torch.ones(1, mid)
                            
                            with torch.no_grad():
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            
                            print(f"Binary search success: {mid}")
                            binary_search_start = mid
                        except:
                            print(f"Binary search failed: {mid}")
                            binary_search_end = mid
                    
                    max_supported_len = binary_search_start
                    break
            else:
                # For non-memory errors, don't try smaller steps (likely a model limitation)
                break
    
    return max_supported_len

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Parse sequence lengths
    sequence_lengths = [int(x) for x in args.seq_lengths.split(',')]
    
    # Create output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f"benchmark_results_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse attention types
    attention_types = args.attention_types.split(',')
    
    # Create configurations based on attention types
    configs = []
    labels = []
    
    max_pos_embeddings = args.max_position_embeddings
    
    if "standard" in attention_types:
        standard_config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            intermediate_size=1024,
            max_position_embeddings=max_pos_embeddings,
        )
        configs.append(("Standard Attention", standard_config))
        labels.append("Standard Attention")
    
    if "mla" in attention_types:
        mla_config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            max_position_embeddings=max_pos_embeddings,
        )
        configs.append(("Multi-Head Latent Attention", mla_config))
        labels.append("Multi-Head Latent Attention")
    
    if "mla_window" in attention_types:
        mla_window_config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            use_sliding_window=True,
            sliding_window_size=512,
            max_position_embeddings=max_pos_embeddings,
        )
        configs.append(("MLA with Sliding Window", mla_window_config))
        labels.append("MLA with Sliding Window")
    
    # Run benchmarks
    results_list = []
    
    for title, config in configs:
        print(f"\nBenchmarking {title}...")
        results = benchmark_memory_usage(
            sequence_lengths,
            config,
            title,
            output_dir,
            args.num_measurements
        )
        results_list.append(results)
    
    # Plot results
    if len(results_list) > 0:
        plot_benchmark_results(results_list, labels, output_dir)
    
    # Find maximum sequence lengths if not skipped
    if not args.skip_max_seq:
        max_lengths = {}
        
        for title, config in configs:
            print(f"\nFinding maximum sequence length for {title}...")
            max_len = find_max_sequence_length(config)
            max_lengths[title] = max_len
        
        # Save max sequence length results
        with open(os.path.join(output_dir, "max_sequence_lengths.txt"), "w") as f:
            for title, length in max_lengths.items():
                f.write(f"{title} maximum sequence length: {length}\n")
        
        print("\nMaximum Sequence Length Results:")
        for title, length in max_lengths.items():
            print(f"{title}: {length}")

if __name__ == "__main__":
    main()