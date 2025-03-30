import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# scripts/phase2_benchmark.py
import argparse
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.models.advanced_optimizer import Phase2Optimizer
from src.utils.logging_utils import setup_logging, get_logger

# Set up logging
logger = setup_logging('phase2_benchmark')

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Benchmarking Tool")
    parser.add_argument('--model_size', type=str, default='small', 
                        choices=['small', 'medium', 'large'],
                        help='Model size to benchmark')
    parser.add_argument('--output_dir', type=str, default='benchmark_results/phase2',
                        help='Directory to save benchmark results')
    parser.add_argument('--sequence_lengths', type=str, default='128,512,1024,2048,4096',
                        help='Comma-separated list of sequence lengths to test')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations for each test')
    parser.add_argument('--with_phase2', action='store_true',
                        help='Run with Phase 2 optimizations')
    return parser.parse_args()

def run_benchmark(model_size, sequence_lengths, iterations, output_dir, with_phase2=False):
    """Run performance benchmarks for the model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize hardware profiler
    optimizer = Phase2Optimizer()
    hardware_profile = optimizer.get_hardware_profile()
    
    # Log hardware profile
    logger.info(f"Hardware profile: {json.dumps(hardware_profile, indent=2)}")
    
    # Save hardware profile
    with open(os.path.join(output_dir, 'hardware_profile.json'), 'w') as f:
        json.dump(hardware_profile, f, indent=2)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = f"edgeformer-{model_size}"
    logger.info(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Apply Phase 2 optimizations if requested
        if with_phase2:
            from src.models.model_optimizer import optimize_model_for_device
            model = optimize_model_for_device(model, model.config, phase2=True)
            logger.info("Applied Phase 2 optimizations")
        
        # Prepare test input
        input_text = "This is a test sentence for benchmarking the EdgeFormer model's performance across different sequence lengths and optimization levels."
        
        # Initialize results container
        results = []
        
        # Run benchmarks for each sequence length
        for seq_length in sequence_lengths:
            logger.info(f"Benchmarking sequence length: {seq_length}")
            
            # Generate a sequence of the desired length
            if seq_length <= len(input_text.split()):
                test_input = ' '.join(input_text.split()[:seq_length])
            else:
                # Repeat the input to reach the desired length
                repetitions = (seq_length // len(input_text.split())) + 1
                test_input = ' '.join([input_text] * repetitions)
                test_input = ' '.join(test_input.split()[:seq_length])
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors='pt').to(device)
            
            # Warm-up run
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=20)
            
            # Benchmark inference
            inference_times = []
            memory_usage = []
            
            for i in range(iterations):
                # Record start time
                start_time = time.time()
                
                # Clear CUDA cache if available
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Run inference
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=20)
                
                # Record end time
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                # Record memory usage
                if device == 'cuda':
                    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                else:
                    import psutil
                    memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # MB
                memory_usage.append(memory_used)
                
                logger.info(f"  Iteration {i+1}/{iterations}: {inference_time:.2f}s, {memory_used:.2f}MB")
                
                # Clear cache
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            
            # Calculate average results
            avg_inference_time = np.mean(inference_times)
            avg_memory_usage = np.mean(memory_usage)
            tokens_per_second = seq_length / avg_inference_time
            
            # Store results
            result = {
                'sequence_length': seq_length,
                'tokens_per_second': tokens_per_second,
                'inference_time': avg_inference_time,
                'memory_usage': avg_memory_usage,
                'phase2_optimizations': with_phase2
            }
            results.append(result)
            
            logger.info(f"Results for {seq_length} tokens: {tokens_per_second:.2f} tokens/s, {avg_memory_usage:.2f}MB")
        
        # Save results
        results_filename = f"{model_size}_phase2_{with_phase2}.json"
        with open(os.path.join(output_dir, results_filename), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        plot_results(results, model_size, with_phase2, output_dir)
        
        return results
    
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        raise

def plot_results(results, model_size, with_phase2, output_dir):
    """Generate visualizations for benchmark results"""
    # Extract data for plotting
    seq_lengths = [r['sequence_length'] for r in results]
    tokens_per_second = [r['tokens_per_second'] for r in results]
    memory_usage = [r['memory_usage'] for r in results]
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot tokens per second
    ax1.plot(seq_lengths, tokens_per_second, 'o-', linewidth=2, markersize=8)
    ax1.set_title(f'Performance - EdgeFormer {model_size}')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Tokens per Second')
    ax1.grid(True)
    
    # Plot memory usage
    ax2.plot(seq_lengths, memory_usage, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_title(f'Memory Usage - EdgeFormer {model_size}')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Memory (MB)')
    ax2.grid(True)
    
    # Add optimization info to title
    opt_status = "with Phase 2 optimizations" if with_phase2 else "with basic optimizations"
    fig.suptitle(f'EdgeFormer {model_size} Benchmark Results {opt_status}')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    filename = f"{model_size}_phase2_{with_phase2}.png"
    plt.savefig(os.path.join(output_dir, filename))
    logger.info(f"Saved visualization to {os.path.join(output_dir, filename)}")

def main():
    """Main function to run the benchmark"""
    args = parse_args()
    
    # Convert sequence lengths string to list of integers
    sequence_lengths = [int(seq_len) for seq_len in args.sequence_lengths.split(',')]
    
    # Run the benchmark
    logger.info(f"Starting benchmark for model size: {args.model_size}")
    logger.info(f"Phase 2 optimizations: {'enabled' if args.with_phase2 else 'disabled'}")
    
    run_benchmark(
        model_size=args.model_size,
        sequence_lengths=sequence_lengths,
        iterations=args.iterations,
        output_dir=args.output_dir,
        with_phase2=args.with_phase2
    )
    
    logger.info("Benchmark completed successfully")

if __name__ == "__main__":
    main()
