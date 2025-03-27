import torch
import logging
import time
import sys
import platform
import matplotlib.pyplot as plt
import numpy as np
import os
from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
import argparse
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device (DirectML, CUDA, ROCm, CPU)"""
    # Try DirectML first for AMD GPUs
    try:
        import torch_directml
        logger.info("Using DirectML for AMD GPU acceleration")
        return torch_directml.device(), "directml"
    except ImportError:
        # Try CUDA for NVIDIA GPUs
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA GPU: {device_name}")
            return torch.device("cuda"), "cuda"
        
        # Try ROCm for AMD GPUs
        if hasattr(torch, 'hip') and torch.hip.is_available():
            try:
                device_name = torch.hip.get_device_name(0)
                logger.info(f"Using ROCm GPU: {device_name}")
                return torch.device("hip"), "rocm"
            except:
                logger.info("ROCm detected but couldn't get device name")
                return torch.device("hip"), "rocm"
        
        # Check specifically for AMD hardware but without GPU support
        system_info = platform.system()
        if system_info == "Windows":
            try:
                import subprocess
                output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
                if "AMD" in output or "Radeon" in output:
                    logger.warning("AMD GPU detected but no DirectML or ROCm support found.")
                    logger.warning("To enable DirectML support: pip install torch-directml")
            except:
                pass
        
        # Fallback to CPU
        logger.info("Using CPU")
        return torch.device("cpu"), "cpu"

def create_output_dir():
    """Create output directory for benchmark results with timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"benchmark_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def benchmark_model(config, sequence_lengths, device, device_type, num_runs=3):
    """Benchmark model performance across sequence lengths"""
    results = {
        'sequence_lengths': sequence_lengths,
        'inference_times': [],
        'memory_usage': [],
        'throughput': []  # tokens per second
    }
    
    for seq_len in tqdm(sequence_lengths, desc=f"Benchmarking on {device_type}"):
        logger.info(f"Benchmarking sequence length: {seq_len}")
        
        # Create model for this run
        model = EdgeFormer(config)
        model.to(device)
        model.eval()
        
        # Create input tensors
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        attention_mask = torch.ones(1, seq_len, device=device)
        
        # Track memory before run
        if device_type == "cpu":
            try:
                import psutil
                memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            except ImportError:
                logger.warning("psutil not available, memory usage will not be tracked accurately")
                memory_before = 0
        else:
            # For GPU we'll estimate based on model parameters and sequence length
            memory_before = 0
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):  # Multiple warmup runs
                model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Synchronize if using GPU
        if device_type in ["cuda", "rocm", "directml"] and device_type != "cpu":
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
        
        # Benchmark
        inference_times = []
        for _ in range(num_runs):
            with torch.no_grad():
                if device_type in ["cuda", "rocm"] and device_type != "cpu":
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
                
                start_time = time.time()
                model(input_ids=input_ids, attention_mask=attention_mask)
                
                if device_type in ["cuda", "rocm"] and device_type != "cpu":
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        # Calculate throughput (tokens per second)
        avg_inference_time = sum(inference_times) / len(inference_times)
        throughput = seq_len / avg_inference_time
        
        # Track memory after run
        if device_type == "cpu":
            try:
                import psutil
                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                memory_used = memory_after - memory_before
            except ImportError:
                # Estimate memory for CPU based on model size
                params_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
                activations_size = seq_len * config.hidden_size * 4 * 3 / (1024 * 1024)  # MB
                memory_used = params_size + activations_size
        else:
            # Estimate memory for GPU based on model size and sequence length
            params_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
            
            # Estimate KV cache size based on whether latent attention is used
            if hasattr(config, 'latent_size_factor') and config.latent_size_factor > 0:
                latent_size = config.hidden_size // config.latent_size_factor
                kv_cache_size = 2 * seq_len * latent_size * config.num_hidden_layers * 4 / (1024 * 1024)  # MB
            else:
                kv_cache_size = 2 * seq_len * config.hidden_size * config.num_hidden_layers * 4 / (1024 * 1024)  # MB
            
            # Add activations
            activations_size = seq_len * config.hidden_size * 4 * 3 / (1024 * 1024)  # MB
            memory_used = params_size + kv_cache_size + activations_size
        
        # Record results
        results['inference_times'].append(avg_inference_time)
        results['memory_usage'].append(memory_used)
        results['throughput'].append(throughput)
        
        logger.info(f"Average inference time: {avg_inference_time:.4f} seconds")
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"Estimated memory usage: {memory_used:.2f} MB")
        
        # Clear memory
        del model, input_ids, attention_mask
        if device_type in ["cuda", "rocm"] and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    return results

def plot_results(results_dict, output_dir):
    """Plot benchmark results for various devices"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the sequence lengths (should be the same for all results)
    sequence_lengths = next(iter(results_dict.values()))['sequence_lengths']
    
    # Plot inference time comparison
    plt.figure(figsize=(12, 7))
    for device_type, results in results_dict.items():
        plt.plot(
            results['sequence_lengths'], 
            results['inference_times'],
            'o-',
            linewidth=2,
            label=device_type.upper()
        )
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Inference Time (seconds)', fontsize=12)
    plt.title('Inference Time vs Sequence Length', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/inference_time_comparison.png', dpi=300)
    
    # Plot throughput comparison
    plt.figure(figsize=(12, 7))
    for device_type, results in results_dict.items():
        plt.plot(
            results['sequence_lengths'], 
            results['throughput'],
            'o-',
            linewidth=2,
            label=device_type.upper()
        )
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Throughput (tokens/sec)', fontsize=12)
    plt.title('Throughput vs Sequence Length', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_comparison.png', dpi=300)
    
    # Plot memory usage
    plt.figure(figsize=(12, 7))
    for device_type, results in results_dict.items():
        plt.plot(
            results['sequence_lengths'], 
            results['memory_usage'],
            'o-',
            linewidth=2,
            label=device_type.upper()
        )
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('Memory Usage vs Sequence Length', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_usage_comparison.png', dpi=300)
    
    # Calculate speedup if more than one device
    if len(results_dict) > 1 and 'cpu' in results_dict:
        plt.figure(figsize=(12, 7))
        
        cpu_times = results_dict['cpu']['inference_times']
        
        for device_type, results in results_dict.items():
            if device_type == 'cpu':
                continue
                
            device_times = results['inference_times']
            speedup = [cpu / device for cpu, device in zip(cpu_times, device_times)]
            
            plt.plot(
                results['sequence_lengths'], 
                speedup, 
                'o-',
                linewidth=2,
                label=f"{device_type.upper()} Speedup"
            )
        
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Speedup (CPU time / Device time)', fontsize=12)
        plt.title('GPU Speedup vs Sequence Length', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gpu_speedup.png', dpi=300)
    
    logger.info(f"Benchmark plots saved to '{output_dir}' directory")

def save_results_table(results_dict, output_dir):
    """Save benchmark results as a formatted text table"""
    with open(f'{output_dir}/benchmark_results.txt', 'w') as f:
        f.write("EdgeFormer Benchmark Results\n")
        f.write("==========================\n\n")
        
        # Get sequence lengths
        sequence_lengths = next(iter(results_dict.values()))['sequence_lengths']
        
        # Write inference time table
        f.write("Inference Time (seconds)\n")
        f.write("-----------------------\n")
        f.write(f"{'Sequence Length':<16} " + " ".join(f"{device.upper():<12}" for device in results_dict.keys()) + "\n")
        
        for i, seq_len in enumerate(sequence_lengths):
            f.write(f"{seq_len:<16} ")
            for device, results in results_dict.items():
                f.write(f"{results['inference_times'][i]:<12.4f} ")
            f.write("\n")
        
        f.write("\n")
        
        # Write throughput table
        f.write("Throughput (tokens/sec)\n")
        f.write("-----------------------\n")
        f.write(f"{'Sequence Length':<16} " + " ".join(f"{device.upper():<12}" for device in results_dict.keys()) + "\n")
        
        for i, seq_len in enumerate(sequence_lengths):
            f.write(f"{seq_len:<16} ")
            for device, results in results_dict.items():
                f.write(f"{results['throughput'][i]:<12.1f} ")
            f.write("\n")
        
        f.write("\n")
        
        # Write memory usage table
        f.write("Memory Usage (MB)\n")
        f.write("----------------\n")
        f.write(f"{'Sequence Length':<16} " + " ".join(f"{device.upper():<12}" for device in results_dict.keys()) + "\n")
        
        for i, seq_len in enumerate(sequence_lengths):
            f.write(f"{seq_len:<16} ")
            for device, results in results_dict.items():
                f.write(f"{results['memory_usage'][i]:<12.1f} ")
            f.write("\n")
        
        # If CPU and GPU results available, write speedup table
        if len(results_dict) > 1 and 'cpu' in results_dict:
            f.write("\n")
            f.write("Speedup (vs CPU)\n")
            f.write("----------------\n")
            f.write(f"{'Sequence Length':<16} " + " ".join(f"{device.upper():<12}" for device in results_dict.keys() if device != 'cpu') + "\n")
            
            cpu_times = results_dict['cpu']['inference_times']
            
            for i, seq_len in enumerate(sequence_lengths):
                f.write(f"{seq_len:<16} ")
                for device, results in results_dict.items():
                    if device == 'cpu':
                        continue
                    speedup = cpu_times[i] / results['inference_times'][i]
                    f.write(f"{speedup:<12.2f} ")
                f.write("\n")

def run_amd_benchmark(args):
    """Run benchmark comparing CPU and DirectML/ROCm performance on AMD GPUs"""
    # Check system info
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # Create output directory
    output_dir = create_output_dir()
    logger.info(f"Benchmark results will be saved to: {output_dir}")
    
    # Create model configuration
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,  # Using latent_size_factor instead of latent_size
        max_position_embeddings=2048,
    )
    
    # Sequence lengths to test
    sequence_lengths = args.seq_lengths
    
    # Dictionary to store results for different devices
    results_dict = {}
    
    # Run CPU benchmark if requested or if no specific device requested
    if args.cpu or (not args.gpu and not args.cpu):
        logger.info("Running CPU benchmark...")
        cpu_device = torch.device("cpu")
        cpu_results = benchmark_model(config, sequence_lengths, cpu_device, "cpu", args.num_runs)
        results_dict['cpu'] = cpu_results
    
    # Run GPU benchmark if requested or if no specific device requested
    if args.gpu or (not args.gpu and not args.cpu):
        # Get best GPU device
        gpu_device, device_type = get_device()
        
        # Only run GPU benchmark if not CPU
        if device_type != "cpu":
            logger.info(f"Running {device_type.upper()} benchmark...")
            gpu_results = benchmark_model(config, sequence_lengths, gpu_device, device_type, args.num_runs)
            results_dict[device_type] = gpu_results
        else:
            logger.warning("No GPU acceleration available")
    
    # Plot and save results if we have any
    if results_dict:
        plot_results(results_dict, output_dir)
        save_results_table(results_dict, output_dir)
    
    # Print summary
    logger.info("\n===== Benchmark Summary =====")
    
    for device_type, results in results_dict.items():
        logger.info(f"\n{device_type.upper()} Performance:")
        for i, seq_len in enumerate(sequence_lengths):
            logger.info(f"  Sequence length {seq_len}: {results['inference_times'][i]:.4f}s, {results['throughput'][i]:.1f} tokens/s")
    
    # Calculate and print speedups if we have both CPU and GPU results
    if 'cpu' in results_dict and len(results_dict) > 1:
        logger.info("\nSpeedup Comparison:")
        cpu_times = results_dict['cpu']['inference_times']
        
        for device_type, results in results_dict.items():
            if device_type == 'cpu':
                continue
                
            logger.info(f"\n{device_type.upper()} vs CPU:")
            for i, seq_len in enumerate(sequence_lengths):
                speedup = cpu_times[i] / results['inference_times'][i]
                logger.info(f"  Sequence length {seq_len}: {speedup:.2f}x faster")
    
    return results_dict, output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark EdgeFormer on AMD GPUs with DirectML or ROCm")
    parser.add_argument("--seq_lengths", type=int, nargs='+', default=[64, 128, 256, 512, 1024, 2048],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs to average for each sequence length")
    parser.add_argument("--cpu", action="store_true",
                        help="Run CPU benchmark only")
    parser.add_argument("--gpu", action="store_true",
                        help="Run GPU benchmark only")
    
    args = parser.parse_args()
    
    run_amd_benchmark(args)