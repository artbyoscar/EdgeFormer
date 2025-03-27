import os
import sys
import torch
import logging
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.device import get_device
from src.utils.rdna3_optimizations import optimize_for_rdna3, is_rdna3_gpu
from src.utils.weight_quantization import weight_only_quantize_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluation_suite")

def create_model(config):
    """Create EdgeFormer model."""
    model = EdgeFormer(config)
    model.eval()
    return model

def evaluate_model(model, device, sequence_lengths, batch_size=1, num_runs=3):
    """Evaluate model performance."""
    import time  # Make sure to import time at the top of your file
    
    results = {
        'sequence_lengths': sequence_lengths,
        'inference_times': [],
        'throughput': [],
        'memory_usage': []
    }
    
    for seq_len in tqdm(sequence_lengths):
        # Create input tensors
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Measure inference time
        inference_times = []
        for _ in range(num_runs):
            # Use simple Python timing for all devices to avoid CUDA-specific code on CPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            inference_time = end_time - start_time
            
            inference_times.append(inference_time)
        
        # Calculate average inference time
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        # Calculate throughput (tokens/sec)
        throughput = seq_len / avg_inference_time
        
        # Record results
        results['inference_times'].append(avg_inference_time)
        results['throughput'].append(throughput)
        
        # Try to estimate memory usage if possible
        try:
            if device.type == 'cuda':
                # For CUDA, we can measure memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                mem_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                results['memory_usage'].append(mem_used)
            else:
                # For CPU, use an estimation based on model size and sequence length
                param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
                activation_size = batch_size * seq_len * model.config.hidden_size * 4 / (1024 * 1024)  # Rough estimate
                results['memory_usage'].append(param_size + activation_size)
        except:
            # If memory measurement fails, just use a placeholder
            results['memory_usage'].append(0)
    
    return results

def compare_optimizations(config, sequence_lengths, num_runs=3):
    """Compare different optimizations."""
    device = get_device()
    
    # Create models with different optimizations
    base_model = create_model(config).to(device)
    
    # RDNA3 optimized model (if applicable)
    if is_rdna3_gpu():
        rdna3_model = optimize_for_rdna3(create_model(config)).to(device)
    else:
        rdna3_model = None
    
    # INT8 quantized model
    int8_model = weight_only_quantize_model(
        create_model(config),
        bits=8,
        group_size=128,
        symmetric=True
    ).to(device)
    
    # INT4 quantized model
    int4_model = weight_only_quantize_model(
        create_model(config),
        bits=4,
        group_size=128,
        symmetric=True
    ).to(device)
    
    # Evaluate models
    results = {}
    results['base'] = evaluate_model(base_model, device, sequence_lengths, num_runs=num_runs)
    
    if rdna3_model:
        results['rdna3'] = evaluate_model(rdna3_model, device, sequence_lengths, num_runs=num_runs)
    
    results['int8'] = evaluate_model(int8_model, device, sequence_lengths, num_runs=num_runs)
    results['int4'] = evaluate_model(int4_model, device, sequence_lengths, num_runs=num_runs)
    
    return results

def plot_results(results, output_dir='benchmark_results'):
    """Plot comparison results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot inference time
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['sequence_lengths'], result['inference_times'], 'o-', label=name.upper())
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
    
    # Plot throughput
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['sequence_lengths'], result['throughput'], 'o-', label=name.upper())
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'))

def main():
    parser = argparse.ArgumentParser(description='EdgeFormer Evaluation Suite')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[128, 256, 512, 1024, 2048],
                        help='Sequence lengths to evaluate')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs to average')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Create model configuration
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
        max_position_embeddings=2048,
    )
    
    # Compare optimizations
    results = compare_optimizations(config, args.seq_lengths, num_runs=args.num_runs)
    
    # Plot results
    plot_results(results, output_dir=args.output_dir)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()