# examples/generate_benchmark_data.py
import os
import json
import time
import argparse
import torch
import logging
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer, EdgeFormerConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('edgeformer')

def run_benchmark(args):
    """Run benchmarks for different configurations"""
    results = []
    
    # Define configurations to benchmark
    configs = [
        {
            "name": "baseline",
            "attention_type": "mla",
            "use_recurrent": False,
            "use_budget": False,
            "use_kv_cache": False
        },
        {
            "name": "recurrent",
            "attention_type": "mla",
            "use_recurrent": True,
            "use_budget": False,
            "use_kv_cache": False,
            "max_iterations": 8,
            "convergence_threshold": 0.005
        },
        {
            "name": "budget",
            "attention_type": "mla",
            "use_recurrent": False,
            "use_budget": True,
            "use_kv_cache": False,
            "max_budget_tokens": 2048,
            "extensions": 2
        },
        {
            "name": "kvcache",
            "attention_type": "mla",
            "use_recurrent": False,
            "use_budget": False,
            "use_kv_cache": True
        },
        {
            "name": "combined",
            "attention_type": "mla",
            "use_recurrent": True,
            "use_budget": True,
            "use_kv_cache": True,
            "max_iterations": 8,
            "convergence_threshold": 0.005,
            "max_budget_tokens": 2048,
            "extensions": 2
        }
    ]
    
    # Sequence lengths to test
    sequence_lengths = [128, 256, 512, 1024, 2048] if not args.quick_test else [256, 1024]
    
    device = torch.device(args.device)
    
    for config in configs:
        logger.info(f"Benchmarking {config['name']} configuration...")
        
        # Create EdgeFormer configuration
        model_config = EdgeFormerConfig(
            vocab_size=50257,  # GPT-2 compatible
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            attention_type=config['attention_type'],
            max_position_embeddings=4096,
            enable_budget_forcing=config.get('use_budget', False),
            max_budget_tokens=config.get('max_budget_tokens', 2048),
            max_thinking_extensions=config.get('extensions', 2),
            extension_token="Wait",
            budget_criteria="balanced",
            enable_recurrent_depth=config.get('use_recurrent', False),
            max_iterations=config.get('max_iterations', 8),
            convergence_threshold=config.get('convergence_threshold', 0.005)
        )
        
        # Create model
        model = EdgeFormer(model_config)
        model.to(device)
        model.eval()
        
        # Initialize KV Cache Manager if needed
        if config.get('use_kv_cache', False):
            from src.utils.kv_cache_manager import KVCacheManager
            kv_cache_manager = KVCacheManager(
                max_batch_size=1,
                max_seq_length=4096,
                num_layers=model_config.num_hidden_layers,
                num_heads=model_config.num_attention_heads,
                head_dim=model_config.hidden_size // model_config.num_attention_heads,
                device=device,
                enable_offload=True
            )
            model.kv_cache_manager = kv_cache_manager
            logger.info(f"KV Cache Manager initialized")
        
        # Run benchmarks for different sequence lengths
        for seq_len in sequence_lengths:
            logger.info(f"Testing sequence length: {seq_len}")
            
            # Generate random input
            input_ids = torch.randint(0, model_config.vocab_size, (1, seq_len), device=device)
            
            # Warm-up run
            with torch.no_grad():
                model(input_ids)
            
            # Benchmark run with timing
            start_time = time.time()
            peak_memory = 0
            
            try:
                with torch.no_grad():
                    # If CUDA is available, track memory usage
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    
                    # Run model
                    outputs = model(input_ids)
                    
                    # Measure memory usage
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            except Exception as e:
                logger.error(f"Error during benchmark: {e}")
                continue
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Record results
            result = {
                "config_name": config["name"],
                "sequence_length": seq_len,
                "inference_time": inference_time,
                "tokens_per_second": seq_len / inference_time,
                "peak_memory_mb": peak_memory if peak_memory > 0 else "N/A",
                "config_details": config
            }
            
            results.append(result)
            
            logger.info(f"Results: {seq_len} tokens in {inference_time:.4f}s ({seq_len / inference_time:.2f} tokens/s)")
            logger.info(f"Peak memory usage: {peak_memory:.2f} MB" if peak_memory > 0 else "Peak memory not available")
            
            # Free up memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save results to JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_file}")
    return output_file

def parse_args():
    parser = argparse.ArgumentParser(description="EdgeFormer Benchmark Generator")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu|cuda)")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with fewer sequence lengths")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_file = run_benchmark(args)
    
    # Print summary
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Total configurations tested: {len(set(r['config_name'] for r in results))}")
    print(f"Sequence lengths tested: {sorted(set(r['sequence_length'] for r in results))}")
    print("\nTop results by tokens/second:")
    
    # Sort by tokens per second (descending)
    sorted_results = sorted(results, key=lambda x: x['tokens_per_second'], reverse=True)
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. {result['config_name']} @ {result['sequence_length']} tokens: {result['tokens_per_second']:.2f} tokens/s")
    
    print("\nRun analyze_benchmark_logs.py for detailed analysis")