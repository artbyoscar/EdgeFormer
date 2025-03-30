#!/usr/bin/env python
# scripts/optimize_model.py
import argparse
import logging
import os
import sys
import torch
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import EdgeFormer
from src.optimization.quantization import quantize_edgeformer
from src.optimization.kv_cache_manager import KVCacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("optimize_model")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Optimize EdgeFormer model")
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Model size to optimize",
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to apply quantization",
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Number of bits for quantization (4 or 8)",
    )
    
    parser.add_argument(
        "--kv-cache-offload",
        action="store_true",
        help="Whether to use KV cache offloading",
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="Sequence length for benchmarking",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_model.pt",
        help="Output file path for optimized model",
    )
    
    parser.add_argument(
        "--attention-type",
        type=str,
        default="mla",
        choices=["standard", "mla", "gqa", "sliding_window"],
        help="Type of attention to use",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    
    return parser.parse_args()

def get_model_config(model_size, attention_type):
    """Get model configuration based on size."""
    if model_size == "tiny":
        return EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            attention_type=attention_type,
        )
    elif model_size == "small":
        return EdgeFormerConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            attention_type=attention_type,
        )
    elif model_size == "medium":
        return EdgeFormerConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            attention_type=attention_type,
        )
    elif model_size == "large":
        return EdgeFormerConfig(
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=4096,
            attention_type=attention_type,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

def benchmark_model(model, seq_length, device, kv_offload=False):
    """Benchmark model performance."""
    logger.info(f"Benchmarking model with sequence length {seq_length}...")
    
    # Create dummy inputs
    batch_size = 1
    input_ids = torch.randint(
        0, 1000, (batch_size, seq_length), dtype=torch.long, device=device
    )
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    
    # Set up KV cache manager if needed
    kv_cache_manager = None
    if kv_offload:
        kv_cache_manager = KVCacheManager(
            device=device,
            cpu_offload_threshold=512,
            chunk_size=256,
            debug=True,
        )
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, attention_mask)
    
    # Benchmark
    num_runs = 5
    latencies = []
    
    with torch.no_grad():
        for i in range(num_runs):
            # Clear cache if using offloading
            if kv_cache_manager:
                kv_cache_manager.clear_cache()
            
            # Benchmark
            start_time = time.time()
            _ = model(input_ids, attention_mask)
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            logger.info(f"Run {i+1}/{num_runs}: {latency:.4f}s")
    
    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    tokens_per_second = seq_length / avg_latency
    
    logger.info(f"Average latency: {avg_latency:.4f}s")
    logger.info(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Memory usage
    if kv_cache_manager:
        stats = kv_cache_manager.get_stats()
        logger.info(f"KV cache memory usage: {stats}")
    
    return {
        "avg_latency": avg_latency,
        "tokens_per_second": tokens_per_second,
    }

def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Optimizing {args.model_size} model with {args.attention_type} attention...")
    
    # Create model config
    config = get_model_config(args.model_size, args.attention_type)
    
    # Create model
    model = EdgeFormer(config)
    
    # Move to device
    device = torch.device(args.device)
    model = model.to(device)
    
    # Apply quantization if requested
    if args.quantize:
        logger.info(f"Applying {args.bits}-bit quantization...")
        model = quantize_edgeformer(model, bits=args.bits)
    
    # Benchmark model
    benchmark_model(
        model, 
        args.sequence_length, 
        device, 
        kv_offload=args.kv_cache_offload
    )
    
    # Save optimized model
    logger.info(f"Saving optimized model to {args.output}...")
    torch.save(model.state_dict(), args.output)
    
    logger.info("Optimization complete!")

if __name__ == "__main__":
    main()