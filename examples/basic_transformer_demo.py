#!/usr/bin/env python
# examples/basic_transformer_demo.py
import argparse
import logging
import os
import sys
import torch
import time
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import EdgeFormer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("basic_transformer_demo")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="EdgeFormer Basic Demo")
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Model size to use",
    )
    
    parser.add_argument(
        "--attention-type",
        type=str,
        default="standard",
        choices=["standard", "sliding_window"],
        help="Type of attention to use",
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Sequence length for benchmarking",
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs for benchmarking",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run text generation demo",
    )
    
    return parser.parse_args()

def get_model_config(model_size, attention_type):
    """Get model configuration based on size."""
    if model_size == "tiny":
        return EdgeFormerConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            attention_type=attention_type,
        )
    elif model_size == "small":
        return EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            attention_type=attention_type,
        )
    elif model_size == "medium":
        return EdgeFormerConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            attention_type=attention_type,
        )
    elif model_size == "large":
        return EdgeFormerConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            attention_type=attention_type,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

def run_benchmark(model, seq_length, num_runs, device):
    """Benchmark model performance."""
    logger.info(f"Benchmarking model with sequence length {seq_length}...")
    
    # Create dummy inputs
    batch_size = 1
    input_ids = torch.randint(
        0, 1000, (batch_size, seq_length), dtype=torch.long, device=device
    )
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, attention_mask)
    
    # Benchmark
    latencies = []
    
    with torch.no_grad():
        for i in range(num_runs):
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
    std_dev = np.std(latencies)
    
    # Print results
    logger.info("-" * 50)
    logger.info(f"Benchmark Results:")
    logger.info(f"  Model Size: {args.model_size}")
    logger.info(f"  Attention Type: {args.attention_type}")
    logger.info(f"  Sequence Length: {seq_length}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Average Latency: {avg_latency:.4f}s")
    logger.info(f"  Standard Deviation: {std_dev:.4f}s")
    logger.info(f"  Tokens per Second: {tokens_per_second:.2f}")
    logger.info("-" * 50)
    
    return {
        "avg_latency": avg_latency,
        "tokens_per_second": tokens_per_second,
        "std_dev": std_dev,
    }

def run_generation_demo(model, device):
    """Run a text generation demo."""
    logger.info("Running text generation demo...")
    
    # Create a simple prompt
    batch_size = 1
    prompt_length = 4
    input_ids = torch.randint(
        0, 1000, (batch_size, prompt_length), dtype=torch.long, device=device
    )
    
    # Time generation
    start_time = time.time()
    
    # Generate text
    generated = model.generate(
        input_ids,
        max_length=32,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    
    end_time = time.time()
    
    # Print results
    generation_time = end_time - start_time
    tokens_generated = generated.shape[1] - prompt_length
    
    logger.info("-" * 50)
    logger.info(f"Generation Results:")
    logger.info(f"  Total tokens generated: {tokens_generated}")
    logger.info(f"  Generation time: {generation_time:.4f}s")
    logger.info(f"  Tokens per second: {tokens_generated / generation_time:.2f}")
    logger.info("-" * 50)
    
    return {
        "tokens_generated": tokens_generated,
        "generation_time": generation_time,
    }

def main(args):
    """Main function."""
    logger.info(f"Running EdgeFormer demo with {args.model_size} model...")
    
    # Set device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Create model config
    config = get_model_config(args.model_size, args.attention_type)
    
    # Print model configuration
    logger.info(f"Model Configuration:")
    logger.info(f"  Hidden Size: {config.hidden_size}")
    logger.info(f"  Layers: {config.num_hidden_layers}")
    logger.info(f"  Attention Heads: {config.num_attention_heads}")
    logger.info(f"  Attention Type: {config.attention_type}")
    
    # Create model
    logger.info("Creating model...")
    model = EdgeFormer(config)
    model.to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {num_params:,} parameters")
    
    # Run benchmark
    benchmark_results = run_benchmark(model, args.sequence_length, args.num_runs, device)
    
    # Run generation demo if requested
    if args.generate:
        generation_results = run_generation_demo(model, device)
    
    logger.info("Demo completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    main(args)