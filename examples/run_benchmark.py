import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.logging import setup_logging
from src.utils.device import get_device
from src.optimization.benchmark import compare_mla_vs_standard, plot_benchmark_results
from src.optimization.quantization import quantize_model

def main():
    # Set up logging
    logger = setup_logging(debug_mode=True)
    logger.info("Starting EdgeFormer benchmark...")
    
    # Benchmark parameters
    seq_lengths = [128, 256, 512, 1024]  # Start with smaller values for testing
    
    # Run benchmarks
    results = compare_mla_vs_standard(
        EdgeFormerConfig, 
        EdgeFormer, 
        seq_lengths=seq_lengths,
        hidden_size=256  # Smaller size for faster testing
    )
    
    # Plot results
    plot_benchmark_results(results)
    
    # Test quantization
    logger.info("Testing quantization...")
    
    config = EdgeFormerConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024
    )
    
    model = EdgeFormer(config)
    device = get_device()
    
    # Only quantize on CPU
    if device.type == "cpu":
        logger.info("Quantizing model...")
        quantized_model = quantize_model(model, quantization="int8")
        
        # Compare model sizes
        original_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Size in MB for fp32
        logger.info(f"Original model size: {original_size:.2f} MB")
        
        # For quantized models, size estimation is more complex
        logger.info("Quantized model created successfully")
    else:
        logger.info("Skipping quantization as it's only supported on CPU")

if __name__ == "__main__":
    main()