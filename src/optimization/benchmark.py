import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import logging

from src.utils.device import get_device

def compare_mla_vs_standard(config_class, model_class, seq_lengths, hidden_size=768):
    """Compare memory usage and speed between MLA and standard attention."""
    logger = logging.getLogger("edgeformer")
    logger.info("Starting MLA vs Standard benchmark...")
    
    results = {
        "mla": {
            "seq_length": seq_lengths,
            "memory_usage": [],
            "inference_time": []
        },
        "standard": {
            "seq_length": seq_lengths,
            "memory_usage": [],
            "inference_time": []
        }
    }
    
    device = get_device()
    logger.info(f"Benchmarking on device: {device}")
    
    for use_mla in [True, False]:
        model_type = "mla" if use_mla else "standard"
        logger.info(f"Testing {model_type} model")
        
        for seq_len in seq_lengths:
            logger.info(f"Sequence length: {seq_len}")
            
            # Create model with or without MLA
            config = config_class(
                hidden_size=hidden_size,
                num_hidden_layers=4,
                latent_size_factor=8 if use_mla else 1,
            )
            
            model = model_class(config)
            model = model.to(device)
            
            # Create input of specific sequence length
            input_ids = torch.randint(0, 100, (1, seq_len)).to(device)
            
            # Measure memory before
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                before_memory = torch.cuda.memory_allocated()
            
            # Measure inference time
            start_time = time.time()
            _ = model.generate(input_ids, max_length=seq_len + 10)
            end_time = time.time()
            
            # Measure memory after
            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() - before_memory
                results[model_type]["memory_usage"].append(peak_memory / (1024 * 1024))  # Convert to MB
            else:
                results[model_type]["memory_usage"].append(0)  # Can't measure on CPU
                
            results[model_type]["inference_time"].append(end_time - start_time)
            logger.info(f"Inference time: {end_time - start_time:.4f}s")
    
    return results

def plot_benchmark_results(results, save_path="benchmark_results.png"):
    """Plot benchmark results comparing MLA and standard attention."""
    plt.figure(figsize=(12, 5))
    
    # Memory usage plot
    plt.subplot(1, 2, 1)
    plt.plot(results["mla"]["seq_length"], results["mla"]["memory_usage"], 'b-', label='MLA')
    plt.plot(results["standard"]["seq_length"], results["standard"]["memory_usage"], 'r-', label='Standard')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    
    # Inference time plot
    plt.subplot(1, 2, 2)
    plt.plot(results["mla"]["seq_length"], results["mla"]["inference_time"], 'b-', label='MLA')
    plt.plot(results["standard"]["seq_length"], results["standard"]["inference_time"], 'r-', label='Standard')
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Speed Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()