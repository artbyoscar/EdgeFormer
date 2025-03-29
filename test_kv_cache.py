import torch
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("edgeformer")

# Add the root directory to the path
sys.path.append(".")
from src.utils.kv_cache_manager import KVCacheManager

def test_kv_cache_manager():
    """Test the KV Cache Manager functionality"""
    print("Testing KV Cache Manager...")
    
    # Create the KV Cache Manager
    num_layers = 4
    num_heads = 8
    head_dim = 64
    max_batch_size = 1
    max_seq_length = 8192
    
    cache_manager = KVCacheManager(
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_gpu_cache_size=256,  # 256 MB max GPU cache size
        enable_offload=True,
        device="cpu"  # Use CPU for testing
    )
    
    print(f"Cache manager initialized")
    
    # Test updating the cache
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 6144, 8192]
    gpu_memory = []
    cpu_memory = []
    
    for seq_len in sequence_lengths:
        print(f"\nTesting with sequence length {seq_len}")
        cache_manager.reset()
        
        # Add tokens incrementally
        tokens_per_update = 128
        updates_needed = seq_len // tokens_per_update
        
        for i in range(updates_needed):
            # Create dummy key and value tensors
            key = torch.rand(
                max_batch_size,
                tokens_per_update,
                num_heads,
                head_dim,
                device="cpu"
            )
            value = torch.rand(
                max_batch_size,
                tokens_per_update,
                num_heads,
                head_dim,
                device="cpu"
            )
            
            # Update each layer
            for layer_idx in range(num_layers):
                cache_manager.update(layer_idx, key, value)
            
            # Print progress
            if (i + 1) % 4 == 0:
                print(f"  Updated {(i + 1) * tokens_per_update} tokens")
        
        # Check memory usage
        gpu_cache_size = cache_manager._estimate_gpu_cache_size() / (1024 * 1024)  # MB
        gpu_memory.append(gpu_cache_size)
        
        # Check CPU cache size
        cpu_cache_size = 0
        for layer_idx in range(num_layers):
            if cache_manager.cpu_cache_k[layer_idx] is not None:
                cpu_tokens = cache_manager.cpu_cache_k[layer_idx].size(1)
                cpu_bytes = (
                    cpu_tokens * 
                    max_batch_size * 
                    num_heads * 
                    head_dim * 
                    2 *  # Keys and values
                    4    # Float32 - 4 bytes
                )
                cpu_cache_size += cpu_bytes / (1024 * 1024)  # MB
        
        cpu_memory.append(cpu_cache_size)
        
        print(f"  GPU cache size: {gpu_cache_size:.2f} MB")
        print(f"  CPU cache size: {cpu_cache_size:.2f} MB")
        
        # Test retrieving from cache
        print("  Testing retrieving from cache")
        start_idx = 0
        end_idx = min(128, seq_len)
        
        for layer_idx in range(num_layers):
            k, v = cache_manager.get(layer_idx, start_idx, end_idx)
            print(f"    Layer {layer_idx}: Retrieved shape K={k.shape}, V={v.shape}")
    
    # Create a plot directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(sequence_lengths, gpu_memory, "b-", label="GPU Memory (MB)")
    plt.plot(sequence_lengths, cpu_memory, "r-", label="CPU Memory (MB)")
    plt.plot(sequence_lengths, [x + y for x, y in zip(gpu_memory, cpu_memory)], "g--", label="Total Memory (MB)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory Usage (MB)")
    plt.title("KV Cache Memory Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/kv_cache_memory.png")
    
    # Plot memory distribution ratio
    plt.figure(figsize=(10, 6))
    cpu_ratio = [c / (g + c) * 100 if g + c > 0 else 0 for g, c in zip(gpu_memory, cpu_memory)]
    gpu_ratio = [g / (g + c) * 100 if g + c > 0 else 0 for g, c in zip(gpu_memory, cpu_memory)]
    
    plt.bar(np.arange(len(sequence_lengths)), gpu_ratio, label="GPU Memory %", color="blue")
    plt.bar(np.arange(len(sequence_lengths)), cpu_ratio, bottom=gpu_ratio, label="CPU Memory %", color="red")
    
    plt.xticks(np.arange(len(sequence_lengths)), sequence_lengths)
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory Distribution (%)")
    plt.title("KV Cache Memory Distribution Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/kv_cache_ratio.png")
    
    print("\nMemory usage test completed. Results saved to plots/")

if __name__ == "__main__":
    test_kv_cache_manager()