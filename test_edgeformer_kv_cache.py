import torch
import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("edgeformer")

# Add the current directory to path
sys.path.append('.')

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.kv_cache_manager import KVCacheManager

def test_edgeformer_with_kv_cache():
    """Test EdgeFormer with KV Cache offloading"""
    print("Testing EdgeFormer with KV Cache offloading...")
    
    # Load vocabulary
    vocab_path = "data/focused/vocab.pt"
    vocab = torch.load(vocab_path)
    vocab_size = vocab["vocab_size"]
    char_to_idx = vocab["char_to_idx"]
    
    # Create EdgeFormer model
    config = EdgeFormerConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_type="standard"
    )
    
    model = EdgeFormer(config)
    model.eval()
    
    # Test different sequence lengths
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    memory_usage = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting with sequence length: {seq_len}")
        
        # Create input tensor
        input_text = "EdgeFormer" * (seq_len // 10 + 1)
        input_text = input_text[:seq_len]
        input_ids = torch.tensor([[char_to_idx.get(char, 0) for char in input_text]])
        
        print(f"Input shape: {input_ids.shape}")
        
        # Reset model state
        if model.kv_cache_manager is not None:
            model.kv_cache_manager.reset()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check KV Cache
        if model.kv_cache_manager is not None:
            gpu_cache_size = model.kv_cache_manager._estimate_gpu_cache_size() / (1024 * 1024)  # MB
            print(f"GPU cache size: {gpu_cache_size:.2f} MB")
            memory_usage.append(gpu_cache_size)
        else:
            print("KV Cache Manager not initialized")
            memory_usage.append(0)
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, memory_usage, "b-o")
    plt.xlabel("Sequence Length")
    plt.ylabel("GPU Memory Usage (MB)")
    plt.title("EdgeFormer KV Cache Memory Usage")
    plt.grid(True)
    plt.savefig("plots/edgeformer_kv_cache.png")
    
    print("Test completed. Results saved to plots/edgeformer_kv_cache.png")

if __name__ == "__main__":
    test_edgeformer_with_kv_cache()