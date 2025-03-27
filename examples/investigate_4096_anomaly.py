# examples/investigate_4096_anomaly.py
import torch
import gc
import time
import sys
import os
import psutil

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_sequence_length(model, seq_len):
    """Test specific sequence length with detailed memory tracking"""
    # Create input tensors
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
    attention_mask = torch.ones(1, seq_len)
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(1.0)
    
    # Baseline memory
    baseline_memory = measure_memory()
    print(f"Baseline memory: {baseline_memory:.2f} MB")
    
    # Track memory at each step
    memories = {}
    
    # Embeddings
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.embeddings(input_ids=input_ids)
        memories["after_embeddings"] = measure_memory() - baseline_memory
        
        # Process through each layer and track memory
        hidden_states = inputs_embeds
        for i, layer in enumerate(model.layers):
            hidden_states = layer(hidden_states, attention_mask=None)[0]
            memories[f"after_layer_{i}"] = measure_memory() - baseline_memory
        
        # Final layer norm
        hidden_states = model.ln_f(hidden_states)
        memories["after_final_ln"] = measure_memory() - baseline_memory
        
        # LM head
        logits = model.lm_head(hidden_states)
        memories["after_lm_head"] = measure_memory() - baseline_memory
    
    # Print memory usage at each step
    print(f"\nMemory usage for sequence length {seq_len}:")
    print(f"{'Component':<20} {'Memory (MB)':<15}")
    print("-" * 35)
    for component, memory in memories.items():
        print(f"{component:<20} {memory:<15.2f}")
    
    return memories

def main():
    # Test sequence lengths around the anomaly
    sequence_lengths = [3584, 4096, 4608]
    
    # Standard attention config
    config = EdgeFormerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
        intermediate_size=1024,
        max_position_embeddings=8192,
    )
    
    # Initialize model
    model = EdgeFormer(config)
    model.eval()
    
    results = {}
    for seq_len in sequence_lengths:
        print(f"\n==== Testing sequence length {seq_len} ====")
        results[seq_len] = test_sequence_length(model, seq_len)
    
    # Compare memory usage between sequence lengths
    print("\n==== Memory Comparison ====")
    print(f"{'Component':<20}", end="")
    for seq_len in sequence_lengths:
        print(f" {seq_len:<10}", end="")
    print()
    print("-" * (20 + 10 * len(sequence_lengths)))
    
    for component in results[sequence_lengths[0]].keys():
        print(f"{component:<20}", end="")
        for seq_len in sequence_lengths:
            print(f" {results[seq_len][component]:<10.2f}", end="")
        print()

if __name__ == "__main__":
    main()