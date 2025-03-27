# examples/profile_mla.py
import torch
import time
import sys
import os
import cProfile
import pstats
from pstats import SortKey
import io

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def profile_model(model, input_ids, attention_mask, num_runs=5):
    """Profile model execution."""
    # Warmup
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Profile
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(num_runs):
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    
    pr.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    print(s.getvalue())
    
    # Save profile results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ps.dump_stats(f"profile_results_{timestamp}.prof")
    print(f"Full profile saved to profile_results_{timestamp}.prof")

def main():
    # Configure models
    print("\n=== Profiling MLA Implementation ===")
    
    # Create test configuration
    mla_config = EdgeFormerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
        max_position_embeddings=8192,
    )
    
    # Create model
    print("Initializing MLA model...")
    model = EdgeFormer(mla_config)
    model.eval()
    
    # Test with different sequence lengths
    sequence_lengths = [512, 1024, 4096, 8192]
    
    for seq_len in sequence_lengths:
        print(f"\nProfiling with sequence length: {seq_len}")
        input_ids = torch.randint(0, mla_config.vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)
        
        # Time execution
        start_time = time.time()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f}s")
        
        # Profile execution
        print(f"Profiling execution for sequence length {seq_len}...")
        profile_model(model, input_ids, attention_mask)
        
        print("-" * 50)

if __name__ == "__main__":
    main()