# examples/test_long_sequences.py
import torch
import time
import sys
import os
import gc

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.memory_tracking import measure_memory_usage, MemoryTracker

def test_sequence_length(model, seq_len, verbose=True):
    """Test a specific sequence length."""
    try:
        # Create input tensors
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)
        
        # Track memory
        tracker = MemoryTracker()
        tracker.start_tracking()
        
        # Take initial snapshot
        tracker.take_snapshot("initial")
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        inference_time = time.time() - start_time
        
        # Take post-inference snapshot
        tracker.take_snapshot("post_inference")
        
        # Print stats
        if verbose:
            print(f"✅ Sequence length {seq_len}: Inference time = {inference_time:.4f}s")
            tracker.print_stats("initial", "post_inference")
        
        return True, inference_time
    except Exception as e:
        if verbose:
            print(f"❌ Failed at sequence length {seq_len}: {str(e)}")
        return False, 0

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test long sequence handling")
    parser.add_argument("--start", type=int, default=4096, help="Starting sequence length")
    parser.add_argument("--end", type=int, default=32768, help="Maximum sequence length to test")
    parser.add_argument("--step", type=int, default=4096, help="Step size for sequence lengths")
    parser.add_argument("--attention", type=str, default="mla", 
                        choices=["standard", "mla", "mla_window"],
                        help="Attention type to test")
    args = parser.parse_args()
    
    # Configure model based on attention type
    max_pos_embeddings = args.end
    
    if args.attention == "standard":
        config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            intermediate_size=1024,
            max_position_embeddings=max_pos_embeddings,
        )
        attention_name = "Standard Attention"
    elif args.attention == "mla":
        config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            max_position_embeddings=max_pos_embeddings,
        )
        attention_name = "Multi-Head Latent Attention"
    else:  # mla_window
        config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            use_sliding_window=True,
            sliding_window_size=512,
            max_position_embeddings=max_pos_embeddings,
        )
        attention_name = "MLA with Sliding Window"
    
    print(f"Testing {attention_name} with long sequences")
    
    # Initialize model
    model = EdgeFormer(config)
    model.eval()
    
    # Test sequence lengths
    results = []
    
    for seq_len in range(args.start, args.end + args.step, args.step):
        print(f"\nTesting sequence length: {seq_len}")
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Test sequence length
        success, time_taken = test_sequence_length(model, seq_len)
        
        # Store results
        results.append((seq_len, success, time_taken))
        
        # Break if failed
        if not success:
            break
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Model: {attention_name}")
    print(f"{'Sequence Length':<15} {'Success':<10} {'Time (s)':<10}")
    print("-" * 35)
    
    for seq_len, success, time_taken in results:
        status = "✅" if success else "❌"
        time_str = f"{time_taken:.4f}" if success else "N/A"
        print(f"{seq_len:<15} {status:<10} {time_str:<10}")

if __name__ == "__main__":
    main()