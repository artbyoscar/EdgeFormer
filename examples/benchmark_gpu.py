# examples/benchmark_gpu.py
import torch
import time
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def run_benchmark(model, input_ids, attention_mask, num_runs=10):
    """Run inference multiple times and measure average performance."""
    total_time = 0.0
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Benchmark
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / num_runs
    return avg_time

def main():
    print("\n=== EdgeFormer CPU Benchmark ===")
    
    # Create models with different configurations for testing
    configs = [
        ("Small", EdgeFormerConfig(vocab_size=1000, hidden_size=256, num_hidden_layers=4, num_attention_heads=8, intermediate_size=1024)),
        ("Medium", EdgeFormerConfig(vocab_size=1000, hidden_size=512, num_hidden_layers=8, num_attention_heads=16, intermediate_size=2048)),
    ]
    
    for name, config in configs:
        print(f"\nBenchmarking {name} model...")
        model = EdgeFormer(config).to(torch.device("cpu"))
        
        # Print model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        # Test with different sequence lengths
        for seq_len in [128, 256, 512]:
            print(f"\n  Sequence length: {seq_len}")
            input_ids = torch.randint(0, 1000, (1, seq_len))
            attention_mask = torch.ones((1, seq_len))
            
            # Run benchmark
            avg_time = run_benchmark(model, input_ids, attention_mask)
            print(f"  Average inference time: {avg_time:.4f} seconds")
            print(f"  Tokens per second: {seq_len / avg_time:.2f}")

if __name__ == "__main__":
    main()