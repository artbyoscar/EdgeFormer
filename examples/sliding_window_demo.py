# examples/sliding_window_demo.py
import torch
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def test_long_sequence(model, vocab_size, seq_length, window_size=None):
    print(f"\nTesting with sequence length: {seq_length}")
    
    # Create a random sequence
    input_ids = torch.randint(0, vocab_size, (1, seq_length))
    attention_mask = torch.ones((1, seq_length))
    
    try:
        start_time = time.time()
        with torch.no_grad():
            if window_size:
                # Process in chunks with sliding window
                chunks = []
                for i in range(0, seq_length, window_size // 2):
                    end = min(i + window_size, seq_length)
                    chunk_input = input_ids[:, i:end]
                    chunk_mask = attention_mask[:, i:end]
                    chunk_output = model(input_ids=chunk_input, attention_mask=chunk_mask)
                    
                    # Extract the output hidden states from the output dictionary
                    if isinstance(chunk_output, dict):
                        chunk_hidden_states = chunk_output.get('last_hidden_state', 
                                                              chunk_output.get('hidden_states', 
                                                                              chunk_output.get('logits')))
                    else:
                        chunk_hidden_states = chunk_output
                    
                    # Only keep the second half of each chunk (except the first and last)
                    if i == 0:
                        chunks.append(chunk_hidden_states)
                    elif end == seq_length:
                        chunks.append(chunk_hidden_states)
                    else:
                        mid_point = window_size // 2
                        chunks.append(chunk_hidden_states[:, mid_point:])
                
                # Concatenate chunks
                outputs = torch.cat(chunks, dim=1)
            else:
                # Process the entire sequence at once
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Handle dictionary output if needed
                if isinstance(outputs, dict):
                    outputs = outputs.get('last_hidden_state', 
                                         outputs.get('hidden_states', 
                                                   outputs.get('logits')))
        
        end_time = time.time()
        
        print(f"  Success! Processed in {end_time - start_time:.4f} seconds")
        return True, end_time - start_time
    
    except Exception as e:
        print(f"  Failed: {e}")
        return False, None

def main():
    print("\n=== EdgeFormer Sliding Window Attention Demo ===")
    
    # Create a model for testing
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024
    )
    
    # Initialize model
    model = EdgeFormer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with increasing sequence lengths
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    # Regular processing results
    regular_results = []
    
    for seq_len in sequence_lengths:
        success, time_taken = test_long_sequence(model, 1000, seq_len)
        if success:
            regular_results.append((seq_len, time_taken))
        else:
            break
    
    # Test with sliding window
    print("\n--- Testing with sliding window attention ---")
    window_size = 512
    print(f"Window size: {window_size}")
    
    sliding_results = []
    
    for seq_len in sequence_lengths:
        success, time_taken = test_long_sequence(model, 1000, seq_len, window_size=window_size)
        if success:
            sliding_results.append((seq_len, time_taken))
        else:
            break
    
    # Plot results
    if regular_results or sliding_results:
        plt.figure(figsize=(10, 6))
        
        if regular_results:
            reg_x, reg_y = zip(*regular_results)
            plt.plot(reg_x, reg_y, 'b-o', label='Regular Processing')
        
        if sliding_results:
            sw_x, sw_y = zip(*sliding_results)
            plt.plot(sw_x, sw_y, 'g-o', label='Sliding Window')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Processing Time (seconds)')
        plt.title('EdgeFormer Performance vs Sequence Length')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('sequence_length_benchmark.png')
        print("\nPerformance plot saved as 'sequence_length_benchmark.png'")
    
    # Print summary
    print("\nSummary:")
    max_regular = max(seq for seq, _ in regular_results) if regular_results else 0
    max_sliding = max(seq for seq, _ in sliding_results) if sliding_results else 0
    
    print(f"Maximum sequence length (regular): {max_regular}")
    print(f"Maximum sequence length (sliding window): {max_sliding}")
    
    if max_sliding > max_regular:
        print(f"Sliding window increases maximum context by {max_sliding / max_regular:.2f}x")

if __name__ == "__main__":
    main()