# examples/long_context_benchmark.py
import torch
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def estimate_memory_usage(model, seq_len, batch_size=1, hidden_size=256, num_attention_heads=8, num_hidden_layers=4):
    """Estimate memory usage for standard attention vs MLA."""
    # Parameter memory is the same for both
    param_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Activations memory - for simplicity we focus on KV cache
    bytes_per_element = 4  # float32
    
    # Standard attention KV cache: 2 * batch_size * seq_len * num_heads * head_dim * num_layers * bytes_per_element
    head_dim = hidden_size // num_attention_heads
    std_kv_cache = 2 * batch_size * seq_len * num_attention_heads * head_dim * num_hidden_layers * bytes_per_element
    
    # MLA KV cache: 2 * batch_size * seq_len * latent_size * num_layers * bytes_per_element
    latent_size = 32  # assuming latent size of 32
    mla_kv_cache = 2 * batch_size * seq_len * latent_size * num_hidden_layers * bytes_per_element
    
    # Convert to MB
    param_mem_mb = param_mem_bytes / (1024 * 1024)
    std_kv_cache_mb = std_kv_cache / (1024 * 1024)
    mla_kv_cache_mb = mla_kv_cache / (1024 * 1024)
    
    return {
        'parameters_mb': param_mem_mb,
        'standard_kv_cache_mb': std_kv_cache_mb,
        'mla_kv_cache_mb': mla_kv_cache_mb,
        'standard_total_mb': param_mem_mb + std_kv_cache_mb,
        'mla_total_mb': param_mem_mb + mla_kv_cache_mb,
        'reduction_factor': std_kv_cache / mla_kv_cache
    }

def test_max_length(model, start_len=128, max_len=16384, step_factor=2):
    """Test progressively longer sequences until we hit memory limits."""
    results = []
    
    seq_len = start_len
    while seq_len <= max_len:
        try:
            print(f"\nTesting sequence length: {seq_len}")
            input_ids = torch.randint(0, 1000, (1, seq_len))
            attention_mask = torch.ones((1, seq_len))
            
            start_time = time.time()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()
            
            inference_time = end_time - start_time
            memory_stats = estimate_memory_usage(model, seq_len)
            
            results.append({
                'seq_len': seq_len,
                'success': True,
                'inference_time': inference_time,
                'inference_tokens_per_sec': seq_len / inference_time,
                'memory_stats': memory_stats
            })
            
            print(f"  Success! Processed in {inference_time:.4f} seconds ({seq_len / inference_time:.2f} tokens/sec)")
            print(f"  Estimated memory usage: {memory_stats['mla_total_mb']:.2f} MB")
            
            # Increase sequence length for next iteration
            seq_len = int(seq_len * step_factor)
            
        except Exception as e:
            print(f"  Failed at sequence length {seq_len}: {e}")
            results.append({
                'seq_len': seq_len,
                'success': False,
                'error': str(e)
            })
            break
    
    return results

def implement_sliding_window(model, seq_len, window_size):
    """Process a long sequence using sliding window approach."""
    try:
        print(f"\nProcessing sequence length {seq_len} with window size {window_size}")
        input_ids = torch.randint(0, 1000, (1, seq_len))
        attention_mask = torch.ones((1, seq_len))
        
        # Process in chunks with sliding window
        start_time = time.time()
        with torch.no_grad():
            chunks = []
            for i in range(0, seq_len, window_size // 2):
                end = min(i + window_size, seq_len)
                print(f"  Processing chunk {i}-{end}...")
                chunk_input = input_ids[:, i:end]
                chunk_mask = attention_mask[:, i:end]
                
                chunk_output = model(input_ids=chunk_input, attention_mask=chunk_mask)
                
                # Extract hidden states if output is a dictionary
                if isinstance(chunk_output, dict):
                    chunk_output = chunk_output.get('last_hidden_state', 
                                                  chunk_output.get('hidden_states', 
                                                                 list(chunk_output.values())[0]))
                
                # Only keep the second half of each chunk (except the first and last)
                if i == 0:
                    chunks.append(chunk_output)
                elif end == seq_len:
                    chunks.append(chunk_output)
                else:
                    mid_point = window_size // 2
                    chunks.append(chunk_output[:, mid_point:])
            
            # Concatenate chunks
            final_output = torch.cat(chunks, dim=1)
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"  Success! Processed in {total_time:.4f} seconds ({seq_len / total_time:.2f} tokens/sec)")
        return True, total_time
    
    except Exception as e:
        print(f"  Failed: {e}")
        return False, None

def plot_results(results):
    """Plot the results of sequence length tests."""
    # Extract data for successful runs
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("No successful runs to plot.")
        return
    
    seq_lengths = [r['seq_len'] for r in successful]
    inference_times = [r['inference_time'] for r in successful]
    tokens_per_sec = [r['inference_tokens_per_sec'] for r in successful]
    memory_usage = [r['memory_stats']['mla_total_mb'] for r in successful]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot inference time
    ax1.plot(seq_lengths, inference_times, 'b-o')
    ax1.set_title('Inference Time vs Sequence Length')
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(True)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    
    # Plot memory usage
    ax2.plot(seq_lengths, memory_usage, 'r-o')
    ax2.set_title('Memory Usage vs Sequence Length')
    ax2.set_xlabel('Sequence Length (tokens)')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.grid(True)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    
    plt.tight_layout()
    plt.savefig('long_context_benchmark.png')
    print("\nResults plot saved as 'long_context_benchmark.png'")

def main():
    print("\n=== EdgeFormer Long Context Benchmark ===")
    
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
    
    # Test progressively longer sequences
    results = test_max_length(model)
    
    # If we found a limit, try with sliding window
    if results and not results[-1]['success']:
        max_success = max([r['seq_len'] for r in results if r['success']])
        
        print(f"\n--- Testing with sliding window attention ---")
        window_size = 512
        print(f"Window size: {window_size}")
        
        # Try a sequence that was too long for standard processing
        if len(results) > 1:
            failed_seq_len = results[-1]['seq_len']
            sliding_success, sliding_time = implement_sliding_window(model, failed_seq_len, window_size)
            
            if sliding_success:
                print(f"\nSliding window enabled processing {failed_seq_len} tokens (previously failed)")
                
                # Try an even longer sequence
                longer_seq_len = failed_seq_len * 2
                sliding_success, sliding_time = implement_sliding_window(model, longer_seq_len, window_size)
                
                if sliding_success:
                    print(f"\nSliding window successfully processed {longer_seq_len} tokens")
    
    # Plot the results
    plot_results(results)
    
    # Print summary
    successful_lens = [r['seq_len'] for r in results if r['success']]
    if successful_lens:
        max_len = max(successful_lens)
        print(f"\nSummary:")
        print(f"Maximum sequence length: {max_len}")
        
        # Memory efficiency
        last_success = next(r for r in reversed(results) if r['success'])
        reduction = last_success['memory_stats']['reduction_factor']
        print(f"KV cache memory reduction from MLA: {reduction:.2f}x")
        
        # Theoretical max under 2GB memory constraint
        max_gpu_mem_gb = 2
        max_gpu_mem_mb = max_gpu_mem_gb * 1024
        mla_usage_per_token = last_success['memory_stats']['mla_total_mb'] / last_success['seq_len']
        theoretical_max = int(max_gpu_mem_mb / mla_usage_per_token)
        print(f"Theoretical max context with {max_gpu_mem_gb}GB memory: ~{theoretical_max:,} tokens")

if __name__ == "__main__":
    main()