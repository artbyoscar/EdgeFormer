# test_long_sequence_kv.py
import torch
import logging
import argparse
import time
import os
import psutil
import matplotlib.pyplot as plt
import numpy as np
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import kv_cache_offload

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("long_sequence_test")

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def test_sequence_length(model, seq_length, use_offloading=True, device="cpu"):
    """Test model with specific sequence length"""
    logger.info(f"Testing sequence length: {seq_length} with offloading={use_offloading}")
    
    # Create input tensors
    config = model.config
    input_ids = torch.randint(0, config.vocab_size, (1, seq_length))
    attention_mask = torch.ones(1, seq_length)
    
    # Apply offloading if requested
    test_model = model
    if use_offloading:
        test_model = kv_cache_offload(model)
        logger.info("KV cache offloading enabled")
    
    # Measure initial memory
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # First forward pass with timing
    start_time = time.time()
    try:
        outputs = test_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        # Get the time for first pass
        first_pass_time = time.time() - start_time
        logger.info(f"First pass completed in {first_pass_time:.4f} seconds")
        
        # Measure peak memory after first pass
        peak_memory = get_memory_usage()
        memory_increase = peak_memory - initial_memory
        logger.info(f"Peak memory after first pass: {peak_memory:.2f} MB (increase: {memory_increase:.2f} MB)")
        
        # Continue with second pass (next token prediction)
        if "past_key_values" in outputs and outputs["past_key_values"] is not None:
            past_key_values = outputs["past_key_values"]
            logger.info(f"Past key values type: {type(past_key_values)}")
            
            # Create next tokens (just a single token for continuation)
            next_input_ids = torch.randint(0, config.vocab_size, (1, 1))
            
            # Important fix: Create the correct attention mask for the second pass
            # The attention mask should cover past tokens + new token
            next_attention_mask = torch.ones(1, seq_length + 1)
            
            # Second forward pass with timing
            start_time = time.time()
            try:
                next_outputs = test_model(
                    input_ids=next_input_ids,
                    attention_mask=next_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                second_pass_time = time.time() - start_time
                logger.info(f"Second pass completed in {second_pass_time:.4f} seconds")
                
                # Get final memory usage
                final_memory = get_memory_usage()
                logger.info(f"Final memory usage: {final_memory:.2f} MB")
                
                return {
                    "sequence_length": seq_length,
                    "offloading": use_offloading,
                    "initial_memory": initial_memory,
                    "peak_memory": peak_memory,
                    "final_memory": final_memory,
                    "memory_increase": memory_increase,
                    "first_pass_time": first_pass_time,
                    "second_pass_time": second_pass_time,
                    "tokens_per_second": seq_length / first_pass_time,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Error during second pass: {str(e)}")
                return {
                    "sequence_length": seq_length,
                    "offloading": use_offloading,
                    "initial_memory": initial_memory,
                    "peak_memory": peak_memory,
                    "memory_increase": memory_increase,
                    "first_pass_time": first_pass_time,
                    "success": False,
                    "error": f"Second pass error: {str(e)}"
                }
        else:
            logger.error("No past_key_values in output")
            return {
                "sequence_length": seq_length,
                "offloading": use_offloading,
                "initial_memory": initial_memory,
                "peak_memory": peak_memory,
                "memory_increase": memory_increase,
                "first_pass_time": first_pass_time,
                "success": False,
                "error": "No past_key_values in output"
            }
    except Exception as e:
        logger.error(f"Error during first pass: {str(e)}")
        return {
            "sequence_length": seq_length,
            "offloading": use_offloading,
            "success": False,
            "error": f"First pass error: {str(e)}"
        }

def run_sequence_scaling_tests(model, seq_lengths):
    """Run tests for multiple sequence lengths with and without offloading"""
    results = []
    
    for seq_length in seq_lengths:
        # First test with offloading
        offload_result = test_sequence_length(
            model, seq_length, use_offloading=True
        )
        results.append(offload_result)
        
        # Only test without offloading for shorter sequences to avoid OOM
        if seq_length <= 4096:
            no_offload_result = test_sequence_length(
                model, seq_length, use_offloading=False
            )
            results.append(no_offload_result)
        else:
            logger.info(f"Skipping sequence length {seq_length} without offloading (too long)")
    
    return results

def plot_results(results):
    """Create visualization of the test results"""
    # Filter successful results
    successful_results = [r for r in results if r.get("success", False)]
    
    # Split results by offloading
    no_offload_results = [r for r in successful_results if not r["offloading"]]
    offload_results = [r for r in successful_results if r["offloading"]]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot memory usage
    if no_offload_results:
        x1 = [r["sequence_length"] for r in no_offload_results]
        y1 = [r["memory_increase"] for r in no_offload_results]
        ax1.plot(x1, y1, 'b-o', label='Without Offloading')
    
    if offload_results:
        x2 = [r["sequence_length"] for r in offload_results]
        y2 = [r["memory_increase"] for r in offload_results]
        ax1.plot(x2, y2, 'r-o', label='With Offloading')
    
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Memory Usage Increase (MB)')
    ax1.set_title('Memory Usage vs Sequence Length')
    ax1.legend()
    ax1.grid(True)
    
    # Plot inference speed
    if no_offload_results:
        x1 = [r["sequence_length"] for r in no_offload_results]
        y1 = [r["tokens_per_second"] for r in no_offload_results]
        ax2.plot(x1, y1, 'b-o', label='Without Offloading')
    
    if offload_results:
        x2 = [r["sequence_length"] for r in offload_results]
        y2 = [r["tokens_per_second"] for r in offload_results]
        ax2.plot(x2, y2, 'r-o', label='With Offloading')
    
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('Inference Speed vs Sequence Length')
    ax2.legend()
    ax2.grid(True)
    
    # Save the plot
    plt.tight_layout()
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"benchmark_results/kv_cache_benchmark_{timestamp}.png")
    logger.info(f"Plot saved to benchmark_results/kv_cache_benchmark_{timestamp}.png")
    
    # Also save the data
    np.save(f"benchmark_results/kv_cache_benchmark_data_{timestamp}.npy", results)
    logger.info(f"Data saved to benchmark_results/kv_cache_benchmark_data_{timestamp}.npy")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Test KV cache offloading with long sequences")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum sequence length to test")
    parser.add_argument("--hidden-size", type=int, default=256, help="Model hidden size")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--latent-factor", type=int, default=8, 
                        help="Latent size factor for MLA (hidden_size/factor)")
    args = parser.parse_args()
    
    # Initialize model
    logger.info("Initializing model...")
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        latent_size_factor=args.latent_factor,
        max_position_embeddings=args.max_length + 1024,  # Add buffer
    )
    
    model = EdgeFormer(config)
    model.eval()
    
    # Define sequence lengths to test (exponential scaling)
    seq_lengths = [
        512,    # 0.5k
        1024,   # 1k
        2048,   # 2k
        4096,   # 4k
    ]
    
    # Add 8k and 16k if requested
    if args.max_length >= 8192:
        seq_lengths.append(8192)  # 8k
    if args.max_length >= 16384:
        seq_lengths.append(16384)  # 16k
    
    # Only test up to max_length
    seq_lengths = [sl for sl in seq_lengths if sl <= args.max_length]
    
    # Run the tests
    logger.info(f"Running tests for sequence lengths: {seq_lengths}")
    results = run_sequence_scaling_tests(model, seq_lengths)
    
    # Create visualization
    plot_results(results)
    
    # Print summary
    logger.info("Test Summary:")
    for result in results:
        if result["success"]:
            logger.info(
                f"Sequence Length: {result['sequence_length']}, "
                f"Offloading: {result['offloading']}, "
                f"Memory Increase: {result.get('memory_increase', 'N/A'):.2f} MB, "
                f"Tokens/sec: {result.get('tokens_per_second', 'N/A'):.2f}"
            )
        else:
            logger.info(
                f"Sequence Length: {result['sequence_length']}, "
                f"Offloading: {result['offloading']}, "
                f"Failed: {result.get('error', 'Unknown error')}"
            )

if __name__ == "__main__":
    main()