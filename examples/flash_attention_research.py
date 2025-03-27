# examples/flash_attention_research.py
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('edgeformer')

def check_flash_attention():
    """
    Check if FlashAttention is available and which version.
    
    Returns:
        tuple: (available, version)
    """
    try:
        try:
            # Try FlashAttention 2
            from flash_attn import flash_attn_func
            logger.info("FlashAttention 2 is available")
            return True, 2
        except ImportError:
            # Try FlashAttention 1
            from flash_attn.flash_attention import FlashAttention
            logger.info("FlashAttention 1 is available")
            return True, 1
    except ImportError:
        logger.info("FlashAttention is not available")
        return False, None

def benchmark_attention_mechanisms(seq_lengths, batch_size=1, hidden_size=256, num_heads=8, repetitions=3, plot=True):
    """
    Benchmark different attention mechanisms across various sequence lengths.
    
    Args:
        seq_lengths: List of sequence lengths to test
        batch_size: Batch size for testing
        hidden_size: Hidden size
        num_heads: Number of attention heads
        repetitions: Number of repetitions for timing
        plot: Whether to generate a plot
        
    Returns:
        dict: Benchmark results
    """
    # Results containers
    standard_times = []
    mla_times = []
    flash_times = []
    
    # Check if FlashAttention is available
    has_flash, flash_version = check_flash_attention()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Import attention mechanisms
        from src.model.attention import StandardAttention, MultiHeadLatentAttention
        logger.info("Successfully imported EdgeFormer attention mechanisms")
    except ImportError:
        logger.error("Could not import EdgeFormer attention mechanisms. Check your project structure.")
        return None
    
    # Benchmark each sequence length
    progress_bar = tqdm(seq_lengths, desc="Benchmarking sequence lengths")
    for seq_len in progress_bar:
        progress_bar.set_description(f"Benchmarking length {seq_len}")
        
        # Create random inputs
        q = torch.randn(batch_size, seq_len, hidden_size, device=device)
        k = torch.randn(batch_size, seq_len, hidden_size, device=device)
        v = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Standard Attention
        attn = StandardAttention(hidden_size, num_heads)
        attn.to(device)
        
        # Warmup
        _ = attn(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Timing
        start = time.time()
        for _ in range(repetitions):
            _ = attn(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / repetitions
        standard_times.append(avg_time)
        logger.info(f"  Standard Attention ({seq_len}): {avg_time:.4f}s")
        
        # MLA Attention
        attn = MultiHeadLatentAttention(hidden_size, num_heads, latent_size=hidden_size//4)
        attn.to(device)
        
        # Warmup
        _ = attn(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Timing
        start = time.time()
        for _ in range(repetitions):
            _ = attn(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / repetitions
        mla_times.append(avg_time)
        logger.info(f"  MLA Attention ({seq_len}): {avg_time:.4f}s")
        
        # Flash Attention (if available)
        if has_flash:
            # Reshape for Flash Attention
            q_reshaped = q.view(batch_size, seq_len, num_heads, hidden_size // num_heads)
            k_reshaped = k.view(batch_size, seq_len, num_heads, hidden_size // num_heads)
            v_reshaped = v.view(batch_size, seq_len, num_heads, hidden_size // num_heads)
            
            q_reshaped = q_reshaped.transpose(1, 2)  # [B, H, S, D]
            k_reshaped = k_reshaped.transpose(1, 2)  # [B, H, S, D]
            v_reshaped = v_reshaped.transpose(1, 2)  # [B, H, S, D]
            
            if flash_version == 2:
                from flash_attn import flash_attn_func
                
                # Warmup
                _ = flash_attn_func(q_reshaped, k_reshaped, v_reshaped)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                # Timing
                start = time.time()
                for _ in range(repetitions):
                    _ = flash_attn_func(q_reshaped, k_reshaped, v_reshaped)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                end = time.time()
            else:
                from flash_attn.flash_attention import FlashAttention
                flash_attn = FlashAttention(softmax_scale=1.0 / np.sqrt(hidden_size // num_heads))
                
                # Warmup
                _ = flash_attn(q_reshaped, k_reshaped, v_reshaped)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                # Timing
                start = time.time()
                for _ in range(repetitions):
                    _ = flash_attn(q_reshaped, k_reshaped, v_reshaped)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                end = time.time()
            
            avg_time = (end - start) / repetitions
            flash_times.append(avg_time)
            logger.info(f"  Flash Attention v{flash_version} ({seq_len}): {avg_time:.4f}s")
    
    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, standard_times, 'o-', label='Standard Attention')
        plt.plot(seq_lengths, mla_times, 's-', label='MLA')
        if has_flash:
            plt.plot(seq_lengths, flash_times, '^-', label=f'Flash Attention v{flash_version}')
            
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (seconds)')
        plt.title('Attention Mechanisms Performance Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        # Save plot
        plt.savefig('attention_benchmark_results.png')
        logger.info("Benchmark results saved to attention_benchmark_results.png")
    
    # Return results
    results = {
        'seq_lengths': seq_lengths,
        'standard_times': standard_times,
        'mla_times': mla_times
    }
    if has_flash:
        results['flash_times'] = flash_times
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Research and benchmark attention mechanisms")
    parser.add_argument("--min_seq_length", type=int, default=128, help="Minimum sequence length")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--num_lengths", type=int, default=7, help="Number of lengths to test")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions for timing")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Generate sequence lengths (log scale)
    min_exp = int(np.log2(args.min_seq_length))
    max_exp = int(np.log2(args.max_seq_length))
    seq_lengths = [2**exp for exp in range(min_exp, max_exp + 1)]
    
    logger.info(f"Testing sequence lengths: {seq_lengths}")
    
    # Run benchmarks
    results = benchmark_attention_mechanisms(
        seq_lengths=seq_lengths,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        repetitions=args.repetitions,
        plot=not args.no_plot
    )
    
    if results is None:
        logger.error("Benchmark failed. Check previous errors.")
        return
    
    # Print summary
    logger.info("\nPerformance summary:")
    for i, seq_len in enumerate(seq_lengths):
        logger.info(f"Sequence length {seq_len}:")
        logger.info(f"  Standard: {results['standard_times'][i]:.4f}s")
        logger.info(f"  MLA:      {results['mla_times'][i]:.4f}s")
        if 'flash_times' in results:
            logger.info(f"  Flash:    {results['flash_times'][i]:.4f}s")
        
        # Calculate speedups
        std_time = results['standard_times'][i]
        mla_time = results['mla_times'][i]
        mla_speedup = std_time / mla_time if mla_time > 0 else float('inf')
        
        logger.info(f"  MLA speedup vs Standard: {mla_speedup:.2f}x")
        
        if 'flash_times' in results:
            flash_time = results['flash_times'][i]
            flash_vs_std = std_time / flash_time if flash_time > 0 else float('inf')
            flash_vs_mla = mla_time / flash_time if flash_time > 0 else float('inf')
            
            logger.info(f"  Flash speedup vs Standard: {flash_vs_std:.2f}x")
            logger.info(f"  Flash speedup vs MLA: {flash_vs_mla:.2f}x")
    
    # Save results to file
    np.save('attention_benchmark_results.npy', results)
    logger.info("Benchmark data saved to attention_benchmark_results.npy")

if __name__ == "__main__":
    main()