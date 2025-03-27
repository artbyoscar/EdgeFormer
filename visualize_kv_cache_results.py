# visualize_kv_cache_results.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def visualize_results(data_file=None):
    """Create detailed visualizations of KV cache benchmark results"""
    # If no file specified, use the most recent
    if data_file is None:
        files = glob.glob("benchmark_results/kv_cache_benchmark_data_*.npy")
        if not files:
            print("No benchmark data found")
            return
        data_file = max(files, key=os.path.getctime)
    
    # Load data
    results = np.load(data_file, allow_pickle=True)
    print(f"Loaded data from {data_file}")
    
    # Filter successful results
    successful_results = [r for r in results if r.get("success", False)]
    
    # Split results by offloading
    no_offload_results = [r for r in successful_results if not r["offloading"]]
    offload_results = [r for r in successful_results if r["offloading"]]
    
    # Create a comprehensive dashboard with multiple plots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Memory Usage
    ax1 = fig.add_subplot(2, 2, 1)
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
    
    # 2. Inference Speed (First Pass)
    ax2 = fig.add_subplot(2, 2, 2)
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
    ax2.set_title('Inference Speed (First Pass)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Second Pass Time
    ax3 = fig.add_subplot(2, 2, 3)
    if no_offload_results:
        x1 = [r["sequence_length"] for r in no_offload_results]
        y1 = [r["second_pass_time"] for r in no_offload_results]
        ax3.plot(x1, y1, 'b-o', label='Without Offloading')
    
    if offload_results:
        x2 = [r["sequence_length"] for r in offload_results]
        y2 = [r["second_pass_time"] for r in offload_results]
        ax3.plot(x2, y2, 'r-o', label='With Offloading')
    
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Continuation Time (Second Pass)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Memory Savings
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Only calculate for sequence lengths that have both offloading and no offloading
    common_seq_lengths = []
    memory_savings = []
    
    for off_result in offload_results:
        seq_length = off_result["sequence_length"]
        # Find matching no-offload result
        matching = [r for r in no_offload_results if r["sequence_length"] == seq_length]
        if matching:
            no_off_result = matching[0]
            saving = (no_off_result["memory_increase"] - off_result["memory_increase"]) / no_off_result["memory_increase"] * 100
            common_seq_lengths.append(seq_length)
            memory_savings.append(saving)
    
    if common_seq_lengths:
        ax4.bar(common_seq_lengths, memory_savings, color='green')
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Memory Savings (%)')
        ax4.set_title('Memory Savings with KV Cache Offloading')
        # Add value labels
        for i, v in enumerate(memory_savings):
            ax4.text(common_seq_lengths[i], v + 1, f"{v:.1f}%", ha='center')
        ax4.grid(True, axis='y')
    
    # Add overall title
    plt.suptitle('KV Cache Offloading Performance Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save the plot
    timestamp = os.path.basename(data_file).split('_')[-1].split('.')[0]
    output_file = f"benchmark_results/kv_cache_performance_dashboard_{timestamp}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Dashboard saved to {output_file}")
    
    # Display plot if running interactively
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize KV cache benchmark results")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to benchmark data file (.npy)")
    args = parser.parse_args()
    
    visualize_results(args.data_file)