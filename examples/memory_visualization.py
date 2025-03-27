# examples/memory_visualization.py
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from pathlib import Path

def plot_component_memory(component_data, title, output_dir="plots"):
    """Create a bar chart of memory usage across model components by sequence length.
    
    Args:
        component_data: Dictionary with sequence lengths as keys and component memory data as values
        title: Title for the plot
        output_dir: Directory to save the plot
    """
    # Extract sequence lengths and component names
    seq_lengths = sorted(list(component_data.keys()))
    components = list(component_data[seq_lengths[0]].keys())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(components))
    width = 0.25  # Width of the bars
    
    # Plot bars for each sequence length
    for i, seq_len in enumerate(seq_lengths):
        memory_values = [component_data[seq_len][component] for component in components]
        offset = width * (i - len(seq_lengths)/2 + 0.5)
        ax.bar(x + offset, memory_values, width, label=f'{seq_len} tokens')
    
    # Set labels and title
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title(f'Memory Usage Across Model Components by Sequence Length')
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.legend()
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'memory_{title.replace(" ", "_").lower()}.png')
    plt.savefig(output_path)
    print(f"Saved component memory plot to {output_path}")
    plt.close()

def plot_memory_vs_sequence(seq_data, title, output_dir="plots"):
    """Create a line chart of memory usage vs sequence length for different attention types.
    
    Args:
        seq_data: Dictionary with attention types as keys and sequence length memory data as values
        title: Title for the plot
        output_dir: Directory to save the plot
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a logarithmic x-axis for sequence lengths
    for attention_type, data in seq_data.items():
        seq_lengths = sorted(list(data.keys()))
        peak_memory = [data[seq_len] for seq_len in seq_lengths]
        
        # Plot as log2 of sequence length
        log_seq_lengths = [np.log2(float(seq_len)) for seq_len in seq_lengths]
        ax.plot(log_seq_lengths, peak_memory, marker='o', label=attention_type)
    
    # Set labels and title
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title(f'Memory Usage vs Sequence Length')
    
    # Set logarithmic x-axis ticks
    ax.set_xticks([np.log2(2**i) for i in range(7, 14)])
    ax.set_xticklabels([f'2^{i}' for i in range(7, 14)])
    
    # Add grid and legend
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'memory_vs_sequence_{title.replace(" ", "_").lower()}.png')
    plt.savefig(output_path)
    print(f"Saved sequence memory plot to {output_path}")
    plt.close()

def plot_inference_time(time_data, title, output_dir="plots"):
    """Create a line chart of inference time vs sequence length for different attention types.
    
    Args:
        time_data: Dictionary with attention types as keys and sequence length time data as values
        title: Title for the plot
        output_dir: Directory to save the plot
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a logarithmic x-axis for sequence lengths
    for attention_type, data in time_data.items():
        seq_lengths = sorted(list(data.keys()))
        inference_times = [data[seq_len] for seq_len in seq_lengths]
        
        # Plot as log2 of sequence length
        log_seq_lengths = [np.log2(float(seq_len)) for seq_len in seq_lengths]
        ax.plot(log_seq_lengths, inference_times, marker='o', label=attention_type)
    
    # Set labels and title
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Inference Time (s)')
    ax.set_title(f'Inference Time vs Sequence Length')
    
    # Set logarithmic x-axis ticks
    ax.set_xticks([np.log2(2**i) for i in range(7, 14)])
    ax.set_xticklabels([f'2^{i}' for i in range(7, 14)])
    
    # Add grid and legend
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'inference_time_{title.replace(" ", "_").lower()}.png')
    plt.savefig(output_path)
    print(f"Saved inference time plot to {output_path}")
    plt.close()

def plot_memory_comparison(data, title, output_dir="plots"):
    """Create a comprehensive memory visualization from test data.
    
    Args:
        data: Dictionary containing memory and inference time data
        title: Title for the plots
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot component-level memory usage
    if "component_memory" in data:
        plot_component_memory(data["component_memory"], title, output_dir)
    
    # Plot memory vs sequence length
    if "sequence_memory" in data:
        plot_memory_vs_sequence(data["sequence_memory"], title, output_dir)
    
    # Plot inference time vs sequence length
    if "inference_time" in data:
        plot_inference_time(data["inference_time"], title, output_dir)
    
    # Create a combined plot with memory and inference time
    if "sequence_memory" in data and "inference_time" in data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Memory usage plot
        for attention_type, mem_data in data["sequence_memory"].items():
            seq_lengths = sorted(list(mem_data.keys()))
            peak_memory = [mem_data[seq_len] for seq_len in seq_lengths]
            log_seq_lengths = [np.log2(float(seq_len)) for seq_len in seq_lengths]
            ax1.plot(log_seq_lengths, peak_memory, marker='o', label=attention_type)
        
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage vs Sequence Length')
        ax1.grid(True, which="both", linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Inference time plot
        for attention_type, time_data in data["inference_time"].items():
            seq_lengths = sorted(list(time_data.keys()))
            inference_times = [time_data[seq_len] for seq_len in seq_lengths]
            log_seq_lengths = [np.log2(float(seq_len)) for seq_len in seq_lengths]
            ax2.plot(log_seq_lengths, inference_times, marker='o', label=attention_type)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Inference Time (s)')
        ax2.set_title('Inference Time vs Sequence Length')
        ax2.set_xticks([np.log2(2**i) for i in range(7, 14)])
        ax2.set_xticklabels([f'2^{i}' for i in range(7, 14)])
        ax2.grid(True, which="both", linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, f'combined_{title.replace(" ", "_").lower()}.png')
        plt.savefig(combined_path)
        print(f"Saved combined plot to {combined_path}")
        plt.close()

def parse_benchmark_data(file_path):
    """Parse benchmark data from a file.
    
    Args:
        file_path: Path to the benchmark data file
        
    Returns:
        Parsed data dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Visualize EdgeFormer memory and performance data")
    parser.add_argument("--data", type=str, required=True, help="Path to benchmark data JSON file")
    parser.add_argument("--title", type=str, default="EdgeFormer", help="Title for the plots")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()
    
    # Parse data from file
    data = parse_benchmark_data(args.data)
    
    # Create visualizations
    plot_memory_comparison(data, args.title, args.output_dir)

if __name__ == "__main__":
    main()