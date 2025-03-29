#!/usr/bin/env python
# visualize_cross_device.py

import argparse
import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_benchmark_results(input_dir):
    """Load all benchmark results from the input directory."""
    results = []
    
    for result_file in glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
                if "device" in data and "results" in data:
                    # Extract data for each sequence length
                    for seq_len, metrics in data["results"].items():
                        if "error" not in metrics:
                            results.append({
                                "device": data["device"],
                                "model_size": data.get("model_size", "unknown"),
                                "sequence_length": int(seq_len),
                                "time_seconds": metrics.get("avg_time_seconds", None),
                                "tokens_per_second": metrics.get("tokens_per_second", None),
                                "memory_mb": metrics.get("avg_memory_mb", None)
                            })
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return pd.DataFrame(results)

def plot_benchmarks(df, output_file):
    """Create visualizations of benchmark results."""
    if df.empty:
        print("No valid benchmark data found.")
        return
    
    # Set up the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 12))
    
    # 1. Tokens per second by sequence length for each device
    plt.subplot(2, 2, 1)
    sns.lineplot(
        data=df, 
        x="sequence_length", 
        y="tokens_per_second", 
        hue="device",
        style="model_size",
        markers=True
    )
    plt.title("Tokens per Second by Sequence Length")
    plt.xscale("log", base=2)
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Tokens per Second")
    plt.grid(True)
    
    # 2. Memory usage by sequence length for each device
    plt.subplot(2, 2, 2)
    sns.lineplot(
        data=df, 
        x="sequence_length", 
        y="memory_mb", 
        hue="device",
        style="model_size",
        markers=True
    )
    plt.title("Memory Usage by Sequence Length")
    plt.xscale("log", base=2)
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True)
    
    # 3. Inference time by sequence length for each device
    plt.subplot(2, 2, 3)
    sns.lineplot(
        data=df, 
        x="sequence_length", 
        y="time_seconds", 
        hue="device",
        style="model_size",
        markers=True
    )
    plt.title("Inference Time by Sequence Length")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True)
    
    # 4. Bar chart comparing devices at specific sequence lengths
    pivot_df = df.pivot_table(
        index=["device", "model_size"], 
        columns="sequence_length", 
        values="tokens_per_second"
    ).reset_index()
    
    # Get a representative sequence length
    rep_seq_len = df["sequence_length"].median()
    if rep_seq_len not in pivot_df.columns:
        rep_seq_len = pivot_df.columns[2]  # Skip device and model_size columns
    
    plt.subplot(2, 2, 4)
    sns.barplot(
        data=pivot_df, 
        x="device", 
        y=rep_seq_len, 
        hue="model_size"
    )
    plt.title(f"Performance Comparison at {rep_seq_len} Tokens")
    plt.xlabel("Device")
    plt.ylabel("Tokens per Second")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    
    # Also save a summary CSV
    summary_file = output_file.replace(".png", ".csv")
    df.to_csv(summary_file, index=False)
    print(f"Summary data saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize cross-device benchmark results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing benchmark results")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for visualization")
    args = parser.parse_args()
    
    # Load benchmark results
    df = load_benchmark_results(args.input_dir)
    
    # Create visualizations
    plot_benchmarks(df, args.output_file)

if __name__ == "__main__":
    main()