# scripts/visualize_optimizations.py
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize optimization comparisons")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing benchmark result JSON files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save visualization output')
    return parser.parse_args()

def load_benchmark_results(input_dir):
    """Load benchmark results from JSON files"""
    results = {
        'with_phase2': {},
        'without_phase2': {}
    }
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    for file_path in json_files:
        # Skip hardware profile files
        if 'hardware_profile' in file_path:
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Determine if this is with or without Phase 2 optimizations
            if isinstance(data, list):
                # Should have a phase2_optimizations field in each result
                phase2 = data[0].get('phase2_optimizations', False)
                
                # Extract model size from filename
                filename = os.path.basename(file_path)
                model_size = filename.split('_')[0]
                
                # Organize data by sequence length
                sequence_lengths = {}
                for result in data:
                    sequence_lengths[result['sequence_length']] = {
                        'tokens_per_second': result['tokens_per_second'],
                        'memory_usage': result['memory_usage'],
                        'inference_time': result['inference_time']
                    }
                
                if phase2:
                    results['with_phase2'][model_size] = sequence_lengths
                else:
                    results['without_phase2'][model_size] = sequence_lengths
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def visualize_optimization_comparison(results, output_file):
    """Create visualizations comparing optimization results"""
    # Check if we have data to visualize
    if not results['with_phase2'] and not results['without_phase2']:
        print("No valid benchmark data found to visualize")
        return
        
    # If we only have one type of result, we can't compare
    if not results['with_phase2'] or not results['without_phase2']:
        print("Need both optimized and non-optimized results for comparison")
        return
        
    # Get model sizes that exist in both result sets
    common_models = set(results['with_phase2'].keys()) & set(results['without_phase2'].keys())
    
    if not common_models:
        print("No common model sizes found for comparison")
        return
        
    # Create comparison plots for each model size
    for model_size in common_models:
        # Get sequence lengths that exist in both result sets
        optimized = results['with_phase2'][model_size]
        baseline = results['without_phase2'][model_size]
        
        common_seq_lengths = sorted(set(optimized.keys()) & set(baseline.keys()))
        
        if not common_seq_lengths:
            print(f"No common sequence lengths for model {model_size}")
            continue
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Prepare data for plotting
        seq_lengths = common_seq_lengths
        optimized_tps = [optimized[seq]['tokens_per_second'] for seq in seq_lengths]
        baseline_tps = [baseline[seq]['tokens_per_second'] for seq in seq_lengths]
        
        optimized_mem = [optimized[seq]['memory_usage'] for seq in seq_lengths]
        baseline_mem = [baseline[seq]['memory_usage'] for seq in seq_lengths]
        
        speedup = [opt/base for opt, base in zip(optimized_tps, baseline_tps)]
        mem_reduction = [100 * (1 - opt/base) for opt, base in zip(optimized_mem, baseline_mem)]
        
        # Plot performance comparison
        width = 0.35
        x = np.arange(len(seq_lengths))
        
        ax1.bar(x - width/2, baseline_tps, width, label='Baseline')
        ax1.bar(x + width/2, optimized_tps, width, label='Phase 2')
        
        # Add speedup annotations
        for i, speed in enumerate(speedup):
            ax1.annotate(f"{speed:.2f}x", 
                         xy=(x[i], max(optimized_tps[i], baseline_tps[i]) + 50),
                         ha='center')
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Tokens per Second')
        ax1.set_title(f'Performance Comparison - {model_size.capitalize()}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(seq_lengths)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot memory usage comparison
        ax2.bar(x - width/2, baseline_mem, width, label='Baseline')
        ax2.bar(x + width/2, optimized_mem, width, label='Phase 2')
        
        # Add memory reduction annotations
        for i, reduction in enumerate(mem_reduction):
            ax2.annotate(f"{reduction:.1f}%", 
                         xy=(x[i], max(optimized_mem[i], baseline_mem[i]) + 50),
                         ha='center')
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title(f'Memory Usage Comparison - {model_size.capitalize()}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(seq_lengths)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Phase 2 Optimization Impact - {model_size.capitalize()} Model', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        output_path = output_file.replace('.png', f'_{model_size}.png')
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        
        plt.close(fig)
    
    # Create a summary visualization with improvement metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Prepare summary data
    model_sizes = sorted(common_models)
    avg_speedups = []
    avg_mem_reductions = []
    
    for model in model_sizes:
        optimized = results['with_phase2'][model]
        baseline = results['without_phase2'][model]
        
        common_seqs = set(optimized.keys()) & set(baseline.keys())
        if not common_seqs:
            continue
            
        # Calculate average speedup for this model
        model_speedups = [optimized[seq]['tokens_per_second'] / baseline[seq]['tokens_per_second'] 
                         for seq in common_seqs]
        avg_speedups.append(sum(model_speedups) / len(model_speedups))
        
        # Calculate average memory reduction for this model
        model_mem_reductions = [100 * (1 - optimized[seq]['memory_usage'] / baseline[seq]['memory_usage']) 
                               for seq in common_seqs]
        avg_mem_reductions.append(sum(model_mem_reductions) / len(model_mem_reductions))
    
    # Plot average speedup by model size
    ax1.bar(model_sizes, avg_speedups, color='green')
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('Average Speedup Factor')
    ax1.set_title('Performance Improvement from Phase 2 Optimizations')
    for i, speedup in enumerate(avg_speedups):
        ax1.annotate(f"{speedup:.2f}x", 
                     xy=(i, speedup + 0.05),
                     ha='center')
    ax1.grid(True, alpha=0.3)
    
    # Plot average memory reduction by model size
    ax2.bar(model_sizes, avg_mem_reductions, color='blue')
    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('Average Memory Reduction (%)')
    ax2.set_title('Memory Efficiency from Phase 2 Optimizations')
    for i, reduction in enumerate(avg_mem_reductions):
        ax2.annotate(f"{reduction:.1f}%", 
                     xy=(i, reduction + 1),
                     ha='center')
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Phase 2 Optimization Summary', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save summary figure
    summary_path = output_file.replace('.png', '_summary.png')
    plt.savefig(summary_path)
    print(f"Saved summary visualization to {summary_path}")

def main():
    args = parse_args()
    results = load_benchmark_results(args.input_dir)
    visualize_optimization_comparison(results, args.output_file)

if __name__ == "__main__":
    main()