#!/usr/bin/env python
# EdgeFormer - Benchmark Analysis Script
# Author: Oscar Nunez (art.by.oscar.n@gmail.com)

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EdgeFormer Benchmark Analysis')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing benchmark result files')
    parser.add_argument('--output_file', type=str, default='benchmark_summary.md',
                        help='Path to output summary markdown file')
    parser.add_argument('--output_dir', type=str, default='benchmark_visualizations',
                        help='Directory to save visualization plots')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive plots')
    parser.add_argument('--filter_outliers', action='store_true',
                        help='Filter extreme outliers from the data')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of top configurations to highlight')
    return parser.parse_args()

def load_benchmark_data(input_dir):
    """Load and parse all benchmark data files from the input directory."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Benchmark directory not found: {input_dir}")
    
    # Find all JSON files in the directory
    benchmark_files = list(Path(input_dir).glob('**/*.json'))
    
    if not benchmark_files:
        print(f"No benchmark JSON files found in {input_dir}")
        return None
    
    print(f"Found {len(benchmark_files)} benchmark files")
    
    # Load and combine all benchmark data
    all_results = []
    
    for file_path in benchmark_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Some older benchmark files might have different structure
                if isinstance(data, list):
                    all_results.extend(data)
                elif isinstance(data, dict) and 'results' in data:
                    all_results.extend(data['results'])
                elif isinstance(data, dict):
                    # Single benchmark result
                    all_results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(all_results)} benchmark results")
    
    if not all_results:
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # Filter out any invalid entries
    df = df.dropna(subset=['execution_time'])
    
    # Add derived metrics
    if 'sequence_length' in df.columns and 'execution_time' in df.columns:
        df['tokens_per_second'] = df['sequence_length'] / df['execution_time']
    
    if 'memory_usage' in df.columns:
        # Convert memory to MB if it's in bytes
        if df['memory_usage'].max() > 1e6:  # Likely in bytes
            df['memory_usage'] = df['memory_usage'] / (1024 * 1024)
    
    return df

def filter_outliers(df, columns=None, threshold=3):
    """Filter extreme outliers from the dataset."""
    if df is None or df.empty:
        return df
    
    if columns is None:
        columns = ['execution_time', 'tokens_per_second', 'memory_usage']
    
    columns = [col for col in columns if col in df.columns]
    
    result_df = df.copy()
    original_count = len(result_df)
    
    for col in columns:
        z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
        result_df = result_df[z_scores < threshold]
    
    filtered_count = original_count - len(result_df)
    if filtered_count > 0:
        print(f"Filtered {filtered_count} outliers ({filtered_count/original_count:.1%} of data)")
    
    return result_df

def create_basic_stats_summary(df):
    """Create a basic statistical summary of benchmark results."""
    if df is None or df.empty:
        return "No valid benchmark data found."
    
    summary = []
    summary.append("# EdgeFormer Benchmark Analysis")
    summary.append(f"\nAnalysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"\nTotal benchmark results analyzed: {len(df)}")
    
    # Basic statistics for key metrics
    metrics = ['execution_time', 'tokens_per_second', 'memory_usage', 'sequence_length']
    metrics = [m for m in metrics if m in df.columns]
    
    summary.append("\n## Overall Performance Metrics")
    
    for metric in metrics:
        summary.append(f"\n### {metric.replace('_', ' ').title()}")
        summary.append(f"- Minimum: {df[metric].min():.4f}")
        summary.append(f"- Maximum: {df[metric].max():.4f}")
        summary.append(f"- Mean: {df[metric].mean():.4f}")
        summary.append(f"- Median: {df[metric].median():.4f}")
        summary.append(f"- Standard Deviation: {df[metric].std():.4f}")
    
    return "\n".join(summary)

def analyze_feature_impact(df):
    """Analyze the impact of different features on performance."""
    if df is None or df.empty:
        return "No valid benchmark data found."
    
    summary = []
    summary.append("\n## Feature Impact Analysis")
    
    # Define feature columns to analyze
    feature_cols = [
        'attention_type', 'use_recurrent', 'use_budget', 'use_kv_cache',
        'use_memory', 'selection_strategy'
    ]
    
    # Filter to only include columns that exist in the dataset
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        return "\n".join(summary + ["No feature columns found in the benchmark data."])
    
    # Define metrics to analyze
    metrics = ['execution_time', 'tokens_per_second', 'memory_usage']
    metrics = [m for m in metrics if m in df.columns]
    
    for feature in feature_cols:
        if len(df[feature].unique()) < 2:
            continue  # Skip features with only one value
        
        summary.append(f"\n### Impact of {feature.replace('_', ' ').title()}")
        
        for metric in metrics:
            summary.append(f"\n#### Effect on {metric.replace('_', ' ').title()}")
            
            # Group by the feature and calculate statistics
            grouped = df.groupby(feature)[metric].agg(['mean', 'std', 'min', 'max', 'count'])
            grouped = grouped.reset_index()
            
            # Format as table
            table_rows = ["| Value | Mean | Std Dev | Min | Max | Count |", 
                          "|-------|------|---------|-----|-----|-------|"]
            
            for _, row in grouped.iterrows():
                table_rows.append(f"| {row[feature]} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} | {row['count']} |")
            
            summary.append("\n".join(table_rows))
    
    return "\n".join(summary)

def analyze_feature_combinations(df):
    """Analyze the performance of different feature combinations."""
    if df is None or df.empty:
        return "No valid benchmark data found."
    
    summary = []
    summary.append("\n## Feature Combination Analysis")
    
    # Define feature columns for combinations
    feature_cols = [
        'attention_type', 'use_recurrent', 'use_budget', 'use_kv_cache',
        'use_memory'
    ]
    
    # Filter to only include columns that exist in the dataset
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        return "\n".join(summary + ["No feature columns found in the benchmark data."])
    
    # Create a combined feature column
    df['feature_combination'] = df.apply(
        lambda row: " + ".join([f"{col}={row[col]}" for col in feature_cols]), 
        axis=1
    )
    
    # Define metrics to analyze
    metrics = ['execution_time', 'tokens_per_second']
    if 'memory_usage' in df.columns:
        metrics.append('memory_usage')
    
    # Find the best combinations for each metric
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        summary.append(f"\n### Best Feature Combinations for {metric.replace('_', ' ').title()}")
        
        # Define whether higher or lower is better for this metric
        ascending = True if metric == 'execution_time' or metric == 'memory_usage' else False
        
        # Group by combination and get average performance
        grouped = df.groupby('feature_combination')[metric].mean().reset_index()
        
        # Sort by the metric
        grouped = grouped.sort_values(by=metric, ascending=ascending)
        
        # Format as table
        table_rows = ["| Rank | Feature Combination | Value |", 
                      "|------|---------------------|-------|"]
        
        for i, (_, row) in enumerate(grouped.head(10).iterrows()):
            table_rows.append(f"| {i+1} | {row['feature_combination']} | {row[metric]:.4f} |")
        
        summary.append("\n".join(table_rows))
    
    return "\n".join(summary)

def find_optimal_configurations(df, top_n=5):
    """Find the most optimal configurations based on multiple metrics."""
    if df is None or df.empty:
        return "No valid benchmark data found."
    
    summary = []
    summary.append("\n## Optimal Configurations")
    
    # Identify key metrics
    metrics = []
    if 'tokens_per_second' in df.columns:
        metrics.append(('tokens_per_second', False))  # False means lower is not better
    if 'execution_time' in df.columns:
        metrics.append(('execution_time', True))  # True means lower is better
    if 'memory_usage' in df.columns:
        metrics.append(('memory_usage', True))
    
    if not metrics:
        return "\n".join(summary + ["No performance metrics found in the benchmark data."])
    
    # Feature columns for identifying unique configurations
    feature_cols = [
        'attention_type', 'model_type', 'sequence_length',
        'use_recurrent', 'use_budget', 'use_kv_cache',
        'use_memory', 'selection_strategy'
    ]
    
    # Filter to only include columns that exist in the dataset
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Create a normalized score for each metric
    df_scored = df.copy()
    
    for metric, lower_is_better in metrics:
        # Calculate z-score (standardization)
        z_score = (df_scored[metric] - df_scored[metric].mean()) / df_scored[metric].std()
        
        # Invert z-score if lower is better
        if lower_is_better:
            z_score = -z_score
        
        # Add as new column
        df_scored[f'{metric}_score'] = z_score
    
    # Calculate overall score (average of normalized scores)
    score_columns = [f'{metric}_score' for metric, _ in metrics]
    df_scored['overall_score'] = df_scored[score_columns].mean(axis=1)
    
    # Group by configuration and average the scores
    grouped = df_scored.groupby(feature_cols, dropna=False)['overall_score'].mean().reset_index()
    
    # Sort by overall score
    grouped = grouped.sort_values(by='overall_score', ascending=False)
    
    # Format the top configurations
    summary.append("\n### Top Overall Configurations")
    table_rows = ["| Rank | Configuration | Overall Score |", 
                  "|------|--------------|---------------|"]
    
    for i, (_, row) in enumerate(grouped.head(top_n).iterrows()):
        config_str = ", ".join([f"{col}={row[col]}" for col in feature_cols])
        table_rows.append(f"| {i+1} | {config_str} | {row['overall_score']:.4f} |")
    
    summary.append("\n".join(table_rows))
    
    # Also provide specific recommendations based on use case
    summary.append("\n### Recommendations by Use Case")
    
    # For maximum speed
    if 'tokens_per_second' in df.columns:
        speed_df = df.sort_values(by='tokens_per_second', ascending=False)
        fastest_row = speed_df.iloc[0]
        
        summary.append("\n#### For Maximum Speed")
        speed_config = ", ".join([f"{col}={fastest_row[col]}" for col in feature_cols if col in fastest_row])
        summary.append(f"- Configuration: {speed_config}")
        summary.append(f"- Performance: {fastest_row['tokens_per_second']:.4f} tokens/second")
    
    # For memory efficiency
    if 'memory_usage' in df.columns:
        memory_df = df.sort_values(by='memory_usage', ascending=True)
        memory_row = memory_df.iloc[0]
        
        summary.append("\n#### For Memory Efficiency")
        memory_config = ", ".join([f"{col}={memory_row[col]}" for col in feature_cols if col in memory_row])
        summary.append(f"- Configuration: {memory_config}")
        summary.append(f"- Memory Usage: {memory_row['memory_usage']:.4f} MB")
    
    # For balanced performance
    balanced_row = grouped.iloc[0]
    
    summary.append("\n#### For Balanced Performance")
    balanced_config = ", ".join([f"{col}={balanced_row[col]}" for col in feature_cols])
    summary.append(f"- Configuration: {balanced_config}")
    summary.append(f"- Overall Score: {balanced_row['overall_score']:.4f}")
    
    return "\n".join(summary)

def create_visualizations(df, output_dir, interactive=False):
    """Create visualization plots for benchmark results."""
    if df is None or df.empty:
        print("No valid data for visualizations")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # Custom color palette
    colors = sns.color_palette("viridis", 8)
    
    # Create a timestamp for the output files
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 1. Performance by Attention Type
    if 'attention_type' in df.columns and 'tokens_per_second' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='attention_type', y='tokens_per_second', data=df, palette=colors)
        plt.title('Performance by Attention Type')
        plt.ylabel('Tokens per Second')
        plt.xlabel('Attention Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_by_attention_type_{timestamp}.png'), dpi=300)
        if interactive:
            plt.show()
        plt.close()
    
    # 2. Memory Usage by Feature Combination
    feature_cols = ['use_recurrent', 'use_budget', 'use_kv_cache', 'use_memory']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if feature_cols and 'memory_usage' in df.columns:
        # Create a feature combination label
        df['feature_set'] = df.apply(
            lambda row: "+".join([col for col in feature_cols if row[col] == True]), 
            axis=1
        )
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='feature_set', y='memory_usage', data=df, palette=colors)
        plt.title('Memory Usage by Feature Combination')
        plt.ylabel('Memory Usage (MB)')
        plt.xlabel('Enabled Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'memory_by_features_{timestamp}.png'), dpi=300)
        if interactive:
            plt.show()
        plt.close()
    
    # 3. Sequence Length vs Performance
    if 'sequence_length' in df.columns and 'tokens_per_second' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Use different markers/colors for different attention types if available
        if 'attention_type' in df.columns:
            for name, group in df.groupby('attention_type'):
                plt.scatter(group['sequence_length'], group['tokens_per_second'], 
                           label=name, alpha=0.7)
            plt.legend(title='Attention Type')
        else:
            plt.scatter(df['sequence_length'], df['tokens_per_second'], alpha=0.7)
        
        plt.title('Performance vs Sequence Length')
        plt.xlabel('Sequence Length')
        plt.ylabel('Tokens per Second')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_vs_sequence_length_{timestamp}.png'), dpi=300)
        if interactive:
            plt.show()
        plt.close()
    
    # 4. Heatmap of Feature Correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
        
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{timestamp}.png'), dpi=300)
        if interactive:
            plt.show()
        plt.close()
    
    # 5. Recurrent Iterations Analysis (if available)
    if 'recurrent_iterations' in df.columns and 'tokens_per_second' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='recurrent_iterations', y='tokens_per_second', 
                       hue='convergence_threshold' if 'convergence_threshold' in df.columns else None,
                       data=df, palette='viridis')
        
        plt.title('Performance vs Recurrent Iterations')
        plt.xlabel('Recurrent Iterations')
        plt.ylabel('Tokens per Second')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_vs_iterations_{timestamp}.png'), dpi=300)
        if interactive:
            plt.show()
        plt.close()
    
    # 6. Combined Feature Impact on Performance
    if feature_cols and 'tokens_per_second' in df.columns:
        # Prepare data for radar chart
        feature_impact = {}
        
        for feature in feature_cols:
            if len(df[feature].unique()) >= 2:  # Need at least two values to compare
                # Compare performance with and without this feature
                with_feature = df[df[feature] == True]['tokens_per_second'].mean()
                without_feature = df[df[feature] == False]['tokens_per_second'].mean()
                
                # Calculate relative impact
                if without_feature > 0:
                    impact = (with_feature / without_feature - 1) * 100  # Percentage change
                    feature_impact[feature.replace('use_', '')] = impact
        
        if feature_impact:
            # Create radar chart
            categories = list(feature_impact.keys())
            values = list(feature_impact.values())
            
            # Number of variables
            N = len(categories)
            
            # Create angles for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add values for closing the loop
            values += values[:1]
            
            # Create figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Plot data
            ax.plot(angles, values, 'o-', linewidth=2)
            
            # Fill area
            ax.fill(angles, values, alpha=0.25)
            
            # Set category labels
            plt.xticks(angles[:-1], categories)
            
            # Configure radial axis
            min_val = min(0, min(values))
            max_val = max(0, max(values))
            plt.ylim(min_val - 10, max_val + 10)
            
            # Add a title
            plt.title('Feature Impact on Performance (% change)', size=15, y=1.1)
            
            # Save the chart
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_impact_radar_{timestamp}.png'), dpi=300)
            if interactive:
                plt.show()
            plt.close()
    
    # 7. Memory Feature Impact
    if 'memory_usage' in df.columns and 'use_memory' in df.columns:
        memory_data = []
        
        # Group by memory feature and selection strategy if available
        group_cols = ['use_memory']
        if 'selection_strategy' in df.columns:
            group_cols.append('selection_strategy')
        
        grouped = df.groupby(group_cols)['memory_usage'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        
        if 'selection_strategy' in group_cols:
            ax = sns.barplot(x='use_memory', y='memory_usage', hue='selection_strategy', data=grouped, palette='viridis')
        else:
            ax = sns.barplot(x='use_memory', y='memory_usage', data=grouped, palette='viridis')
        
        plt.title('Memory Usage by Memory Feature')
        plt.xlabel('Memory Feature Enabled')
        plt.ylabel('Memory Usage (MB)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'memory_feature_impact_{timestamp}.png'), dpi=300)
        if interactive:
            plt.show()
        plt.close()
    
    # 8. Overall Performance Distribution
    if 'tokens_per_second' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['tokens_per_second'], kde=True, bins=20, color=colors[2])
        plt.title('Distribution of Performance')
        plt.xlabel('Tokens per Second')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_distribution_{timestamp}.png'), dpi=300)
        if interactive:
            plt.show()
        plt.close()
    
    # Create an overview plot combining key visualizations
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: Performance by attention type
    if 'attention_type' in df.columns and 'tokens_per_second' in df.columns:
        sns.boxplot(x='attention_type', y='tokens_per_second', data=df, ax=axs[0, 0], palette=colors[:3])
        axs[0, 0].set_title('Performance by Attention Type')
        axs[0, 0].set_ylabel('Tokens per Second')
    
    # Top right: Performance vs Sequence Length
    if 'sequence_length' in df.columns and 'tokens_per_second' in df.columns:
        if 'attention_type' in df.columns:
            for name, group in df.groupby('attention_type'):
                axs[0, 1].scatter(group['sequence_length'], group['tokens_per_second'], 
                                  label=name, alpha=0.7)
            axs[0, 1].legend(title='Attention Type')
        else:
            axs[0, 1].scatter(df['sequence_length'], df['tokens_per_second'], alpha=0.7)
        
        axs[0, 1].set_title('Performance vs Sequence Length')
        axs[0, 1].set_xlabel('Sequence Length')
        axs[0, 1].set_ylabel('Tokens per Second')
    
    # Bottom left: Memory usage by feature
    if feature_cols and 'memory_usage' in df.columns:
        if 'feature_set' not in df.columns:
            df['feature_set'] = df.apply(
                lambda row: "+".join([col for col in feature_cols if col in row and row[col] == True]), 
                axis=1
            )
        
        top_configs = df.groupby('feature_set')['memory_usage'].mean().nlargest(5).index
        subset = df[df['feature_set'].isin(top_configs)]
        
        sns.boxplot(x='feature_set', y='memory_usage', data=subset, ax=axs[1, 0], palette=colors[3:])
        axs[1, 0].set_title('Memory Usage by Top Features')
        axs[1, 0].set_ylabel('Memory Usage (MB)')
        axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Bottom right: Performance distribution
    if 'tokens_per_second' in df.columns:
        sns.histplot(df['tokens_per_second'], kde=True, bins=20, ax=axs[1, 1], color=colors[2])
        axs[1, 1].set_title('Performance Distribution')
        axs[1, 1].set_xlabel('Tokens per Second')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'overview_dashboard_{timestamp}.png'), dpi=300)
    if interactive:
        plt.show()
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    return timestamp

def main():
    """Main function to analyze benchmark results."""
    args = parse_arguments()
    
    # Load benchmark data
    df = load_benchmark_data(args.input_dir)
    
    if df is None or df.empty:
        print("No valid benchmark data found.")
        return
    
    # Filter outliers if requested
    if args.filter_outliers:
        df = filter_outliers(df)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    timestamp = create_visualizations(df, args.output_dir, args.interactive)
    
    # Generate summary
    summary_parts = [
        create_basic_stats_summary(df),
        analyze_feature_impact(df),
        analyze_feature_combinations(df),
        find_optimal_configurations(df, args.top_n)
    ]
    
    # Add visualization references to the summary
    vis_summary = "\n\n## Visualizations\n\n"
    vis_summary += f"Visualizations are available in the `{args.output_dir}` directory.\n\n"
    
    if timestamp:
        vis_summary += "Key visualizations:\n\n"
        vis_summary += f"- [Performance by Attention Type]({args.output_dir}/performance_by_attention_type_{timestamp}.png)\n"
        vis_summary += f"- [Performance vs Sequence Length]({args.output_dir}/performance_vs_sequence_length_{timestamp}.png)\n"
        vis_summary += f"- [Feature Impact Dashboard]({args.output_dir}/overview_dashboard_{timestamp}.png)\n"
    
    summary_parts.append(vis_summary)
    
    # Write the summary to file
    full_summary = "\n".join(summary_parts)
    
    with open(args.output_file, 'w') as f:
        f.write(full_summary)
    
    print(f"Summary written to {args.output_file}")
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()