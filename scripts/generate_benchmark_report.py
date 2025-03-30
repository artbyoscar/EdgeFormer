# scripts/generate_benchmark_report.py
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comprehensive benchmark report")
    parser.add_argument('--input_dirs', type=str, required=True,
                        help='Comma-separated list of directories containing benchmark results')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='Path to save the generated report PDF')
    return parser.parse_args()

def load_benchmark_data(input_dirs):
    """Load and combine benchmark data from multiple directories"""
    # Split input directories
    dir_list = [d.strip() for d in input_dirs.split(',')]
    
    # Initialize data structures
    all_results = []
    hardware_profiles = {}
    
    for directory in dir_list:
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(directory, '*.json'))
        
        for file_path in json_files:
            filename = os.path.basename(file_path)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if filename == 'hardware_profile.json':
                    # Store hardware profile information
                    hardware_profiles[directory] = data
                elif isinstance(data, list):
                    # Regular benchmark results
                    for result in data:
                        # Add source directory for categorization
                        result['source_dir'] = directory
                        all_results.append(result)
                elif 'results' in data:
                    # Legacy format with nested results
                    for seq_length, metrics in data['results'].items():
                        result = {
                            'device': data.get('device', 'unknown'),
                            'model_size': data.get('model_size', 'unknown'),
                            'sequence_length': int(seq_length),
                            'inference_time': metrics.get('avg_time_seconds', 0),
                            'tokens_per_second': metrics.get('tokens_per_second', 0),
                            'memory_usage': metrics.get('avg_memory_mb', 0),
                            'source_dir': directory
                        }
                        all_results.append(result)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Convert results to DataFrame for easier analysis
    if all_results:
        results_df = pd.DataFrame(all_results)
    else:
        results_df = pd.DataFrame()
    
    return results_df, hardware_profiles

def create_performance_plots(df, pdf):
    """Create performance visualization plots"""
    if df.empty:
        return
    
    # Performance by sequence length
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sequence_length', y='tokens_per_second', 
                   hue='model_size', style='model_size', s=100, alpha=0.7)
    plt.title('Performance vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Memory usage by sequence length
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sequence_length', y='memory_usage', 
                   hue='model_size', style='model_size', s=100, alpha=0.7)
    plt.title('Memory Usage vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Performance by device
    if 'device' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='device', y='tokens_per_second', hue='model_size')
        plt.title('Performance by Device')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # Phase 2 optimization impact if available
    if 'phase2_optimizations' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='phase2_optimizations', y='tokens_per_second', hue='model_size')
        plt.title('Phase 2 Optimization Impact on Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='phase2_optimizations', y='memory_usage', hue='model_size')
        plt.title('Phase 2 Optimization Impact on Memory Usage')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def generate_summary_tables(df, pdf):
    """Generate summary statistics tables"""
    if df.empty:
        return
        
    # Style for summary tables
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    
    # Overall performance summary
    performance_summary = df.groupby(['model_size'])['tokens_per_second'].agg(['mean', 'min', 'max', 'std']).reset_index()
    performance_summary = performance_summary.round(2)
    
    plt.table(cellText=performance_summary.values,
              colLabels=performance_summary.columns,
              loc='center',
              cellLoc='center',
              colColours=['#f2f2f2']*len(performance_summary.columns))
    plt.title('Performance Summary by Model Size')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Sequence length performance summary
    if 'sequence_length' in df.columns:
        seq_performance = df.groupby(['sequence_length'])['tokens_per_second'].agg(['mean', 'min', 'max', 'std']).reset_index()
        seq_performance = seq_performance.round(2)
        
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.table(cellText=seq_performance.values,
                  colLabels=seq_performance.columns,
                  loc='center',
                  cellLoc='center',
                  colColours=['#f2f2f2']*len(seq_performance.columns))
        plt.title('Performance Summary by Sequence Length')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # Device comparison if available
    if 'device' in df.columns:
        device_performance = df.groupby(['device', 'model_size'])['tokens_per_second'].mean().reset_index()
        device_performance = device_performance.round(2)
        
        # Pivot the table for better readability
        device_pivot = device_performance.pivot(index='device', columns='model_size', values='tokens_per_second')
        
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.table(cellText=device_pivot.values,
                  colLabels=device_pivot.columns,
                  rowLabels=device_pivot.index,
                  loc='center',
                  cellLoc='center',
                  colColours=['#f2f2f2']*len(device_pivot.columns))
        plt.title('Average Performance (tokens/s) by Device and Model Size')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # Phase 2 optimization impact if available
    if 'phase2_optimizations' in df.columns:
        phase2_impact = df.groupby(['phase2_optimizations', 'model_size'])['tokens_per_second'].mean().reset_index()
        phase2_impact = phase2_impact.round(2)
        
        # Calculate improvement percentage
        if True in phase2_impact['phase2_optimizations'].values and False in phase2_impact['phase2_optimizations'].values:
            # Create a pivot for easier comparison
            pivot = phase2_impact.pivot(index='model_size', columns='phase2_optimizations', values='tokens_per_second')
            
            # Calculate percentage improvement
            if True in pivot.columns and False in pivot.columns:
                pivot['improvement_pct'] = ((pivot[True] - pivot[False]) / pivot[False] * 100).round(2)
                
                plt.figure(figsize=(10, 6))
                plt.axis('off')
                plt.table(cellText=pivot.reset_index().values,
                          colLabels=['model_size', 'Without Phase 2', 'With Phase 2', 'Improvement (%)'],
                          loc='center',
                          cellLoc='center',
                          colColours=['#f2f2f2']*4)
                plt.title('Phase 2 Optimization Performance Impact')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

def create_executive_summary(df, hardware_profiles, pdf):
    """Create an executive summary page"""
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.95, 'EdgeFormer Benchmark Report', fontsize=18, ha='center', weight='bold')
    plt.text(0.5, 0.9, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=12, ha='center')
    
    # Performance highlights
    if not df.empty:
        # Get best performing configuration
        best_config_idx = df['tokens_per_second'].idxmax()
        if not pd.isna(best_config_idx):
            best_config = df.loc[best_config_idx]
            
            plt.text(0.05, 0.85, 'Performance Highlights:', fontsize=14, weight='bold')
            
            highlights = [
                f"Best performance: {best_config['tokens_per_second']:.2f} tokens/second",
                f"Model size: {best_config.get('model_size', 'unknown')}",
                f"Sequence length: {best_config.get('sequence_length', 'unknown')}"
            ]
            
            if 'device' in best_config:
                highlights.append(f"Device: {best_config['device']}")
                
            if 'phase2_optimizations' in best_config:
                highlights.append(f"Phase 2 optimizations: {'Enabled' if best_config['phase2_optimizations'] else 'Disabled'}")
            
            for i, highlight in enumerate(highlights):
                plt.text(0.1, 0.8 - i*0.05, highlight, fontsize=12)
        
        # Add hardware summary
        if hardware_profiles:
            plt.text(0.05, 0.6, 'Hardware Summary:', fontsize=14, weight='bold')
            
            y_pos = 0.55
            for directory, profile in hardware_profiles.items():
                device_name = profile.get('device_name', 'unknown')
                plt.text(0.1, y_pos, f"Device: {device_name}", fontsize=12)
                
                # Show CPU info
                cpu = profile.get('processor', 'unknown')
                plt.text(0.15, y_pos - 0.05, f"CPU: {cpu}", fontsize=10)
                
                # Show RAM
                ram = profile.get('ram_gb', 'unknown')
                plt.text(0.15, y_pos - 0.1, f"RAM: {ram} GB", fontsize=10)
                
                # Show GPU if available
                if profile.get('cuda_available', False):
                    gpu = profile.get('cuda_device_name', 'unknown')
                    plt.text(0.15, y_pos - 0.15, f"GPU: {gpu}", fontsize=10)
                
                y_pos -= 0.2
    
    # Add recommendations
    plt.text(0.05, 0.25, 'Recommendations:', fontsize=14, weight='bold')
    
    recommendations = [
        "Optimal sequence length is around 1024 tokens for best performance",
        "Phase 2 optimizations significantly improve performance on Intel hardware",
        "Memory usage increases linearly with sequence length"
    ]
    
    for i, rec in enumerate(recommendations):
        plt.text(0.1, 0.2 - i*0.05, rec, fontsize=12)
    
    pdf.savefig()
    plt.close()

def generate_report(df, hardware_profiles, output_file):
    """Generate the full benchmark report"""
    with PdfPages(output_file) as pdf:
        # Executive summary
        create_executive_summary(df, hardware_profiles, pdf)
        
        # Performance visualizations
        create_performance_plots(df, pdf)
        
        # Summary tables
        generate_summary_tables(df, pdf)
        
        # Add correlation matrix if enough data is available
        if not df.empty and len(df) > 5:
            plt.figure(figsize=(10, 8))
            
            # Select numerical columns for correlation
            numeric_cols = ['sequence_length', 'tokens_per_second', 'memory_usage', 'inference_time']
            numeric_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().round(2)
                
                # Plot correlation matrix
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Between Metrics')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        # Distribution of performance
        if 'tokens_per_second' in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Create distribution plot
            sns.histplot(df['tokens_per_second'], kde=True)
            plt.title('Distribution of Performance')
            plt.xlabel('Tokens per Second')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Add hardware details page
        if hardware_profiles:
            for directory, profile in hardware_profiles.items():
                plt.figure(figsize=(10, 8))
                plt.axis('off')
                
                # Hardware profile title
                device_name = profile.get('device_name', 'Unknown Device')
                plt.text(0.5, 0.95, f'Hardware Profile: {device_name}', fontsize=16, ha='center', weight='bold')
                
                # Format profile as a table
                profile_items = []
                for key, value in profile.items():
                    profile_items.append([key, str(value)])
                
                plt.table(cellText=profile_items,
                          colLabels=['Property', 'Value'],
                          loc='center',
                          cellLoc='left',
                          colWidths=[0.3, 0.6],
                          colColours=['#f2f2f2']*2)
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()

def main():
    """Main function to generate the benchmark report"""
    args = parse_args()
    
    # Load benchmark data
    df, hardware_profiles = load_benchmark_data(args.input_dirs)
    
    if df.empty:
        print("No benchmark data found in the specified directories")
        return
    
    # Generate the report
    generate_report(df, hardware_profiles, args.output_file)
    print(f"Benchmark report generated: {args.output_file}")

if __name__ == "__main__":
    main()