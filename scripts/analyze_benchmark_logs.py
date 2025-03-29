# scripts/analyze_benchmark_logs.py
import os
import re
import json
import glob
import argparse
from datetime import datetime

def parse_logs(log_file):
    """Parse benchmark log files to extract performance metrics"""
    metrics = {
        'file': os.path.basename(log_file),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'memory_usage': [],
        'processing_times': [],
        'attention_shapes': [],
        'features': {
            'recurrent': False,
            'budget': False,
            'kv_cache': False
        }
    }
    
    # Extract features from filename
    if 'recurrent' in log_file:
        metrics['features']['recurrent'] = True
    if 'budget' in log_file:
        metrics['features']['budget'] = True
    if 'kvcache' in log_file:
        metrics['features']['kv_cache'] = True
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract memory usage
        kv_cache_sizes = re.findall(r'KV Cache entry size: (\d+\.\d+) MB', content)
        for size in kv_cache_sizes:
            metrics['memory_usage'].append(float(size))
        
        # Extract attention shapes
        attention_shapes = re.findall(r'attention_scores shape: torch.Size\(\[(.*?)\]\)', content)
        metrics['attention_shapes'] = attention_shapes
        
        # Simple processing time estimation (from log timestamps)
        timestamps = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', content)
        if len(timestamps) >= 2:
            start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S,%f")
            end = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S,%f")
            duration = (end - start).total_seconds()
            metrics['total_duration'] = duration
    
    return metrics

def analyze_benchmarks(input_dir, output_file):
    """Analyze all benchmark logs in the input directory"""
    log_files = glob.glob(os.path.join(input_dir, "*.log"))
    
    if not log_files:
        print(f"No log files found in {input_dir}")
        return
    
    results = []
    for log_file in log_files:
        print(f"Analyzing {log_file}...")
        metrics = parse_logs(log_file)
        results.append(metrics)
    
    # Save results as JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_file}")
    
    # Generate a simple summary
    print("\nSummary:")
    for metrics in results:
        feature_str = ", ".join([k for k, v in metrics['features'].items() if v])
        mem_usage = sum(metrics['memory_usage']) / len(metrics['memory_usage']) if metrics['memory_usage'] else 0
        
        # Handle the case where total_duration might be a string
        if isinstance(metrics.get('total_duration'), (int, float)):
            duration_str = f"{metrics.get('total_duration'):.2f}s"
        else:
            duration_str = str(metrics.get('total_duration', 'N/A'))
            
        print(f"- {metrics['file']}: Features: {feature_str}, Avg Memory: {mem_usage:.2f} MB, Duration: {duration_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark logs")
    parser.add_argument("--input_dir", type=str, default="benchmark_results", help="Input directory with log files")
    parser.add_argument("--output_file", type=str, default="benchmark_results/analysis.json", help="Output JSON file")
    
    args = parser.parse_args()
    analyze_benchmarks(args.input_dir, args.output_file)