#!/usr/bin/env python
"""Power consumption profiling for EdgeFormer inference."""
import os
import time
import json
import argparse
import logging
import platform
import threading
import datetime
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('power_profiler')

class PowerMonitor:
    """Monitor power consumption during model inference."""
    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval
        self.running = False
        self.samples = []
        self.start_time = None
        
    def start(self):
        """Start power monitoring."""
        self.samples = []
        self.running = True
        self.start_time = time.time()
        
        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Power monitoring started")
        
    def stop(self):
        """Stop power monitoring."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2*self.sample_interval)
        
        logger.info(f"Power monitoring stopped. Collected {len(self.samples)} samples")
        return self.get_stats()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Sample current power usage
                sample = self._get_power_sample()
                self.samples.append(sample)
            except Exception as e:
                logger.error(f"Error sampling power: {e}")
            
            # Sleep until next sample
            time.sleep(self.sample_interval)
    
    def _get_power_sample(self):
        """Get a single power usage sample."""
        sample = {
            'timestamp': time.time() - self.start_time,
            'cpu_percent': 0,
            'memory_percent': 0,
        }
        
        # Platform-specific power sampling
        system = platform.system().lower()
        
        if system == 'linux':
            # TODO: Implement Linux power sampling
            # For now, use psutil as fallback
            try:
                import psutil
                sample['cpu_percent'] = psutil.cpu_percent(interval=None)
                sample['memory_percent'] = psutil.virtual_memory().percent
            except ImportError:
                pass
                
        elif system == 'darwin':  # macOS
            # TODO: Implement macOS power sampling
            # For now, use psutil as fallback
            try:
                import psutil
                sample['cpu_percent'] = psutil.cpu_percent(interval=None)
                sample['memory_percent'] = psutil.virtual_memory().percent
            except ImportError:
                pass
                
        elif system == 'windows':
            # Use psutil on Windows
            try:
                import psutil
                sample['cpu_percent'] = psutil.cpu_percent(interval=None)
                sample['memory_percent'] = psutil.virtual_memory().percent
            except ImportError:
                pass
        
        return sample
    
    def get_stats(self):
        """Get statistics from collected samples."""
        if not self.samples:
            return {'error': 'No samples collected'}
        
        # Extract data series
        timestamps = [s['timestamp'] for s in self.samples]
        cpu_percent = [s['cpu_percent'] for s in self.samples]
        memory_percent = [s['memory_percent'] for s in self.samples]
        
        return {
            'duration': timestamps[-1],
            'samples': len(self.samples),
            'cpu_percent': {
                'mean': np.mean(cpu_percent),
                'max': np.max(cpu_percent),
                'min': np.min(cpu_percent),
                'std': np.std(cpu_percent),
            },
            'memory_percent': {
                'mean': np.mean(memory_percent),
                'max': np.max(memory_percent),
                'min': np.min(memory_percent),
                'std': np.std(memory_percent),
            },
            'raw': {
                'timestamps': timestamps,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
            }
        }
    
    def plot(self, output_file=None):
        """Plot power consumption data."""
        if not self.samples:
            logger.warning("No samples to plot")
            return
        
        stats = self.get_stats()
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot CPU usage
        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU Usage (%)', color=color)
        ax1.plot(stats['raw']['timestamps'], stats['raw']['cpu_percent'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for memory usage
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Memory Usage (%)', color=color)
        ax2.plot(stats['raw']['timestamps'], stats['raw']['memory_percent'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('EdgeFormer Resource Usage Over Time')
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()

def run_inference_benchmark(model_size='small', sequence_length=1024, batch_size=1, duration=30):
    """Run an inference benchmark and monitor power."""
    try:
        import torch
        from examples.htps_associative_memory_demo import initialize_components, add_default_memories
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        return None
    
    # Set up monitoring
    monitor = PowerMonitor(sample_interval=0.1)
    
    # Run mock inference load to create CPU pressure
    class Args:
        def __init__(self):
            self.capacity = 20
            self.strategy = 'htps'
    
    args = Args()
    
    logger.info(f"Starting inference benchmark (size={model_size}, seq_len={sequence_length}, batch={batch_size})")
    
    try:
        # Initialize components
        memory, retriever = initialize_components(args)
        add_default_memories(memory)
        
        # Start monitoring
        monitor.start()
        
        # Create mock inference load
        start_time = time.time()
        end_time = start_time + duration
        
        count = 0
        while time.time() < end_time:
            # Simulate inference with random data
            batch = torch.randn(batch_size, sequence_length, 768)
            
            # Process batch to create CPU load
            for _ in range(10):  # Simulate multiple processing steps
                batch = torch.matmul(batch, torch.randn(768, 768))
                batch = torch.nn.functional.relu(batch)
            
            count += 1
        
        # Calculate throughput
        elapsed = time.time() - start_time
        throughput = count / elapsed
        
        logger.info(f"Completed {count} batches in {elapsed:.2f}s ({throughput:.2f} batches/s)")
        
    finally:
        # Stop monitoring
        stats = monitor.stop()
        
        # Add benchmark info to stats
        stats['benchmark'] = {
            'model_size': model_size,
            'sequence_length': sequence_length,
            'batch_size': batch_size,
            'batches_processed': count,
            'elapsed_time': elapsed,
            'throughput': throughput
        }
        
        return stats

def main():
    """Main entry point for power profiling."""
    parser = argparse.ArgumentParser(description="EdgeFormer power profiling")
    parser.add_argument('--model-size', type=str, default='small', choices=['tiny', 'small', 'medium', 'large'],
                        help='Model size to benchmark')
    parser.add_argument('--sequence-length', type=int, default=1024,
                        help='Sequence length for inference')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--duration', type=int, default=30,
                        help='Benchmark duration in seconds')
    parser.add_argument('--output-dir', type=str, default='benchmark_results/power',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the benchmark
    stats = run_inference_benchmark(
        model_size=args.model_size,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        duration=args.duration
    )
    
    if stats:
        # Generate timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"power_profile_{args.model_size}_{args.sequence_length}_{timestamp}"
        
        # Save stats to JSON
        json_file = os.path.join(args.output_dir, f"{base_filename}.json")
        with open(json_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create monitor to plot the data
        monitor = PowerMonitor()
        monitor.samples = [{'timestamp': t, 'cpu_percent': c, 'memory_percent': m} 
                          for t, c, m in zip(stats['raw']['timestamps'], 
                                            stats['raw']['cpu_percent'],
                                            stats['raw']['memory_percent'])]
        
        # Generate plot
        plot_file = os.path.join(args.output_dir, f"{base_filename}.png")
        monitor.plot(output_file=plot_file)
        
        print(f"\nPower Profile Summary for {args.model_size} model:")
        print(f"Sequence Length: {args.sequence_length}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Average CPU Usage: {stats['cpu_percent']['mean']:.1f}%")
        print(f"Peak CPU Usage: {stats['cpu_percent']['max']:.1f}%")
        print(f"Average Memory Usage: {stats['memory_percent']['mean']:.1f}%")
        print(f"Throughput: {stats['benchmark']['throughput']:.2f} batches/s")
        print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()