# src/utils/memory_tracking.py
import torch
import gc
import psutil
import os
import time
import sys
import tracemalloc

def measure_memory_usage():
    """Measure GPU and CPU memory usage."""
    # Clear cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Wait for memory to settle
    time.sleep(0.5)
    
    # Get CPU memory
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Get GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        return {
            "cpu_memory_mb": cpu_memory,
            "gpu_memory_mb": gpu_memory,
            "max_gpu_memory_mb": max_gpu_memory
        }
    else:
        return {
            "cpu_memory_mb": cpu_memory,
            "gpu_memory_mb": 0,
            "max_gpu_memory_mb": 0
        }

class MemoryTracker:
    """Class for detailed memory tracking of model components."""
    
    def __init__(self, detailed=False):
        self.detailed = detailed
        self.snapshots = {}
        
    def start_tracking(self):
        """Start memory tracking."""
        if self.detailed:
            tracemalloc.start()
    
    def take_snapshot(self, name):
        """Take a memory snapshot with a given name."""
        # Basic memory tracking
        self.snapshots[name] = measure_memory_usage()
        
        # Detailed tracking if enabled
        if self.detailed:
            self.snapshots[f"{name}_detailed"] = tracemalloc.take_snapshot()
    
    def get_memory_diff(self, start_name, end_name):
        """Get memory difference between two snapshots."""
        if start_name not in self.snapshots or end_name not in self.snapshots:
            return None
        
        start = self.snapshots[start_name]
        end = self.snapshots[end_name]
        
        diff = {
            "cpu_diff_mb": end["cpu_memory_mb"] - start["cpu_memory_mb"],
            "gpu_diff_mb": end["gpu_memory_mb"] - start["gpu_memory_mb"]
        }
        
        return diff
    
    def print_stats(self, start_name, end_name):
        """Print memory statistics."""
        diff = self.get_memory_diff(start_name, end_name)
        if diff is None:
            print(f"Cannot compare {start_name} and {end_name}, snapshots not found.")
            return
        
        print(f"\nMemory diff from {start_name} to {end_name}:")
        print(f"  CPU: {diff['cpu_diff_mb']:.2f} MB")
        print(f"  GPU: {diff['gpu_diff_mb']:.2f} MB")
        
        # Detailed comparison if enabled
        if self.detailed and f"{start_name}_detailed" in self.snapshots and f"{end_name}_detailed" in self.snapshots:
            start_snap = self.snapshots[f"{start_name}_detailed"]
            end_snap = self.snapshots[f"{end_name}_detailed"]
            
            stats = end_snap.compare_to(start_snap, 'lineno')
            print("\nTop 10 memory differences by line:")
            for stat in stats[:10]:
                print(f"  {stat.size_diff/1024/1024:.1f} MB: {stat.traceback.format()[0]}")

    def stop_tracking(self):
        """Stop memory tracking."""
        if self.detailed:
            tracemalloc.stop()