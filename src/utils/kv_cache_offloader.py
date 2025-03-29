# kv_cache_offloader.py
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class KVCacheOffloader:
    """Manages offloading of KV cache to CPU RAM to handle long sequences."""
    
    def __init__(self, max_gpu_entries=1024, prefetch_size=128):
        """Initialize KV cache offloader.
        
        Args:
            max_gpu_entries: Maximum number of entries to keep in GPU memory
            prefetch_size: Number of entries to prefetch when loading from CPU
        """
        self.max_gpu_entries = max_gpu_entries
        self.prefetch_size = prefetch_size
        self.gpu_entries = {}  # Entries in GPU memory
        self.cpu_entries = {}  # Entries in CPU memory
        self.access_counts = {}  # Track access frequency
        
    def store(self, key, tensor):
        """Store a tensor in the cache."""
        if len(self.gpu_entries) < self.max_gpu_entries:
            # Store in GPU
            self.gpu_entries[key] = tensor
            self.access_counts[key] = 1
        else:
            # Need to offload something to CPU
            self._offload_least_used()
            # Store in GPU
            self.gpu_entries[key] = tensor
            self.access_counts[key] = 1
            
    def retrieve(self, key):
        """Retrieve a tensor from the cache."""
        if key in self.gpu_entries:
            # Update access count
            self.access_counts[key] += 1
            return self.gpu_entries[key]
        elif key in self.cpu_entries:
            # Move from CPU to GPU
            tensor = self.cpu_entries[key].to(next(iter(self.gpu_entries.values())).device)
            # Remove from CPU
            del self.cpu_entries[key]
            # Store in GPU
            self.gpu_entries[key] = tensor
            self.access_counts[key] = 1
            return tensor
        else:
            return None
            
    def _offload_least_used(self):
        """Offload least recently used entries to CPU."""
        if not self.gpu_entries:
            return
            
        # Find keys with minimum access count
        min_count = min(self.access_counts.values())
        candidates = [k for k, v in self.access_counts.items() if v == min_count]
        
        # Select the oldest entry
        key_to_offload = candidates[0]
        
        # Move to CPU
        self.cpu_entries[key_to_offload] = self.gpu_entries[key_to_offload].cpu()
        
        # Remove from GPU
        del self.gpu_entries[key_to_offload]
        del self.access_counts[key_to_offload]
        
    def clear(self):
        """Clear the cache."""
        self.gpu_entries.clear()
        self.cpu_entries.clear()
        self.access_counts.clear()
        
    def __len__(self):
        """Return the total number of entries in the cache."""
        return len(self.gpu_entries) + len(self.cpu_entries)