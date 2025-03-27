# src/utils/kv_cache.py
import torch
import logging
import time

logger = logging.getLogger('edgeformer')

class KVCacheManager:
    """
    Manages key-value cache for efficient inference with long sequences.
    Implements offloading from GPU VRAM to CPU RAM when necessary.
    """
    
    def __init__(self, max_gpu_cache_size=1024*1024*1024, device="cpu"):  # Default: 1GB
        """
        Initialize the KV cache manager.
        
        Args:
            max_gpu_cache_size: Maximum cache size in bytes to keep on GPU
            device: Default device for caching (cpu or cuda)
        """
        self.max_gpu_cache_size = max_gpu_cache_size
        self.gpu_cache = {}  # layer_idx -> tensor
        self.cpu_cache = {}  # layer_idx -> tensor
        self.total_gpu_size = 0
        self.total_cpu_size = 0
        self.device = device
        self.access_times = {}  # layer_idx -> last access time (for LRU)
        
    def add(self, layer_idx, k_cache, v_cache):
        """
        Add KV cache for a layer.
        
        Args:
            layer_idx: Layer index
            k_cache: Key cache tensor
            v_cache: Value cache tensor
        """
        # Concatenate K and V cache for storage
        combined = torch.cat([k_cache, v_cache], dim=-1)
        tensor_size = combined.element_size() * combined.nelement()
        
        # Check if adding would exceed GPU budget
        if self.device == "cuda" and self.total_gpu_size + tensor_size > self.max_gpu_cache_size:
            # Offload to CPU
            self._offload_least_recently_used_to_cpu()
        
        # Store in appropriate cache
        if self.device == "cuda":
            self.gpu_cache[layer_idx] = combined
            self.total_gpu_size += tensor_size
        else:
            self.cpu_cache[layer_idx] = combined.cpu() if combined.device.type == "cuda" else combined
            self.total_cpu_size += tensor_size
        
        # Update access time
        self.access_times[layer_idx] = time.time()
        
        logger.debug(f"Added cache for layer {layer_idx}, "
                    f"GPU: {self.total_gpu_size/1024/1024:.2f}MB, "
                    f"CPU: {self.total_cpu_size/1024/1024:.2f}MB")
    
    def get(self, layer_idx):
        """
        Get KV cache for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tuple of (k_cache, v_cache)
        """
        # Update access time
        self.access_times[layer_idx] = time.time()
        
        # Check if in GPU cache
        if layer_idx in self.gpu_cache:
            combined = self.gpu_cache[layer_idx]
        elif layer_idx in self.cpu_cache:
            combined = self.cpu_cache[layer_idx]
            
            # If using GPU, move to GPU
            if self.device == "cuda":
                # Move from CPU to GPU
                tensor_size = combined.element_size() * combined.nelement()
                
                # Make space if needed
                while self.total_gpu_size + tensor_size > self.max_gpu_cache_size:
                    self._offload_least_recently_used_to_cpu()
                
                # Move to GPU
                combined = combined.cuda()
                
                # Update caches
                self.gpu_cache[layer_idx] = combined
                self.total_gpu_size += tensor_size
                
                # Remove from CPU cache
                del self.cpu_cache[layer_idx]
                self.total_cpu_size -= tensor_size
        else:
            raise KeyError(f"No cache found for layer {layer_idx}")
        
        # Split combined cache back into K and V
        split_point = combined.size(-1) // 2
        k_cache = combined[..., :split_point]
        v_cache = combined[..., split_point:]
        
        return k_cache, v_cache
    
    def _offload_least_recently_used_to_cpu(self):
        """Offload the least recently used cache entry to CPU."""
        if not self.gpu_cache:
            return
        
        # Find the least recently used layer
        lru_layer = min(self.gpu_cache.keys(), key=lambda k: self.access_times.get(k, 0))
        combined = self.gpu_cache[lru_layer]
        tensor_size = combined.element_size() * combined.nelement()
        
        # Move to CPU
        self.cpu_cache[lru_layer] = combined.cpu()
        self.total_cpu_size += tensor_size
        
        # Remove from GPU
        del self.gpu_cache[lru_layer]
        self.total_gpu_size -= tensor_size
        
        logger.debug(f"Offloaded cache for layer {lru_layer} to CPU, "
                    f"freed {tensor_size/1024/1024:.2f}MB of GPU memory")
    
    def extend(self, layer_idx, new_k, new_v):
        """
        Extend existing KV cache with new key-value pairs.
        
        Args:
            layer_idx: Layer index
            new_k: New key tensor to append
            new_v: New value tensor to append
        """
        # Get existing cache
        if layer_idx in self.gpu_cache or layer_idx in self.cpu_cache:
            k_cache, v_cache = self.get(layer_idx)
            
            # Concatenate with new values
            k_extended = torch.cat([k_cache, new_k], dim=1)
            v_extended = torch.cat([v_cache, new_v], dim=1)
            
            # Store the extended cache
            self.add(layer_idx, k_extended, v_extended)
        else:
            # No existing cache, just add new values
            self.add(layer_idx, new_k, new_v)
    
    def prune(self, layer_idx, max_length):
        """
        Prune KV cache to a maximum length.
        
        Args:
            layer_idx: Layer index
            max_length: Maximum sequence length to keep
        """
        if layer_idx in self.gpu_cache or layer_idx in self.cpu_cache:
            k_cache, v_cache = self.get(layer_idx)
            
            # Check if pruning is needed
            current_length = k_cache.size(1)
            if current_length > max_length:
                # Keep only the last max_length entries
                k_pruned = k_cache[:, -max_length:]
                v_pruned = v_cache[:, -max_length:]
                
                # Store the pruned cache
                self.add(layer_idx, k_pruned, v_pruned)
                
                logger.debug(f"Pruned cache for layer {layer_idx} from {current_length} to {max_length}")
    
    def clear(self):
        """Clear all caches."""
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        self.access_times.clear()
        self.total_gpu_size = 0
        self.total_cpu_size = 0
        
        logger.debug("Cleared all KV caches")
    
    def memory_usage(self):
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory usage statistics
        """
        return {
            'gpu_cache_size_mb': self.total_gpu_size / (1024 * 1024),
            'cpu_cache_size_mb': self.total_cpu_size / (1024 * 1024),
            'gpu_cache_entries': len(self.gpu_cache),
            'cpu_cache_entries': len(self.cpu_cache),
            'total_entries': len(self.gpu_cache) + len(self.cpu_cache)
        }