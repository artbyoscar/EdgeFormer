# src/optimization/kv_cache_manager.py
import torch
import logging
import gc
import numpy as np
import time

logger = logging.getLogger(__name__)

class KVCacheManager:
    """
    Manages KV cache offloading between GPU and CPU memory.
    
    This allows handling long sequences that would exceed GPU memory
    by keeping only the active part of the KV cache in GPU memory.
    """
    
    def __init__(
        self,
        device="cuda",
        cpu_offload_threshold=2048,  # Sequence length to start offloading
        chunk_size=1024,             # Size of chunks to manage
        prefetch_size=128,           # How much to prefetch
        mem_efficient=True,          # Whether to use memory-efficient mode
        debug=False,                 # Whether to print debug info
    ):
        self.device = device
        self.cpu_offload_threshold = cpu_offload_threshold
        self.chunk_size = chunk_size
        self.prefetch_size = prefetch_size
        self.mem_efficient = mem_efficient
        self.debug = debug
        
        # Cache storage
        self.cpu_kv_cache = {}
        self.gpu_kv_cache = {}
        
        # Track which chunks are in GPU
        self.active_chunks = set()
        
        # Performance tracking
        self.transfer_times = []
        
        logger.info(f"Initialized KVCacheManager (offload_threshold={cpu_offload_threshold}, chunk_size={chunk_size})")
    
    def _get_chunk_id(self, seq_pos):
        """Get chunk ID for a sequence position."""
        return seq_pos // self.chunk_size
    
    def _get_chunk_range(self, chunk_id):
        """Get start and end positions for a chunk."""
        start = chunk_id * self.chunk_size
        end = start + self.chunk_size
        return start, end
    
    def _should_offload(self, seq_length):
        """Check if should use CPU offloading."""
        return seq_length > self.cpu_offload_threshold and self.mem_efficient
    
    def register_cache(self, layer_idx, kv_cache):
        """
        Register a new KV cache for a layer.
        
        Args:
            layer_idx: Layer index
            kv_cache: Tuple of (key_cache, value_cache)
        """
        seq_length = kv_cache[0].size(2)  # [batch, heads, seq, dim]
        
        if self._should_offload(seq_length):
            # Initialize CPU storage for this layer
            if layer_idx not in self.cpu_kv_cache:
                self.cpu_kv_cache[layer_idx] = {}
            
            # Store all chunks in CPU initially
            num_chunks = (seq_length + self.chunk_size - 1) // self.chunk_size
            
            for chunk_id in range(num_chunks):
                start, end = self._get_chunk_range(chunk_id)
                end = min(end, seq_length)
                
                # Extract this chunk
                k_chunk = kv_cache[0][:, :, start:end, :].cpu()
                v_chunk = kv_cache[1][:, :, start:end, :].cpu()
                
                # Store in CPU cache
                self.cpu_kv_cache[layer_idx][chunk_id] = (k_chunk, v_chunk)
            
            # Clear GPU cache to save memory
            if self.debug:
                logger.debug(f"Offloaded KV cache for layer {layer_idx} to CPU")
            
            # Return empty cache for GPU
            key_shape = list(kv_cache[0].size())
            value_shape = list(kv_cache[1].size())
            
            key_shape[2] = 0  # Empty sequence dimension
            value_shape[2] = 0
            
            empty_key = torch.empty(key_shape, dtype=kv_cache[0].dtype, device=self.device)
            empty_value = torch.empty(value_shape, dtype=kv_cache[1].dtype, device=self.device)
            
            return (empty_key, empty_value)
        else:
            # No offloading needed, use GPU cache directly
            if layer_idx not in self.gpu_kv_cache:
                self.gpu_kv_cache[layer_idx] = {}
            
            self.gpu_kv_cache[layer_idx]["full"] = kv_cache
            return kv_cache
    
    def get_cache_for_positions(self, layer_idx, positions):
        """
        Get KV cache for specific positions.
        
        Args:
            layer_idx: Layer index
            positions: List of positions to retrieve cache for
            
        Returns:
            retrieved_kv: KV cache for requested positions
        """
        if layer_idx in self.gpu_kv_cache and "full" in self.gpu_kv_cache[layer_idx]:
            # If full cache is in GPU, just index it
            full_kv = self.gpu_kv_cache[layer_idx]["full"]
            return (
                full_kv[0][:, :, positions, :],
                full_kv[1][:, :, positions, :]
            )
        
        # Determine which chunks are needed
        needed_chunks = set()
        for pos in positions:
            needed_chunks.add(self._get_chunk_id(pos))
        
        # Load needed chunks that aren't in GPU already
        for chunk_id in needed_chunks:
            if chunk_id not in self.active_chunks:
                self._load_chunk_to_gpu(layer_idx, chunk_id)
                self.active_chunks.add(chunk_id)
        
        # Collect cache entries for all positions
        result_keys = []
        result_values = []
        
        for pos in positions:
            chunk_id = self._get_chunk_id(pos)
            
            if layer_idx not in self.gpu_kv_cache or chunk_id not in self.gpu_kv_cache[layer_idx]:
                logger.warning(f"Missing cache for layer {layer_idx}, chunk {chunk_id}, pos {pos}")
                continue
                
            kv_chunk = self.gpu_kv_cache[layer_idx][chunk_id]
            
            # Calculate relative position within chunk
            rel_pos = pos - (chunk_id * self.chunk_size)
            
            # Get specific position
            k = kv_chunk[0][:, :, rel_pos:rel_pos+1, :]
            v = kv_chunk[1][:, :, rel_pos:rel_pos+1, :]
            
            result_keys.append(k)
            result_values.append(v)
        
        # Concatenate results
        if not result_keys:
            logger.warning(f"No cache entries found for layer {layer_idx}, positions {positions}")
            return None
            
        return (
            torch.cat(result_keys, dim=2),
            torch.cat(result_values, dim=2)
        )
    
    def _load_chunk_to_gpu(self, layer_idx, chunk_id):
        """Load a specific chunk from CPU to GPU."""
        if layer_idx not in self.cpu_kv_cache or chunk_id not in self.cpu_kv_cache[layer_idx]:
            logger.warning(f"Missing CPU cache for layer {layer_idx}, chunk {chunk_id}")
            return
        
        # Track transfer time
        start_time = time.time()
        
        # Get chunk from CPU
        k_chunk, v_chunk = self.cpu_kv_cache[layer_idx][chunk_id]
        
        # Move to GPU
        k_chunk = k_chunk.to(self.device)
        v_chunk = v_chunk.to(self.device)
        
        # Initialize GPU cache for this layer if needed
        if layer_idx not in self.gpu_kv_cache:
            self.gpu_kv_cache[layer_idx] = {}
        
        # Store in GPU cache
        self.gpu_kv_cache[layer_idx][chunk_id] = (k_chunk, v_chunk)
        
        # Track transfer time
        end_time = time.time()
        self.transfer_times.append(end_time - start_time)
        
        if self.debug:
            logger.debug(f"Loaded chunk {chunk_id} for layer {layer_idx} to GPU ({k_chunk.size(2)} positions)")
    
    def prefetch_chunks(self, current_pos):
        """Prefetch upcoming chunks to GPU."""
        current_chunk = self._get_chunk_id(current_pos)
        next_chunk = current_chunk + 1
        
        # Prefetch next chunk if not already loaded
        if next_chunk not in self.active_chunks:
            for layer_idx in self.cpu_kv_cache.keys():
                if next_chunk in self.cpu_kv_cache[layer_idx]:
                    self._load_chunk_to_gpu(layer_idx, next_chunk)
                    
            self.active_chunks.add(next_chunk)
            
            if self.debug:
                logger.debug(f"Prefetched chunk {next_chunk}")
    
    def add_to_cache(self, layer_idx, position, new_kv):
        """
        Add new KV entry to cache.
        
        Args:
            layer_idx: Layer index
            position: Position to add
            new_kv: New KV cache to add (key, value)
        """
        chunk_id = self._get_chunk_id(position)
        rel_pos = position - (chunk_id * self.chunk_size)
        
        if self._should_offload(position + 1):
            # Initialize CPU cache for this layer if needed
            if layer_idx not in self.cpu_kv_cache:
                self.cpu_kv_cache[layer_idx] = {}
            
            # Check if the chunk exists in CPU
            if chunk_id not in self.cpu_kv_cache[layer_idx]:
                # Create new chunk on CPU
                batch_size = new_kv[0].size(0)
                num_heads = new_kv[0].size(1)
                head_dim = new_kv[0].size(3)
                
                # Create empty tensors
                empty_k = torch.zeros(
                    batch_size, num_heads, self.chunk_size, head_dim,
                    dtype=new_kv[0].dtype, device="cpu"
                )
                empty_v = torch.zeros(
                    batch_size, num_heads, self.chunk_size, head_dim,
                    dtype=new_kv[1].dtype, device="cpu"
                )
                
                self.cpu_kv_cache[layer_idx][chunk_id] = (empty_k, empty_v)
            
            # Update in CPU cache
            k_chunk, v_chunk = self.cpu_kv_cache[layer_idx][chunk_id]
            k_chunk[:, :, rel_pos:rel_pos+1, :] = new_kv[0].cpu()
            v_chunk[:, :, rel_pos:rel_pos+1, :] = new_kv[1].cpu()
            
            # If the chunk is also in GPU, update it there
            if layer_idx in self.gpu_kv_cache and chunk_id in self.gpu_kv_cache[layer_idx]:
                k_gpu, v_gpu = self.gpu_kv_cache[layer_idx][chunk_id]
                k_gpu[:, :, rel_pos:rel_pos+1, :] = new_kv[0]
                v_gpu[:, :, rel_pos:rel_pos+1, :] = new_kv[1]
        else:
            # Just use GPU cache
            if layer_idx not in self.gpu_kv_cache:
                self.gpu_kv_cache[layer_idx] = {}
            
            if "full" not in self.gpu_kv_cache[layer_idx]:
                # Need to initialize full cache
                return
            
            # Update in full GPU cache
            k_full, v_full = self.gpu_kv_cache[layer_idx]["full"]
            
            if position >= k_full.size(2):
                # Need to expand cache
                new_size = position + 1
                new_k = torch.zeros(
                    k_full.size(0), k_full.size(1), new_size, k_full.size(3),
                    dtype=k_full.dtype, device=self.device
                )
                new_v = torch.zeros(
                    v_full.size(0), v_full.size(1), new_size, v_full.size(3),
                    dtype=v_full.dtype, device=self.device
                )
                
                # Copy existing cache
                new_k[:, :, :k_full.size(2), :] = k_full
                new_v[:, :, :v_full.size(2), :] = v_full
                
                # Update full cache reference
                self.gpu_kv_cache[layer_idx]["full"] = (new_k, new_v)
                k_full, v_full = new_k, new_v
            
            # Add new entry
            k_full[:, :, position:position+1, :] = new_kv[0]
            v_full[:, :, position:position+1, :] = new_kv[1]
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cpu_kv_cache.clear()
        self.gpu_kv_cache.clear()
        self.active_chunks.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleared KV cache")
    
    def get_stats(self):
        """Get memory usage statistics."""
        gpu_bytes = 0
        cpu_bytes = 0
        
        # Count GPU memory
        for layer_cache in self.gpu_kv_cache.values():
            for kv in layer_cache.values():
                gpu_bytes += kv[0].element_size() * kv[0].nelement()
                gpu_bytes += kv[1].element_size() * kv[1].nelement()
        
        # Count CPU memory
        for layer_cache in self.cpu_kv_cache.values():
            for kv in layer_cache.values():
                cpu_bytes += kv[0].element_size() * kv[0].nelement()
                cpu_bytes += kv[1].element_size() * kv[1].nelement()
        
        # Get transfer stats
        avg_transfer_time = np.mean(self.transfer_times) if self.transfer_times else 0
        
        return {
            "gpu_memory_mb": gpu_bytes / (1024 * 1024),
            "cpu_memory_mb": cpu_bytes / (1024 * 1024),
            "total_memory_mb": (gpu_bytes + cpu_bytes) / (1024 * 1024),
            "active_chunks": len(self.active_chunks),
            "avg_transfer_time_ms": avg_transfer_time * 1000,
        }