import torch
import numpy as np
import gc
import logging

logger = logging.getLogger("edgeformer")

class KVCacheManager:
    """
    Manages Key-Value cache for transformer models, with support for offloading to CPU
    when VRAM is limited.
    """
    def __init__(
        self, 
        max_batch_size=1, 
        max_seq_length=8192, 
        num_layers=4,
        num_heads=8, 
        head_dim=64,
        max_gpu_cache_size=1024,  # Size in MB beyond which to offload to CPU
        enable_offload=True,      # Whether to enable offloading to CPU
        device="cpu"
    ):
        """
        Initialize KV Cache Manager.
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            max_gpu_cache_size: Maximum GPU cache size in MB before offloading
            enable_offload: Whether to enable offloading to CPU RAM
            device: Default device to use
        """
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_gpu_cache_size = max_gpu_cache_size * 1024 * 1024  # Convert to bytes
        self.enable_offload = enable_offload
        self.device = device
        
        # Calculate cache entry size
        self.element_size = 4  # Float32 - 4 bytes
        
        # Total cache entry size in bytes (keys + values for all layers)
        self.cache_entry_size_bytes = (
            self.max_batch_size * 
            self.max_seq_length * 
            self.num_layers * 
            self.num_heads * 
            self.head_dim * 
            2 *  # Keys and values
            self.element_size
        )
        
        logger.info(f"KV Cache entry size: {self.cache_entry_size_bytes / (1024 * 1024):.2f} MB")
        
        # Initialize cache
        self.reset()
    
    def reset(self):
        """Reset the cache to empty state"""
        # GPU cache (for recent tokens)
        self.gpu_cache_k = []
        self.gpu_cache_v = []
        
        # CPU cache (for older tokens)
        self.cpu_cache_k = []
        self.cpu_cache_v = []
        
        # Current sequence length in cache
        self.current_seq_length = 0
        
        # Create empty caches
        for _ in range(self.num_layers):
            # Initialize with small cache size
            initial_size = min(128, self.max_seq_length)
            
            # GPU cache - initialize with correct dimensions
            layer_k = torch.zeros(
                self.max_batch_size, 
                initial_size,
                self.num_heads, 
                self.head_dim, 
                device=self.device
            )
            layer_v = torch.zeros(
                self.max_batch_size, 
                initial_size,
                self.num_heads, 
                self.head_dim, 
                device=self.device
            )
            
            self.gpu_cache_k.append(layer_k)
            self.gpu_cache_v.append(layer_v)
            
            # CPU cache (initially empty)
            self.cpu_cache_k.append(None)
            self.cpu_cache_v.append(None)
        
        logger.info("KV Cache reset")
    
    def update(self, layer_idx, key, value):
        """
        Update the KV cache for a specific layer with new key and value tensors.
        
        Args:
            layer_idx: Layer index
            key: Key tensor (ensure proper shape)
            value: Value tensor (ensure proper shape)
        """
        # Ensure key and value have the right shape [batch_size, seq_len, num_heads, head_dim]
        if key.dim() != 4:
            raise ValueError(f"Expected key with 4 dimensions, got {key.dim()}")
        
        # Update current sequence length
        new_tokens = key.size(1)
        total_length = self.current_seq_length + new_tokens
        
        # Check if we need to offload
        if (
            self.enable_offload and 
            total_length > 128 and  # Don't offload small caches
            self._estimate_gpu_cache_size() > self.max_gpu_cache_size
        ):
            self._offload_to_cpu()
        
        # Check if we need to expand the GPU cache
        if total_length > self.gpu_cache_k[layer_idx].size(1):
            # Double the size, but don't exceed max_seq_length
            new_size = min(
                self.max_seq_length,
                max(total_length, self.gpu_cache_k[layer_idx].size(1) * 2)
            )
            
            # Create new bigger tensors
            new_k = torch.zeros(
                self.max_batch_size, 
                new_size,
                self.num_heads, 
                self.head_dim, 
                device=self.device
            )
            new_v = torch.zeros(
                self.max_batch_size, 
                new_size,
                self.num_heads, 
                self.head_dim, 
                device=self.device
            )
            
            # Copy existing cache - fix dimensions
            new_k[:, :self.current_seq_length] = self.gpu_cache_k[layer_idx][:, :self.current_seq_length]
            new_v[:, :self.current_seq_length] = self.gpu_cache_v[layer_idx]
            
            # Replace cache
            self.gpu_cache_k[layer_idx] = new_k
            self.gpu_cache_v[layer_idx] = new_v
            
            logger.debug(f"Expanded GPU cache for layer {layer_idx} to size {new_size}")
        
        # Update cache with new key and value
        self.gpu_cache_k[layer_idx][:, self.current_seq_length:total_length] = key
        self.gpu_cache_v[layer_idx][:, self.current_seq_length:total_length] = value
        
        # Update current sequence length
        self.current_seq_length = total_length
    
    def get(self, layer_idx, start_idx=None, end_idx=None):
        """
        Get the cached keys and values for a specific layer.
        
        Args:
            layer_idx: Layer index
            start_idx: Start token index (or None for all)
            end_idx: End token index (or None for all)
            
        Returns:
            Tuple of (keys, values) tensors
        """
        # Default indices
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.current_seq_length
        
        # Check if we need to load from CPU
        if self.cpu_cache_k[layer_idx] is not None and start_idx < self.cpu_cache_k[layer_idx].size(1):
            # We need to fetch from CPU cache
            self._load_from_cpu(layer_idx, start_idx, end_idx)
        
        # Return the requested slice from GPU cache
        k = self.gpu_cache_k[layer_idx][:, start_idx:end_idx]
        v = self.gpu_cache_v[layer_idx][:, start_idx:end_idx]
        
        return k, v
    
    def _estimate_gpu_cache_size(self):
        """
        Estimate the current GPU cache size in bytes.
        
        Returns:
            Size in bytes
        """
        total_elements = 0
        for layer_idx in range(self.num_layers):
            k_elements = self.gpu_cache_k[layer_idx].numel()
            v_elements = self.gpu_cache_v[layer_idx].numel()
            total_elements += k_elements + v_elements
        
        return total_elements * self.element_size
    
    def _offload_to_cpu(self):
        """Offload older tokens to CPU to save GPU memory"""
        # Keep the most recent 128 tokens on GPU, offload the rest
        keep_tokens = min(128, self.current_seq_length)
        offload_tokens = self.current_seq_length - keep_tokens
        
        if offload_tokens <= 0:
            return
        
        logger.info(f"Offloading {offload_tokens} tokens to CPU, keeping {keep_tokens} on GPU")
        
        for layer_idx in range(self.num_layers):
            # Get the tokens to offload
            offload_k = self.gpu_cache_k[layer_idx][:, :offload_tokens].cpu()
            offload_v = self.gpu_cache_v[layer_idx][:, :offload_tokens].cpu()
            
            # Store in CPU cache
            if self.cpu_cache_k[layer_idx] is None:
                self.cpu_cache_k[layer_idx] = offload_k
                self.cpu_cache_v[layer_idx] = offload_v
            else:
                # Concatenate with existing CPU cache
                self.cpu_cache_k[layer_idx] = torch.cat(
                    [self.cpu_cache_k[layer_idx], offload_k], dim=1
                )
                self.cpu_cache_v[layer_idx] = torch.cat(
                    [self.cpu_cache_v[layer_idx], offload_v], dim=1
                )
            
            # Move the remaining tokens to the beginning of the GPU cache
            self.gpu_cache_k[layer_idx][:, :keep_tokens] = self.gpu_cache_k[layer_idx][:, offload_tokens:self.current_seq_length]
            self.gpu_cache_v[layer_idx][:, :keep_tokens] = self.gpu_cache_v[layer_idx][:, offload_tokens:self.current_seq_length]
        
        # Update current sequence length in GPU cache
        self.current_seq_length = keep_tokens
        
        # Run garbage collection to free memory
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()
    
    def _load_from_cpu(self, layer_idx, start_idx, end_idx):
        """
        Load tokens from CPU cache to GPU cache as needed.
        
        Args:
            layer_idx: Layer index
            start_idx: Start token index
            end_idx: End token index
        """
        if self.cpu_cache_k[layer_idx] is None:
            return
        
        cpu_cache_size = self.cpu_cache_k[layer_idx].size(1)
        
        # Calculate how many tokens we need from CPU
        cpu_start = max(0, start_idx)
        cpu_end = min(cpu_cache_size, end_idx)
        
        if cpu_start >= cpu_end:
            return
        
        logger.info(f"Loading tokens {cpu_start}-{cpu_end} from CPU for layer {layer_idx}")
        
        # Adjust GPU cache to have enough space
        total_needed = end_idx - start_idx
        if total_needed > self.gpu_cache_k[layer_idx].size(1):
            # Create new bigger tensors
            new_size = max(total_needed, self.gpu_cache_k[layer_idx].size(1) * 2)
            new_k = torch.zeros(
                self.max_batch_size, 
                new_size,
                self.num_heads, 
                self.head_dim, 
                device=self.device
            )
            new_v = torch.zeros(
                self.max_batch_size, 
                new_size,
                self.num_heads, 
                self.head_dim, 
                device=self.device
            )
            
            # Replace cache
            self.gpu_cache_k[layer_idx] = new_k
            self.gpu_cache_v[layer_idx] = new_v
        
        # Copy from CPU to GPU
        self.gpu_cache_k[layer_idx][:, start_idx:cpu_end] = self.cpu_cache_k[layer_idx][:, cpu_start:cpu_end].to(self.device)
        self.gpu_cache_v[layer_idx][:, start_idx:cpu_end] = self.cpu_cache_v[layer_idx][:, cpu_start:cpu_end].to(self.device)