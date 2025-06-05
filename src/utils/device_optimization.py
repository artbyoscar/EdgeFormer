# src/utils/device_optimization.py

import torch
import platform
import psutil

class DeviceOptimizer:
    """
    Provides device-specific optimization parameters based on hardware detection.
    """
    
    def __init__(self, model_config=None):
        self.model_config = model_config
        self.device_info = self._detect_device()
        self.optimization_profile = self._create_optimization_profile()
    
    def _detect_device(self):
        """Detect device hardware information"""
        device_info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "ram_gb": psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            device_info["gpu_name"] = torch.cuda.get_device_name(0)
            device_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return device_info
    
    def _create_optimization_profile(self):
        """Create device-specific optimization parameters"""
        profile = {
            "attention_strategy": "standard",  # standard, mla, or hybrid
            "attention_switch_length": 1024,   # sequence length to switch from standard to MLA
            "max_chunk_size": 1024,            # maximum chunk size for sequence processing
            "offload_threshold": 2048,         # sequence length to begin KV cache offloading
            "batch_size_factor": 1.0,          # multiplication factor for batch size
            "use_flash_attention": False       # whether to use flash attention when available
        }
        
        # Adjust for detected device
        is_low_end = False
        
        # Detect HP Envy or similar low-end devices
        if not torch.cuda.is_available() and "AMD" not in self.device_info["processor"]:
            is_low_end = True
        
        # Specific optimization for lower-end devices like HP Envy
        if is_low_end:
            profile["attention_strategy"] = "hybrid"
            profile["attention_switch_length"] = 512  # Switch to MLA earlier
            profile["max_chunk_size"] = 512
            profile["offload_threshold"] = 1024
            profile["batch_size_factor"] = 0.5
        
        return profile
    
    def get_optimal_attention_mechanism(self, seq_length):
        """Determine optimal attention mechanism for the given sequence length"""
        if self.optimization_profile["attention_strategy"] == "standard":
            return "standard"
        
        if self.optimization_profile["attention_strategy"] == "mla":
            return "mla"
        
        # For hybrid strategy, switch based on sequence length
        if seq_length < self.optimization_profile["attention_switch_length"]:
            return "standard"
        else:
            return "mla"
    
    def get_optimal_chunk_size(self, seq_length):
        """Determine optimal chunk size for processing sequences"""
        profile = self.optimization_profile
        
        # For short sequences, no chunking needed
        if seq_length <= profile["max_chunk_size"]:
            return seq_length
        
        # For long sequences, use the configured chunk size
        return profile["max_chunk_size"]
    
    def should_offload_kv_cache(self, seq_length):
        """Determine if KV cache should be offloaded for the given sequence length"""
        return seq_length >= self.optimization_profile["offload_threshold"]
    
    def get_optimal_batch_size(self, default_batch_size):
        """Get device-optimized batch size"""
        return max(1, int(default_batch_size * self.optimization_profile["batch_size_factor"]))
