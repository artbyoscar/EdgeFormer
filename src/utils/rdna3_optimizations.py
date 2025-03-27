import torch
import logging
import os
import platform
import subprocess
import re

logger = logging.getLogger("edgeformer")

def is_rdna3_gpu():
    """
    Detect if the system has an AMD RDNA3 architecture GPU.
    
    Returns:
        bool: True if RDNA3 GPU is detected, False otherwise
    """
    if platform.system() == "Windows":
        try:
            # Get GPU info using Windows Management Instrumentation
            output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
            
            # RDNA3 GPUs include RX 7000 series
            rdna3_patterns = [
                r'RX\s+7\d{3}',  # RX 7900, RX 7800, etc.
                r'Radeon\s+RX\s+7\d{3}',
                r'AMD\s+Radeon\s+RX\s+7\d{3}',
                r'Radeon\s+RX\s+7\d{3}\s+XT',
                r'Radeon\s+Pro\s+W7\d{3}'  # Workstation GPUs
            ]
            
            for line in output.strip().split('\n'):
                for pattern in rdna3_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        logger.info(f"Detected RDNA3 GPU: {line.strip()}")
                        return True
        except Exception as e:
            logger.warning(f"Error detecting GPU: {str(e)}")
    
    elif platform.system() == "Linux":
        try:
            # Try using lspci to detect GPU
            output = subprocess.check_output("lspci | grep -i amd", shell=True).decode()
            
            # Look for RDNA3 identifiers
            if "RDNA3" in output or "Navi 3" in output or any(f"Navi 3{x}" in output for x in range(10)):
                for line in output.strip().split('\n'):
                    if "RDNA3" in line or "Navi 3" in line or any(f"Navi 3{x}" in line for x in range(10)):
                        logger.info(f"Detected RDNA3 GPU: {line.strip()}")
                        return True
            
            # Try with AMD GPU detection tool if available
            if os.path.exists("/opt/rocm/bin/rocm-smi"):
                output = subprocess.check_output("/opt/rocm/bin/rocm-smi --showproductname", shell=True).decode()
                for line in output.strip().split('\n'):
                    if "7" in line and ("XT" in line or "XTX" in line):
                        logger.info(f"Detected RDNA3 GPU: {line.strip()}")
                        return True
        except Exception as e:
            logger.warning(f"Error detecting GPU: {str(e)}")
    
    return False

def optimize_for_rdna3(model):
    """
    Apply RDNA3-specific optimizations to the model.
    
    Args:
        model: The EdgeFormer model to optimize
        
    Returns:
        The optimized model
    """
    if not is_rdna3_gpu():
        logger.info("No RDNA3 GPU detected, skipping RDNA3 optimizations")
        return model
    
    logger.info("Applying RDNA3-specific optimizations...")
    
    # RDNA3 architecture has:
    # - 128 KB L1 cache per CU
    # - Larger L2 cache
    # - Hardware matrix units optimized for specific sizes
    # - Better performance with certain memory access patterns
    
    # 1. Optimize block sizes for RDNA3
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
        for layer_idx, layer in enumerate(model.transformer.layer):
            # Optimize attention block sizes
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'latent_attention'):
                attn = layer.attention.latent_attention
                
                # Set optimal block sizes for RDNA3 matrix operations
                # Block size of 64 works well with RDNA3 matrix units
                if hasattr(attn, 'block_size'):
                    old_block_size = attn.block_size
                    attn.block_size = 64
                    logger.info(f"Layer {layer_idx} attention: Changed block size from {old_block_size} to 64")
                
            # Optimize MLP block sizes
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'sparse_mlp'):
                mlp = layer.mlp.sparse_mlp
                
                # Set optimal block sizes for RDNA3 matrix operations
                # Block size of 128 works well with RDNA3 matrix units for larger matrices
                if hasattr(mlp, 'block_size'):
                    old_block_size = mlp.block_size
                    mlp.block_size = 128
                    logger.info(f"Layer {layer_idx} MLP: Changed block size from {old_block_size} to 128")
    
    # 2. Memory access optimizations
    # RDNA3 has better performance with specific memory layouts and striding patterns
    
    # 3. Set configuration flags that may be used by the model
    if not hasattr(model, 'config'):
        model.config = {}
    
    model.config.rdna3_optimized = True
    
    # Log completion
    logger.info("RDNA3 optimizations applied successfully")
    
    return model

def create_directml_provider_options_for_rdna3():
    """
    Create optimized DirectML provider options for RDNA3 GPUs.
    
    Returns:
        A list of provider options for DirectML and CPU providers
    """
    try:
        import onnxruntime as ort
        
        # Check if DirectML provider is available
        if 'DmlExecutionProvider' not in ort.get_available_providers():
            logger.warning("DirectML provider not available")
            return None
        
        # Check if we have an RDNA3 GPU
        if not is_rdna3_gpu():
            logger.info("No RDNA3 GPU detected, using standard DirectML options")
            return [
                {
                    'device_id': 0,
                    'enable_dynamic_block_dma': True,
                }, 
                {}  # Default options for CPU provider
            ]
        
        # RDNA3-specific DirectML provider options
        rdna3_provider_options = [
            {
                'device_id': 0,
                'enable_dynamic_block_dma': True,
                'preferred_scale_for_texture_type': 'FLOAT16',  # Use FP16 for texture operations
                'enable_subgraph_fusion': True,                 # Enable fusion optimizations
                'enable_memory_pool': True,                     # Enable memory pooling
                'memory_pool_size': 1024 * 1024 * 256,          # 256MB memory pool
                'enable_texture_data_type': True,               # Use texture optimizations
                'tiled_resources': True,                        # AMD GPUs benefit from tiled resources
                'enable_dynamic_graph_optimization': True,      # Enable dynamic graph optimizations
                'execution_mode': "sequential"                  # Sequential execution mode often better for AMD
            },
            {}  # Default options for CPU provider
        ]
        
        logger.info("Created RDNA3-optimized DirectML provider options")
        return rdna3_provider_options
    
    except ImportError:
        logger.warning("ONNX Runtime not available")
        return None

def create_torch_directml_config_for_rdna3():
    """
    Create optimized torch-directml configuration for RDNA3 GPUs.
    
    Returns:
        A dictionary with torch-directml configuration options
    """
    try:
        import torch_directml
        
        if not is_rdna3_gpu():
            logger.info("No RDNA3 GPU detected, using standard torch-directml config")
            return {}
        
        # RDNA3-specific torch-directml configuration
        rdna3_config = {
            "allow_tensor_reuse": True,             # Allow tensor reuse for better memory efficiency
            "debug_layer": False,                    # Disable debug layer for better performance
            "default_device_id": 0,                  # Use first GPU
            "use_fp16": True,                        # Enable FP16 where possible
            "enable_tensor_pooling": True,           # Enable tensor pooling
            "tensor_pool_size": 1024 * 1024 * 256,   # 256MB tensor pool
            "enable_tiled_resources": True,          # Enable tiled resources for AMD GPUs
            "execution_mode": "graph",               # Use graph execution mode for better performance
        }
        
        logger.info("Created RDNA3-optimized torch-directml configuration")
        return rdna3_config
    
    except ImportError:
        logger.warning("torch-directml not available")
        return {}

def get_rdna3_device():
    """
    Get the best device for RDNA3 GPUs.
    
    Returns:
        torch.device: The device to use
    """
    # Try torch-directml first
    try:
        if is_rdna3_gpu():
            import torch_directml
            
            # Get RDNA3-optimized configuration
            config = create_torch_directml_config_for_rdna3()
            
            # Create device with config
            device = torch_directml.device(**config)
            logger.info("Using torch-directml for RDNA3 GPU acceleration")
            return device
    except ImportError:
        pass
    
    # Try ROCm
    if hasattr(torch, 'hip') and torch.hip.is_available() and is_rdna3_gpu():
        logger.info("Using ROCm for RDNA3 GPU acceleration")
        return torch.device("hip")
    
    # Fallback to standard device selection
    from src.utils.device import get_device
    return get_device()