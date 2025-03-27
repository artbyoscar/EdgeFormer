# src/utils/device.py
import torch
import platform
import logging
import os
import numpy as np

# Add ONNX Runtime import with fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    
    # Check if DirectML provider is available
    DIRECTML_AVAILABLE = 'DmlExecutionProvider' in ort.get_available_providers()
except ImportError:
    ONNX_AVAILABLE = False
    DIRECTML_AVAILABLE = False

logger = logging.getLogger("edgeformer")

def get_device():
    """
    Get the best available device, with special handling for AMD GPUs.
    
    Returns:
        torch.device: The best available device
    """
    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {device_name}")
        return torch.device("cuda")
    
    # Check for ROCm (AMD)
    if hasattr(torch, 'hip') and torch.hip.is_available():
        try:
            device_name = torch.hip.get_device_name(0)
            logger.info(f"Using ROCm GPU: {device_name}")
            return torch.device("hip")
        except:
            logger.info("ROCm detected but couldn't get device name")
            return torch.device("hip")
    
    # Check specifically for AMD hardware but without ROCm
    system_info = platform.system()
    if system_info == "Windows":
        import subprocess
        try:
            output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
            if "AMD" in output or "Radeon" in output:
                if DIRECTML_AVAILABLE:
                    logger.info("AMD GPU detected with DirectML support available.")
                    # DirectML will be used via ONNX Runtime, but PyTorch still uses CPU
                    return torch.device("cpu")
                else:
                    logger.warning("AMD GPU detected but DirectML support is not available. Using CPU instead.")
                    logger.warning("To enable DirectML GPU support, install onnxruntime-directml:")
                    logger.warning("pip install --no-cache-dir --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-directml/pypi/simple/ onnxruntime-directml")
        except:
            pass
    
    # Fallback to CPU
    logger.info("No GPU detected. Using CPU.")
    return torch.device("cpu")

def print_device_info():
    """Print detailed information about available compute devices."""
    device = get_device()
    
    print("\n=== Device Information ===")
    print(f"Device selected: {device}")
    
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    elif device.type == "hip":
        print("AMD ROCm GPU active")
    else:
        import platform
        print(f"CPU: {platform.processor()}")
        
        # Check for DirectML
        if DIRECTML_AVAILABLE:
            print("DirectML acceleration available for AMD GPUs via ONNX Runtime")
            
            # Try to get more info about the GPU
            try:
                import subprocess
                output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
                for line in output.strip().split('\n')[1:]:  # Skip header
                    if "AMD" in line or "Radeon" in line:
                        print(f"DirectML GPU: {line.strip()}")
            except:
                pass
    
    print(f"PyTorch Version: {torch.__version__}")
    if ONNX_AVAILABLE:
        print(f"ONNX Runtime Version: {ort.__version__}")
        print(f"Available Providers: {ort.get_available_providers()}")
    print("===========================\n")
    
    return device

def export_to_onnx(model, onnx_path, input_shape=(1, 512)):
    """Export PyTorch model to ONNX format optimized for DirectML."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime is required for ONNX export. Install with: pip install onnxruntime")
    
    # Create dummy inputs
    input_ids = torch.randint(0, model.config.vocab_size, input_shape)
    attention_mask = torch.ones(input_shape)
    
    # Set dynamic axes for variable sequence length
    dynamic_axes = {
        'input_ids': {1: 'sequence_length'},
        'attention_mask': {1: 'sequence_length'},
        'logits': {1: 'sequence_length'}
    }
    
    # Make sure model is in eval mode
    model.eval()
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            opset_version=13,
            do_constant_folding=True
        )
    
    logger.info(f"Model exported to ONNX format: {onnx_path}")
    return onnx_path

def get_optimized_directml_session(onnx_path):
    """Create an optimized ONNX session with DirectML provider."""
    if not DIRECTML_AVAILABLE:
        raise ImportError("DirectML provider is not available. Install with: pip install --no-cache-dir --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-directml/pypi/simple/ onnxruntime-directml")
    
    # Create session options with optimizations
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = False  # Helps with DirectML performance
    
    # Create DirectML session
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    
    # DirectML-specific provider options
    provider_options = [
        {
            'device_id': 0,  # Use first GPU
            'enable_dynamic_block_dma': True  # Improve memory transfer
        }, 
        {}  # Default options for CPU provider
    ]
    
    # Create session
    session = ort.InferenceSession(
        onnx_path, 
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options
    )
    
    return session

def run_with_directml(session, input_ids, attention_mask):
    """Run inference using DirectML provider."""
    if not isinstance(session, ort.InferenceSession):
        raise TypeError("session must be an ONNX Runtime InferenceSession")
    
    # Convert inputs to numpy
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.cpu().numpy()
    
    # Create input dictionary
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # Run inference
    outputs = session.run(None, inputs)
    
    # Convert output back to PyTorch tensor
    logits = torch.from_numpy(outputs[0])
    
    return {'logits': logits}

def is_amd_gpu_available():
    """Check if an AMD GPU is available on the system."""
    # Check for ROCm first
    if hasattr(torch, 'hip') and torch.hip.is_available():
        return True
    
    # Check for AMD GPU on Windows via WMI
    if platform.system() == "Windows":
        try:
            import subprocess
            output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
            return "AMD" in output or "Radeon" in output
        except:
            pass
    
    # Check for AMD GPU on Linux via lspci
    if platform.system() == "Linux":
        try:
            import subprocess
            output = subprocess.check_output("lspci | grep -i amd", shell=True).decode()
            return "Radeon" in output or "AMD" in output
        except:
            pass
    
    return False

def optimize_for_amd(model):
    """Apply optimizations specific to AMD GPUs."""
    # For now, just make sure we're using the right device
    device = get_device()
    model.to(device)
    
    # If DirectML is available, we'll use that for inference instead
    # So no need to modify the PyTorch model
    
    return model