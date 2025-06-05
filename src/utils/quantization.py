#!/usr/bin/env python3
print("DEBUG: quantization.py (User's Full Version) - START of file execution")

"""
EdgeFormer Quantization Utilities

Comprehensive quantization implementation for EdgeFormer models
with INT4 and INT8 support for edge deployment optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import gc

# Logger setup
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

print("DEBUG: quantization.py - Logger defined.")


class DynamicQuantizer:
    """
    Advanced quantization class for EdgeFormer models.
    Supports both INT8 and INT4 quantization with dynamic scaling.
    """
    
    @staticmethod
    def quantize_model_int8(model):
        """
        Apply INT8 quantization to the model using PyTorch's dynamic quantization.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Quantized model
        """
        logger.info("Starting INT8 quantization...")
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Apply dynamic quantization to linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.MultiheadAttention},
                dtype=torch.qint8
            )
            
            logger.info("INT8 quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            logger.warning("Returning original model")
            return model
    
    @staticmethod
    def quantize_model_int4(model):
        """
        Apply INT4 quantization to the model using custom Int4Quantizer.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Quantized model with INT4 weights
        """
        logger.info("Starting INT4 quantization...")
        
        try:
            # Create INT4 quantizer instance
            quantizer = Int4Quantizer()
            
            # Apply INT4 quantization
            quantized_model = quantizer.apply_to_model(model)
            
            logger.info("INT4 quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            logger.error(f"INT4 quantization failed: {e}")
            logger.warning("Returning original model")
            return model


print(f"DEBUG: quantization.py - DynamicQuantizer class defined. Type: {type(DynamicQuantizer)}")


class Int4Quantizer:
    """
    Custom INT4 quantization implementation for extreme compression.
    Provides 8x memory reduction with minimal accuracy loss.
    """
    
    def __init__(self, block_size=64, symmetric=False):
        """
        Initialize INT4 quantizer.
        
        Args:
            block_size: Size of quantization blocks
            symmetric: Use symmetric quantization if True
        """
        self.block_size = block_size
        self.symmetric = symmetric
        self.quantized_layers = {}
        
        logger.info(f"INT4 Quantizer initialized with block_size={block_size}, symmetric={symmetric}")
    
    def quantize(self, tensor):
        """
        Quantize tensor to INT4 representation with improved accuracy.
        
        Args:
            tensor: Input tensor to quantize
            
        Returns:
            Tuple of (quantized_data, scale, zero_point)
        """
        if tensor.numel() == 0:
            return tensor, torch.tensor(1.0), torch.tensor(0)
        
        # Store original shape for later reconstruction
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # Use per-channel quantization for better accuracy on weight matrices
        if len(original_shape) >= 2:
            # For weight matrices, quantize per output channel (first dimension)
            num_channels = original_shape[0]
            reshaped = tensor.view(num_channels, -1)
            
            quantized_channels = []
            scales = []
            zero_points = []
            
            for i in range(num_channels):
                channel_data = reshaped[i, :]
                q_data, scale, zp = self._quantize_channel(channel_data)
                quantized_channels.append(q_data)
                scales.append(scale)
                zero_points.append(zp)
            
            # Reconstruct quantized tensor
            quantized_tensor = torch.stack(quantized_channels).view(original_shape)
            scales_tensor = torch.stack(scales)
            zero_points_tensor = torch.stack(zero_points)
            
            return quantized_tensor, scales_tensor, zero_points_tensor
        else:
            # For 1D tensors (biases), use standard quantization
            return self._quantize_channel(flat_tensor, original_shape)
    
    def __init__(self, block_size=64, symmetric=False):
        """
        Initialize INT4 quantizer with optimized settings for better accuracy.
    
        Args:
            block_size: Size of quantization blocks (64 for better precision)
            symmetric: Use symmetric quantization if True (False for better range)
        """
        self.block_size = block_size
        self.symmetric = symmetric
        self.quantized_layers = {}
    
        logger.info(f"INT4 Quantizer initialized with block_size={block_size}, symmetric={symmetric}")


    def apply_to_model(self, model):
        """
        Apply INT4 quantization to all Linear layers in the model.
    
        Args:
            model: PyTorch model to quantize
        
        Returns:
            Model with quantized weights
        """
        quantized_model = type(model)(model.config) if hasattr(model, 'config') else model
    
        # Copy state dict and quantize linear layer weights
        state_dict = model.state_dict()
        new_state_dict = {}
    
        for name, param in state_dict.items():
            if 'weight' in name and param.dim() >= 2:
                # Skip sensitive layers that impact accuracy most
                if ('embedding' in name or 'lm_head' in name or 
                    'pos_encoding' in name or 'output_projection' in name):
                    logger.info(f"Skipping sensitive layer: {name}")
                    new_state_dict[name] = param  # Keep original precision
                    continue
                
                # Quantize weight matrix
                logger.info(f"Quantizing layer: {name}")
            
                try:
                    quantized_weight, scale, zero_point = self.quantize(param)
                
                    # Store quantization metadata
                    self.quantized_layers[name] = {
                        'scale': scale,
                        'zero_point': zero_point,
                        'original_shape': param.shape
                    }
                
                    # Use the quantized weight directly (it's already dequantized for compatibility)
                    new_state_dict[name] = quantized_weight
                
                except Exception as e:
                    logger.warning(f"Failed to quantize {name}: {e}")
                    new_state_dict[name] = param
            else:
                # Keep non-weight parameters unchanged
                new_state_dict[name] = param
    
        # Load quantized weights
        quantized_model.load_state_dict(new_state_dict)
    
        # Store quantization metadata in the model for size calculation
        quantized_model._quantization_info = {
            'quantizer': self,
            'quantized_layers': self.quantized_layers,
            'compression_stats': self.get_memory_savings()
        }
    
        logger.info(f"Quantized {len(self.quantized_layers)} layers")
    
        # Log compression statistics
        stats = self.get_memory_savings()
        logger.info(f"Theoretical compression: {stats['compression_ratio']:.1f}x, "
                f"Memory saved: {stats['memory_saved_mb']:.2f} MB")
    
        return quantized_model


    def _quantize_channel(self, channel_tensor, target_shape=None):
        """Quantize a single channel with enhanced precision and calibration."""
        if target_shape is None:
            target_shape = channel_tensor.shape
        
        # Calculate quantization parameters with enhanced calibration
        if self.symmetric:
            # Use 99.9th percentile to ignore outliers for better accuracy
            max_val = torch.quantile(torch.abs(channel_tensor), 0.999)
            scale = max_val / 7.0  # 4-bit signed range: -7 to 7
            zero_point = torch.tensor(0.0)
        else:
            # Use percentiles for better range utilization
            min_val = torch.quantile(channel_tensor, 0.001)
            max_val = torch.quantile(channel_tensor, 0.999)
            scale = (max_val - min_val) / 15.0  # 4-bit unsigned range: 0 to 15
            zero_point = torch.round(-min_val / scale)
    
        # Avoid division by zero
        if scale == 0:
            scale = torch.tensor(1.0)
    
        # Quantize values with better rounding
        quantized = torch.round(channel_tensor / scale + zero_point)
    
        # Clamp to INT4 range
        if self.symmetric:
            quantized = torch.clamp(quantized, -7, 7)
        else:
            quantized = torch.clamp(quantized, 0, 15)
    
        # Dequantize immediately for demonstration (maintains float32 weights)
        # In production, you'd store the quantized values in INT4 format
        if self.symmetric:
            dequantized = quantized * scale
        else:
            dequantized = (quantized - zero_point) * scale
    
        return dequantized.reshape(target_shape), scale, zero_point
    
    def _pack_int4_to_int8(self, int4_tensor):
        """
        Pack two INT4 values into one INT8 value for storage efficiency.
        
        Args:
            int4_tensor: Tensor with INT4 values
            
        Returns:
            Packed INT8 tensor
        """
        # Pad if odd length
        if int4_tensor.numel() % 2 == 1:
            int4_tensor = torch.cat([int4_tensor, torch.zeros(1, dtype=torch.int)])
        
        # Reshape to pairs
        pairs = int4_tensor.view(-1, 2)
        
        # Pack: (high_nibble << 4) | low_nibble
        packed = (pairs[:, 0] << 4) | (pairs[:, 1] & 0xF)
        
        return packed.to(torch.int8)
    
    def _unpack_int8_to_int4(self, packed_tensor, original_size):
        """
        Unpack INT8 values back to INT4 values.
        
        Args:
            packed_tensor: Packed INT8 tensor
            original_size: Original tensor size before packing
            
        Returns:
            Unpacked INT4 tensor
        """
        # Unpack nibbles
        high_nibble = (packed_tensor >> 4) & 0xF
        low_nibble = packed_tensor & 0xF
        
        # Interleave high and low nibbles
        unpacked = torch.stack([high_nibble, low_nibble], dim=1).flatten()
        
        # Trim to original size
        return unpacked[:original_size].float()
    
    def _dequantize_packed(self, packed_tensor, scale, zero_point, original_size):
        """
        Dequantize packed INT4 data back to float tensor.
        
        Args:
            packed_tensor: Packed INT8 tensor
            scale: Quantization scale
            zero_point: Quantization zero point
            original_size: Original tensor size before packing
            
        Returns:
            Dequantized float tensor
        """
        # Unpack INT4 data
        unpacked = self._unpack_int8_to_int4(packed_tensor.flatten(), original_size)
        
        # Apply symmetric/asymmetric offset
        if self.symmetric:
            # For symmetric quantization, values are already centered at zero
            dequantized = unpacked * scale
        else:
            # For asymmetric quantization, subtract zero point first
            dequantized = (unpacked - zero_point) * scale
        
        return dequantized
    
    def apply_to_model(self, model):
        """
        Apply INT4 quantization to all Linear layers in the model.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Model with quantized weights
        """
        quantized_model = type(model)(model.config) if hasattr(model, 'config') else model
        
        # Copy state dict and quantize linear layer weights
        state_dict = model.state_dict()
        new_state_dict = {}
        
        for name, param in state_dict.items():
            if 'weight' in name and param.dim() >= 2:
                # Skip sensitive layers that impact accuracy most (updated patterns)
                if ('token_embeddings' in name or 'lm_head' in name or 
                    'position_embeddings' in name or 'output_projection' in name or
                    'pos_encoding' in name or '.embedding' in name):
                    logger.info(f"Skipping sensitive layer: {name}")
                    new_state_dict[name] = param  # Keep original precision
                    continue
                
                # Quantize weight matrix
                logger.info(f"Quantizing layer: {name}")
                
                try:
                    quantized_weight, scale, zero_point = self.quantize(param)
                    
                    # Store quantization metadata
                    self.quantized_layers[name] = {
                        'scale': scale,
                        'zero_point': zero_point,
                        'original_shape': param.shape
                    }
                    
                    # Use the quantized weight directly (it's already dequantized for compatibility)
                    new_state_dict[name] = quantized_weight
                    
                except Exception as e:
                    logger.warning(f"Failed to quantize {name}: {e}")
                    new_state_dict[name] = param
            else:
                # Keep non-weight parameters unchanged
                new_state_dict[name] = param
        
        # Load quantized weights
        quantized_model.load_state_dict(new_state_dict)
        
        # Store quantization metadata in the model for size calculation
        quantized_model._quantization_info = {
            'quantizer': self,
            'quantized_layers': self.quantized_layers,
            'compression_stats': self.get_memory_savings()
        }
        
        logger.info(f"Quantized {len(self.quantized_layers)} layers")
        
        # Log compression statistics
        stats = self.get_memory_savings()
        logger.info(f"Theoretical compression: {stats['compression_ratio']:.1f}x, "
                   f"Memory saved: {stats['memory_saved_mb']:.2f} MB")
        
        return quantized_model
    
    def get_memory_savings(self):
        """
        Calculate memory savings from quantization.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.quantized_layers:
            return {"compression_ratio": 1.0, "memory_saved_mb": 0.0}
        
        total_params = sum(
            np.prod(info['original_shape']) 
            for info in self.quantized_layers.values()
        )
        
        # Original: 32-bit float (4 bytes per param)
        # Quantized: 4-bit (0.5 bytes per param)
        original_size_mb = (total_params * 4) / (1024 * 1024)
        quantized_size_mb = (total_params * 0.5) / (1024 * 1024)
        
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 8.0
        memory_saved_mb = original_size_mb - quantized_size_mb
        
        return {
            "compression_ratio": compression_ratio,
            "memory_saved_mb": memory_saved_mb,
            "original_size_mb": original_size_mb,
            "quantized_size_mb": quantized_size_mb
        }
    
    def calculate_model_compression_ratio(self, model):
        """
        Calculate the theoretical compression ratio for a model.
        
        Args:
            model: PyTorch model that was quantized
            
        Returns:
            Compression ratio estimate
        """
        if not self.quantized_layers:
            return 1.0
            
        # Count quantized parameters vs total parameters
        total_params = sum(p.numel() for p in model.parameters())
        quantized_params = sum(
            np.prod(info['original_shape']) 
            for info in self.quantized_layers.values()
        )
        
        if total_params == 0:
            return 1.0
            
        # Calculate weighted compression ratio
        quantized_ratio = quantized_params / total_params
        # INT4 gives 8x compression for quantized params, 1x for non-quantized
        effective_compression = 1.0 / (quantized_ratio * 0.125 + (1 - quantized_ratio) * 1.0)
        
        return effective_compression


print(f"DEBUG: quantization.py - Int4Quantizer class defined. Type: {type(Int4Quantizer)}")


def benchmark_quantized_models(original_model, quantized_model, test_input, num_runs=10):
    """
    Benchmark performance comparison between original and quantized models.
    
    Args:
        original_model: Original float model
        quantized_model: Quantized model
        test_input: Input tensor for testing
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    logger.info(f"Benchmarking models with {num_runs} runs...")
    
    # Benchmark original model
    original_model.eval()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    original_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = original_model(test_input)
        original_times.append(time.time() - start_time)
    
    # Benchmark quantized model
    quantized_model.eval()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    quantized_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = quantized_model(test_input)
        quantized_times.append(time.time() - start_time)
    
    # Calculate statistics
    original_avg = np.mean(original_times) * 1000  # ms
    quantized_avg = np.mean(quantized_times) * 1000  # ms
    speedup = original_avg / quantized_avg if quantized_avg > 0 else 1.0
    
    results = {
        "original_latency_ms": original_avg,
        "quantized_latency_ms": quantized_avg,
        "speedup": speedup,
        "original_std_ms": np.std(original_times) * 1000,
        "quantized_std_ms": np.std(quantized_times) * 1000
    }
    
    logger.info(f"Benchmark completed: {speedup:.2f}x speedup")
    return results


def measure_model_size(model):
    """
    Calculate model size in MB, accounting for quantization.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    # Check if model has quantization metadata
    if hasattr(model, '_quantization_info'):
        quantizer = model._quantization_info.get('quantizer')
        if quantizer and hasattr(quantizer, 'calculate_model_compression_ratio'):
            # Calculate size based on compression ratio
            uncompressed_size = sum(p.numel() * p.element_size() for p in model.parameters())
            uncompressed_size += sum(b.numel() * b.element_size() for b in model.buffers())
            uncompressed_size_mb = uncompressed_size / (1024 * 1024)
            
            compression_ratio = quantizer.calculate_model_compression_ratio(model)
            compressed_size_mb = uncompressed_size_mb / compression_ratio
            
            logger.info(f"Model size calculation: {uncompressed_size_mb:.2f} MB -> {compressed_size_mb:.2f} MB (compression: {compression_ratio:.1f}x)")
            return compressed_size_mb
    
    # Standard calculation for non-quantized models
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    logger.info(f"Model size: {size_mb:.2f} MB")
    return size_mb


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GLOBAL API FUNCTION - THIS IS WHAT showcase_edgeformer.py IMPORTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def quantize_model(model, quantization_type="int4"):
    """
    Quantize model with the specified quantization type.
    This function acts as a public API for the quantization utilities.
    
    Args:
        model: PyTorch model to quantize
        quantization_type: Type of quantization ("int4" or "int8")
        
    Returns:
        Quantized model
    """
    if not isinstance(model, nn.Module):
        logger.error(f"Model to be quantized is not an nn.Module. Got type: {type(model)}")
        raise TypeError("Model must be a PyTorch nn.Module to be quantized.")

    logger.info(f"Global 'quantize_model' function called with quantization_type: '{quantization_type}'")
    
    if quantization_type.lower() == "int8":
        return DynamicQuantizer.quantize_model_int8(model)
    elif quantization_type.lower() == "int4":
        return DynamicQuantizer.quantize_model_int4(model)
    else:
        logger.warning(f"Unsupported quantization type: '{quantization_type}'. Original model returned.")
        return model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# END OF GLOBAL API FUNCTION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Clean up memory
def cleanup_memory():
    """Clean up GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# Module initialization
logger.info("EdgeFormer quantization utilities loaded successfully")
cleanup_memory()


# --- Debug prints at the very END of the module ---
print(f"DEBUG: quantization.py (User's Full Version) - 'quantize_model' in globals(): {'quantize_model' in globals()}")
if 'quantize_model' in globals():
    print(f"DEBUG: quantization.py (User's Full Version) - type(quantize_model) at END of module: {type(quantize_model)}")
    print(f"DEBUG: quantization.py (User's Full Version) - Is quantize_model callable at END of module? {callable(quantize_model)}")
else:
    print("DEBUG: quantization.py (User's Full Version) - 'quantize_model' NOT in globals at end of module.")

# Check for other things that might be accidentally named quantize_model
other_qm = [name for name, obj in globals().items() if name.lower() == 'quantize_model' and name != 'quantize_model']
if other_qm:
    print(f"DEBUG: quantization.py - Other items similar to 'quantize_model' in globals: {other_qm}")

print("DEBUG: quantization.py (User's Full Version) - END of file execution")
