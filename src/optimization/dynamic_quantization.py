# src/optimization/dynamic_quantization.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import gc

logger = logging.getLogger(__name__)

class DynamicQuantizer:
    """Dynamic quantization for EdgeFormer models"""
    
    @staticmethod
    def quantize_model_int8(model):
        """Convert a model to INT8 dynamic quantization using PyTorch's built-in support"""
        logger.info("Applying INT8 dynamic quantization...")
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Quantize only linear layers
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def quantize_model_int4(model):
        """Efficient INT4 quantization implementation"""
        logger.info("Applying INT4 quantization...")
        
        # Create a quantizer object to store metadata
        quantizer = Int4Quantizer(model)
        
        # Perform the quantization
        quantizer.quantize()
        
        # Apply quantized weights to the model
        quantized_model = quantizer.apply_to_model()
        
        # Log memory savings
        savings = quantizer.get_memory_savings()
        logger.info(f"INT4 Quantization results:")
        logger.info(f"  Original size: {savings['original_size_mb']:.2f} MB")
        logger.info(f"  Quantized size: {savings['quantized_size_mb']:.2f} MB")
        logger.info(f"  Compression ratio: {savings['compression_ratio']:.2f}x")
        logger.info(f"  Size reduction: {savings['size_reduction_percent']:.2f}%")
        
        return quantized_model


class Int4Quantizer:
    """Simplified INT4 quantization for EdgeFormer models with shape preservation"""
    
    def __init__(self):
        """Initialize INT4 quantizer"""
        pass
    
    def pack_int4(self, values):
        """
        Pack two INT4 values per byte with shape preservation
        Args:
            values: torch.Tensor with INT4 values in range [-8, 7]
        Returns:
            dict: {
                'packed_data': torch.Tensor of packed bytes,
                'original_shape': tuple of original tensor shape,
                'original_dtype': original tensor dtype,
                'num_elements': total number of elements
            }
        """
        # Store original metadata
        original_shape = values.shape
        original_dtype = values.dtype
        num_elements = values.numel()
        
        # Flatten tensor for processing
        values_flat = values.flatten()
        
        # Ensure even number of elements (pad if necessary)
        if len(values_flat) % 2 != 0:
            values_flat = torch.cat([values_flat, torch.zeros(1, dtype=values_flat.dtype)])
        
        # Convert to INT4 range [-8, 7] to unsigned [0, 15]
        values_unsigned = values_flat + 8
        values_unsigned = torch.clamp(values_unsigned, 0, 15)
        
        # Pack two values per byte: (a << 4) | b
        packed_bytes = []
        for i in range(0, len(values_unsigned), 2):
            a = int(values_unsigned[i].item())
            b = int(values_unsigned[i + 1].item()) if i + 1 < len(values_unsigned) else 0
            packed_byte = (a << 4) | b
            packed_bytes.append(packed_byte)
        
        packed_data = torch.tensor(packed_bytes, dtype=torch.uint8)
        
        return {
            'packed_data': packed_data,
            'original_shape': original_shape,
            'original_dtype': original_dtype,
            'num_elements': num_elements
        }
    
    def unpack_int4(self, packed_data_dict):
        """
        Unpack INT4 values and restore original tensor shape
        Args:
            packed_data_dict: Dictionary from pack_int4() containing metadata
        Returns:
            torch.Tensor: Unpacked tensor with original shape
        """
        packed_data = packed_data_dict['packed_data']
        original_shape = packed_data_dict['original_shape']
        original_dtype = packed_data_dict['original_dtype']
        num_elements = packed_data_dict['num_elements']
        
        # Unpack bytes back to individual values
        unpacked_values = []
        for packed_byte in packed_data:
            byte_val = int(packed_byte.item())
            # Extract first value: (byte >> 4) & 0x0F
            a = (byte_val >> 4) & 0x0F
            # Extract second value: byte & 0x0F  
            b = byte_val & 0x0F
            unpacked_values.extend([a, b])
        
        # Convert back to signed INT4 range: [0, 15] -> [-8, 7]
        unpacked_tensor = torch.tensor(unpacked_values, dtype=torch.float32)
        unpacked_tensor = unpacked_tensor - 8
        
        # Trim to original number of elements (remove padding)
        original_numel = 1
        for dim in original_shape:
            original_numel *= dim
        unpacked_tensor = unpacked_tensor[:original_numel]
        
        # Restore original shape
        unpacked_tensor = unpacked_tensor.reshape(original_shape)
        
        # Convert to original dtype if needed
        if original_dtype != torch.float32:
            unpacked_tensor = unpacked_tensor.to(original_dtype)
        
        return unpacked_tensor
    
    def quantize(self, tensor):
        """
        Quantize tensor to INT4 with shape preservation
        Args:
            tensor: torch.Tensor to quantize
        Returns:
            dict: Quantized data with metadata for reconstruction
        """
        # Store original metadata
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Calculate scale and zero point for quantization
        tensor_min = torch.min(tensor)
        tensor_max = torch.max(tensor)
        
        # Map to INT4 range [-8, 7] (15 levels)
        range_val = tensor_max - tensor_min
        if range_val == 0:
            range_val = 1.0
        scale = range_val / 15.0
        zero_point = tensor_min
        
        # Avoid division by zero
        if scale == 0:
            scale = 1.0
        
        # Quantize: (tensor - zero_point) / scale
        quantized_values = torch.round((tensor - zero_point) / scale)
        quantized_values = torch.clamp(quantized_values, -8, 7)
        
        # Pack the quantized values
        packed_data = self.pack_int4(quantized_values)
        
        # Add quantization metadata
        quantization_metadata = {
            'scale': scale,
            'zero_point': zero_point,
            'original_shape': original_shape,
            'original_dtype': original_dtype
        }
        
        # Combine packed data with quantization metadata
        return {
            **packed_data,
            **quantization_metadata
        }
    
    def dequantize(self, quantized_data, target_shape=None):
        """
        Dequantize INT4 data back to original tensor
        Args:
            quantized_data: dict from quantize() method
            target_shape: optional shape override (for compatibility)
        Returns:
            torch.Tensor: Dequantized tensor with original shape and scale
        """
        # Extract quantization metadata
        scale = quantized_data['scale']
        zero_point = quantized_data['zero_point']
        original_shape = target_shape or quantized_data['original_shape']
        original_dtype = quantized_data['original_dtype']
        
        # Unpack INT4 values
        unpacked_values = self.unpack_int4(quantized_data)
        
        # Ensure correct shape (compatibility with target_shape parameter)
        if target_shape is not None and unpacked_values.shape != target_shape:
            # Handle shape mismatch by reshaping
            if unpacked_values.numel() == torch.prod(torch.tensor(target_shape)):
                unpacked_values = unpacked_values.reshape(target_shape)
            else:
                raise ValueError(f"Cannot reshape {unpacked_values.shape} to {target_shape}")
        
        # Dequantize: values * scale + zero_point
        dequantized = unpacked_values * scale + zero_point
        
        # Convert to original dtype
        dequantized = dequantized.to(original_dtype)
        
        return dequantized
    
    def get_compression_ratio(self, original_tensor, quantized_data):
        """Calculate compression ratio achieved"""
        original_bytes = original_tensor.numel() * 4  # float32 = 4 bytes
        compressed_bytes = quantized_data['packed_data'].numel() * 1  # uint8 = 1 byte
        return original_bytes / compressed_bytes


class DynamicQuantizer:
    """Main quantization interface that handles different quantization types"""
    
    def __init__(self, quantization_type="int8"):
        self.quantization_type = quantization_type
        
        if quantization_type == "int4":
            self.quantizer = Int4Quantizer()
        else:
            raise ValueError(f"Only INT4 quantization is currently supported, got: {quantization_type}")
    
    def quantize(self, tensor):
        """Quantize tensor using the specified quantization type"""
        return self.quantizer.quantize(tensor)
    
    def dequantize(self, quantized_data, target_shape=None):
        """Dequantize tensor using the specified quantization type"""
        return self.quantizer.dequantize(quantized_data, target_shape)
    
    def get_compression_ratio(self, original_tensor, quantized_data):
        """Get compression ratio"""
        if hasattr(self.quantizer, 'get_compression_ratio'):
            return self.quantizer.get_compression_ratio(original_tensor, quantized_data)
        else:
            # Fallback calculation
            original_bytes = original_tensor.numel() * 4
            if 'packed_data' in quantized_data:
                compressed_bytes = quantized_data['packed_data'].numel()
            else:
                compressed_bytes = quantized_data.numel()
            return original_bytes / compressed_bytes


def benchmark_quantized_models(original_model, test_input, test_runs=10):
    """Benchmark different quantization methods on the model"""
    results = {}
    
    # Original FP32 model
    logger.info("Benchmarking original FP32 model...")
    original_model.eval()
    with torch.no_grad():
        # Warmup
        original_output = original_model(**test_input)
        
        # Benchmark
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else torch.tensor(0)
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else torch.tensor(0)
        
        if torch.cuda.is_available():
            start_time.record()
            for _ in range(test_runs):
                original_model(**test_input)
            end_time.record()
            torch.cuda.synchronize()
            fp32_time = start_time.elapsed_time(end_time) / test_runs
        else:
            import time
            start = time.time()
            for _ in range(test_runs):
                original_model(**test_input)
            fp32_time = (time.time() - start) * 1000 / test_runs  # Convert to ms
    
    # Get model size
    fp32_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
    results['fp32'] = {
        'inference_time_ms': fp32_time,
        'memory_usage_bytes': fp32_size,
        'memory_usage_mb': fp32_size / (1024 * 1024),
        'output': original_output
    }
    
    # INT8 quantization
    logger.info("Benchmarking INT8 quantized model...")
    int8_model = DynamicQuantizer.quantize_model_int8(original_model)
    with torch.no_grad():
        # Warmup
        int8_output = int8_model(**test_input)
        
        # Benchmark
        if torch.cuda.is_available():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            for _ in range(test_runs):
                int8_model(**test_input)
            end_time.record()
            torch.cuda.synchronize()
            int8_time = start_time.elapsed_time(end_time) / test_runs
        else:
            import time
            start = time.time()
            for _ in range(test_runs):
                int8_model(**test_input)
            int8_time = (time.time() - start) * 1000 / test_runs  # Convert to ms
    
    # Calculate INT8 model size
    int8_size = 0
    for name, param in int8_model.state_dict().items():
        if hasattr(param, 'dtype') and param.dtype == torch.qint8:
            # INT8 parameters (1 byte per parameter)
            int8_size += param.numel()
        else:
            # Other parameters (typically FP32)
            int8_size += param.numel() * (4 if param.dtype == torch.float32 else param.element_size())
    
    # Calculate similarity between outputs
    if isinstance(original_output, dict) and isinstance(int8_output, dict):
        # Calculate MSE and cosine similarity for each output tensor
        mse_int8 = 0
        similarity_int8 = 0
        count = 0
        
        for key in original_output:
            if torch.is_tensor(original_output[key]) and torch.is_tensor(int8_output[key]):
                # MSE
                mse_int8 += ((original_output[key] - int8_output[key]) ** 2).mean().item()
                
                # Cosine similarity
                orig_flat = original_output[key].view(-1)
                int8_flat = int8_output[key].view(-1)
                
                if orig_flat.numel() > 0:
                    cos_sim = F.cosine_similarity(orig_flat.unsqueeze(0), int8_flat.unsqueeze(0)).item()
                    similarity_int8 += cos_sim
                    count += 1
        
        if count > 0:
            mse_int8 /= count
            similarity_int8 /= count
    else:
        # Single tensor outputs
        mse_int8 = ((original_output - int8_output) ** 2).mean().item()
        orig_flat = original_output.view(-1)
        int8_flat = int8_output.view(-1)
        similarity_int8 = F.cosine_similarity(orig_flat.unsqueeze(0), int8_flat.unsqueeze(0)).item()
    
    results['int8'] = {
        'inference_time_ms': int8_time,
        'memory_usage_bytes': int8_size,
        'memory_usage_mb': int8_size / (1024 * 1024),
        'output_mse': mse_int8,
        'output_similarity': similarity_int8 * 100,  # As percentage
        'memory_reduction': fp32_size / int8_size,
        'speed_impact_percent': (fp32_time - int8_time) / fp32_time * 100
    }
    
    # INT4 quantization
    logger.info("Benchmarking INT4 quantized model...")
    int4_model = DynamicQuantizer.quantize_model_int4(original_model)
    with torch.no_grad():
        # Warmup
        int4_output = int4_model(**test_input)
        
        # Benchmark
        if torch.cuda.is_available():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            for _ in range(test_runs):
                int4_model(**test_input)
            end_time.record()
            torch.cuda.synchronize()
            int4_time = start_time.elapsed_time(end_time) / test_runs
        else:
            import time
            start = time.time()
            for _ in range(test_runs):
                int4_model(**test_input)
            int4_time = (time.time() - start) * 1000 / test_runs  # Convert to ms
    
    # Get int4 quantizer to calculate memory usage
    int4_quantizer = next((obj for obj in gc.get_objects() 
                          if isinstance(obj, Int4Quantizer) 
                          and hasattr(obj, 'model') 
                          and obj.model is original_model), None)
    
    if int4_quantizer:
        savings = int4_quantizer.get_memory_savings()
        int4_size = savings['quantized_size_mb'] * (1024 * 1024)  # Convert MB back to bytes
    else:
        # Approximation if quantizer not found
        int4_size = fp32_size / 8  # Approximate 8x reduction
    
    # Calculate similarity between outputs
    if isinstance(original_output, dict) and isinstance(int4_output, dict):
        # Calculate MSE and cosine similarity for each output tensor
        mse_int4 = 0
        similarity_int4 = 0
        count = 0
        
        for key in original_output:
            if torch.is_tensor(original_output[key]) and torch.is_tensor(int4_output[key]):
                # MSE
                mse_int4 += ((original_output[key] - int4_output[key]) ** 2).mean().item()
                
                # Cosine similarity
                orig_flat = original_output[key].view(-1)
                int4_flat = int4_output[key].view(-1)
                
                if orig_flat.numel() > 0:
                    cos_sim = F.cosine_similarity(orig_flat.unsqueeze(0), int4_flat.unsqueeze(0)).item()
                    similarity_int4 += cos_sim
                    count += 1
        
        if count > 0:
            mse_int4 /= count
            similarity_int4 /= count
    else:
        # Single tensor outputs
        mse_int4 = ((original_output - int4_output) ** 2).mean().item()
        orig_flat = original_output.view(-1)
        int4_flat = int4_output.view(-1)
        similarity_int4 = F.cosine_similarity(orig_flat.unsqueeze(0), int4_flat.unsqueeze(0)).item()
    
    results['int4'] = {
        'inference_time_ms': int4_time,
        'memory_usage_bytes': int4_size,
        'memory_usage_mb': int4_size / (1024 * 1024),
        'output_mse': mse_int4,
        'output_similarity': similarity_int4 * 100,  # As percentage
        'memory_reduction': fp32_size / int4_size,
        'speed_impact_percent': (fp32_time - int4_time) / fp32_time * 100
    }
    
    return results


def measure_model_size(model):
    """Measure model size in MB accounting for quantized models"""
    # Check if this is an INT4 quantized model by looking for Int4Quantizer attributes
    is_int4 = False
    for module in model.modules():
        if hasattr(module, 'forward') and 'Int4Quantizer' in str(module.forward):
            is_int4 = True
            break
    
    if is_int4:
        # Get the quantizer from gc
        quantizer = None
        for obj in gc.get_objects():
            if isinstance(obj, Int4Quantizer) and hasattr(obj, 'model') and obj.model is model:
                quantizer = obj
                break
        
        if quantizer and hasattr(quantizer, 'get_memory_savings'):
            savings = quantizer.get_memory_savings()
            return savings['quantized_size_mb']
    
    # Regular size calculation for non-INT4 models
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def quantize_model(model, quantization_type="int8"):
    """Quantize model with specified quantization type"""
    if quantization_type.lower() == "int8":
        return DynamicQuantizer.quantize_model_int8(model)
    elif quantization_type.lower() == "int4":
        return DynamicQuantizer.quantize_model_int4(model)
    else:
        logger.warning(f"Unsupported quantization type: {quantization_type}, using INT8 instead")
        return DynamicQuantizer.quantize_model_int8(model)