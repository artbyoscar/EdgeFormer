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
    """INT4 quantization for EdgeFormer models"""
    
    def __init__(self, model):
        self.model = model
        self.original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        self.quantized_state_dict = {}
        self.scale_factors = {}
        self.original_shapes = {}
        
    def quantize(self):
        """Quantize model weights to INT4"""
        state_dict = self.model.state_dict()
        
        for name, param in state_dict.items():
            # Only quantize linear layer weights
            if 'weight' in name and param.dim() == 2:
                # Save original shape
                self.original_shapes[name] = param.shape
                
                # Convert to numpy for easier manipulation
                tensor = param.detach().cpu().numpy()
                
                # Calculate scale factor (max / 7.0 for INT4 symmetric quantization)
                abs_max = np.abs(tensor).max()
                scale = abs_max / 7.0
                
                # Quantize to INT4 range (-8 to 7)
                quantized = np.clip(np.round(tensor / scale), -8, 7).astype(np.int8)
                
                # Pack two INT4 values into one INT8 value
                quantized_packed = self._pack_int4(quantized)
                
                # Store quantized weights and scale factors
                self.quantized_state_dict[name] = quantized_packed
                self.scale_factors[name] = scale
                
                logger.info(f"Quantized {name}: original shape {param.shape}, "
                           f"packed shape {quantized_packed.shape}, "
                           f"scale {scale}")
            else:
                # Keep non-weight tensors as is
                self.quantized_state_dict[name] = param
        
        return self
    
    def _pack_int4(self, tensor):
        """Pack two INT4 values into one INT8"""
        # Ensure even number of elements in last dimension
        original_shape = tensor.shape
        tensor = tensor.reshape(-1)
        
        # Pad with zeros if necessary
        padding_needed = (tensor.size % 2) != 0
        if padding_needed:
            tensor = np.pad(tensor, (0, 1), 'constant')
        
        # Reshape for packing
        tensor = tensor.reshape(-1, 2)
        
        # Pack two INT4 into one INT8 (first << 4 | second & 0xF)
        # The first INT4 value gets the high 4 bits, the second gets the low 4 bits
        packed = ((tensor[:, 0] & 0xF) << 4) | (tensor[:, 1] & 0xF)
        
        # Reshape back
        packed_shape = list(original_shape)
        packed_shape[-1] = packed_shape[-1] // 2 + (1 if padding_needed else 0)
        
        return torch.tensor(packed.reshape(packed_shape), dtype=torch.int8)
    
    def _unpack_int4(self, packed_tensor, original_shape):
        """Unpack INT8 tensor back to two INT4 values"""
        # Convert to numpy
        packed = packed_tensor.cpu().numpy().reshape(-1)
    
        # Calculate total elements needed in the original shape
        total_elements = int(np.prod(original_shape))
    
        # Unpack INT8 to two INT4 values
        # Create just enough space for the elements we need
        unpacked = np.zeros(total_elements, dtype=np.int8)

        # Calculate how many packed elements we need to process
        # We might need to handle fewer elements if the original shape isn't even
        packed_needed = (total_elements + 1) // 2
        packed_to_process = min(len(packed), packed_needed)

        # Process complete pairs (2 values per packed byte)
        for i in range(packed_to_process):
            if 2*i < total_elements:
                unpacked[2*i] = (packed[i] >> 4) & 0xF
            if 2*i + 1 < total_elements:
                unpacked[2*i + 1] = packed[i] & 0xF
    
        # Handle negative values (sign extension for 4-bit)
        # Values 8-15 represent negative numbers (-8 to -1)
        neg_mask = unpacked > 7
        unpacked[neg_mask] = unpacked[neg_mask] - 16
    
        # Fix for shape mismatch issue:
        # Make sure unpacked has exactly the same shape as original_shape
        if np.prod(unpacked.shape) != np.prod(original_shape):
            if np.prod(unpacked.shape) > np.prod(original_shape):
                # Too many elements, truncate
                unpacked = unpacked.flatten()[:np.prod(original_shape)].reshape(original_shape)
            else:
                # Too few elements, pad with zeros
                result = np.zeros(original_shape, dtype=np.int8)
                flat_unpacked = unpacked.flatten()
                flat_result = result.flatten()
                flat_result[:len(flat_unpacked)] = flat_unpacked
                unpacked = result
    
    # Reshape to original shape
    return unpacked.reshape(original_shape)
    
    def dequantize(self, name, quantized_tensor, original_shape, device=None):
        """Dequantize a packed INT4 tensor"""
        if name not in self.scale_factors:
            return quantized_tensor
    
        scale = self.scale_factors[name]
    
        # Unpack and convert back to float
        unpacked = self._unpack_int4(quantized_tensor, original_shape)
        dequantized = unpacked * scale
    
        # Create tensor with proper device and shape
        tensor = torch.tensor(dequantized, dtype=torch.float32)
    
        # Ensure tensor has the exact shape we need
        if tensor.shape != original_shape:
            tensor = tensor.reshape(original_shape)
        
        if device is not None:
            tensor = tensor.to(device)
    
        return tensor
    
    def apply_to_model(self, target_model=None):
        """Apply quantized weights to the model using hooks"""
        if target_model is None:
            target_model = self.model
        
        # Keep track of which modules we've modified
        modified_modules = set()
        
        # Create hooks for dequantization during forward pass
        for name, module in target_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight_name = f"{name}.weight"
                if weight_name in self.quantized_state_dict and weight_name in self.scale_factors:
                    # Skip if we've already modified this module
                    if module in modified_modules:
                        continue
                    
                    # Save original forward and parameters
                    original_forward = module.forward
                    quantized_weight = self.quantized_state_dict[weight_name]
                    original_shape = self.original_shapes[weight_name]
                    
                    # Define the new forward method
                    def new_forward(self_module, input, _orig_forward=original_forward, 
                                    _weight_name=weight_name, _quantized=quantized_weight,
                                    _orig_shape=original_shape, _self=self):
                        try:
                            # Dequantize weight on the fly
                            dequantized = _self.dequantize(_weight_name, _quantized, _orig_shape, input.device)
        
                            # Save original weight
                            orig_weight = self_module.weight.data.clone()

                            # Make sure shapes match exactly - very important fix!
                            if dequantized.shape != orig_weight.shape:
                                logger.warning(f"Shape mismatch in INT4 dequantization: got {dequantized.shape}, expected {orig_weight.shape}")
            
                                # Try to reshape directly if possible
                                if dequantized.numel() == orig_weight.numel():
                                    dequantized = dequantized.reshape(orig_weight.shape)
                                # Otherwise handle mismatches more carefully
                                else:
                                    # Create a tensor matching the original weight
                                    temp = torch.zeros_like(orig_weight)
                
                                    # Get the min common sizes for each dimension
                                    common_size0 = min(dequantized.shape[0], orig_weight.shape[0])
                                    common_size1 = min(dequantized.shape[1], orig_weight.shape[1])
                
                                    # Copy the data we can
                                    temp[:common_size0, :common_size1] = dequantized[:common_size0, :common_size1]
                                    dequantized = temp
        
                            # Replace weight temporarily
                            self_module.weight.data = dequantized
        
                            # Run forward
                            result = _orig_forward(input)
        
                            # Restore original weight
                            self_module.weight.data = orig_weight
        
                            return result
                        except Exception as e:
                            # Log the error for debugging
                            logger.error(f"Error in INT4 forward pass: {str(e)}")
                            logger.error(f"Quantized shape: {_quantized.shape}, Original shape: {_orig_shape}")
                            # Fallback to original weights
                            return _orig_forward(input)
                    
                    # Create a bound method
                    bound_forward = lambda x, mod=module: new_forward(mod, x)
                    
                    # Replace the forward method
                    module.forward = bound_forward
                    
                    # Mark this module as modified
                    modified_modules.add(module)
        
        logger.info(f"Applied INT4 quantization to {len(modified_modules)} modules")
        return target_model
    
    def get_memory_savings(self):
        """Calculate memory savings from quantization"""
        original_bytes = 0
        quantized_bytes = 0
        
        for name, param in self.original_state_dict.items():
            if name in self.quantized_state_dict:
                # Original size (FP32 = 4 bytes per parameter)
                original_bytes += param.numel() * 4
                
                # Quantized size
                if 'weight' in name and param.dim() == 2 and name in self.scale_factors:
                    # Packed INT4 weights (4 bits = 0.5 bytes per parameter)
                    # Since we pack two INT4 values into one INT8, the size is half
                    quantized_param = self.quantized_state_dict[name]
                    quantized_bytes += quantized_param.numel() * 1  # INT8 = 1 byte per value
                    
                    # Scale factor (1 FP32 value per tensor)
                    quantized_bytes += 4
                else:
                    # Non-quantized parameters remain in full precision
                    quantized_bytes += param.numel() * 4
        
        return {
            "original_size_mb": original_bytes / (1024 * 1024),
            "quantized_size_mb": quantized_bytes / (1024 * 1024),
            "compression_ratio": original_bytes / quantized_bytes if quantized_bytes > 0 else 0,
            "size_reduction_percent": (1 - quantized_bytes / original_bytes) * 100 if original_bytes > 0 else 0
        }


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