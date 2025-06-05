# src/optimization/quantization.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class QuantizationConfig:
    """Configuration for model quantization."""
    
    def __init__(
        self,
        bits=8,  # 8 or 4
        quantize_weights=True,
        quantize_activations=False,
        sym=True,  # Symmetric quantization
        per_channel=True,  # Per-channel or per-tensor
    ):
        self.bits = bits
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.sym = sym
        self.per_channel = per_channel
        
        if self.bits not in [8, 4]:
            logger.warning(f"Unsupported bits: {bits}, using 8 bits")
            self.bits = 8

def quantize_tensor(tensor, bits=8, sym=True, per_channel=True):
    """
    Quantize a tensor to INT8 or INT4.
    
    Args:
        tensor: Tensor to quantize
        bits: Number of bits (8 or 4)
        sym: Whether to use symmetric quantization
        per_channel: Whether to use per-channel scaling
        
    Returns:
        quantized_tensor: Quantized tensor
        scale: Scale factor
        zero_point: Zero point (for asymmetric quantization)
    """
    # Determine range based on bits
    if bits == 8:
        qmin, qmax = (-128, 127) if sym else (0, 255)
    elif bits == 4:
        qmin, qmax = (-8, 7) if sym else (0, 15)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")
    
    # Handle per-channel quantization
    if per_channel and tensor.dim() > 1:
        # Quantize along the first dimension (out_channels for weights)
        dim = 0
        
        # Compute min/max per channel
        tensor_np = tensor.detach().cpu()
        if sym:
            # For symmetric quantization, we take the max absolute value
            absmax = torch.max(torch.abs(tensor_np), dim=dim, keepdim=True)[0]
            scale = absmax / (float(qmax) - qmin) / 2
            zero_point = torch.zeros_like(scale, dtype=torch.int)
        else:
            # For asymmetric quantization
            tmin = torch.min(tensor_np, dim=dim, keepdim=True)[0]
            tmax = torch.max(tensor_np, dim=dim, keepdim=True)[0]
            scale = (tmax - tmin) / float(qmax - qmin)
            zero_point = torch.round(-tmin / scale).to(torch.int)
            
        # Ensure zero_point is within range
        zero_point = torch.clamp(zero_point, qmin, qmax)
        
        # Quantize
        q_tensor = torch.round(tensor / scale) + zero_point
        q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int)
        
        # Dequantize for simulation
        tensor_dq = (q_tensor - zero_point) * scale
        
    else:
        # Per-tensor quantization
        tensor_np = tensor.detach().cpu()
        if sym:
            # Symmetric quantization
            absmax = torch.max(torch.abs(tensor_np))
            scale = absmax / (float(qmax) - qmin) / 2
            zero_point = 0
        else:
            # Asymmetric quantization
            tmin = torch.min(tensor_np)
            tmax = torch.max(tensor_np)
            scale = (tmax - tmin) / float(qmax - qmin)
            zero_point = torch.round(-tmin / scale).to(torch.int).item()
            
        # Ensure zero_point is within range
        zero_point = max(qmin, min(qmax, zero_point))
        
        # Quantize
        q_tensor = torch.round(tensor / scale) + zero_point
        q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int)
        
        # Dequantize for simulation
        tensor_dq = (q_tensor - zero_point) * scale
    
    return tensor_dq, scale, zero_point

class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights."""
    
    def __init__(self, in_features, out_features, bits=8, sym=True, per_channel=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.sym = sym
        self.per_channel = per_channel
        
        # Create a standard linear layer as reference
        self.ref_linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Quantization parameters
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        self.register_buffer('quantized_weight', None)
        
        # Flag to track quantization state
        self.quantized = False
    
    def quantize_weights(self):
        """Quantize the weights of this layer."""
        if self.quantized:
            return
        
        # Quantize weights
        weight_dq, weight_scale, weight_zero_point = quantize_tensor(
            self.ref_linear.weight, 
            bits=self.bits, 
            sym=self.sym, 
            per_channel=self.per_channel
        )
        
        # Store quantization parameters
        self.weight_scale = weight_scale
        self.weight_zero_point = weight_zero_point
        
        # Replace weights with quantized version
        self.ref_linear.weight.data = weight_dq
        
        self.quantized = True
        
        logger.debug(f"Quantized Linear layer: {self.in_features} -> {self.out_features} ({self.bits}-bit)")
    
    def forward(self, x):
        """Forward pass using quantized weights."""
        # Quantize weights if not already done
        if not self.quantized:
            self.quantize_weights()
            
        # Regular forward pass with quantized weights
        return self.ref_linear(x)

class Quantizer:
    """Utility to quantize EdgeFormer models."""
    
    def __init__(self, config):
        """
        Initialize the quantizer.
        
        Args:
            config: QuantizationConfig instance
        """
        self.config = config
        
    def quantize_model(self, model):
        """
        Quantize a model in-place.
        
        Args:
            model: EdgeFormer model to quantize
            
        Returns:
            model: Quantized model
        """
        logger.info(f"Quantizing model to {self.config.bits}-bit precision...")
        
        # Track number of layers quantized
        quantized_layers = 0
        
        # Create a copy of model
        model_copy = model
        
        # Replace all linear layers with quantized versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                
                # Get parent module
                parent = model if parent_name == '' else getattr(model, parent_name)
                
                # Create quantized linear layer
                quantized_linear = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bits=self.config.bits,
                    sym=self.config.sym,
                    per_channel=self.config.per_channel,
                    bias=(module.bias is not None)
                )
                
                # Copy weights and bias
                quantized_linear.ref_linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    quantized_linear.ref_linear.bias.data.copy_(module.bias.data)
                
                # Quantize weights
                quantized_linear.quantize_weights()
                
                # Replace the module
                setattr(parent, child_name, quantized_linear)
                
                quantized_layers += 1
        
        logger.info(f"Quantized {quantized_layers} linear layers to {self.config.bits}-bit precision")
        
        return model

def quantize_edgeformer(model, bits=8, sym=True, per_channel=True):
    """
    Convenience function to quantize an EdgeFormer model.
    
    Args:
        model: EdgeFormer model
        bits: Number of bits (8 or 4)
        sym: Whether to use symmetric quantization
        per_channel: Whether to use per-channel scaling
        
    Returns:
        model: Quantized model
    """
    # Import here to avoid circular imports
    from .dynamic_quantization import DynamicQuantizer
    
    if bits == 8:
        return DynamicQuantizer.quantize_model_int8(model)
    elif bits == 4:
        return DynamicQuantizer.quantize_model_int4(model)
    else:
        logger.warning(f"Unsupported bit width: {bits}, using 8 bits")
        return DynamicQuantizer.quantize_model_int8(model)
