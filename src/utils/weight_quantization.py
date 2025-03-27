# src/utils/weight_quantization.py

import torch
import numpy as np
import copy
import logging
import os
import uuid
import time
import tempfile
import matplotlib.pyplot as plt

logger = logging.getLogger("edgeformer")

class WeightOnlyQuantizedLinear(torch.nn.Module):
    """
    Linear layer with weight-only quantization.
    """
    def __init__(self, orig_layer, weight, scales, zeros, bits=4, group_size=128, symmetric=True):
        super().__init__()
        self.in_features = orig_layer.in_features
        self.out_features = orig_layer.out_features
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        
        # Register parameters
        self.register_buffer('weight', weight)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        
        # Keep bias
        if hasattr(orig_layer, 'bias') and orig_layer.bias is not None:
            self.register_parameter('bias', torch.nn.Parameter(orig_layer.bias.data))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        # Get dimensions
        output_dim, num_groups, group_size = self.weight.shape
        
        # Process input
        input_flat = input.reshape(-1, input.shape[-1])
        
        # Initialize output
        output = torch.zeros(input_flat.shape[0], output_dim, 
                            dtype=torch.float32, device=input.device)
        
        # Process group by group to save memory
        effective_groups = min(num_groups, (self.in_features + self.group_size - 1) // self.group_size)
        
        for g in range(effective_groups):
            # Calculate start and end indices
            start_idx = g * self.group_size
            end_idx = min(start_idx + self.group_size, self.in_features)
            
            if end_idx <= start_idx:
                continue
            
            # Get input slice
            input_slice = input_flat[:, start_idx:end_idx]
            
            # Skip if the input slice is empty
            if input_slice.shape[1] == 0:
                continue
            
            # Get weight slice
            if self.symmetric:
                # Dequantize symmetrically
                weight_slice = self.weight[:, g, :end_idx-start_idx].float() * self.scales[:, g].unsqueeze(1)
            else:
                # Dequantize asymmetrically
                weight_slice = (self.weight[:, g, :end_idx-start_idx].float() - self.zeros[:, g].unsqueeze(1)) * self.scales[:, g].unsqueeze(1)
            
            # Compute partial output
            output += torch.matmul(input_slice, weight_slice.t())
        
        # Add bias if exists
        if self.bias is not None:
            output += self.bias
        
        # Reshape output to match input shape
        output = output.reshape(input.shape[:-1] + (output_dim,))
        
        return output

def quantize_weight_only(module, bits=4, group_size=128, symmetric=True):
    """
    Quantize only the weights of a linear layer.
    
    Args:
        module: The linear layer to quantize
        bits: Bit width for quantization (4 or 8)
        group_size: Size of quantization groups
        symmetric: Whether to use symmetric quantization
        
    Returns:
        A weight-only quantized linear layer
    """
    if not isinstance(module, torch.nn.Linear):
        return module
    
    # Get original weights
    weight = module.weight.data
    
    # Store original dtype
    orig_dtype = weight.dtype
    
    # Move to float32 for quantization
    weight = weight.float()
    
    # Get dimensions
    output_dim, input_dim = weight.shape
    
    # Compute number of groups
    num_groups = (input_dim + group_size - 1) // group_size
    
    # Reshape weight for group quantization
    # Pad if needed
    padded_input_dim = num_groups * group_size
    if padded_input_dim != input_dim:
        padded_weight = torch.zeros(output_dim, padded_input_dim, device=weight.device)
        padded_weight[:, :input_dim] = weight
        weight = padded_weight
    
    # Reshape to [output_dim, num_groups, group_size]
    grouped_weight = weight.reshape(output_dim, num_groups, group_size)
    
    # Initialize quantized weights and scales
    if bits == 4:
        quant_min, quant_max = -8, 7  # 4-bit signed integer range
    elif bits == 8:
        quant_min, quant_max = -128, 127  # 8-bit signed integer range
    else:
        raise ValueError(f"Unsupported bits: {bits}, only 4 and 8 bits are supported")
    
    # Initialize tensors
    quantized_weight = torch.zeros_like(grouped_weight, dtype=torch.int8)
    scales = torch.zeros(output_dim, num_groups, dtype=torch.float32, device=weight.device)
    zeros = torch.zeros(output_dim, num_groups, dtype=torch.int8, device=weight.device)
    
    # Quantize each group
    for g in range(num_groups):
        # Get group
        group = grouped_weight[:, g, :]
        
        if symmetric:
            # Symmetric quantization (zero point is always 0)
            max_abs = torch.max(torch.abs(group), dim=1)[0]
            scales[:, g] = max_abs / quant_max
            
            # Set small scales to 1.0 to avoid div by zero
            scales[:, g][scales[:, g] < 1e-10] = 1.0
            
            # Quantize
            quantized_group = torch.round(group / scales[:, g].unsqueeze(1)).clamp(quant_min, quant_max)
            quantized_weight[:, g, :] = quantized_group.to(torch.int8)
        else:
            # Asymmetric quantization
            max_val = torch.max(group, dim=1)[0]
            min_val = torch.min(group, dim=1)[0]
            scales[:, g] = (max_val - min_val) / (quant_max - quant_min)
            
            # Set small scales to 1.0 to avoid div by zero
            scales[:, g][scales[:, g] < 1e-10] = 1.0
            
            # Calculate zero point
            zeros[:, g] = torch.round(quant_min - min_val / scales[:, g]).clamp(quant_min, quant_max).to(torch.int8)
            
            # Quantize
            quantized_group = torch.round(group / scales[:, g].unsqueeze(1) + zeros[:, g].unsqueeze(1).float()).clamp(quant_min, quant_max)
            quantized_weight[:, g, :] = quantized_group.to(torch.int8)
    
    # Create quantized module
    quantized_module = WeightOnlyQuantizedLinear(
        module,
        quantized_weight,
        scales,
        zeros,
        bits,
        group_size,
        symmetric
    )
    
    return quantized_module

def weight_only_quantize_model(model, bits=4, group_size=128, symmetric=True, modules_to_exclude=None):
    """
    Apply weight-only quantization to a model's linear layers.
    
    Args:
        model: The model to quantize
        bits: Bit width for quantization (4 or 8)
        group_size: Size of quantization groups
        symmetric: Whether to use symmetric quantization
        modules_to_exclude: List of module names to exclude from quantization
        
    Returns:
        A weight-only quantized model
    """
    if modules_to_exclude is None:
        modules_to_exclude = []
    
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Track quantized parameters
    total_params = 0
    quantized_params = 0
    
    # Get all modules
    for name, module in list(model_copy.named_modules()):
        # Skip non-linear layers and excluded modules
        if not isinstance(module, torch.nn.Linear) or name in modules_to_exclude:
            continue
        
        # Count parameters
        param_count = module.weight.numel()
        total_params += param_count
        quantized_params += param_count
        
        # Get parent module
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        child_name = name.rsplit('.', 1)[1] if '.' in name else name
        
        if parent_name:
            parent = model_copy.get_submodule(parent_name)
        else:
            parent = model_copy
        
        # Quantize linear layer
        logger.info(f"Quantizing layer: {name} ({param_count} parameters)")
        quantized_module = quantize_weight_only(
            module, 
            bits=bits, 
            group_size=group_size, 
            symmetric=symmetric
        )
        
        # Replace original module
        setattr(parent, child_name, quantized_module)
    
    # Log statistics
    logger.info(f"Quantized {quantized_params:,} out of {total_params:,} parameters ({quantized_params/total_params*100:.1f}%)")
    
    # Calculate memory savings
    if bits == 4:
        compression_ratio = 32 / 4  # FP32 to INT4
    elif bits == 8:
        compression_ratio = 32 / 8  # FP32 to INT8
    else:
        compression_ratio = 1
    
    # Account for scales and zeros overhead
    params_per_group = group_size
    scales_overhead = 32 / (params_per_group * bits)  # One FP32 scale per group
    zeros_overhead = 8 / (params_per_group * bits) if not symmetric else 0  # One INT8 zero per group if asymmetric
    
    effective_compression = compression_ratio / (1 + scales_overhead + zeros_overhead)
    
    logger.info(f"Theoretical compression ratio: {compression_ratio:.2f}x")
    logger.info(f"Effective compression ratio (with overhead): {effective_compression:.2f}x")
    
    return model_copy

def test_weight_only_quantization(model, inputs, bits=4, group_size=128, symmetric=True):
    """
    Test weight-only quantization by comparing outputs with the original model.
    
    Args:
        model: The original model
        inputs: Input tensors
        bits: Bit width for quantization (4 or 8)
        group_size: Size of quantization groups
        symmetric: Whether to use symmetric quantization
        
    Returns:
        Dict with comparison metrics
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Get outputs from original model
    with torch.no_grad():
        original_outputs = model(**inputs)
    
    # Quantize model
    quantized_model = weight_only_quantize_model(
        model,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric
    )
    
    # Get outputs from quantized model
    with torch.no_grad():
        quantized_outputs = quantized_model(**inputs)
    
    # Compare logits
    original_logits = original_outputs["logits"]
    quantized_logits = quantized_outputs["logits"]
    
    # Calculate metrics
    mse = torch.mean((original_logits - quantized_logits) ** 2).item()
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        original_logits.reshape(-1), 
        quantized_logits.reshape(-1), 
        dim=0
    ).item() * 100  # Convert to percentage
    
    # Calculate memory usage
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = 0
    
    for name, p in quantized_model.named_parameters():
        quantized_size += p.numel() * p.element_size()
    
    for name, b in quantized_model.named_buffers():
        if 'weight' in name and isinstance(b, torch.Tensor) and b.dtype == torch.int8:
            # For INT4, we're storing in INT8 but only using 4 bits
            if bits == 4:
                quantized_size += b.numel() * 0.5  # 4 bits = 0.5 bytes
            else:
                quantized_size += b.numel() * 1  # 8 bits = 1 byte
        else:
            quantized_size += b.numel() * b.element_size()
    
    size_reduction = original_size / quantized_size
    
    # Calculate inference speed
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(**inputs)
            quantized_model(**inputs)
    
    # Benchmark original model
    start_time = time.time()
    for _ in range(50):
        with torch.no_grad():
            model(**inputs)
    original_time = (time.time() - start_time) / 50
    
    # Benchmark quantized model
    start_time = time.time()
    for _ in range(50):
        with torch.no_grad():
            quantized_model(**inputs)
    quantized_time = (time.time() - start_time) / 50
    
    # Calculate speedup
    speedup = original_time / quantized_time
    
    # Log results
    logger.info(f"Weight-only quantization results ({bits}-bit, group size {group_size}, symmetric={symmetric}):")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  Similarity: {similarity:.2f}%")
    logger.info(f"  Size reduction: {size_reduction:.2f}x")
    logger.info(f"  Inference speedup: {speedup:.2f}x")
    
    return {
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
        "mse": mse,
        "similarity": similarity,
        "size_reduction": size_reduction,
        "original_time": original_time,
        "quantized_time": quantized_time,
        "speedup": speedup,
        "original_model": model,
        "quantized_model": quantized_model
    }

def compare_quantization_configs(model, inputs, configs=None, output_dir=None):
    """
    Compare different quantization configurations and visualize results.
    
    Args:
        model: The original model
        inputs: Input tensors
        configs: List of quantization configs to test
        output_dir: Directory to save results
        
    Returns:
        Dict with comparison results
    """
    if configs is None:
        configs = [
            {"bits": 8, "group_size": 128, "symmetric": True},
            {"bits": 8, "group_size": 128, "symmetric": False},
            {"bits": 4, "group_size": 128, "symmetric": True},
            {"bits": 4, "group_size": 128, "symmetric": False},
            {"bits": 4, "group_size": 64, "symmetric": True},
            {"bits": 4, "group_size": 32, "symmetric": True},
        ]
    
    if output_dir is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"quantization_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for config in configs:
        bits = config["bits"]
        group_size = config["group_size"]
        symmetric = config["symmetric"]
        
        config_name = f"{bits}b_{group_size}g_{'sym' if symmetric else 'asym'}"
        logger.info(f"Testing configuration: {config_name}")
        
        result = test_weight_only_quantization(
            model, 
            inputs, 
            bits=bits, 
            group_size=group_size, 
            symmetric=symmetric
        )
        
        result["config_name"] = config_name
        results.append(result)
    
    # Sort results by similarity (best first)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Save results table
    with open(os.path.join(output_dir, "quantization_comparison.txt"), "w") as f:
        f.write("Weight-Only Quantization Comparison\n")
        f.write("==================================\n\n")
        
        f.write(f"{'Configuration':<15} | {'Size Reduction':<15} | {'Similarity':<10} | {'MSE':<10} | {'Speedup':<8}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            f.write(f"{result['config_name']:<15} | {result['size_reduction']:<15.2f}x | {result['similarity']:<10.2f}% | {result['mse']:<10.6f} | {result['speedup']:<8.2f}x\n")
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Size reduction vs quality
    plt.subplot(2, 2, 1)
    config_names = [r["config_name"] for r in results]
    size_reductions = [r["size_reduction"] for r in results]
    similarities = [r["similarity"] for r in results]
    
    plt.scatter(size_reductions, similarities, s=100)
    for i, config in enumerate(config_names):
        plt.annotate(config, (size_reductions[i], similarities[i]), 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Size Reduction Factor')
    plt.ylabel('Output Similarity (%)')
    plt.title('Quality vs Compression Trade-off')
    plt.grid(True)
    
    # Speedup vs quality
    plt.subplot(2, 2, 2)
    speedups = [r["speedup"] for r in results]
    
    plt.scatter(speedups, similarities, s=100)
    for i, config in enumerate(config_names):
        plt.annotate(config, (speedups[i], similarities[i]), 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Inference Speedup Factor')
    plt.ylabel('Output Similarity (%)')
    plt.title('Quality vs Speed Trade-off')
    plt.grid(True)
    
    # Size reduction comparison
    plt.subplot(2, 2, 3)
    plt.bar(config_names, size_reductions)
    plt.ylabel('Size Reduction Factor')
    plt.title('Model Size Reduction by Configuration')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Similarity comparison
    plt.subplot(2, 2, 4)
    plt.bar(config_names, similarities)
    plt.ylabel('Output Similarity (%)')
    plt.title('Output Quality by Configuration')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantization_comparison.png"))
    
    logger.info(f"Quantization comparison results saved to {output_dir}")
    
    return {
        "results": results,
        "output_dir": output_dir,
        "best_config": results[0]["config_name"]
    }

def kv_cache_offload(model, offload_path=None, kv_cache_dtype=torch.float16):
    """
    Enable KV cache offloading to disk for the given model.
    
    Args:
        model: The EdgeFormer model
        offload_path: Directory to store KV cache files (default: temporary directory)
        kv_cache_dtype: Data type for stored KV cache (default: float16 for memory efficiency)
        
    Returns:
        Modified model with KV cache offloading enabled
    """
    if offload_path is None:
        offload_path = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for KV cache: {offload_path}")
    else:
        os.makedirs(offload_path, exist_ok=True)
        logger.info(f"Using directory for KV cache: {offload_path}")

    
    # Add offload path to model
    model.kv_cache_offload_path = offload_path
    model.kv_cache_dtype = kv_cache_dtype
    
    # Track original forward method
    original_forward = model.forward
    
    # Define new forward method with KV cache offloading
    def forward_with_kv_offload(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sliding_window_size=None,
    ):
        # Debug logging
        logger.debug(f"KV Cache Offload - Input shapes: input_ids={input_ids.shape if input_ids is not None else None}, "
                     f"attention_mask={attention_mask.shape if attention_mask is not None else None}")
        # Set default for use_cache
        use_cache = True if past_key_values is not None else use_cache

        # Check if we have past_key_values as an offload ID
        if isinstance(past_key_values, str) and past_key_values.startswith("offload:"):
            logger.debug(f"Loading KV cache from disk: {past_key_values}")
            # Load past_key_values from disk
            offload_id = past_key_values.split(":")[1]
            loaded_past_key_values = []

            # Try to load each layer's KV cache
            for i in range(self.config.num_hidden_layers):
                layer_path = os.path.join(self.kv_cache_offload_path, f"{offload_id}_layer_{i}.pt")

                if os.path.exists(layer_path):
                    try:
                        # Load KV cache for this layer
                        layer_kv = torch.load(layer_path, map_location=input_ids.device if input_ids is not None else 'cpu')
                        loaded_past_key_values.append(layer_kv)
                        logger.debug(f"Loaded layer {i} KV cache with shapes: {[t.shape for t in layer_kv]}")
                    except Exception as e:
                        logger.error(f"Error loading KV cache for layer {i}: {str(e)}")
                        loaded_past_key_values.append(None)
                else:
                    logger.warning(f"KV cache file not found for layer {i}: {layer_path}")
                    loaded_past_key_values.append(None)

            # Replace string ID with actual tensors
            if loaded_past_key_values:
                past_key_values = tuple(loaded_past_key_values)
                logger.debug(f"Successfully loaded KV cache with {len(past_key_values)} layers")
            else:
                past_key_values = None
                logger.warning("Failed to load any KV cache layers")
        
        # Run forward pass
        outputs = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sliding_window_size=sliding_window_size,
        )
    

        # If past_key_values is not in outputs, just return the outputs
        if use_cache and "past_key_values" in outputs and outputs["past_key_values"] is not None:
            # Generate unique offload ID
            offload_id = str(uuid.uuid4())
            logger.debug(f"Offloading KV cache to disk with ID: {offload_id}")

            # Get new past_key_values
            new_past_key_values = outputs["past_key_values"]

            # Offload each layer's KV cache to disk
            for i, layer_kv in enumerate(new_past_key_values):
                try:
                    # Check if we have valid KV cache for this layer
                    if layer_kv is None or len(layer_kv) != 2:
                        logger.warning(f"Invalid KV cache for layer {i}: {layer_kv}")
                        continue

                    # Convert to specified dtype to save memory and detach from computational graph
                    k, v = layer_kv
                    k_reduced = k.detach().to(kv_cache_dtype)
                    v_reduced = v.detach().to(kv_cache_dtype)
                    layer_kv_reduced = (k_reduced, v_reduced)

                    # Save to disk
                    layer_path = os.path.join(self.kv_cache_offload_path, f"{offload_id}_layer_{i}.pt")
                    torch.save(layer_kv_reduced, layer_path)
                    logger.debug(f"Saved layer {i} KV cache to {layer_path} with shapes: {[t.shape for t in layer_kv_reduced]}")
                except Exception as e:
                    logger.error(f"Error saving KV cache for layer {i}: {str(e)}")

            # Replace past_key_values with offload ID
            outputs["past_key_values"] = f"offload:{offload_id}"
            logger.debug(f"Replaced past_key_values with offload ID: {outputs['past_key_values']}")
        
        return outputs

    
    # Replace forward method
    model.forward = forward_with_kv_offload.__get__(model, type(model))
    
    # Add cleanup method to model
    def cleanup_kv_cache(self, specific_id=None):
        """
        Clean up KV cache files.
        
        Args:
            specific_id: Optional specific offload ID to clean up
        """
        if specific_id is not None:
            # Clean up specific offload ID
            if isinstance(specific_id, str) and specific_id.startswith("offload:"):
                offload_id = specific_id.split(":")[1]
                for i in range(self.config.num_hidden_layers):
                    layer_path = os.path.join(self.kv_cache_offload_path, f"{offload_id}_layer_{i}.pt")
                    if os.path.exists(layer_path):
                        os.remove(layer_path)
                logger.info(f"Cleaned up KV cache files for ID: {specific_id}")
        else:
            # Clean up all offload files
            import glob
            files = glob.glob(os.path.join(self.kv_cache_offload_path, "*.pt"))
            for file in files:
                os.remove(file)
            logger.info(f"Cleaned up all KV cache files in {self.kv_cache_offload_path}")
    
    # Add cleanup method to model
    model.cleanup_kv_cache = cleanup_kv_cache.__get__(model, type(model))
    
    logger.info(f"Enabled KV cache offloading to {offload_path}")
    
    return model