# src/utils/kv_cache_offload.py
import torch
import os
import tempfile
import uuid
import logging

logger = logging.getLogger(__name__)

def kv_cache_offload(model, offload_path=None, kv_cache_dtype=None):
    """
    Enable KV cache offloading to disk for a model.
    
    Args:
        model: The EdgeFormer model to enable offloading for
        offload_path: Path to store KV cache files (default: temporary directory)
        kv_cache_dtype: Data type for KV cache (default: same as model)
        
    Returns:
        Model with KV cache offloading enabled
    """
    # Create temporary directory if not provided
    if offload_path is None:
        temp_dir = tempfile.mkdtemp()
        offload_path = temp_dir
    else:
        os.makedirs(offload_path, exist_ok=True)
        
    logger.info(f"Enabled KV cache offloading to {offload_path}")
    
    # Store offload path
    model.kv_cache_offload_path = offload_path
    
    # Store dtype for offloaded cache
    model.kv_cache_dtype = kv_cache_dtype
    
    # Keep track of offloaded files
    model.offloaded_kv_files = set()
    
    # Hook the forward method
    original_forward = model.forward
    
    def forward_with_offloading(*args, **kwargs):
        """Forward pass with KV cache offloading."""
        # Extract relevant kwargs
        past_key_values = kwargs.get("past_key_values", None)
        use_cache = kwargs.get("use_cache", False)
        
        # If past_key_values is a string starting with "offload:", load from disk
        if isinstance(past_key_values, str) and past_key_values.startswith("offload:"):
            offload_id = past_key_values[len("offload:"):]
            logger.debug(f"Loading KV cache from disk: {offload_id}")
            
            # Load KV cache from disk
            past_key_values = load_kv_cache_from_disk(model, offload_id)
            kwargs["past_key_values"] = past_key_values
        
        # Call original forward pass
        outputs = original_forward(*args, **kwargs)
        
        # If using cache, offload to disk
        if use_cache and "past_key_values" in outputs and outputs["past_key_values"] is not None:
            # Generate unique ID for this KV cache
            offload_id = str(uuid.uuid4())
            logger.debug(f"Offloading KV cache to disk with ID: {offload_id}")
            
            # Save KV cache to disk
            save_kv_cache_to_disk(model, outputs["past_key_values"], offload_id)
            
            # Replace past_key_values with offload ID
            outputs["past_key_values"] = f"offload:{offload_id}"
        
        return outputs
    
    # Replace forward method
    model.forward = forward_with_offloading
    
    # Add cleanup method
    def cleanup_kv_cache():
        """Clean up all KV cache files."""
        logger.info(f"Cleaning up all KV cache files in {model.kv_cache_offload_path}")
        for filename in model.offloaded_kv_files:
            try:
                os.remove(filename)
            except Exception as e:
                logger.warning(f"Error removing KV cache file {filename}: {e}")
        model.offloaded_kv_files.clear()
    
    model.cleanup_kv_cache = cleanup_kv_cache
    
    return model

def save_kv_cache_to_disk(model, past_key_values, offload_id):
    """Save KV cache to disk."""
    for i, layer_kv in enumerate(past_key_values):
        if layer_kv is None:
            continue
        
        # Save keys and values
        filename = os.path.join(model.kv_cache_offload_path, f"{offload_id}_layer_{i}.pt")
        
        # Convert to specified dtype if needed
        if model.kv_cache_dtype is not None:
            layer_kv = tuple(x.to(model.kv_cache_dtype) if x is not None else None for x in layer_kv)
        
        # Save to disk
        torch.save(layer_kv, filename)
        model.offloaded_kv_files.add(filename)
        
        # Log shapes for debugging
        if layer_kv[0] is not None and layer_kv[1] is not None:
            logger.debug(f"Saved layer {i} KV cache to {filename} with shapes: {[layer_kv[0].shape, layer_kv[1].shape]}")

def load_kv_cache_from_disk(model, offload_id):
    """Load KV cache from disk."""
    past_key_values = []
    layer_idx = 0
    num_layers = 0
    
    # Find all layer files
    while True:
        filename = os.path.join(model.kv_cache_offload_path, f"{offload_id}_layer_{layer_idx}.pt")
        if not os.path.exists(filename):
            break
        
        # Load from disk
        layer_kv = torch.load(filename)
        past_key_values.append(layer_kv)
        
        # Log shapes for debugging
        if layer_kv[0] is not None and layer_kv[1] is not None:
            logger.debug(f"Loaded layer {layer_idx} KV cache with shapes: {[layer_kv[0].shape, layer_kv[1].shape]}")
        
        layer_idx += 1
        num_layers += 1
    
    if num_layers > 0:
        logger.debug(f"Successfully loaded KV cache with {num_layers} layers")
    else:
        logger.warning(f"No KV cache files found for ID {offload_id}")
    
    return past_key_values if past_key_values else None