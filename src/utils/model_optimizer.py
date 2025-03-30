from src.utils.device_optimization import DeviceOptimizer

def optimize_model_for_device(model, config=None):
    """
    Apply device-specific optimizations to the model.
    
    Args:
        model: The EdgeFormer model to optimize
        config: Optional model configuration
        
    Returns:
        Optimized model
    """
    # Initialize device optimizer
    optimizer = DeviceOptimizer(model_config=config)
    
    # Log device detection
    device_info = optimizer.device_info
    print(f"Device detected: {device_info['processor']}")
    print(f"RAM: {device_info['ram_gb']:.1f} GB")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']}")
    
    # Apply optimizations based on device profile
    profile = optimizer.optimization_profile
    print(f"Using optimization profile: {profile['attention_strategy']} attention")
    
    # Set device-optimized parameters in the model
    if hasattr(model, 'config'):
        # Set optimal chunk size for long sequences
        model.config.optimal_chunk_size = optimizer.get_optimal_chunk_size(1024)
        
        # Set attention switch point
        model.config.attention_switch_length = profile['attention_switch_length']
        
        # Configure offloading
        if hasattr(model, 'kv_cache_manager'):
            model.kv_cache_manager.offload_threshold = profile['offload_threshold']
    
    return model