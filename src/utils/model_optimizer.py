from src.utils.device_optimization import DeviceOptimizer

def optimize_model_for_device(model, config=None, phase2=False):
    """
    Apply device-specific optimizations to the model.
    
    Args:
        model: The EdgeFormer model to optimize
        config: Optional model configuration
        phase2: Whether to apply Phase 2 optimizations
        
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
        if hasattr(model, 'kv_cache_manager') and model.kv_cache_manager is not None:
            model.kv_cache_manager.offload_threshold = profile['offload_threshold']
    
    # Apply Phase 2 optimizations if requested
    if phase2:
        model = _apply_phase2_optimizations(model, config, device_info, profile)
        print("Applied Phase 2 optimizations")
    
    return model

def _apply_phase2_optimizations(model, config, device_info, profile):
    """
    Apply advanced Phase 2 optimizations based on vendor-specific hardware.
    
    Args:
        model: The EdgeFormer model to optimize
        config: Model configuration
        device_info: Device information dictionary
        profile: Optimization profile
        
    Returns:
        Optimized model
    """
    # Detect vendor-specific optimizations
    vendor = _detect_hardware_vendor(device_info)
    print(f"Applying Phase 2 optimizations for {vendor} hardware")
    
    # Apply vendor-specific optimizations
    if vendor == "AMD":
        model = _optimize_for_amd(model, config, device_info, profile)
    elif vendor == "Intel":
        model = _optimize_for_intel(model, config, device_info, profile)
    elif vendor == "ARM":
        model = _optimize_for_arm(model, config, device_info, profile)
    
    # Apply power-aware optimizations
    model = _apply_power_optimizations(model, config, device_info)
    
    # Apply memory-specific optimizations based on available RAM
    model = _optimize_for_memory(model, config, device_info)
    
    # Enable enterprise features if appropriate
    if profile.get('enterprise_mode', False):
        model = _enable_enterprise_features(model, config)
    
    return model

def _detect_hardware_vendor(device_info):
    """Detect hardware vendor from device information"""
    processor = device_info.get('processor', '').lower()
    
    if 'amd' in processor or 'ryzen' in processor:
        return "AMD"
    elif 'intel' in processor:
        return "Intel"
    elif 'arm' in processor or device_info.get('architecture', '').lower() in ['arm64', 'aarch64']:
        return "ARM"
    else:
        return "Generic"

def _optimize_for_amd(model, config, device_info, profile):
    """Apply AMD-specific optimizations"""
    import torch
    
    # Apply AMD-specific thread optimizations
    if hasattr(torch, 'set_num_threads'):
        # AMD processors often benefit from using all available threads
        cores = device_info.get('cpu_count', 4)
        threads = device_info.get('cpu_threads', 8)
        torch.set_num_threads(threads)
        print(f"Set thread count to {threads} for AMD processor")
    
    # Optimize for AMD GPU if available
    if device_info.get('cuda_available', False) and 'amd' in device_info.get('gpu_name', '').lower():
        if hasattr(model.config, 'attention_implementation'):
            # AMD GPUs benefit from specialized attention implementations
            model.config.attention_implementation = 'amd_optimized'
    
    # Apply Ryzen-specific memory layout optimizations
    if hasattr(model, 'set_memory_layout') and 'ryzen' in device_info.get('processor', '').lower():
        model.set_memory_layout('ccx_optimized')
    
    return model

def _optimize_for_intel(model, config, device_info, profile):
    """Apply Intel-specific optimizations"""
    import torch
    
    # Apply Intel-specific thread optimizations
    if hasattr(torch, 'set_num_threads'):
        # Intel processors often benefit from leaving some threads for the OS
        cores = device_info.get('cpu_count', 4)
        threads = device_info.get('cpu_threads', 8)
        optimal_threads = max(1, threads - 2)  # Leave 2 threads for OS
        torch.set_num_threads(optimal_threads)
        print(f"Set thread count to {optimal_threads} for Intel processor")
    
    # Enable Intel MKL optimizations if available
    try:
        import mkl
        mkl.set_num_threads(optimal_threads)
        print("Enabled Intel MKL optimizations")
    except ImportError:
        pass
    
    # Optimize for Intel GPU if available
    if hasattr(model.config, 'attention_implementation') and device_info.get('gpu_vendor', '').lower() == 'intel':
        model.config.attention_implementation = 'intel_optimized'
    
    return model

def _optimize_for_arm(model, config, device_info, profile):
    """Apply ARM-specific optimizations"""
    import torch
    
    # ARM-specific thread optimizations
    if hasattr(torch, 'set_num_threads'):
        cores = device_info.get('cpu_count', 4)
        optimal_threads = max(1, cores)  # ARM processors often benefit from using all cores
        torch.set_num_threads(optimal_threads)
        print(f"Set thread count to {optimal_threads} for ARM processor")
    
    # Enable ARM NEON optimizations where applicable
    if hasattr(model.config, 'use_neon_acceleration'):
        model.config.use_neon_acceleration = True
        print("Enabled ARM NEON acceleration")
    
    # Apply mobile-specific optimizations
    if device_info.get('ram_gb', 16) < 8:
        # Apply more aggressive memory optimizations for low-memory devices
        if hasattr(model.config, 'kv_cache_strategy'):
            model.config.kv_cache_strategy = 'minimal'
        if hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = min(model.config.max_sequence_length, 1024)
        print("Applied mobile-optimized memory settings")
    
    return model

def _apply_power_optimizations(model, config, device_info):
    """Apply power-aware optimizations"""
    # Detect if running on battery
    on_battery = device_info.get('on_battery', False)
    
    # Apply power-saving optimizations if on battery
    if on_battery and hasattr(model.config, 'power_mode'):
        model.config.power_mode = 'efficient'
        print("Enabled power-efficient mode (on battery)")
        
        # Adjust inference parameters for power efficiency
        if hasattr(model.config, 'max_batch_size'):
            model.config.max_batch_size = min(model.config.max_batch_size, 4)
        
        # Enable compute budget forcing
        if hasattr(model.config, 'enforce_compute_budget'):
            model.config.enforce_compute_budget = True
            model.config.compute_budget_level = 'conservative'
    
    return model

def _optimize_for_memory(model, config, device_info):
    """Apply memory-specific optimizations"""
    ram_gb = device_info.get('ram_gb', 16)
    
    # Apply memory optimizations based on available RAM
    if ram_gb < 4:
        # Ultra low memory mode
        print("Applying ultra low memory optimizations (<4GB RAM)")
        if hasattr(model.config, 'kv_cache_strategy'):
            model.config.kv_cache_strategy = 'minimal'
        if hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = min(getattr(model.config, 'max_sequence_length', 2048), 1024)
        if hasattr(model.config, 'attention_implementation'):
            model.config.attention_implementation = 'memory_efficient'
    elif ram_gb < 8:
        # Low memory mode
        print("Applying low memory optimizations (<8GB RAM)")
        if hasattr(model.config, 'kv_cache_strategy'):
            model.config.kv_cache_strategy = 'efficient'
        if hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = min(getattr(model.config, 'max_sequence_length', 4096), 2048)
    
    # Adjust quantization based on available memory
    if hasattr(model.config, 'quantization_level'):
        if ram_gb < 4:
            model.config.quantization_level = 'int4'
        elif ram_gb < 8:
            model.config.quantization_level = 'int8'
    
    return model

def _enable_enterprise_features(model, config):
    """Enable enterprise-specific features"""
    print("Enabling enterprise features")
    
    # Enable multi-model deployment optimizations
    if hasattr(model.config, 'enable_model_sharing'):
        model.config.enable_model_sharing = True
    
    # Enable robust logging for enterprise deployments
    if hasattr(model.config, 'logging_level'):
        model.config.logging_level = 'detailed'
    
    # Enable advanced security features
    if hasattr(model.config, 'secure_mode'):
        model.config.secure_mode = True
    
    return model
