"""Configuration utilities for EdgeFormer."""
import os
import json
import logging

logger = logging.getLogger('edgeformer')

DEFAULT_CONFIG = {
    "model": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "attention_type": "mla",
        "max_position_embeddings": 2048
    },
    "memory": {
        "capacity": 100,
        "selection_strategy": "htps"
    },
    "optimization": {
        "device": "auto",
        "quantization": "none",
        "kv_offload": False,
        "sequence_chunking": True
    },
    "training": {
        "batch_size": 8,
        "learning_rate": 5e-5,
        "scheduler": "linear",
        "warmup_steps": 100
    }
}

def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update config with user values
            for section, values in user_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values
                    
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    return config

def save_config(config, config_path):
    """Save configuration to file."""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False

def get_device_config():
    """Get device-specific configuration."""
    import platform
    import psutil
    
    # Basic device info
    device_info = {
        "system": platform.system(),
        "processor": platform.processor(),
        "ram_gb": psutil.virtual_memory().total / (1024 ** 3)
    }
    
    # Device-specific optimization settings
    if "amd" in device_info["processor"].lower():
        device_info["recommended_attention"] = "sliding_window"
    elif "intel" in device_info["processor"].lower():
        device_info["recommended_attention"] = "hybrid"
    else:
        device_info["recommended_attention"] = "standard"
    
    # Memory-based recommendations
    if device_info["ram_gb"] < 4:
        device_info["max_sequence_length"] = 1024
        device_info["recommended_quantization"] = "int8"
    elif device_info["ram_gb"] < 8:
        device_info["max_sequence_length"] = 2048
        device_info["recommended_quantization"] = "int8"
    else:
        device_info["max_sequence_length"] = 4096
        device_info["recommended_quantization"] = "fp16"
    
    return device_info