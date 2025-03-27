#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Loading Fix for EdgeFormer

This script properly loads an EdgeFormer model by detecting its configuration
and model structure.
"""

import os
import sys
import torch
import logging
import json

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model-load-fix")

def inspect_model_file(model_path):
    """
    Inspect a model file to determine its configuration.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dict containing estimated model configuration and state_dict
    """
    logger.info(f"Inspecting model file: {model_path}")
    
    # Load the state dict
    state_dict = torch.load(model_path)
    
    # Extract configuration information from state dict keys and shapes
    config = {}
    
    # Check if position_ids exists to determine max_position_embeddings
    if 'embeddings.position_embeddings.weight' in state_dict:
        pos_emb_shape = state_dict['embeddings.position_embeddings.weight'].shape
        config['max_position_embeddings'] = pos_emb_shape[0]
        logger.info(f"Detected max_position_embeddings: {config['max_position_embeddings']}")
    
    # Determine hidden_size
    if 'embeddings.word_embeddings.weight' in state_dict:
        hidden_size = state_dict['embeddings.word_embeddings.weight'].shape[1]
        config['hidden_size'] = hidden_size
        logger.info(f"Detected hidden_size: {config['hidden_size']}")
    
    # Determine vocab_size
    if 'embeddings.word_embeddings.weight' in state_dict:
        vocab_size = state_dict['embeddings.word_embeddings.weight'].shape[0]
        config['vocab_size'] = vocab_size
        logger.info(f"Detected vocab_size: {config['vocab_size']}")
    
    # Determine num_hidden_layers by counting unique layer indices
    layer_indices = set()
    for key in state_dict.keys():
        if 'encoder.layer.' in key:
            parts = key.split('.')
            # Find the index position (it might vary depending on key structure)
            for i, part in enumerate(parts):
                if part == 'layer' and i+1 < len(parts) and parts[i+1].isdigit():
                    layer_indices.add(int(parts[i+1]))
                    break
    
    if layer_indices:
        num_layers = max(layer_indices) + 1
        config['num_hidden_layers'] = num_layers
        logger.info(f"Detected num_hidden_layers: {config['num_hidden_layers']}")
    
    # Check if the model uses the "layers" prefix instead of "encoder.layer"
    if not layer_indices:
        alt_layer_indices = set()
        for key in state_dict.keys():
            if key.startswith('layers.'):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    alt_layer_indices.add(int(parts[1]))
        
        if alt_layer_indices:
            num_layers = max(alt_layer_indices) + 1
            config['num_hidden_layers'] = num_layers
            logger.info(f"Detected num_hidden_layers (alt format): {config['num_hidden_layers']}")
            config['uses_layers_prefix'] = True
    
    # Determine intermediate_size from any intermediate dense weight
    for key in state_dict.keys():
        if 'intermediate.dense.weight' in key or 'mlp.dense_h_to_4h.weight' in key:
            intermediate_weight = state_dict[key]
            config['intermediate_size'] = intermediate_weight.shape[0]
            logger.info(f"Detected intermediate_size: {config['intermediate_size']}")
            break
    
    # Try to infer num_attention_heads and latent_size_factor
    attention_dim_keys = [
        'attention.self.query.weight', 
        'attention.self.key.weight',
        'attention.q_proj.weight',
        'attention.kv_latent_proj.weight'
    ]
    
    for key_pattern in attention_dim_keys:
        for key in state_dict.keys():
            if key_pattern in key:
                attention_weight = state_dict[key]
                if 'hidden_size' in config:
                    # If this is the query weight
                    if 'query' in key or 'q_proj' in key:
                        # Estimate num_attention_heads
                        # Assuming attention head size = hidden_size / num_heads
                        for heads in [4, 8, 12, 16, 24, 32]:
                            if config['hidden_size'] % heads == 0:
                                head_size = config['hidden_size'] // heads
                                if attention_weight.shape[0] == config['hidden_size']:
                                    config['num_attention_heads'] = heads
                                    logger.info(f"Detected num_attention_heads: {config['num_attention_heads']}")
                                    break
                    
                    # If this is the key/value weight
                    elif 'key' in key or 'kv_latent' in key:
                        # Try to detect latent size
                        key_dim = attention_weight.shape[0]
                        if key_dim < config['hidden_size']:
                            # If using MLA, key dimension will be smaller than hidden_size
                            latent_size_factor = config['hidden_size'] / key_dim
                            if latent_size_factor.is_integer():
                                config['latent_size_factor'] = int(latent_size_factor)
                                logger.info(f"Detected latent_size_factor: {config['latent_size_factor']}")
                                break
                break
    
    # If we couldn't determine num_attention_heads, use common values based on hidden_size
    if 'num_attention_heads' not in config and 'hidden_size' in config:
        if config['hidden_size'] <= 128:
            config['num_attention_heads'] = 4
        elif config['hidden_size'] <= 256:
            config['num_attention_heads'] = 8
        elif config['hidden_size'] <= 512:
            config['num_attention_heads'] = 12
        else:
            config['num_attention_heads'] = 16
        logger.info(f"Using estimated num_attention_heads: {config['num_attention_heads']}")
    
    # If latent_size_factor not detected, assume default
    if 'latent_size_factor' not in config:
        config['latent_size_factor'] = 8
        logger.info(f"Using default latent_size_factor: {config['latent_size_factor']}")
    
    # If intermediate_size not detected, use a multiple of hidden_size
    if 'intermediate_size' not in config and 'hidden_size' in config:
        config['intermediate_size'] = config['hidden_size'] * 4
        logger.info(f"Using estimated intermediate_size: {config['intermediate_size']}")
    
    return config, state_dict

def create_model_from_config(config):
    """
    Create a model using the detected configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model_config = EdgeFormerConfig(
        vocab_size=config.get('vocab_size', 30522),
        hidden_size=config.get('hidden_size', 256),
        num_hidden_layers=config.get('num_hidden_layers', 6),
        num_attention_heads=config.get('num_attention_heads', 8),
        latent_size_factor=config.get('latent_size_factor', 8),
        intermediate_size=config.get('intermediate_size', 1024),
        max_position_embeddings=config.get('max_position_embeddings', 2048)
    )
    
    logger.info("Creating model with detected configuration")
    model = EdgeFormer(model_config)
    
    return model, model_config

def analyze_and_save_config(config, state_dict, output_path):
    """
    Analyze the model configuration and save it to a file.
    
    Args:
        config: Model configuration
        state_dict: Model state dictionary
        output_path: Path to save the configuration
        
    Returns:
        None
    """
    # Get all keys in state_dict
    all_keys = list(state_dict.keys())
    
    # Save key structure for debugging
    key_structure = {
        "config": config,
        "key_patterns": {},
        "sample_keys": all_keys[:20] if len(all_keys) > 20 else all_keys
    }
    
    # Count keys with different prefixes
    for key in all_keys:
        prefix = key.split('.')[0]
        if prefix not in key_structure["key_patterns"]:
            key_structure["key_patterns"][prefix] = 0
        key_structure["key_patterns"][prefix] += 1
    
    # Save analysis to file
    with open(output_path, "w") as f:
        json.dump(key_structure, f, indent=2)
    
    logger.info(f"Model analysis saved to {output_path}")

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python model_load_fix.py <model_path>")
        return
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    try:
        # Extract the directory and filename
        dir_path = os.path.dirname(model_path)
        filename = os.path.basename(model_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Analyze the model file
        config, state_dict = inspect_model_file(model_path)
        
        # Save the analysis
        analysis_path = os.path.join(dir_path, f"{name_without_ext}_analysis.json")
        analyze_and_save_config(config, state_dict, analysis_path)
        
        # Try to create and load the model
        model, model_config = create_model_from_config(config)
        
        # Check for key format issues
        if config.get('uses_layers_prefix', False):
            logger.info("Model uses 'layers' prefix instead of 'encoder.layer'")
            logger.info("Creating a model mapping file to help with loading...")
            
            # Find number of layers
            num_layers = config.get('num_hidden_layers', 2)
            
            # Print a sample of the key formats
            sample_keys = list(state_dict.keys())[:10]
            logger.info(f"Sample keys: {sample_keys}")
            
            # Create a new file with model mapping instructions
            mapping_path = os.path.join(dir_path, f"{name_without_ext}_mapping.txt")
            with open(mapping_path, "w") as f:
                f.write("# EdgeFormer Model Loading Instructions\n\n")
                f.write("The model was saved using a different key structure than what the current code expects.\n")
                f.write("To load this model, you need to modify your EdgeFormer model class to match this format, or convert the state dict.\n\n")
                f.write("## Current Key Format\n\n")
                for key in sample_keys:
                    f.write(f"- {key}\n")
                
                f.write("\n## Model Configuration\n\n")
                for key, value in config.items():
                    if key != 'uses_layers_prefix':
                        f.write(f"- {key}: {value}\n")
            
            logger.info(f"Model mapping instructions saved to {mapping_path}")
        
        # Save configuration to file
        config_path = os.path.join(dir_path, f"{name_without_ext}_config.json")
        with open(config_path, "w") as f:
            model_config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('_')}
            json.dump(model_config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        logger.info("The model structure doesn't match your current EdgeFormer implementation.")
        logger.info("Check the analysis file for more details on the model's structure.")
        logger.info("You may need to update your model code or convert the state dict.")
        
    except Exception as e:
        logger.error(f"Failed to process model: {str(e)}")

if __name__ == "__main__":
    main()