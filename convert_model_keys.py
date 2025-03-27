#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert Model Keys for EdgeFormer

This script converts between different key naming conventions in EdgeFormer models.
"""

import os
import sys
import torch
import logging
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("convert-model-keys")

def detect_key_format(state_dict):
    """
    Detect the key format used in the state dictionary.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        String indicating the detected format
    """
    # Check for encoder.layer format
    encoder_layer_keys = [k for k in state_dict.keys() if k.startswith('encoder.layer')]
    
    # Check for layers format
    layers_keys = [k for k in state_dict.keys() if k.startswith('layers.')]
    
    if encoder_layer_keys and not layers_keys:
        return "encoder.layer"
    elif layers_keys and not encoder_layer_keys:
        return "layers"
    elif encoder_layer_keys and layers_keys:
        return "mixed"
    else:
        return "unknown"

def convert_encoder_layer_to_layers(state_dict):
    """
    Convert keys from encoder.layer.X format to layers.X format.
    
    Args:
        state_dict: Original state dictionary
        
    Returns:
        Converted state dictionary
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('encoder.layer.'):
            # Replace 'encoder.layer.' with 'layers.'
            new_key = key.replace('encoder.layer.', 'layers.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    return new_state_dict

def convert_layers_to_encoder_layer(state_dict):
    """
    Convert keys from layers.X format to encoder.layer.X format.
    
    Args:
        state_dict: Original state dictionary
        
    Returns:
        Converted state dictionary
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('layers.'):
            # Replace 'layers.' with 'encoder.layer.'
            new_key = key.replace('layers.', 'encoder.layer.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    return new_state_dict

def convert_mla_to_standard(state_dict):
    """
    Convert Multi-Head Latent Attention (MLA) keys to standard attention keys.
    
    Args:
        state_dict: Original state dictionary
        
    Returns:
        Converted state dictionary
    """
    new_state_dict = {}
    
    # Define mappings for MLA keys to standard keys
    mla_to_standard = {
        'q_proj': 'self.query',
        'kv_latent_proj': 'self.key_value',
        'latent_to_k': 'self.key',
        'latent_to_v': 'self.value',
        'out_proj': 'output.dense'
    }
    
    # Define mappings for MLP keys
    mlp_to_standard = {
        'dense_h_to_4h': 'intermediate.dense',
        'dense_4h_to_h': 'output.dense'
    }
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert attention keys
        for mla_key, std_key in mla_to_standard.items():
            if f'attention.{mla_key}' in key:
                new_key = key.replace(f'attention.{mla_key}', f'attention.{std_key}')
                break
        
        # Convert MLP keys
        for mlp_key, std_key in mlp_to_standard.items():
            if f'mlp.{mlp_key}' in key:
                new_key = key.replace(f'mlp.{mlp_key}', f'mlp.{std_key}')
                break
        
        new_state_dict[new_key] = value
    
    return new_state_dict

def convert_model(input_path, output_path, target_format="auto", convert_attn=False):
    """
    Convert a model's state dictionary to a different key format.
    
    Args:
        input_path: Path to input model file
        output_path: Path to save converted model
        target_format: Target key format ('encoder.layer', 'layers', or 'auto')
        convert_attn: Whether to convert attention keys between MLA and standard
        
    Returns:
        None
    """
    # Load the state dict
    logger.info(f"Loading model from {input_path}")
    state_dict = torch.load(input_path)
    
    # Detect current format
    current_format = detect_key_format(state_dict)
    logger.info(f"Detected key format: {current_format}")
    
    # Determine target format if auto
    if target_format == "auto":
        target_format = "layers" if current_format == "encoder.layer" else "encoder.layer"
    
    logger.info(f"Converting to format: {target_format}")
    
    # Convert keys based on target format
    if current_format == "encoder.layer" and target_format == "layers":
        new_state_dict = convert_encoder_layer_to_layers(state_dict)
        logger.info("Converted from encoder.layer to layers format")
    elif current_format == "layers" and target_format == "encoder.layer":
        new_state_dict = convert_layers_to_encoder_layer(state_dict)
        logger.info("Converted from layers to encoder.layer format")
    else:
        new_state_dict = state_dict.copy()
        logger.info("Keeping the same layer format")
    
    # Convert attention keys if requested
    if convert_attn:
        logger.info("Converting attention keys")
        new_state_dict = convert_mla_to_standard(new_state_dict)
    
    # Save the converted state dict
    logger.info(f"Saving converted model to {output_path}")
    torch.save(new_state_dict, output_path)
    
    # Save a report of the conversion
    report_path = os.path.splitext(output_path)[0] + "_conversion_report.json"
    report = {
        "input_model": input_path,
        "output_model": output_path,
        "original_format": current_format,
        "target_format": target_format,
        "converted_attention_keys": convert_attn,
        "sample_original_keys": list(state_dict.keys())[:10],
        "sample_converted_keys": list(new_state_dict.keys())[:10]
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Conversion report saved to {report_path}")
    logger.info("Model conversion completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Convert EdgeFormer Model Keys")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input model file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save converted model")
    parser.add_argument("--target_format", type=str, default="auto",
                        choices=["encoder.layer", "layers", "auto"],
                        help="Target key format (default: auto)")
    parser.add_argument("--convert_attn", action="store_true",
                        help="Convert attention keys between MLA and standard")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        logger.error(f"Input model file not found: {args.input_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        convert_model(
            args.input_path,
            args.output_path,
            args.target_format,
            args.convert_attn
        )
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")

if __name__ == "__main__":
    main()