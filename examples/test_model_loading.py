#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Model Loading for EdgeFormer

This script tests loading an EdgeFormer model and runs a simple forward pass.
"""

import os
import sys
import torch
import logging
import argparse

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
logger = logging.getLogger("test-model-loading")

def load_model_with_config(model_path, config_path=None):
    """
    Load a model with a specific configuration.
    
    Args:
        model_path: Path to the model file
        config_path: Path to configuration file (optional)
        
    Returns:
        Loaded model
    """
    if config_path and os.path.exists(config_path):
        # Load configuration from file
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create configuration from dict
        config = EdgeFormerConfig(**config_dict)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Use default configuration with large position embeddings
        config = EdgeFormerConfig(
            vocab_size=30522,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            intermediate_size=1024,
            max_position_embeddings=2048  # Important: Match the model's position embedding size
        )
        logger.info("Using default configuration with max_position_embeddings=2048")
    
    # Create model with configuration
    model = EdgeFormer(config)
    
    # Load the state dict
    if os.path.exists(model_path):
        logger.info(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        logger.warning(f"Model file not found: {model_path}, using random initialization")
    
    return model, config

def test_forward_pass(model, seq_length=128):
    """
    Test a forward pass through the model.
    
    Args:
        model: The model to test
        seq_length: Sequence length to test
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Create random input
        input_ids = torch.randint(0, 1000, (1, seq_length))
        attention_mask = torch.ones(1, seq_length)
        
        # Run forward pass
        logger.info(f"Running forward pass with sequence length {seq_length}")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        end_time.record()
        
        # Wait for GPU computation to complete
        torch.cuda.synchronize()
        
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        
        logger.info(f"Forward pass successful in {elapsed_time:.4f} seconds")
        logger.info(f"Output logits shape: {outputs['logits'].shape}")
        
        return True
    
    except Exception as e:
        logger.error(f"Forward pass failed: {str(e)}")
        return False

def main(args):
    # Load model
    model, config = load_model_with_config(args.model_path, args.config_path)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params:,} parameters")
    
    # Test forward pass
    success = test_forward_pass(model, args.seq_length)
    
    if success:
        logger.info("Model test completed successfully")
    else:
        logger.error("Model test failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EdgeFormer Model Loading")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--config_path", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for test")
    
    args = parser.parse_args()
    main(args)