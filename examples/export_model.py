#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Export for EdgeFormer

This script exports an EdgeFormer model to a serialized format for deployment.
"""

import argparse
import os
import sys
import logging
import torch
import glob

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
logger = logging.getLogger("model-export")

def load_model_from_dir(model, model_path):
    """Load model from a directory."""
    # Check if model_path is a directory
    if os.path.isdir(model_path):
        # Look for .pt or .bin files in the directory
        model_files = glob.glob(os.path.join(model_path, "*.pt")) + glob.glob(os.path.join(model_path, "*.bin"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in directory {model_path}")
        
        # Load the first found model file
        model_file = model_files[0]
        logger.info(f"Loading model from {model_file}")
        
        model.load_state_dict(torch.load(model_file))
    else:
        # Direct file loading
        model.load_state_dict(torch.load(model_path))
    
    return model

def export_model(model, output_path, config=None):
    """
    Export EdgeFormer model to deployable format.
    
    Args:
        model: The EdgeFormer model to export
        output_path: Path to save the model
        config: Model configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save model
    logger.info(f"Saving model to {output_path}")
    torch.save(model.state_dict(), output_path)
    
    # Save config if provided
    if config is not None:
        config_path = output_path.replace(".pt", "_config.json")
        logger.info(f"Saving model config to {config_path}")
        with open(config_path, "w") as f:
            import json
            json.dump(config.__dict__, f, indent=2)
    
    # Create mobile package
    mobile_dir = os.path.join(os.path.dirname(output_path), "mobile_package")
    create_mobile_package(output_path, mobile_dir, config)
    
    logger.info(f"Export completed successfully!")
    return True

def create_mobile_package(model_path, output_dir, config):
    """
    Create a mobile deployment package for the model.
    
    Args:
        model_path: Path to the model file
        output_dir: Directory to save the package
        config: Model configuration
    """
    logger.info(f"Creating mobile package in {output_dir}")
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model
    import shutil
    shutil.copy(model_path, os.path.join(output_dir, "model.pt"))
    
    # Save config
    config_dict = {
        "model_type": "EdgeFormer",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "latent_size_factor": config.latent_size_factor,
        "version": "1.0.0",
        "rdna3_optimized": True
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        import json
        json.dump(config_dict, f, indent=2)
    
    # Add README
    readme = f"""# EdgeFormer Mobile Package

This package contains the EdgeFormer model optimized for mobile and edge devices.

## Model Details
- Model type: EdgeFormer
- Hidden size: {config.hidden_size}
- Layers: {config.num_hidden_layers}
- Attention heads: {config.num_attention_heads}
- Vocabulary size: {config.vocab_size}
- MLA latent size factor: {config.latent_size_factor}

## Usage Instructions
Load the model.pt file and config.json using the EdgeFormer library.
    """
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)
    
    logger.info(f"Mobile package created at {output_dir}")

def main(args):
    # Create model configuration
    logger.info("Creating model configuration...")
    config = EdgeFormerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        latent_size_factor=args.latent_size_factor,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = EdgeFormer(config)
    
    # Load model weights if provided
    if args.model_path:
        logger.info(f"Loading model weights from {args.model_path}")
        try:
            model = load_model_from_dir(model, args.model_path)
        except Exception as e:
            logger.warning(f"Failed to load model weights: {str(e)}")
            if not args.continue_on_error:
                logger.error("Aborting export due to model loading failure")
                return
    
    # Set model to evaluation mode
    model.eval()
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params:,} parameters")
    
    # Export model
    export_success = export_model(
        model=model,
        output_path=args.output_path,
        config=config
    )
    
    if export_success:
        logger.info(f"Successfully exported model to {args.output_path}")
    else:
        logger.error("Failed to export model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFormer Model Export")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--latent_size_factor", type=int, default=8, help="Latent size factor for MLA")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Intermediate size for MLP layers")
    parser.add_argument("--max_position_embeddings", type=int, default=128, help="Maximum sequence length for position embeddings")
    
    # Export parameters
    parser.add_argument("--model_path", type=str, default="", help="Path to trained model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save exported model")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue export on non-fatal errors")
    
    args = parser.parse_args()
    main(args)