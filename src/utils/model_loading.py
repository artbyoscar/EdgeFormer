import torch
import os
import importlib
from torch.serialization import safe_globals

def load_custom_model(model_path, device='cpu'):
    """Safely load EdgeFormer model with proper handling of custom classes."""
    
    # Import necessary classes to register them for safe loading
    try:
        # Import the EdgeFormerConfig class
        from src.model.config import EdgeFormerConfig
        from src.model.edgeformer import EdgeFormer
        
        # Add them to safe globals
        with safe_globals([EdgeFormerConfig, EdgeFormer]):
            # Load the model with weights_only=False to allow custom classes
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
            
            # If it's a state dict
            if not isinstance(checkpoint, dict):
                # Create default config and model
                config = EdgeFormerConfig()
                model = EdgeFormer(config)
                model.load_state_dict(checkpoint)
                return model
            
            # If it contains model state dict and config
            if 'model_state_dict' in checkpoint and 'config' in checkpoint:
                config = checkpoint['config']
                model = EdgeFormer(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                return model
            
            # If it's just a model state dict
            if all(isinstance(key, str) and '.' in key for key in checkpoint.keys()):
                # Likely a raw state dict
                config = EdgeFormerConfig()
                model = EdgeFormer(config)
                model.load_state_dict(checkpoint)
                return model
            
            print("Unrecognized checkpoint format")
            return None
            
    except ImportError as e:
        print(f"Error importing model classes: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None