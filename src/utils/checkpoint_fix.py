import torch
import logging
import inspect
from src.model.config import EdgeFormerConfig

logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load checkpoint with proper filtering of unknown parameters.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        Loaded model or checkpoint data
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a complete checkpoint with config
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # Load from training checkpoint
        config_dict = checkpoint['config']
        if isinstance(config_dict, dict):
            # Filter out unknown arguments
            known_args = [p.name for p in inspect.signature(EdgeFormerConfig.__init__).parameters.values()]
            filtered_config = {k: v for k, v in config_dict.items() if k in known_args}
            config = EdgeFormerConfig(**filtered_config)  # Use filtered_config, not config_dict
            logger.info(f"Loaded filtered config from checkpoint with hidden_size={config.hidden_size}")
        else:
            config = config_dict
            
        # Return the checkpoint with fixed config
        checkpoint['config'] = config
        return checkpoint
    
    # If it's just a state dict or other format, return as is
    return checkpoint
