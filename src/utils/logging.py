import logging
import os

def setup_logging(debug_mode=False):
    """Set up logging for the EdgeFormer project.
    
    Args:
        debug_mode: Whether to enable debug level logging
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up logging level based on debug mode
    level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/edgeformer.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create a logger for the project
    logger = logging.getLogger("edgeformer")
    return logger

def log_tensor_shape(tensor, name, logger, debug_mode=False):
    """Log tensor shape during execution for debugging.
    
    Args:
        tensor: PyTorch tensor to log
        name: Name of the tensor for logging
        logger: Logger instance
        debug_mode: Whether to enable debug logging
    """
    if debug_mode:
        shape_str = str(tensor.shape)
        dtype_str = str(tensor.dtype)
        mem_size = tensor.element_size() * tensor.numel() / (1024 * 1024)  # Size in MB
        logger.debug(f"Tensor: {name}, Shape: {shape_str}, Dtype: {dtype_str}, Memory: {mem_size:.2f} MB")