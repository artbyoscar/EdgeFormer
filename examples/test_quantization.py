# examples/test_quantization.py
import argparse
import torch
import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import EdgeFormer classes
from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('edgeformer')

def quantize_model(model, quantization_type="int8"):
    """
    Quantize model weights to lower precision
    
    Args:
        model: The model to quantize
        quantization_type: Type of quantization ("int4" or "int8")
        
    Returns:
        Quantized model
    """
    logger.info(f"Quantizing model to {quantization_type}...")
    
    # Create a copy of the model for quantization
    quantized_model = type(model)(model.config)
    
    # Copy the state dict
    state_dict = model.state_dict()
    
    # Quantize each parameter
    quantized_state_dict = {}
    for name, param in state_dict.items():
        # Skip non-tensor parameters
        if not isinstance(param, torch.Tensor):
            quantized_state_dict[name] = param
            continue
            
        # Skip parameters that shouldn't be quantized (e.g., embeddings)
        if "embed" in name:
            quantized_state_dict[name] = param
            continue
            
        # Determine number of bits for quantization
        num_bits = 4 if quantization_type == "int4" else 8
        
        # Perform simulated quantization 
        # (in a real implementation, you'd use proper quantization libraries)
        with torch.no_grad():
            # Calculate scale
            abs_max = torch.max(torch.abs(param))
            scale = (2**(num_bits-1) - 1) / abs_max
            
            # Quantize
            quantized = torch.round(param * scale)
            quantized = torch.clamp(quantized, -2**(num_bits-1), 2**(num_bits-1)-1)
            
            # Dequantize (for simulation purposes - real quantized models would keep values quantized)
            dequantized = quantized / scale
            
            # Store the dequantized values
            quantized_state_dict[name] = dequantized
    
    # Load the quantized state dict
    quantized_model.load_state_dict(quantized_state_dict)
    
    return quantized_model

def main():
    parser = argparse.ArgumentParser(description="Test EdgeFormer quantization")
    parser.add_argument("--model_type", type=str, default="standard", 
                        choices=["standard", "mla", "mla_window"], 
                        help="Type of attention mechanism")
    parser.add_argument("--quantization_type", type=str, default="int8", 
                        choices=["int4", "int8"], help="Quantization type")
    args = parser.parse_args()
    
    logger.info(f"Creating EdgeFormer with {args.model_type} attention...")
    
    # Create a model with standard config
    # Note: We're removing 'attention_type' since it's not a valid parameter
    config = EdgeFormerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
        intermediate_size=1024,
        max_position_embeddings=4096,
    )
    
    model = EdgeFormer(config)
    
    # Set the attention type manually if needed
    # Assuming EdgeFormer has an 'attention_type' attribute that can be set after initialization
    if hasattr(model, 'attention_type'):
        model.attention_type = args.model_type
        logger.info(f"Set attention type to {args.model_type}")
    else:
        logger.warning(f"Model doesn't have 'attention_type' attribute. Using default attention mechanism.")
    
    # Quantize the model
    logger.info(f"Quantizing model to {args.quantization_type}...")
    quantized_model = quantize_model(model, args.quantization_type)
    
    # Test with a small input
    logger.info("Generating test input...")
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    
    # Measure original model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Measure quantized model size (simulated)
    bytes_per_param = 0.5 if args.quantization_type == "int4" else 1
    quantized_size = sum(p.numel() * bytes_per_param for p in quantized_model.parameters()) / (1024 * 1024)
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    # Test inference speed
    model.eval()
    quantized_model.eval()
    
    # Warm-up run
    logger.info("Performing warm-up inference runs...")
    with torch.no_grad():
        _ = model(input_ids)
        _ = quantized_model(input_ids)
    
    # Measure original model speed
    logger.info("Testing original model inference speed...")
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    original_time = time.time() - start_time
    
    # Measure quantized model speed
    logger.info("Testing quantized model inference speed...")
    start_time = time.time()
    with torch.no_grad():
        _ = quantized_model(input_ids)
    quantized_time = time.time() - start_time
    
    logger.info(f"Original model inference time: {original_time:.4f}s")
    logger.info(f"Quantized model inference time: {quantized_time:.4f}s")
    logger.info(f"Speed ratio: {original_time / quantized_time:.2f}x")
    
    # Print a summary report
    print("\n===== QUANTIZATION TEST SUMMARY =====")
    print(f"Model type: {args.model_type}")
    print(f"Quantization type: {args.quantization_type}")
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    print(f"Original inference time: {original_time:.4f}s")
    print(f"Quantized inference time: {quantized_time:.4f}s")
    print(f"Speedup: {original_time / quantized_time:.2f}x")
    print("=====================================")

if __name__ == "__main__":
    logger.info("Starting EdgeFormer test...")
    main()