def main(args):
    """Main function to export model to ONNX."""
    
    # Create model configuration
    logger.info("Creating model configuration...")
    config = EdgeFormerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        latent_size_factor=args.latent_size_factor,
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = EdgeFormer(config)
    
    # Load model weights if provided
    if args.model_path:
        logger.info(f"Loading model weights from {args.model_path}")
        try:
            model.load_state_dict(torch.load(args.model_path))
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
    
    # Quantize model weights if requested (before ONNX export)
    if args.weight_quantize:
        logger.info(f"Quantizing model weights to {args.weight_quantize_bits} bits...")
        model = quantize_model(
            model, 
            bits=args.weight_quantize_bits,
            group_size=args.weight_quantize_group_size
        )
    
    # Export model to ONNX
    export_success = export_to_onnx(
        model=model,
        output_path=args.output_path,
        optimize=not args.no_optimize,
        quantize=args.quantize,
        quantization_type=args.quantization_type,
        opset_version=args.opset_version,
        create_package=args.create_package
    )
    
    if export_success:
        logger.info(f"Successfully exported model to {args.output_path}")
    else:
        logger.error("Failed to export model to ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFormer ONNX Export")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--latent_size_factor", type=int, default=8, help="Latent size factor for MLA")
    
    # Export parameters
    parser.add_argument("--model_path", type=str, default="", help="Path to trained model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save ONNX model")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--no_optimize", action="store_true", help="Skip ONNX optimization")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue export on non-fatal errors")
    
    # Quantization parameters
    parser.add_argument("--quantize", action="store_true", help="Quantize the ONNX model")
    parser.add_argument("--quantization_type", type=str, default="int8", choices=["int8", "int4"], 
                        help="Type of ONNX quantization")
    parser.add_argument("--weight_quantize", action="store_true", help="Quantize model weights before ONNX export")
    parser.add_argument("--weight_quantize_bits", type=int, default=8, help="Bit width for weight quantization")
    parser.add_argument("--weight_quantize_group_size", type=int, default=64, 
                        help="Group size for weight quantization")
    
    # Additional options
    parser.add_argument("--create_package", action="store_true", 
                        help="Create a mobile deployment package")
    
    args = parser.parse_args()
    main(args)
def export_to_onnx(
    model: EdgeFormer,
    output_path: str,
    optimize: bool = True,
    quantize: bool = False,
    quantization_type: str = "int8",
    opset_version: int = 14,
    create_package: bool = False
) -> bool:
    """
    Export EdgeFormer model to ONNX format.
    
    Args:
        model: The EdgeFormer model to export
        output_path: Path to save the ONNX model
        optimize: Whether to optimize the ONNX model
        quantize: Whether to quantize the model
        quantization_type: Type of quantization to apply ("int8" or "int4")
        opset_version: ONNX opset version to use
        create_package: Whether to create a mobile deployment package
        
    Returns:
        True if export is successful, False otherwise
    """
    if not has_onnx:
        logger.error("ONNX is required for export. Please install onnx.")
        return False
    
    try:
        # Make sure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Prepare sample input for tracing
        logger.info("Preparing sample input for ONNX tracing...")
        batch_size = 1
        seq_length = 32
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Wrap the model for ONNX export
        wrapped_model = create_onnx_tracing_wrapper(model)
        wrapped_model.eval()
        
        # Get PyTorch output for validation later
        with torch.no_grad():
            pytorch_output = wrapped_model(input_ids, attention_mask)
        
        # Input and output names for the ONNX model
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        
        # Dynamic axes for variable batch size and sequence length
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"}
        }
        
        # Export the model to ONNX
        logger.info(f"Exporting model to ONNX (opset version {opset_version})...")
        torch.onnx.export(
            wrapped_model,
            (input_ids, attention_mask),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        
        # Check if the exported model is valid
        if has_onnx:
            try:
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verified successfully")
            except Exception as e:
                logger.warning(f"ONNX model verification failed: {str(e)}")
        
        # Validate the model with sample input
        validation_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        validate_onnx_model(output_path, validation_inputs, pytorch_output)
        
        # Optimize the model if requested
        if optimize:
            optimized_path = output_path.replace(".onnx", "_optimized.onnx")
            if optimize_onnx_model(output_path, optimized_path):
                logger.info(f"Using optimized model for further processing")
                output_path = optimized_path
        
        # Quantize the model if requested
        if quantize:
            quantized_path = output_path.replace(".onnx", f"_{quantization_type}.onnx")
            if quantize_onnx_model(output_path, quantized_path, quantization_type):
                logger.info(f"Using quantized model for further processing")
                output_path = quantized_path
        
        # Create mobile deployment package if requested
        if create_package:
            package_dir = os.path.join(os.path.dirname(output_path), "mobile_package")
            create_mobile_package(output_path, package_dir, model.config)
        
        logger.info(f"Model export completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {str(e)}")
        return False
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX Export for EdgeFormer

This script exports an EdgeFormer model to ONNX format for deployment on mobile
and edge devices, with optional quantization.
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import shutil

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import quantize_model, export_quantized_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("onnx-export")

# Try to import onnxruntime if available
try:
    import onnxruntime as ort
    has_ort = True
except ImportError:
    logger.warning("onnxruntime not found. Validation will be skipped.")
    has_ort = False

# Try to import onnx if available
try:
    import onnx
    from onnx import optimizer
    has_onnx = True
except ImportError:
    logger.warning("onnx not found. Model checking will be skipped.")
    has_onnx = False


def create_onnx_tracing_wrapper(model):
    """
    Create a wrapper around the model for ONNX tracing.
    This handles attention mask creation and other edge cases.
    """
    
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
        ):
            # Create default attention mask if none provided
            if attention_mask is None:
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
                
            # Run model
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=False,  # Disable KV cache for export
                past_key_values=None,
            )
            
            # Return logits for next token prediction
            return outputs["logits"]
    
    return ONNXWrapper(model)


def optimize_onnx_model(onnx_path: str, optimized_path: str) -> bool:
    """
    Optimize ONNX model for inference.
    
    Args:
        onnx_path: Path to the original ONNX model
        optimized_path: Path to save the optimized model
        
    Returns:
        True if optimization is successful, False otherwise
    """
    if not has_onnx:
        logger.warning("Skipping optimization as onnx is not available")
        return False
    
    logger.info("Optimizing ONNX model...")
    
    try:
        # Load model
        model = onnx.load(onnx_path)
        
        # Check model
        onnx.checker.check_model(model)
        
        # Optimize model
        passes = [
            "eliminate_unused_initializer",
            "eliminate_identity",
            "eliminate_nop_pad",
            "extract_constant_to_initializer",
            "eliminate_duplicate_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ]
        
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, optimized_path)
        
        logger.info(f"Optimized model saved to {optimized_path}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX model optimization failed: {str(e)}")
        return False


def quantize_onnx_model(input_path: str, output_path: str, quantization_type: str = "int8") -> bool:
    """
    Quantize ONNX model to INT8 or INT4 for reduced size and faster inference.
    
    Args:
        input_path: Path to the input ONNX model
        output_path: Path to save the quantized model
        quantization_type: Type of quantization ("int8" or "int4")
        
    Returns:
        True if quantization is successful, False otherwise
    """
    if not has_onnx:
        logger.warning("Skipping quantization as onnx is not available")
        return False
    
    try:
        # Import onnxruntime.quantization
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        logger.info(f"Quantizing ONNX model to {quantization_type.upper()}...")
        
        # Define quantization type
        quant_type = QuantType.QInt8 if quantization_type.lower() == "int8" else QuantType.QInt4
        
        # Perform dynamic quantization
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=quant_type,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            op_types_to_quantize=['MatMul', 'Gemm', 'Relu', 'Add', 'Mul']
        )
        
        logger.info(f"Quantized model saved to {output_path}")
        
        # Check model size reduction
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Model size: {original_size:.2f} MB → {quantized_size:.2f} MB ({reduction:.2f}% reduction)")
        
        return True
        
    except ImportError:
        logger.error("Could not import onnxruntime.quantization. Please install onnxruntime-extensions.")
        return False
    except Exception as e:
        logger.error(f"ONNX model quantization failed: {str(e)}")
        return False


def create_mobile_package(model_path: str, output_dir: str, config: EdgeFormerConfig) -> bool:
    """
    Create a mobile deployment package with the ONNX model and necessary files.
    
    Args:
        model_path: Path to the ONNX model
        output_dir: Directory to save the package
        config: The EdgeFormer config used to create the model
        
    Returns:
        True if package creation is successful, False otherwise
    """
    logger.info(f"Creating mobile package in {output_dir}...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy model to output directory
        shutil.copy(model_path, os.path.join(output_dir, "model.onnx"))
        
        # Create config.json with model parameters
        import json
        config_dict = {
            "model_type": "EdgeFormer",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "latent_size_factor": config.latent_size_factor,
            "version": "1.0.0",
            "rdna3_optimized": True,
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
        # Create a README.md with instructions
        readme_content = f"""# EdgeFormer Mobile Package

This package contains the EdgeFormer model optimized for mobile and edge devices.

## Model Details
- Model type: EdgeFormer
- Hidden size: {config.hidden_size}
- Layers: {config.num_hidden_layers}
- Attention heads: {config.num_attention_heads}
- Vocabulary size: {config.vocab_size}
- MLA latent size factor: {config.latent_size_factor}

## Usage
See documentation for integration instructions.
"""
        
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(readme_content)
            
        logger.info(f"Successfully created mobile package at {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create mobile package: {str(e)}")
        return False


def validate_onnx_model(onnx_path: str, input_data: Dict[str, torch.Tensor], pytorch_outputs: torch.Tensor) -> bool:
    """
    Validate the exported ONNX model by comparing its outputs with PyTorch model.
    
    Args:
        onnx_path: Path to the exported ONNX model
        input_data: Dictionary of input tensors for validation
        pytorch_outputs: Original PyTorch model outputs for comparison
        
    Returns:
        True if validation is successful, False otherwise
    """
    if not has_ort:
        logger.warning("Skipping validation as onnxruntime is not available")
        return False
    
    logger.info("Validating ONNX model...")
    
    try:
        # Create ONNX runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        
        # Create inference session
        session = ort.InferenceSession(onnx_path, sess_options)
        
        # Prepare inputs
        ort_inputs = {k: v.cpu().numpy() for k, v in input_data.items()}
        
        # Run inference
        ort_outputs = session.run(None, ort_inputs)
        
        # Compare outputs
        torch_output_np = pytorch_outputs.cpu().numpy()
        onnx_output_np = ort_outputs[0]
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(torch_output_np - onnx_output_np))
        max_diff = np.max(np.abs(torch_output_np - onnx_output_np))
        
        logger.info(f"Output comparison - Mean Absolute Error: {mae:.6f}, Max Difference: {max_diff:.6f}")
        
        # Check if the difference is acceptable
        if mae > 1e-4 or max_diff > 1e-2:
            logger.warning("ONNX model outputs differ significantly from PyTorch model")
            return False
            
        logger.info("ONNX model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"ONNX model validation failed: {str(e)}")
        return False
