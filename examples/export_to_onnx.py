#!/usr/bin/env python
"""Export an EdgeFormer model to ONNX with optional optimizations and quantization."""

import argparse
import logging
import os
import shutil
import sys
from typing import Dict, Optional

import numpy as np
import torch

# Add repository root to path so local modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import quantize_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("onnx-export")

# Optional dependencies
try:
    import onnx
    from onnx import optimizer
    HAS_ONNX = True
except ImportError:  # pragma: no cover - optional dependency
    logger.warning("onnx not found. Model checking will be skipped.")
    HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:  # pragma: no cover - optional dependency
    logger.warning("onnxruntime not found. Validation will be skipped.")
    HAS_ORT = False


def create_onnx_tracing_wrapper(model: EdgeFormer) -> torch.nn.Module:
    """Wrap ``model`` so it can be traced easily for ONNX export."""

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, wrapped: EdgeFormer) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            outputs = self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                past_key_values=None,
            )
            return outputs["logits"]

    return ONNXWrapper(model)


def optimize_onnx_model(onnx_path: str, optimized_path: str) -> bool:
    """Optimize an ONNX model for inference."""
    if not HAS_ONNX:
        logger.warning("Skipping optimization as onnx is not available")
        return False

    logger.info("Optimizing ONNX model...")
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
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
        optimized = optimizer.optimize(model, passes)
        onnx.save(optimized, optimized_path)
        logger.info("Optimized model saved to %s", optimized_path)
        return True
    except Exception as exc:  # pragma: no cover - runtime behaviour
        logger.error("ONNX model optimization failed: %s", exc)
        return False


def quantize_onnx_model(input_path: str, output_path: str, quantization_type: str = "int8") -> bool:
    """Quantize an ONNX model to ``int8`` or ``int4``."""
    if not HAS_ONNX:
        logger.warning("Skipping quantization as onnx is not available")
        return False

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        logger.info("Quantizing ONNX model to %s...", quantization_type.upper())
        quant_type = QuantType.QInt8 if quantization_type.lower() == "int8" else QuantType.QInt4
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=quant_type,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            op_types_to_quantize=["MatMul", "Gemm", "Relu", "Add", "Mul"],
        )
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        quant_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quant_size / original_size) * 100
        logger.info(
            "Model size: %.2f MB → %.2f MB (%.2f%% reduction)",
            original_size,
            quant_size,
            reduction,
        )
        return True
    except ImportError:
        logger.error(
            "Could not import onnxruntime.quantization. Please install onnxruntime-extensions."
        )
        return False
    except Exception as exc:  # pragma: no cover - runtime behaviour
        logger.error("ONNX model quantization failed: %s", exc)
        return False


def create_mobile_package(model_path: str, output_dir: str, config: EdgeFormerConfig) -> bool:
    """Create a deployment package containing the ONNX model and metadata."""
    logger.info("Creating mobile package in %s...", output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(model_path, os.path.join(output_dir, "model.onnx"))

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
        with open(os.path.join(output_dir, "config.json"), "w") as fh:
            json.dump(config_dict, fh, indent=2)

        readme = (
            "# EdgeFormer Mobile Package\n\n"
            "This package contains the EdgeFormer model optimized for mobile and edge devices.\n\n"
            "## Model Details\n"
            f"- Model type: EdgeFormer\n"
            f"- Hidden size: {config.hidden_size}\n"
            f"- Layers: {config.num_hidden_layers}\n"
            f"- Attention heads: {config.num_attention_heads}\n"
            f"- Vocabulary size: {config.vocab_size}\n"
            f"- MLA latent size factor: {config.latent_size_factor}\n\n"
            "## Usage\n"
            "See documentation for integration instructions.\n"
        )
        with open(os.path.join(output_dir, "README.md"), "w") as fh:
            fh.write(readme)
        logger.info("Successfully created mobile package at %s", output_dir)
        return True
    except Exception as exc:  # pragma: no cover - runtime behaviour
        logger.error("Failed to create mobile package: %s", exc)
        return False


def validate_onnx_model(
    onnx_path: str,
    input_data: Dict[str, torch.Tensor],
    pytorch_outputs: torch.Tensor,
) -> bool:
    """Validate the exported ONNX model using onnxruntime."""
    if not HAS_ORT:
        logger.warning("Skipping validation as onnxruntime is not available")
        return False

    logger.info("Validating ONNX model...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, sess_options)
        ort_inputs = {k: v.cpu().numpy() for k, v in input_data.items()}
        ort_outputs = session.run(None, ort_inputs)
        torch_np = pytorch_outputs.cpu().numpy()
        onnx_np = ort_outputs[0]
        mae = np.mean(np.abs(torch_np - onnx_np))
        max_diff = np.max(np.abs(torch_np - onnx_np))
        logger.info(
            "Output comparison - Mean Absolute Error: %.6f, Max Difference: %.6f",
            mae,
            max_diff,
        )
        if mae > 1e-4 or max_diff > 1e-2:
            logger.warning("ONNX model outputs differ significantly from PyTorch model")
            return False
        logger.info("ONNX model validation successful")
        return True
    except Exception as exc:  # pragma: no cover - runtime behaviour
        logger.error("ONNX model validation failed: %s", exc)
        return False


def export_to_onnx(
    model: EdgeFormer,
    output_path: str,
    *,
    optimize: bool = True,
    quantize: bool = False,
    quantization_type: str = "int8",
    opset_version: int = 14,
    create_package: bool = False,
) -> bool:
    """Export ``model`` to ONNX format."""
    if not HAS_ONNX:
        logger.error("ONNX is required for export. Please install onnx.")
        return False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        logger.info("Preparing sample input for ONNX tracing...")
        batch_size = 1
        seq_length = 32
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        wrapper = create_onnx_tracing_wrapper(model)
        wrapper.eval()
        with torch.no_grad():
            pytorch_output = wrapper(input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"},
        }
        logger.info("Exporting model to ONNX (opset version %d)...", opset_version)
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
        if HAS_ONNX:
            try:
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verified successfully")
            except Exception as exc:  # pragma: no cover - runtime behaviour
                logger.warning("ONNX model verification failed: %s", exc)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        validate_onnx_model(output_path, inputs, pytorch_output)
        if optimize:
            optimized_path = output_path.replace(".onnx", "_optimized.onnx")
            if optimize_onnx_model(output_path, optimized_path):
                output_path = optimized_path
        if quantize:
            quantized_path = output_path.replace(".onnx", f"_{quantization_type}.onnx")
            if quantize_onnx_model(output_path, quantized_path, quantization_type):
                output_path = quantized_path
        if create_package:
            package_dir = os.path.join(os.path.dirname(output_path), "mobile_package")
            create_mobile_package(output_path, package_dir, model.config)
        logger.info("Model export completed successfully!")
        return True
    except Exception as exc:  # pragma: no cover - runtime behaviour
        logger.error("Failed to export model to ONNX: %s", exc)
        return False


def main(args: argparse.Namespace) -> None:
    """Entry point for command line execution."""
    logger.info("Creating model configuration...")
    config = EdgeFormerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        latent_size_factor=args.latent_size_factor,
    )
    logger.info("Initializing model...")
    model = EdgeFormer(config)
    if args.model_path:
        logger.info("Loading model weights from %s", args.model_path)
        try:
            model.load_state_dict(torch.load(args.model_path))
        except Exception as exc:  # pragma: no cover - runtime behaviour
            logger.warning("Failed to load model weights: %s", exc)
            if not args.continue_on_error:
                logger.error("Aborting export due to model loading failure")
                return
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model has %d parameters", total_params)
    if args.weight_quantize:
        logger.info("Quantizing model weights to %d bits...", args.weight_quantize_bits)
        model = quantize_model(
            model,
            bits=args.weight_quantize_bits,
            group_size=args.weight_quantize_group_size,
        )
    success = export_to_onnx(
        model=model,
        output_path=args.output_path,
        optimize=not args.no_optimize,
        quantize=args.quantize,
        quantization_type=args.quantization_type,
        opset_version=args.opset_version,
        create_package=args.create_package,
    )
    if success:
        logger.info("Successfully exported model to %s", args.output_path)
    else:
        logger.error("Failed to export model to ONNX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EdgeFormer ONNX Export")
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--latent_size_factor", type=int, default=8, help="Latent size factor for MLA")
    parser.add_argument("--model_path", type=str, default="", help="Path to trained model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save ONNX model")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--no_optimize", action="store_true", help="Skip ONNX optimization")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue export on non-fatal errors")
    parser.add_argument("--quantize", action="store_true", help="Quantize the ONNX model")
    parser.add_argument("--quantization_type", type=str, default="int8", choices=["int8", "int4"], help="Type of ONNX quantization")
    parser.add_argument("--weight_quantize", action="store_true", help="Quantize model weights before ONNX export")
    parser.add_argument("--weight_quantize_bits", type=int, default=8, help="Bit width for weight quantization")
    parser.add_argument(
        "--weight_quantize_group_size",
        type=int,
        default=64,
        help="Group size for weight quantization",
    )
    parser.add_argument("--create_package", action="store_true", help="Create a mobile deployment package")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
