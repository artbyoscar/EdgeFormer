# In examples/optimize_for_rdna3.py

import torch
import os
from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.device import export_to_onnx
from src.utils.amd_optimizations import optimize_for_rdna3, tune_rdna3_parameters, create_rdna3_session

def run_rdna3_optimization():
    """Optimize EdgeFormer model for RDNA3 architecture."""
    # Create output directory
    os.makedirs("optimized_models", exist_ok=True)
    
    # Create model configuration
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
    )
    
    # Initialize model
    model = EdgeFormer(config)
    model.eval()
    
    # Tune model parameters for RDNA3
    model = tune_rdna3_parameters(model)
    
    # Export to ONNX
    onnx_path = export_to_onnx(model, "optimized_models/edgeformer_base.onnx")
    
    # Apply RDNA3 optimizations
    optimized_path = optimize_for_rdna3(
        onnx_path, 
        "optimized_models/edgeformer_rdna3.onnx"
    )
    
    # Create optimized session
    session = create_rdna3_session(optimized_path)
    
    # Test optimized model
    input_ids = torch.randint(0, config.vocab_size, (1, 512))
    attention_mask = torch.ones(1, 512)
    
    inputs = {
        'input_ids': input_ids.cpu().numpy(),
        'attention_mask': attention_mask.cpu().numpy()
    }
    
    outputs = session.run(None, inputs)
    
    print(f"RDNA3 optimization completed successfully!")
    print(f"Optimized model saved to: {optimized_path}")
    
    return {
        'base_onnx_path': onnx_path,
        'optimized_onnx_path': optimized_path,
        'session': session
    }

if __name__ == "__main__":
    run_rdna3_optimization()