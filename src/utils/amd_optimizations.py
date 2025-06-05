# In src/utils/amd_optimizations.py

import torch
import onnxruntime as ort

def optimize_for_rdna3(onnx_path, optimized_path):
    """Apply RDNA3-specific optimizations to the ONNX model."""
    from onnxruntime.transformers import optimizer
    
    # Create optimization options for RDNA3 architecture
    opt_options = optimizer.OptimizationOptions()
    opt_options.enable_gelu_approximation = True
    opt_options.enable_layer_norm_fusion = True
    opt_options.enable_attention_fusion = True
    opt_options.enable_embed_layer_norm_fusion = True
    
    # Add RDNA3-specific optimizations
    opt_options.enable_gpu_specific_optimizations = True
    
    # Optimize model
    optimized_model = optimizer.optimize_model(
        onnx_path,
        'bert',
        num_heads=8,
        hidden_size=256,
        optimization_options=opt_options
    )
    
    # Save optimized model
    optimized_model.save_model_to_file(optimized_path)
    
    return optimized_path

def create_rdna3_session(onnx_path):
    """Create an ONNX session optimized for RDNA3 architecture."""
    # DirectML session options optimized for RDNA3
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = False
    
    # RDNA3-specific provider options
    provider_options = [
        {
            'device_id': 0,
            'enable_dynamic_block_dma': True,
            'preferred_scale_for_texture_type': 'FLOAT16',
            'enable_subgraph_fusion': True,
            'enable_memory_pool': True,
            'memory_pool_size': 1024 * 1024 * 256,  # 256MB memory pool
            'enable_texture_data_type': True
        },
        {}
    ]
    
    # Create session
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=['DmlExecutionProvider', 'CPUExecutionProvider'],
        provider_options=provider_options
    )
    
    return session

def tune_rdna3_parameters(model, input_shape=(1, 512)):
    """Tune model parameters specifically for RDNA3 architecture."""
    # RDNA3 works best with certain block sizes for matrix operations
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
        for layer in model.transformer.layer:
            # Tune attention block sizes
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'latent_attention'):
                attn = layer.attention.latent_attention
                # Set optimal block sizes for RDNA3
                attn.block_size = 64
                # RDNA3 benefits from reshaping certain operations
                attn.use_rdna3_reshape = True
            
            # Tune MLP block sizes
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'sparse_mlp'):
                mlp = layer.mlp.sparse_mlp
                # Set optimal block sizes for RDNA3
                mlp.block_size = 128
    
    return model
