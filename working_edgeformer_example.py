#!/usr/bin/env python3
"""
Working EdgeFormer Example
Demonstrates correct usage with proper config format
"""

import torch
import sys
import os
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_edgeformer_config():
    """Create properly formatted EdgeFormer config"""
    
    config_dict = {
        'vocab_size': 1000,
        'hidden_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 1024,
        'max_position_embeddings': 512,
        'attention_type': 'standard'
    }
    
    # Try EdgeFormerConfig first, fallback to SimpleNamespace
    try:
        from src.model.transformer.config import EdgeFormerConfig
        return EdgeFormerConfig(**config_dict)
    except ImportError:
        return SimpleNamespace(**config_dict)

def main():
    """Test EdgeFormer with correct configuration"""
    
    print("🧪 Testing EdgeFormer with correct config format")
    
    # Create model
    from src.model.transformer.base_transformer import EdgeFormer
    config = create_edgeformer_config()
    
    model = EdgeFormer(config)
    print(f"✅ EdgeFormer created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    test_input = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Forward pass successful: {output.shape}")
    
    # Test with INT4 quantization
    from src.optimization.dynamic_quantization import DynamicQuantizer
    quantizer = DynamicQuantizer("int4")
    
    # Test on embedding layer
    embedding_weight = model.transformer.embeddings.word_embeddings.weight
    print(f"\nTesting INT4 on embedding: {embedding_weight.shape}")
    
    quantized = quantizer.quantize(embedding_weight.data)
    dequantized = quantizer.dequantize(quantized)
    
    compression_ratio = quantizer.get_compression_ratio(embedding_weight.data, quantized)
    mse = torch.mean((embedding_weight.data - dequantized) ** 2).item()
    relative_error = (mse / torch.mean(embedding_weight.data**2).item()) * 100
    
    print(f"  Compression: {compression_ratio:.2f}x")
    print(f"  Error: {relative_error:.3f}%")
    
    if compression_ratio >= 7.5:
        print("🎉 EdgeFormer + INT4 quantization WORKING!")
        return True
    else:
        print("⚠️ Compression below target")
        return False

if __name__ == "__main__":
    main()
