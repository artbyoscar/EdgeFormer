import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.dynamic_quantization import Int4Quantizer

def test_roadmap_model_sizes():
    """Test INT4 quantization on model sizes from roadmap: 32, 64, 128"""
    quantizer = Int4Quantizer()
    
    # Model sizes from your roadmap
    test_cases = [
        {"size": 32, "shape": (32, 512), "name": "Small Model"},
        {"size": 64, "shape": (64, 768), "name": "Medium Model"},
        {"size": 128, "shape": (128, 1024), "name": "Large Model"}
    ]
    
    print("=== TESTING ROADMAP MODEL SIZES ===")
    
    for case in test_cases:
        print(f"\n{case['name']} (Size {case['size']}): {case['shape']}")
        
        # Create realistic model weights
        weights = torch.randn(case['shape']) * 0.5
        
        # Test quantization
        quantized = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized)
        
        # Calculate metrics
        compression = quantizer.get_compression_ratio(weights, quantized)
        mse = torch.mean((weights - dequantized) ** 2).item()
        max_error = torch.max(torch.abs(weights - dequantized)).item()
        
        print(f"  ✅ Compression: {compression:.2f}x")
        print(f"  📊 MSE: {mse:.6f}")
        print(f"  📏 Max Error: {max_error:.4f}")
        print(f"  💾 Original: {weights.numel() * 4} bytes → {quantized['packed_data'].numel()} bytes")
        
        # Success criteria
        if 4.0 <= compression <= 8.0:
            print(f"  ✅ Compression target met!")
        else:
            print(f"  ❌ Compression target missed")

if __name__ == "__main__":
    test_roadmap_model_sizes()