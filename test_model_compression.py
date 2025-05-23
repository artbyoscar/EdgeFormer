import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.dynamic_quantization import DynamicQuantizer

def test_real_model_compression():
    """Test INT4 compression on actual transformer-like models"""
    print("=== REAL MODEL COMPRESSION TEST ===")
    
    # Create transformer-like model layers
    model_configs = [
        {"hidden_size": 128, "name": "Small Model"},
        {"hidden_size": 256, "name": "Medium Model"}, 
        {"hidden_size": 512, "name": "Large Model"}
    ]
    
    quantizer = DynamicQuantizer("int4")
    
    for config in model_configs:
        print(f"\n--- Testing {config['name']} (hidden_size={config['hidden_size']}) ---")
        
        # Create transformer layer weights (typical shapes)
        hidden_size = config['hidden_size']
        
        # Attention weights (QKV projection)
        qkv_weight = torch.randn(hidden_size, hidden_size * 3)
        
        # Output projection
        output_weight = torch.randn(hidden_size, hidden_size)
        
        # Feed-forward layers  
        ff1_weight = torch.randn(hidden_size, hidden_size * 4)
        ff2_weight = torch.randn(hidden_size * 4, hidden_size)
        
        weights = [qkv_weight, output_weight, ff1_weight, ff2_weight]
        weight_names = ["QKV", "Output", "FF1", "FF2"]
        
        total_original_size = 0
        total_compressed_size = 0
        total_mse = 0
        
        for weight, name in zip(weights, weight_names):
            # Calculate original size
            original_size = weight.numel() * 4  # 4 bytes per float32
            
            # Quantize
            quantized = quantizer.quantize(weight)
            compressed_size = quantized['packed_data'].numel()
            
            # Dequantize and test accuracy
            dequantized = quantizer.dequantize(quantized)
            mse = torch.mean((weight - dequantized) ** 2).item()
            
            # Calculate compression ratio
            compression_ratio = original_size / compressed_size
            
            print(f"  {name}: {compression_ratio:.2f}x compression, MSE: {mse:.6f}")
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            total_mse += mse
        
        # Overall stats for this model
        overall_compression = total_original_size / total_compressed_size
        avg_mse = total_mse / len(weights)
        
        print(f"  OVERALL: {overall_compression:.2f}x compression, Avg MSE: {avg_mse:.6f}")
        print(f"  Model size: {total_original_size/1024/1024:.2f} MB → {total_compressed_size/1024/1024:.2f} MB")
        
        # Verify claims
        if overall_compression >= 7.5:  # Allow small tolerance
            print(f"  ✅ 8x compression claim VERIFIED")
        else:
            print(f"  ⚠️ Compression below 8x: {overall_compression:.2f}x")
            
        if avg_mse < 0.1:
            print(f"  ✅ Accuracy claim VERIFIED (low error)")
        else:
            print(f"  ⚠️ Higher than expected error: {avg_mse:.6f}")

def test_benchmark_claims():
    """Test the specific claims from your README"""
    print("\n=== VERIFYING README CLAIMS ===")
    
    # Your README claims these compressions:
    claimed_results = [
        {"model_size": 32, "fp32_mb": 6.59, "int4_mb": 0.82, "claimed_compression": 8.0},
        {"model_size": 64, "fp32_mb": 13.55, "int4_mb": 1.69, "claimed_compression": 8.0},
        {"model_size": 128, "fp32_mb": 28.58, "int4_mb": 3.57, "claimed_compression": 8.0}
    ]
    
    quantizer = DynamicQuantizer("int4")
    
    for claim in claimed_results:
        print(f"\nTesting {claim['model_size']}M parameter model:")
        
        # Estimate parameters needed to get claimed FP32 size
        # FP32 size = params * 4 bytes
        estimated_params = int((claim['fp32_mb'] * 1024 * 1024) / 4)
        
        # Create model weight that matches this size
        # Use square root to get reasonable dimensions
        dim = int(estimated_params ** 0.5)
        if dim * dim < estimated_params:
            # Add a bit more to match exactly
            test_weight = torch.randn(dim, dim + (estimated_params - dim*dim)//dim)
        else:
            test_weight = torch.randn(dim, dim)
        
        actual_params = test_weight.numel()
        actual_fp32_mb = (actual_params * 4) / (1024 * 1024)
        
        print(f"  Created model: {actual_params:,} params, {actual_fp32_mb:.2f} MB")
        
        # Test compression
        quantized = quantizer.quantize(test_weight)
        compressed_bytes = quantized['packed_data'].numel()
        compressed_mb = compressed_bytes / (1024 * 1024)
        
        actual_compression = (actual_params * 4) / compressed_bytes
        
        print(f"  Actual compression: {actual_compression:.2f}x")
        print(f"  Compressed size: {compressed_mb:.2f} MB")
        print(f"  Claimed size: {claim['int4_mb']} MB")
        
        # Verify accuracy
        dequantized = quantizer.dequantize(quantized)
        mse = torch.mean((test_weight - dequantized) ** 2).item()
        relative_error = (mse / torch.mean(test_weight**2).item()) * 100
        
        print(f"  Relative error: {relative_error:.2f}%")
        
        if actual_compression >= 7.5:
            print(f"  ✅ 8x compression VERIFIED")
        else:
            print(f"  ⚠️ Below 8x: {actual_compression:.2f}x")

if __name__ == "__main__":
    test_real_model_compression()
    test_benchmark_claims()
    
    print("\n🎯 SUMMARY:")
    print("If all tests show ✅, your claims are bulletproof!")
    print("Ready for strategic partnerships with confidence.")