import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.dynamic_quantization import Int4Quantizer

def test_int4_accuracy_fix():
    """Test if INT4 accuracy issue is fixed"""
    print("🔧 TESTING INT4 ACCURACY FIX")
    print("=" * 40)
    
    quantizer = Int4Quantizer()
    
    # Test cases that were failing
    test_cases = [
        ("Small tensor", torch.randn(10, 10) * 0.5),
        ("Model weights", torch.randn(32, 512) * 0.3),
        ("Large matrix", torch.randn(100, 100) * 1.0)
    ]
    
    all_good = True
    
    for name, weights in test_cases:
        quantized = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized)
        
        # Calculate relative error
        weight_range = torch.max(weights) - torch.min(weights)
        max_error = torch.max(torch.abs(weights - dequantized))
        relative_error = (max_error / weight_range * 100).item() if weight_range > 0 else 0
        
        compression = quantizer.get_compression_ratio(weights, quantized)
        
        print(f"{name}: {compression:.2f}x compression, {relative_error:.2f}% error")
        
        # Check if fixed
        if relative_error < 15.0:  # Much better threshold than 53%
            print(f"  ✅ FIXED - Accuracy much improved!")
        else:
            print(f"  ❌ STILL NEEDS WORK - Error too high")
            all_good = False
    
    if all_good:
        print("\n🎉 INT4 ACCURACY SIGNIFICANTLY IMPROVED!")
        print("✅ Ready for strategic partnerships!")
    else:
        print("\n⚠️ Still some accuracy issues to resolve")

if __name__ == "__main__":
    test_int4_accuracy_fix()