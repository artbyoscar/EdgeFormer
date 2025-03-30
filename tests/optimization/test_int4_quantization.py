import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.optimization.quantization import (
    QuantizationConfig, 
    quantize_tensor,
    QuantizedLinear4Bit,
    Quantizer,
    pack_int4_tensor,
    unpack_int4_tensor
)

class TestINT4Quantization(unittest.TestCase):
    def test_int4_ranges(self):
        """Test INT4 range handling."""
        # Test symmetric range (-8 to 7)
        tensor = torch.tensor([-9, -8, -1, 0, 1, 7, 8])
        _, _, _, packed, shape = quantize_tensor(tensor, bits=4, sym=True, memory_efficient=True)
        unpacked = unpack_int4_tensor(packed, shape)
        self.assertTrue(torch.all(unpacked <= 7))
        self.assertTrue(torch.all(unpacked >= -8))
        
        # Test asymmetric range (0 to 15)
        tensor = torch.tensor([-1, 0, 1, 14, 15, 16])
        _, _, _, packed, shape = quantize_tensor(tensor, bits=4, sym=False, memory_efficient=True)
        unpacked = unpack_int4_tensor(packed, shape)
        self.assertTrue(torch.all(unpacked <= 15))
        self.assertTrue(torch.all(unpacked >= 0))
    
    def test_packing_unpacking(self):
        """Test packing and unpacking INT4 values."""
        # Even length tensor
        tensor = torch.tensor([-8, -4, 0, 3, 7, 1], dtype=torch.int8)
        packed, orig_shape = pack_int4_tensor(tensor)
        unpacked = unpack_int4_tensor(packed, orig_shape)
        self.assertTrue(torch.equal(tensor, unpacked))
        
        # Odd length tensor
        tensor = torch.tensor([-8, -4, 0, 3, 7], dtype=torch.int8)
        packed, orig_shape = pack_int4_tensor(tensor)
        unpacked = unpack_int4_tensor(packed, orig_shape)
        self.assertTrue(torch.equal(tensor, unpacked))
    
    def test_memory_savings(self):
        """Test memory savings of INT4 quantization."""
        # Create a moderate sized tensor
        tensor = torch.randn(1024, 1024)
        
        # Original memory (32-bit float)
        original_bytes = tensor.element_size() * tensor.numel()
        
        # Quantize to INT4
        _, _, _, packed, _ = quantize_tensor(tensor, bits=4, sym=True, memory_efficient=True)
        
        # Packed memory (4-bit)
        packed_bytes = packed.element_size() * packed.numel()
        
        # Should be approximately 8x smaller (32-bit to 4-bit)
        # With a small overhead for scales and zero points
        self.assertLess(packed_bytes, original_bytes / 6)
    
    def test_quantized_linear_layer(self):
        """Test INT4 quantized linear layer."""
        in_features, out_features = 768, 256
        
        # Create quantized layer
        quant_linear = QuantizedLinear4Bit(in_features, out_features)
        
        # Input tensor
        x = torch.randn(2, in_features)
        
        # Forward pass before quantization
        out_before = quant_linear(x)
        
        # Manually trigger quantization
        quant_linear.quantize_weights()
        
        # Forward pass after quantization
        out_after = quant_linear(x)
        
        # Output should be roughly similar but not identical
        self.assertFalse(torch.allclose(out_before, out_after, rtol=1e-2))
        self.assertTrue(torch.allclose(out_before, out_after, rtol=1e-1))
        
        # Check that weights were packed
        self.assertIsNotNone(quant_linear.packed_weight)

if __name__ == "__main__":
    unittest.main()