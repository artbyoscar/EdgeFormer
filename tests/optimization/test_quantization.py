# tests/optimization/test_quantization.py
import unittest
import torch
from src.optimization.quantization import quantize_tensor, QuantizedLinear

class TestQuantization(unittest.TestCase):
    def test_int8_quantization(self):
        """Test INT8 quantization of tensors."""
        # Create a test tensor
        test_tensor = torch.randn(10, 10)
        
        # Quantize to INT8
        quantized, scale, zero_point = quantize_tensor(test_tensor, bits=8)
        
        # Check shape
        self.assertEqual(quantized.shape, test_tensor.shape)
        
        # Check that quantization reduced precision (some information loss is expected)
        self.assertFalse(torch.all(torch.eq(test_tensor, quantized)))
        
        # Check that values are within a reasonable range of the original
        max_diff = torch.max(torch.abs(test_tensor - quantized))
        self.assertLess(max_diff, 3.0)  # Ensure reasonable precision
    
    def test_quantized_linear(self):
        """Test QuantizedLinear layer."""
        # Create a test layer
        in_features, out_features = 64, 32
        linear = QuantizedLinear(in_features, out_features, bits=8)
        
        # Create a test input
        x = torch.randn(2, in_features)
        
        # Test forward pass
        y = linear(x)
        
        # Check output shape
        self.assertEqual(y.shape, (2, out_features))
        
        # Ensure quantization happened
        self.assertTrue(linear.quantized)