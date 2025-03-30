# tests/optimization/test_quantization.py
import unittest
import torch
from src.optimization.quantization import quantize_tensor, QuantizedLinear, quantize_edgeformer

class TestQuantization(unittest.TestCase):
    def test_int8_quantization(self):
        # Create test tensor
        test_tensor = torch.randn(10, 10)
        
        # Quantize to INT8
        quantized, scale, zero_point = quantize_tensor(test_tensor, bits=8)
        
        # Check shape
        self.assertEqual(quantized.shape, test_tensor.shape)
        
        # Check range (values should be within INT8 range after dequantization)
        max_diff = torch.max(torch.abs(test_tensor - quantized))
        self.assertLess(max_diff, 1.0)  # Ensure reasonable precision